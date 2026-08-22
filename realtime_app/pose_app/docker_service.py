from __future__ import annotations
from dataclasses import dataclass
import logging
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from .config import AppConfig
from .http_client import PMPosePipelineClient, PoseServiceClient, ServiceError
from .docker_image import resolve_docker_image

LOG = logging.getLogger(__name__)


@dataclass
class RunningService:
    name: str
    container_name: str
    process: subprocess.Popen
    log_handle: Any
    client: Any


class DockerPoseService:
    def __init__(self, config: AppConfig, run_dir: Path, model_name: str = "yolo26x_pose") -> None:
        self.config = config
        self.run_dir = run_dir
        self.model_name = model_name
        self.services: list[RunningService] = []

    @staticmethod
    def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def _check_docker(self) -> None:
        docker = self._run(["docker", "version", "--format", "{{.Server.Version}}"])
        if docker.returncode != 0:
            raise RuntimeError("Docker Desktop未启动或Docker引擎不可用。\n" + (docker.stdout or ""))

    def _resolve(self, image: str) -> str:
        resolution = resolve_docker_image(image)
        if resolution.matched_by == "image-list":
            LOG.warning("镜像标签%s无法直接解析，已改用镜像ID运行：%s", image, resolution.image_id)
        return resolution.runtime_reference

    def check(self) -> None:
        self._check_docker()
        if self.model_name == "yolo26x_pose":
            self._resolve(self.config.docker.image)
            if not self.config.model.repo.is_dir():
                raise FileNotFoundError(f"YOLO工程目录不存在：{self.config.model.repo}")
            if not self.config.model.weight.is_file():
                raise FileNotFoundError(f"YOLO26x-pose权重不存在：{self.config.model.weight}")
            return
        if self.model_name != "pmpose":
            raise ValueError(f"暂不支持模型：{self.model_name}")
        self._resolve(self.config.detector.image)
        self._resolve(self.config.pmpose.image)
        for label, path, is_file in (
            ("YOLO检测工程", self.config.detector.repo, False),
            ("YOLO26x检测权重", self.config.detector.weight, True),
            ("BBoxMaskPose工程", self.config.pmpose.repo, False),
        ):
            exists = path.is_file() if is_file else path.is_dir()
            if not exists:
                raise FileNotFoundError(f"{label}不存在：{path}")
        self.config.pmpose.cache.mkdir(parents=True, exist_ok=True)

    def _start_service(
        self,
        name: str,
        container_name: str,
        args: list[str],
        host_port: int,
        log_name: str,
    ) -> Any:
        self._run(["docker", "rm", "-f", container_name])
        log_dir = self.run_dir / "docker_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / log_name
        log_handle = log_path.open("w", encoding="utf-8", buffering=1)
        LOG.debug("Docker command: %s", subprocess.list2cmdline(args))
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if sys.platform == "win32" else 0
        process = subprocess.Popen(
            args,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=flags,
        )
        client = PoseServiceClient(
            base_url=f"http://127.0.0.1:{host_port}",
            timeout=self.config.docker.request_timeout_sec,
            jpeg_quality=self.config.output.jpeg_quality,
        )
        service = RunningService(name, container_name, process, log_handle, client)
        self.services.append(service)
        deadline = time.monotonic() + self.config.docker.startup_timeout_sec
        last_error = ""
        while time.monotonic() < deadline:
            code = process.poll()
            if code is not None:
                details = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
                raise RuntimeError(
                    f"{name}容器提前退出，返回码{code}。\n日志：{log_path}\n{details[-8000:]}"
                )
            try:
                health = client.health()
                if health.get("status") == "ready":
                    LOG.info("%s服务已就绪。", name)
                    return client
            except ServiceError as exc:
                last_error = str(exc)
            time.sleep(1.0)
        raise TimeoutError(f"等待{name}启动超时。最后错误：{last_error}；日志：{log_path}")

    def _base(self, name: str, host_port: int, container_port: int) -> list[str]:
        args = [
            "docker", "run", "--rm", "--name", name,
            "--shm-size", self.config.docker.shm_size,
            "-p", f"127.0.0.1:{host_port}:{container_port}",
        ]
        if self.config.docker.use_gpu:
            args += ["--gpus", "all"]
        return args

    def _start_yolo_pose(self) -> PoseServiceClient:
        image = self._resolve(self.config.docker.image)
        args = self._base("pose-video-yolo26x-pose", self.config.docker.host_port, self.config.docker.container_port)
        args += [
            "-e", "YOLO_CONFIG_DIR=/tmp/Ultralytics",
            "-v", f"{self.config.app_root.resolve()}:/workspace/pose_app:ro",
            "-v", f"{self.config.model.repo.resolve()}:/workspace/yolo:ro",
            "-v", f"{self.config.model.weight.parent.resolve()}:/workspace/weights:ro",
            "-w", "/workspace/yolo",
            image,
            "python", "/workspace/pose_app/server/yolo_pose_server.py",
            "--weights", f"/workspace/weights/{self.config.model.weight.name}",
            "--device", self.config.model.device,
            "--imgsz", str(self.config.model.imgsz),
            "--conf", str(self.config.model.candidate_conf),
            "--iou", str(self.config.model.iou),
            "--max-det", str(self.config.model.max_det),
            "--keypoint-score-thr", str(self.config.model.keypoint_score_threshold),
            "--pose-thr", str(self.config.model.pose_threshold),
            "--port", str(self.config.docker.container_port),
        ]
        return self._start_service(
            "YOLO26x-pose",
            "pose-video-yolo26x-pose",
            args,
            self.config.docker.host_port,
            "yolo26x_pose.log",
        )

    def _start_detector(self) -> PoseServiceClient:
        det = self.config.detector
        image = self._resolve(det.image)
        args = self._base("pose-video-yolo26x-detector", det.host_port, det.container_port)
        args += [
            "-e", "YOLO_CONFIG_DIR=/tmp/Ultralytics",
            "-v", f"{self.config.app_root.resolve()}:/workspace/pose_app:ro",
            "-v", f"{det.repo.resolve()}:/workspace/yolo:ro",
            "-v", f"{det.weight.parent.resolve()}:/workspace/detector_weights:ro",
            "-w", "/workspace/yolo",
            image,
            "python", "/workspace/pose_app/server/yolo_detector_server.py",
            "--weights", f"/workspace/detector_weights/{det.weight.name}",
            "--device", det.device,
            "--imgsz", str(det.imgsz),
            "--conf", str(det.conf),
            "--iou", str(det.iou),
            "--max-det", str(det.max_det),
            "--port", str(det.container_port),
        ]
        return self._start_service(
            "YOLO26x检测",
            "pose-video-yolo26x-detector",
            args,
            det.host_port,
            "yolo26x_detector.log",
        )

    def _start_pmpose(self) -> PoseServiceClient:
        pm = self.config.pmpose
        image = self._resolve(pm.image)
        args = self._base("pose-video-pmpose", pm.host_port, pm.container_port)
        args += [
            "-e", "TORCH_HOME=/workspace/cache",
            "-e", "HF_HOME=/workspace/cache/huggingface",
            "-v", f"{self.config.app_root.resolve()}:/workspace/pose_app:ro",
            "-v", f"{pm.repo.resolve()}:/workspace/BBoxMaskPose:ro",
            "-v", f"{pm.cache.resolve()}:/workspace/cache",
            "-w", "/workspace/BBoxMaskPose",
            image,
            "python", "/workspace/pose_app/server/pmpose_server.py",
            "--device", pm.device,
            "--variant", pm.variant,
            "--pose-thr", str(pm.pose_threshold),
            "--keypoint-score-thr", str(pm.keypoint_score_threshold),
            "--mask-mode", pm.mask_mode,
            "--port", str(pm.container_port),
        ]
        return self._start_service(
            "PMPose",
            "pose-video-pmpose",
            args,
            pm.host_port,
            "pmpose.log",
        )

    def start(self) -> Any:
        self.check()
        if self.model_name == "yolo26x_pose":
            return self._start_yolo_pose()
        detector = self._start_detector()
        pose = self._start_pmpose()
        client = PMPosePipelineClient(
            detector_url=f"http://127.0.0.1:{self.config.detector.host_port}",
            pose_url=f"http://127.0.0.1:{self.config.pmpose.host_port}",
            timeout=self.config.docker.request_timeout_sec,
            jpeg_quality=self.config.output.jpeg_quality,
        )
        client.health()
        return client

    def stop(self) -> None:
        for service in reversed(self.services):
            self._run(["docker", "rm", "-f", service.container_name])
            try:
                service.process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                service.process.terminate()
            try:
                service.log_handle.close()
            except Exception:
                pass
        self.services.clear()
