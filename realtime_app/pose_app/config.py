from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DockerConfig:
    image: str
    host_port: int
    container_port: int
    startup_timeout_sec: float
    request_timeout_sec: float
    shm_size: str
    use_gpu: bool


@dataclass(frozen=True)
class ModelConfig:
    repo: Path
    weight: Path
    device: str
    imgsz: int
    candidate_conf: float
    iou: float
    max_det: int
    keypoint_score_threshold: float
    pose_threshold: float


@dataclass(frozen=True)
class DetectorConfig:
    image: str
    repo: Path
    weight: Path
    host_port: int
    container_port: int
    device: str
    imgsz: int
    conf: float
    iou: float
    max_det: int


@dataclass(frozen=True)
class PMPoseConfig:
    image: str
    repo: Path
    cache: Path
    host_port: int
    container_port: int
    device: str
    variant: str
    keypoint_score_threshold: float
    pose_threshold: float
    mask_mode: str


@dataclass(frozen=True)
class OutputConfig:
    root: Path
    jpeg_quality: int
    realtime_output_fps: float
    draw_keypoint_threshold: float


@dataclass(frozen=True)
class CameraConfig:
    width: int
    height: int
    fps: int
    backend: str


@dataclass(frozen=True)
class AppConfig:
    app_root: Path
    project_root: Path
    docker: DockerConfig
    model: ModelConfig
    detector: DetectorConfig
    pmpose: PMPoseConfig
    output: OutputConfig
    camera: CameraConfig


def _obj(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name}必须是JSON对象。")
    return value


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8-sig"))
    raw = _obj(raw, "配置根节点")
    app_root = config_path.parent.resolve()
    project_raw = Path(str(raw.get("project_root", "..")))
    project_root = project_raw if project_raw.is_absolute() else (app_root / project_raw).resolve()

    d = _obj(raw.get("docker", {}), "docker")
    m = _obj(raw.get("model", {}), "model")
    det = _obj(raw.get("detector", {}), "detector")
    pm = _obj(raw.get("pmpose", {}), "pmpose")
    o = _obj(raw.get("output", {}), "output")
    c = _obj(raw.get("camera", {}), "camera")

    def project_path(value: Any) -> Path:
        p = Path(str(value))
        return p if p.is_absolute() else (project_root / p).resolve()

    output_raw = Path(str(o.get("root", "pose_realtime_app/outputs")))
    output_root = output_raw if output_raw.is_absolute() else (project_root / output_raw).resolve()

    return AppConfig(
        app_root=app_root,
        project_root=project_root,
        docker=DockerConfig(
            image=str(d.get("image", "yolo26-realtime:latest")),
            host_port=int(d.get("host_port", 18080)),
            container_port=int(d.get("container_port", 18080)),
            startup_timeout_sec=float(d.get("startup_timeout_sec", 600)),
            request_timeout_sec=float(d.get("request_timeout_sec", 180)),
            shm_size=str(d.get("shm_size", "4g")),
            use_gpu=bool(d.get("use_gpu", True)),
        ),
        model=ModelConfig(
            repo=project_path(m.get("repo", "YOLO26-test")),
            weight=project_path(m.get("weight", "external_models/YOLO26x-pose/yolo26x-pose.pt")),
            device=str(m.get("device", "0")),
            imgsz=int(m.get("imgsz", 1280)),
            candidate_conf=float(m.get("candidate_conf", 0.01)),
            iou=float(m.get("iou", 0.70)),
            max_det=int(m.get("max_det", 300)),
            keypoint_score_threshold=float(m.get("keypoint_score_threshold", 0.20)),
            pose_threshold=float(m.get("pose_threshold", 0.40)),
        ),
        detector=DetectorConfig(
            image=str(det.get("image", d.get("image", "yolo26-realtime:latest"))),
            repo=project_path(det.get("repo", "YOLO26-test")),
            weight=project_path(det.get("weight", "YOLO26-test/yolo26x.pt")),
            host_port=int(det.get("host_port", 18081)),
            container_port=int(det.get("container_port", 18081)),
            device=str(det.get("device", "0")),
            imgsz=int(det.get("imgsz", 1280)),
            conf=float(det.get("conf", 0.05)),
            iou=float(det.get("iou", 0.70)),
            max_det=int(det.get("max_det", 300)),
        ),
        pmpose=PMPoseConfig(
            image=str(pm.get("image", "bboxmaskpose:latest")),
            repo=project_path(pm.get("repo", "BBoxMaskPose")),
            cache=project_path(pm.get("cache", "model_cache/pmpose")),
            host_port=int(pm.get("host_port", 18082)),
            container_port=int(pm.get("container_port", 18082)),
            device=str(pm.get("device", "cuda:0")),
            variant=str(pm.get("variant", "PMPose-b")),
            keypoint_score_threshold=float(pm.get("keypoint_score_threshold", 0.20)),
            pose_threshold=float(pm.get("pose_threshold", 0.30)),
            mask_mode=str(pm.get("mask_mode", "bbox")),
        ),
        output=OutputConfig(
            root=output_root,
            jpeg_quality=max(1, min(100, int(o.get("jpeg_quality", 90)))),
            realtime_output_fps=max(1.0, float(o.get("realtime_output_fps", 10.0))),
            draw_keypoint_threshold=float(o.get("draw_keypoint_threshold", 0.20)),
        ),
        camera=CameraConfig(
            width=int(c.get("width", 1280)),
            height=int(c.get("height", 720)),
            fps=int(c.get("fps", 30)),
            backend=str(c.get("backend", "dshow")).lower(),
        ),
    )
