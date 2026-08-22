from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np
from pose_app.config import load_config
from pose_app.docker_image import resolve_docker_image


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, encoding="utf-8", errors="replace",
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def check_image(name: str, errors: list[str]) -> None:
    try:
        image = resolve_docker_image(name)
        if image.matched_by == "inspect":
            print(f"[OK] 镜像 {image.configured}，ID={image.image_id[:19]}")
        else:
            print(f"[OK] 镜像标签 {image.configured} 已通过镜像列表匹配，运行时使用ID={image.image_id[:19]}")
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        errors.append(f"image:{name}")


def check_path(label: str, path: Path, file_expected: bool, errors: list[str]) -> None:
    ok = path.is_file() if file_expected else path.is_dir()
    if ok:
        print(f"[OK] {label}：{path}")
    else:
        print(f"[ERROR] {label}：{path}")
        errors.append(label)


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(root / "config.json"))
    parser.add_argument("--model", choices=["yolo26x_pose", "pmpose", "all"], default="yolo26x_pose")
    parser.add_argument("--video")
    args = parser.parse_args()
    config = load_config(args.config)
    errors: list[str] = []
    print(f"[OK] Python {sys.version.split()[0]}")
    print(f"[OK] NumPy {np.__version__}")
    print(f"[OK] OpenCV {cv2.__version__}")
    docker = run(["docker", "version", "--format", "{{.Server.Version}}"])
    if docker.returncode == 0:
        print(f"[OK] Docker {docker.stdout.strip()}")
    else:
        print("[ERROR] Docker Desktop未启动")
        errors.append("docker")

    if args.model in {"yolo26x_pose", "all"}:
        print("\n[YOLO26x-pose]")
        check_image(config.docker.image, errors)
        check_path("YOLO工程", config.model.repo, False, errors)
        check_path("YOLO26x-pose权重", config.model.weight, True, errors)

    if args.model in {"pmpose", "all"}:
        print("\n[YOLO26x + PMPose]")
        check_image(config.detector.image, errors)
        check_image(config.pmpose.image, errors)
        check_path("YOLO检测工程", config.detector.repo, False, errors)
        check_path("YOLO26x检测权重", config.detector.weight, True, errors)
        check_path("BBoxMaskPose工程", config.pmpose.repo, False, errors)
        config.pmpose.cache.mkdir(parents=True, exist_ok=True)
        print(f"[OK] PMPose缓存目录：{config.pmpose.cache}")

    if args.video:
        path = Path(args.video).resolve()
        capture = cv2.VideoCapture(str(path))
        try:
            if path.is_file() and capture.isOpened():
                print(f"[OK] 视频：{path}，FPS={capture.get(cv2.CAP_PROP_FPS):.3f}，帧数={int(capture.get(cv2.CAP_PROP_FRAME_COUNT))}")
            else:
                print(f"[ERROR] 视频无法打开：{path}")
                errors.append("video")
        finally:
            capture.release()

    print(f"\n错误数量：{len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
