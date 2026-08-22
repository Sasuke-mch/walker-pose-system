from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time

import cv2
import numpy as np

from .config import CameraConfig

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def natural_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def read_image(path: str | Path) -> np.ndarray | None:
    """Read an image safely when a Windows path contains Chinese characters."""
    file_path = Path(path)
    try:
        data = np.fromfile(str(file_path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


@dataclass
class SourceFrame:
    frame_id: int
    timestamp_sec: float
    image: np.ndarray


class FrameSource:
    name: str
    fps: float
    width: int
    height: int
    total_frames: int | None
    is_live: bool = False

    def read(self) -> SourceFrame | None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class VideoSource(FrameSource):
    def __init__(self, path: str | Path, start_frame: int = 0, loop: bool = False) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"视频不存在：{self.path}")
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise RuntimeError(f"OpenCV无法打开视频：{self.path}")
        self.name = str(self.path)
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0:
            self.fps = 25.0
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames = count if count > 0 else None
        self.start_frame = max(0, int(start_frame))
        self.loop = loop
        self.current = self.start_frame
        if self.start_frame:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

    def read(self) -> SourceFrame | None:
        ok, image = self.capture.read()
        if not ok or image is None:
            if not self.loop:
                return None
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            self.current = self.start_frame
            ok, image = self.capture.read()
            if not ok or image is None:
                return None
        frame_id = self.current
        self.current += 1
        return SourceFrame(frame_id, frame_id / self.fps, image)

    def close(self) -> None:
        self.capture.release()


class ImageDirectorySource(FrameSource):
    def __init__(
        self,
        directory: str | Path,
        fps: float,
        start_frame: int = 0,
        loop: bool = False,
    ) -> None:
        self.directory = Path(directory).resolve()
        if not self.directory.is_dir():
            raise NotADirectoryError(f"图片目录不存在：{self.directory}")
        self.files = sorted(
            [
                p
                for p in self.directory.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=natural_key,
        )
        if not self.files:
            raise FileNotFoundError(f"图片目录中没有支持的图像：{self.directory}")
        self.name = str(self.directory)
        self.fps = max(0.1, float(fps))
        self.start_frame = max(0, int(start_frame))
        if self.start_frame >= len(self.files):
            raise ValueError(
                f"start_frame={self.start_frame}超出图片数量{len(self.files)}。"
            )
        self.index = self.start_frame
        self.loop = loop
        self.total_frames = len(self.files)
        sample = read_image(self.files[self.start_frame])
        if sample is None:
            raise RuntimeError(f"无法读取图片：{self.files[self.start_frame]}")
        self.height, self.width = sample.shape[:2]

    def read(self) -> SourceFrame | None:
        if self.index >= len(self.files):
            if not self.loop:
                return None
            self.index = self.start_frame
        path = self.files[self.index]
        image = read_image(path)
        if image is None:
            raise RuntimeError(f"无法读取图片：{path}")
        frame_id = self.index
        self.index += 1
        return SourceFrame(frame_id, frame_id / self.fps, image)


class CameraSource(FrameSource):
    is_live = True

    def __init__(self, camera_id: int, config: CameraConfig) -> None:
        self.camera_id = int(camera_id)
        self.name = f"camera:{self.camera_id}"
        if config.backend == "dshow" and hasattr(cv2, "CAP_DSHOW"):
            self.capture = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        elif config.backend == "msmf" and hasattr(cv2, "CAP_MSMF"):
            self.capture = cv2.VideoCapture(self.camera_id, cv2.CAP_MSMF)
        else:
            self.capture = cv2.VideoCapture(self.camera_id)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        self.capture.set(cv2.CAP_PROP_FPS, config.fps)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"无法打开摄像头{self.camera_id}。")
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS)) or float(config.fps)
        self.total_frames = None
        self.current = 0
        self.started = time.monotonic()

    def read(self) -> SourceFrame | None:
        ok, image = self.capture.read()
        if not ok or image is None:
            raise RuntimeError("摄像头停止返回画面。")
        frame = SourceFrame(
            self.current, time.monotonic() - self.started, image
        )
        self.current += 1
        return frame

    def close(self) -> None:
        self.capture.release()
