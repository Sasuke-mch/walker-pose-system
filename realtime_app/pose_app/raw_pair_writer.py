from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class RawStereoPairWriter:
    """Persist the pre-inference left/right frames for a paired replay."""

    def __init__(self, output_dir: Path, output_fps: float) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_fps = float(output_fps)
        if not np.isfinite(self.output_fps) or self.output_fps <= 0:
            raise ValueError("output_fps must be positive and finite")
        self._left_writer: Any = None
        self._right_writer: Any = None
        self._size: tuple[int, int] | None = None
        self._metadata = self.output_dir / "raw_pairs.jsonl"
        self._metadata_fp = self._metadata.open("w", encoding="utf-8")
        self._frames = 0

    @staticmethod
    def _validate_image(image: np.ndarray, side: str) -> tuple[int, int]:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{side} raw frame must be HxWx3 ndarray")
        if image.dtype != np.uint8:
            raise ValueError(f"{side} raw frame must be uint8 BGR")
        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError(f"{side} raw frame has invalid dimensions")
        return width, height

    def _ensure_writers(self, left: np.ndarray, right: np.ndarray) -> None:
        left_size = self._validate_image(left, "left")
        right_size = self._validate_image(right, "right")
        if left_size != right_size:
            raise ValueError(f"left/right raw frame sizes differ: {left_size} vs {right_size}")
        if self._size is not None:
            if left_size != self._size:
                raise ValueError(f"raw frame dimensions changed: {self._size} -> {left_size}")
            return
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._left_writer = cv2.VideoWriter(
            str(self.output_dir / "raw_left.mp4"), fourcc, self.output_fps, left_size
        )
        self._right_writer = cv2.VideoWriter(
            str(self.output_dir / "raw_right.mp4"), fourcc, self.output_fps, right_size
        )
        if not self._left_writer.isOpened() or not self._right_writer.isOpened():
            self.close()
            raise RuntimeError("could not open raw left/right video writers")
        self._size = left_size

    def write(self, pair: Any) -> None:
        left = pair.left.image
        right = pair.right.image
        self._ensure_writers(left, right)
        assert self._left_writer is not None and self._right_writer is not None
        self._left_writer.write(left)
        self._right_writer.write(right)
        record = {
            "pair_id": int(pair.pair_id),
            "left_frame_id": int(pair.left.frame_id),
            "right_frame_id": int(pair.right.frame_id),
            "left_timestamp_sec": float(pair.left.timestamp_sec),
            "right_timestamp_sec": float(pair.right.timestamp_sec),
            "timestamp_skew_ms": float(pair.timestamp_skew_sec * 1000.0),
            "dropped_left": int(pair.dropped_left),
            "dropped_right": int(pair.dropped_right),
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
        self._metadata_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._metadata_fp.flush()
        self._frames += 1

    def close(self) -> dict[str, Any]:
        if self._left_writer is not None:
            self._left_writer.release()
            self._left_writer = None
        if self._right_writer is not None:
            self._right_writer.release()
            self._right_writer = None
        if not self._metadata_fp.closed:
            self._metadata_fp.close()
        return {
            "output_dir": str(self.output_dir),
            "left_video": str(self.output_dir / "raw_left.mp4"),
            "right_video": str(self.output_dir / "raw_right.mp4"),
            "metadata_jsonl": str(self._metadata),
            "frames": self._frames,
            "frame_size": list(self._size) if self._size else None,
            "fps": self.output_fps,
        }
