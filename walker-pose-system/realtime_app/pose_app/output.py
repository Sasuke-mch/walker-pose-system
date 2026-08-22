from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import time

import cv2
import numpy as np

from .schema import InferenceResult


class OutputWriter:
    def __init__(
        self,
        run_dir: Path,
        save_video: bool,
        save_json: bool,
        output_fps: float,
        source_name: str,
        mode: str,
    ) -> None:
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self.save_video = save_video
        self.save_json = save_json
        self.output_fps = max(1.0, float(output_fps))
        self.writer: cv2.VideoWriter | None = None
        self.size: tuple[int, int] | None = None
        self.video_path = run_dir / "annotated.mp4"
        self.temporary_video_path: Path | None = None
        self.json_file = (
            (run_dir / "results.jsonl").open(
                "w", encoding="utf-8", buffering=1
            )
            if save_json
            else None
        )
        self.source_name = source_name
        self.mode = mode
        self.started = time.time()
        self.model_times: list[float] = []
        self.roundtrip_times: list[float] = []
        self.processed = 0
        self.dropped = 0

    def _make_writer(self, path: Path, size: tuple[int, int]) -> cv2.VideoWriter:
        return cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.output_fps,
            size,
        )

    def _open_video(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        self.size = (width, height)
        writer = self._make_writer(self.video_path, self.size)
        if writer.isOpened():
            self.writer = writer
            return
        writer.release()

        # OpenCV VideoWriter may reject Windows paths containing Chinese
        # characters.  Write to an ASCII temporary path, then move on close.
        handle = tempfile.NamedTemporaryFile(
            prefix="pose_video_", suffix=".mp4", delete=False
        )
        handle.close()
        temporary = Path(handle.name)
        writer = self._make_writer(temporary, self.size)
        if not writer.isOpened():
            writer.release()
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"无法创建输出视频：{self.video_path}")
        self.temporary_video_path = temporary
        self.writer = writer

    def write(
        self, frame: np.ndarray, result: InferenceResult, dropped: int
    ) -> None:
        if self.save_video:
            if self.writer is None:
                self._open_video(frame)
            assert self.size is not None
            height, width = frame.shape[:2]
            if (width, height) != self.size:
                frame = cv2.resize(frame, self.size)
            assert self.writer is not None
            self.writer.write(frame)
        if self.json_file:
            self.json_file.write(
                json.dumps(result.to_dict(), ensure_ascii=False) + "\n"
            )
        self.processed += 1
        self.dropped = dropped
        self.model_times.append(result.model_ms)
        self.roundtrip_times.append(result.roundtrip_ms)

    def close(self) -> dict:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.temporary_video_path is not None:
            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(self.temporary_video_path), str(self.video_path))
            self.temporary_video_path = None
        if self.json_file is not None:
            self.json_file.close()
            self.json_file = None

        elapsed = max(1e-9, time.time() - self.started)
        summary = {
            "source": self.source_name,
            "mode": self.mode,
            "processed_frames": self.processed,
            "dropped_frames": self.dropped,
            "wall_time_sec": elapsed,
            "effective_fps": self.processed / elapsed,
            "mean_model_ms": (
                sum(self.model_times) / len(self.model_times)
                if self.model_times
                else 0.0
            ),
            "mean_roundtrip_ms": (
                sum(self.roundtrip_times) / len(self.roundtrip_times)
                if self.roundtrip_times
                else 0.0
            ),
            "annotated_video": str(self.video_path) if self.save_video else None,
            "results_jsonl": (
                str(self.run_dir / "results.jsonl") if self.save_json else None
            ),
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary
