from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import time

import cv2
import numpy as np

from .calibration import StereoCalibration
from .schema import InferenceResult
from .stereo_sources import StereoFramePair
from .triangulation import TriangulatedPerson


class StereoOutputWriter:
    def __init__(
        self,
        run_dir: Path,
        save_video: bool,
        save_json: bool,
        output_fps: float,
        source_name: str,
        calibration: StereoCalibration,
    ) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.save_video = save_video
        self.save_json = save_json
        self.output_fps = max(1.0, float(output_fps))
        self.source_name = source_name
        self.calibration = calibration
        self.writer: cv2.VideoWriter | None = None
        self.size: tuple[int, int] | None = None
        self.video_path = self.run_dir / "stereo_annotated.mp4"
        self.temporary_video_path: Path | None = None
        self.json_path = self.run_dir / "stereo_results.jsonl"
        self.json_file = (
            self.json_path.open("w", encoding="utf-8", buffering=1)
            if save_json
            else None
        )
        self.started = time.time()
        self.processed = 0
        self.skews_ms: list[float] = []
        self.left_model_ms: list[float] = []
        self.right_model_ms: list[float] = []
        self.valid_3d_counts: list[int] = []
        self.timestamp_types: set[str] = set()

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
        handle = tempfile.NamedTemporaryFile(
            prefix="stereo_pose_", suffix=".mp4", delete=False
        )
        handle.close()
        temporary = Path(handle.name)
        writer = self._make_writer(temporary, self.size)
        if not writer.isOpened():
            writer.release()
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"无法创建双目输出视频：{self.video_path}")
        self.temporary_video_path = temporary
        self.writer = writer

    def write(
        self,
        frame: np.ndarray,
        pair: StereoFramePair,
        left_result: InferenceResult,
        right_result: InferenceResult,
        persons_3d: list[TriangulatedPerson],
    ) -> None:
        if self.save_video:
            if self.writer is None:
                self._open_video(frame)
            assert self.size is not None and self.writer is not None
            height, width = frame.shape[:2]
            if (width, height) != self.size:
                frame = cv2.resize(frame, self.size)
            self.writer.write(frame)

        payload = {
            "pair_id": pair.pair_id,
            "pair_timestamp_sec": pair.timestamp_sec,
            "left_frame_id": pair.left.frame_id,
            "right_frame_id": pair.right.frame_id,
            "left_timestamp_sec": pair.left.timestamp_sec,
            "right_timestamp_sec": pair.right.timestamp_sec,
            "timestamp_skew_ms": pair.timestamp_skew_sec * 1000.0,
            "timestamp_type": pair.timestamp_type,
            "left_host_timestamp_ns": pair.left_host_timestamp_ns,
            "right_host_timestamp_ns": pair.right_host_timestamp_ns,
            "signed_host_delta_ms_right_minus_left": pair.signed_host_delta_ms,
            "left_read_duration_ms": pair.left_read_duration_ms,
            "right_read_duration_ms": pair.right_read_duration_ms,
            "dropped_left": pair.dropped_left,
            "dropped_right": pair.dropped_right,
            "coordinate_frame": "left_camera",
            "length_unit": self.calibration.length_unit,
            "left": left_result.to_dict(),
            "right": right_result.to_dict(),
            "persons_3d": [person.to_dict() for person in persons_3d],
        }
        if self.json_file is not None:
            self.json_file.write(
                json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
            )

        self.processed += 1
        self.skews_ms.append(pair.timestamp_skew_sec * 1000.0)
        self.left_model_ms.append(left_result.model_ms)
        self.right_model_ms.append(right_result.model_ms)
        self.valid_3d_counts.append(
            sum(person.valid_keypoints for person in persons_3d)
        )
        self.timestamp_types.add(pair.timestamp_type)

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
        skews = np.asarray(self.skews_ms, dtype=np.float64)
        summary = {
            "source": self.source_name,
            "processed_pairs": self.processed,
            "wall_time_sec": elapsed,
            "effective_pair_fps": self.processed / elapsed,
            "mean_timestamp_skew_ms": float(np.mean(skews)) if skews.size else 0.0,
            "p95_timestamp_skew_ms": float(np.percentile(skews, 95)) if skews.size else 0.0,
            "max_timestamp_skew_ms": float(np.max(skews)) if skews.size else 0.0,
            "mean_left_model_ms": (
                float(np.mean(self.left_model_ms)) if self.left_model_ms else 0.0
            ),
            "mean_right_model_ms": (
                float(np.mean(self.right_model_ms)) if self.right_model_ms else 0.0
            ),
            "mean_valid_3d_keypoints_per_pair": (
                float(np.mean(self.valid_3d_counts)) if self.valid_3d_counts else 0.0
            ),
            "camera_model": self.calibration.camera_model,
            "baseline": self.calibration.baseline,
            "length_unit": self.calibration.length_unit,
            "coordinate_frame": "left_camera",
            "timestamp_type": (
                next(iter(self.timestamp_types))
                if len(self.timestamp_types) == 1
                else sorted(self.timestamp_types)
                if self.timestamp_types
                else "unknown"
            ),
            "annotated_video": str(self.video_path) if self.save_video else None,
            "results_jsonl": str(self.json_path) if self.save_json else None,
        }
        (self.run_dir / "stereo_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return summary
