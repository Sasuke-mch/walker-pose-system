from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path

import numpy as np

from .config import CameraConfig
from .sources import SourceFrame, VideoSource
from .stereo_camera import (
    StereoCameraConfig,
    StereoCameraSource as _CoreStereoCameraSource,
    StereoPair as CameraStereoPair,
)


@dataclass(frozen=True)
class StereoFramePair:
    """Runtime stereo pair consumed by pose/triangulation code.

    ``timestamp_sec`` is a timeline value suitable for model metadata.
    For live cameras it comes from the host monotonic clock recorded after
    ``VideoCapture.read()`` returns. It is not an exposure timestamp.
    """

    pair_id: int
    left: SourceFrame
    right: SourceFrame
    timestamp_skew_sec: float
    dropped_left: int = 0
    dropped_right: int = 0
    timestamp_type: str = "source_timeline"
    left_host_timestamp_ns: int | None = None
    right_host_timestamp_ns: int | None = None
    signed_host_delta_ms: float | None = None
    left_read_duration_ms: float | None = None
    right_read_duration_ms: float | None = None

    @property
    def timestamp_sec(self) -> float:
        return 0.5 * (self.left.timestamp_sec + self.right.timestamp_sec)


class StereoFrameSource:
    name: str
    is_live: bool
    left_width: int
    left_height: int
    right_width: int
    right_height: int
    fps: float

    def read(self, timeout_sec: float | None = None) -> StereoFramePair | None:
        raise NotImplementedError

    def close(self) -> None:
        pass


def _camera_pair_to_runtime_pair(pair: CameraStereoPair) -> StereoFramePair:
    left_timestamp_sec = pair.left.host_return_timestamp_ns / 1_000_000_000.0
    right_timestamp_sec = pair.right.host_return_timestamp_ns / 1_000_000_000.0
    return StereoFramePair(
        pair_id=pair.pair_id,
        left=SourceFrame(pair.left.frame_id, left_timestamp_sec, pair.left.image),
        right=SourceFrame(pair.right.frame_id, right_timestamp_sec, pair.right.image),
        timestamp_skew_sec=pair.abs_host_delta_ms / 1000.0,
        dropped_left=pair.left_dropped_before,
        dropped_right=pair.right_dropped_before,
        timestamp_type=_CoreStereoCameraSource.TIMESTAMP_TYPE,
        left_host_timestamp_ns=pair.left.host_return_timestamp_ns,
        right_host_timestamp_ns=pair.right.host_return_timestamp_ns,
        signed_host_delta_ms=pair.signed_host_delta_ms,
        left_read_duration_ms=pair.left.read_duration_ms,
        right_read_duration_ms=pair.right.read_duration_ms,
    )


class StereoCameraInput(StereoFrameSource):
    """Adapter that exposes the validated dual-camera module to run_stereo.py.

    The actual camera acquisition and pairing algorithm live only in
    ``pose_app.stereo_camera``. This wrapper exists so the stereo pose pipeline
    can share one input interface with offline stereo videos without duplicating
    camera logic.
    """

    is_live = True

    def __init__(
        self,
        left_camera_id: int,
        right_camera_id: int,
        config: CameraConfig,
        queue_size: int = 8,
        max_pair_delta_ms: float = 25.0,
        backend: str | None = None,
    ) -> None:
        core_config = StereoCameraConfig(
            left_id=int(left_camera_id),
            right_id=int(right_camera_id),
            width=int(config.width),
            height=int(config.height),
            fps=float(config.fps),
            backend=(
                str(backend).lower()
                if backend is not None
                else str(config.backend).lower()
            ),
            max_pair_delta_ms=float(max_pair_delta_ms),
            queue_size=int(queue_size),
        )
        core_config.validate()
        self._source = _CoreStereoCameraSource(core_config)
        self._source.start()

        self.name = (
            f"stereo-camera:{core_config.left_id},{core_config.right_id}"
            f"@{core_config.backend}"
        )
        self.left_width = self._source.left_info.actual_width
        self.left_height = self._source.left_info.actual_height
        self.right_width = self._source.right_info.actual_width
        self.right_height = self._source.right_info.actual_height
        left_fps = self._source.left_info.reported_fps or core_config.fps
        right_fps = self._source.right_info.reported_fps or core_config.fps
        self.fps = min(float(left_fps), float(right_fps))

    def read(self, timeout_sec: float | None = None) -> StereoFramePair | None:
        pair = self._source.read(timeout_sec=timeout_sec)
        if pair is None:
            return None
        return _camera_pair_to_runtime_pair(pair)

    def close(self) -> None:
        self._source.close()

    def stats(self) -> dict:
        return self._source.stats().to_dict()


class StereoVideoSource(StereoFrameSource):
    """Read two already-aligned videos by frame index.

    This mode assumes the two videos are already aligned frame-for-frame. For
    raw asynchronous camera recordings produced by ``tools/capture_stereo.py``,
    use the accompanying ``stereo_pairs.csv`` when building an offline dataset;
    do not assume left frame N corresponds to right frame N.
    """

    is_live = False

    def __init__(
        self,
        left_video: str | Path,
        right_video: str | Path,
        left_start_frame: int = 0,
        right_start_frame: int = 0,
        loop: bool = False,
    ) -> None:
        self.left_source = VideoSource(left_video, left_start_frame, loop)
        try:
            self.right_source = VideoSource(right_video, right_start_frame, loop)
        except Exception:
            self.left_source.close()
            raise
        self.name = f"stereo-video:{self.left_source.path}|{self.right_source.path}"
        self.left_width = self.left_source.width
        self.left_height = self.left_source.height
        self.right_width = self.right_source.width
        self.right_height = self.right_source.height
        self.fps = min(self.left_source.fps, self.right_source.fps)
        self.pair_id = 0

    def read(self, timeout_sec: float | None = None) -> StereoFramePair | None:
        left = self.left_source.read()
        right = self.right_source.read()
        if left is None or right is None:
            return None
        pair = StereoFramePair(
            pair_id=self.pair_id,
            left=left,
            right=right,
            timestamp_skew_sec=abs(left.timestamp_sec - right.timestamp_sec),
            timestamp_type="video_frame_index_over_fps",
        )
        self.pair_id += 1
        return pair

    def close(self) -> None:
        self.left_source.close()
        self.right_source.close()


@dataclass(frozen=True)
class _RecordedStereoPair:
    """One validated row of ``capture_stereo.py`` recording metadata."""

    pair_id: int
    left_frame_id: int
    right_frame_id: int
    left_recorded_index: int
    right_recorded_index: int
    left_host_timestamp_ns: int
    right_host_timestamp_ns: int
    signed_host_delta_ms: float
    abs_host_delta_ms: float
    left_read_duration_ms: float
    right_read_duration_ms: float


class StereoCaptureReplaySource(StereoFrameSource):
    """Replay exactly the frame pairs accepted by ``capture_stereo.py``.

    ``left_capture.avi`` and ``right_capture.avi`` are independent camera
    recordings.  Their decoded AVI positions therefore are *not* a stereo
    correspondence.  This source uses ``stereo_pairs.csv`` to select each
    historical one-to-one pair, and uses ``left_frames.csv`` / ``right_frames.csv``
    to translate a camera frame ID into its encoded AVI position.  It refuses
    incomplete or ambiguous recordings rather than silently falling back to
    same-index pairing.
    """

    is_live = False

    def __init__(self, capture_dir: str | Path) -> None:
        self.capture_dir = Path(capture_dir).resolve()
        if not self.capture_dir.is_dir():
            raise NotADirectoryError(
                f"双目采集目录不存在：{self.capture_dir}"
            )

        self.left_video_path = self.capture_dir / "left_capture.avi"
        self.right_video_path = self.capture_dir / "right_capture.avi"
        self.left_frames_path = self.capture_dir / "left_frames.csv"
        self.right_frames_path = self.capture_dir / "right_frames.csv"
        self.pairs_path = self.capture_dir / "stereo_pairs.csv"
        required_paths = (
            self.left_video_path,
            self.right_video_path,
            self.left_frames_path,
            self.right_frames_path,
            self.pairs_path,
        )
        missing = [str(path) for path in required_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "按真实帧对离线重放需要同一采集目录中的 left/right_capture.avi、"
                "left/right_frames.csv 与 stereo_pairs.csv；缺少："
                + "; ".join(missing)
            )

        self.left_recorded_indices = self._load_recorded_frame_indices(
            self.left_frames_path, "LEFT"
        )
        self.right_recorded_indices = self._load_recorded_frame_indices(
            self.right_frames_path, "RIGHT"
        )
        self.pairs = self._load_pairs(
            self.pairs_path,
            self.left_recorded_indices,
            self.right_recorded_indices,
        )

        self.left_source = VideoSource(self.left_video_path, loop=False)
        try:
            self.right_source = VideoSource(self.right_video_path, loop=False)
        except Exception:
            self.left_source.close()
            raise
        try:
            self._validate_video_frame_counts()
        except Exception:
            self.left_source.close()
            self.right_source.close()
            raise

        self.name = f"stereo-capture-replay:{self.capture_dir}"
        self.left_width = self.left_source.width
        self.left_height = self.left_source.height
        self.right_width = self.right_source.width
        self.right_height = self.right_source.height
        self.fps = min(self.left_source.fps, self.right_source.fps)
        self.next_pair_index = 0
        self.capture_summary = self._load_optional_summary()

    @staticmethod
    def _load_recorded_frame_indices(path: Path, side: str) -> dict[int, int]:
        required_columns = {
            "frame_id",
            "host_return_timestamp_ns",
            "read_duration_ms",
        }
        records: dict[int, int] = {}
        with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
                raise ValueError(
                    f"{side}逐帧记录字段不完整：{path}；需要{sorted(required_columns)!r}。"
                )
            for recorded_index, row in enumerate(reader, 2):
                try:
                    frame_id = int(row["frame_id"])
                    host_timestamp_ns = int(row["host_return_timestamp_ns"])
                    read_duration_ms = float(row["read_duration_ms"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{side}逐帧记录无效：{path}:{recorded_index}。"
                    ) from exc
                if frame_id < 0 or host_timestamp_ns < 0 or not math.isfinite(read_duration_ms):
                    raise ValueError(
                        f"{side}逐帧记录包含非法值：{path}:{recorded_index}。"
                    )
                if frame_id in records:
                    raise ValueError(
                        f"{side}逐帧记录含重复frame_id={frame_id}：{path}:{recorded_index}。"
                    )
                records[frame_id] = recorded_index - 2
        if not records:
            raise ValueError(f"{side}逐帧记录为空：{path}。")
        return records

    @staticmethod
    def _load_pairs(
        path: Path,
        left_recorded_indices: dict[int, int],
        right_recorded_indices: dict[int, int],
    ) -> list[_RecordedStereoPair]:
        required_columns = {
            "pair_id",
            "left_frame_id",
            "right_frame_id",
            "left_host_return_timestamp_ns",
            "right_host_return_timestamp_ns",
            "signed_host_delta_ms_right_minus_left",
            "abs_host_delta_ms",
            "left_read_duration_ms",
            "right_read_duration_ms",
        }
        pairs: list[_RecordedStereoPair] = []
        previous_pair_id: int | None = None
        previous_left_frame_id: int | None = None
        previous_right_frame_id: int | None = None
        with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
                raise ValueError(
                    f"双目配对记录字段不完整：{path}；需要{sorted(required_columns)!r}。"
                )
            for line_number, row in enumerate(reader, 2):
                try:
                    pair_id = int(row["pair_id"])
                    left_frame_id = int(row["left_frame_id"])
                    right_frame_id = int(row["right_frame_id"])
                    left_host_timestamp_ns = int(row["left_host_return_timestamp_ns"])
                    right_host_timestamp_ns = int(row["right_host_return_timestamp_ns"])
                    signed_host_delta_ms = float(
                        row["signed_host_delta_ms_right_minus_left"]
                    )
                    abs_host_delta_ms = float(row["abs_host_delta_ms"])
                    left_read_duration_ms = float(row["left_read_duration_ms"])
                    right_read_duration_ms = float(row["right_read_duration_ms"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"双目配对记录无效：{path}:{line_number}。") from exc
                values = (
                    signed_host_delta_ms,
                    abs_host_delta_ms,
                    left_read_duration_ms,
                    right_read_duration_ms,
                )
                if (
                    pair_id < 0
                    or left_frame_id < 0
                    or right_frame_id < 0
                    or left_host_timestamp_ns < 0
                    or right_host_timestamp_ns < 0
                    or not all(math.isfinite(value) for value in values)
                    or abs_host_delta_ms < 0
                ):
                    raise ValueError(f"双目配对记录包含非法值：{path}:{line_number}。")
                if (
                    previous_pair_id is not None
                    and pair_id <= previous_pair_id
                ):
                    raise ValueError(
                        f"双目配对pair_id必须严格递增：{path}:{line_number}。"
                    )
                if (
                    previous_left_frame_id is not None
                    and left_frame_id <= previous_left_frame_id
                ):
                    raise ValueError(
                        f"左相机frame_id必须在配对记录中严格递增：{path}:{line_number}。"
                    )
                if (
                    previous_right_frame_id is not None
                    and right_frame_id <= previous_right_frame_id
                ):
                    raise ValueError(
                        f"右相机frame_id必须在配对记录中严格递增：{path}:{line_number}。"
                    )
                if left_frame_id not in left_recorded_indices:
                    raise RuntimeError(
                        f"配对所需LEFT frame_id={left_frame_id}未写入AVI；"
                        f"采集记录不完整，不能安全重放：{path}:{line_number}。"
                    )
                if right_frame_id not in right_recorded_indices:
                    raise RuntimeError(
                        f"配对所需RIGHT frame_id={right_frame_id}未写入AVI；"
                        f"采集记录不完整，不能安全重放：{path}:{line_number}。"
                    )
                measured_signed_delta_ms = (
                    right_host_timestamp_ns - left_host_timestamp_ns
                ) / 1_000_000.0
                if not math.isclose(
                    signed_host_delta_ms, measured_signed_delta_ms, abs_tol=0.01
                ) or not math.isclose(
                    abs_host_delta_ms, abs(measured_signed_delta_ms), abs_tol=0.01
                ):
                    raise ValueError(
                        f"双目配对时间字段不自洽：{path}:{line_number}。"
                    )
                pairs.append(
                    _RecordedStereoPair(
                        pair_id=pair_id,
                        left_frame_id=left_frame_id,
                        right_frame_id=right_frame_id,
                        left_recorded_index=left_recorded_indices[left_frame_id],
                        right_recorded_index=right_recorded_indices[right_frame_id],
                        left_host_timestamp_ns=left_host_timestamp_ns,
                        right_host_timestamp_ns=right_host_timestamp_ns,
                        signed_host_delta_ms=signed_host_delta_ms,
                        abs_host_delta_ms=abs_host_delta_ms,
                        left_read_duration_ms=left_read_duration_ms,
                        right_read_duration_ms=right_read_duration_ms,
                    )
                )
                previous_pair_id = pair_id
                previous_left_frame_id = left_frame_id
                previous_right_frame_id = right_frame_id
        if not pairs:
            raise ValueError(f"双目配对记录为空：{path}。")
        return pairs

    def _validate_video_frame_counts(self) -> None:
        for side, source, records in (
            ("LEFT", self.left_source, self.left_recorded_indices),
            ("RIGHT", self.right_source, self.right_recorded_indices),
        ):
            if source.total_frames is None:
                continue
            if len(records) != source.total_frames:
                raise RuntimeError(
                    f"{side} AVI帧数与{side.lower()}_frames.csv行数不一致："
                    f"video={source.total_frames} csv={len(records)}；不能安全重放。"
                )

    def _load_optional_summary(self) -> dict | None:
        path = self.capture_dir / "summary.json"
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"采集summary.json无法读取：{path}。") from exc
        if not isinstance(value, dict):
            raise ValueError(f"采集summary.json不是JSON对象：{path}。")
        return value

    @staticmethod
    def _read_recorded_index(source: VideoSource, index: int, side: str) -> SourceFrame:
        if index < source.current:
            raise RuntimeError(
                f"{side}回放索引倒退：目标={index}，当前={source.current}；"
                "stereo_pairs.csv不是单调一对一配对记录。"
            )
        frame: SourceFrame | None = None
        while source.current <= index:
            frame = source.read()
            if frame is None:
                raise RuntimeError(
                    f"{side} AVI在读取所需编码帧{index}前结束。"
                )
        assert frame is not None and frame.frame_id == index
        return frame

    def read(self, timeout_sec: float | None = None) -> StereoFramePair | None:
        del timeout_sec
        if self.next_pair_index >= len(self.pairs):
            return None
        recorded = self.pairs[self.next_pair_index]
        left_recorded = self._read_recorded_index(
            self.left_source, recorded.left_recorded_index, "LEFT"
        )
        right_recorded = self._read_recorded_index(
            self.right_source, recorded.right_recorded_index, "RIGHT"
        )
        emitted_before = self.next_pair_index
        self.next_pair_index += 1
        return StereoFramePair(
            pair_id=recorded.pair_id,
            left=SourceFrame(
                recorded.left_frame_id,
                recorded.left_host_timestamp_ns / 1_000_000_000.0,
                left_recorded.image,
            ),
            right=SourceFrame(
                recorded.right_frame_id,
                recorded.right_host_timestamp_ns / 1_000_000_000.0,
                right_recorded.image,
            ),
            timestamp_skew_sec=recorded.abs_host_delta_ms / 1000.0,
            dropped_left=recorded.left_frame_id - emitted_before,
            dropped_right=recorded.right_frame_id - emitted_before,
            timestamp_type="capture_stereo_csv_host_read_return",
            left_host_timestamp_ns=recorded.left_host_timestamp_ns,
            right_host_timestamp_ns=recorded.right_host_timestamp_ns,
            signed_host_delta_ms=recorded.signed_host_delta_ms,
            left_read_duration_ms=recorded.left_read_duration_ms,
            right_read_duration_ms=recorded.right_read_duration_ms,
        )

    def integrity_metadata(self) -> dict:
        recording_integrity = None
        if self.capture_summary is not None:
            recording_integrity = self.capture_summary.get("recording_integrity")
        return {
            "mode": "capture_stereo_csv_true_pair_replay",
            "capture_dir": str(self.capture_dir),
            "stereo_pairs_csv": str(self.pairs_path),
            "left_frames_csv": str(self.left_frames_path),
            "right_frames_csv": str(self.right_frames_path),
            "validated_pair_count": len(self.pairs),
            "left_recorded_frame_count": len(self.left_recorded_indices),
            "right_recorded_frame_count": len(self.right_recorded_indices),
            "timestamp_type": "host_read_return_restored_from_stereo_pairs_csv",
            "capture_summary_recording_integrity": recording_integrity,
        }

    def close(self) -> None:
        self.left_source.close()
        self.right_source.close()


class StereoSideBySideVideoSource(StereoFrameSource):
    """Replay a single side-by-side stereo recording without resampling it.

    Each decoded source frame is split at one explicit vertical boundary.  The
    left and right panels are copied directly from the decoded raster; there
    is intentionally no resize, crop within either panel, rectification, or
    re-encoding step before the pose model receives them.  It is useful for a
    controlled replay of a previously recorded full-resolution stereo video:
    live camera acquisition and host-time matching are removed from the
    experiment while the model still sees the recorded pixels.
    """

    is_live = False

    def __init__(
        self,
        video: str | Path,
        *,
        left_panel_width: int | None = None,
        metadata_jsonl: str | Path | None = None,
        start_frame: int = 0,
        loop: bool = False,
    ) -> None:
        self.source = VideoSource(video, start_frame=start_frame, loop=loop)
        self.name = f"stereo-side-by-side-video:{self.source.path}"
        self.input_width = int(self.source.width)
        self.input_height = int(self.source.height)
        if self.input_width <= 1 or self.input_height <= 0:
            self.source.close()
            raise RuntimeError(
                "Side-by-side video reported an invalid input size: "
                f"{self.input_width}x{self.input_height}."
            )

        if left_panel_width is None:
            if self.input_width % 2:
                self.source.close()
                raise ValueError(
                    "An odd-width side-by-side video needs --sbs-left-panel-width "
                    "so its split boundary is explicit."
                )
            left_panel_width = self.input_width // 2
        self.left_width = int(left_panel_width)
        self.right_width = self.input_width - self.left_width
        if self.left_width <= 0 or self.right_width <= 0:
            self.source.close()
            raise ValueError(
                "Side-by-side split must leave a positive-width left and right panel: "
                f"input={self.input_width}, left={self.left_width}, right={self.right_width}."
            )

        self.left_height = self.input_height
        self.right_height = self.input_height
        self.fps = self.source.fps
        self.decoded_frames = 0
        self.metadata_path = (
            Path(metadata_jsonl).resolve() if metadata_jsonl is not None else None
        )
        try:
            self.pair_metadata = self._load_pair_metadata(self.metadata_path)
        except Exception:
            self.source.close()
            raise

    @staticmethod
    def _load_pair_metadata(path: Path | None) -> dict[int, dict]:
        """Load original pair timing when replaying a previous run's JSONL.

        The visualizer video puts the two already-paired camera images in one
        video frame.  It does not itself preserve their original host-time
        delta, so treating that delta as a real zero would be false.  The
        optional JSONL sidecar restores it exactly for a saved run.
        """

        if path is None:
            return {}
        if not path.is_file():
            raise FileNotFoundError(f"Side-by-side replay metadata does not exist: {path}")
        records: dict[int, dict] = {}
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                pair_id = int(item["pair_id"])
                skew_ms = float(item["timestamp_skew_ms"])
                left_frame_id = int(item["left_frame_id"])
                right_frame_id = int(item["right_frame_id"])
                left_timestamp_sec = float(item["left_timestamp_sec"])
                right_timestamp_sec = float(item["right_timestamp_sec"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid side-by-side replay metadata at {path}:{line_number}."
                ) from exc
            values = (skew_ms, left_timestamp_sec, right_timestamp_sec)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"Non-finite timing values in side-by-side replay metadata at "
                    f"{path}:{line_number}."
                )
            if pair_id in records:
                raise ValueError(
                    f"Duplicate pair_id={pair_id} in side-by-side replay metadata: {path}."
                )
            records[pair_id] = {
                "timestamp_skew_sec": abs(skew_ms) / 1000.0,
                "timestamp_type": str(
                    item.get("timestamp_type", "replayed_side_by_side_pair")
                ),
                "left_frame_id": left_frame_id,
                "right_frame_id": right_frame_id,
                "left_timestamp_sec": left_timestamp_sec,
                "right_timestamp_sec": right_timestamp_sec,
                "left_host_timestamp_ns": item.get("left_host_timestamp_ns"),
                "right_host_timestamp_ns": item.get("right_host_timestamp_ns"),
                "signed_host_delta_ms": item.get(
                    "signed_host_delta_ms_right_minus_left"
                ),
                "left_read_duration_ms": item.get("left_read_duration_ms"),
                "right_read_duration_ms": item.get("right_read_duration_ms"),
                "dropped_left": int(item.get("dropped_left", 0)),
                "dropped_right": int(item.get("dropped_right", 0)),
            }
        if not records:
            raise ValueError(f"Side-by-side replay metadata is empty: {path}")
        return records

    def _validate_composite(self, image) -> None:
        if not isinstance(image, np.ndarray):
            raise RuntimeError("Side-by-side video decoder returned a non-image frame.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(
                "Side-by-side video decoder returned an unexpected layout: "
                f"shape={tuple(image.shape)!r}; expected HxWx3 BGR."
            )
        height, width, channels = (int(value) for value in image.shape)
        if (width, height) != (self.input_width, self.input_height):
            raise RuntimeError(
                "Side-by-side video frame dimensions changed mid-stream: "
                f"got={width}x{height}x{channels}, "
                f"expected={self.input_width}x{self.input_height}x3. "
                "Refusing to resize or crop the replay source."
            )
        if image.dtype != np.uint8:
            raise RuntimeError(
                "Side-by-side video decoder returned an unexpected dtype: "
                f"got={image.dtype}, expected=uint8 BGR."
            )

    def read(self, timeout_sec: float | None = None) -> StereoFramePair | None:
        # ``timeout_sec`` is accepted for the common StereoFrameSource API;
        # file decoding is sequential and does not have an asynchronous wait.
        del timeout_sec
        frame = self.source.read()
        if frame is None:
            return None
        self._validate_composite(frame.image)
        self.decoded_frames += 1

        left_image = frame.image[:, : self.left_width].copy()
        right_image = frame.image[:, self.left_width :].copy()
        metadata = self.pair_metadata.get(frame.frame_id)
        if self.metadata_path is not None and metadata is None:
            raise RuntimeError(
                "Side-by-side video has a frame without a matching JSONL pair_id: "
                f"frame_id={frame.frame_id}, metadata={self.metadata_path}."
            )
        if metadata is None:
            # Decoding one composite frame does not prove that its two panels
            # came from one exposure.  Keep the historical skew explicitly
            # unknown unless the accompanying JSONL supplies it.
            left = SourceFrame(frame.frame_id, frame.timestamp_sec, left_image)
            right = SourceFrame(frame.frame_id, frame.timestamp_sec, right_image)
            timestamp_skew_sec = 0.0
            timestamp_type = "side_by_side_video_original_skew_unknown"
            fields = {}
        else:
            left = SourceFrame(
                metadata["left_frame_id"], metadata["left_timestamp_sec"], left_image
            )
            right = SourceFrame(
                metadata["right_frame_id"], metadata["right_timestamp_sec"], right_image
            )
            timestamp_skew_sec = metadata["timestamp_skew_sec"]
            timestamp_type = metadata["timestamp_type"]
            fields = metadata
        return StereoFramePair(
            pair_id=frame.frame_id,
            left=left,
            right=right,
            timestamp_skew_sec=timestamp_skew_sec,
            dropped_left=int(fields.get("dropped_left", 0)),
            dropped_right=int(fields.get("dropped_right", 0)),
            timestamp_type=timestamp_type,
            left_host_timestamp_ns=fields.get("left_host_timestamp_ns"),
            right_host_timestamp_ns=fields.get("right_host_timestamp_ns"),
            signed_host_delta_ms=fields.get("signed_host_delta_ms"),
            left_read_duration_ms=fields.get("left_read_duration_ms"),
            right_read_duration_ms=fields.get("right_read_duration_ms"),
        )

    def integrity_metadata(self) -> dict:
        return {
            "mode": "side_by_side_video_split_without_resize",
            "decoded_frames": self.decoded_frames,
            "input_frame_size": [self.input_width, self.input_height],
            "left_panel_x_range": [0, self.left_width],
            "right_panel_x_range": [self.left_width, self.input_width],
            "left_panel_size": [self.left_width, self.left_height],
            "right_panel_size": [self.right_width, self.right_height],
            "timing_metadata_jsonl": str(self.metadata_path) if self.metadata_path else None,
            "timing_metadata_records": len(self.pair_metadata),
            "timestamp_type": (
                "restored_from_sidecar_jsonl"
                if self.metadata_path is not None
                else "original_skew_unknown"
            ),
        }

    def close(self) -> None:
        self.source.close()
