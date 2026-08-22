from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

    def read(self) -> StereoFramePair | None:
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
    ) -> None:
        core_config = StereoCameraConfig(
            left_id=int(left_camera_id),
            right_id=int(right_camera_id),
            width=int(config.width),
            height=int(config.height),
            fps=float(config.fps),
            backend=str(config.backend).lower(),
            max_pair_delta_ms=float(max_pair_delta_ms),
            queue_size=int(queue_size),
        )
        core_config.validate()
        self._source = _CoreStereoCameraSource(core_config)
        self._source.start()

        self.name = f"stereo-camera:{core_config.left_id},{core_config.right_id}"
        self.left_width = self._source.left_info.actual_width
        self.left_height = self._source.left_info.actual_height
        self.right_width = self._source.right_info.actual_width
        self.right_height = self._source.right_info.actual_height
        left_fps = self._source.left_info.reported_fps or core_config.fps
        right_fps = self._source.right_info.reported_fps or core_config.fps
        self.fps = min(float(left_fps), float(right_fps))

    def read(self) -> StereoFramePair | None:
        pair = self._source.read(timeout_sec=None)
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

    def read(self) -> StereoFramePair | None:
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
