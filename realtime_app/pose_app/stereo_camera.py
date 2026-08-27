from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import threading
import time
from typing import Callable, Deque, Literal

import cv2
import numpy as np


BackendName = Literal["msmf", "dshow", "auto"]
FrameListener = Callable[[str, "CameraFrame"], None]


@dataclass(frozen=True)
class StereoCameraConfig:
    left_id: int = 1
    right_id: int = 0
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    backend: BackendName = "msmf"
    max_pair_delta_ms: float = 25.0
    queue_size: int = 8

    def validate(self) -> None:
        if self.left_id == self.right_id:
            raise ValueError("left_id and right_id must be different camera indices.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive.")
        if self.fps <= 0:
            raise ValueError("fps must be positive.")
        if self.max_pair_delta_ms <= 0:
            raise ValueError("max_pair_delta_ms must be positive.")
        if self.queue_size < 2:
            raise ValueError("queue_size must be >= 2 because pairing uses one-frame lookahead.")
        if self.backend not in {"msmf", "dshow", "auto"}:
            raise ValueError("backend must be one of: msmf, dshow, auto.")


@dataclass(frozen=True)
class CameraFrame:
    """One decoded frame returned by one physical camera.

    host_return_timestamp_ns is recorded immediately after VideoCapture.read()
    returns on the host. It is NOT a sensor exposure timestamp.
    """

    frame_id: int
    host_return_timestamp_ns: int
    read_duration_ms: float
    image: np.ndarray


@dataclass(frozen=True)
class StereoPair:
    """One-to-one left/right frame pair selected by host-side timestamps.

    ``left_dropped_before`` and ``right_dropped_before`` are cumulative counts
    of pairing-queue drops that happened before this pair was emitted. They do
    not include recorder queue drops from ``tools/capture_stereo.py``.
    """

    pair_id: int
    left: CameraFrame
    right: CameraFrame
    signed_host_delta_ms: float  # right timestamp - left timestamp
    abs_host_delta_ms: float
    left_dropped_before: int = 0
    right_dropped_before: int = 0

    @property
    def pair_host_timestamp_ns(self) -> int:
        return (self.left.host_return_timestamp_ns + self.right.host_return_timestamp_ns) // 2


@dataclass(frozen=True)
class CameraInfo:
    camera_id: int
    requested_width: int
    requested_height: int
    requested_fps: float
    actual_width: int
    actual_height: int
    reported_fps: float
    backend: str


@dataclass(frozen=True)
class StereoCameraStats:
    left_captured: int
    right_captured: int
    stereo_pairs: int

    # Count only frames whose decoded array has been checked to be the exact
    # requested uint8 BGR raster.  This is deliberately separate from
    # captured_count: a backend format/resolution change must fail closed
    # rather than silently entering calibration/triangulation.
    left_validated_frames: int
    right_validated_frames: int
    left_last_decoded_shape: tuple[int, int, int] | None
    right_last_decoded_shape: tuple[int, int, int] | None

    left_read_failures: int
    right_read_failures: int

    left_match_drops: int
    right_match_drops: int

    left_overflow_drops: int
    right_overflow_drops: int

    left_listener_drops: int
    right_listener_drops: int

    left_queue_remaining: int
    right_queue_remaining: int

    left_first_host_timestamp_ns: int | None
    left_last_host_timestamp_ns: int | None
    right_first_host_timestamp_ns: int | None
    right_last_host_timestamp_ns: int | None

    left_capture_span_sec: float | None
    right_capture_span_sec: float | None
    left_measured_fps: float | None
    right_measured_fps: float | None

    last_abs_host_delta_ms: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def _backend_api(name: BackendName) -> int:
    if name == "msmf":
        if not hasattr(cv2, "CAP_MSMF"):
            raise RuntimeError("This OpenCV build does not expose CAP_MSMF.")
        return cv2.CAP_MSMF
    if name == "dshow":
        if not hasattr(cv2, "CAP_DSHOW"):
            raise RuntimeError("This OpenCV build does not expose CAP_DSHOW.")
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def _pair_action(
    left0_ns: int,
    left1_ns: int,
    right0_ns: int,
    right1_ns: int,
    max_delta_ns: int,
) -> Literal["pair", "drop_left", "drop_right"]:
    """Choose the next online pairing action with one-frame lookahead.

    The objective is low-latency monotonic one-to-one pairing. The lookahead
    prevents a common greedy error: accepting L0-R0 merely because it lies
    under the threshold when L1-R0 or L0-R1 is clearly closer.

    This is an ONLINE matcher, not an offline global optimum over a complete
    recording. That distinction is intentional for real-time use.
    """

    d00 = abs(left0_ns - right0_ns)
    d10 = abs(left1_ns - right0_ns)
    d01 = abs(left0_ns - right1_ns)

    # If R0 is closer to L1 than L0, then L0 is already too early relative to
    # the current right stream head and can be discarded.
    if d10 < d00:
        return "drop_left"

    # Symmetric case: L0 is closer to R1 than R0, so R0 can be discarded.
    if d01 < d00:
        return "drop_right"

    if d00 <= max_delta_ns:
        return "pair"

    # Current heads are locally nearest but still outside the acceptance
    # window. The earlier head cannot improve against later future frames.
    if left0_ns < right0_ns:
        return "drop_left"
    return "drop_right"


def _capture_span_and_fps(
    captured_count: int,
    first_timestamp_ns: int | None,
    last_timestamp_ns: int | None,
) -> tuple[float | None, float | None]:
    if (
        captured_count < 2
        or first_timestamp_ns is None
        or last_timestamp_ns is None
        or last_timestamp_ns <= first_timestamp_ns
    ):
        return None, None

    span_sec = (last_timestamp_ns - first_timestamp_ns) / 1_000_000_000.0
    measured_fps = (captured_count - 1) / span_sec
    return span_sec, measured_fps


def validate_camera_frame_image(
    image: object,
    *,
    side: str,
    expected_width: int,
    expected_height: int,
) -> np.ndarray:
    """Fail closed when OpenCV returns anything but the calibrated BGR raster.

    ``VideoCapture`` is requested to produce a particular resolution in
    :meth:`_CameraReader.open`, but a device/backend can still change its
    decoded output later.  The stereo calibration and model-coordinate
    restoration are valid only for the exact raw-camera pixel grid.  Never
    resize, crop, or coerce a returned frame here: report the fault instead.
    """

    if not isinstance(image, np.ndarray):
        raise RuntimeError(
            f"{side} camera returned a non-array frame: {type(image).__name__}."
        )
    if image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError(
            f"{side} camera returned an unexpected decoded layout: "
            f"shape={tuple(image.shape)!r}; expected HxWx3 BGR."
        )
    height, width, channels = (int(value) for value in image.shape)
    if (width, height) != (expected_width, expected_height):
        raise RuntimeError(
            f"{side} camera decoded frame resolution changed: "
            f"got={width}x{height}x{channels}, "
            f"expected={expected_width}x{expected_height}x3. "
            "Refusing to resize or crop a calibrated frame."
        )
    if image.dtype != np.uint8:
        raise RuntimeError(
            f"{side} camera decoded frame dtype changed: got={image.dtype}, "
            "expected=uint8 BGR."
        )
    return image


class _CameraReader:
    def __init__(
        self,
        side: str,
        camera_id: int,
        config: StereoCameraConfig,
        new_frame_condition: threading.Condition,
        frame_listener: FrameListener | None,
    ) -> None:
        self.side = side
        self.camera_id = int(camera_id)
        self.config = config
        self.new_frame_condition = new_frame_condition
        self.frame_listener = frame_listener

        self.capture: cv2.VideoCapture | None = None
        self.info: CameraInfo | None = None

        self.queue: Deque[CameraFrame] = deque()
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.running = False
        self.thread_error: BaseException | None = None

        self.captured_count = 0
        self.validated_frame_count = 0
        self.last_decoded_shape: tuple[int, int, int] | None = None
        self.read_failure_count = 0
        self.overflow_drop_count = 0
        self.match_drop_count = 0
        self.listener_drop_count = 0

        self.first_timestamp_ns: int | None = None
        self.last_timestamp_ns: int | None = None

    def open(self) -> None:
        api = _backend_api(self.config.backend)
        capture = cv2.VideoCapture(self.camera_id, api)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Cannot open {self.side} camera index {self.camera_id} "
                f"with backend={self.config.backend!r}."
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)

        actual_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        actual_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
        try:
            actual_backend = capture.getBackendName()
        except Exception:
            actual_backend = self.config.backend.upper()

        if (actual_width, actual_height) != (self.config.width, self.config.height):
            capture.release()
            raise RuntimeError(
                f"{self.side} camera did not accept the requested resolution: "
                f"requested={self.config.width}x{self.config.height}, "
                f"reported={actual_width}x{actual_height}."
            )

        self.capture = capture
        self.info = CameraInfo(
            camera_id=self.camera_id,
            requested_width=self.config.width,
            requested_height=self.config.height,
            requested_fps=self.config.fps,
            actual_width=actual_width,
            actual_height=actual_height,
            reported_fps=reported_fps,
            backend=actual_backend,
        )

    def start(self) -> None:
        if self.capture is None:
            raise RuntimeError(f"{self.side} camera is not open.")
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError(f"{self.side} camera is already running.")

        self.running = True
        self.thread_error = None
        self.thread = threading.Thread(
            target=self._capture_loop,
            name=f"stereo-{self.side.lower()}-capture",
            daemon=True,
        )
        self.thread.start()

    def _capture_loop(self) -> None:
        assert self.capture is not None
        try:
            while self.running:
                read_start_ns = time.perf_counter_ns()
                ok, image = self.capture.read()
                read_return_ns = time.perf_counter_ns()

                # If stop was requested while read() was blocked, discard that
                # final returned frame so shutdown does not extend the capture
                # interval asymmetrically.
                if not self.running:
                    break

                if not ok or image is None:
                    self.read_failure_count += 1
                    continue

                image = validate_camera_frame_image(
                    image,
                    side=self.side,
                    expected_width=self.config.width,
                    expected_height=self.config.height,
                )
                self.validated_frame_count += 1
                self.last_decoded_shape = tuple(int(value) for value in image.shape)

                if self.last_timestamp_ns is not None and read_return_ns <= self.last_timestamp_ns:
                    raise RuntimeError(
                        f"Non-monotonic host timestamp on {self.side}: "
                        f"previous={self.last_timestamp_ns}, current={read_return_ns}."
                    )

                if self.first_timestamp_ns is None:
                    self.first_timestamp_ns = read_return_ns
                self.last_timestamp_ns = read_return_ns

                frame = CameraFrame(
                    frame_id=self.captured_count,
                    host_return_timestamp_ns=read_return_ns,
                    read_duration_ms=(read_return_ns - read_start_ns) / 1_000_000.0,
                    image=image,
                )
                self.captured_count += 1

                with self.lock:
                    if len(self.queue) >= self.config.queue_size:
                        self.queue.popleft()
                        self.overflow_drop_count += 1
                    self.queue.append(frame)

                # Listener must be non-blocking. capture_stereo.py implements
                # it as a queue put; listener errors are counted, not allowed
                # to kill camera acquisition.
                if self.frame_listener is not None:
                    try:
                        self.frame_listener(self.side, frame)
                    except Exception:
                        self.listener_drop_count += 1

                with self.new_frame_condition:
                    self.new_frame_condition.notify_all()

        except BaseException as exc:
            self.thread_error = exc
            self.running = False
            with self.new_frame_condition:
                self.new_frame_condition.notify_all()

    def request_stop(self) -> None:
        # Deliberately separate "signal stop" from "join/release" so BOTH
        # cameras can be asked to stop before either thread is joined. This
        # prevents one camera from continuing to capture while the other is
        # already shutting down.
        self.running = False

    def finish_stop(self) -> None:
        if self.thread is not None:
            self.thread.join(timeout=2.0)

        # Some backends can leave read() blocked. Releasing the capture object
        # can unblock it; then give the thread one final short join window.
        if self.thread is not None and self.thread.is_alive() and self.capture is not None:
            self.capture.release()
            self.capture = None
            self.thread.join(timeout=1.0)

        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def queue_length(self) -> int:
        with self.lock:
            return len(self.queue)


class StereoCameraSource:
    """Dual-camera acquisition and online one-to-one nearest-time pairing.

    Scope: camera acquisition only. This class knows nothing about pose,
    calibration or triangulation.

    The pairing timestamp is a host-side timestamp recorded immediately after
    cv2.VideoCapture.read() returns. It must not be described as exposure
    synchronization error.
    """

    TIMESTAMP_TYPE = "host_read_return_perf_counter_ns"

    def __init__(
        self,
        config: StereoCameraConfig | None = None,
        *,
        frame_listener: FrameListener | None = None,
    ) -> None:
        self.config = config or StereoCameraConfig()
        self.config.validate()
        self.frame_listener = frame_listener

        self._condition = threading.Condition()
        self._left = _CameraReader(
            "LEFT", self.config.left_id, self.config, self._condition, frame_listener
        )
        self._right = _CameraReader(
            "RIGHT", self.config.right_id, self.config, self._condition, frame_listener
        )
        self._started = False
        self._closed = False
        self._pair_id = 0
        self._last_abs_host_delta_ms: float | None = None

    @property
    def left_info(self) -> CameraInfo:
        if self._left.info is None:
            raise RuntimeError("StereoCameraSource has not been started.")
        return self._left.info

    @property
    def right_info(self) -> CameraInfo:
        if self._right.info is None:
            raise RuntimeError("StereoCameraSource has not been started.")
        return self._right.info

    def start(self) -> None:
        if self._started:
            return
        if self._closed:
            raise RuntimeError("A closed StereoCameraSource cannot be restarted.")

        self._left.open()
        try:
            self._right.open()
        except Exception:
            self._left.request_stop()
            self._left.finish_stop()
            raise

        try:
            # Both devices are opened/configured before either capture thread
            # starts. Cameras remain free-running; this is not hardware sync.
            self._left.start()
            self._right.start()
        except Exception:
            self.close()
            raise

        self._started = True

    def _raise_thread_errors(self) -> None:
        if self._left.thread_error is not None:
            raise RuntimeError("LEFT capture thread failed.") from self._left.thread_error
        if self._right.thread_error is not None:
            raise RuntimeError("RIGHT capture thread failed.") from self._right.thread_error

    def _try_pair(self) -> StereoPair | None:
        max_delta_ns = int(round(self.config.max_pair_delta_ms * 1_000_000.0))

        # Lock order is always LEFT -> RIGHT.
        with self._left.lock:
            with self._right.lock:
                while len(self._left.queue) >= 2 and len(self._right.queue) >= 2:
                    left0 = self._left.queue[0]
                    left1 = self._left.queue[1]
                    right0 = self._right.queue[0]
                    right1 = self._right.queue[1]

                    action = _pair_action(
                        left0.host_return_timestamp_ns,
                        left1.host_return_timestamp_ns,
                        right0.host_return_timestamp_ns,
                        right1.host_return_timestamp_ns,
                        max_delta_ns,
                    )

                    if action == "drop_left":
                        self._left.queue.popleft()
                        self._left.match_drop_count += 1
                        continue

                    if action == "drop_right":
                        self._right.queue.popleft()
                        self._right.match_drop_count += 1
                        continue

                    left = self._left.queue.popleft()
                    right = self._right.queue.popleft()

                    signed_delta_ms = (
                        right.host_return_timestamp_ns - left.host_return_timestamp_ns
                    ) / 1_000_000.0

                    pair = StereoPair(
                        pair_id=self._pair_id,
                        left=left,
                        right=right,
                        signed_host_delta_ms=signed_delta_ms,
                        abs_host_delta_ms=abs(signed_delta_ms),
                        left_dropped_before=(
                            self._left.match_drop_count + self._left.overflow_drop_count
                        ),
                        right_dropped_before=(
                            self._right.match_drop_count + self._right.overflow_drop_count
                        ),
                    )
                    self._pair_id += 1
                    self._last_abs_host_delta_ms = pair.abs_host_delta_ms
                    return pair

        return None

    def read(self, timeout_sec: float | None = 2.0) -> StereoPair | None:
        """Return the next paired frame, or None on timeout.

        The source must already be started. A timeout of None waits until a
        pair is available or the source is closed/failed.
        """

        if not self._started:
            raise RuntimeError("Call start() before read().")
        if timeout_sec is not None and timeout_sec < 0:
            raise ValueError("timeout_sec must be non-negative or None.")

        deadline = None if timeout_sec is None else time.monotonic() + timeout_sec

        while True:
            self._raise_thread_errors()

            pair = self._try_pair()
            if pair is not None:
                return pair

            if self._closed:
                return None

            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                wait_for = min(remaining, 0.25)
            else:
                wait_for = 0.25

            with self._condition:
                self._condition.wait(timeout=wait_for)

    def stats(self) -> StereoCameraStats:
        left_span, left_measured_fps = _capture_span_and_fps(
            self._left.captured_count,
            self._left.first_timestamp_ns,
            self._left.last_timestamp_ns,
        )
        right_span, right_measured_fps = _capture_span_and_fps(
            self._right.captured_count,
            self._right.first_timestamp_ns,
            self._right.last_timestamp_ns,
        )

        return StereoCameraStats(
            left_captured=self._left.captured_count,
            right_captured=self._right.captured_count,
            stereo_pairs=self._pair_id,
            left_validated_frames=self._left.validated_frame_count,
            right_validated_frames=self._right.validated_frame_count,
            left_last_decoded_shape=self._left.last_decoded_shape,
            right_last_decoded_shape=self._right.last_decoded_shape,
            left_read_failures=self._left.read_failure_count,
            right_read_failures=self._right.read_failure_count,
            left_match_drops=self._left.match_drop_count,
            right_match_drops=self._right.match_drop_count,
            left_overflow_drops=self._left.overflow_drop_count,
            right_overflow_drops=self._right.overflow_drop_count,
            left_listener_drops=self._left.listener_drop_count,
            right_listener_drops=self._right.listener_drop_count,
            left_queue_remaining=self._left.queue_length(),
            right_queue_remaining=self._right.queue_length(),
            left_first_host_timestamp_ns=self._left.first_timestamp_ns,
            left_last_host_timestamp_ns=self._left.last_timestamp_ns,
            right_first_host_timestamp_ns=self._right.first_timestamp_ns,
            right_last_host_timestamp_ns=self._right.last_timestamp_ns,
            left_capture_span_sec=left_span,
            right_capture_span_sec=right_span,
            left_measured_fps=left_measured_fps,
            right_measured_fps=right_measured_fps,
            last_abs_host_delta_ms=self._last_abs_host_delta_ms,
        )

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True

        # Signal BOTH camera loops before joining either one. In the old
        # implementation LEFT was fully stopped before RIGHT was told to stop,
        # which let RIGHT keep capturing during shutdown and could create fake
        # end-of-run imbalance/overflow.
        self._left.request_stop()
        self._right.request_stop()

        with self._condition:
            self._condition.notify_all()

        self._left.finish_stop()
        self._right.finish_stop()

    def __enter__(self) -> "StereoCameraSource":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
