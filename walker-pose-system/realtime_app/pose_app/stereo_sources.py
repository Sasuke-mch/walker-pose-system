from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time

import cv2

from .config import CameraConfig
from .sources import SourceFrame, VideoSource


@dataclass
class StereoFramePair:
    pair_id: int
    left: SourceFrame
    right: SourceFrame
    timestamp_skew_sec: float
    dropped_left: int = 0
    dropped_right: int = 0

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


def _open_camera(camera_id: int, config: CameraConfig) -> cv2.VideoCapture:
    if config.backend == "dshow" and hasattr(cv2, "CAP_DSHOW"):
        capture = cv2.VideoCapture(int(camera_id), cv2.CAP_DSHOW)
    elif config.backend == "msmf" and hasattr(cv2, "CAP_MSMF"):
        capture = cv2.VideoCapture(int(camera_id), cv2.CAP_MSMF)
    else:
        capture = cv2.VideoCapture(int(camera_id))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    capture.set(cv2.CAP_PROP_FPS, config.fps)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"无法打开摄像头{camera_id}。")
    return capture


class StereoCameraSource(StereoFrameSource):
    """Continuously capture two independent cameras and pair frames by host time.

    The timestamps are taken immediately after ``VideoCapture.read`` returns.
    They are host receive times, not hardware exposure timestamps.  This is
    intentional: ordinary UVC webcams usually do not expose reliable exposure
    timestamps, so the measured skew must be recorded rather than mistaken for
    hardware synchronization.
    """

    is_live = True

    def __init__(
        self,
        left_camera_id: int,
        right_camera_id: int,
        config: CameraConfig,
        queue_size: int = 8,
    ) -> None:
        if int(left_camera_id) == int(right_camera_id):
            raise ValueError("左右摄像头编号不能相同。")
        self.left_camera_id = int(left_camera_id)
        self.right_camera_id = int(right_camera_id)
        self.name = f"stereo-camera:{self.left_camera_id},{self.right_camera_id}"
        self.condition = threading.Condition()
        self.closed = False
        self.errors: dict[str, BaseException] = {}
        self.started = time.monotonic()
        self.pair_id = 0
        self.queue_size = max(2, int(queue_size))
        self.queues: dict[str, deque[SourceFrame]] = {
            "left": deque(),
            "right": deque(),
        }
        self.overflow_drops = {"left": 0, "right": 0}
        self.consumed_drops = {"left": 0, "right": 0}

        self.left_capture = _open_camera(self.left_camera_id, config)
        try:
            self.right_capture = _open_camera(self.right_camera_id, config)
        except Exception:
            self.left_capture.release()
            raise

        self.left_width = int(self.left_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.left_height = int(self.left_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.right_width = int(self.right_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.right_height = int(self.right_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        left_fps = float(self.left_capture.get(cv2.CAP_PROP_FPS)) or float(config.fps)
        right_fps = float(self.right_capture.get(cv2.CAP_PROP_FPS)) or float(config.fps)
        self.fps = min(left_fps, right_fps)

        self.threads = [
            threading.Thread(
                target=self._capture_loop,
                args=("left", self.left_capture),
                name=f"capture-left-{self.left_camera_id}",
                daemon=True,
            ),
            threading.Thread(
                target=self._capture_loop,
                args=("right", self.right_capture),
                name=f"capture-right-{self.right_camera_id}",
                daemon=True,
            ),
        ]
        for thread in self.threads:
            thread.start()

    def _capture_loop(self, side: str, capture: cv2.VideoCapture) -> None:
        frame_id = 0
        try:
            while True:
                with self.condition:
                    if self.closed:
                        return
                ok, image = capture.read()
                timestamp = time.monotonic() - self.started
                if not ok or image is None:
                    raise RuntimeError(f"{side}摄像头停止返回画面。")
                frame = SourceFrame(frame_id, timestamp, image)
                frame_id += 1
                with self.condition:
                    queue = self.queues[side]
                    if len(queue) >= self.queue_size:
                        queue.popleft()
                        self.overflow_drops[side] += 1
                    queue.append(frame)
                    self.condition.notify_all()
        except BaseException as exc:
            with self.condition:
                self.errors[side] = exc
                self.closed = True
                self.condition.notify_all()

    def _best_pair(self) -> tuple[int, int] | None:
        left = self.queues["left"]
        right = self.queues["right"]
        if not left or not right:
            return None

        # Low latency matters more than recovering an old pair with a marginally
        # smaller timestamp difference.  Form one candidate around each side's
        # newest frame, then keep the candidate with the smaller skew.
        newest_left = len(left) - 1
        nearest_right = min(
            range(len(right)),
            key=lambda index: abs(
                left[newest_left].timestamp_sec - right[index].timestamp_sec
            ),
        )
        newest_right = len(right) - 1
        nearest_left = min(
            range(len(left)),
            key=lambda index: abs(
                left[index].timestamp_sec - right[newest_right].timestamp_sec
            ),
        )
        candidates = [(newest_left, nearest_right), (nearest_left, newest_right)]
        return min(
            candidates,
            key=lambda item: (
                abs(left[item[0]].timestamp_sec - right[item[1]].timestamp_sec),
                -0.5
                * (
                    left[item[0]].timestamp_sec
                    + right[item[1]].timestamp_sec
                ),
            ),
        )

    def read(self) -> StereoFramePair | None:
        deadline = time.monotonic() + 5.0
        with self.condition:
            while True:
                if self.errors:
                    side, error = next(iter(self.errors.items()))
                    raise RuntimeError(f"{side}摄像头采集失败：{error}") from error
                pair_indices = self._best_pair()
                if pair_indices is not None:
                    left_index, right_index = pair_indices
                    left_queue = self.queues["left"]
                    right_queue = self.queues["right"]
                    left_frame = left_queue[left_index]
                    right_frame = right_queue[right_index]
                    for _ in range(left_index + 1):
                        left_queue.popleft()
                    for _ in range(right_index + 1):
                        right_queue.popleft()
                    self.consumed_drops["left"] += left_index
                    self.consumed_drops["right"] += right_index
                    pair = StereoFramePair(
                        pair_id=self.pair_id,
                        left=left_frame,
                        right=right_frame,
                        timestamp_skew_sec=abs(
                            left_frame.timestamp_sec - right_frame.timestamp_sec
                        ),
                        dropped_left=self.overflow_drops["left"]
                        + self.consumed_drops["left"],
                        dropped_right=self.overflow_drops["right"]
                        + self.consumed_drops["right"],
                    )
                    self.pair_id += 1
                    return pair
                if self.closed:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("等待双目摄像头画面超时。")
                self.condition.wait(min(remaining, 0.25))

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.left_capture.release()
        self.right_capture.release()
        for thread in self.threads:
            thread.join(timeout=2.0)


class StereoVideoSource(StereoFrameSource):
    is_live = False

    def __init__(
        self,
        left_video: str,
        right_video: str,
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
        )
        self.pair_id += 1
        return pair

    def close(self) -> None:
        self.left_source.close()
        self.right_source.close()
