from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REALTIME_APP_DIR = Path(__file__).resolve().parents[1]
if str(REALTIME_APP_DIR) not in sys.path:
    sys.path.insert(0, str(REALTIME_APP_DIR))

from pose_app.stereo_camera import (
    CameraFrame,
    StereoPair,
    _capture_span_and_fps,
    _pair_action,
)
from pose_app.stereo_sources import _camera_pair_to_runtime_pair


MS = 1_000_000
NS = 1_000_000_000


class PairActionTests(unittest.TestCase):
    def test_pairs_current_heads_when_locally_best(self) -> None:
        self.assertEqual(
            _pair_action(100 * MS, 133 * MS, 115 * MS, 148 * MS, 25 * MS),
            "pair",
        )

    def test_drops_left_when_next_left_is_closer_to_current_right(self) -> None:
        self.assertEqual(
            _pair_action(100 * MS, 133 * MS, 128 * MS, 161 * MS, 30 * MS),
            "drop_left",
        )

    def test_drops_right_when_next_right_is_closer_to_current_left(self) -> None:
        self.assertEqual(
            _pair_action(128 * MS, 161 * MS, 100 * MS, 133 * MS, 30 * MS),
            "drop_right",
        )

    def test_outside_threshold_drops_earlier_head(self) -> None:
        self.assertEqual(
            _pair_action(100 * MS, 133 * MS, 140 * MS, 173 * MS, 10 * MS),
            "drop_left",
        )

    def test_exact_threshold_is_accepted(self) -> None:
        self.assertEqual(
            _pair_action(100 * MS, 180 * MS, 125 * MS, 180 * MS, 25 * MS),
            "pair",
        )

    def test_outside_threshold_drops_right_when_right_is_earlier(self) -> None:
        self.assertEqual(
            _pair_action(140 * MS, 173 * MS, 100 * MS, 133 * MS, 10 * MS),
            "drop_right",
        )

    def test_tie_keeps_current_heads_deterministically(self) -> None:
        self.assertEqual(
            _pair_action(100 * MS, 150 * MS, 125 * MS, 175 * MS, 25 * MS),
            "pair",
        )


class CaptureStatsTests(unittest.TestCase):
    def test_measured_fps_uses_first_to_last_frame_span(self) -> None:
        span, fps = _capture_span_and_fps(301, 10 * NS, 20 * NS)
        self.assertAlmostEqual(span, 10.0)
        self.assertAlmostEqual(fps, 30.0)

    def test_measured_fps_requires_at_least_two_frames(self) -> None:
        self.assertEqual(_capture_span_and_fps(1, 10 * NS, 10 * NS), (None, None))

    def test_measured_fps_rejects_nonpositive_span(self) -> None:
        self.assertEqual(_capture_span_and_fps(2, 20 * NS, 10 * NS), (None, None))


class RuntimeAdapterTests(unittest.TestCase):
    def test_camera_pair_adapter_preserves_timestamp_metadata(self) -> None:
        image = np.zeros((8, 12, 3), dtype=np.uint8)
        left = CameraFrame(10, 1_000_000_000, 31.0, image)
        right = CameraFrame(12, 1_005_000_000, 32.0, image)
        pair = StereoPair(
            4,
            left,
            right,
            signed_host_delta_ms=5.0,
            abs_host_delta_ms=5.0,
            left_dropped_before=2,
            right_dropped_before=3,
        )
        runtime = _camera_pair_to_runtime_pair(pair)
        self.assertEqual(runtime.left.frame_id, 10)
        self.assertEqual(runtime.right.frame_id, 12)
        self.assertAlmostEqual(runtime.left.timestamp_sec, 1.0)
        self.assertAlmostEqual(runtime.right.timestamp_sec, 1.005)
        self.assertEqual(runtime.left_host_timestamp_ns, 1_000_000_000)
        self.assertEqual(runtime.right_host_timestamp_ns, 1_005_000_000)
        self.assertEqual(runtime.signed_host_delta_ms, 5.0)
        self.assertAlmostEqual(runtime.timestamp_skew_sec, 0.005)
        self.assertEqual(runtime.dropped_left, 2)
        self.assertEqual(runtime.dropped_right, 3)

    def test_stereo_pair_drop_counts_default_to_zero(self) -> None:
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        frame = CameraFrame(0, 1, 0.0, image)
        pair = StereoPair(0, frame, frame, 0.0, 0.0)
        self.assertEqual(pair.left_dropped_before, 0)
        self.assertEqual(pair.right_dropped_before, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
