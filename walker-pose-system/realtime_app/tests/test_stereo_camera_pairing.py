from __future__ import annotations

from pathlib import Path
import sys
import unittest


REALTIME_APP_DIR = Path(__file__).resolve().parents[1]
if str(REALTIME_APP_DIR) not in sys.path:
    sys.path.insert(0, str(REALTIME_APP_DIR))

from pose_app.stereo_camera import _capture_span_and_fps, _pair_action


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
