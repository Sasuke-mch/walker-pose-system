import unittest

import numpy as np

from pose_app.pose_observation_partition import (
    REASON_ACCEPTED_RELAXED,
    REASON_ACCEPTED_STRICT,
    REASON_LOW_SCORE,
    REASON_MISSING_OR_NONFINITE,
    REASON_OUTSIDE_RAW_IMAGE,
    REASON_UPSTREAM_NON_STRICT_HIGH_ERROR,
    UPSTREAM_ERROR_THRESHOLD_PX,
    ViewObservationDecision,
    classify_view_observation,
    classify_view_observations_batch,
)
from pose_app.reprojection_common import build_view_2d_weights_from_obs


def _cal(width=1920, height=1080):
    return type("Cal", (), {"image_width": width, "image_height": height})()


class PoseObservationPartitionTests(unittest.TestCase):
    def test_strict_valid_point_accepted(self):
        decision = classify_view_observation(
            np.asarray([960.0, 540.0]), 0.9, True, 5.0, 1920, 1080
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, REASON_ACCEPTED_STRICT)
        self.assertEqual(decision.score, 0.9)
        self.assertTrue(decision.upstream_strict)

    def test_nonfinite_point_rejected(self):
        decision = classify_view_observation(
            np.asarray([np.nan, 540.0]), 0.9, True, 5.0, 1920, 1080
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, REASON_MISSING_OR_NONFINITE)

    def test_outside_raw_image_rejected(self):
        decision = classify_view_observation(
            np.asarray([1920.0, 540.0]), 0.9, True, 5.0, 1920, 1080
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, REASON_OUTSIDE_RAW_IMAGE)

    def test_low_score_rejected(self):
        decision = classify_view_observation(
            np.asarray([960.0, 540.0]), 0.10, True, 5.0, 1920, 1080
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, REASON_LOW_SCORE)

    def test_non_strict_high_error_rejected(self):
        decision = classify_view_observation(
            np.asarray([960.0, 540.0]), 0.9, False, UPSTREAM_ERROR_THRESHOLD_PX + 1.0, 1920, 1080
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, REASON_UPSTREAM_NON_STRICT_HIGH_ERROR)

    def test_non_strict_unknown_error_rejected(self):
        decision = classify_view_observation(
            np.asarray([960.0, 540.0]), 0.9, False, None, 1920, 1080
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, REASON_UPSTREAM_NON_STRICT_HIGH_ERROR)

    def test_non_strict_low_error_accepted_relaxed(self):
        decision = classify_view_observation(
            np.asarray([960.0, 540.0]), 0.9, False, 3.0, 1920, 1080
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, REASON_ACCEPTED_RELAXED)

    def test_left_right_decisions_are_independent(self):
        # One view good does not silently promote the other view.
        pixels_l = np.asarray([[[960.0, 540.0], [960.0, 540.0]]])
        pixels_r = np.asarray([[[960.0, 540.0], [np.nan, np.nan]]])
        scores = np.asarray([[0.9, 0.9]])
        strict = np.asarray([[True, False]])
        errors = np.asarray([[5.0, 5.0]])
        part_l = classify_view_observations_batch(pixels_l, scores, strict, errors, 1920, 1080)
        part_r = classify_view_observations_batch(pixels_r, scores, strict, errors, 1920, 1080)
        np.testing.assert_array_equal(part_l["accepted"], [[True, True]])
        np.testing.assert_array_equal(part_r["accepted"], [[True, False]])
        self.assertEqual(part_r["reasons"][0, 1], REASON_MISSING_OR_NONFINITE)

    def test_rejection_reasons_preserved(self):
        pixels = np.asarray([
            [[np.nan, 540.0], [2000.0, 540.0], [960.0, 540.0], [960.0, 540.0], [960.0, 540.0]],
        ])
        scores = np.asarray([[0.9, 0.9, 0.10, 0.9, 0.9]])
        strict = np.asarray([[True, True, True, False, False]])
        errors = np.asarray([[5.0, 5.0, 5.0, 11.0, 3.0]])
        part = classify_view_observations_batch(pixels, scores, strict, errors, 1920, 1080)
        reasons = part["reasons"][0]
        self.assertEqual(reasons[0], REASON_MISSING_OR_NONFINITE)
        self.assertEqual(reasons[1], REASON_OUTSIDE_RAW_IMAGE)
        self.assertEqual(reasons[2], REASON_LOW_SCORE)
        self.assertEqual(reasons[3], REASON_UPSTREAM_NON_STRICT_HIGH_ERROR)
        self.assertEqual(reasons[4], REASON_ACCEPTED_RELAXED)
        self.assertIn(REASON_MISSING_OR_NONFINITE, part["rejection_counts"])
        self.assertIn(REASON_OUTSIDE_RAW_IMAGE, part["rejection_counts"])
        self.assertIn(REASON_LOW_SCORE, part["rejection_counts"])
        self.assertIn(REASON_UPSTREAM_NON_STRICT_HIGH_ERROR, part["rejection_counts"])
        self.assertIn(REASON_ACCEPTED_STRICT, part["rejection_counts"])
        self.assertIn(REASON_ACCEPTED_RELAXED, part["rejection_counts"])

    def test_weights_do_not_read_3d_depth(self):
        # build_view_2d_weights_from_obs must depend only on scores/strict/errors.
        obs = {
            "score_left": np.asarray([[0.9, 0.9, 0.9]]),
            "score_right": np.asarray([[0.9, 0.9, 0.9]]),
            "accepted_left": np.asarray([[True, True, False]]),
            "accepted_right": np.asarray([[True, False, True]]),
            "strict_left": np.asarray([[True, False, True]]),
            "strict_right": np.asarray([[True, True, False]]),
            "errors_left_px": np.asarray([[5.0, 5.0, 5.0]]),
            "errors_right_px": np.asarray([[5.0, 5.0, 5.0]]),
        }
        wl, wr = build_view_2d_weights_from_obs(obs)
        self.assertEqual(wl.shape, (1, 3))
        self.assertEqual(wr.shape, (1, 3))
        np.testing.assert_array_equal(wl[0, 2], 0.0)  # rejected -> 0
        np.testing.assert_array_equal(wr[0, 1], 0.0)
        self.assertGreater(wl[0, 0], 0.0)   # strict accepted
        self.assertGreater(wr[0, 2], 0.0)   # relaxed accepted (strict flag False)


if __name__ == "__main__":
    unittest.main()
