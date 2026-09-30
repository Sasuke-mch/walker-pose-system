import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_app.ground_external_validation import (compare_ground_poses,
                                                   failure_reason_counts,
                                                   foot_support_speed,
                                                   ground_normal_drift,
                                                   stage_switch_jumps)


class GroundExternalValidationTests(unittest.TestCase):
    def test_pose_comparison_identity(self):
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        result = compare_ground_poses(poses, poses)
        self.assertEqual(result["translation_p95_m"], 0.0)
        self.assertEqual(result["rotation_p95_deg"], 0.0)

    def test_support_speed_uses_only_adjacent_support_pairs(self):
        points = np.zeros((3, 2, 3))
        points[1:, :, 0] = 0.01
        result = foot_support_speed(points, np.array([0.1, 0.1]),
                                    np.array([[True, True], [True, False], [False, True]]))
        self.assertEqual(result["support_pairs"], 1)
        self.assertAlmostEqual(result["median_speed_mps"], 0.1)

    def test_stage_switch_jumps_are_reported(self):
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[1, 0, 3] = 0.02
        result = stage_switch_jumps(poses, ["stage1", "stage2", "stage2"])
        self.assertEqual(result["switches"], 1)
        self.assertAlmostEqual(result["translation_jumps_m"][0], 0.02)

    def test_ground_normal_drift_is_relative_to_first_normal(self):
        normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.1, 0.99503719]])
        result = ground_normal_drift(normals)
        self.assertEqual(result["frames"], 2)
        self.assertAlmostEqual(result["median_deg"], result["angles_deg"][1] / 2.0, places=4)
        self.assertGreater(result["p95_deg"], result["median_deg"])

    def test_failure_reason_counts_preserve_empty_reason(self):
        result = failure_reason_counts(["accepted", "", None, "accepted"])
        self.assertEqual(result["frames"], 4)
        self.assertEqual(result["counts"]["accepted"], 2)
        self.assertEqual(result["counts"]["<missing>"], 2)


if __name__ == "__main__":
    unittest.main()
