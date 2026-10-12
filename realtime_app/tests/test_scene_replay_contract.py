import sys
import unittest
from unittest.mock import patch

import numpy as np
from walker_tools.scene.replay_fitted_body_scene import args, strict_scene_arrays


class SceneReplayContractTests(unittest.TestCase):
    def test_association_rejection_cannot_reappear_in_export(self):
        rows = [
            {"index": 0, "persons_3d": []},
            {
                "index": 1,
                "persons_3d": [
                    {
                        "left_person_id": 0,
                        "right_person_id": 0,
                        "keypoints_3d": [
                            {
                                "index": 14,
                                "valid": True,
                                "xyz": [1, 2, 3],
                                "depth_left": 3,
                                "depth_right": 4,
                            },
                            {"index": 15, "valid": False, "xyz": [5, 6, 7]},
                        ],
                    }
                ],
            },
        ]
        points, mask = strict_scene_arrays(rows, [0, 1])
        self.assertFalse(mask[0].any())
        self.assertEqual(mask.sum(), 1)
        np.testing.assert_array_equal(points[1, 14], [1, 2, 3])
        self.assertTrue(np.isnan(points[~mask]).all())

    def test_duplicate_or_wrong_person_is_rejected(self):
        person = {
            "left_person_id": 0,
            "right_person_id": 0,
            "keypoints_3d": [{"index": 15, "valid": False}] * 2,
        }
        with self.assertRaises(ValueError):
            strict_scene_arrays([{"index": 0, "persons_3d": [person]}], [0])
        person.update(right_person_id=1, keypoints_3d=[])
        with self.assertRaises(ValueError):
            strict_scene_arrays([{"index": 0, "persons_3d": [person]}], [0])

    def test_nonfinite_depth_is_rejected(self):
        row = {
            "index": 0,
            "persons_3d": [
                {
                    "left_person_id": 0,
                    "right_person_id": 0,
                    "keypoints_3d": [
                        {
                            "index": 15,
                            "valid": True,
                            "xyz": [1, 2, 3],
                            "depth_left": float("nan"),
                            "depth_right": 3,
                        }
                    ],
                }
            ],
        }
        with self.assertRaises(ValueError):
            strict_scene_arrays([row], [0])

    def parse(self, extra):
        required = [
            "--left-video",
            "l",
            "--right-video",
            "r",
            "--left-pmpose",
            "l.json",
            "--right-pmpose",
            "r.json",
            "--calibration",
            "c",
            "--ground-reference",
            "g",
            "--walker-model",
            "w",
            "--output",
            "o",
            "--strict-jsonl",
            "s",
        ]
        with patch.object(sys, "argv", ["replay", *required, *extra]):
            return args()

    def test_baseline_keeps_algorithm_defaults(self):
        profile = self.parse(["--scene-profile", "baseline"])
        self.assertFalse(profile.stage_handoff)
        self.assertFalse(profile.disable_stage1_reanchor)
        self.assertEqual(profile.landing_recovery, "none")
        self.assertIsNone(profile.landing_protocol)
        self.assertEqual(profile.foot_observation_mode, "causal_median")

    def test_mainline_and_default_use_verified_supported_stop(self):
        selected = self.parse(["--scene-profile", "mainline"])
        supported = self.parse(["--scene-profile", "supported_stop_fixed_lag"])
        self.assertEqual(selected.resolved_scene_profile, "supported_stop_fixed_lag")
        self.assertEqual(vars(self.parse([])), vars(selected))
        for key in vars(supported):
            if key != "scene_profile":
                self.assertEqual(getattr(selected, key), getattr(supported, key))
        self.assertTrue(selected.stage_handoff)
        self.assertTrue(selected.disable_stage1_reanchor)
        self.assertEqual(selected.landing_protocol, "supported_at_camera_stop")
        self.assertEqual(selected.landing_recovery, "fixed_lag")
        with self.assertRaises(SystemExit):
            self.parse(["--scene-profile", "mainline", "--landing-recovery", "none"])
        with self.assertRaises(SystemExit):
            self.parse(["--scene-profile", "mainline", "--single-ankle-fallback"])

    def test_candidate_requires_real_timing_and_rejects_conflicting_flag(self):
        with self.assertRaises(SystemExit):
            self.parse(["--scene-profile", "landing_fixed_lag"])
        result = self.parse(
            ["--scene-profile", "landing_fixed_lag", "--frames-metadata", "f"]
        )
        self.assertTrue(result.stage_handoff)
        self.assertEqual(result.landing_recovery, "fixed_lag")
        with self.assertRaises(SystemExit):
            self.parse(
                [
                    "--scene-profile",
                    "landing_fixed_lag",
                    "--frames-metadata",
                    "f",
                    "--landing-recovery",
                    "none",
                ]
            )


if __name__ == "__main__":
    unittest.main()
