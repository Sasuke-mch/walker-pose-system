import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pose_app.anchored_ankle_translation import solve_ankle_bearing_translation
from pose_app.calibration import StereoCalibration
from pose_app.causal_stereo_background_pose import estimate_background_edge
from pose_app.dynamic_ground_live import STAGE1, STAGE2
from pose_app.stage_handoff import StageHandoffGroundWriter, StageHandoffPolicy


def evidence(moving=False, static=False, instant=1.8):
    return {
        "stage": {
            "operational": "transition",
            "background": {
                "fused": {
                    "available": True,
                    "motion_active": moving,
                    "static_confident": static,
                },
                "left": {
                    "available": True,
                    "inlier_ratio": 0.9,
                    "median_motion_px": instant,
                },
                "right": {
                    "available": True,
                    "inlier_ratio": 0.9,
                    "median_motion_px": instant,
                },
            },
        }
    }


class HandoffTests(unittest.TestCase):
    def test_ankle_bearings_recover_metric_translation(self):
        K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1.0]])
        cal = StereoCalibration(
            "pinhole",
            (1920, 1080),
            (1920, 1080),
            K,
            np.zeros(5),
            K,
            np.zeros(5),
            np.eye(3),
            np.array([-400.0, 0, 0]),
            "mm",
        )
        anchors = {"left": np.array([-120, 0, 2000]), "right": np.array([120, 0, 2000])}
        expected = np.array([20, 10, 5])
        observations = {s: np.zeros((17, 3)) for s in ["left", "right"]}
        for foot, j in [("left", 15), ("right", 16)]:
            for view in ["left", "right"]:
                fn = cal.project_left if view == "left" else cal.project_right
                observations[view][j] = np.r_[
                    fn((anchors[foot] - expected)[None])[0], 1.0
                ]
        translation, audit = solve_ankle_bearing_translation(
            cal, np.eye(3), anchors, observations
        )
        np.testing.assert_allclose(translation, expected, atol=1e-8)
        self.assertFalse(audit["triangulation_mask_modified"])
        observations["right"][16, 0] += 200
        translation, audit = solve_ankle_bearing_translation(
            cal, np.eye(3), anchors, observations
        )
        self.assertIsNone(translation)
        self.assertEqual(audit["reason"], "ankle_anchor_reprojection_above_10px")

    def test_single_strict_ankle_updates_and_preserves_failed_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference.json"
            reference.write_text(
                json.dumps(
                    {
                        "schema_version": "static_charuco_ground_reference_v1",
                        "status": "measured_static_reference",
                        "source_capture_session": "synthetic",
                        "rotation_ground_from_left": np.eye(3).tolist(),
                        "translation_ground_from_left_mm": [0, 0, 700],
                        "plane_left_camera": {"normal": [0, 0, 1], "offset_mm": 700},
                    }
                )
            )
            writer = StageHandoffGroundWriter(
                root / "ground.jsonl",
                reference,
                reanchor=False,
                single_ankle_fallback=True,
                latest_anchor=True,
            )
            policy = StageHandoffPolicy()

            def payload(i, x=0, single=False):
                return {
                    "pair_id": i,
                    "pair_timestamp_sec": i * 0.05,
                    "persons_3d": [
                        {
                            "keypoints_3d": [
                                {"index": j, "valid": True, "xyz": [v - x, 0, -700]}
                                for j, v in (
                                    [(15, -120)] if single else [(15, -120), (16, 120)]
                                )
                            ]
                        }
                    ],
                }

            try:
                for i in range(3):
                    writer.consume(
                        payload(i), policy.update(evidence(static=True, instant=0))
                    )
                row = writer.consume(
                    payload(3, x=20, single=True),
                    policy.update(evidence(moving=True)),
                    {"status": "accepted", "R_to_from": np.eye(3).tolist()},
                )
                self.assertEqual(row["pose_audit"]["status"], "accepted_single_ankle")
                self.assertEqual(row["pose_audit"]["measurement_status"], "unavailable")
                self.assertAlmostEqual(
                    row["translation_world_from_left_camera_mm"][0], 9.0
                )
                self.assertEqual(set(row["strict_points_ground_mm"]), {"15"})
                self.assertFalse(row["constraint_projection"]["applied"])
                row = writer.consume(
                    payload(4, x=30, single=True),
                    policy.update(evidence(moving=True)),
                    {"status": "unavailable"},
                )
                self.assertNotEqual(
                    row["pose_audit"]["status"], "accepted_single_ankle"
                )
                with self.assertRaises(ValueError):
                    writer.consume(payload(4), policy.update(evidence(moving=True)))
            finally:
                writer.close(completed=True)

    def test_motion_onset_has_action_before_confirmation(self):
        policy = StageHandoffPolicy()
        first = policy.update(evidence(moving=True))
        self.assertEqual(first["stage"]["operational"], "transition")
        self.assertEqual(first["stage"]["handoff"]["ground_stage"], STAGE2)
        self.assertEqual(
            policy.update(evidence(moving=True))["stage"]["operational"], STAGE2
        )

    def test_static_human_pause_is_not_indefinite_transition(self):
        policy = StageHandoffPolicy()
        policy.update(evidence(moving=True))
        policy.update(evidence(moving=True))
        policy.update(evidence(static=True, instant=0))
        self.assertEqual(
            policy.update(evidence(static=True, instant=0))["stage"]["operational"],
            STAGE1,
        )

    def test_zero_flow_stops_stale_window_motion(self):
        policy = StageHandoffPolicy()
        policy.update(evidence(moving=True))
        policy.update(evidence(moving=True, instant=0))
        row = policy.update(evidence(moving=True, instant=0))
        self.assertEqual(row["stage"]["operational"], STAGE1)

    def test_ambiguous_background_does_not_force_stage2(self):
        row = StageHandoffPolicy().update(evidence(instant=1))
        self.assertEqual(row["stage"]["operational"], "warming_up")
        self.assertEqual(row["stage"]["handoff"]["action"], "uncertain_pose_hold")

    def test_background_solver_uses_six_to_twenty_points(self):
        rng = np.random.default_rng(719)
        for count in [6, 8, 12, 20, 60]:
            xyz = rng.uniform([-500, -400, 1200], [500, 400, 3000], (count, 3))
            moved = xyz + [10, 2, -3]
            result = estimate_background_edge(xyz, moved[:, :2] / moved[:, 2:3])
            self.assertEqual(result["status"], "accepted", result)
            self.assertLessEqual(result["final_solver_point_count"], 20)
            self.assertEqual(result["final_solver_point_count"], min(count, 20))
            np.testing.assert_allclose(result["t_to_from_mm"], [10, 2, -3], atol=0.01)

    def test_collinear_points_remain_rejected(self):
        xyz = np.column_stack((np.arange(8) * 20, np.zeros(8), np.full(8, 1500)))
        result = estimate_background_edge(xyz, xyz[:, :2] / xyz[:, 2:3])
        self.assertEqual(result["reason"], "spatially_degenerate_links")


if __name__ == "__main__":
    unittest.main()
