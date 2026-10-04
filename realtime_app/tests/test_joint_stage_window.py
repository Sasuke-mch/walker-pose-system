import copy
import io
import json
import unittest
from collections import deque
from types import SimpleNamespace

import numpy as np
from pose_app.dynamic_ground_live import STAGE1, STAGE2
from pose_app.joint_stage_window import solve_stage_window
from pose_app.landing_pose_recovery import LandingPoseRecovery
from pose_app.two_stage_motion import WorldPose


class JointStageWindowTests(unittest.TestCase):
    def test_buffer_controller_publishes_final_pose_and_preserves_reasons(self):
        initial = WorldPose(np.eye(3), np.array([0.0, 0.0, 700.0]))
        recovery = LandingPoseRecovery(
            initial, mode="joint_window", protocol="flat_ground_supported_after_motion"
        )
        estimator = SimpleNamespace(pose=initial, accepted_translation_history=deque())
        output = io.StringIO()
        for row, payload in self.make_entries():
            action = row["pose_audit"]["window_observation"]["ground_stage"]
            row["stage"] = action
            row["pair_id"] = payload["pair_id"]
            row["pose_audit"]["reason"] = "original_measurement_unavailable"
            row["pose_audit"]["window_observation"]["pose_update_active"] = (
                action != STAGE1
            )
            estimator.pose = WorldPose(
                np.asarray(row["rotation_world_from_left_camera"]),
                np.asarray(row["translation_world_from_left_camera_mm"]),
            )
            recovery.update(
                estimator,
                row,
                payload,
                {"stage": {"handoff": {"ground_stage": action}, "pose_motion": {}}},
            )
            recovery.flush(output, payload["pair_timestamp_sec"])
        self.assertEqual(output.getvalue(), "")
        recovery.flush(output)
        emitted = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([r["pair_id"] for r in emitted], list(range(8)))
        self.assertTrue(all(not r["causal"] for r in emitted))
        self.assertTrue(
            all(
                r["pose_audit"]["reason"] == "original_measurement_unavailable"
                for r in emitted
            )
        )
        self.assertTrue(
            any("joint_window_solution" in r["pose_audit"] for r in emitted)
        )

    def test_inactive_transition_cannot_be_filled_by_window(self):
        entries = self.make_entries()
        entries[3][0]["pose_audit"]["window_observation"].update(
            ground_stage="transition", pose_update_active=False
        )
        poses, audit = solve_stage_window(
            entries, WorldPose(np.eye(3), np.array([0.0, 0.0, 700.0]))
        )
        self.assertIsNone(poses)
        self.assertEqual(audit["reason"], "window_contains_inactive_transition")

    def make_entries(self, *, conflict=False, observations=True, jitter=10):
        entries = []
        for i in range(8):
            k = min(i, 5)
            true = np.array(
                [k * 5.0, 0.0, 700.0 + (20 if conflict else 4) * np.sin(k / 5 * np.pi)]
            )
            if conflict:
                true[2] = 730.0
            noisy = true + np.array([0.0, 0.0, jitter * (k % 2)])
            row = {
                **WorldPose(np.eye(3), noisy).as_mapping(),
                "operational_ankle_anchors_mm": {
                    "left": [-100.0, 0.0, 0.0],
                    "right": [100.0, 0.0, 0.0],
                },
                "pose_audit": {
                    "status": "held",
                    "window_observation": {
                        "ground_stage": STAGE1 if i >= 6 else STAGE2,
                        "rotation_to_from": np.eye(3).tolist()
                        if observations
                        else None,
                    },
                },
            }
            payload = {
                "pair_id": i,
                "pair_timestamp_sec": i * 0.05,
                "persons_3d": [
                    {
                        "keypoints_3d": [
                            {
                                "index": j,
                                "valid": True,
                                "xyz": (np.array([x, 0.0, 0.0]) - true).tolist(),
                            }
                            for j, x in [(15, -100), (16, 100)]
                        ]
                    }
                ]
                if observations
                else [],
            }
            entries.append((row, payload))
        return entries

    def test_joint_motion_decreases_cost_and_holds_stop_exactly(self):
        entries = self.make_entries()
        saved = copy.deepcopy(entries)
        poses, audit = solve_stage_window(
            entries, WorldPose(np.eye(3), np.array([0.0, 0.0, 700.0]))
        )
        self.assertIsNotNone(poses, audit)
        self.assertLess(audit["cost_after"], audit["cost_before"])
        np.testing.assert_allclose(
            poses[0].translation_world_from_camera_mm, [0, 0, 700]
        )
        for i in (6, 7):
            np.testing.assert_array_equal(
                poses[i].translation_world_from_camera_mm,
                poses[5].translation_world_from_camera_mm,
            )
        self.assertEqual(entries, saved)

    def test_support_cannot_override_incompatible_feet(self):
        poses, audit = solve_stage_window(
            self.make_entries(conflict=True),
            WorldPose(np.eye(3), np.array([0.0, 0.0, 700.0])),
            support=True,
        )
        self.assertIsNone(poses)
        self.assertIn(
            audit["reason"],
            [
                "foot_anchor_residual_worsened",
                "support_and_foot_constraints_incompatible",
            ],
        )

    def test_smoothness_cannot_worsen_already_good_foot_observations(self):
        poses, audit = solve_stage_window(
            self.make_entries(jitter=3),
            WorldPose(np.eye(3), np.array([0.0, 0.0, 700.0])),
        )
        self.assertIsNone(poses)
        self.assertEqual(audit["reason"], "foot_anchor_residual_worsened")

    def test_missing_observations_do_not_create_trajectory(self):
        poses, audit = solve_stage_window(
            self.make_entries(observations=False),
            WorldPose(np.eye(3), np.array([0.0, 0.0, 700.0])),
            support=True,
        )
        self.assertIsNone(poses)
        self.assertEqual(audit["reason"], "insufficient_strict_feet_or_rotation")

    def test_repeated_windows_use_original_measurement_guard(self):
        entries = self.make_entries()
        for row, payload in entries:
            i = min(payload["pair_id"], 5)
            row["pose_audit"]["measurement_pose_before_support"] = WorldPose(
                np.eye(3), np.array([i * 5.0, 0.0, 700.0 + 4 * np.sin(i / 5 * np.pi)])
            ).as_mapping()
        poses, audit = solve_stage_window(
            entries, WorldPose(np.eye(3), np.array([0.0, 0.0, 700.0]))
        )
        self.assertIsNone(poses)
        self.assertAlmostEqual(audit["foot_p95_before_mm"], 0, places=8)
        self.assertEqual(audit["reason"], "foot_anchor_residual_worsened")

    def test_invalid_rotation_is_not_silently_projected(self):
        entries = self.make_entries()
        entries[3][0]["rotation_world_from_left_camera"][0][0] = 2
        with self.assertRaises(ValueError):
            solve_stage_window(
                entries, WorldPose(np.eye(3), np.array([0.0, 0.0, 700.0]))
            )

    def test_irregular_time_and_invalid_time(self):
        entries = self.make_entries()
        for i, (_, p) in enumerate(entries):
            p["pair_timestamp_sec"] = i * 0.05 + (i % 2) * 0.01
        _, audit = solve_stage_window(
            entries, WorldPose(np.eye(3), np.array([0.0, 0.0, 700.0]))
        )
        self.assertIn(audit["status"], ["accepted", "rejected"])
        entries[2][1]["pair_timestamp_sec"] = entries[1][1]["pair_timestamp_sec"]
        with self.assertRaises(ValueError):
            solve_stage_window(
                entries, WorldPose(np.eye(3), np.array([0.0, 0.0, 700.0]))
            )


if __name__ == "__main__":
    unittest.main()
