import unittest

import numpy as np

from pose_app.local_plane_propagation import STATIC_BACKGROUND_DOMAIN
from pose_app.two_stage_motion import (
    FEET_STATIC_WALKER_MOVING,
    SETTLING,
    TRANSITION,
    UNKNOWN,
    WALKER_STATIC_HUMAN_MOVING,
    MotionEvidence,
    TwoStageCriteria,
    TwoStageMotionTracker,
    WorldPose,
    apply_soft_foot_translation_correction,
    foot_anchor_residuals_mm,
    propagate_world_pose,
)


def evidence(translation, rotation, human_speed, foot_residual, status="accepted"):
    return MotionEvidence(status, translation, rotation, human_speed, foot_residual)


class WorldPoseTests(unittest.TestCase):
    def test_relative_camera_edge_is_inverted_when_accumulated_into_world(self):
        initial = WorldPose(np.eye(3), np.zeros(3))
        # A camera translated +10 mm in world sees a stationary feature shift
        # from x=100 to x=90, hence X_current = X_previous - 10.
        current = propagate_world_pose(initial, {
            "status": "accepted",
            "feature_domain": STATIC_BACKGROUND_DOMAIN,
            "ground_region_used_for_motion": False,
            "R_to_from": np.eye(3).tolist(),
            "q_to_from": [-10.0, 0.0, 0.0],
        })
        self.assertTrue(np.allclose(current.translation_world_from_camera_mm, [10.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(current.transform([90.0, 0.0, 0.0]), [100.0, 0.0, 0.0]))

    def test_pose_propagation_rejects_non_static_or_ground_dependent_input(self):
        initial = WorldPose(np.eye(3), np.zeros(3))
        common = {"R_to_from": np.eye(3), "q_to_from": np.zeros(3), "feature_domain": STATIC_BACKGROUND_DOMAIN}
        with self.assertRaisesRegex(ValueError, "status"):
            propagate_world_pose(initial, {**common, "status": "rejected", "ground_region_used_for_motion": False})
        with self.assertRaisesRegex(ValueError, "ground-fit"):
            propagate_world_pose(initial, {**common, "status": "accepted", "ground_region_used_for_motion": True})

    def test_two_foot_correction_is_translation_only_and_bounded(self):
        pose = WorldPose(np.eye(3), np.asarray([10.0, 0.0, 0.0]))
        observed = {"left": [0.0, 0.0, 0.0], "right": [100.0, 0.0, 0.0]}
        anchors = {"left": [0.0, 0.0, 0.0], "right": [100.0, 0.0, 0.0]}
        corrected, audit = apply_soft_foot_translation_correction(
            pose, observed, anchors, gain=0.5, maximum_pre_correction_residual_mm=20.0
        )
        self.assertEqual(audit["status"], "corrected")
        self.assertIsNotNone(corrected)
        self.assertTrue(np.allclose(corrected.rotation_world_from_camera, np.eye(3)))
        self.assertTrue(np.allclose(corrected.translation_world_from_camera_mm, [5.0, 0.0, 0.0]))
        self.assertAlmostEqual(audit["post_correction_residual_maximum_mm"], 5.0)

    def test_foot_constraint_requires_both_feet_and_rejects_large_residual(self):
        pose = WorldPose(np.eye(3), np.zeros(3))
        missing = foot_anchor_residuals_mm(pose, {"left": [0, 0, 0]}, {"left": [0, 0, 0]})
        self.assertEqual(missing["status"], "unavailable")
        corrected, audit = apply_soft_foot_translation_correction(
            pose,
            {"left": [0, 0, 0], "right": [100, 0, 0]},
            {"left": [1000, 0, 0], "right": [1100, 0, 0]},
            gain=0.2,
            maximum_pre_correction_residual_mm=100.0,
        )
        self.assertIsNone(corrected)
        self.assertEqual(audit["status"], "rejected")


class TwoStageTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = TwoStageMotionTracker(TwoStageCriteria(confirmation_frames=2))

    def test_requires_confirmation_for_each_controlled_stage(self):
        first = self.tracker.update(evidence(1.0, 0.1, 50.0, None))
        second = self.tracker.update(evidence(1.0, 0.1, 50.0, None))
        self.assertEqual(first["state"], TRANSITION)
        self.assertEqual(second["state"], WALKER_STATIC_HUMAN_MOVING)
        third = self.tracker.update(evidence(12.0, 0.1, 5.0, 10.0))
        fourth = self.tracker.update(evidence(12.0, 0.1, 5.0, 10.0))
        self.assertEqual(third["state"], TRANSITION)
        self.assertEqual(fourth["state"], FEET_STATIC_WALKER_MOVING)

    def test_moving_camera_without_two_stationary_feet_never_enters_stage_two(self):
        for _ in range(5):
            result = self.tracker.update(evidence(12.0, 0.1, 5.0, 80.0))
        self.assertEqual(result["state"], TRANSITION)
        self.assertNotEqual(result["active_confirmed_state"], FEET_STATIC_WALKER_MOVING)

    def test_missing_background_pose_fails_closed_and_settling_is_explicit(self):
        self.tracker.update(evidence(12.0, 0.1, 5.0, 10.0))
        self.tracker.update(evidence(12.0, 0.1, 5.0, 10.0))
        settling = self.tracker.update(evidence(1.0, 0.1, 0.0, 5.0))
        self.assertEqual(settling["state"], SETTLING)
        unavailable = self.tracker.update(evidence(None, None, None, None, status="unavailable"))
        self.assertEqual(unavailable["state"], UNKNOWN)
        self.assertEqual(unavailable["active_confirmed_state"], UNKNOWN)


if __name__ == "__main__":
    unittest.main()
