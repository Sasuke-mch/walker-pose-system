import unittest

import numpy as np

from pose_app.dynamic_ground_pose import (
    FullFootPlanePoseCriteria,
    PlanarFootPoseCriteria,
    RotationFootPoseCriteria,
    solve_full_pose_from_feet_and_plane,
    solve_pose_from_relative_rotation_and_feet,
    solve_planar_pose_from_feet,
    solve_planar_translation_from_feet,
)
from pose_app.two_stage_motion import WorldPose


class DynamicGroundPoseTests(unittest.TestCase):
    def test_recovers_planar_translation_and_yaw(self):
        reference = WorldPose(np.eye(3), np.asarray([0.0, 0.0, 700.0]))
        yaw = np.deg2rad(4.0)
        rotation = np.asarray([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ])
        expected = WorldPose(rotation, np.asarray([40.0, -20.0, 700.0]))
        anchors = {"left": [-120.0, 20.0, 0.0], "right": [120.0, 20.0, 0.0]}
        observed = {
            name: expected.rotation_world_from_camera.T @ (np.asarray(point) - expected.translation_world_from_camera_mm)
            for name, point in anchors.items()
        }
        solved, audit = solve_planar_pose_from_feet(
            reference, reference, observed, anchors,
            PlanarFootPoseCriteria(smoothing_gain=1.0),
        )
        self.assertEqual(audit["status"], "accepted")
        np.testing.assert_allclose(solved.translation_world_from_camera_mm, expected.translation_world_from_camera_mm, atol=1e-8)
        np.testing.assert_allclose(solved.rotation_world_from_camera, expected.rotation_world_from_camera, atol=1e-8)

    def test_rejects_implausible_realtime_jump(self):
        pose = WorldPose(np.eye(3), np.asarray([0.0, 0.0, 700.0]))
        observed = {"left": [-620.0, 0.0, -700.0], "right": [-380.0, 0.0, -700.0]}
        anchors = {"left": [-120.0, 0.0, 0.0], "right": [120.0, 0.0, 0.0]}
        solved, audit = solve_planar_pose_from_feet(pose, pose, observed, anchors)
        self.assertIsNone(solved)
        self.assertEqual(audit["reason"], "planar_pose_step_exceeds_realtime_limit")

    def test_translation_only_recovers_straight_push(self):
        reference = WorldPose(np.eye(3), np.asarray([0.0, 0.0, 700.0]))
        expected = WorldPose(np.eye(3), np.asarray([45.0, -25.0, 700.0]))
        anchors = {"left": [-120.0, 20.0, 0.0], "right": [120.0, 20.0, 0.0]}
        observed = {name: np.asarray(point) - expected.translation_world_from_camera_mm for name, point in anchors.items()}
        solved, audit = solve_planar_translation_from_feet(
            reference, reference, observed, anchors, PlanarFootPoseCriteria(smoothing_gain=1.0)
        )
        self.assertEqual(audit["status"], "accepted")
        np.testing.assert_allclose(solved.translation_world_from_camera_mm, expected.translation_world_from_camera_mm)

    def test_full_pose_recovers_rotation_height_and_xyz(self):
        yaw, pitch, roll = np.deg2rad([4.0, -2.0, 3.0])
        rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        expected = WorldPose(rz @ ry @ rx, np.asarray([35.0, -18.0, 715.0]))
        previous = WorldPose(np.eye(3), np.asarray([0.0, 0.0, 700.0]))
        anchors = {"left": np.array([-130.0, 30.0, 65.0]), "right": np.array([130.0, 30.0, 65.0])}
        observed = {
            name: expected.rotation_world_from_camera.T @ (point - expected.translation_world_from_camera_mm)
            for name, point in anchors.items()
        }
        normal_camera = expected.rotation_world_from_camera.T @ np.asarray([0.0, 0.0, 1.0])
        solved, audit = solve_full_pose_from_feet_and_plane(
            previous, observed, anchors, normal_camera, expected.translation_world_from_camera_mm[2],
            FullFootPlanePoseCriteria(smoothing_gain=1.0),
        )
        self.assertEqual(audit["status"], "accepted")
        np.testing.assert_allclose(solved.rotation_world_from_camera, expected.rotation_world_from_camera, atol=1e-8)
        np.testing.assert_allclose(solved.translation_world_from_camera_mm, expected.translation_world_from_camera_mm, atol=1e-8)

    def test_full_pose_requires_valid_plane(self):
        pose = WorldPose(np.eye(3), np.asarray([0.0, 0.0, 700.0]))
        feet = {"left": [-120.0, 0.0, -635.0], "right": [120.0, 0.0, -635.0]}
        anchors = {"left": [-120.0, 0.0, 65.0], "right": [120.0, 0.0, 65.0]}
        solved, audit = solve_full_pose_from_feet_and_plane(pose, feet, anchors, [0, 0, 0], 700.0)
        self.assertIsNone(solved)
        self.assertEqual(audit["reason"], "valid_ground_plane_required")

    def test_relative_rotation_and_feet_recover_full_pose(self):
        previous = WorldPose(np.eye(3), np.asarray([0.0, 0.0, 700.0]))
        angle = np.deg2rad(2.0)
        relative = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        expected_rotation = relative.T
        expected_translation = np.asarray([35.0, -15.0, 712.0])
        anchors = {"left": np.array([-120.0, 20.0, 65.0]), "right": np.array([120.0, 20.0, 65.0])}
        observed = {
            name: expected_rotation.T @ (point - expected_translation)
            for name, point in anchors.items()
        }
        solved, audit = solve_pose_from_relative_rotation_and_feet(
            previous, relative, observed, anchors,
            RotationFootPoseCriteria(smoothing_gain=1.0),
        )
        self.assertEqual(audit["status"], "accepted")
        np.testing.assert_allclose(solved.rotation_world_from_camera, expected_rotation, atol=1e-8)
        np.testing.assert_allclose(solved.translation_world_from_camera_mm, expected_translation, atol=1e-8)

    def test_relative_rotation_rejects_disagreeing_per_foot_translations(self):
        previous = WorldPose(np.eye(3), np.zeros(3))
        observed = {"left": np.array([0.0, 0.0, 0.0]), "right": np.array([100.0, 0.0, 0.0])}
        anchors = {"left": np.array([0.0, 0.0, 0.0]), "right": np.array([180.0, 0.0, 0.0])}
        solved, audit = solve_pose_from_relative_rotation_and_feet(
            previous, np.eye(3), observed, anchors,
            RotationFootPoseCriteria(
                maximum_foot_separation_error_mm=100.0,
                maximum_two_foot_translation_disagreement_mm=60.0,
                smoothing_gain=1.0,
            ),
        )
        self.assertIsNone(solved)
        self.assertEqual(audit["reason"], "two_foot_translation_disagreement")
        self.assertAlmostEqual(audit["two_foot_translation_disagreement_mm"], 80.0)

    def test_relative_rotation_scales_translation_gate_after_a_short_gap(self):
        previous = WorldPose(np.eye(3), np.zeros(3))
        anchors = {"left": np.array([-120.0, 0.0, 0.0]), "right": np.array([120.0, 0.0, 0.0])}
        observed = {name: point - np.array([200.0, 0.0, 0.0]) for name, point in anchors.items()}
        criteria = RotationFootPoseCriteria(maximum_translation_step_mm=120.0, smoothing_gain=1.0)
        rejected, rejected_audit = solve_pose_from_relative_rotation_and_feet(
            previous, np.eye(3), observed, anchors, criteria,
        )
        self.assertIsNone(rejected)
        self.assertEqual(rejected_audit["effective_maximum_translation_step_mm"], 120.0)
        solved, audit = solve_pose_from_relative_rotation_and_feet(
            previous, np.eye(3), observed, anchors, criteria,
            elapsed_frames_since_update=2,
        )
        self.assertEqual(audit["status"], "accepted")
        self.assertEqual(audit["effective_maximum_translation_step_mm"], 240.0)
        np.testing.assert_allclose(solved.translation_world_from_camera_mm, [200.0, 0.0, 0.0])

    def test_single_foot_recovery_requires_a_clear_motion_prior_winner(self):
        previous = WorldPose(np.eye(3), np.zeros(3))
        anchors = {"left": np.array([0.0, 0.0, 0.0]), "right": np.array([200.0, 0.0, 0.0])}
        observed = {"left": np.array([-100.0, 0.0, 0.0]), "right": np.array([20.0, 0.0, 0.0])}
        solved, audit = solve_pose_from_relative_rotation_and_feet(
            previous, np.eye(3), observed, anchors,
            RotationFootPoseCriteria(smoothing_gain=1.0),
            translation_prediction_world_mm=np.array([105.0, 0.0, 0.0]),
            allow_single_foot_recovery=True,
        )
        self.assertEqual(audit["status"], "accepted")
        self.assertTrue(audit["single_foot_recovery_applied"])
        self.assertEqual(audit["selected_translation_foot"], "left")
        np.testing.assert_allclose(solved.translation_world_from_camera_mm, [100.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
