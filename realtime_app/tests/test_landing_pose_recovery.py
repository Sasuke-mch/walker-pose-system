import io
import json
import unittest
from collections import deque
from types import SimpleNamespace

import cv2
import numpy as np
from pose_app.dynamic_ground_live import STAGE1, STAGE2
from pose_app.landing_pose_recovery import LandingPoseRecovery, supported_pose
from pose_app.two_stage_motion import WorldPose


class RecoveryTests(unittest.TestCase):
    def simulate(self, mode="fixed_lag", human=True, height=64, tilt=0.05, ankles=True):
        initial = WorldPose(np.eye(3), np.array([0.0, 0.0, 700.0]))
        recovery = LandingPoseRecovery(
            initial, mode=mode, protocol="flat_ground_supported_after_motion"
        )
        yaw = cv2.Rodrigues(np.array([0.0, 0.0, 0.2]))[0]
        rotation = yaw @ cv2.Rodrigues(np.array([tilt, 0.0, 0.0]))[0]
        estimator = SimpleNamespace(
            pose=initial,
            anchors={"left": np.ones(3)},
            recent_feet=deque(),
            recent_anchor_feet_world=deque(),
            accepted_translation_history=deque(),
            latest_stationary_feet_world=None,
            _strict_points=lambda p: {
                15: np.array([-120.0, 0.0, 1000.0]),
                16: np.array([120.0, 0.0, 1000.0]),
            },
        )
        rows, output = [], io.StringIO()
        for i in range(18):
            if i == 1:
                estimator.pose = WorldPose(
                    rotation, np.array([50.0, 20.0, 700.0 + height])
                )
            moving = 1 <= i <= 8
            payload = {
                "pair_id": i,
                "pair_timestamp_sec": i * 0.05,
                "persons_3d": [
                    {
                        "keypoints_3d": [
                            {"index": 15, "valid": True, "xyz": [-120, 0, 1000]},
                            {"index": 16, "valid": True, "xyz": [120, 0, 1000]},
                        ]
                    }
                ],
            }
            stage = {
                "stage": {
                    "handoff": {"ground_stage": STAGE2 if moving else STAGE1},
                    "pose_motion": {
                        "left": {
                            "upper_body_keypoints": 8,
                            "upper_body_compensated_motion_px": 2 if human else 0,
                            "ankle_count":2,
                            "ankle_background_residual_median_px":2 if ankles else 0,
                        }
                    },
                }
            }
            row = {
                **estimator.pose.as_mapping(),
                "stage": STAGE2 if moving else STAGE1,
                "pair_id": i,
                "pose_audit": {"status": "held", "reason": "original_failure"},
            }
            rows.append(row)
            recovery.update(estimator, row, payload, stage)
            recovery.flush(output, i * 0.05)
        recovery.flush(output)
        return (
            initial,
            estimator,
            recovery,
            rows,
            [json.loads(x) for x in output.getvalue().splitlines()],
        )

    def test_landing_closes_height_tilt_and_preserves_xy_yaw(self):
        _initial, estimator, recovery, rows, emitted = self.simulate()
        np.testing.assert_allclose(
            estimator.pose.translation_world_from_camera_mm, [50, 20, 700]
        )
        np.testing.assert_allclose(
            estimator.pose.rotation_world_from_camera,
            cv2.Rodrigues(np.array([0.0, 0.0, 0.2]))[0],
            atol=1e-10,
        )
        self.assertEqual(len(recovery.events), 1)
        self.assertTrue(all(not x["causal"] for x in emitted))
        self.assertGreater(
            emitted[0]["pose_audit"]["landing_recovery"]["release_delay_sec"], 0
        )
        self.assertEqual([x["pair_id"] for x in emitted], list(range(18)))
        self.assertTrue(
            all(x["translation_world_from_left_camera_mm"][2] == 700 for x in rows[9:])
        )
        self.assertEqual(rows[12]["pose_audit"]["reason"], "original_failure")
        self.assertIsNone(estimator.anchors)
        np.testing.assert_allclose(
            estimator.latest_stationary_feet_world["left"],
            estimator.pose.transform(np.array([-120.0, 0.0, 1000.0])),
        )
        # Outputs released before stop confirmation must never be rewritten.
        for i in range(3):
            self.assertEqual(
                rows[i]["translation_world_from_left_camera_mm"],
                emitted[i]["translation_world_from_left_camera_mm"],
            )
        self.assertFalse(rows[1]["pose_audit"]["landing_recovery"]["applied"])

    def test_stationary_image_without_human_resume_does_not_restore(self):
        _, estimator, recovery, _, _ = self.simulate(human=False)
        self.assertEqual(estimator.pose.translation_world_from_camera_mm[2], 764)
        self.assertFalse(recovery.events)

    def test_arm_motion_alone_does_not_imply_landing(self):
        _, estimator, recovery, _, _ = self.simulate(ankles=False)
        self.assertEqual(estimator.pose.translation_world_from_camera_mm[2], 764)
        self.assertFalse(recovery.events)

    def test_excessive_correction_is_rejected_not_clipped(self):
        _, estimator, recovery, rows, _ = self.simulate(height=200)
        self.assertEqual(estimator.pose.translation_world_from_camera_mm[2], 900)
        self.assertFalse(recovery.events)
        self.assertEqual(
            rows[-1]["pose_audit"]["landing_recovery"]["reason"],
            "support_correction_exceeds_150mm_or_8deg",
        )

    def test_causal_recovery_has_rate_bound(self):
        _, estimator, _, rows, _ = self.simulate(mode="causal")
        heights = np.array(
            [x["translation_world_from_left_camera_mm"][2] for x in rows[9:]]
        )
        self.assertLess(estimator.pose.translation_world_from_camera_mm[2], 764)
        self.assertLessEqual(abs(np.diff(heights)).max(), 10.00001)

    def test_reference_normal_not_identity(self):
        initial = WorldPose(
            cv2.Rodrigues(np.array([0.4, 0.1, 0.0]))[0], np.array([0.0, 0.0, 706.0])
        )
        pose = WorldPose(
            cv2.Rodrigues(np.array([0.04, -0.02, 0.3]))[0]
            @ initial.rotation_world_from_camera,
            np.array([120.0, -30.0, 780.0]),
        )
        target = supported_pose(pose, initial)
        np.testing.assert_allclose(
            target.rotation_world_from_camera[2],
            initial.rotation_world_from_camera[2],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            target.translation_world_from_camera_mm, [120, -30, 706]
        )

    def test_static_alone_has_no_landing_event(self):
        initial = WorldPose(np.eye(3), np.array([0.0, 0.0, 700.0]))
        recovery = LandingPoseRecovery(
            initial, mode="fixed_lag", protocol="flat_ground_supported_after_motion"
        )
        estimator = SimpleNamespace(
            pose=WorldPose(np.eye(3), np.array([0.0, 0.0, 780.0]))
        )
        for i in range(8):
            row = {
                **estimator.pose.as_mapping(),
                "stage": STAGE1,
                "pose_audit": {"status": "held"},
            }
            payload = {"pair_id": i, "pair_timestamp_sec": i * 0.05, "persons_3d": []}
            stage = {
                "stage": {
                    "handoff": {"ground_stage": STAGE1},
                    "pose_motion": {
                        "left": {
                            "upper_body_keypoints": 8,
                            "upper_body_compensated_motion_px": 5,
                        }
                    },
                }
            }
            self.assertFalse(recovery.update(estimator, row, payload, stage))
        self.assertEqual(estimator.pose.translation_world_from_camera_mm[2], 780)
        self.assertFalse(recovery.events)

    def test_explicit_protocol_required(self):
        initial = WorldPose(np.eye(3), np.zeros(3))
        with self.assertRaises(ValueError):
            LandingPoseRecovery(initial, mode="fixed_lag", protocol=None)


if __name__ == "__main__":
    unittest.main()
