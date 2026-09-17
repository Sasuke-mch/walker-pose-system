import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pose_app.dynamic_ground_live import RealtimeDynamicGroundWriter


class DynamicGroundLiveTests(unittest.TestCase):
    @staticmethod
    def ground_reference(path):
        path.write_text(json.dumps({
            "schema_version": "static_charuco_ground_reference_v1",
            "status": "measured_static_reference", "source_capture_session": "unit-test",
            "rotation_ground_from_left": np.eye(3).tolist(),
            "translation_ground_from_left_mm": [0.0, 0.0, 700.0],
            "plane_left_camera": {"normal": [0.0, 0.0, 1.0], "offset_mm": 700.0},
        }), encoding="utf-8")

    def test_writes_causal_ground_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "ground.json"
            reference.write_text(json.dumps({
                "schema_version": "static_charuco_ground_reference_v1",
                "status": "measured_static_reference",
                "source_capture_session": "unit-test",
                "rotation_ground_from_left": np.eye(3).tolist(),
                "translation_ground_from_left_mm": [0.0, 0.0, 700.0],
                "plane_left_camera": {"normal": [0.0, 0.0, 1.0], "offset_mm": 700.0},
            }), encoding="utf-8")
            writer = RealtimeDynamicGroundWriter(root / "live.jsonl", reference)
            points = [
                {"index": 15, "valid": True, "xyz": [-120.0, 0.0, -700.0]},
                {"index": 16, "valid": True, "xyz": [120.0, 0.0, -700.0]},
            ]
            payload = {"pair_id": 0, "persons_3d": [{"keypoints_3d": points}]}
            stage1 = {"stage": {"operational": "stage1_walker_static_human_moving"}}
            first = writer.consume(payload, stage1)
            self.assertTrue(first["causal"])
            payload["pair_id"] = 1
            stage2 = {"stage": {"operational": "stage2_feet_static_walker_moving", "vote_confidence": 1.0}}
            second = writer.consume(payload, stage2)
            self.assertEqual(second["pose_audit"]["status"], "accepted")
            summary = writer.close(completed=True)
            self.assertEqual(summary["record_count"], 2)
            self.assertEqual(len((root / "live.jsonl").read_text(encoding="utf-8").splitlines()), 2)

    def test_full_se3_live_mode_updates_xyz_with_rotation_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "ground.json"
            reference.write_text(json.dumps({
                "schema_version": "static_charuco_ground_reference_v1",
                "status": "measured_static_reference", "source_capture_session": "unit-test",
                "rotation_ground_from_left": np.eye(3).tolist(),
                "translation_ground_from_left_mm": [0.0, 0.0, 700.0],
                "plane_left_camera": {"normal": [0.0, 0.0, 1.0], "offset_mm": 700.0},
            }), encoding="utf-8")
            writer = RealtimeDynamicGroundWriter(root / "live.jsonl", reference, pose_mode="full_se3")
            def payload(pair_id, left, right):
                return {"pair_id": pair_id, "persons_3d": [{"keypoints_3d": [
                    {"index": 15, "valid": True, "xyz": left},
                    {"index": 16, "valid": True, "xyz": right},
                ]}]}
            stage1 = {"stage": {"operational": "stage1_walker_static_human_moving"}}
            writer.consume(payload(0, [-120, 0, -700], [120, 0, -700]), stage1)
            stage2 = {"stage": {"operational": "stage2_feet_static_walker_moving", "vote_confidence": 1.0}}
            rotation = {"status": "accepted", "R_to_from": np.eye(3).tolist()}
            updated = writer.consume(payload(1, [-130, 0, -710], [110, 0, -710]), stage2, rotation)
            self.assertEqual(updated["pose_audit"]["status"], "accepted")
            self.assertGreater(updated["translation_world_from_left_camera_mm"][0], 0.0)
            self.assertGreater(updated["translation_world_from_left_camera_mm"][2], 700.0)
            summary = writer.close(completed=True)
            self.assertIn("full_se3", summary["pose_model"])

    def test_missing_current_feet_never_reuses_window_and_low_vote_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "ground.json"
            self.ground_reference(reference)
            writer = RealtimeDynamicGroundWriter(root / "live.jsonl", reference, pose_mode="full_se3")
            feet = {"pair_id": 0, "persons_3d": [{"keypoints_3d": [
                {"index": 15, "valid": True, "xyz": [-120, 0, -700]},
                {"index": 16, "valid": True, "xyz": [120, 0, -700]},
            ]}]}
            writer.consume(feet, {"stage": {"operational": "stage1_walker_static_human_moving"}})
            feet["pair_id"] = 1
            low_vote = writer.consume(
                feet, {"stage": {"operational": "stage2_feet_static_walker_moving", "vote_confidence": 0.33}},
                {"status": "accepted", "R_to_from": np.eye(3).tolist()},
            )
            self.assertEqual(low_vote["pose_audit"]["reason"], "stage2_vote_confidence_below_gate")
            pose_before_missing = low_vote["translation_world_from_left_camera_mm"]
            missing = writer.consume(
                {"pair_id": 2, "persons_3d": [{"keypoints_3d": []}]},
                {"stage": {"operational": "stage2_feet_static_walker_moving", "vote_confidence": 1.0}},
                {"status": "accepted", "R_to_from": np.eye(3).tolist()},
            )
            self.assertEqual(missing["pose_audit"]["reason"], "strict_two_foot_input_missing")
            self.assertEqual(missing["translation_world_from_left_camera_mm"], pose_before_missing)
            writer.close(completed=True)

    def test_stage2_latch_tolerates_two_low_vote_frames_then_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "ground.json"
            self.ground_reference(reference)
            writer = RealtimeDynamicGroundWriter(
                root / "live.jsonl", reference, pose_mode="full_se3",
                maximum_stage2_vote_dropout_frames=2,
            )
            def payload(pair_id):
                return {"pair_id": pair_id, "persons_3d": [{"keypoints_3d": [
                    {"index": 15, "valid": True, "xyz": [-120, 0, -700]},
                    {"index": 16, "valid": True, "xyz": [120, 0, -700]},
                ]}]}
            identity = {"status": "accepted", "R_to_from": np.eye(3).tolist()}
            writer.consume(payload(0), {"stage": {"operational": "stage1_walker_static_human_moving"}})
            high = {"stage": {"operational": "stage2_feet_static_walker_moving", "vote_confidence": 1.0}}
            low = {"stage": {"operational": "stage2_feet_static_walker_moving", "vote_confidence": 0.33}}
            self.assertEqual(writer.consume(payload(1), high, identity)["pose_audit"]["status"], "accepted")
            first = writer.consume(payload(2), low, identity)
            second = writer.consume(payload(3), low, identity)
            third = writer.consume(payload(4), low, identity)
            self.assertEqual(first["pose_audit"]["status"], "accepted")
            self.assertTrue(first["pose_audit"]["stage2_vote_grace_applied"])
            self.assertEqual(second["pose_audit"]["status"], "accepted")
            self.assertEqual(third["pose_audit"]["reason"], "stage2_vote_confidence_below_gate")
            writer.close(completed=True)

    def test_no_wheel_transition_carry_and_confirmed_landing_reanchor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "ground.json"
            self.ground_reference(reference)
            writer = RealtimeDynamicGroundWriter(
                root / "live.jsonl", reference, pose_mode="full_se3",
                stage1_landed_support_reanchor=True,
            )
            def payload(pair_id, z):
                return {"pair_id": pair_id, "persons_3d": [{"keypoints_3d": [
                    {"index": 15, "valid": True, "xyz": [-120, 0, z]},
                    {"index": 16, "valid": True, "xyz": [120, 0, z]},
                ]}]}
            identity = {"status": "accepted", "R_to_from": np.eye(3).tolist()}
            stage1 = {"stage": {"operational": "stage1_walker_static_human_moving"}}
            stage2 = {"stage": {"operational": "stage2_feet_static_walker_moving", "vote_confidence": 1.0}}
            transition = {"stage": {"operational": "transition", "vote_confidence": 0.0}}
            writer.consume(payload(0, -700), stage1)
            lifted = writer.consume(payload(1, -730), stage2, identity)
            carried = writer.consume(payload(2, -720), transition, identity)
            landed = writer.consume(payload(3, -700), stage1, identity)
            self.assertEqual(lifted["pose_audit"]["status"], "accepted")
            self.assertTrue(carried["pose_audit"]["stage2_transition_carry_applied"])
            self.assertIn(carried["pose_audit"]["status"], {"accepted", "accepted_rotation_held"})
            self.assertEqual(landed["pose_audit"]["status"], "landed_support_reanchored")
            self.assertAlmostEqual(landed["translation_world_from_left_camera_mm"][2], 700.0)
            writer.close(completed=True)


if __name__ == "__main__":
    unittest.main()
