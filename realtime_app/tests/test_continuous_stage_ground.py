import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pose_app.causal_stereo_background_pose import estimate_background_edge
from pose_app.continuous_stage_ground import ContinuousStageGroundWriter
from pose_app.dynamic_ground_live import STAGE2
from pose_app.replay_pair_timing import load_pair_timing
from pose_app.two_stage_motion import WorldPose


def stage(static=False, support=False):
    return {
        "stage": {
            "operational": STAGE2 if support else "transition",
            "candidate": STAGE2 if support else "transition",
            "background": {
                "fused": {
                    "available": True,
                    "motion_active": not static,
                    "static_confident": static,
                }
            },
        }
    }


def payload(i, feet=True, x=0):
    return {
        "pair_id": i,
        "pair_timestamp_sec": i * 0.05,
        "persons_3d": [
            {
                "keypoints_3d": [
                    {"index": j, "valid": True, "xyz": [v - x, 0, -700]}
                    for j, v in [(15, -120), (16, 120)]
                ]
                if feet
                else []
            }
        ],
    }


class ContinuousTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ref = self.root / "ground.json"
        self.ref.write_text(
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

    def writer(self):
        writer = ContinuousStageGroundWriter(self.root / "live.jsonl", self.ref)
        self.addCleanup(lambda: writer.close(completed=True))
        return writer

    def test_handoff_updates_before_confirmed_stage2_and_no_landing_reset(self):
        writer = self.writer()
        for i in range(3):
            writer.consume(payload(i), stage(static=True))
        edge = {"status": "accepted", "R_to_from": np.eye(3).tolist()}
        candidate = stage(support=True)
        candidate["stage"]["operational"] = "transition"
        row = writer.consume(payload(3, x=20), candidate, edge)
        self.assertEqual(row["pose_audit"]["status"], "accepted")
        self.assertEqual(row["pose_audit"]["mode"], "handoff")
        self.assertAlmostEqual(row["translation_world_from_left_camera_mm"][0], 3.0)
        landed = writer.consume(payload(4, x=20), stage(static=True), edge)
        self.assertEqual(
            landed["translation_world_from_left_camera_mm"],
            row["translation_world_from_left_camera_mm"],
        )

    def test_missing_rotation_never_claims_accepted_and_prediction_expires(self):
        writer = self.writer()
        for i in range(3):
            writer.consume(payload(i), stage(static=True))
        row = writer.consume(payload(3), stage(support=True), {"status": "unavailable"})
        self.assertEqual(row["pose_audit"]["reason"], "incomplete_rotation_chain")
        self.assertEqual(row["ankle_constraint_weight"], 0.0)
        for i in range(4, 10):
            row = writer.consume(
                payload(i, feet=False),
                stage(support=True),
                {"status": "accepted", "R_to_from": np.eye(3).tolist()},
            )
        self.assertEqual(row["pose_audit"]["status"], "unavailable")
        self.assertFalse(row["pose_audit"]["estimate_available"])
        static = writer.consume(payload(10), stage(static=True))
        self.assertFalse(static["pose_audit"]["world_rotation_chain_complete"])
        self.assertFalse(static["pose_audit"]["absolute_world_estimate_available"])

    def test_static_return_keeps_height_and_tilt(self):
        writer = self.writer()
        writer.consume(payload(0), stage(static=True))
        angle = np.deg2rad(3.0)
        rotation = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)],
            ]
        )
        writer.pose = WorldPose(rotation, np.array([5.0, 12.0, 730.0]))
        row = writer.consume(payload(1), stage(static=True))
        np.testing.assert_array_equal(row["rotation_world_from_left_camera"], rotation)
        np.testing.assert_array_equal(
            row["translation_world_from_left_camera_mm"], [5, 12, 730]
        )

    def test_one_foot_requires_independent_rotation(self):
        writer = self.writer()
        for i in range(3):
            writer.consume(payload(i), stage(static=True))
        p = payload(3, x=10)
        p["persons_3d"][0]["keypoints_3d"] = p["persons_3d"][0]["keypoints_3d"][:1]
        row = writer.consume(
            p,
            stage(support=True),
            {"status": "accepted", "R_to_from": np.eye(3).tolist()},
        )
        self.assertEqual(
            row["pose_audit"]["source"], "single_ankle_with_rotation_and_motion_gate"
        )

    def test_moving_stage1_ankles_are_not_anchors(self):
        writer = self.writer()
        for i in range(3):
            writer.consume(payload(i, x=i * 100), stage(static=True))
        writer.consume(
            payload(3),
            stage(support=True),
            {"status": "accepted", "R_to_from": np.eye(3).tolist()},
        )
        self.assertEqual(writer.anchors, {})

    def test_nonmonotonic_time_rejected(self):
        writer = self.writer()
        writer.consume(payload(0), stage(static=True))
        with self.assertRaises(ValueError):
            writer.consume(payload(0), stage(static=True))

    def test_first_frame_needs_no_previous_rotation_edge(self):
        writer = self.writer()
        row = writer.consume(
            payload(0),
            {"stage": {"operational": "warming_up"}},
            {"status": "unavailable", "reason": "first_frame"},
        )
        self.assertEqual(
            row["pose_audit"]["source"], "measured_static_reference_initialization"
        )
        self.assertTrue(row["pose_audit"]["world_rotation_chain_complete"])

    def test_host_timing_preserves_skew_and_elapsed_time(self):
        path = self.root / "frames.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "index": i,
                        "left_host_return_timestamp_ns": 1000000000 + i * 50000000,
                        "right_host_return_timestamp_ns": 1010000000 + i * 50000000,
                        "signed_host_delta_ms_right_minus_left": 10,
                    }
                    for i in range(3)
                ]
            )
        )
        values = load_pair_timing(path, 3)
        self.assertAlmostEqual(values[2][0] - values[0][0], 0.1)
        self.assertAlmostEqual(values[0][1] - values[0][0], 0.01)
        with self.assertRaises(ValueError):
            load_pair_timing(path, 2)

    def test_capture_pair_times_preserve_nanosecond_skew_and_reject_identity(self):
        from pose_app.replay_pair_timing import load_capture_pair_times

        capture = self.root / "capture"
        capture.mkdir()
        (capture / "metadata.json").write_text(
            json.dumps({"timestamp_type": "host_read_return_perf_counter_ns"})
        )
        pairs = [
            {
                "pair_id": i,
                "left_frame_id": 10 + i,
                "right_frame_id": 20 + i,
                "left_host_return_timestamp_ns": 5_000_000_000 + i * 50_000_000,
                "right_host_return_timestamp_ns": 5_010_000_000 + i * 50_000_000,
            }
            for i in range(3)
        ]
        with (capture / "stereo_pairs.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
            writer.writeheader()
            writer.writerows(pairs)
        manifest = [
            {"sequence_index": i, "source_pair_index": i, "left_frame_id": 10 + i,
             "right_frame_id": 20 + i}
            for i in range(3)
        ]
        manifest_path = self.root / "selection_manifest.csv"

        def write_manifest(rows):
            with manifest_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["sequence_index", "source_pair_index",
                                                            "left_frame_id", "right_frame_id"])
                writer.writeheader()
                writer.writerows(rows)

        write_manifest(manifest)
        (self.root / "metadata.json").write_text(json.dumps({"source_session": str(capture)}))
        times, skew, pair_ids, capture_path = load_capture_pair_times(self.root, 3)
        self.assertEqual(capture_path, capture / "stereo_pairs.csv")
        self.assertAlmostEqual(times[1] - times[0], 0.05)
        self.assertAlmostEqual(skew[0], 10.0)
        self.assertEqual(pair_ids.tolist(), [0, 1, 2])
        manifest[1]["left_frame_id"] = 999
        write_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "identity"):
            load_capture_pair_times(self.root, 3)

    def test_metric_background_pose_and_withheld_failure(self):
        rng = np.random.default_rng(71)
        xyz = rng.uniform([-500, -400, 1200], [500, 400, 3000], (80, 3))
        moved = xyz + [10, 2, -3]
        uv = moved[:, :2] / moved[:, 2:3]
        result = estimate_background_edge(xyz, uv)
        self.assertEqual(result["status"], "accepted")
        np.testing.assert_allclose(result["t_to_from_mm"], [10, 2, -3], atol=1e-3)
        uv[::5] += 0.1
        result = estimate_background_edge(xyz, uv)
        self.assertEqual(result["reason"], "withheld_reprojection_failed")

    def test_sparse_background_unavailable(self):
        result = estimate_background_edge(np.zeros((3, 3)), np.zeros((3, 2)))
        self.assertEqual(result["status"], "unavailable")

    def test_uncertified_vo_cannot_enter_world_pose(self):
        writer = self.writer()
        for i in range(3):
            writer.consume(payload(i), stage(static=True))
        row = writer.consume(
            payload(3, feet=False),
            stage(support=True),
            {"status": "accepted", "R_to_from": np.eye(3).tolist()},
            {
                "status": "accepted",
                "metric_pose_usable": False,
                "R_to_from": np.eye(3).tolist(),
                "t_to_from_mm": [100, 0, 0],
            },
        )
        self.assertEqual(row["translation_world_from_left_camera_mm"], [0, 0, 700])
        self.assertNotEqual(row["pose_audit"]["status"], "accepted")

    def test_timestamp_image_identity_is_checked(self):
        path = self.root / "frames.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "index": 0,
                        "left_host_return_timestamp_ns": 1000000000,
                        "right_host_return_timestamp_ns": 1010000000,
                        "signed_host_delta_ms_right_minus_left": 10,
                        "left_image": str(self.root / "wrong.png"),
                        "right_image": str(self.root / "wrong.png"),
                    }
                ]
            )
        )
        with self.assertRaisesRegex(ValueError, "image identity"):
            load_pair_timing(path, 1, input_pair_dir=self.root)


if __name__ == "__main__":
    unittest.main()
