from __future__ import annotations

import math
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from pose_app.fixed_coordinate import (
    RigidTransform,
    load_coordinate_transform,
    summarize_static_reference,
    summarize_transformed_trajectory,
    transform_trajectory_record,
)


def transform_mapping() -> dict:
    return {
        "status": "measured_locked",
        "transform_id": "test_left_to_walker_ground",
        "source_coordinate_frame": "left_camera",
        "target_coordinate_frame": "walker_ground",
        "length_unit": "millimeter",
        "rotation_target_from_source": [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        "translation_target_from_source_mm": [100.0, -50.0, 25.0],
        "reference_definition": {
            "method": "synthetic unit-test reference",
            "physical_reference": "synthetic marker set",
            "capture_session": "unit_test_static_session",
        },
    }


def trajectory_record() -> dict:
    points = {}
    for index, name in ((11, "left_hip"), (12, "right_hip"), (13, "left_knee"), (14, "right_knee"), (15, "left_ankle"), (16, "right_ankle")):
        points[name] = {
            "index": index,
            "name": name,
            "observed_3d": True,
            "xyz_left_camera_mm": [float(index), float(index + 1), float(index + 2)],
            "reason": None,
        }
    points["right_ankle"] = {
        "index": 16,
        "name": "right_ankle",
        "observed_3d": False,
        "xyz_left_camera_mm": None,
        "reason": "high_reprojection_error",
    }
    return {
        "pair_id": 4,
        "pair_timestamp_sec": 1.0,
        "coordinate_frame": "left_camera",
        "length_unit": "millimeter",
        "frame_status": "accepted_single_person",
        "points": points,
    }


class FixedCoordinateTests(unittest.TestCase):
    def test_proper_rigid_transform_round_trips(self) -> None:
        transform = RigidTransform.from_mapping(transform_mapping())
        target = transform.apply([10.0, 20.0, 30.0])
        self.assertEqual(target, [80.0, -40.0, 55.0])
        recovered = transform.invert(target)
        self.assertLess(max(abs(a - b) for a, b in zip(recovered, [10.0, 20.0, 30.0])), 1e-10)

    def test_rejects_template_and_non_rigid_rotation(self) -> None:
        template = transform_mapping()
        template["status"] = "template_not_measured"
        with self.assertRaisesRegex(ValueError, "measured_locked"):
            RigidTransform.from_mapping(template)

        non_rigid = transform_mapping()
        non_rigid["rotation_target_from_source"] = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        with self.assertRaisesRegex(ValueError, "proper rigid rotation"):
            RigidTransform.from_mapping(non_rigid)

    def test_test_only_transform_requires_explicit_opt_in(self) -> None:
        test_only = transform_mapping()
        test_only["status"] = "test_only"
        with self.assertRaisesRegex(ValueError, "allow_test_transform"):
            RigidTransform.from_mapping(test_only)
        self.assertEqual(
            RigidTransform.from_mapping(test_only, allow_test_transform=True).status,
            "test_only",
        )

    def test_transforms_observations_and_propagates_rejection(self) -> None:
        result = transform_trajectory_record(trajectory_record(), RigidTransform.from_mapping(transform_mapping()))
        self.assertEqual(result["source_coordinate_frame"], "left_camera")
        self.assertEqual(result["target_coordinate_frame"], "walker_ground")
        self.assertEqual(result["points"]["left_hip"]["xyz_left_camera_mm"], [11.0, 12.0, 13.0])
        self.assertEqual(result["points"]["left_hip"]["xyz_target_mm"], [88.0, -39.0, 38.0])
        self.assertEqual(result["points"]["left_hip"]["target_coordinate_status"], "transformed_direct_observation")
        self.assertIsNone(result["points"]["right_ankle"]["xyz_target_mm"])
        self.assertEqual(result["points"]["right_ankle"]["target_coordinate_reason"], "high_reprojection_error")

    def test_summary_roundtrip_and_static_reference_residual(self) -> None:
        transform = RigidTransform.from_mapping(transform_mapping())
        rows = [transform_trajectory_record(trajectory_record(), transform)]
        summary = summarize_transformed_trajectory(rows, transform)
        self.assertEqual(summary["source_observed_points"], 5)
        self.assertEqual(summary["target_transformed_points"], 5)
        self.assertEqual(summary["source_rejections_propagated"], 1)
        self.assertLess(summary["max_numeric_inverse_roundtrip_error_mm"], 1e-10)

        reference = [{
            "reference_id": "marker_a",
            "sample_id": "marker_a_001",
            "capture_session": "unit_test_static_session",
            "xyz_left_camera_mm": [10.0, 20.0, 30.0],
            "expected_xyz_target_mm": [80.0, -40.0, 55.0],
        }, {
            "reference_id": "marker_a",
            "sample_id": "marker_a_002",
            "capture_session": "unit_test_static_session",
            "xyz_left_camera_mm": [10.0, 20.0, 30.0],
            "expected_xyz_target_mm": [80.0, -40.0, 56.0],
        }]
        static = summarize_static_reference(reference, transform)
        self.assertEqual(static["samples"], 2)
        self.assertAlmostEqual(static["overall_max_residual_mm"], 1.0)
        self.assertAlmostEqual(static["overall_median_residual_mm"], 0.5)

    def test_measured_transform_requires_accepted_static_evidence(self) -> None:
        samples = [{
            "reference_id": "marker_a",
            "sample_id": "marker_a_001",
            "capture_session": "unit_test_static_session",
            "xyz_left_camera_mm": [10.0, 20.0, 30.0],
            "expected_xyz_target_mm": [80.0, -40.0, 55.0],
        }, {
            "reference_id": "marker_b",
            "sample_id": "marker_b_001",
            "capture_session": "unit_test_static_session",
            "xyz_left_camera_mm": [11.0, 20.0, 30.0],
            "expected_xyz_target_mm": [80.0, -39.0, 55.0],
        }]
        criteria = {
            "criterion_id": "unit_test_field_criteria",
            "minimum_distinct_reference_ids": 2,
            "minimum_samples_per_reference": 1,
            "maximum_p95_residual_mm": 0.0,
            "maximum_max_residual_mm": 0.0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "static_reference.jsonl"
            raw.write_text("".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8")
            transform_mapping_with_evidence = transform_mapping()
            transform_mapping_with_evidence["reference_definition"]["static_reference_evidence_file"] = "evidence.json"
            transform_file = root / "transform.json"
            transform_file.write_text(json.dumps(transform_mapping_with_evidence), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evidence is missing"):
                load_coordinate_transform(transform_file)

            (root / "criteria.json").write_text(json.dumps(criteria), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parents[1] / "tools" / "validate_static_reference_lock.py"),
                    "--transform", str(transform_file),
                    "--static-reference", str(raw),
                    "--criteria", str(root / "criteria.json"),
                    "--output", str(root / "evidence.json"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            evidence = json.loads((root / "evidence.json").read_text(encoding="utf-8"))
            self.assertEqual(evidence["status"], "accepted")
            self.assertEqual(load_coordinate_transform(transform_file).status, "measured_locked")

            changed_transform = json.loads(transform_file.read_text(encoding="utf-8"))
            changed_transform["translation_target_from_source_mm"][0] = 101.0
            transform_file.write_text(json.dumps(changed_transform), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "translation_target_from_source_mm"):
                load_coordinate_transform(transform_file)


if __name__ == "__main__":
    unittest.main()
