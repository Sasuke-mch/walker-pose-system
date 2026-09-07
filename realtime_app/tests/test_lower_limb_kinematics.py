from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_app.lower_limb_kinematics import (  # noqa: E402
    derive_frame_kinematics,
    derive_kinematics,
    summarize_kinematics,
)


def trajectory_record(pair_id: int = 0, *, missing: set[str] | None = None) -> dict:
    missing = missing or set()
    xyz = {
        "left_hip": [0.0, 0.0, 0.0],
        "right_hip": [4.0, 0.0, 0.0],
        "left_knee": [0.0, -3.0, 0.0],
        "right_knee": [4.0, -3.0, 0.0],
        "left_ankle": [4.0, -3.0, 0.0],
        "right_ankle": [8.0, -3.0, 0.0],
    }
    return {
        "pair_id": pair_id,
        "pair_timestamp_sec": pair_id / 30.0,
        "coordinate_frame": "left_camera",
        "length_unit": "millimeter",
        "frame_status": "accepted_single_person",
        "points": {
            name: {
                "observed_3d": name not in missing,
                "xyz_left_camera_mm": None if name in missing else value,
                "reason": "high_reprojection_error" if name in missing else None,
            }
            for name, value in xyz.items()
        },
    }


def rigid_transform(record: dict) -> dict:
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.asarray([100.0, -30.0, 50.0])
    copied = trajectory_record(record["pair_id"])
    for name, point in record["points"].items():
        if not point["observed_3d"]:
            copied["points"][name] = point.copy()
            continue
        value = rotation @ np.asarray(point["xyz_left_camera_mm"]) + translation
        copied["points"][name]["xyz_left_camera_mm"] = value.tolist()
    return copied


class LowerLimbKinematicsTests(unittest.TestCase):
    def test_expected_lengths_angles_and_pelvis(self):
        result = derive_frame_kinematics(trajectory_record())
        self.assertEqual(
            result["metrics"]["pelvis_center"]["xyz_left_camera_mm"], [2.0, 0.0, 0.0]
        )
        self.assertEqual(result["metrics"]["left_thigh_length_mm"]["value"], 3.0)
        self.assertEqual(result["metrics"]["left_shank_length_mm"]["value"], 4.0)
        self.assertEqual(result["metrics"]["left_knee_angle_deg"]["value"], 90.0)
        self.assertEqual(result["metrics"]["ankle_separation_mm"]["value"], 4.0)

    def test_missing_input_remains_unavailable(self):
        result = derive_frame_kinematics(trajectory_record(missing={"left_hip"}))
        self.assertFalse(result["metrics"]["pelvis_center"]["available"])
        self.assertEqual(
            result["metrics"]["pelvis_center"]["reason"],
            "missing_or_rejected:left_hip",
        )
        self.assertFalse(result["metrics"]["left_knee_angle_deg"]["available"])
        self.assertTrue(result["metrics"]["right_knee_angle_deg"]["available"])

    def test_rigid_transform_preserves_invariant_scalars(self):
        source = derive_frame_kinematics(trajectory_record())
        transformed = derive_frame_kinematics(rigid_transform(trajectory_record()))
        for name in (
            "left_thigh_length_mm",
            "right_thigh_length_mm",
            "left_shank_length_mm",
            "right_shank_length_mm",
            "left_knee_angle_deg",
            "right_knee_angle_deg",
            "ankle_separation_mm",
        ):
            self.assertAlmostEqual(
                source["metrics"][name]["value"],
                transformed["metrics"][name]["value"],
            )

    def test_summary_does_not_bridge_unavailable_frame(self):
        rows = derive_kinematics(
            [
                trajectory_record(0),
                trajectory_record(1, missing={"left_ankle"}),
                trajectory_record(2),
            ]
        )
        summary = summarize_kinematics(rows)
        left_knee = summary["per_metric"]["left_knee_angle_deg"]
        self.assertEqual(left_knee["available_frames"], 2)
        self.assertEqual(left_knee["longest_contiguous_available_run_frames"], 1)
        self.assertEqual(
            left_knee["unavailable_reasons"],
            {"missing_or_rejected:left_ankle": 1},
        )


if __name__ == "__main__":
    unittest.main()
