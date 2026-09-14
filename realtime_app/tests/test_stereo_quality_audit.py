from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_app.stereo_quality_audit import audit_saved_stereo_records  # noqa: E402


class IdentityCalibration:
    R = np.eye(3)

    def undistort_normalized(self, points, side):
        del side
        return np.asarray(points, dtype=float)


def record(pair_id: int, *, accepted: bool, include_person: bool = True) -> dict:
    left = [[0.1, 0.0, 0.9] for _ in range(17)]
    right = [[0.0, 0.0, 0.9] for _ in range(17)]
    if not include_person:
        return {"pair_id": pair_id, "file_name": f"pair_{pair_id:04d}.png", "pair_outcome": "association_failed", "timestamp_skew_ms": 2.0, "left": {"persons": [{"keypoints": left}]}, "right": {"persons": [{"keypoints": right}]}, "persons_3d": []}
    names = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
    xyz = {"left_hip": [0, 0, 2], "left_knee": [0, -1, 2], "left_ankle": [0, -2, 2], "right_hip": [1, 0, 2], "right_knee": [1, -1, 2], "right_ankle": [1, -2, 2]}
    points = []
    for index, name in enumerate(names):
        valid = accepted and name in xyz
        points.append({"name": name, "valid": valid, "xyz": xyz.get(name, [0, 0, 2]), "left_score": 0.9, "right_score": 0.9, "depth_left": 2.0, "depth_right": 2.0, "reprojection_error_mean_px": 1.0, "reason": None if valid else "high_reprojection_error"})
    return {"pair_id": pair_id, "file_name": f"pair_{pair_id:04d}.png", "pair_outcome": "matched", "timestamp_skew_ms": 2.0, "left": {"persons": [{"keypoints": left}]}, "right": {"persons": [{"keypoints": right}]}, "persons_3d": [{"association_cost": 0.01, "common_keypoints": 17, "keypoints_3d": points}]}


class StereoQualityAuditTests(unittest.TestCase):
    def test_keeps_association_failure_and_does_not_bridge_bone_gap(self):
        records = [record(0, accepted=True), record(1, accepted=False, include_person=False), record(2, accepted=True)]
        audit = audit_saved_stereo_records(records, calibration=IdentityCalibration(), keypoint_threshold=0.25, manifest_by_file_name={item["file_name"]: {"abs_host_delta_ms": "3.5"} for item in records})
        self.assertEqual(audit["summary"]["frames"], 3)
        self.assertEqual(audit["summary"]["matched_frames"], 2)
        left_thigh = next(row for row in audit["bone_rows"] if row["bone"] == "left_thigh")
        self.assertEqual(left_thigh["direct_observed_frames"], 2)
        self.assertEqual(left_thigh["longest_direct_observed_run_frames"], 1)
        self.assertEqual(left_thigh["consecutive_abs_change_mm_n"], 0)
        reasons = {(row["joint"], row["outcome_or_reason"]): row["count"] for row in audit["rejection_rows"]}
        self.assertEqual(reasons[("left_ankle", "pair_association_failed")], 1)
        self.assertEqual(audit["summary"]["host_pairing_delta_ms"]["median"], 3.5)


if __name__ == "__main__":
    unittest.main()
