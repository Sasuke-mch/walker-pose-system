from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_app.lower_limb_trajectory import (  # noqa: E402
    normalize_stereo_records,
    summarize_trajectory,
)


def point(index: int, name: str, *, valid: bool = True, reason=None) -> dict:
    return {
        "index": index,
        "name": name,
        "valid": valid,
        "xyz": [float(index), 1.0, 1000.0] if valid else None,
        "score": 0.9,
        "left_score": 0.8,
        "right_score": 0.85,
        "reprojection_error_left_px": 1.0,
        "reprojection_error_right_px": 1.2,
        "reprojection_error_mean_px": 1.1,
        "reason": reason,
    }


def person(*, right_ankle_valid: bool = True) -> dict:
    names = (
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )
    return {
        "keypoints_3d": [
            point(
                index,
                name,
                valid=right_ankle_valid or name != "right_ankle",
                reason=(
                    None
                    if right_ankle_valid or name != "right_ankle"
                    else "high_reprojection_error"
                ),
            )
            for index, name in zip(range(11, 17), names)
        ]
    }


def record(pair_id: int, persons: list[dict]) -> dict:
    return {
        "pair_id": pair_id,
        "pair_timestamp_sec": pair_id / 30.0,
        "left_frame_id": pair_id,
        "right_frame_id": pair_id,
        "timestamp_skew_ms": 2.0,
        "timestamp_type": "host_monotonic_read_return",
        "coordinate_frame": "left_camera",
        "length_unit": "millimeter",
        "persons_3d": persons,
    }


class LowerLimbTrajectoryTests(unittest.TestCase):
    def test_preserves_valid_points_and_rejection_reason(self):
        rows = normalize_stereo_records([record(0, [person(right_ankle_valid=False)])])
        self.assertTrue(rows[0]["points"]["left_hip"]["observed_3d"])
        self.assertEqual(
            rows[0]["points"]["left_hip"]["xyz_left_camera_mm"],
            [11.0, 1.0, 1000.0],
        )
        self.assertFalse(rows[0]["points"]["right_ankle"]["observed_3d"])
        self.assertEqual(
            rows[0]["points"]["right_ankle"]["reason"],
            "high_reprojection_error",
        )

    def test_never_selects_among_multiple_people(self):
        rows = normalize_stereo_records([record(0, [person(), person()])])
        self.assertEqual(
            rows[0]["frame_status"], "ambiguous_multiple_stereo_persons"
        )
        self.assertTrue(
            all(not value["observed_3d"] for value in rows[0]["points"].values())
        )

    def test_summary_does_not_bridge_missing_frame(self):
        rows = normalize_stereo_records(
            [record(0, [person()]), record(1, []), record(2, [person()])]
        )
        summary = summarize_trajectory(rows)
        self.assertEqual(summary["observed_lower_limb_points"], 12)
        self.assertEqual(
            summary["per_joint"]["left_knee"][
                "longest_contiguous_observed_run_frames"
            ],
            1,
        )
        self.assertEqual(
            summary["per_joint"]["left_knee"]["missing_or_rejected_reasons"],
            {"no_accepted_stereo_person": 1},
        )

    def test_rejects_non_increasing_pair_ids(self):
        with self.assertRaises(ValueError):
            normalize_stereo_records([record(1, []), record(1, [])])


if __name__ == "__main__":
    unittest.main()
