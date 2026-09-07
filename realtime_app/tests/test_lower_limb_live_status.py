from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from pose_app.lower_limb_live_status import LowerLimbLiveStatusWriter


def stereo_record(pair_id: int, knee_x: float) -> dict:
    points = {
        "left_hip": [0.0, 0.0, 1000.0],
        "right_hip": [100.0, 0.0, 1000.0],
        "left_knee": [knee_x, 0.0, 700.0],
        "right_knee": [100.0, 0.0, 700.0],
        "left_ankle": [0.0, 0.0, 400.0],
        "right_ankle": [100.0, 0.0, 400.0],
    }
    indices = {
        "left_hip": 11, "right_hip": 12, "left_knee": 13,
        "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
    }
    return {
        "pair_id": pair_id,
        "pair_timestamp_sec": pair_id / 30.0,
        "coordinate_frame": "left_camera",
        "length_unit": "millimeter",
        "persons_3d": [{
            "keypoints_3d": [
                {"index": indices[name], "name": name, "valid": True, "xyz": xyz, "reason": None}
                for name, xyz in points.items()
            ]
        }],
    }


class LowerLimbLiveStatusTests(unittest.TestCase):
    def test_appends_current_state_and_delays_turning_point_until_following_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "lower_limb_live_status.jsonl"
            writer = LowerLimbLiveStatusWriter(output)
            first = writer.consume(stereo_record(0, 0.0))
            second = writer.consume(stereo_record(1, 90.0))
            third = writer.consume(stereo_record(2, 0.0))
            summary = writer.close(completed=True)

            self.assertEqual(first["t3_coordinates"]["status"], "not_configured")
            self.assertEqual(second["t4_noncontact_candidates"]["newly_confirmed_candidates"], [])
            self.assertTrue(any(
                event["signal"] == "left_knee_angle_deg"
                for event in third["t4_noncontact_candidates"]["newly_confirmed_candidates"]
            ))
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["pairs"], 3)
            self.assertEqual(summary["accepted_contact_events"], 0)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["pair_id"] for row in rows], [0, 1, 2])

    def test_rejects_nonincreasing_pair_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            writer = LowerLimbLiveStatusWriter(Path(temporary) / "status.jsonl")
            writer.consume(stereo_record(1, 0.0))
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                writer.consume(stereo_record(1, 0.0))
            writer.close(completed=False)


if __name__ == "__main__":
    unittest.main()
