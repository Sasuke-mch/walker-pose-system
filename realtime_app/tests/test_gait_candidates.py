from __future__ import annotations

import unittest

from pose_app.gait_candidates import derive_gait_candidates


def record(pair_id: int, left: float | None, right: float | None, separation: float | None) -> dict:
    def metric(value: float | None, unit: str) -> dict:
        return {
            "available": value is not None,
            "value": value,
            "unit": unit,
            "reason": None if value is not None else "missing_or_rejected:source",
        }

    return {
        "pair_id": pair_id,
        "pair_timestamp_sec": pair_id / 30.0,
        "coordinate_frame": "left_camera",
        "length_unit": "millimeter",
        "source_frame_status": "accepted_single_person",
        "metrics": {
            "left_knee_angle_deg": metric(left, "degree"),
            "right_knee_angle_deg": metric(right, "degree"),
            "ankle_separation_mm": metric(separation, "millimeter"),
        },
    }


class GaitCandidateTests(unittest.TestCase):
    def test_only_strict_adjacent_direct_observations_make_turning_points(self) -> None:
        output = derive_gait_candidates(
            [
                record(0, 170.0, 170.0, 200.0),
                record(1, 150.0, 180.0, 240.0),
                record(2, 160.0, 175.0, 220.0),
            ]
        )
        events = output["events"]
        self.assertEqual(len(events), 3)
        self.assertEqual(
            {(event["signal"], event["turning_point"]) for event in events},
            {
                ("left_knee_angle_deg", "strict_local_min"),
                ("right_knee_angle_deg", "strict_local_max"),
                ("ankle_separation_mm", "strict_local_max"),
            },
        )
        self.assertTrue(all(event["support_pair_ids"] == [0, 1, 2] for event in events))
        self.assertTrue(all("not heel strike" in event["interpretation"] for event in events))

    def test_missing_middle_value_cannot_bridge_to_candidate(self) -> None:
        output = derive_gait_candidates(
            [
                record(0, 170.0, 170.0, 200.0),
                record(1, None, 180.0, 240.0),
                record(2, 160.0, 175.0, 220.0),
            ]
        )
        self.assertFalse(any(event["signal"] == "left_knee_angle_deg" for event in output["events"]))
        self.assertEqual(output["summary"]["signal_summary"]["left_knee_angle_deg"]["available_frames"], 2)

    def test_gait_parameters_are_explicitly_unavailable(self) -> None:
        output = derive_gait_candidates(
            [record(0, 170.0, 170.0, 200.0), record(1, 160.0, 180.0, 220.0), record(2, 170.0, 170.0, 200.0)]
        )
        parameters = output["summary"]["gait_parameter_candidates"]
        self.assertFalse(any(parameter["available"] for parameter in parameters.values()))
        self.assertEqual(parameters["step_length_mm"]["reason"], "target_coordinate_frame_not_measured_locked")
        self.assertEqual(parameters["cadence_steps_per_min"]["reason"], "no_valid_contact_event_definition")


if __name__ == "__main__":
    unittest.main()
