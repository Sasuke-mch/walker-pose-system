from pathlib import Path
import sys
import unittest
from dataclasses import replace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.analyze_jointwise_triangulation_continuity import (
    LOWER_BODY,
    JointObservation,
    analyze_conditions,
)


class JointwiseContinuityTests(unittest.TestCase):
    def condition(self):
        observations = {}
        for pair_id, thigh_length in enumerate((100.0, 110.0, None, 200.0)):
            row = {}
            for joint in LOWER_BODY:
                valid = not (pair_id == 2 and joint == "left_knee")
                xyz = np.zeros(3, dtype=float) if valid else None
                if joint == "left_knee" and valid:
                    xyz = np.asarray([thigh_length, 0.0, 0.0], dtype=float)
                elif joint == "right_hip":
                    xyz = np.asarray([0.0, 10.0, 0.0], dtype=float)
                elif joint == "right_knee":
                    xyz = np.asarray([0.0, 110.0, 0.0], dtype=float)
                elif joint == "left_ankle":
                    xyz = np.asarray([0.0, 200.0, 0.0], dtype=float)
                elif joint == "right_ankle":
                    xyz = np.asarray([0.0, 210.0, 0.0], dtype=float)
                row[joint] = JointObservation(
                    pair_id=pair_id,
                    file_name=f"pair_{pair_id:04d}.png",
                    joint=joint,
                    valid=valid,
                    reason=None if valid else "out_of_raw_image_bounds",
                    xyz=xyz,
                )
            observations[pair_id] = row
        return observations

    def test_segment_delta_does_not_bridge_a_missing_frame(self):
        frame_rows, summary_rows, rejection_rows, _ = analyze_conditions(
            {"pmpose": self.condition()}, fps=30.0
        )
        left_thigh = [row for row in frame_rows if row["segment"] == "left_thigh"]
        self.assertEqual([row["segment_observed"] for row in left_thigh], [True, True, False, True])
        self.assertEqual(left_thigh[1]["frame_to_frame_abs_length_delta_mm"], 10.0)
        self.assertIsNone(left_thigh[3]["frame_to_frame_abs_length_delta_mm"])

        summary = next(row for row in summary_rows if row["segment"] == "left_thigh")
        self.assertEqual(summary["observed_segment_frames"], 3)
        self.assertEqual(summary["adjacent_observed_segment_pairs"], 1)
        self.assertEqual(summary["absolute_delta_max_mm"], 10.0)
        self.assertEqual(summary["longest_observed_segment_run_frames"], 2)
        self.assertIn(
            {"condition": "pmpose", "joint": "left_knee", "outcome": "out_of_raw_image_bounds", "frames": 1},
            rejection_rows,
        )

    def test_segment_run_does_not_cross_a_pair_id_gap(self):
        source = self.condition()
        gapped = {
            0: source[0],
            2: {
                joint: replace(
                    observation,
                    pair_id=2,
                    file_name="pair_0002.png",
                )
                for joint, observation in source[1].items()
            },
        }
        _, summary_rows, _, _ = analyze_conditions({"pmpose": gapped}, fps=30.0)
        summary = next(row for row in summary_rows if row["segment"] == "left_thigh")
        self.assertEqual(summary["observed_segment_frames"], 2)
        self.assertEqual(summary["observed_segment_runs"], 2)
        self.assertEqual(summary["longest_observed_segment_run_frames"], 1)
        self.assertEqual(summary["adjacent_observed_segment_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
