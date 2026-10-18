from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.audit_foot_support_labels import build_audit, choose_window


class FootSupportAuditTests(unittest.TestCase):
    @staticmethod
    def motion() -> dict:
        frames = []
        for index in range(12):
            stage = "Stage 1" if index < 5 else "Transition" if index < 7 else "Stage 2"
            points = [[0.0, 0.0, 100.0] for _ in range(17)]
            left_x = 0.0 if index < 4 else float(index * 20)
            right_x = 0.0 if index < 5 else float(index * 5)
            points[15] = [left_x, 0.0, 10.0]
            points[16] = [right_x, 20.0, 12.0]
            frames.append({"pair": index, "stage": stage, "points_mm": points,
                           "strict": [True] * 17})
        return {"frames": frames}

    def test_transition_disables_contact_and_stage2_requires_both_feet(self):
        rows, summary = build_audit(self.motion(), minimum_support_frames=2)
        self.assertEqual(summary["stage_counts"]["transition"], 2)
        self.assertTrue(all(not row["contact_enabled"] for row in rows[5:7]))
        self.assertTrue(all(row["stage2_both_feet_candidate"] for row in rows[7:]))
        self.assertTrue(all(row["left"]["label"] == "stage2_contact_candidate" for row in rows[7:]))

    def test_invalid_is_not_interpolated(self):
        motion = self.motion()
        motion["frames"][2]["points_mm"][15] = None
        rows, _ = build_audit(motion, minimum_support_frames=2)
        self.assertEqual(rows[2]["left"]["label"], "invalid")
        self.assertIsNone(rows[2]["left"]["ground_position_mm"])
        self.assertEqual(rows[2]["left"]["contact_weight"], 0.0)

    def test_window_selection_requires_stage_boundary_and_evidence(self):
        rows, _ = build_audit(self.motion(), minimum_support_frames=2)
        result = choose_window(rows, [(0, 11), (0, 3)])
        self.assertEqual(result["selection_status"], "selected")
        self.assertEqual(result["selected"]["start"], 0)
        self.assertFalse(result["candidates"][1]["eligible"])


if __name__ == "__main__":
    unittest.main()
