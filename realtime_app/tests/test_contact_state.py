import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pose_app.contact_state import (
    ContactConfig, ContactObservation, ContactState, ContactStateMachine,
    surface_partition_metrics,
)


class ContactStateTests(unittest.TestCase):
    def test_enter_hysteresis_is_time_based(self):
        sm = ContactStateMachine(ContactConfig(enter_duration_ms=80.0))
        states = []
        for t in (0.00, 0.03, 0.07, 0.081):
            states.append(sm.update(ContactObservation(t, 0.010, 0.01, 0.01, 1.0)).state)
        self.assertEqual(states[:3], [ContactState.CONTACT_CANDIDATE] * 3)
        self.assertEqual(states[3], ContactState.STICKING_CANDIDATE)

    def test_exit_hysteresis_and_unknown_reset(self):
        sm = ContactStateMachine(ContactConfig(enter_duration_ms=1.0, exit_duration_ms=100.0))
        sm.update(ContactObservation(0.0, 0.01, 0.0, 0.0, 1.0))
        sm.update(ContactObservation(0.01, 0.01, 0.0, 0.0, 1.0))
        self.assertEqual(sm.state, ContactState.STICKING_CANDIDATE)
        self.assertEqual(sm.update(ContactObservation(0.02, 0.03, 0.0, 0.0, 1.0)).state,
                         ContactState.STICKING_CANDIDATE)
        self.assertEqual(sm.update(ContactObservation(0.13, 0.03, 0.0, 0.0, 1.0)).state,
                         ContactState.SEPARATED)
        self.assertEqual(sm.update(ContactObservation(0.14, np.nan, 0.0, 0.0, 1.0)).state,
                         ContactState.UNKNOWN)

    def test_quality_and_ground_gates_do_not_hold_previous_state(self):
        sm = ContactStateMachine(ContactConfig(enter_duration_ms=1.0))
        sm.update(ContactObservation(0.0, 0.01, 0.0, 0.0, 1.0))
        sm.update(ContactObservation(0.01, 0.01, 0.0, 0.0, 1.0))
        d = sm.update(ContactObservation(0.02, 0.01, 0.0, 0.0, 0.1))
        self.assertEqual(d.state, ContactState.UNKNOWN)
        self.assertEqual(d.reason, "quality_below_gate")

    def test_surface_partition_reports_penetration(self):
        points = np.array([[0, 0, -0.01], [0, 0, 0.02], [0, 0, 0.03]], float)
        got = surface_partition_metrics(points, {"heel": [0, 1], "toe": [2], "bad": [9]})
        self.assertAlmostEqual(got["heel"]["penetration_fraction"], 0.5)
        self.assertAlmostEqual(got["toe"]["minimum_height_m"], 0.03)
        self.assertEqual(got["bad"]["available"], 0)


if __name__ == "__main__":
    unittest.main()
