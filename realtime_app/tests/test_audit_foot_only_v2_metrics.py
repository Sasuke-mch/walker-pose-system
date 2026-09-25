"""Regression tests for the read-only foot-only v2 metric audit."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.audit_foot_only_v2_metrics import corrected_metrics


def _synthetic(n: int = 5):
    rng = np.random.default_rng(0)
    pred = np.abs(rng.normal(0.0, 0.5, (n, 17, 3))) + np.array([0.0, 0.0, 1.0])
    R01 = np.eye(3)
    T01_mm = np.array([-50.0, 0.0, 0.0])
    Rgc = np.tile(np.eye(3), (n, 1, 1))
    Tgc_mm = np.zeros((n, 3))
    foot_w = np.zeros((n, 2))
    foot_w[:, 0] = 0.5
    return pred, R01, T01_mm, Rgc, Tgc_mm, foot_w


class CorrectedMetricsTests(unittest.TestCase):
    def test_counts_and_rms_shapes(self):
        out = corrected_metrics(*_synthetic())
        self.assertEqual(out["negative_depth_count"],
                         out["negative_left_count"] + out["negative_right_count"])
        self.assertEqual(out["active_contact_frames"], 5)
        self.assertEqual(out["active_foot_observations"], 5)
        self.assertGreater(out["foot_proxy_weighted_rms_mm"], 0.0)

    def test_zero_weights_give_none_rms(self):
        pred, R01, T01, Rgc, Tgc, foot_w = _synthetic()
        out = corrected_metrics(pred, R01, T01, Rgc, Tgc, np.zeros_like(foot_w))
        self.assertIsNone(out["foot_proxy_weighted_rms_mm"])
        self.assertEqual(out["active_contact_frames"], 0)

    def test_shape_mismatch_and_negative_weights_rejected(self):
        pred, R01, T01, Rgc, Tgc, foot_w = _synthetic()
        with self.assertRaises(ValueError):
            corrected_metrics(pred[:, :10, :], R01, T01, Rgc, Tgc, foot_w)
        with self.assertRaises(ValueError):
            corrected_metrics(pred, R01, T01, Rgc, Tgc, -foot_w)


if __name__ == "__main__":
    unittest.main()
