from __future__ import annotations

import unittest
import subprocess
import sys

import numpy as np

from pose_app.temporal_body_prefit import (
    TemporalBodyPrefitConfig,
    observation_weights,
    prefit_acceptance_audit,
    sequence_metrics,
    temporal_prefit,
)


class TemporalBodyPrefitTests(unittest.TestCase):
    def test_quality_weights_prefer_strict_low_error_observations(self) -> None:
        errors = np.array([[2.0, 20.0, 2.0]])
        strict = np.array([[True, True, False]])
        valid = np.ones_like(strict)
        weights = observation_weights(errors, strict, valid)
        self.assertGreater(weights[0, 0], weights[0, 1])
        self.assertGreater(weights[0, 0], weights[0, 2])

    def test_prefit_reduces_jitter_and_contact_residual(self) -> None:
        code = r'''
import numpy as np
from pose_app.temporal_body_prefit import TemporalBodyPrefitConfig, sequence_metrics, temporal_prefit
rng = np.random.default_rng(7)
frames = 24
base = np.zeros((frames, 17, 3), dtype=np.float64)
for joint in range(17):
    base[:, joint, 0] = joint * 25.0 + np.linspace(0, 80, frames)
    base[:, joint, 1] = (joint % 3) * 40.0
    base[:, joint, 2] = 900.0 + (joint % 5) * 35.0
noisy = base + rng.normal(0, 7.0, base.shape)
errors = np.full((frames, 17), 3.0)
strict = np.ones((frames, 17), dtype=bool)
targets = np.zeros_like(noisy)
contact = np.zeros((frames, 17))
targets[5:18, 15] = np.median(noisy[5:18, 15], axis=0)
contact[5:18, 15] = 1.0
fitted, _ = temporal_prefit(
    noisy, errors, strict, targets, contact,
    TemporalBodyPrefitConfig(iterations=140, learning_rate=0.02),
)
metrics = sequence_metrics(noisy, fitted, targets, contact)
assert metrics["fitted_motion"]["acceleration_p95_mm_per_frame2"] < metrics["raw_motion"]["acceleration_p95_mm_per_frame2"]
assert metrics["contact_residual_after_median_mm"] < metrics["contact_residual_before_median_mm"]
'''
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)

    def test_acceptance_audit_passes_bounded_improvement(self) -> None:
        frames = 12
        raw = np.zeros((frames, 17, 3), dtype=np.float64)
        fitted = np.zeros_like(raw)
        for joint in range(17):
            jitter = ((-1) ** np.arange(frames)) * 4.0
            jitter[[0, -1]] = 0.0
            raw[:, joint, 0] = np.linspace(0.0, 110.0, frames) + jitter
            fitted[:, joint, 0] = np.linspace(0.0, 110.0, frames)
            raw[:, joint, 1] = fitted[:, joint, 1] = joint * 20.0
            raw[:, joint, 2] = fitted[:, joint, 2] = 800.0 + joint * 10.0
        raw[5, 3, 0] += 60.0
        errors = np.full((frames, 17), 2.0)
        strict = np.ones((frames, 17), dtype=bool)
        errors[5, 3] = 30.0
        strict[5, 3] = False
        contacts = np.zeros_like(raw)
        contact_weights = np.zeros((frames, 17))
        metrics = sequence_metrics(raw, fitted, contacts, contact_weights)
        audit = prefit_acceptance_audit(raw, fitted, errors, strict, metrics)
        self.assertTrue(audit["gates"]["acceleration_p95_reduction"]["passed"])
        self.assertLess(audit["gates"]["pelvis_net_displacement_change"]["value"], 0.02)
        self.assertTrue(audit["top_corrections"][0]["non_strict_high_error"])


if __name__ == "__main__":
    unittest.main()
