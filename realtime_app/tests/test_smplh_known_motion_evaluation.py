"""The evaluation's analytic derivatives must include camera-motion terms."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/"tools"))
from evaluate_smplh_motion_known_motion import analytic_motion


@pytest.mark.parametrize("kind", ["sine", "turn", "camera"])
def test_analytic_truth_matches_continuous_positions_and_initial_geometry(kind):
    points = np.random.default_rng(3).normal(0, .2, (17, 3))+[0, 0, 2]
    initial = analytic_motion(points, np.array([0.]), kind)[0][0]
    np.testing.assert_allclose(initial, points, atol=1e-14)
    now, eps = .713, 1e-4
    samples = analytic_motion(points, now+eps*np.array([-2, -1, 0, 1, 2]), kind)
    positions = samples[0]
    velocity = (positions[0]-8*positions[1]+8*positions[3]-positions[4])/(12*eps)
    acceleration = (-positions[0]+16*positions[1]-30*positions[2]+16*positions[3]-positions[4])/(12*eps**2)
    np.testing.assert_allclose(samples[1][2], velocity, atol=1e-10)
    np.testing.assert_allclose(samples[2][2], acceleration, atol=3e-7)
    assert np.linalg.norm(samples[1][2]) > .01
    lengths = np.linalg.norm(positions[:, 5]-positions[:, 6], axis=-1)
    np.testing.assert_allclose(lengths, lengths[0], atol=1e-13)
