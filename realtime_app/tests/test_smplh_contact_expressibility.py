"""Static chart and robust-loss contracts; no learned assets required."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from probe_smplh_contact_expressibility import (
    FROZEN,
    LOWER,
    local_bodies,
    radial_huber_residuals,
)


class ContactExpressibilityTests(unittest.TestCase):
    def test_right_rotation_retraction_preserves_all_other_joints(self):
        base = Rotation.from_rotvec(np.arange(63).reshape(21, 3)*.001).as_matrix()
        offsets = np.zeros((2, 6, 3))
        offsets[1, 4, 0] = .2
        result = local_bodies(base, offsets)
        np.testing.assert_array_equal(result[:, FROZEN], np.broadcast_to(base[FROZEN], result[:, FROZEN].shape))
        np.testing.assert_allclose(result[0], base, atol=1e-15)
        np.testing.assert_allclose(result[1, LOWER[4]], base[LOWER[4]] @ Rotation.from_rotvec([.2, 0, 0]).as_matrix())

    def test_pixel_huber_is_radial_and_does_not_double_robustify_contact(self):
        raw = np.array([[[0., 0.], [3., 4.], [30., 40.]]])
        encoded = radial_huber_residuals(raw)
        expected = np.array([[0., .5*25, 25*(50-12.5)]])/64
        np.testing.assert_allclose(.5*np.sum(encoded**2, axis=-1), expected)


if __name__ == '__main__':
    unittest.main()
