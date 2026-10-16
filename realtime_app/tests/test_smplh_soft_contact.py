"""Verify contact geometry, normalization and fixed support hypotheses."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from probe_smplh_soft_contact import BALL, SURFACE, contact_residuals, support_labels


class SoftContactTests(unittest.TestCase):
    def test_unilateral_tolerance_and_vertex_count_normalization(self):
        points = np.zeros((1, len(SURFACE), 3))
        points[..., 2] = -.005
        np.testing.assert_array_equal(contact_residuals(points, np.eye(3), np.zeros(3), [0, 0]), 0.)
        points[..., 2] = -.010
        r = contact_residuals(points, np.eye(3), np.zeros(3), [0, 0])
        self.assertAlmostEqual((r@r.T).item(), 1.)

    def test_true_ball_minimum_and_swing_not_attracted(self):
        points = np.zeros((2, len(SURFACE), 3))
        points[..., 2] = .030
        points[:, BALL[0][0], 2] = .010
        r = contact_residuals(points, np.eye(3), np.zeros(3), [1, 0])
        self.assertAlmostEqual(.5*r[0, -2]**2, np.sqrt(2.)-1.)
        np.testing.assert_array_equal(r[:, -1], 0.)
        # A camera-space z offset is removed only through the supplied ground.
        shifted = contact_residuals(points + [0, 0, 2], np.eye(3), np.array([0, 0, -2]), [1, 0])
        np.testing.assert_allclose(r, shifted, atol=1e-13)
        rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        camera_points = points @ rotation
        rotated = contact_residuals(camera_points, rotation, np.zeros(3), [1, 0])
        np.testing.assert_allclose(r, rotated, atol=1e-13)

    def test_labels_leave_transitions_and_preexperiment_unconstrained(self):
        mask = support_labels()
        self.assertEqual(mask.sum(), 26)
        self.assertFalse(mask[:35].any())
        self.assertFalse(mask[48].any())
        self.assertFalse(mask[35:49, 0].any())
        self.assertFalse(mask[48:, 1].any())


if __name__ == '__main__':
    unittest.main()
