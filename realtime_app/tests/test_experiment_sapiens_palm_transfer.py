import unittest

import numpy as np
from experiment_sapiens_palm_transfer import coefficient_target, to_raw


class PalmTransferTests(unittest.TestCase):
    def test_inverse_rotations(self):
        point = np.array([[100., 200.]])
        np.testing.assert_array_equal(to_raw(point, "left"), [[1719., 100.]])
        np.testing.assert_array_equal(to_raw(point, "right"), [[200., 979.]])

    def test_projected_coefficient_and_degenerate(self):
        w, p, m = np.array([0., 0.]), np.array([10., 0.]), np.array([2., 3.])
        self.assertEqual(coefficient_target(w, p, m), .2)
        self.assertEqual(coefficient_target(w * 3, p * 3, m * 3), .2)
        self.assertIsNone(coefficient_target(w, w, m))


if __name__ == "__main__":
    unittest.main()
