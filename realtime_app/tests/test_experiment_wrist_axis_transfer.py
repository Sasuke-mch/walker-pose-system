import unittest

import numpy as np
from experiment_wrist_axis_transfer import axis_target, correct


class WristAxisTransferTests(unittest.TestCase):
    def test_projection_does_not_fit_transverse_component(self):
        self.assertEqual(axis_target([10., 0.], [0., 0.], [12., 3.]), (2., .2))
        np.testing.assert_allclose(correct([10., 0.], [0., 0.], 2., "pixels"), [12., 0.])
        np.testing.assert_allclose(correct([10., 0.], [0., 0.], .2, "fraction"), [12., 0.])

    def test_degenerate_axis_is_explicit_and_unchanged(self):
        self.assertIsNone(axis_target([0., 0.], [0., 0.], [12., 3.]))
        np.testing.assert_array_equal(correct([0., 0.], [0., 0.], 20., "pixels"), [0., 0.])


if __name__ == "__main__":
    unittest.main()
