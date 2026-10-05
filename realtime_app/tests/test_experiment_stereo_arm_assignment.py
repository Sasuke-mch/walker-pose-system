import unittest

import numpy as np
from experiment_stereo_arm_assignment import prefer_crossed, score, swap_arm


class StereoArmAssignmentTests(unittest.TestCase):
    def test_swap_involution_and_preserve_other_joints(self):
        points = np.arange(2 * 17 * 3).reshape(2, 17, 3)
        changed = swap_arm(points)
        np.testing.assert_array_equal(swap_arm(changed), points)
        np.testing.assert_array_equal(changed[:, :5], points[:, :5])
        np.testing.assert_array_equal(changed[:, 11:], points[:, 11:])
        np.testing.assert_array_equal(changed[:, 5], points[:, 6])

    def test_conservative_rule(self):
        self.assertTrue(prefer_crossed({"pass": 2, "mean_px": 20}, {"pass": 4, "mean_px": 10}))
        self.assertFalse(prefer_crossed({"pass": 3, "mean_px": 20}, {"pass": 4, "mean_px": 10}))
        self.assertFalse(prefer_crossed({"pass": 2, "mean_px": None}, {"pass": 4, "mean_px": 10}))
        self.assertFalse(prefer_crossed({"pass": 2, "mean_px": 20}, {"pass": 4, "mean_px": 15}))

    def test_missing_slots_never_look_like_zero_error(self):
        self.assertEqual(score([{}] * 6), {"pass": 0, "mean_px": None})


if __name__ == "__main__":
    unittest.main()
