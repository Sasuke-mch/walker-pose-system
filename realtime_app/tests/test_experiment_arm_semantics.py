import unittest

import numpy as np
from experiment_arm_semantics import features, flagged


class ArmSemanticTests(unittest.TestCase):
    def setUp(self):
        self.points = np.zeros((17, 3))
        self.points[[7, 8, 9, 10], :2] = [[0, 0], [100, 0], [0, 100], [100, 100]]

    def test_scale_translation_invariant(self):
        a = features(self.points)
        b = self.points.copy()
        b[:, :2] = b[:, :2] * 2 + 300
        self.assertAlmostEqual(a["wrist_separation_ratio"], features(b)["wrist_separation_ratio"])
        self.assertIsNone(a["jump_ratio"])

    def test_collapse_and_jump_are_flags_only(self):
        b = self.points.copy()
        b[10] = b[9]
        a = features(b, self.points)
        self.assertTrue(flagged(a, .2, 2.))
        self.assertEqual(a["jump_ratio"], 1.)
        np.testing.assert_array_equal(b[9], b[10])

    def test_invalid_points_explicit(self):
        self.points[9, 0] = np.nan
        a = features(self.points)
        self.assertEqual(a["state"], "invalid_pixels")
        self.assertFalse(flagged(a, .2, 1.))


if __name__ == "__main__":
    unittest.main()
