import unittest

import numpy as np
from experiment_body_translation_audit import optimal_translation, weighted_rms


class BodyTranslationTests(unittest.TestCase):
    def test_known_optimum_and_invalid_points(self):
        r = np.array([[[2., 0, 0], [4., 0, 0], [np.nan, np.nan, np.nan]]])
        w = np.array([[1., 3., 0.]])
        delta, valid = optimal_translation(r, w)
        np.testing.assert_allclose(delta, [[-3.5, 0, 0]])
        self.assertTrue(valid[0])
        self.assertLess(weighted_rms(r + delta[:, None], w), weighted_rms(r, w))

    def test_no_support_kept_unavailable(self):
        delta, valid = optimal_translation(np.full((1, 3, 3), np.nan), np.zeros((1, 3)))
        np.testing.assert_array_equal(delta, [[0, 0, 0]])
        self.assertFalse(valid[0])


if __name__ == "__main__":
    unittest.main()
