import unittest

import numpy as np
from experiment_forearm_residual_parts import decompose


class ResidualPartsTests(unittest.TestCase):
    def test_translation_only(self):
        p = np.array([[-1., 0., 0.], [1., 0., 0.]])
        v = decompose(p + [0, 2, 0], p)
        self.assertEqual(v["center_mm2"], 8)
        self.assertEqual(v["length_mm2"], 0)
        self.assertEqual(v["direction_mm2"], 0)

    def test_length_and_rotation_separate(self):
        p = np.array([[-1., 0., 0.], [1., 0., 0.]])
        v = decompose(p * 2, p)
        self.assertEqual(v["length_mm2"], 2)
        self.assertEqual(v["direction_mm2"], 0)
        r = np.array([[0., -1., 0.], [0., 1., 0.]])
        self.assertEqual(decompose(r, p)["direction_mm2"], 4)

    def test_random_and_zero_length_identity(self):
        rng = np.random.default_rng(7)
        a, b = rng.normal(size=(2, 100, 2, 3))
        a[0] = 0
        v = decompose(a, b)
        np.testing.assert_allclose(v["center_mm2"] + v["length_mm2"] + v["direction_mm2"], v["total_mm2"], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
