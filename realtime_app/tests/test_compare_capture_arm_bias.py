import unittest

import numpy as np
from compare_capture_arm_bias import signed_components, stats


class ArmBiasMetricsTests(unittest.TestCase):
    def test_empty_metrics_remain_unavailable(self):
        self.assertEqual(stats([float("nan"), float("inf")])["n"], 0)
        self.assertIsNone(stats([])["median"])

    def test_toward_elbow_sign_and_transverse_component(self):
        result = signed_components([8, 3], [10, 0], [0, 0])
        self.assertEqual(result["along_px"], -2)
        self.assertEqual(result["absolute_across_px"], 3)
        self.assertIsNone(signed_components([8, 3], [0, 0], [0, 0]))

    def test_image_rigid_change_preserves_signed_arm_components(self):
        p, m, e = np.array([8., 3.]), np.array([10., 0.]), np.array([0., 0.])
        R = np.array([[0., -1.], [1., 0.]])
        t = np.array([500., 300.])
        before = signed_components(p, m, e)
        after = signed_components(R @ p + t, R @ m + t, R @ e + t)
        for key in before:
            self.assertAlmostEqual(before[key], after[key])


if __name__ == "__main__":
    unittest.main()
