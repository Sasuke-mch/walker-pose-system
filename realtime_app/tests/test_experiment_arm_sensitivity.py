import unittest

import numpy as np
from experiment_arm_sensitivity import length_sigma, projections, sensitivity
from pose_app.fisheye_camera import StereoFisheyeCalibration
from walker_tools.hands.solve_annotated_wrist_targets import solve_point


class SensitivityTests(unittest.TestCase):
    def test_covariance_positive_and_step_stable(self):
        k = np.array([[500., 0., 960.], [0., 500., 540.], [0., 0., 1.]])
        cal = StereoFisheyeCalibration(k, np.zeros(4), k.copy(), np.zeros(4), np.eye(3),
                                       np.array([-120., 0., 0.]), 1920, 1080)
        a, _ = sensitivity([0., 0., 800.], cal, .01)
        b, _ = sensitivity([0., 0., 800.], cal, .001)
        self.assertTrue(np.all(np.linalg.eigvalsh(a) > 0))
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8)
        self.assertIsNone(sensitivity([0., 0., -800.], cal))

    def test_length_propagates_along_bone_only(self):
        c = np.diag([4., 9., 16.])
        self.assertAlmostEqual(length_sigma([0., 0., 0.], [10., 0., 0.], c, c), np.sqrt(8))
        self.assertIsNone(length_sigma([0., 0., 0.], [0., 0., 0.], c, c))

    def test_small_noise_monte_carlo_matches_linear_scale(self):
        k = np.array([[500., 0., 960.], [0., 500., 540.], [0., 0., 1.]])
        cal = StereoFisheyeCalibration(k, np.zeros(4), k.copy(), np.zeros(4), np.eye(3),
                                       np.array([-120., 0., 0.]), 1920, 1080)
        point = np.array([20., 30., 800.])
        covariance, _ = sensitivity(point, cal)
        pixels = projections(point, cal)
        rng = np.random.default_rng(20261005)
        errors = []
        for noise in rng.normal(0., .1, (300, 4)):
            observed = pixels + noise
            result = solve_point(observed[:2], observed[2:], cal)
            errors.append(np.asarray(result["wrist_left_camera_m"]) * 1000 - point)
        observed_variance = np.mean(np.sum(np.square(errors), axis=1))
        ratio = observed_variance / (np.trace(covariance) * .1 ** 2)
        self.assertGreater(ratio, .75)
        self.assertLess(ratio, 1.25)


if __name__ == "__main__":
    unittest.main()
