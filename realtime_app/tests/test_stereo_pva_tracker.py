"""Physics, pixel feedback and failure boundaries of the causal PVA baseline."""
import unittest

import numpy as np

from pose_app.fisheye_camera import StereoFisheyeCalibration, fisheye_project_numpy
from pose_app.stereo_pva_tracker import (
    StereoPVAConfig, StereoPVATracker, continuous_jerk_covariance,
    constant_acceleration_transition,
)


def calibration() -> StereoFisheyeCalibration:
    K = np.array([[300., 0., 320.], [0., 300., 240.], [0., 0., 1.]])
    angle = 0.07
    rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                         [-np.sin(angle), 0., np.cos(angle)]])
    return StereoFisheyeCalibration(K, np.array([0.01, 0., 0., 0.]), K, np.zeros(4),
                                    rotation, np.array([-120., 10., 5.]), 640, 480)


def observe(tracker, time_s, point, initial=None):
    cal = tracker.calibration
    point = np.asarray(point, dtype=float).reshape(1, 3)
    other = point @ cal.R_cam0_to_cam1.T + cal.T_cam0_to_cam1_mm
    return tracker.step(time_s, fisheye_project_numpy(point, cal.K0, cal.D0),
                        fisheye_project_numpy(other, cal.K1, cal.D1),
                        np.ones(1), np.ones(1), initial)


class StereoPVATests(unittest.TestCase):
    def test_physical_integration_and_noise_composition(self):
        transition = constant_acceleration_transition(0.1)
        np.testing.assert_allclose(transition @ np.arange(9.),
                                   [0.33, 1.435, 2.54, 3.6, 4.7, 5.8, 6., 7., 8.])
        q1, q2 = continuous_jerk_covariance(0.07, 1e4), continuous_jerk_covariance(0.03, 1e4)
        f2 = constant_acceleration_transition(0.03)
        np.testing.assert_allclose(f2 @ q1 @ f2.T + q2, continuous_jerk_covariance(0.1, 1e4), atol=1e-10)
        self.assertGreater(np.linalg.eigvalsh(q1).min(), 0)
        self.assertEqual(q1[0, 1], 0)

    def test_initializer_is_not_updated_again_from_same_pixels(self):
        tracker = StereoPVATracker(calibration(), 1)
        prior = np.array([[150., 30., 1800.]])
        result = observe(tracker, 0., [100., 40., 1800.], prior)
        self.assertEqual(result["status"].tolist(), ["initialized"])
        np.testing.assert_array_equal(result["state"], np.c_[prior, np.zeros((1, 6))])
        np.testing.assert_array_equal(result["covariance"][0], tracker.P0)
        self.assertFalse(result["world_state_available"])

    def test_irregular_time_constant_acceleration_is_recovered_from_pixels(self):
        config = StereoPVAConfig(pixel_sigma_px=0.1, jerk_spectral_density_mm2_s5=10.)
        tracker = StereoPVATracker(calibration(), 1, config)
        origin = np.array([20., 30., 1800.])
        acceleration = np.array([200., -100., 40.])
        observe(tracker, 0., origin, origin[None])
        time_s = 0.
        for i in range(80):
            time_s += (0.025, 0.035, 0.045)[i % 3]
            point = origin + 0.5 * acceleration * time_s**2
            result = observe(tracker, time_s, point)
        self.assertEqual(result["status"].tolist(), ["updated"])
        np.testing.assert_allclose(result["state"][0, :3], point, atol=2.)
        np.testing.assert_allclose(result["state"][0, 3:6], acceleration * time_s, atol=10.)
        np.testing.assert_allclose(result["state"][0, 6:], acceleration, atol=15.)
        self.assertGreater(np.linalg.eigvalsh(result["covariance"][0]).min(), -1e-8)

    def test_missing_data_predicts_then_expires_and_restarts_without_old_velocity(self):
        tracker = StereoPVATracker(calibration(), 1)
        point = np.array([100., 40., 1800.])
        observe(tracker, 0., point, point[None])
        before = observe(tracker, 0.1, point + [20., 0., 0.])
        missing = np.full((1, 2), np.nan)
        predicted = tracker.step(0.2, missing, missing, np.zeros(1), np.zeros(1))
        self.assertEqual(predicted["status"].tolist(), ["predicted"])
        np.testing.assert_allclose(predicted["state"][0], constant_acceleration_transition(0.1) @ before["state"][0])
        self.assertGreater(np.trace(predicted["covariance"][0]), np.trace(before["covariance"][0]))
        expired = tracker.step(0.7, missing, missing, np.zeros(1), np.zeros(1))
        self.assertEqual(expired["status"].tolist(), ["expired"])
        self.assertTrue(np.isnan(expired["state"]).all())
        restored = observe(tracker, 0.8, point)
        self.assertEqual(restored["status"].tolist(), ["reinitialized"])
        self.assertEqual(restored["segment_id"].tolist(), [1])
        np.testing.assert_array_equal(restored["state"][0, 3:], np.zeros(6))

    def test_inconsistent_pair_and_out_of_bounds_do_not_update(self):
        tracker = StereoPVATracker(calibration(), 1)
        point = np.array([100., 40., 1800.])
        before = observe(tracker, 0., point, point[None])
        cal = tracker.calibration
        left = fisheye_project_numpy(point[None], cal.K0, cal.D0)
        right = fisheye_project_numpy((cal.R_cam0_to_cam1 @ point + cal.T_cam0_to_cam1_mm)[None], cal.K1, cal.D1)
        rejected = tracker.step(0.1, left, right + [0., 100.], np.ones(1), np.ones(1))
        self.assertEqual(rejected["status"].tolist(), ["predicted"])
        self.assertEqual(rejected["reason"].tolist(), ["stereo_reprojection_rejected"])
        np.testing.assert_array_equal(rejected["state"], before["state"])
        bounds = tracker.step(0.2, np.array([[640., 200.]]), right, np.ones(1), np.ones(1))
        self.assertEqual(bounds["reason"].tolist(), ["out_of_raw_image_bounds"])

    def test_geometrically_valid_large_innovation_is_rejected(self):
        tracker = StereoPVATracker(calibration(), 1)
        point = np.array([100., 40., 1800.])
        observe(tracker, 0., point, point[None])
        for i in range(1, 31):
            observe(tracker, i * 0.03, point)
        rejected = observe(tracker, 0.93, point + [1200., 0., 0.])
        self.assertEqual(rejected["status"].tolist(), ["predicted"])
        self.assertEqual(rejected["reason"].tolist(), ["innovation_rejected"])
        np.testing.assert_allclose(rejected["state"][0, :3], point, atol=1e-6)

    def test_small_linear_innovation_with_bad_nonlinear_update_is_rejected(self):
        config = StereoPVAConfig(initial_position_sigma_mm=10000.)
        tracker = StereoPVATracker(calibration(), 1, config)
        point = np.array([100., 40., 1800.])
        observe(tracker, 0., point, point[None])
        result = observe(tracker, 0.1, [1500., 600., 3000.])
        self.assertLess(result["innovation_squared_mahalanobis"][0], config.innovation_gate_squared)
        self.assertEqual(result["status"].tolist(), ["predicted"])
        self.assertEqual(result["reason"].tolist(), ["nonlinear_update_rejected"])
        np.testing.assert_array_equal(result["state"][0, :3], point)

    def test_time_and_initialization_contracts(self):
        tracker = StereoPVATracker(calibration(), 1)
        point = np.array([100., 40., 1800.])
        observe(tracker, 0., point, point[None])
        with self.assertRaises(ValueError):
            observe(tracker, 0., point)
        with self.assertRaises(ValueError):
            observe(tracker, 0.1, point, point[None])
        with self.assertRaises(ValueError):
            StereoPVAConfig(jerk_spectral_density_mm2_s5=float("nan"))


if __name__ == "__main__":
    unittest.main()
