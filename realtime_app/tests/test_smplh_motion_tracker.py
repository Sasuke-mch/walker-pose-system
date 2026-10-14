"""Model-parameter motion tests; real licensed-model parity is checked in replay."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from pose_app.fisheye_camera import StereoFisheyeCalibration
from pose_app.smplh_motion_tracker import MotionPose, SMPLHMotionConfig, SMPLHMotionTracker, retract


class AnalyticBodyMap:
    """A small rigid body with an articulated arm, for known derivative truth."""
    dimension = 7

    def __init__(self):
        self.initial_pose = MotionPose(np.array([0., 0., 2.]), np.eye(3), np.zeros(1))
        self.base = np.array([[.15*np.cos(i), .2*np.sin(i), .03*(i % 3)] for i in range(17)])

    def evaluate_batch(self, translations, roots, latents, mesh=False):
        points = np.repeat(self.base[None], len(translations), axis=0)
        for i in range(len(translations)):
            points[i, 9] = points[i, 7] + Rotation.from_rotvec([0, 0, latents[i, 0]]).apply([.12, 0, 0])
        points = np.einsum("bij,bnj->bni", roots, points) + translations[:, None]
        data = {"coco_m": points, "joints_m": points.copy(),
                "body_rotations": np.repeat(np.eye(3)[None, None], len(translations), axis=0)}
        if mesh:
            data["vertices_m"] = points.copy()
        return data

    def evaluate(self, pose, mesh=False):
        return self.evaluate_batch(pose.translation_m[None], pose.root_rotation[None], pose.latent[None], mesh)


def tracker(config=None):
    K = np.array([[300., 0., 320.], [0., 300., 240.], [0., 0., 1.]])
    cal = StereoFisheyeCalibration(K, np.zeros(4), K, np.zeros(4), np.eye(3), np.array([-200., 0., 0.]), 640, 480)
    return SMPLHMotionTracker(AnalyticBodyMap(), cal, config)


def observation(track, pose=None):
    data = track.model.evaluate(pose or track.pose)
    pixels = track._pixels(data["coco_m"])[0]
    return np.concatenate((pixels, np.ones((2, 12, 1))), axis=2)


class SMPLHMotionTests(unittest.TestCase):
    def test_first_frame_keeps_model_and_does_not_reuse_pixels(self):
        tr = tracker()
        px = observation(tr)
        result = tr.step(0., px[0], px[1])
        self.assertEqual(result["status"], "initialized")
        self.assertEqual(result["joint_status"].tolist(), ["initialized"] * 12)
        self.assertFalse(result["used_views"].any())
        self.assertFalse(result["derivative_valid"].any())
        np.testing.assert_allclose(result["position_mm"], tr.model.evaluate(tr.pose)["coco_m"][0, 5:17]*1000)

    def test_known_root_acceleration_with_irregular_times(self):
        tr = tracker(SMPLHMotionConfig(pixel_sigma_px=.2))
        px = observation(tr)
        tr.step(0., px[0], px[1])
        now = 0.
        acceleration = np.array([.2, -.1, .04])
        origin = tr.pose.translation_m.copy()
        for i in range(80):
            now += (.025, .035, .045)[i % 3]
            true = MotionPose(origin + .5*acceleration*now**2, np.eye(3), np.zeros(1))
            px = observation(tr, true)
            result = tr.step(now, px[0], px[1])
        self.assertEqual(result["status"], "updated")
        np.testing.assert_allclose(result["translation_m"], true.translation_m, atol=.01)
        np.testing.assert_allclose(result["generalized_velocity"][:3], acceleration*now, atol=.04)
        np.testing.assert_allclose(result["generalized_acceleration"][:3], acceleration, atol=.08)
        self.assertGreater(np.linalg.eigvalsh(tr.covariance).min(), -1e-8)

    def test_derivatives_include_rotation_curvature(self):
        tr = tracker(SMPLHMotionConfig(max_position_std_m=100.))
        tr.velocity[5] = 1.
        tr.last_joint_support[:] = .2
        tr._linearize()
        result = tr._output(.2, "updated", np.ones((2, 12), bool), np.full((2, 12), "used_2d"))
        point = tr.model.base[5]
        expected_v = np.array([-point[1], point[0], 0.])
        expected_a = np.array([-point[0], -point[1], 0.])
        np.testing.assert_allclose(result["velocity_mm_s"][0]/1000, expected_v, atol=1e-5)
        np.testing.assert_allclose(result["acceleration_mm_s2"][0]/1000, expected_a, atol=1e-5)

    def test_short_missing_prediction_expires_and_restarts_from_2d(self):
        tr = tracker()
        px = observation(tr)
        tr.step(0., px[0], px[1])
        tr.velocity[:3] = [.5, 0, 0]
        missing = np.full((12, 3), np.nan)
        predicted = tr.step(.05, missing, missing)
        self.assertEqual(predicted["status"], "predicted")
        self.assertTrue(predicted["pose_available"])
        expired = tr.step(.2, missing, missing)
        self.assertEqual(expired["status"], "expired")
        self.assertFalse(expired["pose_available"])
        self.assertTrue(np.isnan(expired["vertices_m"]).all())
        restored = tr.step(.3, px[0], px[1])
        self.assertEqual(restored["status"], "reinitialized")
        self.assertEqual(restored["joint_status"].tolist(), ["reinitialized"] * 12)
        self.assertEqual(restored["segment_id"], 1)
        np.testing.assert_array_equal(restored["generalized_velocity"], np.zeros(7))
        self.assertFalse(restored["derivative_valid"].any())

    def test_non_observed_joint_derivatives_are_unavailable(self):
        tr = tracker(SMPLHMotionConfig(max_position_std_m=100.))
        px = observation(tr)
        tr.step(0., px[0], px[1])
        for i in range(1, 8):
            masked = px.copy()
            masked[:, 4] = np.nan
            result = tr.step(i*.03, masked[0], masked[1])
        self.assertEqual(result["joint_status"][4], "inferred")
        self.assertFalse(result["derivative_valid"][4])
        self.assertTrue(np.isnan(result["acceleration_mm_s2"][4]).all())
        self.assertTrue(np.isfinite(result["position_mm"][4]).all())

    def test_correction_is_bounded_and_actual_2d_cost_improves(self):
        tr = tracker()
        px = observation(tr)
        tr.step(0., px[0], px[1])
        true = tr.pose.copy()
        true.translation_m += [.3, .1, .0]
        px = observation(tr, true)
        before = tr.pose.copy()
        prior_error = np.linalg.norm(observation(tr)[..., :2]-px[..., :2])
        result = tr.step(.03, px[0], px[1])
        self.assertEqual(result["status"], "updated")
        self.assertLessEqual(np.linalg.norm(tr.pose.translation_m-before.translation_m), tr.config.max_translation_correction_m+1e-9)
        self.assertLess(np.linalg.norm(result["projected_pixels"]-px[..., :2]), prior_error)

    def test_bad_stereo_pair_is_rejected_in_2d_without_disabling_other_points(self):
        tr = tracker()
        px = observation(tr)
        tr.step(0., px[0], px[1])
        px[1, 0, 1] += 100.
        result = tr.step(.03, px[0], px[1])
        self.assertEqual(result["status"], "updated")
        self.assertFalse(result["used_views"][:, 0].any())
        self.assertEqual(result["view_reason"][:, 0].tolist(), ["stereo_epipolar_rejected"]*2)
        self.assertTrue(result["used_views"][:, 1:].all())

    def test_tiny_clipped_correction_does_not_claim_full_measurement_information(self):
        tr = tracker(SMPLHMotionConfig(max_translation_correction_m=1e-5,
                                      max_root_correction_rad=1e-5, max_latent_correction_norm=1e-5))
        px = observation(tr)
        tr.step(0., px[0], px[1])
        true = tr.pose.copy()
        true.translation_m += [.3, .1, 0.]
        px = observation(tr, true)
        variance_before = tr.covariance[0, 0]
        result = tr.step(.03, px[0], px[1])
        self.assertEqual(result["status"], "updated")
        self.assertLess(result["update_scale"], .001)
        self.assertGreater(tr.covariance[0, 0], .95*variance_before)
        self.assertGreater(np.linalg.eigvalsh(tr.covariance).min(), -1e-8)

    def test_poor_pixel_fit_keeps_raw_derivative_and_reports_unavailable(self):
        tr = tracker(SMPLHMotionConfig(max_position_std_m=100.))
        tr.velocity[0] = .2
        tr.last_joint_support[:] = .2
        tr._linearize()
        px = observation(tr)
        px[1, 0, 0] += 50.
        result = tr._output(.2, "updated", np.ones((2, 12), bool),
                            np.full((2, 12), "used_2d"), px)
        self.assertFalse(result["derivative_valid"][0])
        self.assertEqual(result["derivative_reason"][0], "poor_or_stale_pixel_fit")
        self.assertTrue(np.isfinite(result["raw_velocity_mm_s"][0]).all())
        self.assertTrue(np.isnan(result["velocity_mm_s"][0]).all())
        self.assertTrue(result["derivative_valid"][1:].all())
        tr.last_joint_good_fit[:] = .2
        rejected = tr._output(.2, "predicted", np.zeros((2, 12), bool),
                              np.full((2, 12), "nonlinear_update_rejected"), px)
        self.assertFalse(rejected["derivative_valid"][0])
        self.assertEqual(rejected["derivative_reason"][0], "poor_or_stale_pixel_fit")

    def test_so3_retraction_preserves_rotation_near_pi(self):
        pose = MotionPose(np.zeros(3), Rotation.from_rotvec([3.13, 0., 0.]).as_matrix(), np.zeros(1))
        for _ in range(100):
            pose = retract(pose, np.array([0., 0., 0., .03, .01, -.02, 0.]))
        np.testing.assert_allclose(pose.root_rotation.T @ pose.root_rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(pose.root_rotation), 1., places=12)

    def test_bounds_time_and_covariance_availability(self):
        tr = tracker(SMPLHMotionConfig(max_position_std_m=1e-8))
        px = observation(tr)
        tr.step(0., px[0], px[1])
        px[:, 0, 0] = 700.
        result = tr.step(.2, px[0], px[1])
        self.assertTrue(np.all(result["view_reason"][:, 0] == "out_of_raw_image_bounds"))
        self.assertFalse(result["derivative_valid"].any())
        with self.assertRaises(ValueError):
            tr.step(.2, px[0], px[1])


if __name__ == "__main__":
    unittest.main()
