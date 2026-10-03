import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from pose_app.wrist_processing import (CausalPointFilter, epipolar_select,
    perspective_view, raw_to_upright, upright_to_raw, upper_box, refine_forearm,
    project_raw, PerspectiveView)


class WristProcessingTest(unittest.TestCase):
    def test_camera_rotation_roundtrip(self):
        p = np.array([[0., 0.], [1919, 1079], [500.2, 900.7]])
        for camera in ('left', 'right'):
            np.testing.assert_allclose(upright_to_raw(raw_to_upright(p, camera), camera), p, atol=1e-12)

    def test_perspective_mapping_matches_opencv_remap(self):
        K = np.array([[600., 0, 960], [0, 600, 540], [0, 0, 1]])
        D = np.zeros(4)
        for camera in ('left', 'right'):
            v = perspective_view([250, 450, 850, 1400], K, D, camera)
            points = np.array([[100., 100.], [650., 800.], [384., 512.]])
            raw = v.to_raw(points)
            np.testing.assert_allclose(v.from_raw(raw), points, atol=1e-8)
            mx, my = cv2.fisheye.initUndistortRectifyMap(K, D, v.rotation.T, v.intrinsic, v.size, cv2.CV_32FC1)
            np.testing.assert_allclose(raw, np.column_stack((mx[points[:, 1].astype(int), points[:, 0].astype(int)], my[points[:, 1].astype(int), points[:, 0].astype(int)])), atol=1e-4)

    def test_roi_requires_confident_complete_arm(self):
        p = np.ones((17, 3))
        p[:, :2] = [500, 600]
        self.assertIsNotNone(upper_box(p))
        p[7, 2] = .1
        self.assertIsNone(upper_box(p))

    def test_perspective_rejects_behind_camera(self):
        K = np.array([[600., 0, 960], [0, 600, 540], [0, 0, 1]])
        v = PerspectiveView(np.diag([1., -1., -1.]), K, K, np.zeros(4), 'left')
        with self.assertRaises(ValueError):
            v.from_raw([[960., 540.]])

    def test_missing_filter_resets_without_fill(self):
        f = CausalPointFilter()
        f.update([10., 20.], 0)
        p, state = f.update([np.nan, np.nan], .03)
        self.assertEqual(state, 'missing')
        self.assertTrue(np.isnan(p).all())
        p, state = f.update([90., 80.], .06)
        np.testing.assert_array_equal(p, [90, 80])
        self.assertEqual(state, 'initialized')

    def test_filter_causal_and_attenuates_static_noise(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(90, 2))*2
        f = CausalPointFilter()
        y = np.array([f.update(p, i/30)[0] for i, p in enumerate(x)])
        self.assertLess(np.std(y[10:]), np.std(x[10:]))
        g = CausalPointFilter()
        prefix = np.array([g.update(p, i/30)[0] for i, p in enumerate(x[:30])])
        np.testing.assert_array_equal(prefix, y[:30])

    def test_epipolar_selection_cannot_leave_candidates(self):
        K = np.array([[600., 0, 960], [0, 600, 540], [0, 0, 1]])
        cal = SimpleNamespace(K0=K, K1=K, D0=np.zeros(4), D1=np.zeros(4), T_cam0_to_cam1_mm=np.array([-100., 0, 0]), R_cam0_to_cam1=np.eye(3))
        a = np.array([[960., 540.], [960., 542.]])
        b = np.array([[900., 545.], [900., 540.]])
        l, r, audit = epipolar_select((a, np.ones(2)), (b, np.ones(2)), cal)
        self.assertTrue(any(np.array_equal(l, p) for p in a))
        self.assertTrue(any(np.array_equal(r, p) for p in b))
        self.assertLess(audit['sampson_px'], .1)

    def test_forearm_refinement_respects_image_movement_limit(self):
        K = np.array([[600., 0, 960], [0, 600, 540], [0, 0, 1]])
        cal = SimpleNamespace(K0=K, K1=K, D0=np.zeros(4), D1=np.zeros(4), T_cam0_to_cam1_mm=np.array([-100., 0, 0]), R_cam0_to_cam1=np.eye(3))
        x = np.array([[0., 0., 1.], [.25, 0., 1.]])
        obs = np.stack((project_raw(x, K, cal.D0), project_raw(x+[-.1, 0, 0], K, cal.D1)))
        p, audit = refine_forearm(x, obs, .24, cal)
        self.assertTrue(audit['accepted_update'])
        self.assertLess(np.max(np.linalg.norm(p-obs, axis=-1)), 12)
        self.assertLess(abs(audit['final_length_m']-.24), .01)


if __name__ == '__main__':
    unittest.main()
