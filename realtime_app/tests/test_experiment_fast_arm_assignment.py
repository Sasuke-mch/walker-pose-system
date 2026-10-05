import unittest

import numpy as np
from experiment_fast_arm_assignment import diagnostic
from pose_app.fisheye_camera import StereoFisheyeCalibration, fisheye_project_numpy


class FastAssignmentTests(unittest.TestCase):
    def setUp(self):
        k = np.array([[500., 0., 960.], [0., 500., 540.], [0., 0., 1.]])
        self.cal = StereoFisheyeCalibration(k, np.zeros(4), k.copy(), np.zeros(4), np.eye(3),
                                           np.array([-120., 0., 0.]), 1920, 1080)

    def test_exact_stereo_and_negative_depth(self):
        x = np.array([[0., 50., 800.], [50., 30., 900.]])
        left = fisheye_project_numpy(x, self.cal.K0, self.cal.D0)[None]
        right = fisheye_project_numpy(x + self.cal.T_cam0_to_cam1_mm, self.cal.K1, self.cal.D1)[None]
        error, valid = diagnostic(left, right, self.cal)
        np.testing.assert_allclose(error, 0., atol=1e-8)
        self.assertTrue(valid.all())
        _, wrong = diagnostic(right, left, self.cal)
        self.assertFalse(wrong.any())

    def test_raw_bounds_retained_as_failure(self):
        left = np.array([[[960., 540.]]])
        right = np.array([[[-1., 540.]]])
        _, valid = diagnostic(left, right, self.cal)
        self.assertFalse(valid.any())

    def test_batched_and_streamed_match(self):
        left = np.array([[[950., 540.]], [[940., 530.]]])
        right = left - [70., 2.]
        a, mask = diagnostic(left, right, self.cal)
        for i in range(2):
            b, m = diagnostic(left[i:i + 1], right[i:i + 1], self.cal)
            np.testing.assert_allclose(a[i], b[0])
            np.testing.assert_array_equal(mask[i], m[0])


if __name__ == "__main__":
    unittest.main()
