import unittest

import numpy as np
from experiment_fast_arm_assignment import normalized_derivatives, normalized_rays
from experiment_stereo_rotation_diagnostic import residual
from pose_app.fisheye_camera import fisheye_project_numpy


class StereoRotationDiagnosticTests(unittest.TestCase):
    def test_exact_geometry_zero_and_perturbation_nonzero(self):
        k = np.array([[500., 0., 960.], [0., 500., 540.], [0., 0., 1.]])
        d, R, t = np.zeros(4), np.eye(3), np.array([-120., 0., 0.])
        x = np.array([[10., 40., 800.], [70., 80., 900.]])
        a, b = fisheye_project_numpy(x, k, d), fisheye_project_numpy(x + t, k, d)
        arguments = (normalized_rays(a, k, d), normalized_rays(b, k, d),
                     normalized_derivatives(a, k, d), normalized_derivatives(b, k, d), R, t)
        np.testing.assert_allclose(residual([0, 0, 0], *arguments), 0, atol=1e-8)
        self.assertGreater(np.abs(residual([.01, 0, 0], *arguments)).mean(), 1)


if __name__ == "__main__":
    unittest.main()
