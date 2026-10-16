"""Zero-angle SO3 and raw-fisheye derivative contracts for contact Jacobians."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from pose_app.fisheye_camera import fisheye_project_numpy, fisheye_project_torch
from smplh_contact_autograd import so3_exp_torch


class ContactAutogradTests(unittest.TestCase):
    def test_so3_matches_scipy_and_has_zero_angle_derivatives(self):
        values = torch.tensor([[0., 0., 0.], [1e-7, -2e-8, 4e-8], [.13, -.21, .31]], dtype=torch.float64)
        np.testing.assert_allclose(so3_exp_torch(values).numpy(), Rotation.from_rotvec(values.numpy()).as_matrix(), atol=2e-15)
        jacobian = torch.autograd.functional.jacobian(so3_exp_torch, values[0])
        self.assertTrue(torch.isfinite(jacobian).all())
        np.testing.assert_allclose(jacobian[:, :, 0], [[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]], atol=0)

    def test_fisheye_values_and_axis_gradient(self):
        matrix = np.array([[510., 0., 950.], [0., 505., 540.], [0., 0., 1.]])
        distortion = np.array([.03, -.02, .001, -.0001])
        points = torch.tensor([[0., 0., 2.], [.6, -.5, 2.]], dtype=torch.float64, requires_grad=True)
        actual = fisheye_project_torch(points, matrix, distortion)
        np.testing.assert_allclose(actual.detach().numpy(), fisheye_project_numpy(points.detach().numpy(), matrix, distortion), atol=1e-12)
        jacobian = torch.autograd.functional.jacobian(lambda value: fisheye_project_torch(value, matrix, distortion), points)
        self.assertTrue(torch.isfinite(jacobian).all())
        np.testing.assert_allclose(jacobian[0, :, 0], [[255., 0., 0.], [0., 252.5, 0.]], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
