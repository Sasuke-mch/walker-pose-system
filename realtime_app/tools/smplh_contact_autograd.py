"""Autograd Jacobians for the unchanged direction/contact MAP objective.

The model remains float32, exactly as in its NumPy observation map. Input
retraction and projection use float64 to match SciPy/NumPy around that model;
this neither converts the learned model to double nor changes the cost path.
"""
from __future__ import annotations

import numpy as np
import torch
from pose_app.fisheye_camera import fisheye_project_torch
from pose_app.foot_ground_reference import SHOE_HEIGHT_M
from probe_smplh_foot_observations import TOE_VERTICES
from probe_smplh_soft_contact import BALL, SURFACE, ContactTracker


def so3_exp_torch(vector):
    """Right-retraction exponential, with finite first derivatives at zero."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(*vector.shape[:-1], 3, 3)
    squared = (vector*vector).sum(dim=-1)
    safe_squared = squared.clamp_min(1e-8)
    angle = safe_squared.sqrt()
    small = squared < 1e-8
    a = torch.where(small, 1-squared/6+squared*squared/120, angle.sin()/angle)
    b = torch.where(small, .5-squared/24+squared*squared/720, (1-angle.cos())/safe_squared)
    return torch.eye(3, dtype=vector.dtype, device=vector.device) + a[..., None, None]*skew + b[..., None, None]*(skew@skew)


class AutogradContactTracker(ContactTracker):
    """Change objective derivatives only; retain the original NumPy costs/P path."""

    def _torch_objective_values(self, prior, delta, selected):
        model = self.model
        dtype, device = delta.dtype, delta.device

        def tensor(value):
            return torch.as_tensor(value, dtype=dtype, device=device)

        # NumPy first retracts float64 values, then casts the model inputs to32.
        translation = (tensor(prior.translation_m)+delta[:3]).float()[None]
        root = (tensor(prior.root_rotation)@so3_exp_torch(delta[3:6])).float()[None]
        latent = (tensor(prior.latent)+delta[6:]).float()[None]
        body = model.vposer.decode(latent)['pose_body_matrot'].reshape(1, 21, 3, 3).clone()
        body[:, model.fixed_body_indices] = model.fixed_body[list(model.fixed_body_indices)]
        result = model.model(betas=model.betas, transl=translation, global_orient=root,
            body_pose=body, left_hand_pose=model.hand_rotations[0], right_hand_pose=model.hand_rotations[1])
        vertices = result.vertices
        coco = torch.einsum('jv,bvc->bjc', model.regressor, vertices).to(dtype)
        if self.solve_with_feet:
            coco = torch.cat((coco, vertices[:, TOE_VERTICES].to(dtype)), dim=1)
        points = coco[:, 5:21 if self.solve_with_feet else 17]
        right = points@tensor(self.cal.R_cam0_to_cam1).T+tensor(self.cal.T_cam0_to_cam1_mm)/1000
        projected = torch.stack((fisheye_project_torch(points, self.cal.K0, self.cal.D0),
                                 fisheye_project_torch(right, self.cal.K1, self.cal.D1)), dim=1)
        if self.solve_with_feet and self.observation_mode == 'direction':
            toe = projected[:, :, 12:].reshape(1, 2, 2, 2, 2)
            vector = toe[..., 1, :]-toe[..., 0, :]
            measured = np.asarray(self.extra_pixels)[..., :2].reshape(2, 2, 2, 2)
            observed_length = np.linalg.norm(measured[..., 1, :]-measured[..., 0, :], axis=-1)
            observed_length = np.where(np.isfinite(observed_length), observed_length, 0.)
            direction = vector/torch.linalg.vector_norm(vector, dim=-1, keepdim=True)*tensor(observed_length)[None, ..., None]
            encoded = torch.stack((direction, torch.zeros_like(direction)), dim=-2).reshape(1, 2, 4, 2)
            projected = torch.cat((projected[:, :, :12], encoded), dim=2)
        pixel_values = projected.reshape(-1)[torch.as_tensor(selected, device=device)]
        if self.contact_rotation is None:
            return pixel_values
        surface = vertices[:, SURFACE].to(dtype)
        heights = (surface@tensor(self.contact_rotation).T+tensor(self.contact_translation))[..., 2]
        penetration = torch.relu((SHOE_HEIGHT_M-.005)-heights)/(.005*np.sqrt(len(SURFACE)))
        support_height = torch.stack([heights[:, ids].min(dim=1).values for ids in BALL], dim=1)
        u = (support_height-SHOE_HEIGHT_M)/.010
        attachment = u*torch.sqrt(2/(torch.sqrt(1+u*u)+1))*tensor(self.support)
        return torch.cat((pixel_values, penetration.reshape(-1), attachment.reshape(-1)))

    def _objective_jacobians(self, prior, delta, selected, engineering_count):
        expected = len(SURFACE)+2 if self.contact_rotation is not None else 0
        if engineering_count != expected:
            raise ValueError('Autograd contact residual count differs from NumPy objective')
        values = torch.as_tensor(delta, dtype=torch.float64, device=self.model.device).requires_grad_(True)
        function = lambda value: self._torch_objective_values(prior, value, selected)
        jacobian = torch.autograd.functional.jacobian(function, values, vectorize=True).detach().cpu().numpy()
        pixel_count = int(np.count_nonzero(selected))
        if jacobian.shape != (pixel_count+engineering_count, self.n):
            raise ValueError('Autograd objective Jacobian has incompatible dimensions')
        return jacobian[:pixel_count], jacobian[pixel_count:]
