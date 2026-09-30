"""Tests for differentiable SMPL surface contact primitives."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import torch

from pose_app import smpl_surface_contact as sc


class SurfaceContactTests(unittest.TestCase):
    def test_nonpenetration_only_penalizes_below_margin(self):
        z = torch.tensor([[-0.010, 0.002, 0.020]], requires_grad=True)
        loss = sc.nonpenetration_loss(z, margin_m=0.003)
        self.assertAlmostEqual(float(loss), ((0.013 ** 2) + (0.001 ** 2)) / 3.0, places=8)

    def test_tangential_velocity_removes_normal_component(self):
        now = torch.tensor([[[0.1, 0.0, 0.2], [0.0, 0.1, 0.2]]])
        prev = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        tangent = sc.tangential_velocity(now, prev, 0.1, torch.tensor([0.0, 0.0, 1.0]))
        self.assertTrue(torch.allclose(tangent[..., 2], torch.zeros(1, 2)))
        self.assertAlmostEqual(float(sc.tangential_velocity_loss(now, prev, 0.1, [0, 0, 1])), 1.0, places=6)

    def test_ground_distances_shape(self):
        v = torch.randn(4, 10, 3)
        self.assertEqual(sc.signed_ground_distances(v).shape, (4, 10))

    def test_point_segment_and_capsule(self):
        p = torch.tensor([[[0.0, 0.0, 0.05]]])
        a = torch.tensor([[[0.0, 0.0, 0.0]]])
        b = torch.tensor([[[1.0, 0.0, 0.0]]])
        d = sc.point_segment_distance(p, a, b)
        self.assertAlmostEqual(float(d), 0.05, places=5)
        r = sc.capsule_surface_residual(p, a, b, 0.02)
        self.assertAlmostEqual(float(r), 0.03, places=5)
        with self.assertRaises(ValueError):
            sc.capsule_surface_residual(p, a, b, 0.0)

    def test_robust_scalar_keeps_k(self):
        x = torch.tensor([[0.0, 0.05, -0.03]])
        out = sc.robust_scalar(x, 0.025)
        self.assertEqual(out.shape, (1, 3))
        self.assertTrue(bool((out >= 0).all()))

    def test_softmin_below_min(self):
        vals = torch.tensor([[0.05, 0.02, 0.08]])
        sm = sc.softmin(vals, 0.005)
        self.assertEqual(sm.shape, (1,))
        self.assertLess(float(sm), 0.02)

    def test_foot_loss_prefers_lowest_vertex(self):
        sole = torch.tensor([[[0.10, 0.005, 0.12]]], requires_grad=True)
        loss = sc.foot_surface_loss(sole).sum()
        loss.backward()
        self.assertTrue(np.isfinite(float(loss.detach())))
        self.assertTrue(np.isfinite(sole.grad.detach().numpy()).all())

    def test_hand_loss_differentiable(self):
        p = torch.tensor([[[0.0, 0.0, 0.03]]], requires_grad=True)
        a = torch.zeros(1, 1, 3)
        b = torch.ones(1, 1, 3)
        loss = sc.hand_surface_loss(p, a, b, 0.02).sum()
        loss.backward()
        self.assertTrue(np.isfinite(float(loss.detach())))


if __name__ == "__main__":
    unittest.main()
