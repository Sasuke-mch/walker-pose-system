from types import SimpleNamespace

import pytest
import torch

from pose_app.smplh_fitting.losses import (
    body_reprojection_loss,
    body_temporal_loss,
    hand_observation_term,
)
from pose_app.smplh_fitting.contact import surface_contact_losses


@pytest.mark.parametrize("stage,weight", [(0, 1.), (3, 0.), (4, 0.)])
def test_unused_hand_projection_nan_cannot_poison_body(stage, weight):
    invalid = torch.tensor(float("nan"), requires_grad=True)
    value = hand_observation_term(weight=weight, stage_index=stage,
                                  left_loss=invalid, right_loss=invalid, body_loss=torch.tensor(1.))
    assert torch.isfinite(value) and value == 0 and not value.requires_grad


def test_active_hand_projection_nan_is_not_silently_sanitized():
    value = hand_observation_term(weight=.1, stage_index=3, left_loss=torch.tensor(float("nan")),
                                  right_loss=torch.tensor(1.), body_loss=torch.tensor(1.))
    assert not torch.isfinite(value)


@pytest.mark.parametrize("stage,weight", [(0, 0), (3, 1), (4, 1)])
def test_inactive_body_terms_need_neither_geometry_nor_finite_diagnostics(stage, weight):
    args = SimpleNamespace(body_reprojection_weight=weight, body_temporal_weight=weight)
    diagnostic = torch.tensor(float("nan"), requires_grad=True)
    values = body_reprojection_loss(
        args=args, stage_index=stage, body_loss=diagnostic,
        proj_cam0=None, proj_cam1=None, raw_left_2d=None, raw_right_2d=None,
        raw_left_conf=None, raw_right_conf=None, raw_left_valid=None, raw_right_valid=None)
    temporal = body_temporal_loss(
        args=args, n=4, stage_index=stage, body_loss=diagnostic, coco=None, mask_body=None)
    assert all(v.item() == 0 and not v.requires_grad for v in (*values, temporal))


def test_disabled_surface_contact_uses_finite_geometry_not_projection():
    vertices = torch.ones((2, 3, 3), requires_grad=True)
    values = surface_contact_losses(left_in_cam0=torch.full((2, 21, 2), float("nan")),
        contact_enabled=False, stage_index=4, vertices=vertices,
        contact=SimpleNamespace(global_hand_offset=None))
    assert all(torch.isfinite(value) and value == 0 for value in values)
    sum(values).backward()
    assert torch.isfinite(vertices.grad).all() and not vertices.grad.any()


def test_disabled_contact_without_vertices_handles_invalid_unused_values():
    points = torch.full((2, 21, 2), float("nan"), requires_grad=True)
    values = surface_contact_losses(left_in_cam0=points, contact_enabled=False,
        stage_index=4, vertices=None, contact=SimpleNamespace(global_hand_offset=None))
    assert all(torch.isfinite(value) and value == 0 for value in values)
    sum(values).backward()
    assert torch.isfinite(points.grad).all() and not points.grad.any()
