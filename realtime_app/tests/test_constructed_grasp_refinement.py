"""State isolation, real FK and surface-gradient checks for opt-in refinement."""
import numpy as np
import pytest
import torch

from pose_app.constructed_grasp_refinement import (
    UPPER_JOINTS, compose_body, wrist_rotations, foot_terms,
    frozen_surface_states, validate_rotation, load_grasp, triangle_separation_loss,
    align_wrist_rotations, screened_triangle_pairs)


def test_upper_updates_leave_lower_rotations_exactly_unchanged():
    base = torch.eye(3).expand(4, 21, 3, 3).clone()
    delta = torch.randn(4, len(UPPER_JOINTS), 3, requires_grad=True)
    body = compose_body(base, delta, UPPER_JOINTS)
    lower = [j - 1 for j in range(1, 22) if j not in UPPER_JOINTS]
    assert torch.equal(body[:, lower], base[:, lower])
    body[:, 19].sum().backward()
    assert delta.grad.norm() > 0


def test_wrist_fk_includes_local_wrist_rotation_and_gradients():
    base = torch.eye(3).expand(1, 21, 3, 3).clone()
    delta = torch.zeros(1, 1, 3, requires_grad=True)
    delta.data[..., 2] = .5
    body = compose_body(base, delta, (20,))
    parents = [-1] + [0] * 21
    r = wrist_rotations(torch.eye(3)[None], body, parents)
    assert torch.allclose(r[:, 0], body[:, 19])
    assert torch.allclose(r[:, 1], torch.eye(3)[None])
    r[:, 0, 0, 1].sum().backward()
    assert delta.grad.norm() > 0


def test_foot_c2_requires_both_support_frames_and_frame_validity():
    vertices = torch.zeros(3, 4, 3)
    vertices[1:, :, 0] = torch.tensor([.01, .02])[:, None]
    vertices[..., 2] = -.01
    vertices.requires_grad_()
    weights = torch.ones(3, 2)
    support = torch.tensor([[True, False], [True, True], [True, True]])
    terms, pairs = foot_terms(vertices, [np.array([0, 1]), np.array([2, 3])],
                              weights, support, torch.tensor([True, True, False]))
    assert pairs == [1, 0]
    assert terms["foot_nonpenetration"] > 0
    assert terms["foot_tangential"] > 0
    sum(terms.values()).backward()
    assert vertices.grad[..., 2].abs().sum() > 0
    assert vertices.grad[..., 0].abs().sum() > 0


def test_unknown_support_retains_nonpenetration_without_fabricated_c2():
    vertices = torch.zeros(4, 2, 3, requires_grad=True)
    terms, pairs = foot_terms(vertices, [np.array([0]), np.array([1])],
        torch.zeros(4, 2), torch.zeros(4, 2, dtype=torch.bool), torch.ones(4, dtype=torch.bool))
    assert pairs == [0, 0]
    assert terms["foot_tangential"] == 0
    assert terms["foot_nonpenetration"] > 0


def test_penetrating_baseline_cannot_become_support():
    vertices = np.zeros((8, 2, 3))
    vertices[..., 2] = -.02
    support, _, reasons = frozen_surface_states(vertices, [np.array([0]), np.array([1])], np.ones(8, bool))
    assert not support.any()
    assert (reasons == "baseline_penetration_exceeds_3mm").all()


def test_static_near_ground_surface_enters_support_with_hysteresis():
    vertices = np.zeros((8, 2, 3))
    vertices[..., 2] = .001
    support, _, _ = frozen_surface_states(vertices, [np.array([0]), np.array([1])], np.ones(8, bool))
    assert not support[:3].any()
    assert support[-1].all()


def test_reflection_is_not_a_valid_rotation():
    with pytest.raises(ValueError):
        validate_rotation(np.diag([-1., 1., 1.]))


def test_loader_rejects_observed_or_world_stationary_target(tmp_path):
    import json
    path = tmp_path / "grasp.json"
    path.write_text(json.dumps({"schema": "constructed_bilateral_grasp_v1",
        "coordinate_frame": "world", "observed_pose": True}))
    with pytest.raises(ValueError, match="contract"):
        load_grasp(path)


def test_invalid_fk_parent_and_correction_shape_rejected():
    base = torch.eye(3).expand(1, 21, 3, 3).clone()
    with pytest.raises(ValueError, match="shape"):
        compose_body(base, torch.zeros(2, 1, 3), (20,))
    with pytest.raises(ValueError, match="parent"):
        wrist_rotations(torch.eye(3)[None], base, [-1, 1] + [0] * 20)


def test_screened_triangle_plane_separation_has_surface_gradient():
    v = torch.tensor([[[-.01, -.01, 0], [.01, -.01, 0], [0, .01, 0],
                       [0, -.01, -.01], [0, .01, -.01], [0, 0, .01]]], requires_grad=True)
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
    value = triangle_separation_loss(v, faces, torch.tensor([[0, 1]]))
    assert value > 0
    value.backward()
    assert torch.isfinite(v.grad).all() and v.grad.abs().sum() > 0


def test_empty_triangle_candidates_remain_differentiable_zero():
    v = torch.randn(2, 6, 3, requires_grad=True)
    value = triangle_separation_loss(v, torch.tensor([[0, 1, 2], [3, 4, 5]]), torch.empty(0, 2, dtype=torch.long))
    assert value == 0
    value.backward()
    assert torch.equal(v.grad, torch.zeros_like(v))


def test_frame_specific_collision_cannot_push_unrelated_frames():
    v = torch.tensor([[[-.01, -.01, 0], [.01, -.01, 0], [0, .01, 0],
                       [0, -.01, -.01], [0, .01, -.01], [0, 0, .01]]]).expand(2, -1, -1).clone().requires_grad_()
    value = triangle_separation_loss(v, torch.tensor([[0, 1, 2], [3, 4, 5]]), torch.tensor([[0, 0, 1]]))
    value.backward()
    assert v.grad[0].abs().sum() > 0
    assert torch.equal(v.grad[1], torch.zeros_like(v.grad[1]))


def test_locked_palm_rotation_compensates_moving_parent_with_gradient():
    base = torch.eye(3).expand(2, 21, 3, 3).clone()
    delta = torch.tensor([[[.2, -.3, .1]], [[-.1, .4, .2]]], requires_grad=True)
    body = compose_body(base, delta, (18,))
    parents = [-1] + [0] * 21
    parents[20] = 18
    target = torch.eye(3).expand(2, 3, 3).clone()
    aligned = align_wrist_rotations(torch.eye(3).expand(2, 3, 3), body, parents, target)
    actual = wrist_rotations(torch.eye(3).expand(2, 3, 3), aligned, parents)
    assert torch.allclose(actual, target[None].expand(2, -1, -1, -1), atol=1e-6)
    aligned[:, 19, 0, 1].sum().backward()
    assert torch.isfinite(delta.grad).all() and delta.grad.abs().sum() > 0


def test_fast_screen_reports_crossing_and_excludes_shared_vertex():
    vertices = np.array([[-.01, -.01, 0], [.01, -.01, 0], [0, .01, 0],
                         [0, -.01, -.01], [0, .01, -.01], [0, 0, .01]])
    pairs = screened_triangle_pairs(vertices, np.array([[0, 1, 2], [3, 4, 5]]))
    assert np.array_equal(pairs, [[0, 1]])
    assert len(screened_triangle_pairs(vertices, np.array([[0, 1, 2], [0, 4, 5]]))) == 0
