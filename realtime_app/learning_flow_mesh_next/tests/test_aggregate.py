from dataclasses import replace

import cv2
import numpy as np
import pytest
import torch

from realtime_app.learning_flow_mesh.rotations import matrix_to_six

from ..aggregate import decide, diagnostic_joint_controls
from ..candidate_store import from_parameters
from ..geometry_observation import Camera, NoiseModel, observation_errors, project
from ..schema import Topology
from ..tangent import TangentScale
from .test_firewall import observation


def store_fixture():
    rotations = matrix_to_six(torch.eye(3, dtype=torch.float64)).repeat(5, 52)
    translation = torch.tensor([[0., 0., 3.], [.1, 0., 3.], [.2, 0., 3.], [2., 0., 3.], [4., 0., 3.]], dtype=torch.float64)
    beta = torch.arange(5, dtype=torch.float64)[:, None].expand(-1, 10)*.01
    return from_parameters(torch.cat((rotations, translation, beta), -1), Topology(),
                           ids=[f"id{i}" for i in range(5)], seeds=list(range(5)),
                           current_observed=True, provenance={}, resources={})


def rig():
    K = torch.tensor([[300., 0., 320.], [0., 290., 240.], [0., 0., 1.]], dtype=torch.float64)
    D = torch.tensor([.03, -.01, .001, 0.], dtype=torch.float64)
    return tuple(Camera(f"PnP_{role}_fixture", role, K, D, torch.eye(3, dtype=torch.float64),
                        torch.tensor([-i*.2, 0., 0.], dtype=torch.float64), (640, 480))
                 for i, role in enumerate(("LEFT", "RIGHT")))


def test_project_opencv_and_optical_axis_jacobian():
    points = torch.tensor([[0., 0., 3.], [.5, -.2, 2.], [-.3, .2, 1.]], dtype=torch.float64, requires_grad=True)
    uv, valid = project(points, rig()[0])
    expected, _ = cv2.fisheye.projectPoints(points.detach().numpy()[:, None], np.zeros(3), np.zeros(3),
                                          rig()[0].K.numpy(), rig()[0].D.numpy())
    assert np.max(np.abs(uv.detach().numpy()-expected[:, 0])) < 1e-8
    assert valid.all()
    uv.sum().backward()
    assert torch.isfinite(points.grad).all()
    assert not project(torch.tensor([[0., 0., -1.]], dtype=torch.float64), rig()[0])[1].any()


def test_whole_candidate_permutation_and_raw_expectation():
    store = store_fixture()
    scale = TangentScale.fixture(store.topology)
    first = decide(store, "A4", scale)
    order = torch.tensor([4, 1, 3, 0, 2])
    assert decide(store.permuted(order), "A4", scale).selected_id == first.selected_id
    assert first.human(store)["beta"].equal(store.beta[first.selected_index])
    store.actions[3:] = False
    restricted = decide(store, "A4", scale)
    assert torch.allclose(restricted.costs, first.costs)
    store.actions[:] = False
    assert decide(store, "A4", scale).selected_id is None


def test_reprojection_own_translation_and_missing_nan():
    obs = replace(observation(), xy=torch.tensor([320., 240.], dtype=torch.float64).expand(2, 17, 2))
    points = torch.tensor([[[0., 0., 3.]]*17, [[.5, 0., 3.]]*17], dtype=torch.float64)
    errors, per_joint, _, _ = observation_errors(points, obs, rig(), NoiseModel.proxy())
    assert errors[0] < errors[1]
    missing = replace(obs, present=torch.zeros_like(obs.present), xy=torch.full_like(obs.xy, torch.nan))
    assert torch.isinf(observation_errors(points, missing, rig(), NoiseModel.proxy())[0]).all()
    results = diagnostic_joint_controls(points, per_joint, torch.ones(2, dtype=torch.bool), ids=["a", "b"])
    assert results["jpma_coco17_absolute"].equal(points[0])
    assert "no_mesh" in results["scope"]


def test_masks_failures_never_resampled():
    store = store_fixture()
    raw = store.raw.clone()
    raw[1, :6] = 0
    raw[2, -1] = torch.nan
    failed = from_parameters(raw, store.topology, ids=store.ids, seeds=store.seeds,
                             current_observed=True, provenance={}, resources={})
    assert len(failed.raw) == 5 and not failed.numerical[1:3].any()
    assert failed.reasons[1] == ["invalid_rotation"]
    assert failed.reasons[2] == ["nonfinite_joint_state"]
    with pytest.raises(ValueError):
        failed.select(1)
