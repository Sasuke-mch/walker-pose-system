from dataclasses import replace

import torch

from ..geometry_information import local_information, schur_shape, whitened_shape_audit
from ..geometry_observation import NoiseModel, observation_errors, project
from ..tangent import TangentScale
from .test_aggregate import rig, store_fixture
from .test_firewall import observation


def test_exact_observation_cost_has_zero_finite_derivative():
    cameras = rig()
    points = torch.tensor([.1, -.05, 3.], dtype=torch.float64).expand(1, 17, 3).clone().requires_grad_()
    xy = torch.stack([project(points.detach(), camera)[0][0] for camera in cameras])
    obs = replace(observation(), xy=xy)
    costs = observation_errors(points, obs, cameras, NoiseModel.proxy())[0]
    derivative = torch.autograd.grad(costs.sum(), points)[0]
    assert torch.equal(costs, torch.zeros_like(costs))
    assert torch.isfinite(derivative).all()
    assert torch.equal(derivative, torch.zeros_like(derivative))


def test_two_stage_frozen_g0_and_state_chunk_roundtrip():
    import json

    from realtime_app.learning_flow_mesh.rotations import matrix_to_six

    from ..data_firewall import Normalization
    from ..joint_flow import JointFlow
    from ..pipeline import TwoStageInference, TwoStageState

    torch.set_num_threads(1)
    torch.manual_seed(7)
    g0, g1 = JointFlow().double().eval(), JointFlow().double().eval()
    mean = torch.cat((matrix_to_six(torch.eye(3, dtype=torch.float64)).repeat(52),
                      torch.tensor([0., 0., 3.], dtype=torch.float64), torch.zeros(10, dtype=torch.float64)))
    normal = Normalization(mean, torch.ones(325, dtype=torch.float64)*.03, g0.topology, "analytic_fixture_only")
    pipeline = TwoStageInference(g0, g1, normal, TangentScale.fixture(g0.topology), rigid_surface, rig(), NoiseModel.proxy())
    before = {k: v.clone() for k, v in g0.state_dict().items()}
    with torch.no_grad():
        _, _, info, state, report = pipeline(replace(observation(), xy=observation().xy.double()), k=2, anchor_k=2, steps=1)
        restored = TwoStageState.from_dict(json.loads(json.dumps(state.to_dict())), dtype=torch.float64)
        obs = replace(observation(1, .1), xy=observation().xy.double())
        a = pipeline(obs, state, k=2, anchor_k=2, steps=1)[0]
        b = pipeline(obs, restored, k=2, anchor_k=2, steps=1)[0]
    assert torch.equal(a.raw, b.raw) and info.matrix is not None
    assert report["generators"] == 2 and report["network_calls"] == 2
    assert report["candidate_state_evaluations"] == 4
    assert not any(p.requires_grad for p in g0.parameters())
    assert all(torch.equal(v, before[k]) for k, v in g0.state_dict().items())


def rigid_surface(rotations, translation, beta):
    points = translation[:, None].expand(-1, 17, -1)
    return {"coco17": points}


def test_jacobian_against_independent_autograd_and_scaled_coordinates():
    store = store_fixture()
    store.actions[3:] = False  # Anchor still uses all numerical candidates.
    cameras = rig()
    scale = TangentScale.fixture(store.topology)
    values = scale.values.clone()
    values[156:159] = torch.tensor([.2, .4, .3])
    scale = TangentScale(values, "analytic_fixture_only")
    info = local_information(store, observation(), cameras, NoiseModel.proxy(), scale, rigid_surface)
    assert info.matrix is not None and not info.matrix.requires_grad
    anchor = store.ids.index(info.anchor_id)
    position = store.translation[anchor]
    def image(delta):
        points = (position+delta*values[156:159])[None].expand(17, 3)
        return torch.stack([project(points, camera)[0] for camera in cameras]).flatten()
    jac = torch.autograd.functional.jacobian(image, torch.zeros(3, dtype=torch.float64))
    assert torch.allclose(info.jacobian[:, 156:159], jac, atol=1e-5)
    assert torch.count_nonzero(info.jacobian[:, :156]) == 0
    assert torch.count_nonzero(info.jacobian[:, -10:]) == 0
    assert info.resources["decoder_state_evaluations"] == 339


def test_shape_pose_confounding_and_pinv_sensitivity():
    J = torch.zeros(2, 13, dtype=torch.float64)
    J[0, 0] = J[0, 3] = 1
    J[1, 1] = J[1, 4] = 1
    schur, report = schur_shape(J.T@J)
    assert schur.abs().max() < 1e-12
    assert report["nuisance_rank"] == 2


def test_weak_nuisance_false_shape_signal_and_svd_cutoff_match():
    # Shape is exactly explained by a weak nuisance direction. A Gram cutoff
    # of 1e-6 discards it; the same cutoff applied to J retains it.
    J = torch.zeros(3, 13, dtype=torch.float64)
    J[0, 0], J[1, 1], J[1, 3] = 1., 1e-4, 1.
    gram_schur, old = schur_shape(J.T@J, rtol=1e-6)
    residual, audit = whitened_shape_audit(J, rtol=1e-6)
    matched, _ = schur_shape(J.T@J, rtol=1e-12)
    assert old["nuisance_rank"] == 1 and gram_schur[0, 0] == 1
    assert audit["nuisance_rank"] == 2
    assert audit["equivalent_gram_rtol"] == 1e-12
    assert residual.abs().max() < 1e-14 and torch.allclose(residual, matched, atol=1e-14)
    assert audit["residual_shape_energy_fraction"] == 0
    truncated, _ = whitened_shape_audit(J, rtol=1e-3)
    assert truncated[0, 0] == 1  # SVD also needs threshold sensitivity.


def test_svd_shape_audit_independent_direction_and_coordinate_invariance():
    J = torch.zeros(3, 13, dtype=torch.float64)
    J[0, 0], J[1, 1], J[0, 3], J[2, 4] = 1., 2., 3., 4.
    expected, report = whitened_shape_audit(J)
    assert expected[1, 1] == 16 and expected.count_nonzero() == 1
    assert abs(report["residual_shape_energy_fraction"] - 16/25) < 1e-14
    Q = torch.tensor([[0., 1., 0.], [0., 0., 1.], [-1., 0., 0.]], dtype=torch.float64)
    permuted = J[:, [2, 0, 1, *range(3, 13)]]
    rotated, _ = whitened_shape_audit(Q @ permuted)
    assert torch.allclose(expected, rotated, atol=1e-14)
    empty, report = whitened_shape_audit(torch.zeros_like(J))
    assert empty.count_nonzero() == 0 and report["residual_shape_energy_fraction"] is None


def test_no_measurement_and_bad_projection_information_failure():
    store = store_fixture()
    missing = replace(observation(), present=torch.zeros(2, 17, dtype=torch.bool))
    info = local_information(store, missing, rig(), NoiseModel.proxy(), TangentScale.fixture(store.topology), rigid_surface)
    assert info.matrix is None and info.reasons == ["no_current_observation"]
    store.translation[:, 2] = -1
    info = local_information(store, observation(), rig(), NoiseModel.proxy(), TangentScale.fixture(store.topology), rigid_surface)
    assert info.matrix is None and "projection_invalid" in info.reasons[0]
