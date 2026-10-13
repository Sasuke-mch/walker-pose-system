from dataclasses import replace

import pytest
import torch

from ..geometry_information import (
    endpoint_cost_vjp,
    local_information,
    schur_shape,
    whitened_shape_audit,
)
from ..geometry_observation import NoiseModel, observation_errors, project
from ..input_binding import observation_snapshot
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


@pytest.mark.parametrize("missing_value", [torch.nan, torch.inf])
def test_unobserved_surface_values_do_not_enter_projection_backward(missing_value):
    cameras = rig()
    present = observation().present.clone()
    present[:, 0], present[1, 1] = False, False
    xy = torch.tensor([320., 240.], dtype=torch.float64).expand(2, 17, 2).clone()
    xy[~present] = torch.nan
    obs = replace(observation(), xy=xy, present=present)
    reference = torch.tensor([.1, -.05, 3.], dtype=torch.float64).expand(1, 17, 3).clone().requires_grad_()
    expected = observation_errors(reference, obs, cameras, NoiseModel.proxy())[0]
    expected_gradient = torch.autograd.grad(expected.sum(), reference)[0]
    points = reference.detach().clone()
    points[:, 0] = missing_value
    points.requires_grad_()
    costs, per_joint, uv, valid = observation_errors(points, obs, cameras, NoiseModel.proxy())
    gradient = torch.autograd.grad(costs.sum(), points)[0]
    assert torch.isfinite(gradient).all()
    assert torch.equal(costs, expected) and torch.equal(gradient, expected_gradient)
    assert gradient[:, 0].count_nonzero() == 0 and torch.isinf(per_joint[:, 0]).all()
    assert torch.isnan(uv[:, ~obs.admitted()]).all() and not valid[:, ~obs.admitted()].any()
    # The same corrupt value remains a failure if its observation is admitted.
    admitted = replace(obs, xy=torch.nan_to_num(xy), present=torch.ones_like(present))
    failed_cost, _, _, failed_valid = observation_errors(points, admitted, cameras, NoiseModel.proxy())
    assert torch.isinf(failed_cost).all() and not failed_valid[:, :, 0].any()


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
        first_obs = replace(observation(), xy=observation().xy.double())
        first, _, info, state, report = pipeline(first_obs, k=2, anchor_k=2, steps=1)
        studied, _, full_info, full_state, full_report = pipeline(first_obs, k=2, anchor_k=2, steps=1, diagnostics=True)
        restored = TwoStageState.from_dict(json.loads(json.dumps(state.to_dict())), dtype=torch.float64)
        obs = replace(observation(1, .1), xy=observation().xy.double())
        a = pipeline(obs, state, k=2, anchor_k=2, steps=1)[0]
        b = pipeline(obs, restored, k=2, anchor_k=2, steps=1)[0]
    assert torch.equal(a.raw, b.raw) and info.matrix is not None
    assert torch.equal(first.raw, studied.raw) and first.reasons == studied.reasons
    assert torch.equal(info.matrix, full_info.matrix) and state.to_dict() == full_state.to_dict()
    assert report["information_diagnostics"] == "not_requested"
    assert full_report["information_diagnostics"] == "computed"
    assert report["generators"] == 2 and report["network_calls"] == 2
    assert report["candidate_state_evaluations"] == 4
    assert not any(p.requires_grad for p in g0.parameters())
    assert all(torch.equal(v, before[k]) for k, v in g0.state_dict().items())


def rigid_surface(rotations, translation, beta):
    points = translation[:, None].expand(-1, 17, -1)
    return {"coco17": points}


@pytest.mark.parametrize("device,dtype,step", [("cpu", torch.float64, 1e-4),
                                              ("cuda", torch.float32, 1e-3),
                                              ("cuda", torch.float32, 1e-4)])
def test_observation_only_licensed_surface_values_gradients_and_information(device, dtype, step):
    """Cropping LBS must retain the surface operator, not use skeletal joints."""
    import json
    from pathlib import Path

    from realtime_app.learning_flow_mesh.body import BodyDecoder
    from realtime_app.learning_flow_mesh.rotations import matrix_to_six
    from realtime_app.pose_app.smpl_coco_observation import load_coco17_regressor

    from ..body import SurfaceDecoder
    from ..candidate_store import from_parameters
    from ..schema import Topology
    from ..tangent import skew

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = json.loads(Path("realtime_app/learning_flow_mesh/config.json").read_text(encoding="utf-8"))
    if not all(Path(path).is_file() for path in config["assets"].values()):
        pytest.skip("licensed SMPL-H/MANO assets unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(83)
    decoder = BodyDecoder(**config["assets"]).to(device=device, dtype=dtype)
    regressor = torch.tensor(load_coco17_regressor("models/smpl/J_regressor_coco.npy"), device=device, dtype=dtype)
    surface = SurfaceDecoder(decoder, regressor)
    rotation = torch.matrix_exp(skew(torch.randn(3, 52, 3, device=device, dtype=dtype)*.2)).requires_grad_()
    translation = torch.tensor([[0., 0., 3.], [.1, -.1, 3.2], [-.1, .1, 2.8]], device=device, dtype=dtype).requires_grad_()
    beta = (torch.randn(3, 10, device=device, dtype=dtype)*.2).requires_grad_()
    full = surface(rotation, translation, beta)["coco17"]
    fast = surface.coco17(rotation, translation, beta)
    atol = 1e-10 if dtype == torch.float64 else 2e-6
    torch.testing.assert_close(fast, full, atol=atol, rtol=atol)
    weight = torch.randn_like(full)
    expected = torch.autograd.grad((full*weight).sum(), (rotation, translation, beta))
    actual = torch.autograd.grad((fast*weight).sum(), (rotation, translation, beta))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=atol*10, rtol=atol*10)
    assert len(surface.observation_vertices) == int(regressor.ne(0).any(0).sum())
    cameras = tuple(replace(c, **{name: getattr(c, name).to(device=device, dtype=dtype)
                                  for name in ("K", "D", "R", "T")}) for c in rig())
    fixture = observation()
    obs = replace(fixture, **{name: getattr(fixture, name).to(device) for name in
        ("confidence", "present", "quality", "geometry", "views", "origins", "directions")},
        xy=torch.stack([project(full.detach()[0], camera)[0] for camera in cameras]))
    raw = torch.cat((matrix_to_six(rotation.detach()).flatten(1), translation.detach(), beta.detach()), -1)
    store = from_parameters(raw, Topology(), ids=["a", "b", "c"], seeds=[0, 1, 2],
                            current_observed=True, provenance={}, resources={})
    scale = TangentScale.fixture(store.topology)
    args = (store, obs, cameras, NoiseModel.proxy(), scale)
    # A callable without coco17 is the original full-mesh/chunk16 reference.
    reference = local_information(*args, lambda r, t, b: surface(r, t, b), step=step, diagnostics=False)
    result = local_information(*args, surface, step=step, diagnostics=False)
    assert result.matrix is not None and reference.matrix is not None
    relative_j = float((result.jacobian-reference.jacobian).norm()/reference.jacobian.norm())
    relative_matrix = float((result.matrix-reference.matrix).norm()/reference.matrix.norm())
    tolerance = 1e-8 if dtype == torch.float64 else (2e-3 if step == 1e-3 else 2e-2)
    assert relative_j < tolerance and relative_matrix < tolerance
    assert result.anchor_id == reference.anchor_id and result.reasons == reference.reasons == []
    assert result.resources["decoder_state_evaluations"] == reference.resources["decoder_state_evaluations"] == 339
    assert result.resources["decoder_batch_calls"] == 2 and reference.resources["decoder_batch_calls"] == 23
    print(f"{device}/{dtype}/step={step}: surface_max={float((fast-full).detach().abs().max()):.3g}, "
          f"J_relative={relative_j:.3g}, Lambda_relative={relative_matrix:.3g}")
    bad_translation = translation.detach().clone()
    bad_translation[0, 0] = torch.nan
    for decode in (surface, surface.coco17):
        with pytest.raises(ValueError, match="nonfinite shape/translation"):
            decode(rotation.detach(), bad_translation, beta.detach())
    # Both reject a valid current anchor whose original -Tz probe goes behind
    # the cameras. No removed +/- probe may turn this into usable information.
    extreme = replace(scale, values=scale.values.clone())
    extreme.values[158] = 1e5
    old_failure = local_information(store, obs, cameras, NoiseModel.proxy(), extreme,
                                    lambda r, t, b: surface(r, t, b), step=step, diagnostics=False)
    new_failure = local_information(store, obs, cameras, NoiseModel.proxy(), extreme,
                                    surface, step=step, diagnostics=False)
    assert old_failure.matrix is None and new_failure.matrix is None
    assert old_failure.reasons == new_failure.reasons
    if device == "cpu":
        state = surface.state_dict()
        state["regressor"] = state["regressor"].roll(7, 1)
        state["decoder.layer.v_template"] = state["decoder.layer.v_template"] + .02
        surface.load_state_dict(state)
        surface.float().double()
        with torch.no_grad():
            rebuilt = surface.coco17(rotation, translation, beta)
            updated_full = surface(rotation, translation, beta)["coco17"]
        torch.testing.assert_close(rebuilt, updated_full, atol=1e-10, rtol=1e-10)


def endpoint_fixture():
    from ..candidate_store import from_parameters

    original = store_fixture()
    raw = original.raw.clone()
    raw[:, 312] = torch.arange(5, dtype=raw.dtype)*.01
    store = from_parameters(raw, original.topology, ids=original.ids, seeds=original.seeds,
                            current_observed=True, provenance={}, resources={})
    points = torch.tensor([.017, -.015, 3.05], dtype=torch.float64).expand(17, 3)
    xy = torch.stack([project(points, camera)[0] for camera in rig()])
    obs = replace(observation(), xy=xy)
    store.provenance["observation_binding"] = observation_snapshot(obs)
    return store, obs


def analytic_pixel_jacobian(point, camera):
    """Closed equidistant derivative; no AD, FD or helper implementation."""
    xyz = camera.R @ point + camera.T
    x, y, z = xyz
    radius = (x*x+y*y+1e-24).sqrt()
    theta = torch.atan2(radius, z)
    dr = torch.stack((x/radius, y/radius, z*0))
    dt = (z*dr-torch.tensor([0., 0., 1.], dtype=point.dtype)*radius)/(radius*radius+z*z)
    f = theta + sum(d*theta**(2*i+3) for i, d in enumerate(camera.D))
    fp = 1 + sum((2*i+3)*d*theta**(2*i+2) for i, d in enumerate(camera.D))
    da = fp/radius*dt-f/radius**2*dr
    planar = torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=point.dtype)
    return camera.K[:2, :2] @ (f/radius*planar+xyz[:2, None]*da) @ camera.R


def test_endpoint_analytic_scaled_masked_direction_and_batch_independence():
    from ..candidate_store import from_parameters

    store, obs = endpoint_fixture()
    present, quality = obs.present.clone(), obs.quality.clone()
    present[0, ::2], quality[1, 1::3] = False, False
    xy = obs.xy.clone()
    xy[~present] = torch.nan
    obs = replace(obs, xy=xy, present=present, quality=quality)
    store.provenance["observation_binding"] = observation_snapshot(obs)
    precision = torch.tensor([[.02, .004], [.004, .03]], dtype=torch.float64).expand(2, 17, 2, 2).clone()
    noise = NoiseModel(precision, "analytic_fixture_anisotropic")
    values = torch.full((169,), .1, dtype=torch.float64)
    values[156:159] = torch.tensor([.2, .4, .3])
    scale = TangentScale(values, "analytic_fixture_nonuniform")
    result = endpoint_cost_vjp(store, obs, rig(), noise, scale, rigid_surface)
    assert result.available.all() and not result.gradient.requires_grad
    for index, point in enumerate(store.translation):
        expected = torch.zeros(3, dtype=torch.float64)
        for view, camera in enumerate(rig()):
            mask = obs.admitted()[view]
            residual = project(point.expand(17, 3), camera)[0][mask]-obs.xy[view, mask]
            assert (torch.einsum("ji,jil,jl->j", residual, precision[view, mask], residual) < 9).all()
            expected += analytic_pixel_jacobian(point, camera).T @ (precision[view, mask] @ residual[..., None]).sum(0)[:, 0]
        expected *= values[156:159]/obs.admitted().sum()
        assert torch.allclose(result.gradient[index, 156:159], expected, atol=1e-12, rtol=1e-10)
        assert result.gradient[index, :156].count_nonzero() == 0
        assert result.gradient[index, -10:].count_nonzero() == 0
    single = from_parameters(store.raw[2:3], store.topology, ids=[store.ids[2]], seeds=[2],
                             current_observed=True, provenance=store.provenance, resources={})
    alone = endpoint_cost_vjp(single, obs, rig(), noise, scale, rigid_surface)
    assert torch.equal(result.gradient[2], alone.gradient[0])  # No division by K.
    assert result.resources["decoder_state_evaluations"] == 5 and result.resources["reverse_calls"] == 1


@pytest.mark.parametrize("mode", ["no_grad", "inference_mode"])
def test_endpoint_zero_gradient_available_and_frozen_inputs(mode):
    store, obs = endpoint_fixture()
    with getattr(torch, mode)():
        store = replace(store, raw=store.raw.clone(), rotations=store.rotations.clone(),
                        translation=store.translation.clone(), beta=store.beta.clone())
        store.translation[:] = store.translation[0].clone()
        points = rigid_surface(store.rotations, store.translation, store.beta)["coco17"][0]
        obs = replace(obs, xy=torch.stack([project(points, camera)[0] for camera in rig()]))
        store.provenance["observation_binding"] = observation_snapshot(obs)
        result = endpoint_cost_vjp(store, obs, rig(), NoiseModel.proxy(),
                                   TangentScale.fixture(store.topology), rigid_surface)
    assert result.available.all() and result.gradient.count_nonzero() == 0
    assert result.costs.count_nonzero() == 0 and torch.isfinite(result.gradient).all()
    assert not result.gradient.requires_grad and not result.costs.requires_grad


def test_endpoint_missing_stale_binding_and_no_observations_unavailable():
    store, obs = endpoint_fixture()
    for fixture, bound, reason in (
        (obs, None, "candidate_observation_binding_missing"),
        (replace(obs, frame=1), store.provenance["observation_binding"], "candidate_observation_binding_mismatch"),
        (replace(obs, xy=obs.xy+1), store.provenance["observation_binding"], "candidate_observation_binding_mismatch"),
    ):
        bad = replace(store, provenance={} if bound is None else {"observation_binding": bound})
        result = endpoint_cost_vjp(bad, fixture, rig(), NoiseModel.proxy(),
                                   TangentScale.fixture(store.topology), rigid_surface)
        assert not result.available.any() and result.gradient.isnan().all()
        assert all(reason in row for row in result.reasons) and result.resources["decoder_batch_calls"] == 0
    missing = replace(obs, present=torch.zeros_like(obs.present))
    store.provenance["observation_binding"] = observation_snapshot(missing)
    result = endpoint_cost_vjp(store, missing, rig(), NoiseModel.proxy(),
                               TangentScale.fixture(store.topology), rigid_surface)
    assert not result.available.any() and result.gradient.isnan().all()
    assert all("no_current_observation" in row for row in result.reasons)


def test_endpoint_invalid_slots_retained_and_batch_exception_not_retried():
    store, obs = endpoint_fixture()
    store.translation[1, 2] = -1  # Numerically finite, invalid admitted projection.
    store.beta[2, 0] = torch.nan  # Detect actual tensors even if cached numerical flag is True.
    store.rotations[3, 0] *= 2  # Finite but not a proper rotation.
    store.reasons[2].append("existing_failure")
    result = endpoint_cost_vjp(store, obs, rig(), NoiseModel.proxy(),
                               TangentScale.fixture(store.topology), rigid_surface)
    assert result.ids == store.ids and result.available.tolist() == [True, False, False, False, True]
    assert result.gradient[1:4].isnan().all() and result.resources["decoder_state_evaluations"] == 3
    assert "admitted_projection_invalid" in result.reasons[1]
    assert "existing_failure" in result.reasons[2] and "numerically_invalid_endpoint" in result.reasons[3]
    calls = []
    def broken(rotations, translation, beta):
        calls.append(len(beta))
        raise RuntimeError("test_batch_failure")
    failed = endpoint_cost_vjp(store, obs, rig(), NoiseModel.proxy(),
                               TangentScale.fixture(store.topology), broken)
    assert calls == [3] and not failed.available.any() and failed.gradient.isnan().all()
    assert all("test_batch_failure" in failed.reasons[index][-1] for index in (0, 1, 4))


def test_endpoint_nonfinite_gradient_is_not_usable_zero():
    store, obs = endpoint_fixture()
    def singular(rotations, translation, beta):
        # Forward value is finite; derivative of sqrt(0) is not.
        extra = (beta-beta.detach()).square().sum(-1).sqrt()
        return {"coco17": translation[:, None].expand(-1, 17, -1)+extra[:, None, None]}
    result = endpoint_cost_vjp(store, obs, rig(), NoiseModel.proxy(),
                               TangentScale.fixture(store.topology), singular)
    assert not result.available.any() and result.gradient.isnan().all()
    assert all("nonfinite_endpoint_gradient" in row for row in result.reasons)


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


@pytest.mark.parametrize("diagnostics", [False, True])
def test_no_measurement_and_bad_projection_information_failure(diagnostics):
    store = store_fixture()
    missing = replace(observation(), present=torch.zeros(2, 17, dtype=torch.bool))
    info = local_information(store, missing, rig(), NoiseModel.proxy(), TangentScale.fixture(store.topology),
                             rigid_surface, diagnostics=diagnostics)
    assert info.matrix is None and info.reasons == ["no_current_observation"]
    store.translation[:, 2] = -1
    info = local_information(store, observation(), rig(), NoiseModel.proxy(), TangentScale.fixture(store.topology),
                             rigid_surface, diagnostics=diagnostics)
    assert info.matrix is None and "projection_invalid" in info.reasons[0]


@pytest.mark.parametrize("nodes", [22, 52])
def test_optional_diagnostics_preserve_information_without_redundant_schur(monkeypatch, nodes):
    from .. import geometry_information as geometry
    from ..candidate_store import from_parameters
    from ..schema import Topology

    original = store_fixture()
    store = from_parameters(torch.cat((original.raw[:, :6*nodes], original.raw[:, -13:]), -1),
                            Topology(nodes), ids=original.ids, seeds=original.seeds,
                            current_observed=True, provenance={}, resources={})
    args = (store, observation(), rig(), NoiseModel.proxy(), TangentScale.fixture(store.topology), rigid_surface)
    calls, schur = [], geometry.schur_shape
    def counted(matrix, *, rtol=1e-6):
        calls.append(rtol)
        return schur(matrix, rtol=rtol)
    monkeypatch.setattr(geometry, "schur_shape", counted)
    full = local_information(*args)
    assert full.diagnostics["status"] == "computed"
    assert calls == [1e-6, 1e-4, 1e-5, 1e-8, 1e-10]
    assert full.diagnostics["rank_sensitivity"][2]["pinv_rtol"] == 1e-6
    def forbidden(*args, **kwargs):
        raise AssertionError("minimal acquisition must not run rank/shape diagnostics")
    monkeypatch.setattr(geometry, "information_diagnostics", forbidden)
    minimal = local_information(*args, diagnostics=False)
    assert torch.equal(minimal.matrix, full.matrix) and torch.equal(minimal.jacobian, full.jacobian)
    assert torch.equal(minimal.anchor_rotation, full.anchor_rotation)
    assert minimal.anchor_id == full.anchor_id and minimal.reasons == full.reasons == []
    assert minimal.binding == full.binding and minimal.provenance == full.provenance
    assert minimal.diagnostics["status"] == "not_requested"
    assert minimal.diagnostics["candidate_current_observation_binding"] == full.diagnostics["candidate_current_observation_binding"]
    assert minimal.resources["decoder_state_evaluations"] == full.resources["decoder_state_evaluations"] == 2*(3*nodes+13)+1


def test_diagnostic_failure_does_not_discard_valid_information(monkeypatch):
    from .. import geometry_information as geometry

    store = store_fixture()
    args = (store, observation(), rig(), NoiseModel.proxy(), TangentScale.fixture(store.topology), rigid_surface)
    reference = local_information(*args, diagnostics=False)
    def failed(*args, **kwargs):
        raise RuntimeError("diagnostic decomposition failed")
    monkeypatch.setattr(geometry, "schur_shape", failed)
    result = local_information(*args)
    assert torch.equal(result.matrix, reference.matrix) and torch.equal(result.jacobian, reference.jacobian)
    assert result.reasons == [] and result.diagnostics["status"] == "unavailable"
    assert result.diagnostics["reason"] == "RuntimeError:diagnostic decomposition failed"
