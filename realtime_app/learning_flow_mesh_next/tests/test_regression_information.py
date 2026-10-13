"""Fair information controls without training or updating any parameters."""
from dataclasses import replace

import pytest
import torch

from ..candidate_store import infer
from ..data_firewall import Normalization
from ..joint_flow import JointFlow, ModelConfig
from ..train import objective
from .test_firewall import labels, observation


@pytest.mark.parametrize("nodes", [22, 52])
def test_default_and_zero_information_preserve_regression(nodes):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = JointFlow(ModelConfig(nodes=nodes)).eval()
    context = torch.randn(2, 128)
    count = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        baseline = model.velocity(torch.zeros(2, model.topology.dimensions),
                                  torch.ones(2), context)
        assert torch.equal(model.regression(context), baseline)
        assert torch.equal(model.regression(context, torch.zeros(
            model.topology.tangent_dimensions, model.topology.tangent_dimensions)), baseline)
    assert count == (711443 if nodes == 52 else 707603)


def test_objective_uses_information_and_detaches_geometry():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = JointFlow().eval()
    context = torch.zeros(2, 128)
    normalization = Normalization.fixture(model.topology)
    reference = labels()
    information = torch.eye(169, requires_grad=True)
    parameters_before = {name: p.detach().clone() for name, p in model.named_parameters()}
    baseline, _ = objective("regression", model, context, reference, normalization)
    conditioned, _ = objective("regression", model, context, reference, normalization,
                               information=information)
    assert torch.isfinite(conditioned) and not torch.equal(baseline, conditioned)
    direct = model.regression(context, information)
    target, _ = reference.packed(require_complete=True)
    error = (direct - normalization.normalize(target)).square()
    expected = (error[:, :312].mean() + error[:, 312:315].mean()
                + error[:, -10:].mean()) / 3
    assert torch.equal(conditioned, expected)
    conditioned.backward()  # No optimizer step: dependency check only.
    assert information.grad is None
    for projection in (model.info_pose, model.info_position, model.info_shape):
        assert projection.weight.grad is not None
        assert projection.weight.grad.abs().sum() > 0
    assert all(torch.equal(p, parameters_before[name]) for name, p in model.named_parameters())


def test_regression_rejects_invalid_information_and_unexpected_options():
    torch.set_num_threads(1)
    model = JointFlow().eval()
    args = ("regression", model, torch.zeros(2, 128), labels(),
            Normalization.fixture(model.topology))
    with pytest.raises(ValueError, match="invalid frozen anchor information"):
        objective(*args, information=torch.eye(168))
    with pytest.raises(ValueError, match="invalid frozen anchor information"):
        objective(*args, information=torch.full((169, 169), torch.nan))
    with pytest.raises(TypeError, match="unexpected keyword"):
        objective(*args, informaton=torch.eye(169))


@pytest.mark.parametrize("nodes", [22, 52])
def test_information_regression_through_candidate_inference(nodes):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = JointFlow(ModelConfig(nodes=nodes)).double().eval()
    obs = observation()
    obs = replace(obs, xy=obs.xy.double(), confidence=obs.confidence.double(),
                  origins=obs.origins.double(), directions=obs.directions.double())
    dimensions = model.topology.dimensions
    normal = Normalization(torch.arange(dimensions, dtype=torch.float64)*.001,
                           torch.linspace(.5, 1.5, dimensions, dtype=torch.float64),
                           model.topology, "analytic_fixture_only")
    information = torch.eye(model.topology.tangent_dimensions, dtype=torch.float64)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        context, direct_state, _ = model.encoder(obs)
        expected = normal.denormalize(model.regression(context[None], information))
        result, state = infer(model, normal, obs, k=1, regression=True,
                              information=information, stream="R_info")
        repeated, _ = infer(model, normal, obs, k=1, regression=True,
                            information=information, stream="R_other_seed")
        baseline, _ = infer(model, normal, obs, k=1, regression=True)
        zero, _ = infer(model, normal, obs, k=1, regression=True,
                        information=torch.zeros_like(information))
        missing = replace(obs, frame=1, timestamp=.1, present=torch.zeros_like(obs.present),
                          xy=torch.full_like(obs.xy, torch.nan))
        unavailable, _ = infer(model, normal, missing, state, k=1, regression=True,
                                information=information)
    assert torch.equal(result.raw, expected) and torch.equal(result.raw, repeated.raw)
    assert result.seeds != repeated.seeds  # These bookkeeping seeds are unused by regression.
    assert torch.equal(baseline.raw, zero.raw) and not torch.equal(baseline.raw, result.raw)
    assert torch.equal(state.hidden, direct_state.hidden)
    assert result.resources["solver"] == "regression"
    assert result.resources["network_calls"] == result.resources["candidate_state_evaluations"] == 1
    assert result.raw.shape == (1, dimensions) and len(result.ids) == len(result.reasons) == 1
    assert not unavailable.actions.any() and "no_current_observation" in unavailable.reasons[0]
    assert sum(p.numel() for p in model.parameters()) == (711443 if nodes == 52 else 707603)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    with pytest.raises(ValueError, match="single deterministic control"):
        infer(model, normal, obs, k=2, regression=True, information=information)
    for invalid in (information[:-1, :-1], torch.full_like(information, torch.nan)):
        with pytest.raises(ValueError, match="invalid frozen anchor information"):
            infer(model, normal, obs, k=1, regression=True, information=invalid)


@pytest.mark.parametrize("nodes", [22, 52])
def test_frozen_information_regression_path_matches_acquisition_and_direct_control(nodes):
    from realtime_app.learning_flow_mesh.rotations import matrix_to_six

    from ..geometry_observation import NoiseModel
    from ..pipeline import TwoStageInference, TwoStageState
    from ..tangent import TangentScale
    from .test_aggregate import rig
    from .test_information import rigid_surface

    torch.set_num_threads(1)
    torch.manual_seed(7)
    g0, g1 = JointFlow(ModelConfig(nodes=nodes)).double().eval(), JointFlow(ModelConfig(nodes=nodes)).double().eval()
    mean = torch.cat((matrix_to_six(torch.eye(3, dtype=torch.float64)).repeat(nodes),
                      torch.tensor([0., 0., 3.], dtype=torch.float64), torch.zeros(10, dtype=torch.float64)))
    normal = Normalization(mean, torch.ones_like(mean)*.03, g0.topology, "analytic_pipeline_not_real_labels",
                           split="engineering_fixture", source_kind="analytic_fixture")
    decoded = []
    def counted_surface(rotations, translation, beta):
        decoded.append(len(rotations))
        return rigid_surface(rotations, translation, beta)
    pipeline = TwoStageInference(g0, g1, normal, TangentScale.fixture(g0.topology),
                                 counted_surface, rig(), NoiseModel.proxy())
    before = {key: value.clone() for key, value in g0.state_dict().items()}
    obs = replace(observation(), xy=observation().xy.double())
    with torch.no_grad():
        result, anchor, info, state, report = pipeline(obs, k=1, anchor_k=2, steps=2, regression=True)
        acquisition_calls = decoded.copy()
        decoded.clear()
        _, flow_anchor, flow_info, flow_state, flow_report = pipeline(obs, k=3, anchor_k=2, steps=2)
        context, _, _ = g1.encoder(obs)
        expected = normal.denormalize(g1.regression(context[None], info.matrix))
        restored = TwoStageState.from_dict(state.to_dict(), dtype=torch.float64)
        next_obs = replace(observation(1, .1), xy=obs.xy.clone())
        next_result, _, _, next_state, _ = pipeline(next_obs, state, k=1, anchor_k=2, steps=2, regression=True)
        replayed = pipeline(next_obs, restored, k=1, anchor_k=2, steps=2, regression=True)[0]
        missing = replace(next_obs, frame=2, timestamp=.2, present=torch.zeros_like(obs.present),
                          xy=torch.full_like(obs.xy, torch.nan))
        unavailable, _, missing_info, missing_state, missing_report = pipeline(
            missing, next_state, k=1, anchor_k=2, steps=2, regression=True)
    assert info.matrix is not None and torch.count_nonzero(info.matrix) > 0
    assert torch.equal(anchor.raw, flow_anchor.raw) and anchor.ids == flow_anchor.ids
    assert torch.equal(info.matrix, flow_info.matrix) and info.anchor_id == flow_info.anchor_id
    assert state.to_dict() == flow_state.to_dict()
    assert torch.equal(result.raw, expected) and torch.equal(next_result.raw, replayed.raw)
    assert result.resources["solver"] == "regression" and result.raw.shape == (1, g0.topology.dimensions)
    assert report["generators"] == report["regressors"] == 1 and report["second_stage_objective"] == "regression"
    assert report["network_calls"] == 3 and report["candidate_state_evaluations"] == 5
    assert report["J_decoder_calls"] == len(acquisition_calls)
    assert report["J_decoder_state_evaluations"] == sum(acquisition_calls)
    assert report["J_decoder_state_evaluations"] == flow_report["J_decoder_state_evaluations"]
    assert report["matched_total_sampling_G0_control_K"] is None
    assert flow_report["generators"] == 2 and flow_report["regressors"] == 0
    assert flow_report["matched_total_sampling_G0_control_K"] == 5
    assert not unavailable.actions.any() and "no_current_observation" in unavailable.reasons[0]
    assert missing_info.matrix is None and missing_report["J_decoder_state_evaluations"] == 0
    assert torch.equal(missing_state.g1.hidden, next_state.g1.hidden)
    assert result.provenance["information"]["calibrated"] is False
    assert not any(p.requires_grad for p in g0.parameters())
    assert all(torch.equal(value, before[key]) for key, value in g0.state_dict().items())


def test_frozen_information_regression_path_rejects_multiple_outputs_before_anchor(monkeypatch):
    from ..geometry_observation import NoiseModel
    from ..pipeline import TwoStageInference
    from ..tangent import TangentScale
    from .test_aggregate import rig
    from .test_information import rigid_surface

    g0, g1 = JointFlow().eval(), JointFlow().eval()
    pipeline = TwoStageInference(g0, g1, Normalization.fixture(g0.topology),
                                 TangentScale.fixture(g0.topology), rigid_surface, rig(), NoiseModel.proxy())
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid regression budget must fail before anchor work")
    monkeypatch.setattr(g0.encoder, "forward", forbidden)
    with pytest.raises(ValueError, match="single deterministic"):
        pipeline(observation(), k=2, regression=True)


def information_sequence_fixture(nodes=52):
    from realtime_app.learning_flow_mesh.rotations import matrix_to_six

    from ..geometry_observation import NoiseModel
    from ..tangent import TangentScale
    from ..train import acquire_sequence_information
    from .test_aggregate import rig
    from .test_information import rigid_surface
    from .test_joint import sequence_fixture

    torch.set_num_threads(1)
    torch.manual_seed(7)
    g0 = JointFlow(ModelConfig(nodes=nodes)).double().eval().requires_grad_(False)
    g1 = JointFlow(ModelConfig(nodes=nodes)).double().eval()
    mean = torch.cat((matrix_to_six(torch.eye(3, dtype=torch.float64)).repeat(nodes),
                      torch.tensor([0., 0., 3.], dtype=torch.float64), torch.zeros(10, dtype=torch.float64)))
    normal = Normalization(mean, torch.ones_like(mean)*.03, g0.topology, "synthetic_G1_sequence_not_training_data",
                           split="engineering_fixture", source_kind="analytic_fixture")
    original = sequence_fixture(nodes)
    chunk = replace(original, observations=tuple(replace(obs, xy=obs.xy.double()) for obs in original.observations),
                    labels=replace(original.labels, rotations=original.labels.rotations.double(),
                                   translation=original.labels.translation.double(), beta=original.labels.beta.double()))
    calls = []
    def surface(rotations, translation, beta):
        calls.append(len(rotations))
        return rigid_surface(rotations, translation, beta)
    before_acquisition = {key: value.clone() for key, value in g0.state_dict().items()}
    bundle = acquire_sequence_information(g0, chunk.observations, normal, TangentScale.fixture(g0.topology),
                                          surface, rig(), NoiseModel.proxy(), anchor_k=2, steps=2)
    assert all(torch.equal(value, before_acquisition[key]) for key, value in g0.state_dict().items())
    return g0, g1, chunk, normal, bundle, calls


@pytest.mark.parametrize("nodes", [22, 52])
@pytest.mark.parametrize("mode", ["CFM", "regression"])
def test_G1_sequence_matches_manual_objective_and_counted_acquisition(nodes, mode):
    from ..train import sequence_objective
    from .test_joint import chunk_slice

    g0, g1, chunk, normal, bundle, decoded = information_sequence_fixture(nodes)
    before = {key: value.clone() for key, value in g0.state_dict().items()}
    generator = torch.Generator().manual_seed(17)
    with torch.no_grad():
        result = sequence_objective(mode, g1, chunk, normal, information=bundle, generator=generator)
        state, generator, expected = None, torch.Generator().manual_seed(17), []
        for index, obs in enumerate(chunk.observations):
            context, state, _ = g1.encoder(obs, state)
            assert torch.equal(context, result.contexts[index])
            if index in (0, 2, 4):
                options = {"generator": generator} if mode == "CFM" else {}
                loss, _ = objective(mode, g1, context[None], chunk_slice(chunk, index, index+1).labels,
                                    normal, information=bundle.frames[index].matrix, **options)
                expected.append(loss)
                assert torch.equal(loss, result.frame_losses[index])
        torch.testing.assert_close(result.loss, torch.stack(expected).mean(), rtol=1e-14, atol=0)
        generator = torch.Generator().manual_seed(17)
        first_info = replace(bundle, frames=bundle.frames[:3], anchor_resources=bundle.anchor_resources[:3])
        last_info = replace(bundle, frames=bundle.frames[3:], anchor_resources=bundle.anchor_resources[3:])
        first = sequence_objective(mode, g1, chunk_slice(chunk, 0, 3), normal,
                                   information=first_info, generator=generator)
        last = sequence_objective(mode, g1, chunk_slice(chunk, 3, 5), normal, first.state,
                                  information=last_info, generator=generator)
    assert torch.equal(result.contexts, torch.cat((first.contexts, last.contexts)))
    assert torch.allclose(result.frame_losses, torch.cat((first.frame_losses, last.frame_losses)), equal_nan=True)
    assert result.state.to_dict() == last.state.to_dict()
    assert result.supervised.tolist() == [True, False, True, False, True]
    assert len(bundle.frames) == 5 and bundle.frames[1].matrix is None
    assert bundle.state.hidden.grad_fn is None and not bundle.state.hidden.requires_grad
    assert "no_current_observation" in result.reasons[1]["information_unavailable"]
    resources = result.resources["information_acquisition"]
    assert resources["G0_network_calls"] == 10 and resources["G0_candidate_state_evaluations"] == 20
    assert resources["J_decoder_batch_calls"] == len(decoded)
    assert resources["J_decoder_state_evaluations"] == sum(decoded) == 4*(1+2*g0.topology.tangent_dimensions)
    assert resources["available_information_frames"] == 4
    assert not resources["computed_in_this_objective"]
    assert all(torch.equal(value, before[key]) for key, value in g0.state_dict().items())


@pytest.mark.parametrize("fault", ["frame", "pixels", "clock", "rig", "noise", "scale", "shape", "nonfinite", "rows", "gradient",
                                  "counts", "reason", "legacy_anchor", "SO3"])
def test_G1_sequence_rejects_bad_information_before_encoding(monkeypatch, fault):
    from ..geometry_observation import NoiseModel
    from ..so3_flow import SO3JointFlow
    from ..train import sequence_objective

    _, g1, chunk, normal, bundle, _ = information_sequence_fixture()
    frames = list(bundle.frames)
    info = frames[-1]
    if fault == "frame":
        frames[-1] = replace(info, provenance={**info.provenance, "frame": 105})
    elif fault in ("pixels", "clock"):
        obs = chunk.observations[-1]
        altered = replace(obs, xy=obs.xy+1) if fault == "pixels" else replace(obs, timestamp_kind="other_clock")
        chunk = replace(chunk, observations=(*chunk.observations[:-1], altered))
    elif fault == "rig":
        camera = bundle.cameras[0]
        bundle = replace(bundle, cameras=(replace(camera, T=camera.T+.01), bundle.cameras[1]))
    elif fault == "noise":
        bundle = replace(bundle, noise=NoiseModel.proxy(5))
    elif fault == "scale":
        frames[-1] = replace(info, scale=info.scale*2)
    elif fault == "shape":
        frames[-1] = replace(info, matrix=info.matrix[:-1, :-1])
    elif fault == "nonfinite":
        frames[-1] = replace(info, matrix=torch.full_like(info.matrix, torch.nan))
    elif fault == "rows":
        frames = frames[:-1]
    elif fault == "gradient":
        frames[-1] = replace(info, matrix=info.matrix.clone().requires_grad_())
    elif fault == "counts":
        frames[-1] = replace(info, resources={**info.resources, "decoder_state_evaluations": -1})
    elif fault == "reason":
        frames[1] = replace(frames[1], reasons=[])
    elif fault == "legacy_anchor":
        frames[-1] = replace(info, diagnostics={**info.diagnostics, "candidate_current_observation_binding": "legacy_unverified"})
    else:
        g1 = SO3JointFlow(normal).double().eval()
    bundle = replace(bundle, frames=tuple(frames))
    def forbidden(*args, **kwargs):
        raise AssertionError("all original information must validate before any G1 encoding")
    monkeypatch.setattr(g1.encoder, "forward", forbidden)
    with pytest.raises(ValueError):
        sequence_objective("regression", g1, chunk, normal, information=bundle)


def test_G1_sequence_gradients_reach_observation_and_core_but_not_acquisition():
    from ..schema import TemporalState
    from ..train import sequence_objective

    g0, g1, original, normal, bundle, _ = information_sequence_fixture()
    observations = tuple(replace(obs, xy=obs.xy.clone().requires_grad_()) for obs in original.observations)
    targets = replace(original.labels, translation=original.labels.translation.clone().requires_grad_(),
                      beta=original.labels.beta.clone().requires_grad_())
    chunk = replace(original, observations=observations, labels=targets)
    first = observations[0]
    carry = TemporalState(first.identity, first.sequence, first.camera_ids,
                          torch.zeros(128, dtype=torch.float64, requires_grad=True), torch.zeros(2, 17, dtype=torch.float64),
                          -.1, 99, torch.zeros(2, 17, dtype=torch.bool))
    result = sequence_objective("regression", g1, chunk, normal, carry, information=bundle)
    result.frame_losses[-1].backward()  # Dependency verification, not optimization.
    assert observations[0].xy.grad is not None and observations[0].xy.grad.abs().sum() > 0
    assert observations[3].xy.grad is not None and observations[3].xy.grad.abs().sum() > 0
    assert observations[1].xy.grad is None or observations[1].xy.grad.count_nonzero() == 0
    assert targets.translation.grad is None and targets.beta.grad is None and carry.hidden.grad is None
    assert g1.info_position.weight.grad.abs().sum() > 0 and g1.encoder.gru.weight_hh.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in g0.parameters())
    assert all(info.matrix is None or info.matrix.grad is None for info in bundle.frames)
    assert bundle.state.hidden.grad_fn is None and result.state.hidden.grad_fn is None


@pytest.mark.parametrize("mode", ["CFM", "regression"])
def test_G1_sequence_acquisition_future_isolation_and_chunk_carry(mode):
    from ..train import acquire_sequence_information, sequence_objective
    from .test_information import rigid_surface

    g0, g1, chunk, normal, bundle, _ = information_sequence_fixture()
    acquire = lambda observations, state=None: acquire_sequence_information(
        g0, observations, normal, bundle.scale, rigid_surface, bundle.cameras, bundle.noise,
        state, anchor_k=2, steps=2)
    first = acquire(chunk.observations[:3])
    last = acquire(chunk.observations[3:], first.state)
    for expected, actual in zip(bundle.frames, (*first.frames, *last.frames)):
        assert expected.binding == actual.binding
        assert (expected.matrix is actual.matrix is None) or torch.equal(expected.matrix, actual.matrix)
    assert bundle.state.to_dict() == last.state.to_dict()
    observations = chunk.observations[:3]+tuple(replace(obs, xy=obs.xy*2) for obs in chunk.observations[3:])
    targets = chunk.labels.translation.clone()
    targets[3:] += 1
    future = replace(chunk, observations=observations, labels=replace(chunk.labels, translation=targets))
    changed = acquire(future.observations)
    with torch.no_grad():
        before = sequence_objective(mode, g1, chunk, normal, information=bundle,
                                    generator=torch.Generator().manual_seed(17))
        after = sequence_objective(mode, g1, future, normal, information=changed,
                                   generator=torch.Generator().manual_seed(17))
    assert torch.equal(before.contexts[:3], after.contexts[:3])
    assert torch.allclose(before.frame_losses[:3], after.frame_losses[:3], equal_nan=True)


def test_G1_sequence_unavailable_information_keeps_fallback_rows_and_reasons():
    from ..train import acquire_sequence_information, sequence_objective

    g0, g1, chunk, normal, bundle, _ = information_sequence_fixture()
    def unavailable_surface(*args):
        raise ValueError("controlled_surface_unavailable_not_missing_target")
    unavailable = acquire_sequence_information(g0, chunk.observations, normal, bundle.scale, unavailable_surface,
                                               bundle.cameras, bundle.noise, anchor_k=2, steps=2)
    assert all(info.matrix is None and info.reasons for info in unavailable.frames)
    with torch.no_grad():
        f0 = sequence_objective("regression", g1, chunk, normal)
        fallback = sequence_objective("regression", g1, chunk, normal, information=unavailable)
    assert torch.equal(f0.loss, fallback.loss) and torch.equal(f0.contexts, fallback.contexts)
    assert torch.allclose(f0.frame_losses, fallback.frame_losses, equal_nan=True)
    assert fallback.supervised.tolist() == [True, False, True, False, True]
    assert fallback.resources["information_acquisition"]["unavailable_information_frames"] == 5
    assert fallback.reasons[3]["unavailable"] == ["beta_label_unavailable"]
    assert "controlled_surface_unavailable" in fallback.reasons[3]["information_unavailable"][0]
    assert "no_current_observation" in fallback.reasons[1]["information_unavailable"]


@pytest.mark.parametrize("fault", ["unfrozen", "order", "noise_rig", "carry_type"])
def test_G1_acquisition_rejects_bad_source_before_encoder(monkeypatch, fault):
    from ..train import acquire_sequence_information
    from .test_information import rigid_surface

    g0, _, chunk, normal, bundle, _ = information_sequence_fixture()
    observations, noise, state = chunk.observations, bundle.noise, None
    if fault == "unfrozen":
        g0.requires_grad_(True)
    elif fault == "order":
        observations = observations[:-2]+(observations[-1], observations[-2])
    elif fault == "noise_rig":
        noise = replace(noise, camera_ids=("other_LEFT", "other_RIGHT"))
    else:
        state = "not_a_temporal_state"
    def forbidden(*args, **kwargs):
        raise AssertionError("acquisition prerequisites must fail before any G0 frame")
    monkeypatch.setattr(g0.encoder, "forward", forbidden)
    with pytest.raises((ValueError, TypeError)):
        acquire_sequence_information(g0, observations, normal, bundle.scale, rigid_surface,
                                     bundle.cameras, noise, state, anchor_k=2, steps=2)


def test_G1_multi_stream_batch_preserves_bound_information_and_acquisition_counts(monkeypatch):
    from ..train import (
        acquire_sequence_information,
        sequence_batch_objective,
        sequence_objective,
    )
    from .test_information import rigid_surface
    from .test_joint import chunk_slice

    g0, g1, first, normal, first_info, _ = information_sequence_fixture()
    observations = tuple(replace(obs, identity="second_person", sequence="second_sequence") for obs in first.observations)
    bindings = tuple(replace(frame, identity="second_person", sequence="second_sequence") for frame in first.label_frames)
    second = replace(first, observations=observations, label_frames=bindings,
                     labels=replace(first.labels, translation=first.labels.translation+1))
    second_info = acquire_sequence_information(g0, observations, normal, first_info.scale, rigid_surface,
                                               first_info.cameras, first_info.noise, anchor_k=2, steps=2)
    chunks = (chunk_slice(first, 0, 3), second, chunk_slice(first, 3, 5))
    information = (replace(first_info, frames=first_info.frames[:3], anchor_resources=first_info.anchor_resources[:3]),
                   second_info, replace(first_info, frames=first_info.frames[3:], anchor_resources=first_info.anchor_resources[3:]))
    with torch.no_grad():
        batch = sequence_batch_objective("regression", g1, chunks, normal, information=information)
        direct_second = sequence_objective("regression", g1, second, normal, information=second_info)
    assert torch.equal(batch.chunks[1].contexts, direct_second.contexts)
    assert torch.equal(batch.chunks[1].loss, direct_second.loss)
    counts = batch.resources["information_acquisition"]
    assert counts["G0_network_calls"] == 20 and counts["G0_candidate_state_evaluations"] == 40
    assert counts["J_decoder_state_evaluations"] == 8*(1+2*g0.topology.tangent_dimensions)
    assert counts["available_information_frames"] == 8 and counts["unavailable_information_frames"] == 2
    assert not counts["computed_in_this_objective"]
    bad = replace(information[-1], frames=information[-1].frames[::-1])
    def forbidden(*args, **kwargs):
        raise AssertionError("late stale information must reject before all batch windows")
    monkeypatch.setattr(g1.encoder, "forward", forbidden)
    with pytest.raises(ValueError):
        sequence_batch_objective("regression", g1, chunks, normal, information=(*information[:-1], bad))
