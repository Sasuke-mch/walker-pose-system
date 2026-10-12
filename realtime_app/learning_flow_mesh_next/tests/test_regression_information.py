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
