"""Fair information controls without training or updating any parameters."""
import pytest
import torch

from ..data_firewall import Normalization
from ..joint_flow import JointFlow, ModelConfig
from ..train import objective
from .test_firewall import labels


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
