"""H8 controlled dependency, availability and policy-bound streaming state."""
import json
from pathlib import Path

import pytest
import torch

from realtime_app.learning_flow_mesh.beta_conditioning import (
    BetaConditionedFlow,
    BetaConditionedMeshCandidate,
)
from realtime_app.learning_flow_mesh.body import BodyDecoder
from realtime_app.learning_flow_mesh.interfaces import JointSchema
from realtime_app.learning_flow_mesh.reproduction.check_hand_candidate import (
    slice_inputs,
)
from realtime_app.learning_flow_mesh.reproduction.check_joint_candidate import parity
from realtime_app.learning_flow_mesh.tests.test_hand_candidate import (
    setup as hand_setup,
)
from realtime_app.learning_flow_mesh.tests.test_shape_features import fixture


def setup():
    model, _old, control, batch, admitted, ids = fixture()
    torch.manual_seed(127)
    wrapper = BetaConditionedFlow(model).eval()
    candidate = BetaConditionedMeshCandidate(wrapper, control.conditioner,
                                             schema=JointSchema.COCO17_SURFACE, calibration_frames=2, steps=8)
    return wrapper, control, candidate, batch, admitted, ids


def test_disabled_exact_H7_and_unchanged_beta_translation_hidden():
    wrapper, control, candidate, batch, admitted, ids = setup()
    with torch.no_grad():
        original = control.step(batch, ids, ("sample",), admitted)
        wrapper.enabled = False
        disabled = candidate.step(batch, ids, ("sample",), admitted)
        wrapper.enabled = True
        active = candidate.step(batch, ids, ("sample",), admitted)
    assert all(value == 0 for value in parity(original.tensors, disabled.tensors).values())
    untouched = {key: value for key, value in original.tensors.items() if key not in
                 ("rotations", "raw_rotations_diagnostic", "raw_state_diagnostic")}
    assert all(value == 0 for value in parity(untouched, active.tensors).values())
    assert torch.equal(original.state.hidden, active.state.hidden)
    assert torch.equal(original.state.initial_noise, active.state.initial_noise)
    assert not torch.equal(original.tensors["raw_state_diagnostic"], active.tensors["raw_state_diagnostic"])


def test_ready_beta_has_flow_gradient_unknown_beta_noneffective():
    wrapper, _control, _candidate, _batch, _admitted, _ids = setup()
    condition = torch.randn(1, 3, 52, wrapper.cfg.width)
    betas = torch.randn(1, 3, 10)
    betas[:, 0] = torch.nan
    betas.requires_grad_()
    ready = torch.tensor([[False, True, True]])
    readout = {"condition": condition, "betas": betas, "shape_available": ready,
               "translation": torch.zeros(1, 3, 3), "frame_observed": ready}
    noise = torch.randn(1, 3, 52, 6)
    active = wrapper.transport(readout, steps=2, initial=noise)
    grad = torch.autograd.grad(active["state"].square().mean(), betas)[0]
    assert grad[:, 0].eq(0).all() and grad[:, 1:].abs().sum() > 0
    assert torch.isfinite(grad).all() and torch.isfinite(active["state"]).all()
    clean = dict(readout, betas=torch.zeros_like(betas))
    wrapper.enabled = False
    baseline = wrapper.transport(clean, steps=2, initial=noise)
    wrapper.enabled = True
    neutral = wrapper.transport(clean, steps=2, initial=noise)
    assert torch.equal(neutral["state"], baseline["state"])


@pytest.mark.parametrize("bad", ["ready_nan", "numeric_mask", "wrong_shape", "dtype"])
def test_invalid_declared_beta_rejected(bad):
    wrapper, _control, _candidate, _batch, _admitted, _ids = setup()
    condition = torch.zeros(1, 2, 52, wrapper.cfg.width)
    beta = torch.zeros(1, 2, 10)
    ready = torch.ones(1, 2, dtype=torch.bool)
    if bad == "ready_nan":
        beta[0, 0, 0] = torch.nan
    elif bad == "numeric_mask":
        ready = ready.float()
    elif bad == "wrong_shape":
        beta = beta[:, :, :9]
    else:
        beta = beta.double()
    with pytest.raises(ValueError):
        wrapper.beta_residual(condition, beta, ready)


def test_stream_policy_reset_guard_and_masked_nonfinite_invariance():
    wrapper, _control, candidate, batch, admitted, ids = setup()
    with torch.no_grad():
        first = candidate.step(slice_inputs(batch, 0, 3), ids[:, :3], ("sample",), admitted[:, :3])
    wrapper.enabled = False
    with pytest.raises(ValueError, match="reset"):
        candidate.step(slice_inputs(batch, 3, 6), ids[:, 3:], ("sample",), admitted[:, 3:], first.state)
    condition = torch.zeros(1, 2, 52, wrapper.cfg.width)
    unknown = torch.full((1, 2, 10), torch.nan)
    ready = torch.zeros(1, 2, dtype=torch.bool)
    assert wrapper.beta_residual(condition, unknown, ready).eq(0).all()
    unknown[:] = torch.inf
    assert wrapper.beta_residual(condition, unknown, ready).eq(0).all()


def test_beta_to_pose_to_realmesh_gradient_without_direct_mesh_beta_route():
    config = json.loads(Path("realtime_app/learning_flow_mesh/config.json").read_text(encoding="utf-8"))
    decoder = BodyDecoder(**config["assets"])
    model, _hand, _body, _candidate = hand_setup(decoder.parents)
    torch.manual_seed(127)
    wrapper = BetaConditionedFlow(model).eval()
    beta = torch.randn(1, 2, 10, requires_grad=True)
    ready = torch.tensor([[False, True]])
    readout = {"condition": torch.randn(1, 2, 52, model.cfg.width), "betas": beta,
               "shape_available": ready, "translation": torch.zeros(1, 2, 3), "frame_observed": ready}
    output = wrapper.transport(readout, steps=2, initial=torch.randn(1, 2, 52, 6))
    # Body beta is constant: any beta gradient below must pass through pose.
    mesh = decoder(output["rotations"], torch.zeros(1, 10), torch.zeros(1, 2, 3))
    beta_grad, weight_grad = torch.autograd.grad(mesh["vertices"].square().mean(), (beta, wrapper.projection.weight))
    assert torch.isfinite(beta_grad).all() and beta_grad[:, 0].eq(0).all()
    assert beta_grad[:, 1].abs().sum() > 0
    assert torch.isfinite(weight_grad).all() and weight_grad.abs().sum() > 0
