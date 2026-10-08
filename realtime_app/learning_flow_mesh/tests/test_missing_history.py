"""H6 state invariants, duration blindness and recovered gradient contracts."""
import pytest
import torch

from realtime_app.learning_flow_mesh.candidate import OutputStatus
from realtime_app.learning_flow_mesh.hand_candidate import disabled_native_hands
from realtime_app.learning_flow_mesh.interfaces import JointSchema
from realtime_app.learning_flow_mesh.missing_history import (
    MissingHistoryMeshCandidate,
    freeze_missing_transitions,
)
from realtime_app.learning_flow_mesh.reproduction.check_hand_candidate import (
    slice_inputs,
)
from realtime_app.learning_flow_mesh.reproduction.check_joint_candidate import parity
from realtime_app.learning_flow_mesh.tests.test_hand_candidate import hands_for, setup
from realtime_app.learning_flow_mesh.tests.test_joint_candidate import inputs


def fixture():
    model, branch, _body, control = setup()
    batch, admitted, ids = inputs()
    batch = dict(batch, native_hands=hands_for(ids, model.parents))
    options = {"schema": JointSchema.COCO17_SURFACE, "calibration_frames": 2, "steps": 8}
    return model, control, MissingHistoryMeshCandidate(model, branch, **options), batch, admitted, ids


def test_disabled_history_policy_is_exact_H5():
    model, control, candidate, batch, admitted, ids = fixture()
    candidate.missing_policy = "update_all"
    with torch.no_grad():
        before = control.step(batch, ids, ("sample",), admitted)
        after = candidate.step(batch, ids, ("sample",), admitted)
    assert all(value == 0 for value in parity(before.tensors, after.tensors).values())
    assert torch.equal(before.state.hidden, after.state.hidden)
    assert not model.encoder.history._forward_hooks


def test_all_observed_matches_original_gru():
    torch.manual_seed(13)
    gru = torch.nn.GRU(5, 7, batch_first=True).eval()
    pooled = torch.randn(2, 9, 5)
    observed = torch.ones(2, 9, dtype=torch.bool)
    expected, expected_hidden = gru(pooled)
    actual, actual_hidden = freeze_missing_transitions(gru, pooled, observed)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=0)
    torch.testing.assert_close(actual_hidden, expected_hidden, atol=2e-5, rtol=0)


@pytest.mark.parametrize("duration", [1, 8, 64])
def test_missing_duration_ignores_elapsed_time_and_preserves_carried_state(duration):
    gru = torch.nn.GRU(5, 7, batch_first=True).eval()
    hidden = torch.randn(1, 2, 7)
    original = hidden.clone()
    recovery = torch.randn(2, 1, 5)
    gap = torch.randn(2, duration, 5)
    observed = torch.zeros(2, duration + 1, dtype=torch.bool)
    observed[:, -1] = True
    output, state = freeze_missing_transitions(gru, torch.cat((gap, recovery), 1), observed, hidden)
    expected, expected_hidden = gru(recovery, hidden)
    assert torch.equal(hidden, original)
    assert torch.equal(output[:, :-1], hidden[-1, :, None].expand(-1, duration, -1))
    assert torch.equal(state, expected_hidden)
    assert torch.equal(output[:, -1:], expected)


def test_mixed_rows_missing_prefix_and_gradients():
    torch.manual_seed(15)
    gru = torch.nn.GRU(5, 7, batch_first=True).eval()
    pooled = torch.randn(2, 4, 5, requires_grad=True)
    observed = torch.tensor([[False, False, True, True], [True, False, False, True]])
    output, hidden = freeze_missing_transitions(gru, pooled, observed)
    assert output[0, :2].eq(0).all()
    assert torch.equal(output[1, 0], output[1, 1]) and torch.equal(output[1, 1], output[1, 2])
    output[:, -1].square().sum().backward()
    assert pooled.grad[~observed].eq(0).all()
    assert torch.isfinite(pooled.grad).all() and pooled.grad[observed].abs().sum() > 0
    assert gru.weight_hh_l0.grad.abs().sum() > 0 and torch.isfinite(gru.weight_hh_l0.grad).all()
    assert torch.equal(hidden[-1], output[:, -1])


def test_scoped_hooks_removed_on_failure():
    model, _control, candidate, batch, admitted, ids = fixture()
    batch["xy"][batch["observation_mask"]] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        candidate.step(batch, ids, ("sample",), admitted)
    assert not model.encoder.history._forward_hooks


def test_shape_admission_and_chunk_state_remain_explicit():
    _model, _control, candidate, batch, admitted, ids = fixture()
    before_params = {key: value.clone() for key, value in candidate.model.state_dict().items()}
    with torch.no_grad():
        whole = candidate.step(batch, ids, ("sample",), admitted)
        first = candidate.step(slice_inputs(batch, 0, 3), ids[:, :3], ("sample",), admitted[:, :3])
        last = candidate.step(slice_inputs(batch, 3, 6), ids[:, 3:], ("sample",), admitted[:, 3:], first.state)
    combined = {key: torch.cat((first.tensors[key], last.tensors[key]), 1) for key in whole.tensors}
    assert all(value <= 2e-5 for value in parity(whole.tensors, combined).values())
    assert torch.equal(whole.tensors["shape_admitted"], admitted)
    assert torch.equal(whole.tensors["available"], whole.tensors["frame_observed"] & whole.tensors["shape_ready"])
    assert all(torch.equal(before_params[key], value) for key, value in candidate.model.state_dict().items())


def test_invalid_policy_and_training_mode_rejected():
    model, _control, candidate, _batch, _admitted, _ids = fixture()
    with pytest.raises(ValueError, match="policy"):
        MissingHistoryMeshCandidate(model, candidate.conditioner, schema=JointSchema.COCO17_SURFACE, missing_policy="unknown")
    gru = torch.nn.GRU(5, 7, batch_first=True)
    with pytest.raises(ValueError, match="eval"):
        freeze_missing_transitions(gru, torch.randn(1, 2, 5), torch.ones(1, 2, dtype=torch.bool))


def test_missing_body_freezes_hidden_but_keeps_frame_identity_and_unavailable_output():
    _model, _control, candidate, batch, admitted, ids = fixture()
    batch["native_hands"] = disabled_native_hands(batch["native_hands"])
    batch["observation_mask"][:, 2:4] = False
    batch["geometry_mask"][:, 2:4] = False
    admitted[:, 2:4] = False
    with torch.no_grad():
        prefix = candidate.step(slice_inputs(batch, 0, 2), ids[:, :2], ("sample",), admitted[:, :2])
        saved_hidden = prefix.state.hidden.clone()
        gap = candidate.step(slice_inputs(batch, 2, 4), ids[:, 2:4], ("sample",), admitted[:, 2:4], prefix.state)
        recovered = candidate.step(slice_inputs(batch, 4, 6), ids[:, 4:], ("sample",), admitted[:, 4:], gap.state)
    assert torch.equal(prefix.state.hidden, saved_hidden) and torch.equal(gap.state.hidden, saved_hidden)
    assert gap.state.shape.last_frame.item() == 3
    assert gap.state.shape.count.item() == 2
    assert not gap.tensors["available"].any()
    assert (gap.tensors["status"] == OutputStatus.NO_CURRENT_OBSERVATION).all()
    assert torch.isnan(gap.tensors["rotations"]).all() and torch.isnan(gap.tensors["translation_m"]).all()
    assert recovered.tensors["available"].all()


def test_recovered_generation_has_observed_gradients_and_no_masked_input_gradient():
    model, _control, candidate, batch, admitted, ids = fixture()
    batch["native_hands"] = disabled_native_hands(batch["native_hands"])
    batch["observation_mask"][:, 2] = False
    batch["geometry_mask"][:, 2] = False
    admitted[:, 0] = False  # A current observation may update history without being shape evidence.
    admitted[:, 2] = False
    batch["xy"] = batch["xy"].clone().requires_grad_()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    result = candidate.step(batch, ids, ("sample",), admitted)
    assert result.tensors["shape_ready"][0, :3].eq(False).all()
    assert result.tensors["shape_ready"][0, 3]
    loss = result.tensors["raw_state_diagnostic"][:, 3].square().mean()
    loss = loss + result.tensors["raw_translation_diagnostic_m"][:, 3].square().mean()
    loss.backward()
    assert torch.isfinite(batch["xy"].grad).all()
    assert batch["xy"].grad[:, 2].eq(0).all() and batch["xy"].grad[:, 4:].eq(0).all()
    assert batch["xy"].grad[:, 1].abs().sum() > 0 and batch["xy"].grad[:, 3].abs().sum() > 0
    assert torch.isfinite(model.encoder.history.weight_hh_l0.grad).all()
    assert model.encoder.history.weight_hh_l0.grad.abs().sum() > 0
    assert not model.encoder.history._forward_hooks
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
