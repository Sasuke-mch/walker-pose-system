"""H7 information attribution and stream-state contracts."""
import pytest
import torch

from realtime_app.learning_flow_mesh.hand_candidate import disabled_native_hands
from realtime_app.learning_flow_mesh.interfaces import JointSchema
from realtime_app.learning_flow_mesh.missing_history import MissingHistoryMeshCandidate
from realtime_app.learning_flow_mesh.reproduction.check_hand_candidate import (
    slice_inputs,
)
from realtime_app.learning_flow_mesh.reproduction.check_joint_candidate import parity
from realtime_app.learning_flow_mesh.shape_features import ShapeFeatureMeshCandidate
from realtime_app.learning_flow_mesh.tests.test_hand_candidate import hands_for, setup
from realtime_app.learning_flow_mesh.tests.test_joint_candidate import inputs


def fixture():
    model, branch, _body, _hand = setup()
    batch, admitted, ids = inputs()
    batch = dict(batch, native_hands=hands_for(ids, model.parents))
    options = {"schema": JointSchema.COCO17_SURFACE, "calibration_frames": 2, "steps": 8}
    control = MissingHistoryMeshCandidate(model, branch, **options)
    candidate = ShapeFeatureMeshCandidate(model, branch, **options)
    return model, control, candidate, batch, admitted, ids


def test_history_control_and_local_leave_all_nonbeta_outputs_exact():
    model, control, candidate, batch, admitted, ids = fixture()
    history = ShapeFeatureMeshCandidate(model, candidate.conditioner,
                                       shape_feature_source="history", schema=JointSchema.COCO17_SURFACE,
                                       calibration_frames=2, steps=8)
    with torch.no_grad():
        before = control.step(batch, ids, ("sample",), admitted)
        same = history.step(batch, ids, ("sample",), admitted)
        local = candidate.step(batch, ids, ("sample",), admitted)
    assert all(value == 0 for value in parity(before.tensors, same.tensors).values())
    nonbeta = {key: value for key, value in before.tensors.items() if key != "betas"}
    assert all(value == 0 for value in parity(nonbeta, local.tensors).values())
    assert torch.equal(before.state.hidden, local.state.hidden)
    assert torch.equal(before.state.initial_noise, local.state.initial_noise)


def test_nonadmitted_observed_frame_has_no_local_shape_gradient():
    _model, control, candidate, batch, _admitted, ids = fixture()
    # Three observed frames, but only frames1/2 admitted as shape evidence.
    batch = slice_inputs(batch, 0, 3)
    batch["view_mask"][:] = True
    batch["observation_mask"][:] = True
    batch["geometry_mask"][:] = False
    xy = batch["xy"].clone().requires_grad_()
    batch["xy"] = xy
    mask = torch.tensor([[False, True, True]])
    local = candidate.step(batch, ids[:, :3], ("sample",), mask)
    history = control.step(batch, ids[:, :3], ("sample",), mask)
    local_grad = torch.autograd.grad(local.tensors["betas"][:, -1].square().sum(), xy)[0]
    history_grad = torch.autograd.grad(history.tensors["betas"][:, -1].square().sum(), xy)[0]
    assert local_grad[:, 0].eq(0).all()
    assert local_grad[:, 1:].abs().sum() > 0
    assert history_grad[:, 0].abs().sum() > 0
    assert torch.isfinite(local_grad).all() and torch.isfinite(history_grad).all()


def test_later_local_features_ignore_earlier_inputs_but_history_does_not():
    _model, control, candidate, batch, _admitted, ids = fixture()
    changed = dict(batch, xy=batch["xy"].clone())
    changed["xy"][:, 0] += .8
    with torch.no_grad():
        _, local = candidate._encode(batch, ids, None)
        _, changed_local = candidate._encode(changed, ids, None)
        _, history = control._encode(batch, ids, None)
        _, changed_history = control._encode(changed, ids, None)
    assert torch.equal(local[:, 1:], changed_local[:, 1:])
    assert not torch.equal(history[:, 1:], changed_history[:, 1:])


def test_policy_bound_state_and_chunk_parity():
    _model, _control, candidate, batch, admitted, ids = fixture()
    with torch.no_grad():
        whole = candidate.step(batch, ids, ("sample",), admitted)
        first = candidate.step(slice_inputs(batch, 0, 3), ids[:, :3], ("sample",), admitted[:, :3])
        last = candidate.step(slice_inputs(batch, 3, 6), ids[:, 3:], ("sample",), admitted[:, 3:], first.state)
    combined = {key: torch.cat((first.tensors[key], last.tensors[key]), 1) for key in whole.tensors}
    assert all(value <= 2e-5 for value in parity(whole.tensors, combined).values())
    candidate.shape_feature_source = "history"
    with pytest.raises(ValueError, match="reset"):
        candidate.step(slice_inputs(batch, 3, 6), ids[:, 3:], ("sample",), admitted[:, 3:], first.state)
    candidate.shape_feature_source = "local"
    candidate.missing_policy = "update_all"
    with pytest.raises(ValueError, match="reset"):
        candidate.step(slice_inputs(batch, 3, 6), ids[:, 3:], ("sample",), admitted[:, 3:], first.state)


def test_mixed_rows_have_independent_calibration_and_explicit_reset():
    model, _control, candidate, batch, admitted, ids = fixture()
    both = {key: value.repeat(2, *([1] * (value.ndim - 1)))
            for key, value in batch.items() if key != "native_hands"}
    both_ids = ids.repeat(2, 1)
    both["native_hands"] = hands_for(both_ids, model.parents)
    both_admitted = admitted.repeat(2, 1)
    both_admitted[0, :2] = False
    with torch.no_grad():
        rows = candidate.step(both, both_ids, ("delayed", "sample"), both_admitted)
        original = candidate.step(batch, ids, ("sample",), admitted)
        reset = candidate.step(batch, ids, ("new_sequence",), admitted)
    assert rows.tensors["shape_ready"][0].tolist() == [False, False, False, True, True, True]
    assert rows.tensors["shape_ready"][1].tolist() == [False, True, True, True, True, True]
    # Same seed draws different per-row noise when batch size changes. Compare
    # shape/state here; cross-batch pose equality would compare different noise.
    shape = {key: original.tensors[key] for key in ("betas", "shape_ready", "shape_admitted")}
    assert all(value <= 2e-5 for value in parity(
        shape, {key: value[1:2] for key, value in rows.tensors.items()}).values())
    torch.testing.assert_close(rows.state.hidden[:, 1:2], original.state.hidden, atol=2e-5, rtol=0)
    assert all(value == 0 for value in parity(original.tensors, reset.tensors).values())
    assert reset.state.shape.sequence_ids == ("new_sequence",)


def test_missing_nan_prefix_is_unavailable_and_hooks_clean_on_failure():
    model, _control, candidate, batch, _admitted, ids = fixture()
    batch["view_mask"][:] = False
    batch["native_hands"] = disabled_native_hands(batch["native_hands"])
    batch["xy"][:] = torch.nan
    admitted = torch.zeros_like(ids, dtype=torch.bool)
    with torch.no_grad():
        output = candidate.step(batch, ids, ("sample",), admitted)
    assert not output.tensors["available"].any()
    assert torch.isnan(output.tensors["betas"]).all()
    assert output.state.shape.count.eq(0).all()
    batch["view_mask"][:] = True
    with pytest.raises(ValueError, match="finite"):
        candidate.step(batch, ids, ("sample",), admitted)
    assert not model.encoder.history._forward_pre_hooks
    assert not model.encoder.history._forward_hooks
