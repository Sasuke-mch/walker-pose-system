"""H5 single-factor fusion, unchanged body state and causal gradient contracts."""
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from smplx.lbs import batch_rodrigues

from realtime_app.learning_flow_mesh.body import BodyDecoder
from realtime_app.learning_flow_mesh.candidate import (
    CausalMeshCandidate,
    OutputStatus,
    decode_available,
)
from realtime_app.learning_flow_mesh.hand_candidate import (
    NativeHandConditioner,
    NativeHandMeshCandidate,
    disabled_native_hands,
    slice_native_hands,
)
from realtime_app.learning_flow_mesh.hand_conditions import (
    SIDE_ORDER,
    HandConvention,
    native_hand_tokens,
)
from realtime_app.learning_flow_mesh.interfaces import JointSchema
from realtime_app.learning_flow_mesh.model import FlowMesh, ModelConfig
from realtime_app.learning_flow_mesh.reproduction.check_hand_candidate import (
    slice_inputs,
    stream,
)
from realtime_app.learning_flow_mesh.reproduction.check_joint_candidate import parity
from realtime_app.learning_flow_mesh.tests.test_hand_conditions import parents
from realtime_app.learning_flow_mesh.tests.test_joint_candidate import inputs
from realtime_app.pose_app.smpl_coco_observation import load_coco17_regressor


def hands_for(ids, tree, *, rotations=None, score=None):
    b, t = ids.shape
    if rotations is None:
        theta = torch.zeros(b * t * 2 * 2 * 15, 3)
        theta[:, 0] = .2
        rotations = batch_rodrigues(theta).reshape(b, t, 2, 2, 15, 3, 3)
    if score is None:
        score = torch.full((b, t, 2, 2), .7)
    return native_hand_tokens(rotations, score, source_frame_ids=ids,
                              target_frame_ids=ids, view_available=torch.ones(b, t, 2, dtype=torch.bool),
                              convention=HandConvention.SMPLH_PARENT_LOCAL_NATIVE,
                              side_order=SIDE_ORDER, parents=tree)


def setup(tree=None):
    tree = parents() if tree is None else tree
    torch.manual_seed(19)
    model = FlowMesh(ModelConfig(observation_joints=17), tree).eval()
    torch.manual_seed(29)
    branch = NativeHandConditioner().eval()
    options = {"schema": JointSchema.COCO17_SURFACE, "calibration_frames": 2, "steps": 8}
    return model, branch, CausalMeshCandidate(model, **options), NativeHandMeshCandidate(model, branch, **options)


def test_missing_hand_intervention_reproduces_H3_exactly():
    model, branch, control, candidate = setup()
    batch, admitted, ids = inputs()
    hands = hands_for(ids, model.parents)
    missing = disabled_native_hands(hands)
    assert missing.source_present.all() and not missing.admitted.any()
    before = {k: v.clone() for k, v in branch.state_dict().items()}
    with torch.no_grad():
        base = control.step(batch, ids, ("sample",), admitted)
        other = candidate.step(dict(batch, native_hands=missing), ids, ("sample",), admitted)
        assert all(value == 0 for value in parity(base.tensors, other.tensors).values())
        assert torch.equal(base.state.hidden, other.state.hidden)
        assert torch.equal(base.state.initial_noise, other.state.initial_noise)
        encoded, _ = control._encode(batch, ids, None)
        assert torch.equal(branch(encoded["condition"], missing), torch.zeros_like(encoded["condition"]))
    assert all(torch.equal(before[k], v) for k, v in branch.state_dict().items())


def test_native_changes_only_pose_condition_and_preserves_body_state():
    model, branch, control, candidate = setup()
    batch, admitted, ids = inputs()
    hands = hands_for(ids, model.parents)
    with torch.no_grad():
        original = control.step(batch, ids, ("sample",), admitted)
        result = candidate.step(dict(batch, native_hands=hands), ids, ("sample",), admitted)
        for key in ("betas", "translation_m", "shape_ready", "shape_admitted", "available", "status"):
            assert all(value == 0 for value in parity({key: original.tensors[key]}, {key: result.tensors[key]}).values())
        assert torch.equal(original.state.hidden, result.state.hidden)
        assert not torch.equal(original.tensors["raw_state_diagnostic"], result.tensors["raw_state_diagnostic"])
        encoded, _ = control._encode(batch, ids, None)
        delta = branch(encoded["condition"], hands)
        assert not delta[:, :, 22:].eq(0).all()
        assert delta[:, :, :22].eq(0).all()


def test_same_hand_attention_and_masked_values_have_no_cross_hand_leak():
    model, branch, control, _candidate = setup()
    batch, _admitted, ids = inputs()
    score = torch.full((1, 6, 2, 2), .7)
    score[:, :, 1] = 0  # second camera entirely absent
    hands = hands_for(ids, model.parents, score=score)
    changed = hands.tokens.clone()
    changed[:, :, 0, 0, :, :6] += .2
    changed[:, :, 1] = 9999  # all masked, must have no effect
    with torch.no_grad():
        encoded, _ = control._encode(batch, ids, None)
        first = branch(encoded["condition"], hands)
        second = branch(encoded["condition"], replace(hands, tokens=changed))
    assert not torch.equal(first[:, :, 22:37], second[:, :, 22:37])
    assert torch.equal(first[:, :, 37:], second[:, :, 37:])
    assert torch.equal(first[:, :, :22], second[:, :, :22])


@pytest.mark.parametrize("fault", ["frames", "nodes", "mask", "nonfinite", "dtype", "unavailable_view"])
def test_misaligned_or_invalid_hand_conditions_rejected(fault):
    model, _branch, _control, candidate = setup()
    batch, admitted, ids = inputs()
    hands = hands_for(ids, model.parents)
    if fault == "frames":
        hands = replace(hands, frame_ids=hands.frame_ids + 1)
    elif fault == "nodes":
        hands = replace(hands, node_ids=hands.node_ids.flip(0))
    elif fault == "mask":
        hands = replace(hands, present=hands.present.float())
    elif fault == "nonfinite":
        value = hands.tokens.clone()
        value[0, 0, 0, 0, 0, 0] = torch.nan
        hands = replace(hands, tokens=value)
    elif fault == "dtype":
        hands = replace(hands, tokens=hands.tokens.double())
    else:
        batch["view_mask"][:, :, 0] = False
    with pytest.raises(ValueError):
        candidate.step(dict(batch, native_hands=hands), ids, ("sample",), admitted)


def test_current_hands_do_not_publish_body_missing_frames_or_count_shape():
    model, _branch, _control, candidate = setup()
    batch, admitted, ids = inputs()
    hands = hands_for(ids, model.parents)
    for name in ("observation_mask", "geometry_mask"):
        batch[name][:, 3] = False
    admitted[:, 3] = False
    with torch.no_grad():
        result = candidate.step(dict(batch, native_hands=hands), ids, ("sample",), admitted).tensors
    assert result["status"][0, 3] == OutputStatus.NO_CURRENT_OBSERVATION
    assert not result["available"][0, 3]
    assert torch.isnan(result["rotations"][0, 3]).all()
    assert result["shape_ready"][0, 3] and not result["shape_admitted"][0, 3]


def test_native_stream_and_future_isolation():
    model, _branch, _control, candidate = setup()
    batch, admitted, ids = inputs()
    hands = hands_for(ids, model.parents)
    batch["native_hands"] = hands
    with torch.no_grad():
        whole = candidate.step(batch, ids, ("repositioned360_H5",), admitted).tensors
        for size in (1, 3):
            parity(whole, stream(candidate, batch, ids, admitted, size))
        future = hands.tokens.clone()
        future[:, 4:, ..., :6] += .25
        changed = candidate.step(dict(batch, native_hands=replace(hands, tokens=future)),
                                 ids, ("repositioned360_H5",), admitted).tensors
        prefix = parity({k: v[:, :4] for k, v in whole.items()}, {k: v[:, :4] for k, v in changed.items()})
        assert all(value == 0 for value in prefix.values())
    assert torch.equal(slice_native_hands(hands, 2, 3).frame_ids, ids[:, 2:3])
    assert slice_inputs(batch, 2, 3)["native_hands"].tokens.shape[1] == 1


def test_real_mesh_hand_condition_gradients_without_training():
    config = json.loads(Path("realtime_app/learning_flow_mesh/config.json").read_text(encoding="utf-8"))
    decoder = BodyDecoder(**config["assets"])
    model, branch, _control, candidate = setup(decoder.parents)
    batch, admitted, ids = inputs()
    theta = torch.full((1, 6, 2, 2, 15, 3), .1, requires_grad=True)
    rotations = batch_rodrigues(theta.reshape(-1, 3)).reshape(1, 6, 2, 2, 15, 3, 3)
    score = torch.full((1, 6, 2, 2), .7)
    score[:, 1] = 0
    score[:, 2, 1] = 0  # current-frame absent camera must receive no gradient
    score.requires_grad_()
    hands = hands_for(ids, decoder.parents, rotations=rotations, score=score)
    before = {k: v.detach().clone() for k, v in branch.state_dict().items()}
    result = candidate.step(dict(batch, native_hands=hands), ids, ("gradient",), admitted).tensors
    selected = {key: value[:, 2:3] for key, value in result.items()}
    regressor = torch.tensor(load_coco17_regressor(Path("models/smpl/J_regressor_coco.npy")))
    mesh = decode_available(decoder, selected, regressor)
    (mesh["vertices"].square().mean() + mesh["coco17_surface_m"].square().mean()).backward()
    for gradient in (theta.grad[:, 2], score.grad[:, 2]):
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    for frame in (0, 1, 3, 4, 5):
        assert theta.grad[:, frame].eq(0).all() and score.grad[:, frame].eq(0).all()
    assert theta.grad[:, 2, 1].eq(0).all() and score.grad[:, 2, 1].eq(0).all()
    for parameter in (branch.projection[0].weight, branch.node_identity, branch.view_identity,
                      branch.attention.in_proj_weight, branch.attention.out_proj.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert all(torch.equal(before[k], v) for k, v in branch.state_dict().items())
    assert model.velocity_head[-1].weight.grad is not None
