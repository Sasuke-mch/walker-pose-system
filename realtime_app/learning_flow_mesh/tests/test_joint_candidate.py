"""H3 composition, availability and real differentiable body contracts."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from realtime_app.learning_flow_mesh.body import BodyDecoder
from realtime_app.learning_flow_mesh.candidate import (
    CausalMeshCandidate,
    OutputStatus,
    decode_available,
    saved_contract_conditions,
)
from realtime_app.learning_flow_mesh.interfaces import JointSchema
from realtime_app.learning_flow_mesh.model import FlowMesh, ModelConfig
from realtime_app.learning_flow_mesh.reproduction.check_joint_candidate import (
    parity,
    stream,
)
from realtime_app.pose_app.smpl_coco_observation import load_coco17_regressor


def fixture_arrays():
    shape = (6, 2, 17)
    rays = np.zeros((*shape, 3))
    rays[..., 2] = 1
    gate = np.ones((6, 17), dtype=bool)
    geometry = np.broadcast_to(gate[:, None], shape).copy()
    return {"image_normalized_xy": np.zeros((*shape, 2)), "confidence": np.ones(shape) * 0.7,
            "observation_mask": geometry.copy(), "geometry_mask": geometry,
            "strict_gate": gate, "ray_origins_m": np.zeros_like(rays),
            "ray_directions_cam0": rays, "selected_people": np.zeros((6, 2), dtype=np.int64),
            "frame_ids": np.arange(6, dtype=np.int32), "strict_xyz_m_diagnostic": np.ones((6, 17, 3))}


def inputs():
    return saved_contract_conditions(fixture_arrays(), schema=JointSchema.COCO17_SURFACE,
                                     frame="cam0", unit="meter")


@pytest.fixture
def model():
    torch.manual_seed(33)
    # Interface tests need no licensed body asset; actual body is tested separately.
    return FlowMesh(ModelConfig(observation_joints=17), [-1] + [0] * 51).eval()


def candidate(model, k=2, **kwargs):
    return CausalMeshCandidate(model, schema=JointSchema.COCO17_SURFACE,
                               calibration_frames=k, **kwargs)


def test_checked_adapter_retains_confidence_masks_and_excludes_xyz():
    arrays = fixture_arrays()
    arrays["confidence"][0, 0, 0] = 0
    batch, admitted, ids = saved_contract_conditions(arrays, schema=JointSchema.COCO17_SURFACE,
                                                     frame="cam0", unit="meter")
    assert batch["observation_mask"][0, 0, 0, 0] and batch["confidence"][0, 0, 0, 0] == 0
    assert ids.dtype == torch.long and admitted.all()
    assert not batch["image_mask"].any() and not any("xyz" in key for key in batch)
    assert arrays["image_normalized_xy"].dtype == np.float64
    arrays["strict_xyz_m_diagnostic"][:] = np.nan
    other, _admitted, _ids = saved_contract_conditions(arrays, schema=JointSchema.COCO17_SURFACE,
                                                      frame="cam0", unit="meter")
    assert all(torch.equal(value, other[key]) for key, value in batch.items())
    for schema, frame, unit in ((JointSchema.MOTIONBERT_H36M17, "cam0", "meter"),
                                (JointSchema.COCO17_SURFACE, "world", "meter"),
                                (JointSchema.COCO17_SURFACE, "cam0", "mm")):
        with pytest.raises(ValueError):
            saved_contract_conditions(arrays, schema=schema, frame=frame, unit=unit)


def test_checked_adapter_rejects_mask_erosion_person_substitution_and_numeric_gate():
    for failure in ("eroded_geometry", "selected_person", "numeric_gate"):
        arrays = fixture_arrays()
        if failure == "eroded_geometry":
            arrays["geometry_mask"][0, 0, 0] = False
        elif failure == "selected_person":
            arrays["selected_people"][0, 0] = -1
        else:
            arrays["strict_gate"] = arrays["strict_gate"].astype(int)
        with pytest.raises(ValueError):
            saved_contract_conditions(arrays, schema=JointSchema.COCO17_SURFACE,
                                       frame="cam0", unit="meter")


def test_complete_stream_composition_retains_noise_beta_and_generator(model):
    batch, admitted, ids = inputs()
    pipe = candidate(model)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        whole = pipe.step(batch, ids, ("one",), admitted).tensors
        for size in (1, 3):
            parity(whole, stream(pipe, batch, ids, admitted, ("one",), size))
        control = candidate(model, 1).step(batch, ids, ("one",), admitted).tensors
    assert torch.equal(whole["raw_state_diagnostic"], control["raw_state_diagnostic"])
    assert torch.equal(whole["raw_translation_diagnostic_m"], control["raw_translation_diagnostic_m"])
    assert whole["status"][0, 0] == OutputStatus.CALIBRATION_PENDING
    assert torch.isnan(whole["rotations"][0, 0]).all()
    assert torch.equal(whole["betas"][0, 1:], whole["betas"][0, 1:2].expand(5, -1))
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_missing_after_freeze_cannot_be_published_and_cannot_be_admitted(model):
    batch, admitted, ids = inputs()
    for key in ("observation_mask", "geometry_mask", "view_mask", "image_mask"):
        batch[key][:, 3] = False
    for key in ("xy", "confidence", "ray_origins", "ray_directions"):
        batch[key][:, 3] = torch.nan
    admitted[:, 3] = False
    pipe = candidate(model)
    with torch.no_grad():
        result = pipe.step(batch, ids, ("one",), admitted).tensors
    assert result["shape_ready"][0, 3] and not result["available"][0, 3]
    assert result["status"][0, 3] == OutputStatus.NO_CURRENT_OBSERVATION
    assert torch.isnan(result["rotations"][0, 3]).all()
    assert torch.isfinite(result["raw_rotations_diagnostic"][0, 3]).all()
    assert torch.equal(result["betas"][0, 2], result["betas"][0, 3])
    admitted[:, 3] = True
    with pytest.raises(ValueError, match="shape evidence"):
        pipe.step(batch, ids, ("one",), admitted)


def test_state_cannot_cross_sequence_repeat_frames_or_change_noise(model):
    batch, admitted, ids = inputs()
    pipe = candidate(model)
    first_batch = {key: value[:, :3] for key, value in batch.items()}
    next_batch = {key: value[:, 3:] for key, value in batch.items()}
    with torch.no_grad():
        first = pipe.step(first_batch, ids[:, :3], ("one",), admitted[:, :3])
        with pytest.raises(ValueError, match="another sequence"):
            pipe.step(next_batch, ids[:, 3:], ("two",), admitted[:, 3:], first.state)
        with pytest.raises(ValueError, match="duplicate or reordered"):
            pipe.step(first_batch, ids[:, :3], ("one",), admitted[:, :3], first.state)
        with pytest.raises(ValueError, match="noise seed"):
            candidate(model, noise_seed=999).step(next_batch, ids[:, 3:], ("one",), admitted[:, 3:], first.state)
        unavailable = candidate(model, k=8).step(batch, ids, ("one",), admitted).tensors
    assert not unavailable["shape_ready"].any() and not unavailable["available"].any()
    assert torch.isnan(unavailable["betas"]).all()


def test_future_input_changes_do_not_rewrite_prior_parameters(model):
    batch, admitted, ids = inputs()
    pipe = candidate(model)
    with torch.no_grad():
        original = pipe.step(batch, ids, ("one",), admitted).tensors
        batch["xy"][:, 4:] += 2
        changed = pipe.step(batch, ids, ("one",), admitted).tensors
    parity({key: value[:, :4] for key, value in original.items()},
           {key: value[:, :4] for key, value in changed.items()})
    torch.testing.assert_close(original["betas"], changed["betas"], atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("failure", ["translation", "velocity"])
def test_numerical_failures_are_unavailable_and_retain_diagnostics(model, failure):
    batch, admitted, ids = inputs()
    with torch.no_grad():
        if failure == "translation":
            model.encoder.translation_head.bias.fill_(torch.nan)
        else:
            model.velocity_head[-1].bias.fill_(torch.nan)
        result = candidate(model).step(batch, ids, ("one",), admitted).tensors
    expected = (OutputStatus.NONFINITE_PARAMETERS if failure == "translation"
                else OutputStatus.INVALID_ROTATION)
    assert (result["status"][0, 1:] == expected).all()
    assert not result["available"].any() and torch.isnan(result["rotations"]).all()
    key = "raw_translation_diagnostic_m" if failure == "translation" else "raw_state_diagnostic"
    assert torch.isnan(result[key]).any()
    assert torch.isfinite(result["betas"][0, 1:]).all()


def test_real_body_surface_and_flow_composition_are_differentiable_without_training():
    config = json.loads(Path("realtime_app/learning_flow_mesh/config.json").read_text(encoding="utf-8"))
    body = BodyDecoder(**config["assets"])
    torch.manual_seed(33)
    model = FlowMesh(ModelConfig(observation_joints=17), body.parents).eval()
    batch, admitted, ids = inputs()
    batch = {key: value[:, :3] for key, value in batch.items()}
    batch["xy"].requires_grad_(True)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    output = candidate(model).step(batch, ids[:, :3], ("one",), admitted[:, :3]).tensors
    regressor = load_coco17_regressor("models/smpl/J_regressor_coco.npy")
    decoded = decode_available(body, output, regressor)
    assert decoded["vertices"].shape == (1, 3, 6890, 3)
    assert decoded["faces"].shape == (13776, 3)
    assert torch.isnan(decoded["coco17_surface_m"][:, 0]).all()
    decoded["coco17_surface_m"][output["available"]].square().mean().backward()
    for parameter in (model.encoder.shape_head.weight, model.encoder.translation_head.weight,
                      model.velocity_head[-1].weight, model.encoder.obs_identity):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().max() > 0
    assert torch.isfinite(batch["xy"].grad).all() and batch["xy"].grad.abs().max() > 0
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
