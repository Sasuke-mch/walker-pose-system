"""H3 eager candidate: typed COCO conditions, causal shape and explicit availability.

This composes the unchanged generator with H1/H2; it is not the engineering
mainline or a pretrained real-data estimator. Internal missing-frame predictions
are retained as diagnostics but never published as available body outputs.
"""
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import torch

from realtime_app.pose_app.smpl_coco_observation import regress_coco17_torch

from .interfaces import JointSchema, require_joint_schema
from .shared_shape import PrefixSharedShape, ShapeState, encode_with_global_features


class OutputStatus(IntEnum):
    AVAILABLE = 0
    CALIBRATION_PENDING = 1
    NO_CURRENT_OBSERVATION = 2
    INVALID_ROTATION = 3
    NONFINITE_PARAMETERS = 4


@dataclass(frozen=True)
class CandidateState:
    shape: ShapeState
    hidden: torch.Tensor | None
    initial_noise: torch.Tensor
    noise_seed: int


@dataclass(frozen=True)
class CandidateReadout:
    tensors: dict[str, torch.Tensor]
    state: CandidateState


def saved_contract_conditions(arrays, *, schema: JointSchema, frame: str, unit: str,
                              device="cpu", dtype=torch.float32):
    """Select conditions only; neither triangulated XYZ nor labels enter the model.

    The H2 sidecar declares these conventions. Strict acceptance stays separate
    from 2D availability. No confidence threshold, coordinate fix or new geometric
    gate is applied. Raw reasons and rejected XYZ remain with the caller.
    """
    if frame != "cam0" or unit != "meter":
        raise ValueError("need explicitly cam0, metre H2 contract")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("candidate requires explicit floating model dtype")
    xy = np.asarray(arrays["image_normalized_xy"])
    require_joint_schema(xy, actual=schema, expected=JointSchema.COCO17_SURFACE)
    if xy.ndim != 4 or xy.shape[1:] != (2, 17, 2) or len(xy) < 1:
        raise ValueError("need a nonempty T,2,17,2 H2 sequence")
    shape = xy.shape[:-1]
    for key, expected in (("observation_mask", shape), ("geometry_mask", shape),
                          ("strict_gate", (len(xy), 17))):
        value = np.asarray(arrays[key])
        if value.dtype != np.bool_ or value.shape != expected:
            raise ValueError(f"{key} must be a boolean array of shape {expected}")
    people = np.asarray(arrays["selected_people"])
    if people.shape != (len(xy), 2) or not np.issubdtype(people.dtype, np.integer):
        raise ValueError("explicit T,2 integer person selections required")
    views = people >= 0
    observed = arrays["observation_mask"]
    geometry = arrays["geometry_mask"]
    gate = arrays["strict_gate"]
    if np.any(observed & ~views[..., None]) or np.any(geometry & ~observed):
        raise ValueError("presence/geometry contradicts selected people or observed pixels")
    if not np.array_equal(geometry, np.broadcast_to(gate[:, None], shape)):
        raise ValueError("H2 geometry must retain both-view immutable strict mask")
    expected_fields = {"confidence": shape, "ray_origins_m": (*shape, 3),
                       "ray_directions_cam0": (*shape, 3)}
    for key, expected in expected_fields.items():
        if np.asarray(arrays[key]).shape != expected:
            raise ValueError(f"{key} shape differs from H2 input")
    ids = np.asarray(arrays["frame_ids"])
    if (ids.shape != (len(xy),) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or np.any(np.diff(ids) <= 0)):
        raise ValueError("nonnegative increasing integer frame IDs required")

    def floating(value):
        # Declared float64 H2 geometry -> model dtype once, without changing source.
        return torch.as_tensor(np.array(value, copy=True), device=device, dtype=dtype)[None]

    def boolean(value):
        return torch.as_tensor(np.array(value, copy=True), device=device, dtype=torch.bool)[None]

    batch = {"xy": floating(xy), "confidence": floating(arrays["confidence"]),
             "observation_mask": boolean(observed), "geometry_mask": boolean(geometry),
             "ray_origins": floating(arrays["ray_origins_m"]),
             "ray_directions": floating(arrays["ray_directions_cam0"]),
             "view_mask": boolean(views), "image_mask": boolean(np.zeros_like(views))}
    admitted = boolean(gate.any(-1))
    frame_ids = torch.as_tensor(ids.astype(np.int64, copy=True), device=device)[None]
    return batch, admitted, frame_ids


class CausalMeshCandidate:
    """Eager verification adapter; no new parameters or copied generator logic."""

    def __init__(self, model, *, schema: JointSchema, calibration_frames=8,
                 steps=8, noise_seed=97):
        if schema is not JointSchema.COCO17_SURFACE or model.cfg.observation_joints != 17:
            raise ValueError("candidate needs explicitly COCO17 model conditions")
        if type(steps) is not int or steps < 1:
            raise ValueError("positive integer Euler steps required")
        self.model = model
        self.shape = PrefixSharedShape(model.encoder.shape_head, calibration_frames)
        self.steps = steps
        self.noise_seed = noise_seed

    def step(self, batch, frame_ids, sequence_ids, admitted, state=None):
        if admitted.dtype != torch.bool or admitted.shape != batch["xy"].shape[:2]:
            raise ValueError("shape admission must be explicit boolean B,T")
        shape_state = state.shape if state is not None else None
        hidden = state.hidden if state is not None else None
        encoded, features = encode_with_global_features(self.model.encoder, batch, hidden=hidden)
        if (admitted & ~encoded["frame_observed"]).any():
            raise ValueError("missing current observations cannot be shape evidence")
        shape = self.shape(features, admitted, frame_ids, sequence_ids, shape_state)
        b, t = batch["xy"].shape[:2]
        if state is None:
            generator = torch.Generator(device=features.device).manual_seed(self.noise_seed)
            noise = torch.randn(b, 1, 52, 6, generator=generator, device=features.device,
                                dtype=features.dtype)
        else:
            if state.noise_seed != self.noise_seed:
                raise ValueError("carried noise seed changed; reset explicitly")
            noise = state.initial_noise
            if noise.shape != (b, 1, 52, 6) or noise.device != features.device or noise.dtype != features.dtype:
                raise ValueError("carried noise batch/device/dtype changed")
        readout = dict(encoded, betas=shape.betas, shape_available=shape.ready)
        prediction = self.model.transport(readout, steps=self.steps,
                                          initial=noise.expand(-1, t, -1, -1))
        finite = torch.isfinite(shape.betas).all(-1) & torch.isfinite(prediction["translation"]).all(-1)
        rotation_ok = prediction["rotation_valid"].all(-1)
        available = encoded["frame_observed"] & shape.ready & finite & rotation_ok
        status = torch.full_like(available, OutputStatus.AVAILABLE, dtype=torch.uint8)
        status[~rotation_ok] = OutputStatus.INVALID_ROTATION
        status[~finite] = OutputStatus.NONFINITE_PARAMETERS
        status[~shape.ready] = OutputStatus.CALIBRATION_PENDING
        status[~encoded["frame_observed"]] = OutputStatus.NO_CURRENT_OBSERVATION
        # Keep internal extrapolation explicitly separate from published parameters.
        published_rotation = torch.where(available[..., None, None, None], prediction["rotations"], torch.nan)
        published_translation = torch.where(available[..., None], prediction["translation"], torch.nan)
        tensors = {"betas": shape.betas, "shape_ready": shape.ready,
                   "available": available, "status": status, "frame_ids": frame_ids.clone(),
                   "frame_observed": encoded["frame_observed"], "shape_admitted": admitted.clone(),
                   "rotations": published_rotation, "translation_m": published_translation,
                   "raw_rotations_diagnostic": prediction["rotations"],
                   "raw_translation_diagnostic_m": prediction["translation"],
                   "raw_state_diagnostic": prediction["state"],
                   "rotation_valid": prediction["rotation_valid"]}
        return CandidateReadout(tensors, CandidateState(shape.state, encoded["hidden"], noise, self.noise_seed))


def decode_available(decoder, tensors, regressor):
    """One shared beta per sequence, actual body and surface points, no prefix fill."""
    available = tensors["available"]
    vertices = tensors["translation_m"].new_full((*available.shape, 6890, 3), torch.nan)
    joints = vertices.new_full((*available.shape, 52, 3), torch.nan)
    coco = vertices.new_full((*available.shape, 17, 3), torch.nan)
    if available.any():
        decoded = decoder(tensors["rotations"][available][:, None], tensors["betas"][available],
                          tensors["translation_m"][available][:, None])
        vertices[available] = decoded["vertices"][:, 0]
        joints[available] = decoded["joints"][:, 0]
        coco[available] = regress_coco17_torch(decoded["vertices"][:, 0], regressor)
    return {"vertices": vertices, "kinematic_joints": joints, "coco17_surface_m": coco,
            "faces": decoder.faces}
