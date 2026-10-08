"""Explicit candidate interfaces; never infer semantics from tensor dimensions.

This module does not replace engineering mainline readers or strict gates.
COCO observations are surface landmarks, while SMPL-H rotations act on 52
kinematic nodes. Their shared words (e.g. wrist) do not define an identity map.
"""
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

from realtime_app.pose_app.geometry_input import raw_point_rejection_reason
from realtime_app.pose_app.rotation import model_image_size, model_to_raw_point
from realtime_app.pose_app.smpl_coco_observation import COCO17_NAMES

from .rotations import matrix_to_six, six_to_matrix, validate_rotation_matrices


class JointSchema(Enum):
    COCO17_SURFACE = "coco17_surface_observations"
    MOTIONBERT_H36M17 = "motionbert_h36m17_observations"
    SMPLH52_ROTATIONS = "smplh52_local_rotation_nodes"


MOTIONBERT_H36M17_NAMES = (
    "pelvis", "right_hip", "right_knee", "right_ankle", "left_hip",
    "left_knee", "left_ankle", "torso", "neck", "nose", "head_top",
    "left_shoulder", "left_elbow", "left_wrist", "right_shoulder",
    "right_elbow", "right_wrist",
)


def require_joint_schema(tensor, *, actual: JointSchema, expected: JointSchema):
    """No automatic reindexing, padding, missing head synthesis or wrist substitution."""
    if not isinstance(actual, JointSchema) or not isinstance(expected, JointSchema):
        raise TypeError("joint schemas must be explicit JointSchema values")
    if actual is not expected:
        raise ValueError(f"joint semantics mismatch: {actual.value} -> {expected.value}")
    count = 52 if actual is JointSchema.SMPLH52_ROTATIONS else len(COCO17_NAMES)
    if actual is JointSchema.SMPLH52_ROTATIONS:
        if tuple(tensor.shape[-3:]) != (count, 3, 3):
            raise ValueError("SMPL-H requires 52 local rotation matrices")
    elif tuple(tensor.shape[-2:]) not in ((count, 2), (count, 3)):
        raise ValueError("landmarks require 17 ordered points with 2 or 3 coordinates")
    return tensor


class SixDLayout(Enum):
    PACKED_COLUMNS = "packed_columns"  # ProHMR/HMR2/project
    INTERLEAVED_COLUMNS = "interleaved_columns"  # HuManiFlow/MotionBERT
    PACKED_ROWS = "packed_rows"  # WHAM actual body/root paths


def encode_rotation(matrix, layout: SixDLayout):
    if not isinstance(layout, SixDLayout):
        raise TypeError("6D layout must be explicitly declared")
    validate_rotation_matrices(matrix)
    if layout is SixDLayout.PACKED_COLUMNS:
        return matrix_to_six(matrix)
    if layout is SixDLayout.INTERLEAVED_COLUMNS:
        return matrix[..., :, :2].reshape(*matrix.shape[:-2], 6)
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def decode_rotation(six, layout: SixDLayout):
    if not isinstance(layout, SixDLayout):
        raise TypeError("6D layout must be explicitly declared")
    if six.shape[-1] != 6:
        raise ValueError("6D rotation must end in six coordinates")
    if layout is SixDLayout.INTERLEAVED_COLUMNS:
        pairs = six.reshape(*six.shape[:-1], 3, 2)
        packed = torch.cat((pairs[..., 0], pairs[..., 1]), dim=-1)
    else:
        packed = six
    matrix, valid = six_to_matrix(packed)
    if layout is SixDLayout.PACKED_ROWS:
        matrix = matrix.transpose(-1, -2)
    return matrix, valid


def convert_rotation(six, *, source: SixDLayout, target: SixDLayout):
    matrix, valid = decode_rotation(six, source)
    # Only valid matrices go through the strict encoder; invalid remains NaN.
    output = torch.full_like(six, float("nan"))
    if valid.any():
        output[valid] = encode_rotation(matrix[valid], target)
    elif not isinstance(target, SixDLayout):
        raise TypeError("6D layout must be explicitly declared")
    return output, valid


class MaskMeaning(Enum):
    PRESENT = "true_is_present"
    MISSING = "true_is_missing"  # WHAM preprocessing


def presence_mask(mask, meaning: MaskMeaning):
    if not isinstance(meaning, MaskMeaning):
        raise TypeError("mask polarity must be explicitly declared")
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise TypeError("presence/missing mask must be a boolean tensor")
    return mask.clone() if meaning is MaskMeaning.PRESENT else ~mask


class PixelDomain(Enum):
    RAW = "raw_sensor_pixels"
    MODEL = "rotated_model_pixels"


def length_scale_to_metre(unit: str):
    scales = {"meter": 1.0, "m": 1.0, "millimeter": 0.001, "mm": 0.001}
    if unit not in scales:
        raise ValueError(f"unsupported declared length unit: {unit}")
    return scales[unit]


def metric_coordinates(value, *, camera_kind: str, unit: str, frame: str):
    """Weak-perspective scale/offset cannot be relabeled metric camera XYZ."""
    if camera_kind != "metric_xyz" or frame != "cam0":
        raise ValueError("expected metric XYZ in cam0; weak camera/global frame needs an explicit model")
    points = np.asarray(value, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 3:
        raise ValueError("metric coordinates must end in XYZ")
    return points * length_scale_to_metre(unit)


@dataclass(frozen=True)
class CameraObservations:
    raw_xy: np.ndarray
    image_normalized_xy: np.ndarray
    ray_origins_m: np.ndarray
    ray_directions_cam0: np.ndarray
    point_valid: np.ndarray
    point_reasons: np.ndarray
    ray_valid: np.ndarray
    ray_reasons: np.ndarray
    frame: str = "cam0"
    length_unit: str = "meter"


def camera_observations(points, calibration, *, side: str, domain: PixelDomain,
                        model_rotation: str):
    """Reuse current inverse pixel rotation and calibrated unprojection.

    Image tokens use centered raw pixels divided by min(W,H)/2, independently
    of calibrated rays. Geometric availability is *not* strict acceptance;
    callers must intersect ray_valid with their immutable upstream gate.
    No detector score/association/triangulation gate is recomputed here.
    """
    if side not in ("left", "right") or not isinstance(domain, PixelDomain):
        raise ValueError("explicit physical side and pixel domain required")
    size = calibration.left_image_size if side == "left" else calibration.right_image_size
    model_image_size(*size, model_rotation)  # also validates declared rotation
    raw = np.asarray(points, dtype=np.float64).copy()
    if raw.ndim < 1 or raw.shape[-1] != 2:
        raise ValueError("pixel observations must end in XY")
    if domain is PixelDomain.MODEL:
        raw = np.asarray([model_to_raw_point(*p, *size, model_rotation)
                          for p in raw.reshape(-1, 2)]).reshape(raw.shape)
    reasons = np.asarray([raw_point_rejection_reason(p, size)
                          for p in raw.reshape(-1, 2)], dtype=object).reshape(raw.shape[:-1])
    valid = np.asarray([x is None for x in reasons.flat], dtype=bool).reshape(reasons.shape)
    ray_valid = valid.copy()
    ray_reasons = reasons.copy()
    rays = np.full((*raw.shape[:-1], 3), np.nan, dtype=np.float64)
    origins = np.full_like(rays, np.nan)
    normalized = np.full_like(raw, np.nan)
    R = np.asarray(calibration.R, dtype=np.float64)
    T = np.asarray(calibration.T, dtype=np.float64).reshape(3)
    if (R.shape != (3, 3) or not np.isfinite(R).all() or not np.isfinite(T).all()
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-5, rtol=0)
            or not np.isclose(np.linalg.det(R), 1, atol=1e-5, rtol=0)):
        raise ValueError("invalid cam0-to-cam1 extrinsics")
    scale = length_scale_to_metre(calibration.length_unit)
    if valid.any():
        normalized[valid] = (raw[valid] - (np.asarray(size) - 1) / 2) / (min(size) / 2)
        xy = calibration.undistort_normalized(raw[valid], side)
        direction = np.column_stack((xy, np.ones(len(xy))))
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        # OpenCV may return a finite sentinel for an unconverged fisheye inverse.
        # Report it separately instead of exposing a spurious valid unit ray.
        projection = (calibration.project_left(direction) if side == "left"
                      else calibration.project_right((direction - T) @ R))
        checked = np.isfinite(direction).all(-1) & (np.linalg.norm(projection - raw[valid], axis=-1) < 1e-4)
        flat_indices = np.flatnonzero(valid)
        failed = flat_indices[~checked]
        ray_reasons.flat[failed] = "unprojection_inconsistent"
        ray_valid.flat[failed] = False
        direction = direction[checked]
        if side == "right":
            direction = direction @ R
        rays[ray_valid] = direction
        origins[ray_valid] = np.zeros(3) if side == "left" else -R.T @ T * scale
    return CameraObservations(raw, normalized, origins, rays, valid, reasons,
                              ray_valid, ray_reasons)
