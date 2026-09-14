"""Quality-gated gait and hand--walker interaction candidates.

This module consumes already reconstructed human joints, a current local
ground plane, and walker line primitives.  It never upgrades PMPose ankles to
heel/toe landmarks and never calls geometric proximity physical contact.
Thresholds are explicit configuration values so experiments can report them.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


def _vector3(value: Any, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must be a finite 3-vector")
    return vector


def signed_plane_height_mm(point: Any, normal: Any, offset_mm: float) -> float:
    point = _vector3(point, "point")
    normal = _vector3(normal, "normal")
    normal /= np.linalg.norm(normal)
    if not math.isfinite(float(offset_mm)):
        raise ValueError("offset_mm must be finite")
    return float(normal @ point + offset_mm)


def point_segment_distance_mm(point: Any, start: Any, end: Any) -> tuple[float, np.ndarray, float]:
    point = _vector3(point, "point")
    start = _vector3(start, "start")
    end = _vector3(end, "end")
    direction = end - start
    length2 = float(direction @ direction)
    if length2 < 1e-9:
        raise ValueError("handle segment must have non-zero length")
    fraction = float(np.clip(((point - start) @ direction) / length2, 0.0, 1.0))
    closest = start + fraction * direction
    return float(np.linalg.norm(point - closest)), closest, fraction


@dataclass(frozen=True)
class GaitCriteria:
    minimum_window_frames: int = 8
    minimum_valid_fraction: float = 0.75
    minimum_ankle_motion_mm: float = 35.0
    minimum_interankle_range_mm: float = 25.0
    alternation_correlation_maximum: float = -0.15


def classify_walking_window(
    timestamps_s: np.ndarray,
    left_ankle_left_camera_mm: np.ndarray,
    right_ankle_left_camera_mm: np.ndarray,
    valid: np.ndarray,
    criteria: GaitCriteria = GaitCriteria(),
) -> dict[str, Any]:
    """Classify relative alternating leg motion as a walking candidate.

    The camera is walker-mounted, so the test uses ankle trajectories relative
    to their midpoint.  It does not require a fixed world origin and does not
    mistake camera translation for patient translation.
    """
    times = np.asarray(timestamps_s, dtype=np.float64)
    left = np.asarray(left_ankle_left_camera_mm, dtype=np.float64)
    right = np.asarray(right_ankle_left_camera_mm, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    count = len(times)
    if times.shape != (count,) or left.shape != (count, 3) or right.shape != (count, 3) or valid.shape != (count,):
        raise ValueError("walking-window arrays must have aligned lengths")
    finite = valid & np.all(np.isfinite(left), axis=1) & np.all(np.isfinite(right), axis=1) & np.isfinite(times)
    reasons: list[str] = []
    if count < criteria.minimum_window_frames:
        reasons.append("window_too_short")
    valid_fraction = float(finite.mean()) if count else 0.0
    if valid_fraction < criteria.minimum_valid_fraction:
        reasons.append("insufficient_valid_ankle_fraction")
    indices = np.flatnonzero(finite)
    if len(indices) < 3:
        return {
            "status": "unavailable", "state": "unavailable", "reasons": reasons or ["insufficient_valid_ankles"],
            "metrics": {"valid_fraction": valid_fraction},
        }
    left, right = left[indices], right[indices]
    midpoint = 0.5 * (left + right)
    left_relative, right_relative = left - midpoint, right - midpoint
    left_path = float(np.sum(np.linalg.norm(np.diff(left_relative, axis=0), axis=1)))
    right_path = float(np.sum(np.linalg.norm(np.diff(right_relative, axis=0), axis=1)))
    interankle = np.linalg.norm(left - right, axis=1)
    interankle_range = float(np.ptp(interankle))
    # Estimate the dominant relative leg-motion axis in this window.  Using
    # speed magnitudes would be wrong: two alternating legs often have similar
    # speed magnitudes and therefore positive correlation.  Signed positions
    # along the dominant inter-ankle axis preserve the opposition.
    centered_separation = (left - right) - np.mean(left - right, axis=0)
    _, singular, vt = np.linalg.svd(centered_separation, full_matrices=False)
    motion_axis = vt[0] if len(singular) and singular[0] > 1e-9 else np.asarray([1.0, 0.0, 0.0])
    left_phase = left_relative @ motion_axis
    right_phase = right_relative @ motion_axis
    if np.std(left_phase) < 1e-9 or np.std(right_phase) < 1e-9:
        alternation = 0.0
    else:
        alternation = float(np.corrcoef(left_phase, right_phase)[0, 1])
    motion_ok = max(left_path, right_path) >= criteria.minimum_ankle_motion_mm
    separation_ok = interankle_range >= criteria.minimum_interankle_range_mm
    alternation_ok = alternation <= criteria.alternation_correlation_maximum
    if reasons:
        state = "unavailable"
    elif motion_ok and separation_ok and alternation_ok:
        state = "walking_candidate"
    elif not motion_ok and not separation_ok:
        state = "stationary_candidate"
    else:
        state = "motion_ambiguous"
    return {
        "status": "candidate" if state != "unavailable" else "unavailable",
        "state": state,
        "reasons": reasons,
        "metrics": {
            "valid_fraction": valid_fraction,
            "left_relative_path_mm": left_path,
            "right_relative_path_mm": right_path,
            "interankle_range_mm": interankle_range,
            "ankle_motion_correlation": alternation,
        },
        "criteria": criteria.__dict__,
        "interpretation": (
            "Relative alternating-ankle motion candidate. COCO ankles do not justify heel strike, toe off, "
            "clinical gait phase, or a diagnosis."
        ),
    }


@dataclass(frozen=True)
class FootGroundCriteria:
    maximum_near_plane_height_mm: float = 80.0
    maximum_tangential_speed_mm_s: float = 120.0


def foot_ground_candidate(
    ankle_left_camera_mm: Any,
    ankle_velocity_left_camera_mm_s: Any,
    plane_normal_left_camera: Any,
    plane_offset_mm: float,
    *,
    ground_status: str,
    criteria: FootGroundCriteria = FootGroundCriteria(),
) -> dict[str, Any]:
    if ground_status != "direct":
        return {"status": "unavailable", "reasons": ["requires_current_direct_ground_plane"]}
    ankle = _vector3(ankle_left_camera_mm, "ankle")
    velocity = _vector3(ankle_velocity_left_camera_mm_s, "ankle_velocity")
    normal = _vector3(plane_normal_left_camera, "plane_normal")
    normal /= np.linalg.norm(normal)
    height = signed_plane_height_mm(ankle, normal, plane_offset_mm)
    tangent_velocity = velocity - float(velocity @ normal) * normal
    tangent_speed = float(np.linalg.norm(tangent_velocity))
    near = abs(height) <= criteria.maximum_near_plane_height_mm
    slow = tangent_speed <= criteria.maximum_tangential_speed_mm_s
    return {
        "status": "candidate",
        "state": "near_static_ankle_candidate" if near and slow else "not_near_static_ankle",
        "ankle_signed_height_mm": height,
        "ankle_tangential_speed_mm_s": tangent_speed,
        "criteria": criteria.__dict__,
        "interpretation": "Ankle proxy only; not shoe-sole contact, stance, heel strike, or toe off.",
    }


@dataclass(frozen=True)
class HandHandleCriteria:
    maximum_surface_distance_mm: float = 60.0
    maximum_relative_speed_mm_s: float = 180.0
    distance_softness_mm: float = 15.0
    speed_softness_mm_s: float = 50.0


def _logistic_margin(value: float, limit: float, softness: float) -> float:
    softness = max(float(softness), 1e-6)
    exponent = np.clip((value - limit) / softness, -60.0, 60.0)
    return float(1.0 / (1.0 + np.exp(exponent)))


def hand_handle_proximity_candidate(
    wrist_left_camera_mm: Any,
    wrist_velocity_left_camera_mm_s: Any,
    handle_start_left_camera_mm: Any,
    handle_end_left_camera_mm: Any,
    handle_velocity_left_camera_mm_s: Any,
    *,
    handle_radius_mm: float,
    walker_geometry_status: str,
    criteria: HandHandleCriteria = HandHandleCriteria(),
) -> dict[str, Any]:
    if walker_geometry_status != "candidate":
        return {"status": "unavailable", "reasons": ["walker_handle_geometry_unavailable"]}
    wrist = _vector3(wrist_left_camera_mm, "wrist")
    wrist_velocity = _vector3(wrist_velocity_left_camera_mm_s, "wrist_velocity")
    handle_velocity = _vector3(handle_velocity_left_camera_mm_s, "handle_velocity")
    centerline_distance, closest, fraction = point_segment_distance_mm(
        wrist, handle_start_left_camera_mm, handle_end_left_camera_mm
    )
    surface_distance = max(0.0, centerline_distance - float(handle_radius_mm))
    relative_speed = float(np.linalg.norm(wrist_velocity - handle_velocity))
    distance_probability = _logistic_margin(
        surface_distance, criteria.maximum_surface_distance_mm, criteria.distance_softness_mm
    )
    speed_probability = _logistic_margin(
        relative_speed, criteria.maximum_relative_speed_mm_s, criteria.speed_softness_mm_s
    )
    score = distance_probability * speed_probability
    return {
        "status": "candidate",
        "state": "hand_handle_proximity_candidate" if score >= 0.5 else "proximity_not_supported",
        "proximity_score_0_1": score,
        "surface_distance_mm": surface_distance,
        "relative_speed_mm_s": relative_speed,
        "closest_handle_point_left_camera_mm": closest.tolist(),
        "closest_segment_fraction": fraction,
        "criteria": criteria.__dict__,
        "interpretation": (
            "Probability-like geometric proximity score only. It is not tactile contact or load support, and "
            "must not hard-snap the wrist to the handle."
        ),
    }
