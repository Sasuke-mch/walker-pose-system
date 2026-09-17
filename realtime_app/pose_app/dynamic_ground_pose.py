"""Ground-fixed camera pose helpers for the controlled two-stage walker motion.

The deployable fallback in this module uses the experiment's physical contract:
the walker is constrained to planar motion and both feet are stationary during
Stage 2.  Two labelled feet then determine planar yaw and horizontal camera
translation.  Camera height, roll and pitch remain tied to the measured static
ground reference; the solver deliberately does not claim unconstrained SE(3).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import cv2
import numpy as np

from .two_stage_motion import WorldPose


def _point3(value: Any, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return point


def _rotation_z(angle_rad: float) -> np.ndarray:
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    return np.asarray(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rotation_angle_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    relative = np.asarray(rotation_a) @ np.asarray(rotation_b).T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


@dataclass(frozen=True)
class PlanarFootPoseCriteria:
    minimum_foot_separation_mm: float = 90.0
    maximum_foot_separation_error_mm: float = 140.0
    maximum_translation_step_mm: float = 90.0
    maximum_yaw_step_deg: float = 7.0
    smoothing_gain: float = 0.45

    def __post_init__(self) -> None:
        if self.minimum_foot_separation_mm <= 0 or self.maximum_foot_separation_error_mm <= 0:
            raise ValueError("foot separation thresholds must be positive")
        if self.maximum_translation_step_mm <= 0 or self.maximum_yaw_step_deg <= 0:
            raise ValueError("pose step thresholds must be positive")
        if not 0.0 < self.smoothing_gain <= 1.0:
            raise ValueError("smoothing_gain must be in (0, 1]")


@dataclass(frozen=True)
class FullFootPlanePoseCriteria:
    minimum_foot_separation_mm: float = 90.0
    maximum_foot_separation_error_mm: float = 140.0
    maximum_translation_step_mm: float = 120.0
    maximum_rotation_step_deg: float = 10.0
    maximum_plane_offset_step_mm: float = 100.0
    smoothing_gain: float = 0.45

    def __post_init__(self) -> None:
        if min(
            self.minimum_foot_separation_mm,
            self.maximum_foot_separation_error_mm,
            self.maximum_translation_step_mm,
            self.maximum_rotation_step_deg,
            self.maximum_plane_offset_step_mm,
        ) <= 0.0:
            raise ValueError("full foot-plane pose thresholds must be positive")
        if not 0.0 < self.smoothing_gain <= 1.0:
            raise ValueError("smoothing_gain must be in (0, 1]")


@dataclass(frozen=True)
class RotationFootPoseCriteria:
    maximum_translation_step_mm: float = 120.0
    maximum_rotation_step_deg: float = 5.0
    maximum_foot_separation_error_mm: float = 140.0
    maximum_two_foot_translation_disagreement_mm: float = 60.0
    maximum_recovery_gap_frames: int = 4
    maximum_single_foot_prediction_error_mm: float = 80.0
    minimum_single_foot_prediction_advantage_mm: float = 20.0
    smoothing_gain: float = 0.45

    def __post_init__(self) -> None:
        if min(
            self.maximum_translation_step_mm,
            self.maximum_rotation_step_deg,
            self.maximum_foot_separation_error_mm,
            self.maximum_two_foot_translation_disagreement_mm,
            self.maximum_single_foot_prediction_error_mm,
            self.minimum_single_foot_prediction_advantage_mm,
        ) <= 0.0:
            raise ValueError("rotation-foot pose thresholds must be positive")
        if self.maximum_recovery_gap_frames < 1:
            raise ValueError("maximum_recovery_gap_frames must be positive")
        if not 0.0 < self.smoothing_gain <= 1.0:
            raise ValueError("smoothing_gain must be in (0, 1]")


def solve_pose_from_relative_rotation_and_feet(
    previous_pose: WorldPose,
    rotation_current_from_previous: Any,
    observed_feet_camera_mm: Mapping[str, Any],
    anchors_world_mm: Mapping[str, Any],
    criteria: RotationFootPoseCriteria = RotationFootPoseCriteria(),
    *,
    elapsed_frames_since_update: int = 1,
    translation_prediction_world_mm: Any | None = None,
    allow_single_foot_recovery: bool = False,
) -> tuple[WorldPose | None, dict[str, Any]]:
    """Fuse a background rotation edge with two stationary metric feet.

    The visual edge supplies all three rotational degrees of freedom.  Given
    that rotation, either stationary foot supplies XYZ translation; using both
    provides a separation-consistency check and an averaged translation.
    """

    if any(name not in observed_feet_camera_mm or name not in anchors_world_mm for name in ("left", "right")):
        return None, {"status": "unavailable", "reason": "both_labelled_feet_required"}
    relative = np.asarray(rotation_current_from_previous, dtype=np.float64)
    if relative.shape != (3, 3) or not np.all(np.isfinite(relative)):
        return None, {"status": "unavailable", "reason": "valid_relative_rotation_required"}
    if np.linalg.norm(relative.T @ relative - np.eye(3), ord="fro") > 1e-3 or np.linalg.det(relative) < 0.999:
        return None, {"status": "rejected", "reason": "relative_rotation_not_proper"}
    observed = {
        name: _point3(observed_feet_camera_mm[name], f"{name}_foot_camera")
        for name in ("left", "right")
    }
    anchors = {
        name: _point3(anchors_world_mm[name], f"{name}_foot_anchor")
        for name in ("left", "right")
    }
    observed_separation = float(np.linalg.norm(observed["right"] - observed["left"]))
    anchor_separation = float(np.linalg.norm(anchors["right"] - anchors["left"]))
    separation_error = abs(observed_separation - anchor_separation)
    if separation_error > criteria.maximum_foot_separation_error_mm:
        return None, {
            "status": "rejected", "reason": "foot_separation_inconsistent",
            "foot_separation_error_mm": separation_error,
        }
    candidate_rotation = previous_pose.rotation_world_from_camera @ relative.T
    translations = np.asarray([
        anchors[name] - candidate_rotation @ observed[name] for name in ("left", "right")
    ])
    if elapsed_frames_since_update < 1:
        raise ValueError("elapsed_frames_since_update must be positive")
    recovery_gap_frames = min(elapsed_frames_since_update, criteria.maximum_recovery_gap_frames)
    effective_translation_limit = criteria.maximum_translation_step_mm * recovery_gap_frames
    translation_disagreement = float(np.linalg.norm(translations[0] - translations[1]))
    translation_source = "two_foot_mean"
    single_foot_audit: dict[str, Any] = {}
    if translation_disagreement > criteria.maximum_two_foot_translation_disagreement_mm:
        prediction = None
        if translation_prediction_world_mm is not None:
            prediction = _point3(translation_prediction_world_mm, "translation_prediction_world")
        prediction_errors = None if prediction is None else np.linalg.norm(translations - prediction, axis=1)
        ordered = None if prediction_errors is None else np.argsort(prediction_errors)
        recoverable = bool(
            allow_single_foot_recovery
            and ordered is not None
            and prediction_errors[ordered[0]] <= criteria.maximum_single_foot_prediction_error_mm * recovery_gap_frames
            and prediction_errors[ordered[1]] - prediction_errors[ordered[0]]
            >= criteria.minimum_single_foot_prediction_advantage_mm
        )
        if not recoverable:
            return None, {
                "status": "rejected", "reason": "two_foot_translation_disagreement",
                "two_foot_translation_disagreement_mm": translation_disagreement,
                "maximum_two_foot_translation_disagreement_mm": (
                    criteria.maximum_two_foot_translation_disagreement_mm
                ),
                "left_foot_translation_world_mm": translations[0].tolist(),
                "right_foot_translation_world_mm": translations[1].tolist(),
                "translation_prediction_world_mm": None if prediction is None else prediction.tolist(),
                "per_foot_prediction_error_mm": None if prediction_errors is None else {
                    "left": float(prediction_errors[0]), "right": float(prediction_errors[1]),
                },
                "single_foot_recovery_allowed": bool(allow_single_foot_recovery),
                "foot_separation_error_mm": separation_error,
            }
        selected_index = int(ordered[0])
        selected_name = ("left", "right")[selected_index]
        candidate_translation = translations[selected_index]
        translation_source = f"single_{selected_name}_foot_motion_prior_recovery"
        single_foot_audit = {
            "single_foot_recovery_applied": True,
            "selected_translation_foot": selected_name,
            "translation_prediction_world_mm": prediction.tolist(),
            "per_foot_prediction_error_mm": {
                "left": float(prediction_errors[0]), "right": float(prediction_errors[1]),
            },
            "minimum_single_foot_prediction_advantage_mm": criteria.minimum_single_foot_prediction_advantage_mm,
        }
    else:
        candidate_translation = translations.mean(axis=0)
    translation_step = float(np.linalg.norm(
        candidate_translation - previous_pose.translation_world_from_camera_mm
    ))
    rotation_step = rotation_angle_deg(candidate_rotation, previous_pose.rotation_world_from_camera)
    if translation_step > effective_translation_limit or rotation_step > criteria.maximum_rotation_step_deg:
        return None, {
            "status": "rejected", "reason": "rotation_foot_pose_step_exceeds_realtime_limit",
            "translation_step_mm": translation_step, "rotation_step_deg": rotation_step,
            "elapsed_frames_since_update": elapsed_frames_since_update,
            "effective_maximum_translation_step_mm": effective_translation_limit,
            "foot_separation_error_mm": separation_error,
            **single_foot_audit,
        }
    gain = criteria.smoothing_gain
    relative_world = candidate_rotation @ previous_pose.rotation_world_from_camera.T
    relative_vector, _ = cv2.Rodrigues(relative_world)
    filtered_delta, _ = cv2.Rodrigues(gain * relative_vector)
    filtered_rotation = filtered_delta @ previous_pose.rotation_world_from_camera
    filtered_translation = previous_pose.translation_world_from_camera_mm + gain * (
        candidate_translation - previous_pose.translation_world_from_camera_mm
    )
    filtered = WorldPose(filtered_rotation, filtered_translation)
    residuals = [
        float(np.linalg.norm(filtered.transform(observed[name]) - anchors[name]))
        for name in ("left", "right")
    ]
    return filtered, {
        "status": "accepted", "source": "stage2_background_rotation_plus_two_stationary_feet_full_se3",
        "translation_step_mm": translation_step, "rotation_step_deg": rotation_step,
        "foot_separation_error_mm": separation_error,
        "two_foot_translation_disagreement_mm": translation_disagreement,
        "maximum_two_foot_translation_disagreement_mm": criteria.maximum_two_foot_translation_disagreement_mm,
        "left_foot_translation_world_mm": translations[0].tolist(),
        "right_foot_translation_world_mm": translations[1].tolist(),
        "translation_source": translation_source,
        "elapsed_frames_since_update": elapsed_frames_since_update,
        "effective_maximum_translation_step_mm": effective_translation_limit,
        "post_filter_foot_residual_median_mm": float(np.median(residuals)),
        "post_filter_foot_residual_maximum_mm": float(np.max(residuals)),
        "constraints": (
            "visual background rotation plus two stationary labelled metric feet"
            if translation_source == "two_foot_mean" else
            "visual background rotation plus one motion-prior-selected stationary metric foot"
        ),
        **single_foot_audit,
    }


def solve_full_pose_from_feet_and_plane(
    previous_pose: WorldPose,
    observed_feet_camera_mm: Mapping[str, Any],
    anchors_world_mm: Mapping[str, Any],
    plane_normal_camera: Any,
    plane_offset_mm: float,
    criteria: FullFootPlanePoseCriteria = FullFootPlanePoseCriteria(),
) -> tuple[WorldPose | None, dict[str, Any]]:
    """Recover a full pose from two labelled foot anchors and a ground plane.

    A plane normal alone leaves yaw free; two feet alone leave rotation about
    their connecting line free.  Together, when the feet are separated and the
    foot line is not parallel to the normal, they define a complete orthonormal
    frame.  The plane offset supplies camera height, while the two anchors
    supply horizontal translation.
    """

    if any(name not in observed_feet_camera_mm or name not in anchors_world_mm for name in ("left", "right")):
        return None, {"status": "unavailable", "reason": "both_labelled_feet_required"}
    normal = _point3(plane_normal_camera, "plane_normal_camera")
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-9 or not math.isfinite(float(plane_offset_mm)) or plane_offset_mm <= 0.0:
        return None, {"status": "unavailable", "reason": "valid_ground_plane_required"}
    normal /= normal_norm
    left_camera = _point3(observed_feet_camera_mm["left"], "left_foot_camera")
    right_camera = _point3(observed_feet_camera_mm["right"], "right_foot_camera")
    left_anchor = _point3(anchors_world_mm["left"], "left_foot_anchor")
    right_anchor = _point3(anchors_world_mm["right"], "right_foot_anchor")

    camera_delta = right_camera - left_camera
    camera_tangent = camera_delta - normal * float(camera_delta @ normal)
    world_up = np.asarray((0.0, 0.0, 1.0))
    world_delta = right_anchor - left_anchor
    world_tangent = world_delta - world_up * float(world_delta @ world_up)
    camera_separation = float(np.linalg.norm(camera_tangent))
    anchor_separation = float(np.linalg.norm(world_tangent))
    if min(camera_separation, anchor_separation) < criteria.minimum_foot_separation_mm:
        return None, {"status": "rejected", "reason": "feet_too_close_for_full_pose"}
    separation_error = abs(camera_separation - anchor_separation)
    if separation_error > criteria.maximum_foot_separation_error_mm:
        return None, {
            "status": "rejected", "reason": "foot_separation_inconsistent",
            "foot_separation_error_mm": separation_error,
        }

    camera_x = camera_tangent / camera_separation
    camera_y = np.cross(normal, camera_x)
    camera_y /= np.linalg.norm(camera_y)
    camera_x = np.cross(camera_y, normal)
    world_x = world_tangent / anchor_separation
    world_y = np.cross(world_up, world_x)
    world_y /= np.linalg.norm(world_y)
    world_x = np.cross(world_y, world_up)
    camera_basis = np.column_stack((camera_x, camera_y, normal))
    world_basis = np.column_stack((world_x, world_y, world_up))
    candidate_rotation = world_basis @ camera_basis.T

    observed_center = 0.5 * (left_camera + right_camera)
    anchor_center = 0.5 * (left_anchor + right_anchor)
    candidate_translation = anchor_center - candidate_rotation @ observed_center
    candidate_translation[2] = float(plane_offset_mm) / normal_norm
    candidate = WorldPose(candidate_rotation, candidate_translation)
    translation_step = float(np.linalg.norm(
        candidate.translation_world_from_camera_mm - previous_pose.translation_world_from_camera_mm
    ))
    rotation_step = rotation_angle_deg(
        candidate.rotation_world_from_camera, previous_pose.rotation_world_from_camera
    )
    plane_offset_step = abs(
        float(candidate.translation_world_from_camera_mm[2] - previous_pose.translation_world_from_camera_mm[2])
    )
    if (
        translation_step > criteria.maximum_translation_step_mm
        or rotation_step > criteria.maximum_rotation_step_deg
        or plane_offset_step > criteria.maximum_plane_offset_step_mm
    ):
        return None, {
            "status": "rejected", "reason": "full_pose_step_exceeds_realtime_limit",
            "translation_step_mm": translation_step, "rotation_step_deg": rotation_step,
            "plane_offset_step_mm": plane_offset_step,
        }

    gain = criteria.smoothing_gain
    relative_rotation = candidate.rotation_world_from_camera @ previous_pose.rotation_world_from_camera.T
    relative_vector, _ = cv2.Rodrigues(relative_rotation)
    filtered_delta, _ = cv2.Rodrigues(gain * relative_vector)
    filtered_rotation = filtered_delta @ previous_pose.rotation_world_from_camera
    filtered_translation = previous_pose.translation_world_from_camera_mm + gain * (
        candidate.translation_world_from_camera_mm - previous_pose.translation_world_from_camera_mm
    )
    filtered = WorldPose(filtered_rotation, filtered_translation)
    foot_residuals = [
        float(np.linalg.norm(filtered.transform(observed) - anchor))
        for observed, anchor in ((left_camera, left_anchor), (right_camera, right_anchor))
    ]
    mapped_normal = filtered.rotation_world_from_camera @ normal
    normal_residual = math.degrees(math.acos(float(np.clip(mapped_normal @ world_up, -1.0, 1.0))))
    return filtered, {
        "status": "accepted", "source": "stage2_two_feet_plus_direct_ground_plane_full_se3",
        "translation_step_mm": translation_step, "rotation_step_deg": rotation_step,
        "plane_offset_step_mm": plane_offset_step,
        "foot_separation_error_mm": separation_error,
        "post_filter_foot_residual_median_mm": float(np.median(foot_residuals)),
        "post_filter_foot_residual_maximum_mm": float(np.max(foot_residuals)),
        "post_filter_plane_normal_residual_deg": normal_residual,
        "constraints": "two stationary labelled feet plus independently observed local ground plane",
        "observability": "full_se3_only_when_both_feet_and_plane_are_valid",
    }


def solve_planar_pose_from_feet(
    reference_pose: WorldPose,
    previous_pose: WorldPose,
    observed_feet_camera_mm: Mapping[str, Any],
    anchors_world_mm: Mapping[str, Any],
    criteria: PlanarFootPoseCriteria = PlanarFootPoseCriteria(),
) -> tuple[WorldPose | None, dict[str, Any]]:
    """Estimate ground-constrained pose from two stationary labelled feet.

    ``reference_pose`` supplies the fixed camera roll, pitch and height.  The
    returned pose changes only world yaw and horizontal translation.
    """

    if any(name not in observed_feet_camera_mm or name not in anchors_world_mm for name in ("left", "right")):
        return None, {"status": "unavailable", "reason": "both_labelled_feet_required"}
    left_camera = _point3(observed_feet_camera_mm["left"], "left_foot_camera")
    right_camera = _point3(observed_feet_camera_mm["right"], "right_foot_camera")
    left_anchor = _point3(anchors_world_mm["left"], "left_foot_anchor")
    right_anchor = _point3(anchors_world_mm["right"], "right_foot_anchor")

    observed_delta_reference = reference_pose.rotation_world_from_camera @ (right_camera - left_camera)
    anchor_delta = right_anchor - left_anchor
    observed_separation = float(np.linalg.norm(observed_delta_reference[:2]))
    anchor_separation = float(np.linalg.norm(anchor_delta[:2]))
    if min(observed_separation, anchor_separation) < criteria.minimum_foot_separation_mm:
        return None, {"status": "rejected", "reason": "feet_too_close_for_planar_yaw"}
    separation_error = abs(observed_separation - anchor_separation)
    if separation_error > criteria.maximum_foot_separation_error_mm:
        return None, {
            "status": "rejected", "reason": "foot_separation_inconsistent",
            "foot_separation_error_mm": separation_error,
        }

    observed_angle = math.atan2(observed_delta_reference[1], observed_delta_reference[0])
    anchor_angle = math.atan2(anchor_delta[1], anchor_delta[0])
    yaw = _wrap_pi(anchor_angle - observed_angle)
    candidate_rotation = _rotation_z(yaw) @ reference_pose.rotation_world_from_camera
    observed_center = 0.5 * (left_camera + right_camera)
    anchor_center = 0.5 * (left_anchor + right_anchor)
    rotated_center = candidate_rotation @ observed_center
    candidate_translation = previous_pose.translation_world_from_camera_mm.copy()
    candidate_translation[:2] = anchor_center[:2] - rotated_center[:2]
    candidate_translation[2] = reference_pose.translation_world_from_camera_mm[2]
    candidate = WorldPose(candidate_rotation, candidate_translation)

    translation_step = float(np.linalg.norm(
        candidate.translation_world_from_camera_mm[:2] - previous_pose.translation_world_from_camera_mm[:2]
    ))
    yaw_step = rotation_angle_deg(candidate.rotation_world_from_camera, previous_pose.rotation_world_from_camera)
    if translation_step > criteria.maximum_translation_step_mm or yaw_step > criteria.maximum_yaw_step_deg:
        return None, {
            "status": "rejected", "reason": "planar_pose_step_exceeds_realtime_limit",
            "translation_step_mm": translation_step, "yaw_step_deg": yaw_step,
        }

    gain = criteria.smoothing_gain
    previous_forward = previous_pose.rotation_world_from_camera[:, 0]
    candidate_forward = candidate.rotation_world_from_camera[:, 0]
    previous_yaw = math.atan2(previous_forward[1], previous_forward[0])
    candidate_yaw = math.atan2(candidate_forward[1], candidate_forward[0])
    filtered_delta_yaw = gain * _wrap_pi(candidate_yaw - previous_yaw)
    filtered_rotation = _rotation_z(filtered_delta_yaw) @ previous_pose.rotation_world_from_camera
    filtered_translation = previous_pose.translation_world_from_camera_mm.copy()
    filtered_translation[:2] += gain * (
        candidate.translation_world_from_camera_mm[:2] - filtered_translation[:2]
    )
    filtered_translation[2] = reference_pose.translation_world_from_camera_mm[2]
    filtered = WorldPose(filtered_rotation, filtered_translation)

    residuals = []
    for observed, anchor in ((left_camera, left_anchor), (right_camera, right_anchor)):
        residuals.append(float(np.linalg.norm(filtered.transform(observed) - anchor)))
    return filtered, {
        "status": "accepted",
        "source": "stage2_two_foot_planar_se2",
        "translation_step_mm": translation_step,
        "yaw_step_deg": yaw_step,
        "foot_separation_error_mm": separation_error,
        "post_filter_foot_residual_median_mm": float(np.median(residuals)),
        "post_filter_foot_residual_maximum_mm": float(np.max(residuals)),
        "constraints": "fixed camera height/roll/pitch; planar yaw+XY only",
    }


def solve_planar_translation_from_feet(
    reference_pose: WorldPose,
    previous_pose: WorldPose,
    observed_feet_camera_mm: Mapping[str, Any],
    anchors_world_mm: Mapping[str, Any],
    criteria: PlanarFootPoseCriteria = PlanarFootPoseCriteria(),
) -> tuple[WorldPose | None, dict[str, Any]]:
    """Estimate only ground XY translation from the stationary two-foot centre.

    This conservative model is intended for straight walker pushes.  It avoids
    turning noisy stereo foot-baseline direction into a spurious camera yaw.
    """
    if any(name not in observed_feet_camera_mm or name not in anchors_world_mm for name in ("left", "right")):
        return None, {"status": "unavailable", "reason": "both_labelled_feet_required"}
    observed = np.asarray([
        _point3(observed_feet_camera_mm["left"], "left_foot_camera"),
        _point3(observed_feet_camera_mm["right"], "right_foot_camera"),
    ])
    anchors = np.asarray([
        _point3(anchors_world_mm["left"], "left_foot_anchor"),
        _point3(anchors_world_mm["right"], "right_foot_anchor"),
    ])
    pre_residuals = [float(np.linalg.norm(previous_pose.transform(point) - anchor)) for point, anchor in zip(observed, anchors)]
    observed_separation = float(np.linalg.norm((reference_pose.rotation_world_from_camera @ (observed[1] - observed[0]))[:2]))
    anchor_separation = float(np.linalg.norm((anchors[1] - anchors[0])[:2]))
    separation_error = abs(observed_separation - anchor_separation)
    if min(observed_separation, anchor_separation) < criteria.minimum_foot_separation_mm:
        return None, {"status": "rejected", "reason": "feet_too_close_for_translation_constraint"}
    if separation_error > criteria.maximum_foot_separation_error_mm:
        return None, {"status": "rejected", "reason": "foot_separation_inconsistent", "foot_separation_error_mm": separation_error}
    rotated_center = reference_pose.rotation_world_from_camera @ observed.mean(axis=0)
    anchor_center = anchors.mean(axis=0)
    candidate_translation = previous_pose.translation_world_from_camera_mm.copy()
    candidate_translation[:2] = anchor_center[:2] - rotated_center[:2]
    candidate_translation[2] = reference_pose.translation_world_from_camera_mm[2]
    step = float(np.linalg.norm(candidate_translation[:2] - previous_pose.translation_world_from_camera_mm[:2]))
    if step > criteria.maximum_translation_step_mm:
        return None, {"status": "rejected", "reason": "planar_translation_step_exceeds_realtime_limit", "translation_step_mm": step}
    gain = criteria.smoothing_gain
    filtered_translation = previous_pose.translation_world_from_camera_mm.copy()
    filtered_translation[:2] += gain * (candidate_translation[:2] - filtered_translation[:2])
    filtered_translation[2] = reference_pose.translation_world_from_camera_mm[2]
    filtered = WorldPose(reference_pose.rotation_world_from_camera, filtered_translation)
    residuals = [float(np.linalg.norm(filtered.transform(point) - anchor)) for point, anchor in zip(observed, anchors)]
    return filtered, {
        "status": "accepted", "source": "stage2_two_foot_planar_translation",
        "translation_step_mm": step, "yaw_step_deg": 0.0,
        "foot_separation_error_mm": separation_error,
        "post_filter_foot_residual_median_mm": float(np.median(residuals)),
        "post_filter_foot_residual_maximum_mm": float(np.max(residuals)),
        "pre_update_foot_residual_median_mm": float(np.median(pre_residuals)),
        "pre_update_foot_residual_maximum_mm": float(np.max(pre_residuals)),
        "constraints": "fixed camera height/roll/pitch/yaw; ground XY translation only",
    }
