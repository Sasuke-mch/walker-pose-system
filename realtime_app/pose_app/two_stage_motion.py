"""Two-stage motion model for a walker-mounted stereo rig.

The module separates two controlled experimental states from the evidence used
to enter them.  It never infers a usable camera pose from missing odometry and
only offers a translation-only foot-anchor correction; two feet must not be
used to manufacture an unconstrained six-degree-of-freedom pose.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from .local_plane_propagation import STATIC_BACKGROUND_DOMAIN


UNKNOWN = "unknown"
WALKER_STATIC_HUMAN_MOVING = "walker_static_human_moving"
TRANSITION = "transition"
FEET_STATIC_WALKER_MOVING = "feet_static_walker_moving"
SETTLING = "settling"


def _vector3(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite 3-vector")
    return result


def _rotation(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite 3x3 matrix")
    if np.linalg.norm(result.T @ result - np.eye(3), ord="fro") > 1e-6 or abs(np.linalg.det(result) - 1.0) > 1e-6:
        raise ValueError(f"{label} must be a proper rotation")
    return result


@dataclass(frozen=True)
class WorldPose:
    """Rigid transform ``X_world = R_world_from_camera X_camera + t``."""

    rotation_world_from_camera: np.ndarray
    translation_world_from_camera_mm: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation_world_from_camera", _rotation(
            self.rotation_world_from_camera, "rotation_world_from_camera"
        ))
        object.__setattr__(self, "translation_world_from_camera_mm", _vector3(
            self.translation_world_from_camera_mm, "translation_world_from_camera_mm"
        ))

    def transform(self, points_camera_mm: Any) -> np.ndarray:
        points = np.asarray(points_camera_mm, dtype=np.float64)
        if points.shape == (3,):
            return self.rotation_world_from_camera @ points + self.translation_world_from_camera_mm
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("points_camera_mm must be a finite 3-vector or Nx3 array")
        return (self.rotation_world_from_camera @ points.T).T + self.translation_world_from_camera_mm

    def as_mapping(self) -> dict[str, Any]:
        return {
            "coordinate_transform": "world_from_left_camera",
            "length_unit": "millimeter",
            "rotation_world_from_left_camera": self.rotation_world_from_camera.tolist(),
            "translation_world_from_left_camera_mm": self.translation_world_from_camera_mm.tolist(),
        }


def propagate_world_pose(previous: WorldPose, relative_pose: Mapping[str, Any]) -> WorldPose:
    """Propagate a world pose from accepted static-background relative pose.

    The input convention is ``X_current = R_current_from_previous X_previous + q``.
    Consequently the world-from-current transform uses the inverse edge.
    """

    if relative_pose.get("status") != "accepted":
        raise ValueError("relative pose must have status='accepted'")
    if relative_pose.get("feature_domain") != STATIC_BACKGROUND_DOMAIN:
        raise ValueError("relative pose must use independently masked static background")
    if relative_pose.get("ground_region_used_for_motion") is not False:
        raise ValueError("ground-fit pixels must not be used for camera motion")
    rotation_current_from_previous = _rotation(relative_pose.get("R_to_from"), "R_to_from")
    translation_current_from_previous = _vector3(relative_pose.get("q_to_from"), "q_to_from")
    inverse_rotation = rotation_current_from_previous.T
    rotation_world_from_current = previous.rotation_world_from_camera @ inverse_rotation
    translation_world_from_current = (
        previous.translation_world_from_camera_mm
        - rotation_world_from_current @ translation_current_from_previous
    )
    return WorldPose(rotation_world_from_current, translation_world_from_current)


def foot_anchor_residuals_mm(
    pose: WorldPose,
    observed_feet_camera_mm: Mapping[str, Any],
    anchors_world_mm: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure left/right foot-anchor residuals without altering the pose."""

    names = ("left", "right")
    if any(name not in observed_feet_camera_mm or name not in anchors_world_mm for name in names):
        return {"status": "unavailable", "reasons": ["both_foot_anchors_required"]}
    residuals: dict[str, list[float]] = {}
    vectors = []
    for name in names:
        observed_world = pose.transform(_vector3(observed_feet_camera_mm[name], f"{name}_foot_camera"))
        anchor = _vector3(anchors_world_mm[name], f"{name}_foot_anchor_world")
        vector = anchor - observed_world
        vectors.append(vector)
        residuals[name] = vector.tolist()
    norms = np.linalg.norm(np.asarray(vectors), axis=1)
    return {
        "status": "available",
        "residual_vectors_world_mm": residuals,
        "residual_median_mm": float(np.median(norms)),
        "residual_maximum_mm": float(np.max(norms)),
        "mean_translation_correction_world_mm": np.mean(vectors, axis=0).tolist(),
    }


def apply_soft_foot_translation_correction(
    pose: WorldPose,
    observed_feet_camera_mm: Mapping[str, Any],
    anchors_world_mm: Mapping[str, Any],
    *,
    gain: float,
    maximum_pre_correction_residual_mm: float,
) -> tuple[WorldPose | None, dict[str, Any]]:
    """Apply a bounded translation-only correction from two stationary feet."""

    if not math.isfinite(gain) or not 0.0 <= gain <= 1.0:
        raise ValueError("gain must be finite and in [0, 1]")
    if not math.isfinite(maximum_pre_correction_residual_mm) or maximum_pre_correction_residual_mm <= 0:
        raise ValueError("maximum_pre_correction_residual_mm must be positive")
    audit = foot_anchor_residuals_mm(pose, observed_feet_camera_mm, anchors_world_mm)
    audit["correction_kind"] = "translation_only_two_foot_soft_constraint"
    audit["gain"] = float(gain)
    if audit["status"] != "available":
        return None, audit
    if audit["residual_maximum_mm"] > maximum_pre_correction_residual_mm:
        audit["status"] = "rejected"
        audit["reasons"] = ["foot_anchor_residual_exceeds_limit"]
        return None, audit
    delta = gain * np.asarray(audit["mean_translation_correction_world_mm"], dtype=np.float64)
    corrected = WorldPose(
        pose.rotation_world_from_camera,
        pose.translation_world_from_camera_mm + delta,
    )
    after = foot_anchor_residuals_mm(corrected, observed_feet_camera_mm, anchors_world_mm)
    audit["status"] = "corrected"
    audit["applied_translation_world_mm"] = delta.tolist()
    audit["post_correction_residual_median_mm"] = after["residual_median_mm"]
    audit["post_correction_residual_maximum_mm"] = after["residual_maximum_mm"]
    audit["interpretation"] = (
        "Two stationary feet provide a bounded translation correction only. "
        "They do not independently determine a complete six-degree-of-freedom camera pose."
    )
    return corrected, audit


@dataclass(frozen=True)
class TwoStageCriteria:
    maximum_static_translation_mm: float = 3.0
    maximum_static_rotation_deg: float = 0.25
    minimum_moving_translation_mm: float = 8.0
    minimum_moving_rotation_deg: float = 0.7
    minimum_human_relative_speed_mm_s: float = 30.0
    maximum_stationary_feet_residual_mm: float = 35.0
    confirmation_frames: int = 3

    def __post_init__(self) -> None:
        values = (
            self.maximum_static_translation_mm,
            self.maximum_static_rotation_deg,
            self.minimum_moving_translation_mm,
            self.minimum_moving_rotation_deg,
            self.minimum_human_relative_speed_mm_s,
            self.maximum_stationary_feet_residual_mm,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("two-stage thresholds must be finite and non-negative")
        if self.minimum_moving_translation_mm <= self.maximum_static_translation_mm:
            raise ValueError("moving translation threshold must exceed static threshold")
        if self.minimum_moving_rotation_deg <= self.maximum_static_rotation_deg:
            raise ValueError("moving rotation threshold must exceed static threshold")
        if self.confirmation_frames < 1:
            raise ValueError("confirmation_frames must be positive")


@dataclass(frozen=True)
class MotionEvidence:
    background_pose_status: str
    camera_translation_mm: float | None
    camera_rotation_deg: float | None
    human_relative_speed_mm_s: float | None
    feet_anchor_residual_maximum_mm: float | None


class TwoStageMotionTracker:
    """Hysteretic state tracker that fails closed on missing background motion."""

    def __init__(self, criteria: TwoStageCriteria = TwoStageCriteria()) -> None:
        self.criteria = criteria
        self.active_state = UNKNOWN
        self.pending_state: str | None = None
        self.pending_count = 0

    @staticmethod
    def _finite(value: float | None) -> bool:
        return value is not None and math.isfinite(float(value))

    def update(self, evidence: MotionEvidence) -> dict[str, Any]:
        c = self.criteria
        if evidence.background_pose_status != "accepted" or not self._finite(evidence.camera_translation_mm) or not self._finite(evidence.camera_rotation_deg):
            self.active_state, self.pending_state, self.pending_count = UNKNOWN, None, 0
            return self._record(UNKNOWN, ["accepted_static_background_motion_required"])
        translation = float(evidence.camera_translation_mm)
        rotation = float(evidence.camera_rotation_deg)
        camera_static = translation <= c.maximum_static_translation_mm and rotation <= c.maximum_static_rotation_deg
        camera_moving = translation >= c.minimum_moving_translation_mm or rotation >= c.minimum_moving_rotation_deg
        human_moving = self._finite(evidence.human_relative_speed_mm_s) and float(evidence.human_relative_speed_mm_s) >= c.minimum_human_relative_speed_mm_s
        feet_stationary = self._finite(evidence.feet_anchor_residual_maximum_mm) and float(evidence.feet_anchor_residual_maximum_mm) <= c.maximum_stationary_feet_residual_mm

        candidate: str | None = None
        if camera_static and human_moving:
            candidate = WALKER_STATIC_HUMAN_MOVING
        elif camera_moving and feet_stationary:
            candidate = FEET_STATIC_WALKER_MOVING

        if candidate is not None:
            if candidate == self.active_state:
                self.pending_state, self.pending_count = None, 0
                return self._record(candidate, [])
            if self.pending_state == candidate:
                self.pending_count += 1
            else:
                self.pending_state, self.pending_count = candidate, 1
            if self.pending_count >= c.confirmation_frames:
                self.active_state = candidate
                self.pending_state, self.pending_count = None, 0
                return self._record(candidate, [])
            return self._record(TRANSITION, ["state_confirmation_pending"])

        self.pending_state, self.pending_count = None, 0
        if camera_static and not human_moving and self.active_state == FEET_STATIC_WALKER_MOVING:
            return self._record(SETTLING, ["camera_static_after_walker_motion"])
        if camera_moving and not feet_stationary:
            return self._record(TRANSITION, ["moving_camera_without_stationary_two_foot_support"])
        if not camera_static and not camera_moving:
            return self._record(UNKNOWN, ["camera_motion_inside_deadband"])
        return self._record(UNKNOWN, ["stage_motion_evidence_incomplete"])

    def _record(self, state: str, reasons: list[str]) -> dict[str, Any]:
        return {
            "schema_version": "two_stage_walker_motion_state_v1",
            "state": state,
            "active_confirmed_state": self.active_state,
            "pending_state": self.pending_state,
            "pending_count": self.pending_count,
            "reasons": reasons,
            "criteria": self.criteria.__dict__,
            "interpretation": (
                "Controlled-experiment motion state. Unknown and transition frames are retained; "
                "the state is not a natural-gait phase or physical ground truth."
            ),
        }
