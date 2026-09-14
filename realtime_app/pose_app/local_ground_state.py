"""Conditioned local-ground state for a moving walker-mounted stereo rig.

This module deliberately does *not* create a permanent world frame.  A plane
is always expressed in the current left-camera frame::

    n_left_camera^T X_left_camera + d_mm = 0

It has two deliberately separate consumers:

* a directly observed plane may be audited against a locked walker support
  template; the template never creates a plane by itself;
* a previously direct plane may be propagated by independently accepted visual
  relative pose plus IMU rotation, but only with explicit time/extrinsic and
  disagreement gates.

There is no acceleration integration, no inferred camera height, and no
contact label in this module.  ``lift_candidate`` means only that a locked
support template is geometrically separated from a directly observed plane.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping
from typing import Any

import numpy as np


LEFT_CAMERA_FRAME = "left_camera"
WALKER_FRAME = "walker"
LENGTH_UNIT = "millimeter"
GROUND_STATE_SCHEMA = "local_ground_state_v1"
WALKER_SUPPORT_TEMPLATE_SCHEMA = "walker_support_template_v1"
VISUAL_INERTIAL_SCHEMA = "local_ground_visual_inertial_v1"


def _required_text(value: Any, label: str) -> str:
    rendered = str(value or "").strip()
    if not rendered:
        raise ValueError(f"{label} must be a non-empty string")
    return rendered


def _finite_vector(value: Any, label: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must be a three-element vector")
    vector = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain finite values")
    return vector


def _unit_vector(value: Any, label: str) -> np.ndarray:
    vector = _finite_vector(value, label)
    magnitude = float(np.linalg.norm(vector))
    if magnitude < 1e-9:
        raise ValueError(f"{label} must have non-zero magnitude")
    return vector / magnitude


def _proper_rotation(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 3x3 matrix")
    orthogonality_error = float(np.linalg.norm(matrix.T @ matrix - np.eye(3), ord="fro"))
    determinant = float(np.linalg.det(matrix))
    if orthogonality_error > 1e-6 or abs(determinant - 1.0) > 1e-6:
        raise ValueError(
            f"{label} must be a proper rigid rotation "
            f"(orthogonality_error={orthogonality_error:.3g}, determinant={determinant:.9g})"
        )
    return matrix


def _finite_nonnegative(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and non-negative") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return number


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if number < 1 or number != value:
        raise ValueError(f"{label} must be a positive integer")
    return number


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative integer") from exc
    if number < 0 or number != value:
        raise ValueError(f"{label} must be a non-negative integer")
    return number


def _angle_degrees(first: np.ndarray, second: np.ndarray, *, unsigned: bool = False) -> float:
    dot = float(first @ second)
    if unsigned:
        dot = abs(dot)
    return float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))


@dataclass(frozen=True)
class LocalGroundPlane:
    """A local plane state, never an implicit world-ground coordinate system."""

    status: str
    normal_left_camera: np.ndarray | None
    offset_mm: float | None
    normal_orientation: str | None
    reasons: tuple[str, ...]
    provenance: str | None = None

    @classmethod
    def unavailable(cls, *reasons: str) -> "LocalGroundPlane":
        return cls("unavailable", None, None, None, tuple(reasons), None)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LocalGroundPlane":
        """Read the explicit route-1 ground-state contract.

        Accepted direct records need ``status=direct``, left-camera millimetre
        coordinates, a plane normal, offset, and ``normal_orientation`` equal
        to ``toward_camera``.  Legacy RANSAC candidate CSV/JSON fields are
        intentionally not accepted as direct ground observations.
        """

        container: Mapping[str, Any] = value
        for field in ("ground_state", "ground_observation"):
            candidate = value.get(field)
            if isinstance(candidate, Mapping):
                container = candidate
                break
        status = str(container.get("status", "")).strip()
        reasons_raw = container.get("reasons", container.get("reason", []))
        if isinstance(reasons_raw, str):
            reasons = (reasons_raw,)
        elif isinstance(reasons_raw, list):
            reasons = tuple(str(reason) for reason in reasons_raw if str(reason).strip())
        else:
            reasons = ()
        provenance = str(container.get("provenance") or "").strip() or None
        if status in {"unavailable", ""}:
            return cls.unavailable(*(reasons or ("ground_not_directly_observed",)))
        if status not in {"direct", "propagated"}:
            raise ValueError("ground state status must be direct, propagated, or unavailable")
        if container.get("coordinate_frame", value.get("coordinate_frame")) != LEFT_CAMERA_FRAME:
            raise ValueError("ground state must use left_camera coordinates")
        if container.get("length_unit", value.get("length_unit")) not in {"mm", "millimeter", "millimeters"}:
            raise ValueError("ground state must use millimetres")
        plane = container.get("plane")
        plane_mapping = plane if isinstance(plane, Mapping) else container
        normal = _unit_vector(plane_mapping.get("normal_left_camera"), "normal_left_camera")
        try:
            offset = float(plane_mapping.get("offset_mm"))
        except (TypeError, ValueError) as exc:
            raise ValueError("offset_mm must be finite") from exc
        if not math.isfinite(offset):
            raise ValueError("offset_mm must be finite")
        orientation = str(plane_mapping.get("normal_orientation", "")).strip()
        if orientation != "toward_camera":
            raise ValueError(
                "normal_orientation must be toward_camera; a signless plane cannot support a lift audit"
            )
        # With n pointing from the plane toward the camera origin, n^T 0+d is
        # the positive camera-to-plane distance.  This rejects an arbitrary
        # RANSAC sign flip before support-point signed distances are interpreted.
        if offset <= 0.0:
            raise ValueError("toward_camera plane must have positive offset_mm")
        return cls(status, normal, offset, orientation, reasons, provenance)

    def as_mapping(self) -> dict[str, Any]:
        if self.status == "unavailable":
            return {
                "status": "unavailable",
                "reasons": list(self.reasons),
                "provenance": self.provenance,
            }
        return {
            "status": self.status,
            "coordinate_frame": LEFT_CAMERA_FRAME,
            "length_unit": LENGTH_UNIT,
            "plane": {
                "normal_left_camera": self.normal_left_camera.tolist() if self.normal_left_camera is not None else None,
                "offset_mm": self.offset_mm,
                "normal_orientation": self.normal_orientation,
            },
            "reasons": list(self.reasons),
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class WalkerSupportPoint:
    support_point_id: str
    xyz_walker_mm: np.ndarray


@dataclass(frozen=True)
class WalkerSupportTemplate:
    """Measured or provisional support geometry, expressed in a walker frame."""

    configuration_id: str
    status: str
    rotation_left_camera_from_walker: np.ndarray
    translation_left_camera_from_walker_mm: np.ndarray
    support_points: tuple[WalkerSupportPoint, ...]
    measurement_evidence: dict[str, str] | None

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, allow_test_template: bool = False
    ) -> "WalkerSupportTemplate":
        schema = value.get("schema")
        if schema not in {None, WALKER_SUPPORT_TEMPLATE_SCHEMA}:
            raise ValueError("unsupported walker support template schema")
        status = str(value.get("status", "")).strip()
        allowed = {"template_not_measured", "measured_locked"}
        if allow_test_template:
            allowed.add("test_only")
        if status not in allowed:
            raise ValueError("walker template status must be template_not_measured or measured_locked")
        if value.get("walker_coordinate_frame") != WALKER_FRAME:
            raise ValueError("walker_coordinate_frame must be walker")
        if value.get("length_unit") not in {"mm", "millimeter", "millimeters"}:
            raise ValueError("walker template length_unit must be millimeter")
        raw_points = value.get("support_points_walker_mm")
        if not isinstance(raw_points, list) or len(raw_points) < 3:
            raise ValueError("support_points_walker_mm must contain at least three points")
        support_points: list[WalkerSupportPoint] = []
        seen_ids: set[str] = set()
        for raw_point in raw_points:
            if not isinstance(raw_point, Mapping):
                raise ValueError("each support point must be an object")
            point_id = _required_text(raw_point.get("support_point_id"), "support_point_id")
            if point_id in seen_ids:
                raise ValueError(f"duplicate support_point_id: {point_id}")
            seen_ids.add(point_id)
            support_points.append(
                WalkerSupportPoint(point_id, _finite_vector(raw_point.get("xyz_walker_mm"), f"{point_id}.xyz_walker_mm"))
            )
        measurement_evidence: dict[str, str] | None = None
        if status == "measured_locked":
            raw_evidence = value.get("measurement_evidence")
            if not isinstance(raw_evidence, Mapping):
                raise ValueError("measured_locked walker template requires measurement_evidence")
            measurement_evidence = {
                field: _required_text(raw_evidence.get(field), f"measurement_evidence.{field}")
                for field in ("method", "capture_session", "evidence_id")
            }
        return cls(
            configuration_id=_required_text(value.get("configuration_id"), "configuration_id"),
            status=status,
            rotation_left_camera_from_walker=_proper_rotation(
                value.get("rotation_left_camera_from_walker"), "rotation_left_camera_from_walker"
            ),
            translation_left_camera_from_walker_mm=_finite_vector(
                value.get("translation_left_camera_from_walker_mm"),
                "translation_left_camera_from_walker_mm",
            ),
            support_points=tuple(support_points),
            measurement_evidence=measurement_evidence,
        )

    def point_in_left_camera(self, point: WalkerSupportPoint) -> np.ndarray:
        return self.rotation_left_camera_from_walker @ point.xyz_walker_mm + self.translation_left_camera_from_walker_mm


@dataclass(frozen=True)
class SupportCriteria:
    support_compatibility_distance_mm: float
    minimum_lift_clearance_mm: float
    criterion_id: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SupportCriteria":
        compatibility = _finite_nonnegative(
            value.get("support_compatibility_distance_mm"), "support_compatibility_distance_mm"
        )
        lift = _finite_nonnegative(value.get("minimum_lift_clearance_mm"), "minimum_lift_clearance_mm")
        if lift <= compatibility:
            raise ValueError("minimum_lift_clearance_mm must exceed support_compatibility_distance_mm")
        return cls(compatibility, lift, _required_text(value.get("criterion_id"), "criterion_id"))


def audit_walker_support(
    ground: LocalGroundPlane,
    template: WalkerSupportTemplate | None,
    criteria: SupportCriteria | None,
) -> dict[str, Any]:
    """Audit a direct external plane against walker support geometry.

    This function intentionally refuses propagated planes.  A propagated plane
    may be useful for visualization but cannot independently establish physical
    support or lifting in the current frame.
    """

    output: dict[str, Any] = {
        "status": "contact_unconfirmed",
        "configuration_id": None if template is None else template.configuration_id,
        "template_status": None if template is None else template.status,
        "support_point_signed_distances_mm": [],
        "reasons": [],
        "interpretation": (
            "This is a geometry-condition audit only. It is not a contact, support, "
            "lift, gait-event, or gait-parameter ground truth label."
        ),
    }
    if template is None:
        output["reasons"] = ["walker_support_template_missing"]
        return output
    if template.status != "measured_locked":
        output["reasons"] = [f"walker_template_status:{template.status}"]
        return output
    if criteria is None:
        output["reasons"] = ["support_criteria_missing"]
        return output
    output["criterion_id"] = criteria.criterion_id
    if ground.status != "direct":
        output["reasons"] = ["requires_current_direct_external_ground_plane"]
        return output
    if ground.normal_left_camera is None or ground.offset_mm is None or ground.normal_orientation != "toward_camera":
        output["reasons"] = ["direct_ground_plane_sign_not_verified"]
        return output
    distances: list[dict[str, Any]] = []
    values: list[float] = []
    for point in template.support_points:
        xyz_left = template.point_in_left_camera(point)
        signed_height = float(ground.normal_left_camera @ xyz_left + ground.offset_mm)
        values.append(signed_height)
        distances.append(
            {
                "support_point_id": point.support_point_id,
                "xyz_walker_mm": point.xyz_walker_mm.tolist(),
                "xyz_left_camera_mm": xyz_left.tolist(),
                "h_signed_mm": signed_height,
            }
        )
    output["support_point_signed_distances_mm"] = distances
    if all(abs(height) <= criteria.support_compatibility_distance_mm for height in values):
        output["status"] = "support-compatible"
        output["reasons"] = []
    elif all(height >= criteria.minimum_lift_clearance_mm for height in values):
        output["status"] = "lift-candidate"
        output["reasons"] = ["all_locked_support_points_clear_of_direct_plane"]
    elif any(height < -criteria.support_compatibility_distance_mm for height in values):
        output["reasons"] = ["support_template_inconsistent_with_direct_plane"]
    else:
        output["reasons"] = ["support_geometry_inconclusive"]
    return output


@dataclass(frozen=True)
class VisualRelativePose:
    from_pair_id: int
    to_pair_id: int
    timestamp_from_ns: int
    timestamp_to_ns: int
    rotation_current_from_previous: np.ndarray
    translation_current_from_previous_mm: np.ndarray
    status: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VisualRelativePose":
        status = str(value.get("status", "")).strip()
        if status != "accepted_static_background":
            raise ValueError("visual relative pose status must be accepted_static_background")
        result = cls(
            _nonnegative_integer(value.get("from_pair_id"), "from_pair_id"),
            _nonnegative_integer(value.get("to_pair_id"), "to_pair_id"),
            _positive_integer(value.get("timestamp_from_ns"), "timestamp_from_ns"),
            _positive_integer(value.get("timestamp_to_ns"), "timestamp_to_ns"),
            _proper_rotation(value.get("rotation_current_from_previous"), "rotation_current_from_previous"),
            _finite_vector(value.get("translation_current_from_previous_mm"), "translation_current_from_previous_mm"),
            status,
        )
        if result.to_pair_id <= result.from_pair_id or result.timestamp_to_ns <= result.timestamp_from_ns:
            raise ValueError("visual relative pose must advance pair id and timestamp")
        return result


@dataclass(frozen=True)
class ImuRelativeRotation:
    from_pair_id: int
    to_pair_id: int
    rotation_current_from_previous: np.ndarray
    gravity_current_left_camera_unit: np.ndarray
    status: str
    imu_to_left_camera_extrinsics_status: str
    camera_imu_time_alignment_status: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ImuRelativeRotation":
        # Acceleration integration is intentionally prohibited.  In particular,
        # an IMU stream may constrain orientation/gravity but not manufacture a
        # translated plane or camera height in this local-ground interface.
        forbidden = {"integrated_translation_mm", "height_from_acceleration_mm", "accelerometer_integrated_height_mm"}
        found = sorted(field for field in forbidden if field in value)
        if found:
            raise ValueError(f"IMU relative input must not contain acceleration-integrated height: {', '.join(found)}")
        if str(value.get("status", "")).strip() != "accepted_rotation_only":
            raise ValueError("IMU relative rotation status must be accepted_rotation_only")
        if value.get("imu_to_left_camera_extrinsics_status") != "measured_locked":
            raise ValueError("IMU-to-camera extrinsics must be measured_locked")
        if value.get("camera_imu_time_alignment_status") != "verified":
            raise ValueError("camera-IMU time alignment must be verified")
        return cls(
            _nonnegative_integer(value.get("from_pair_id"), "from_pair_id"),
            _nonnegative_integer(value.get("to_pair_id"), "to_pair_id"),
            _proper_rotation(value.get("rotation_current_from_previous"), "imu.rotation_current_from_previous"),
            _unit_vector(value.get("gravity_current_left_camera_unit"), "gravity_current_left_camera_unit"),
            "accepted_rotation_only",
            "measured_locked",
            "verified",
        )


@dataclass(frozen=True)
class PropagationCriteria:
    criterion_id: str
    maximum_propagation_interval_ms: float
    maximum_visual_imu_rotation_disagreement_deg: float
    maximum_plane_gravity_angle_deg: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PropagationCriteria":
        return cls(
            _required_text(value.get("criterion_id"), "criterion_id"),
            _finite_nonnegative(value.get("maximum_propagation_interval_ms"), "maximum_propagation_interval_ms"),
            _finite_nonnegative(
                value.get("maximum_visual_imu_rotation_disagreement_deg"),
                "maximum_visual_imu_rotation_disagreement_deg",
            ),
            _finite_nonnegative(value.get("maximum_plane_gravity_angle_deg"), "maximum_plane_gravity_angle_deg"),
        )


def propagate_plane_with_visual_imu(
    previous_direct_ground: LocalGroundPlane,
    visual: VisualRelativePose | None,
    imu: ImuRelativeRotation | None,
    criteria: PropagationCriteria | None,
) -> tuple[LocalGroundPlane, dict[str, Any]]:
    """Propagate one direct plane by a separately gated local camera motion.

    Given ``X_t = R X_prev + q``, the implementation uses
    ``n_t = R n_prev`` and ``d_t = d_prev - n_t^T q``.  The returned audit
    always records why it refused propagation; callers must not silently turn
    that refusal into a held plane.
    """

    audit: dict[str, Any] = {
        "schema": VISUAL_INERTIAL_SCHEMA,
        "status": "unavailable",
        "criterion_id": None if criteria is None else criteria.criterion_id,
        "reasons": [],
        "visual_imu_rotation_disagreement_deg": None,
        "plane_gravity_angle_deg": None,
        "interpretation": (
            "Visual-inertial propagation is a short-term local state only. IMU acceleration is "
            "not integrated into height or translation; visual translation remains independently required."
        ),
    }
    if previous_direct_ground.status != "direct":
        audit["reasons"] = ["previous_anchor_is_not_direct"]
        return LocalGroundPlane.unavailable(*audit["reasons"]), audit
    if visual is None:
        audit["reasons"] = ["visual_relative_pose_missing"]
        return LocalGroundPlane.unavailable(*audit["reasons"]), audit
    if imu is None:
        audit["reasons"] = ["imu_rotation_input_missing"]
        return LocalGroundPlane.unavailable(*audit["reasons"]), audit
    if criteria is None:
        audit["reasons"] = ["propagation_criteria_missing"]
        return LocalGroundPlane.unavailable(*audit["reasons"]), audit
    if visual.from_pair_id != imu.from_pair_id or visual.to_pair_id != imu.to_pair_id:
        audit["reasons"] = ["visual_imu_pair_interval_mismatch"]
        return LocalGroundPlane.unavailable(*audit["reasons"]), audit
    interval_ms = (visual.timestamp_to_ns - visual.timestamp_from_ns) / 1_000_000.0
    audit["propagation_interval_ms"] = interval_ms
    if interval_ms > criteria.maximum_propagation_interval_ms:
        audit["reasons"] = ["propagation_interval_exceeds_limit"]
        return LocalGroundPlane.unavailable(*audit["reasons"]), audit
    rotation_disagreement = _angle_degrees(
        np.asarray([1.0, 0.0, 0.0]),
        (visual.rotation_current_from_previous.T @ imu.rotation_current_from_previous) @ np.asarray([1.0, 0.0, 0.0]),
    )
    # The trace form is a true SO(3) angle; the vector form above is not enough
    # for rotations around x, so replace it with the complete relative rotation.
    relative_rotation = visual.rotation_current_from_previous.T @ imu.rotation_current_from_previous
    rotation_disagreement = float(
        np.degrees(np.arccos(np.clip((float(np.trace(relative_rotation)) - 1.0) / 2.0, -1.0, 1.0)))
    )
    audit["visual_imu_rotation_disagreement_deg"] = rotation_disagreement
    if rotation_disagreement > criteria.maximum_visual_imu_rotation_disagreement_deg:
        audit["reasons"] = ["visual_imu_rotation_disagreement_exceeds_limit"]
        return LocalGroundPlane.unavailable(*audit["reasons"]), audit
    assert previous_direct_ground.normal_left_camera is not None
    assert previous_direct_ground.offset_mm is not None
    normal = visual.rotation_current_from_previous @ previous_direct_ground.normal_left_camera
    offset = float(previous_direct_ground.offset_mm - normal @ visual.translation_current_from_previous_mm)
    if offset <= 0.0:
        audit["reasons"] = ["propagated_toward_camera_offset_nonpositive"]
        return LocalGroundPlane.unavailable(*audit["reasons"]), audit
    gravity_angle = _angle_degrees(normal, imu.gravity_current_left_camera_unit, unsigned=True)
    audit["plane_gravity_angle_deg"] = gravity_angle
    if gravity_angle > criteria.maximum_plane_gravity_angle_deg:
        audit["reasons"] = ["propagated_plane_gravity_disagreement_exceeds_limit"]
        return LocalGroundPlane.unavailable(*audit["reasons"]), audit
    audit["status"] = "propagated"
    audit["reasons"] = []
    audit["from_pair_id"] = visual.from_pair_id
    audit["to_pair_id"] = visual.to_pair_id
    return (
        LocalGroundPlane(
            "propagated",
            normal,
            offset,
            "toward_camera",
            (),
            "previous_direct_plane_plus_accepted_static_background_vo_and_imu_rotation",
        ),
        audit,
    )
