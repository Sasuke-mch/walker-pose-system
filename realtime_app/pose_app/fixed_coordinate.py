"""T3: rigidly express direct stereo observations in a locked physical frame."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import copy
import math
from typing import Any

import numpy as np


SOURCE_FRAME = "left_camera"
LENGTH_UNIT = "millimeter"
RIGID_TOLERANCE = 1e-6


def _finite_vector(value: Any, label: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must be a three-element vector")
    vector = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain only finite values")
    return vector


def _finite_matrix(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 3x3 matrix")
    return matrix


class RigidTransform:
    """One locked source-to-target coordinate transform in millimetres."""

    def __init__(
        self,
        *,
        transform_id: str,
        source_frame: str,
        target_frame: str,
        rotation_target_from_source: np.ndarray,
        translation_target_from_source_mm: np.ndarray,
        reference_definition: Mapping[str, Any],
        status: str,
    ) -> None:
        if not transform_id.strip():
            raise ValueError("transform_id must not be empty")
        if source_frame != SOURCE_FRAME:
            raise ValueError(f"source_coordinate_frame must be {SOURCE_FRAME!r}")
        if not target_frame.strip() or target_frame == source_frame:
            raise ValueError("target_coordinate_frame must be non-empty and different")

        rotation = _finite_matrix(rotation_target_from_source, "rotation_target_from_source")
        translation = _finite_vector(
            translation_target_from_source_mm,
            "translation_target_from_source_mm",
        )
        orthogonality_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro"))
        determinant = float(np.linalg.det(rotation))
        if orthogonality_error > RIGID_TOLERANCE or abs(determinant - 1.0) > RIGID_TOLERANCE:
            raise ValueError(
                "rotation_target_from_source must be a proper rigid rotation "
                f"(orthogonality_error={orthogonality_error:.3g}, determinant={determinant:.9g})"
            )

        self.transform_id = transform_id
        self.source_frame = source_frame
        self.target_frame = target_frame
        self.rotation = rotation
        self.translation_mm = translation
        self.reference_definition = dict(reference_definition)
        self.status = status

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, allow_test_transform: bool = False
    ) -> "RigidTransform":
        status = value.get("status")
        allowed_statuses = {"measured_locked"}
        if allow_test_transform:
            allowed_statuses.add("test_only")
        if status not in allowed_statuses:
            raise ValueError(
                "T3 transform must have status='measured_locked'. A test_only transform requires "
                "explicit allow_test_transform; template or provisional transforms cannot be applied."
            )
        if value.get("length_unit") not in {"mm", "millimeter", "millimeters"}:
            raise ValueError("T3 transform length_unit must be millimeter")
        reference = value.get("reference_definition")
        if not isinstance(reference, Mapping):
            raise ValueError("reference_definition must be an object describing the physical reference")
        return cls(
            transform_id=str(value.get("transform_id", "")).strip(),
            source_frame=str(value.get("source_coordinate_frame", "")).strip(),
            target_frame=str(value.get("target_coordinate_frame", "")).strip(),
            rotation_target_from_source=value.get("rotation_target_from_source"),
            translation_target_from_source_mm=value.get("translation_target_from_source_mm"),
            reference_definition=reference,
            status=str(status),
        )

    def apply(self, xyz_source_mm: Any) -> list[float]:
        point = _finite_vector(xyz_source_mm, "source point")
        return [float(component) for component in self.rotation @ point + self.translation_mm]

    def invert(self, xyz_target_mm: Any) -> list[float]:
        point = _finite_vector(xyz_target_mm, "target point")
        return [float(component) for component in self.rotation.T @ (point - self.translation_mm)]

    def audit(self) -> dict[str, Any]:
        return {
            "transform_id": self.transform_id,
            "transform_status": self.status,
            "source_coordinate_frame": self.source_frame,
            "target_coordinate_frame": self.target_frame,
            "length_unit": LENGTH_UNIT,
            "rotation_target_from_source": self.rotation.tolist(),
            "translation_target_from_source_mm": self.translation_mm.tolist(),
            "rotation_orthogonality_error_fro": float(
                np.linalg.norm(self.rotation.T @ self.rotation - np.eye(3), ord="fro")
            ),
            "rotation_determinant": float(np.linalg.det(self.rotation)),
            "reference_definition": self.reference_definition,
        }


def transform_trajectory_record(record: Mapping[str, Any], transform: RigidTransform) -> dict[str, Any]:
    """Add target-frame coordinates while retaining every source observation and rejection."""

    if record.get("coordinate_frame") != SOURCE_FRAME:
        raise ValueError("T3 input trajectory must use left_camera coordinates")
    if record.get("length_unit") != LENGTH_UNIT:
        raise ValueError("T3 input trajectory must use millimeter coordinates")

    output = copy.deepcopy(dict(record))
    output["source_coordinate_frame"] = SOURCE_FRAME
    output["target_coordinate_frame"] = transform.target_frame
    output["coordinate_transform_id"] = transform.transform_id
    for name, point in output.get("points", {}).items():
        if bool(point.get("observed_3d")):
            source_xyz = point.get("xyz_left_camera_mm")
            if source_xyz is None:
                raise ValueError(f"pair {record.get('pair_id')}: {name} marked observed without source xyz")
            point["xyz_target_mm"] = transform.apply(source_xyz)
            point["target_coordinate_status"] = "transformed_direct_observation"
            point["target_coordinate_reason"] = None
        else:
            point["xyz_target_mm"] = None
            point["target_coordinate_status"] = "source_missing_or_rejected"
            point["target_coordinate_reason"] = point.get("reason") or "source_not_observed"
    return output


def transform_trajectory_records(
    records: Iterable[Mapping[str, Any]], transform: RigidTransform
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    previous_pair_id: int | None = None
    for record in records:
        transformed = transform_trajectory_record(record, transform)
        pair_id = int(transformed["pair_id"])
        if previous_pair_id is not None and pair_id <= previous_pair_id:
            raise ValueError("pair_id values must be strictly increasing")
        previous_pair_id = pair_id
        output.append(transformed)
    if not output:
        raise ValueError("trajectory is empty")
    return output


def summarize_transformed_trajectory(
    records: list[Mapping[str, Any]], transform: RigidTransform
) -> dict[str, Any]:
    source_observed = 0
    target_transformed = 0
    propagated_rejections: dict[str, int] = {}
    max_roundtrip_error_mm = 0.0
    for record in records:
        for point in record["points"].values():
            if point.get("observed_3d"):
                source_observed += 1
                target_transformed += int(point.get("xyz_target_mm") is not None)
                recovered = transform.invert(point["xyz_target_mm"])
                source = _finite_vector(point["xyz_left_camera_mm"], "source point")
                max_roundtrip_error_mm = max(
                    max_roundtrip_error_mm,
                    float(np.linalg.norm(np.asarray(recovered) - source)),
                )
            else:
                reason = point.get("target_coordinate_reason") or "source_not_observed"
                propagated_rejections[reason] = propagated_rejections.get(reason, 0) + 1
    return {
        "frames": len(records),
        "source_observed_points": source_observed,
        "target_transformed_points": target_transformed,
        "source_rejections_propagated": sum(propagated_rejections.values()),
        "propagated_rejection_reasons": propagated_rejections,
        "max_numeric_inverse_roundtrip_error_mm": max_roundtrip_error_mm,
        "transform": transform.audit(),
        "interpretation": (
            "Rigid coordinate expression of direct accepted stereo observations only. "
            "No point is corrected, smoothed, interpolated, ground-contact labeled, or "
            "converted into a gait parameter."
        ),
    }


def summarize_static_reference(
    samples: Iterable[Mapping[str, Any]], transform: RigidTransform
) -> dict[str, Any]:
    """Report residuals for a static, physically measured target-frame reference."""

    by_reference: dict[str, list[float]] = {}
    total = 0
    for sample in samples:
        reference_id = str(sample.get("reference_id", "")).strip()
        if not reference_id:
            raise ValueError("static reference sample requires reference_id")
        estimated = np.asarray(transform.apply(sample.get("xyz_left_camera_mm")), dtype=np.float64)
        expected = _finite_vector(sample.get("expected_xyz_target_mm"), "expected_xyz_target_mm")
        residual = float(np.linalg.norm(estimated - expected))
        if not math.isfinite(residual):
            raise ValueError("static reference residual is non-finite")
        by_reference.setdefault(reference_id, []).append(residual)
        total += 1
    if total == 0:
        raise ValueError("static reference sample stream is empty")

    per_reference = {
        reference_id: {
            "samples": len(values),
            "median_residual_mm": float(np.median(values)),
            "p95_residual_mm": float(np.percentile(values, 95)),
            "max_residual_mm": float(np.max(values)),
        }
        for reference_id, values in by_reference.items()
    }
    all_values = [value for values in by_reference.values() for value in values]
    return {
        "samples": total,
        "per_reference": per_reference,
        "overall_median_residual_mm": float(np.median(all_values)),
        "overall_p95_residual_mm": float(np.percentile(all_values, 95)),
        "overall_max_residual_mm": float(np.max(all_values)),
        "interpretation": (
            "Residual to the supplied physical target-frame reference. This audits the "
            "locked transform plus static observation stability; it is not a human-pose "
            "or gait-parameter accuracy measurement."
        ),
    }
