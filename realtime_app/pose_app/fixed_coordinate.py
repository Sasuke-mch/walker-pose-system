"""T3: rigidly express direct stereo observations in a locked physical frame."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_FRAME = "left_camera"
LENGTH_UNIT = "millimeter"
RIGID_TOLERANCE = 1e-6
STATIC_REFERENCE_EVIDENCE_SCHEMA = "t3_static_reference_lock_evidence_v1"


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


def _required_text(value: Any, label: str) -> str:
    rendered = str(value or "").strip()
    if not rendered:
        raise ValueError(f"{label} must be a non-empty string")
    return rendered


def _nonnegative_finite_number(value: Any, label: str) -> float:
    try:
        rendered = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite non-negative number") from exc
    if not math.isfinite(rendered) or rendered < 0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return rendered


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        rendered = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if rendered < 1 or rendered != value:
        raise ValueError(f"{label} must be a positive integer")
    return rendered


def _validate_reference_definition(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("reference_definition must be an object describing the physical reference")
    reference = dict(value)
    for field in ("method", "physical_reference", "capture_session"):
        reference[field] = _required_text(reference.get(field), f"reference_definition.{field}")
    return reference


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
        self.reference_definition = _validate_reference_definition(reference_definition)
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
        reference = _validate_reference_definition(value.get("reference_definition"))
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
    sample_ids: set[str] = set()
    capture_session = transform.reference_definition["capture_session"]
    total = 0
    for sample in samples:
        reference_id = _required_text(sample.get("reference_id"), "static reference sample reference_id")
        sample_id = _required_text(sample.get("sample_id"), "static reference sample sample_id")
        if sample_id in sample_ids:
            raise ValueError(f"static reference sample_id is duplicated: {sample_id}")
        sample_ids.add(sample_id)
        if _required_text(sample.get("capture_session"), "static reference sample capture_session") != capture_session:
            raise ValueError("static reference sample capture_session must match transform reference_definition")
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
        "reference_ids": sorted(by_reference),
        "capture_session": capture_session,
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


def assess_static_reference_lock(
    samples: Iterable[Mapping[str, Any]],
    transform: RigidTransform,
    criteria: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate declared field criteria without inventing a physical acceptance threshold."""

    if not isinstance(criteria, Mapping):
        raise ValueError("static-reference criteria must be an object")
    normalized_criteria = {
        "criterion_id": _required_text(criteria.get("criterion_id"), "criteria.criterion_id"),
        "minimum_distinct_reference_ids": _positive_integer(
            criteria.get("minimum_distinct_reference_ids"),
            "criteria.minimum_distinct_reference_ids",
        ),
        "minimum_samples_per_reference": _positive_integer(
            criteria.get("minimum_samples_per_reference"),
            "criteria.minimum_samples_per_reference",
        ),
        "maximum_p95_residual_mm": _nonnegative_finite_number(
            criteria.get("maximum_p95_residual_mm"),
            "criteria.maximum_p95_residual_mm",
        ),
        "maximum_max_residual_mm": _nonnegative_finite_number(
            criteria.get("maximum_max_residual_mm"),
            "criteria.maximum_max_residual_mm",
        ),
    }
    summary = summarize_static_reference(samples, transform)
    failures: list[str] = []
    if len(summary["reference_ids"]) < normalized_criteria["minimum_distinct_reference_ids"]:
        failures.append("too_few_distinct_reference_ids")
    for reference_id, reference_summary in summary["per_reference"].items():
        if reference_summary["samples"] < normalized_criteria["minimum_samples_per_reference"]:
            failures.append(f"too_few_samples:{reference_id}")
    if summary["overall_p95_residual_mm"] > normalized_criteria["maximum_p95_residual_mm"]:
        failures.append("overall_p95_residual_exceeds_limit")
    if summary["overall_max_residual_mm"] > normalized_criteria["maximum_max_residual_mm"]:
        failures.append("overall_max_residual_exceeds_limit")
    return {
        "schema": STATIC_REFERENCE_EVIDENCE_SCHEMA,
        "status": "accepted" if not failures else "rejected",
        "transform": transform.audit(),
        "reference_capture_session": transform.reference_definition["capture_session"],
        "acceptance_criteria": normalized_criteria,
        "static_reference_stability": summary,
        "acceptance_failures": failures,
        "interpretation": (
            "A passed result verifies only the supplied static-reference observations against "
            "the declared transform and field criteria. It is not human-pose, gait-event, "
            "or gait-parameter accuracy validation."
        ),
    }


def load_coordinate_transform(
    path: str | Path, *, allow_test_transform: bool = False
) -> RigidTransform:
    """Load a transform and require independent static-reference evidence for physical use."""

    transform_path = Path(path).resolve()
    transform = RigidTransform.from_mapping(
        json.loads(transform_path.read_text(encoding="utf-8")),
        allow_test_transform=allow_test_transform,
    )
    if transform.status == "test_only":
        return transform

    evidence_name = _required_text(
        transform.reference_definition.get("static_reference_evidence_file"),
        "reference_definition.static_reference_evidence_file",
    )
    evidence_path = Path(evidence_name)
    if not evidence_path.is_absolute():
        evidence_path = transform_path.parent / evidence_path
    if not evidence_path.is_file():
        raise ValueError(f"measured_locked transform static-reference evidence is missing: {evidence_path}")
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid static-reference evidence JSON: {evidence_path}") from exc
    if evidence.get("schema") != STATIC_REFERENCE_EVIDENCE_SCHEMA:
        raise ValueError("static-reference evidence has an unsupported schema")
    if evidence.get("status") != "accepted":
        raise ValueError("measured_locked transform requires accepted static-reference evidence")
    evidence_transform = evidence.get("transform")
    if not isinstance(evidence_transform, Mapping):
        raise ValueError("static-reference evidence is missing transform audit data")
    for field, expected in (
        ("transform_id", transform.transform_id),
        ("source_coordinate_frame", transform.source_frame),
        ("target_coordinate_frame", transform.target_frame),
        ("rotation_target_from_source", transform.rotation.tolist()),
        ("translation_target_from_source_mm", transform.translation_mm.tolist()),
    ):
        if evidence_transform.get(field) != expected:
            raise ValueError(f"static-reference evidence {field} does not match transform")
    if evidence.get("reference_capture_session") != transform.reference_definition["capture_session"]:
        raise ValueError("static-reference evidence capture session does not match transform")
    static_reference_input = _required_text(
        evidence.get("static_reference_input"),
        "static-reference evidence static_reference_input",
    )
    raw_reference_path = Path(static_reference_input)
    if not raw_reference_path.is_absolute():
        raw_reference_path = evidence_path.parent / raw_reference_path
    if not raw_reference_path.is_file():
        raise ValueError(f"static-reference evidence raw input is missing: {raw_reference_path}")
    recomputed = assess_static_reference_lock(
        [
            json.loads(line)
            for line in raw_reference_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ],
        transform,
        evidence.get("acceptance_criteria"),
    )
    if recomputed["status"] != "accepted":
        raise ValueError("static-reference evidence raw input no longer satisfies its acceptance criteria")
    if recomputed["static_reference_stability"] != evidence.get("static_reference_stability"):
        raise ValueError("static-reference evidence no longer matches its raw input")
    return transform
