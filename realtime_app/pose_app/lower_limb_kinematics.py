from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any

import numpy as np


DISTANCE_METRICS = (
    "left_thigh_length_mm",
    "right_thigh_length_mm",
    "left_shank_length_mm",
    "right_shank_length_mm",
    "ankle_separation_mm",
)
ANGLE_METRICS = ("left_knee_angle_deg", "right_knee_angle_deg")
VECTOR_METRICS = ("left_thigh_vector_mm", "right_thigh_vector_mm", "left_shank_vector_mm", "right_shank_vector_mm")


def _point(record: dict[str, Any], name: str) -> np.ndarray | None:
    point = record["points"][name]
    if not point.get("observed_3d"):
        return None
    xyz = point.get("xyz_left_camera_mm")
    if not isinstance(xyz, list) or len(xyz) != 3:
        return None
    value = np.asarray(xyz, dtype=np.float64)
    return value if np.all(np.isfinite(value)) else None


def _missing_reason(record: dict[str, Any], *names: str) -> str:
    missing = [name for name in names if _point(record, name) is None]
    return "missing_or_rejected:" + ",".join(missing)


def _available_scalar(value: float, unit: str) -> dict[str, Any]:
    return {"available": True, "value": float(value), "unit": unit, "reason": None}


def _unavailable(unit: str, reason: str) -> dict[str, Any]:
    return {"available": False, "value": None, "unit": unit, "reason": reason}


def _available_vector(value: np.ndarray) -> dict[str, Any]:
    return {
        "available": True,
        "xyz_left_camera_mm": [float(component) for component in value],
        "unit": "millimeter",
        "reason": None,
    }


def _unavailable_vector(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "xyz_left_camera_mm": None,
        "unit": "millimeter",
        "reason": reason,
    }


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return None
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def derive_frame_kinematics(record: dict[str, Any]) -> dict[str, Any]:
    """Derive frame-local lower-limb quantities without filtering or gap filling."""

    if record.get("coordinate_frame") != "left_camera":
        raise ValueError("T1 trajectory must use left_camera coordinates")
    if record.get("length_unit") != "millimeter":
        raise ValueError("T1 trajectory must use millimeters")

    hips = (_point(record, "left_hip"), _point(record, "right_hip"))
    ankles = (_point(record, "left_ankle"), _point(record, "right_ankle"))
    metrics: dict[str, dict[str, Any]] = {}
    if all(value is not None for value in hips):
        metrics["pelvis_center"] = _available_vector((hips[0] + hips[1]) * 0.5)
    else:
        metrics["pelvis_center"] = _unavailable_vector(
            _missing_reason(record, "left_hip", "right_hip")
        )

    for side in ("left", "right"):
        hip_name, knee_name, ankle_name = (
            f"{side}_hip",
            f"{side}_knee",
            f"{side}_ankle",
        )
        hip, knee, ankle = (
            _point(record, hip_name),
            _point(record, knee_name),
            _point(record, ankle_name),
        )
        if hip is not None and knee is not None:
            thigh_vector = knee - hip
            metrics[f"{side}_thigh_vector_mm"] = _available_vector(thigh_vector)
            metrics[f"{side}_thigh_length_mm"] = _available_scalar(
                float(np.linalg.norm(thigh_vector)), "millimeter"
            )
        else:
            reason = _missing_reason(record, hip_name, knee_name)
            metrics[f"{side}_thigh_vector_mm"] = _unavailable_vector(reason)
            metrics[f"{side}_thigh_length_mm"] = _unavailable("millimeter", reason)

        if knee is not None and ankle is not None:
            shank_vector = ankle - knee
            metrics[f"{side}_shank_vector_mm"] = _available_vector(shank_vector)
            metrics[f"{side}_shank_length_mm"] = _available_scalar(
                float(np.linalg.norm(shank_vector)), "millimeter"
            )
        else:
            reason = _missing_reason(record, knee_name, ankle_name)
            metrics[f"{side}_shank_vector_mm"] = _unavailable_vector(reason)
            metrics[f"{side}_shank_length_mm"] = _unavailable("millimeter", reason)

        if hip is not None and knee is not None and ankle is not None:
            angle = _angle_deg(hip - knee, ankle - knee)
            if angle is not None:
                metrics[f"{side}_knee_angle_deg"] = _available_scalar(angle, "degree")
            else:
                metrics[f"{side}_knee_angle_deg"] = _unavailable(
                    "degree", "degenerate_zero_length_segment"
                )
        else:
            metrics[f"{side}_knee_angle_deg"] = _unavailable(
                "degree", _missing_reason(record, hip_name, knee_name, ankle_name)
            )

    if all(value is not None for value in ankles):
        metrics["ankle_separation_mm"] = _available_scalar(
            float(np.linalg.norm(ankles[0] - ankles[1])), "millimeter"
        )
    else:
        metrics["ankle_separation_mm"] = _unavailable(
            "millimeter", _missing_reason(record, "left_ankle", "right_ankle")
        )

    return {
        "pair_id": int(record["pair_id"]),
        "pair_timestamp_sec": record.get("pair_timestamp_sec"),
        "coordinate_frame": "left_camera",
        "length_unit": "millimeter",
        "source_frame_status": record.get("frame_status"),
        "source_observation_policy": "direct_accepted_stereo_only",
        "metrics": metrics,
    }


def derive_kinematics(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    previous_pair_id: int | None = None
    for record in records:
        derived = derive_frame_kinematics(record)
        pair_id = derived["pair_id"]
        if previous_pair_id is not None and pair_id <= previous_pair_id:
            raise ValueError("pair_id values must be strictly increasing")
        previous_pair_id = pair_id
        output.append(derived)
    if not output:
        raise ValueError("trajectory is empty")
    return output


def _longest_available_run(records: list[dict[str, Any]], metric: str) -> int:
    longest = 0
    current = 0
    previous_pair_id: int | None = None
    for record in records:
        pair_id = int(record["pair_id"])
        available = bool(record["metrics"][metric]["available"])
        adjacent = previous_pair_id is not None and pair_id == previous_pair_id + 1
        if available:
            current = current + 1 if adjacent else 1
            longest = max(longest, current)
        else:
            current = 0
        previous_pair_id = pair_id
    return longest


def summarize_kinematics(records: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_metrics = DISTANCE_METRICS + ANGLE_METRICS
    summary: dict[str, dict[str, Any]] = {}
    for metric in scalar_metrics:
        values = [
            float(record["metrics"][metric]["value"])
            for record in records
            if record["metrics"][metric]["available"]
        ]
        unavailable_reasons: dict[str, int] = {}
        for record in records:
            source = record["metrics"][metric]
            if source["available"]:
                continue
            reason = source["reason"] or "unspecified"
            unavailable_reasons[reason] = unavailable_reasons.get(reason, 0) + 1
        summary[metric] = {
            "available_frames": len(values),
            "coverage": len(values) / len(records),
            "longest_contiguous_available_run_frames": _longest_available_run(
                records, metric
            ),
            "median": float(np.median(values)) if values else None,
            "p10": float(np.percentile(values, 10)) if values else None,
            "p90": float(np.percentile(values, 90)) if values else None,
            "unavailable_reasons": unavailable_reasons,
        }
    return {
        "frames": len(records),
        "first_pair_id": records[0]["pair_id"],
        "last_pair_id": records[-1]["pair_id"],
        "per_metric": summary,
        "coordinate_frame": "left_camera",
        "length_unit": "millimeter",
        "interpretation": (
            "Frame-local, direct-observation lower-limb geometry only. These are not "
            "ground-frame gait parameters, gait events, filtered trajectories, or "
            "external accuracy measurements."
        ),
    }
