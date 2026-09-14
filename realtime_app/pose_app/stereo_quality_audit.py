"""Read-only diagnostics for a frozen single-person stereo replay.

This module intentionally consumes saved ``offline_stereo_results.jsonl``
records.  It never re-associates people, re-triangulates a point, fills a gap,
or changes a rejection reason.  The resulting figures are internal geometric
quality diagnostics, not human-pose accuracy measurements.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any

import numpy as np


COCO17 = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)
BONES = {
    "left_thigh": ("left_hip", "left_knee"),
    "right_thigh": ("right_hip", "right_knee"),
    "left_shank": ("left_knee", "left_ankle"),
    "right_shank": ("right_knee", "right_ankle"),
}


def numeric_summary(values: list[float]) -> dict[str, int | float | None]:
    """Return robust descriptive statistics, retaining empty inputs explicitly."""
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return {"n": 0, "mean": None, "median": None, "mad": None, "p10": None, "p90": None, "p95": None, "max": None}
    median = float(np.median(finite))
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": median,
        "mad": float(np.median(np.abs(finite - median))),
        "p10": float(np.percentile(finite, 10)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _unit_ray(normalized_xy: np.ndarray) -> np.ndarray:
    ray = np.asarray([normalized_xy[0], normalized_xy[1], 1.0], dtype=float)
    return ray / np.linalg.norm(ray)


def ray_intersection_angle_deg(
    calibration: Any, left_point: list[float], right_point: list[float]
) -> float | None:
    """Return the angle between two raw-pixel viewing rays in left-camera axes.

    It is a geometric conditioning indicator: a small angle can make depth
    sensitive to 2-D noise.  It says nothing by itself about pose accuracy.
    """
    try:
        left = calibration.undistort_normalized(np.asarray([left_point[:2]], dtype=float), "left")[0]
        right = calibration.undistort_normalized(np.asarray([right_point[:2]], dtype=float), "right")[0]
        left_ray = _unit_ray(left)
        right_ray_in_left = calibration.R.T @ _unit_ray(right)
        cosine = float(np.clip(np.dot(left_ray, right_ray_in_left), -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))
    except (ValueError, IndexError, TypeError, np.linalg.LinAlgError):
        return None


def _point_vector(point: dict[str, Any] | None) -> np.ndarray | None:
    if not point or not point.get("valid"):
        return None
    raw = point.get("xyz") or point.get("xyz_left_camera")
    try:
        vector = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        return None
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        return None
    return vector


def _longest_run(pair_ids: list[int], observed: list[bool]) -> int:
    longest = current = 0
    previous_id: int | None = None
    for pair_id, present in zip(pair_ids, observed):
        if present and previous_id is not None and pair_id == previous_id + 1:
            current += 1
        elif present:
            current = 1
        else:
            current = 0
        longest = max(longest, current)
        previous_id = pair_id
    return longest


def audit_saved_stereo_records(
    records: list[dict[str, Any]], *, calibration: Any, keypoint_threshold: float,
    manifest_by_file_name: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Audit ordered saved strict-replay records without changing observations."""
    if not records:
        raise ValueError("At least one saved stereo record is required")
    pair_ids = [int(record["pair_id"]) for record in records]
    if pair_ids != sorted(pair_ids) or len(set(pair_ids)) != len(pair_ids):
        raise ValueError("pair_id values must be unique and ordered")

    frame_rows: list[dict[str, Any]] = []
    joint_state: dict[str, dict[str, list[float] | list[bool] | int]] = {
        name: defaultdict(list) for name in COCO17
    }
    rejections: Counter[tuple[str, str]] = Counter()
    bone_observations: dict[str, list[float]] = defaultdict(list)
    bone_deltas: dict[str, list[float]] = defaultdict(list)
    bone_valid: dict[str, list[bool]] = defaultdict(list)
    previous_bones: dict[str, tuple[int, float]] = {}

    manifest_by_file_name = manifest_by_file_name or {}
    for record in records:
        pair_id = int(record["pair_id"])
        file_name = str(record["file_name"])
        capture_meta = manifest_by_file_name.get(file_name, {})
        outcome = str(record.get("pair_outcome") or "missing_pair_outcome")
        persons = record.get("persons_3d") or []
        if len(persons) > 1:
            raise ValueError(f"pair_id {pair_id}: expected max_matches=1, got {len(persons)} persons")
        person = persons[0] if persons else None
        keypoints = {str(point.get("name")): point for point in (person or {}).get("keypoints_3d", [])}
        left_keypoints = ((record.get("left") or {}).get("persons") or [{}])[0].get("keypoints") if person else None
        right_keypoints = ((record.get("right") or {}).get("persons") or [{}])[0].get("keypoints") if person else None
        scores, reprojections, angles = [], [], []
        eligible_2d = finite_3d = positive_depth = direct_valid = 0
        direct_xyz: dict[str, np.ndarray] = {}

        for index, joint in enumerate(COCO17):
            point = keypoints.get(joint)
            state = joint_state[joint]
            state["frames"].append(True)
            if point is None:
                state["matched"].append(False)
                state["eligible_2d"].append(False)
                state["finite_3d"].append(False)
                state["positive_depth"].append(False)
                state["direct_valid"].append(False)
                rejections[(joint, f"pair_{outcome}")] += 1
                bone_valid[joint].append(False)
                continue

            state["matched"].append(True)
            left_point = left_keypoints[index] if left_keypoints and len(left_keypoints) > index else None
            right_point = right_keypoints[index] if right_keypoints and len(right_keypoints) > index else None
            left_score = _finite_float(point.get("left_score"))
            right_score = _finite_float(point.get("right_score"))
            if left_score is not None and right_score is not None:
                scores.append(min(left_score, right_score))
            raw_eligible = (
                left_score is not None and right_score is not None
                and left_score >= keypoint_threshold and right_score >= keypoint_threshold
                and isinstance(left_point, list) and isinstance(right_point, list)
            )
            angle = ray_intersection_angle_deg(calibration, left_point, right_point) if raw_eligible else None
            raw_eligible = raw_eligible and angle is not None
            state["eligible_2d"].append(raw_eligible)
            if raw_eligible:
                eligible_2d += 1
                angles.append(angle)
                state["ray_angle_deg"].append(angle)

            reprojection = _finite_float(point.get("reprojection_error_mean_px"))
            if reprojection is not None:
                reprojections.append(reprojection)
                state["reprojection_px"].append(reprojection)
            depth_left = _finite_float(point.get("depth_left"))
            depth_right = _finite_float(point.get("depth_right"))
            finite = depth_left is not None and depth_right is not None and reprojection is not None
            state["finite_3d"].append(finite)
            if finite:
                finite_3d += 1
            positive = bool(finite and depth_left > 0.0 and depth_right > 0.0)
            state["positive_depth"].append(positive)
            if positive:
                positive_depth += 1
            accepted = bool(point.get("valid"))
            state["direct_valid"].append(accepted)
            if accepted:
                direct_valid += 1
                rejections[(joint, "accepted")]+= 1
            else:
                rejections[(joint, str(point.get("reason") or "missing_rejection_reason"))] += 1
            vector = _point_vector(point)
            if vector is not None:
                direct_xyz[joint] = vector

        for bone, (start, end) in BONES.items():
            observed = start in direct_xyz and end in direct_xyz
            bone_valid[bone].append(observed)
            if not observed:
                previous_bones.pop(bone, None)
                continue
            length = float(np.linalg.norm(direct_xyz[start] - direct_xyz[end]))
            bone_observations[bone].append(length)
            previous = previous_bones.get(bone)
            if previous is not None and previous[0] == pair_id - 1:
                bone_deltas[bone].append(abs(length - previous[1]))
            previous_bones[bone] = (pair_id, length)

        frame_rows.append({
            "pair_id": pair_id,
            "file_name": file_name,
            "pair_outcome": outcome,
            "host_pairing_delta_ms": capture_meta.get("abs_host_delta_ms"),
            "host_pairing_delta_source": "selection_manifest.abs_host_delta_ms" if capture_meta else None,
            "saved_replay_timestamp_skew_ms": record.get("timestamp_skew_ms"),
            "saved_replay_timestamp_type": record.get("timestamp_type"),
            "matched_stereo_persons": len(persons),
            "association_cost": None if person is None else person.get("association_cost"),
            "common_keypoints": None if person is None else person.get("common_keypoints"),
            "two_d_score_eligible_joint_count": eligible_2d,
            "finite_geometry_joint_count": finite_3d,
            "positive_depth_joint_count": positive_depth,
            "direct_valid_3d_joint_count": direct_valid,
            "min_left_right_score_median": numeric_summary(scores)["median"],
            "reprojection_mean_px_median": numeric_summary(reprojections)["median"],
            "ray_intersection_angle_deg_median": numeric_summary(angles)["median"],
        })

    joint_rows = []
    for joint in COCO17:
        state = joint_state[joint]
        frames = len(state["frames"])
        row = {"joint": joint, "frames": frames}
        for label, key in (("matched_frames", "matched"), ("two_d_score_eligible_frames", "eligible_2d"), ("finite_geometry_frames", "finite_3d"), ("positive_depth_frames", "positive_depth"), ("direct_valid_3d_frames", "direct_valid")):
            count = int(sum(state[key]))
            row[label] = count
            row[f"{label}_rate"] = count / frames
        row.update({f"reprojection_{key}": value for key, value in numeric_summary(state["reprojection_px"]).items()})
        row.update({f"ray_angle_deg_{key}": value for key, value in numeric_summary(state["ray_angle_deg"]).items()})
        joint_rows.append(row)

    rejection_rows = [
        {"joint": joint, "outcome_or_reason": reason, "count": count}
        for (joint, reason), count in sorted(rejections.items())
    ]
    bone_rows = []
    for bone in BONES:
        observed = bone_valid[bone]
        row = {"bone": bone, "frames": len(records), "direct_observed_frames": int(sum(observed)), "direct_observed_rate": sum(observed) / len(records), "longest_direct_observed_run_frames": _longest_run(pair_ids, observed)}
        row.update({f"length_mm_{key}": value for key, value in numeric_summary(bone_observations[bone]).items()})
        row.update({f"consecutive_abs_change_mm_{key}": value for key, value in numeric_summary(bone_deltas[bone]).items()})
        bone_rows.append(row)

    matched = sum(1 for row in frame_rows if row["matched_stereo_persons"] == 1)
    summary = {
        "frames": len(records),
        "matched_frames": matched,
        "matched_frame_rate": matched / len(records),
        "host_pairing_delta_ms": numeric_summary([_finite_float(row["host_pairing_delta_ms"]) for row in frame_rows if _finite_float(row["host_pairing_delta_ms"]) is not None]),
        "host_pairing_timing_note": "abs_host_delta_ms is host-side capture pairing metadata, not an exposure-synchronization measurement. Saved replay timestamps are not used as a synchronization claim.",
        "association_cost": numeric_summary([_finite_float(row["association_cost"]) for row in frame_rows if _finite_float(row["association_cost"]) is not None]),
        "direct_valid_3d_points": int(sum(row["direct_valid_3d_joint_count"] for row in frame_rows)),
        "note": "Read-only internal geometry diagnostics. Reprojection, viewing-ray angle, coverage and segment stability do not establish human-pose accuracy or physical gait validity.",
    }
    return {"frame_rows": frame_rows, "joint_rows": joint_rows, "rejection_rows": rejection_rows, "bone_rows": bone_rows, "summary": summary}
