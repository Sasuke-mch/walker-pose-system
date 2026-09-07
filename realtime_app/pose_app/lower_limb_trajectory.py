from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any


LOWER_LIMB_JOINTS: tuple[tuple[int, str], ...] = (
    (11, "left_hip"),
    (12, "right_hip"),
    (13, "left_knee"),
    (14, "right_knee"),
    (15, "left_ankle"),
    (16, "right_ankle"),
)


def _finite_xyz(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        xyz = [float(component) for component in value]
    except (TypeError, ValueError):
        return None
    return xyz if all(math.isfinite(component) for component in xyz) else None


def _empty_point(index: int, name: str, reason: str) -> dict[str, Any]:
    return {
        "index": index,
        "name": name,
        "observed_3d": False,
        "xyz_left_camera_mm": None,
        "score": None,
        "left_score": None,
        "right_score": None,
        "reprojection_error_left_px": None,
        "reprojection_error_right_px": None,
        "reprojection_error_mean_px": None,
        "reason": reason,
    }


def normalize_stereo_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert one stereo result into the model-agnostic lower-limb contract.

    The function never selects among multiple reconstructed people, fills a gap,
    filters a point, or changes a 3-D coordinate.
    """

    pair_id = int(record["pair_id"])
    coordinate_frame = record.get("coordinate_frame", "left_camera")
    length_unit = record.get("length_unit", "millimeter")
    if coordinate_frame != "left_camera":
        raise ValueError(
            f"pair {pair_id}: expected left_camera coordinates, got {coordinate_frame!r}"
        )
    if length_unit not in {"mm", "millimeter", "millimeters"}:
        raise ValueError(
            f"pair {pair_id}: expected millimeter coordinates, got {length_unit!r}"
        )

    persons = record.get("persons_3d") or []
    if len(persons) == 0:
        frame_status = "no_accepted_stereo_person"
        selected_person = None
    elif len(persons) == 1:
        frame_status = "accepted_single_person"
        selected_person = persons[0]
    else:
        frame_status = "ambiguous_multiple_stereo_persons"
        selected_person = None

    source_points: dict[str, dict[str, Any]] = {}
    if selected_person is not None:
        for point in selected_person.get("keypoints_3d", []):
            name = point.get("name")
            if isinstance(name, str):
                source_points[name] = point

    points: dict[str, dict[str, Any]] = {}
    for expected_index, name in LOWER_LIMB_JOINTS:
        if selected_person is None:
            points[name] = _empty_point(expected_index, name, frame_status)
            continue
        source = source_points.get(name)
        if source is None:
            points[name] = _empty_point(expected_index, name, "missing_keypoint_record")
            continue

        xyz = _finite_xyz(source.get("xyz"))
        observed = bool(source.get("valid")) and xyz is not None
        reason = source.get("reason")
        if not observed and reason is None:
            reason = "invalid_or_non_finite_3d"
        points[name] = {
            "index": int(source.get("index", expected_index)),
            "name": name,
            "observed_3d": observed,
            "xyz_left_camera_mm": xyz if observed else None,
            "score": source.get("score"),
            "left_score": source.get("left_score"),
            "right_score": source.get("right_score"),
            "reprojection_error_left_px": source.get(
                "reprojection_error_left_px"
            ),
            "reprojection_error_right_px": source.get(
                "reprojection_error_right_px"
            ),
            "reprojection_error_mean_px": source.get(
                "reprojection_error_mean_px"
            ),
            "reason": None if observed else reason,
        }

    timestamp = record.get("pair_timestamp_sec")
    if timestamp is not None:
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            timestamp = None

    return {
        "pair_id": pair_id,
        "pair_timestamp_sec": timestamp,
        "left_frame_id": record.get("left_frame_id"),
        "right_frame_id": record.get("right_frame_id"),
        "timestamp_skew_ms": record.get("timestamp_skew_ms"),
        "timestamp_type": record.get("timestamp_type"),
        "coordinate_frame": "left_camera",
        "length_unit": "millimeter",
        "frame_status": frame_status,
        "source_person_count": len(persons),
        "points": points,
    }


def normalize_stereo_records(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    previous_pair_id: int | None = None
    for record in records:
        normalized = normalize_stereo_record(record)
        pair_id = normalized["pair_id"]
        if previous_pair_id is not None and pair_id <= previous_pair_id:
            raise ValueError("pair_id values must be strictly increasing")
        previous_pair_id = pair_id
        output.append(normalized)
    if not output:
        raise ValueError("stereo result stream is empty")
    return output


def _longest_observed_run(records: list[dict[str, Any]], joint: str) -> int:
    longest = 0
    current = 0
    previous_pair_id: int | None = None
    for record in records:
        pair_id = int(record["pair_id"])
        observed = bool(record["points"][joint]["observed_3d"])
        adjacent = previous_pair_id is not None and pair_id == previous_pair_id + 1
        if observed:
            current = current + 1 if adjacent else 1
            longest = max(longest, current)
        else:
            current = 0
        previous_pair_id = pair_id
    return longest


def summarize_trajectory(records: list[dict[str, Any]]) -> dict[str, Any]:
    frame_status_counts: dict[str, int] = {}
    for record in records:
        status = record["frame_status"]
        frame_status_counts[status] = frame_status_counts.get(status, 0) + 1

    per_joint: dict[str, dict[str, Any]] = {}
    for _, name in LOWER_LIMB_JOINTS:
        observed = sum(
            bool(record["points"][name]["observed_3d"]) for record in records
        )
        reasons: dict[str, int] = {}
        for record in records:
            point = record["points"][name]
            if point["observed_3d"]:
                continue
            reason = point["reason"] or "unspecified"
            reasons[reason] = reasons.get(reason, 0) + 1
        per_joint[name] = {
            "observed_frames": observed,
            "coverage": observed / len(records),
            "longest_contiguous_observed_run_frames": _longest_observed_run(
                records, name
            ),
            "missing_or_rejected_reasons": reasons,
        }

    observed_points = sum(
        point["observed_3d"]
        for record in records
        for point in record["points"].values()
    )
    expected_points = len(records) * len(LOWER_LIMB_JOINTS)
    return {
        "frames": len(records),
        "first_pair_id": records[0]["pair_id"],
        "last_pair_id": records[-1]["pair_id"],
        "frame_status_counts": frame_status_counts,
        "expected_lower_limb_points": expected_points,
        "observed_lower_limb_points": observed_points,
        "observed_lower_limb_coverage": observed_points / expected_points,
        "per_joint": per_joint,
        "coordinate_frame": "left_camera",
        "length_unit": "millimeter",
        "interpretation": (
            "Direct accepted stereo observations only. Missing and rejected points "
            "remain missing; no filtering, interpolation, gait event, or accuracy "
            "claim is introduced."
        ),
    }
