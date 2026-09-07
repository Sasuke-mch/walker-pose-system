"""Summarize a frozen C3/D1 saved-prediction stereo replay.

The script compares one already-computed C3 replay with one already-computed
D1 replay.  It never changes 2-D observations, matches, gates, or 3-D points.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


LOWER_BODY = (
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            name = record.get("file_name")
            if not isinstance(name, str) or not name:
                raise ValueError(f"{path}:{line_number}: missing file_name.")
            if name in records:
                raise ValueError(f"{path}:{line_number}: duplicate file_name {name}.")
            if len(record.get("persons_3d", [])) > 1:
                raise ValueError(
                    f"{path}:{line_number}: expected max_matches=1, found multiple people."
                )
            records[name] = record
    if not records:
        raise ValueError(f"{path}: no records.")
    return records


def _as_bool(value: str) -> bool:
    if value.strip().lower() == "true":
        return True
    if value.strip().lower() == "false":
        return False
    raise ValueError(f"Expected Boolean CSV value, got {value!r}.")


def _read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            name = row.get("file_name")
            condition = row.get("condition")
            changed = row.get("changed_from_c3")
            if not name or not condition or changed is None:
                raise ValueError(f"{path}: requires file_name, condition, changed_from_c3.")
            if name in records:
                raise ValueError(f"{path}: duplicate file_name {name}.")
            records[name] = {"condition": condition, "changed": _as_bool(changed)}
    if not records:
        raise ValueError(f"{path}: no records.")
    return records


def _roi_change_stratum(left_changed: bool, right_changed: bool) -> str:
    if left_changed and right_changed:
        return "both_changed"
    if left_changed or right_changed:
        return "one_side_changed"
    return "neither_changed"


def _pair_status(record: dict[str, Any]) -> str:
    outcome = record.get("pair_outcome")
    if outcome in {"matched", "no_target_person", "association_failed"}:
        return outcome
    people = record.get("persons_3d", [])
    if people:
        return "matched"
    left_people = record.get("left", {}).get("persons", [])
    right_people = record.get("right", {}).get("persons", [])
    return "no_target_person" if not left_people or not right_people else "association_failed"


def _joint_map(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    people = record.get("persons_3d", [])
    if len(people) != 1:
        return {}
    return {
        point["name"]: point
        for point in people[0].get("keypoints_3d", [])
        if point.get("name") in LOWER_BODY
    }


def _point_state(
    pair_status: str, points: dict[str, dict[str, Any]], joint: str
) -> tuple[str, bool, float | None]:
    if pair_status != "matched":
        return pair_status, False, None
    point = points.get(joint)
    if point is None:
        return "missing_keypoint", False, None
    status = "valid" if point.get("valid") else str(point.get("reason") or "invalid")
    value = point.get("reprojection_error_mean_px")
    error = None if value is None else float(value)
    if error is not None and not math.isfinite(error):
        error = None
    return status, bool(point.get("valid")), error


def _percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    c3_statuses = Counter(row["c3_status"] for row in rows)
    d1_statuses = Counter(row["d1_status"] for row in rows)
    transitions = Counter(
        f"{row['c3_status']}__to__{row['d1_status']}" for row in rows
    )
    c3_errors = [row["c3_reprojection_error_px"] for row in rows if row["common_computable"]]
    d1_errors = [row["d1_reprojection_error_px"] for row in rows if row["common_computable"]]
    delta_errors = [row["d1_minus_c3_reprojection_error_px"] for row in rows if row["common_computable"]]
    return {
        "joint_trials": len(rows),
        "c3_valid_joint_count": sum(row["c3_valid"] for row in rows),
        "d1_valid_joint_count": sum(row["d1_valid"] for row in rows),
        "net_valid_joint_change_d1_minus_c3": (
            sum(row["d1_valid"] for row in rows) - sum(row["c3_valid"] for row in rows)
        ),
        "c3_status_counts": dict(c3_statuses),
        "d1_status_counts": dict(d1_statuses),
        "status_transition_counts": dict(transitions),
        "common_computable_joint_count": len(c3_errors),
        "c3_common_computable_reprojection_px": {
            "median": median(c3_errors) if c3_errors else None,
            "p90": _percentile(c3_errors, 90),
        },
        "d1_common_computable_reprojection_px": {
            "median": median(d1_errors) if d1_errors else None,
            "p90": _percentile(d1_errors, 90),
        },
        "d1_minus_c3_common_computable_reprojection_px": {
            "median": median(delta_errors) if delta_errors else None,
            "p90": _percentile(delta_errors, 90),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired lower-body audit for frozen C3/D1 stereo replays."
    )
    parser.add_argument("--c3-results", required=True, type=Path)
    parser.add_argument("--d1-results", required=True, type=Path)
    parser.add_argument("--left-d1-manifest", required=True, type=Path)
    parser.add_argument("--right-d1-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    c3_records = _read_jsonl(args.c3_results.resolve())
    d1_records = _read_jsonl(args.d1_results.resolve())
    left_manifest = _read_manifest(args.left_d1_manifest.resolve())
    right_manifest = _read_manifest(args.right_d1_manifest.resolve())
    expected_names = set(c3_records)
    for label, records in (
        ("D1 replay", d1_records),
        ("left D1 manifest", left_manifest),
        ("right D1 manifest", right_manifest),
    ):
        if set(records) != expected_names:
            raise ValueError(f"{label}: file_name set does not match C3 replay.")

    point_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for name in sorted(expected_names):
        c3_record, d1_record = c3_records[name], d1_records[name]
        c3_pair_status, d1_pair_status = _pair_status(c3_record), _pair_status(d1_record)
        c3_points, d1_points = _joint_map(c3_record), _joint_map(d1_record)
        condition = left_manifest[name]["condition"]
        if right_manifest[name]["condition"] != condition:
            raise ValueError(f"{name}: left/right manifests disagree on condition.")
        left_changed = left_manifest[name]["changed"]
        right_changed = right_manifest[name]["changed"]
        stratum = _roi_change_stratum(left_changed, right_changed)
        c3_valid_count = 0
        d1_valid_count = 0
        for joint in LOWER_BODY:
            c3_status, c3_valid, c3_error = _point_state(c3_pair_status, c3_points, joint)
            d1_status, d1_valid, d1_error = _point_state(d1_pair_status, d1_points, joint)
            c3_valid_count += c3_valid
            d1_valid_count += d1_valid
            common_computable = c3_error is not None and d1_error is not None
            point_rows.append(
                {
                    "file_name": name,
                    "condition": condition,
                    "roi_change_stratum": stratum,
                    "left_roi_changed": left_changed,
                    "right_roi_changed": right_changed,
                    "joint": joint,
                    "c3_pair_status": c3_pair_status,
                    "d1_pair_status": d1_pair_status,
                    "c3_status": c3_status,
                    "d1_status": d1_status,
                    "c3_valid": c3_valid,
                    "d1_valid": d1_valid,
                    "c3_reprojection_error_px": c3_error,
                    "d1_reprojection_error_px": d1_error,
                    "common_computable": common_computable,
                    "d1_minus_c3_reprojection_error_px": (
                        d1_error - c3_error if common_computable else None
                    ),
                }
            )
        pair_rows.append(
            {
                "file_name": name,
                "condition": condition,
                "roi_change_stratum": stratum,
                "left_roi_changed": left_changed,
                "right_roi_changed": right_changed,
                "c3_pair_status": c3_pair_status,
                "d1_pair_status": d1_pair_status,
                "c3_valid_lower_body_joint_count": c3_valid_count,
                "d1_valid_lower_body_joint_count": d1_valid_count,
                "d1_minus_c3_valid_lower_body_joint_count": d1_valid_count - c3_valid_count,
            }
        )

    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_roi_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_joint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in point_rows:
        by_condition[row["condition"]].append(row)
        by_roi_stratum[row["roi_change_stratum"]].append(row)
        by_joint[row["joint"]].append(row)
    result = {
        "purpose": (
            "Paired C3/D1 lower-body geometry audit using saved predictions and "
            "the frozen raw-fisheye calibration replay."
        ),
        "frames": len(pair_rows),
        "lower_body_joints": list(LOWER_BODY),
        "pair_outcomes": {
            "c3": dict(Counter(row["c3_pair_status"] for row in pair_rows)),
            "d1": dict(Counter(row["d1_pair_status"] for row in pair_rows)),
        },
        "overall": _metrics(point_rows),
        "by_condition": {
            key: _metrics(rows) for key, rows in sorted(by_condition.items())
        },
        "by_roi_change_stratum": {
            key: _metrics(rows) for key, rows in sorted(by_roi_stratum.items())
        },
        "by_joint": {key: _metrics(rows) for key, rows in sorted(by_joint.items())},
        "interpretation_boundary": (
            "Valid-point counts and reprojection residuals measure internal "
            "consistency of saved 2-D observations with the frozen calibration. "
            "They are not human 2-D labels, metric 3-D accuracy, gait validity, "
            "or evidence that DA3 depth entered geometry."
        ),
    }
    (args.output_dir / "paired_lower_body_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    for name, rows in (("paired_lower_body_per_joint.csv", point_rows), ("paired_lower_body_per_pair.csv", pair_rows)):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
