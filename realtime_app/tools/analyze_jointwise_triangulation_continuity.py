#!/usr/bin/env python3
"""Audit observed lower-limb segment continuity without changing stereo data.

The input is the ``per_joint_geometry_diagnostic.csv`` written by
``experiment_single_person_jointwise_geometry.py``.  A segment is observed
only when both of its endpoints are already marked ``direct_valid`` in the
same frame.  A frame-to-frame length change is reported only for two directly
adjacent pair IDs whose segment observations both exist.  Missing endpoints,
rejections, and gaps are written out; no coordinate is interpolated, smoothed,
filtered, or re-triangulated here.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


LOWER_BODY = (
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
BONES = {
    "left_thigh": ("left_hip", "left_knee"),
    "right_thigh": ("right_hip", "right_knee"),
    "left_shank": ("left_knee", "left_ankle"),
    "right_shank": ("right_knee", "right_ankle"),
}
COLORS = ("#1f4e79", "#c65d00", "#558b2f", "#7b1fa2", "#00838f")
REQUIRED_COLUMNS = {
    "pair_id",
    "file_name",
    "joint",
    "direct_valid",
    "direct_reason",
    "direct_xyz_left_camera_mm",
}


@dataclass(frozen=True)
class JointObservation:
    pair_id: int
    file_name: str
    joint: str
    valid: bool
    reason: str | None
    xyz: np.ndarray | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        metavar="LABEL=CSV",
        help="One ordered-model diagnostic CSV; repeat for each baseline.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def parse_bool(value: str, *, field: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Row {row_number}: {field} must be True or False, got {value!r}")


def parse_xyz(value: str, *, valid: bool, row_number: int) -> np.ndarray | None:
    if not value:
        if valid:
            raise ValueError(f"Row {row_number}: direct-valid point has no XYZ")
        return None
    try:
        parsed = np.asarray(json.loads(value), dtype=np.float64)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"Row {row_number}: invalid direct XYZ") from error
    if parsed.shape != (3,) or not np.all(np.isfinite(parsed)):
        if valid:
            raise ValueError(f"Row {row_number}: direct-valid point has non-finite XYZ")
        return None
    return parsed if valid else None


def parse_condition(specification: str) -> tuple[str, Path]:
    label, separator, path_text = specification.partition("=")
    label = label.strip()
    if not separator or not label or not path_text.strip():
        raise ValueError(f"Invalid --condition {specification!r}; expected LABEL=CSV")
    if any(character in label for character in "/\\"):
        raise ValueError(f"Condition label must not contain a path separator: {label!r}")
    return label, Path(path_text.strip()).resolve()


def read_condition(path: Path) -> dict[int, dict[str, JointObservation]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"{path}: missing required columns {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no diagnostic rows")

    by_pair: dict[int, dict[str, JointObservation]] = {}
    file_names: dict[int, str] = {}
    for row_number, row in enumerate(rows, start=2):
        try:
            pair_id = int(row["pair_id"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Row {row_number}: pair_id must be an integer") from error
        file_name = row["file_name"].strip()
        joint = row["joint"].strip()
        if not file_name or joint not in LOWER_BODY:
            raise ValueError(f"Row {row_number}: unexpected file_name or lower-body joint")
        known_name = file_names.setdefault(pair_id, file_name)
        if known_name != file_name:
            raise ValueError(f"Pair {pair_id}: inconsistent file_name values")
        valid = parse_bool(row["direct_valid"], field="direct_valid", row_number=row_number)
        reason = row["direct_reason"].strip() or None
        if valid and reason is not None:
            raise ValueError(f"Row {row_number}: a direct-valid point cannot have a rejection reason")
        xyz = parse_xyz(row["direct_xyz_left_camera_mm"], valid=valid, row_number=row_number)
        observations = by_pair.setdefault(pair_id, {})
        if joint in observations:
            raise ValueError(f"Pair {pair_id}: duplicate lower-body joint {joint}")
        observations[joint] = JointObservation(pair_id, file_name, joint, valid, reason, xyz)

    for pair_id, observations in by_pair.items():
        missing_joints = set(LOWER_BODY) - set(observations)
        if missing_joints:
            raise ValueError(f"Pair {pair_id}: missing lower-body joints {sorted(missing_joints)}")
    return by_pair


def observed_run_summary(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """Count observed runs while treating a pair-ID gap as a hard break."""

    runs = longest = current = 0
    previous_pair_id: int | None = None
    previous_observed = False
    for row in rows:
        pair_id = int(row["pair_id"])
        observed = bool(row["segment_observed"])
        continues = (
            observed
            and previous_observed
            and previous_pair_id is not None
            and pair_id == previous_pair_id + 1
        )
        if observed:
            if continues:
                current += 1
            else:
                runs += 1
                current = 1
            longest = max(longest, current)
        else:
            current = 0
        previous_pair_id = pair_id
        previous_observed = observed
    return runs, longest


def numeric_summary(values: list[float]) -> dict[str, int | float | None]:
    if not values:
        return {"n": 0, "median": None, "mad": None, "p10": None, "p90": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    return {
        "n": int(array.size),
        "median": median,
        "mad": float(np.median(np.abs(array - median))),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def rejection_label(observation: JointObservation) -> str:
    return "accepted" if observation.valid else (observation.reason or "missing_rejection_reason")


def same_sequence(conditions: dict[str, dict[int, dict[str, JointObservation]]]) -> list[int]:
    labels = list(conditions)
    reference_label = labels[0]
    reference = conditions[reference_label]
    reference_keys = set(reference)
    reference_names = {pair_id: items[LOWER_BODY[0]].file_name for pair_id, items in reference.items()}
    for label, current in conditions.items():
        if label == reference_label:
            continue
        if set(current) != reference_keys:
            raise ValueError(f"{label}: pair IDs do not exactly match {reference_label}")
        for pair_id in reference_keys:
            if current[pair_id][LOWER_BODY[0]].file_name != reference_names[pair_id]:
                raise ValueError(f"{label}: pair {pair_id} file_name does not match {reference_label}")
    return sorted(reference_keys)


def analyze_conditions(
    conditions: dict[str, dict[int, dict[str, JointObservation]]], fps: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, list[dict[str, Any]]]]]:
    """Return frame, segment-summary, rejection-summary rows and plot series."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    pair_ids = same_sequence(conditions)
    frame_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    series: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for label, by_pair in conditions.items():
        series[label] = {}
        reasons: Counter[tuple[str, str]] = Counter()
        for pair_id in pair_ids:
            for joint in LOWER_BODY:
                observation = by_pair[pair_id][joint]
                reasons[(joint, rejection_label(observation))] += 1

        for (joint, outcome), count in sorted(reasons.items()):
            rejection_rows.append({
                "condition": label,
                "joint": joint,
                "outcome": outcome,
                "frames": count,
            })

        for segment, (start_joint, end_joint) in BONES.items():
            segment_rows: list[dict[str, Any]] = []
            lengths: list[float] = []
            absolute_deltas: list[float] = []
            relative_absolute_deltas: list[float] = []
            signed_deltas: list[float] = []
            previous_pair_id: int | None = None
            previous_observed = False
            previous_length: float | None = None

            for pair_id in pair_ids:
                start = by_pair[pair_id][start_joint]
                end = by_pair[pair_id][end_joint]
                observed = start.valid and end.valid
                length = float(np.linalg.norm(start.xyz - end.xyz)) if observed else None
                is_adjacent_pair_id = previous_pair_id is not None and pair_id == previous_pair_id + 1
                adjacent_observed = bool(is_adjacent_pair_id and previous_observed and observed)
                signed_delta = length - previous_length if adjacent_observed else None
                absolute_delta = abs(signed_delta) if signed_delta is not None else None
                relative_absolute_delta = (
                    100.0 * absolute_delta / previous_length
                    if absolute_delta is not None and previous_length is not None and previous_length > 0.0
                    else None
                )
                if length is not None:
                    lengths.append(length)
                if signed_delta is not None:
                    signed_deltas.append(signed_delta)
                    absolute_deltas.append(absolute_delta)
                if relative_absolute_delta is not None:
                    relative_absolute_deltas.append(relative_absolute_delta)
                frame_rows.append({
                    "condition": label,
                    "pair_id": pair_id,
                    "file_name": start.file_name,
                    "segment": segment,
                    "start_joint": start_joint,
                    "end_joint": end_joint,
                    "start_valid": start.valid,
                    "start_outcome": rejection_label(start),
                    "end_valid": end.valid,
                    "end_outcome": rejection_label(end),
                    "segment_observed": observed,
                    "segment_length_mm": length,
                    "directly_adjacent_pair_id": is_adjacent_pair_id,
                    "previous_segment_observed": previous_observed if is_adjacent_pair_id else False,
                    "adjacent_observed_pair": adjacent_observed,
                    "frame_to_frame_length_delta_mm": signed_delta,
                    "frame_to_frame_abs_length_delta_mm": absolute_delta,
                    "frame_to_frame_relative_abs_length_delta_pct": relative_absolute_delta,
                })
                segment_rows.append(frame_rows[-1])
                previous_pair_id = pair_id
                previous_observed = observed
                previous_length = length

            observed_runs, longest_observed_run = observed_run_summary(segment_rows)
            length_stats = numeric_summary(lengths)
            signed_stats = numeric_summary(signed_deltas)
            absolute_stats = numeric_summary(absolute_deltas)
            relative_stats = numeric_summary(relative_absolute_deltas)
            summary_rows.append({
                "condition": label,
                "segment": segment,
                "start_joint": start_joint,
                "end_joint": end_joint,
                "sequence_frames": len(pair_ids),
                "observed_segment_frames": len(lengths),
                "segment_observation_rate": len(lengths) / len(pair_ids),
                "unobserved_segment_frames": len(pair_ids) - len(lengths),
                "observed_segment_runs": observed_runs,
                "longest_observed_segment_run_frames": longest_observed_run,
                "possible_adjacent_frame_pairs": max(0, len(pair_ids) - 1),
                "adjacent_observed_segment_pairs": len(absolute_deltas),
                "unobservable_adjacent_pairs": max(0, len(pair_ids) - 1) - len(absolute_deltas),
                "length_n": length_stats["n"],
                "length_median_mm": length_stats["median"],
                "length_mad_mm": length_stats["mad"],
                "length_p10_mm": length_stats["p10"],
                "length_p90_mm": length_stats["p90"],
                "length_max_mm": length_stats["max"],
                "signed_delta_n": signed_stats["n"],
                "signed_delta_median_mm": signed_stats["median"],
                "absolute_delta_n": absolute_stats["n"],
                "absolute_delta_median_mm": absolute_stats["median"],
                "absolute_delta_mad_mm": absolute_stats["mad"],
                "absolute_delta_p90_mm": absolute_stats["p90"],
                "absolute_delta_max_mm": absolute_stats["max"],
                "relative_absolute_delta_n": relative_stats["n"],
                "relative_absolute_delta_median_pct": relative_stats["median"],
                "relative_absolute_delta_p90_pct": relative_stats["p90"],
                "relative_absolute_delta_max_pct": relative_stats["max"],
            })
            series[label][segment] = segment_rows
    return frame_rows, summary_rows, rejection_rows, series


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_timeseries(
    series: dict[str, dict[str, list[dict[str, Any]]]], output: Path, *, delta: bool
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True, constrained_layout=True)
    value_key = "frame_to_frame_abs_length_delta_mm" if delta else "segment_length_mm"
    ylabel = "adjacent absolute length change (mm)" if delta else "observed segment length (mm)"
    for axis, segment in zip(axes.flat, BONES):
        for index, (label, by_segment) in enumerate(series.items()):
            rows = by_segment[segment]
            x = [row["pair_id"] for row in rows if row[value_key] is not None]
            y = [row[value_key] for row in rows if row[value_key] is not None]
            if delta:
                axis.scatter(x, y, s=11, color=COLORS[index % len(COLORS)], label=label)
            else:
                axis.plot(x, y, linewidth=1.05, color=COLORS[index % len(COLORS)], label=label)
        axis.set_title(segment.replace("_", " "))
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("ordered stereo-pair ID")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=len(handles))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    condition_specs = [parse_condition(value) for value in args.condition]
    labels = [label for label, _ in condition_specs]
    if len(labels) != len(set(labels)):
        raise ValueError("Each --condition label must be unique")
    conditions = {label: read_condition(path) for label, path in condition_specs}
    frame_rows, summary_rows, rejection_rows, series = analyze_conditions(conditions, args.fps)

    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "segment_continuity_per_frame.csv", frame_rows)
    write_csv(args.output_dir / "segment_continuity_summary.csv", summary_rows)
    write_csv(args.output_dir / "joint_observation_outcomes.csv", rejection_rows)
    render_timeseries(series, args.output_dir / "segment_length_timeseries.png", delta=False)
    render_timeseries(series, args.output_dir / "segment_frame_to_frame_abs_delta.png", delta=True)

    pair_ids = same_sequence(conditions)
    metadata = {
        "conditions": {label: str(path) for label, path in condition_specs},
        "frames": len(pair_ids),
        "fps": args.fps,
        "coordinate_frame": "left_camera",
        "length_unit": "mm",
        "segment_definitions": BONES,
        "observation_rule": "A segment is observed only when both endpoints are already direct_valid in the same stored frame.",
        "frame_to_frame_rule": "A length change is reported only when pair IDs differ by exactly one and the segment is observed in both frames; no gap is bridged.",
        "coordinate_mutation": "none",
        "selection_or_rejection_from_bone_length": "none",
        "interpretation": "Length and adjacent-frame changes are within-sequence triangulation-continuity diagnostics. They are not bone-length truth, gait accuracy, or external 3-D accuracy metrics.",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_dir), "frames": len(pair_ids), "conditions": labels}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
