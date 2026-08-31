#!/usr/bin/env python3
"""Summarize ordered PMPose lower-body and Sapiens2 foot stereo results.

The program does not alter coordinates or fill missing points.  It writes the
per-frame acceptance state, geometric residual summaries, bone-length
stability, and a compact time-series figure for a frozen sequence run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PRIMARY = ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
FOOT = ("left_ankle", "left_big_toe", "left_small_toe", "left_heel", "right_ankle", "right_big_toe", "right_small_toe", "right_heel")
BONES = {
    "left_thigh": ("left_hip", "left_knee"),
    "right_thigh": ("right_hip", "right_knee"),
    "left_shank": ("left_knee", "left_ankle"),
    "right_shank": ("right_knee", "right_ankle"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pmpose-jsonl", required=True, type=Path)
    parser.add_argument("--foot-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> dict[str, dict]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[row["file_name"]] = row
    return result


def values(items: list[float]) -> dict[str, float | int | None]:
    if not items:
        return {"n": 0, "median": None, "mad": None, "p90": None}
    array = np.asarray(items, dtype=float)
    median = float(np.median(array))
    return {
        "n": int(array.size),
        "median": median,
        "mad": float(np.median(np.abs(array - median))),
        "p90": float(np.percentile(array, 90)),
    }


def longest_invalid_run(valid: list[bool]) -> int:
    best = current = 0
    for item in valid:
        if item:
            current = 0
        else:
            current += 1
            best = max(best, current)
    return best


def point_map(record: dict) -> dict[str, dict]:
    persons = record.get("persons_3d", [])
    if not persons:
        return {}
    return {point["name"]: point for point in persons[0].get("keypoints_3d", [])}


def vector(point: dict | None) -> np.ndarray | None:
    if not point or not point.get("valid"):
        return None
    xyz = point.get("xyz") or point.get("xyz_left_camera")
    if not isinstance(xyz, list) or len(xyz) != 3:
        return None
    array = np.asarray(xyz, dtype=float)
    return array if np.all(np.isfinite(array)) else None


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    manifest = read_csv(args.manifest.resolve())
    pmpose_by_name = read_jsonl(args.pmpose_jsonl.resolve())
    foot_by_name = read_jsonl(args.foot_jsonl.resolve())
    expected = [row["file_name"] for row in manifest]
    if set(expected) != set(pmpose_by_name) or set(expected) != set(foot_by_name):
        raise RuntimeError("Manifest, PMPose and foot file names must match exactly")

    output.mkdir(parents=True)
    primary_valid = defaultdict(list); foot_valid = defaultdict(list)
    primary_errors = defaultdict(list); foot_errors = defaultdict(list)
    bone_values = defaultdict(list)
    frame_rows = []
    pmpose_error_series, foot_error_series, primary_count_series, foot_count_series = [], [], [], []
    previous_xyz: dict[str, np.ndarray] = {}
    speed_values = defaultdict(list)

    for sequence_index, meta in enumerate(manifest):
        name = meta["file_name"]
        primary = point_map(pmpose_by_name[name])
        foot_records = {point["name"]: point for point in foot_by_name[name].get("foot_points", [])}
        primary_errors_frame, foot_errors_frame = [], []
        primary_count, foot_count = 0, 0
        vectors = {}
        for joint in PRIMARY:
            point = primary.get(joint)
            valid = bool(point and point.get("valid"))
            primary_valid[joint].append(valid)
            xyz = vector(point)
            if valid and xyz is not None:
                primary_count += 1; vectors[joint] = xyz
                error = point.get("reprojection_error_mean_px")
                if error is not None:
                    primary_errors[joint].append(float(error)); primary_errors_frame.append(float(error))
                if joint in previous_xyz:
                    speed_values[joint].append(float(np.linalg.norm(xyz - previous_xyz[joint]) * args.fps))
                previous_xyz[joint] = xyz
            else:
                previous_xyz.pop(joint, None)
        for joint in FOOT:
            point = foot_records.get(joint)
            valid = bool(point and point.get("valid_at_reprojection_gate"))
            foot_valid[joint].append(valid)
            if valid:
                foot_count += 1
                error = point.get("reprojection_error_mean_px")
                if error is not None:
                    foot_errors[joint].append(float(error)); foot_errors_frame.append(float(error))
        for bone, (start, end) in BONES.items():
            if start in vectors and end in vectors:
                bone_values[bone].append(float(np.linalg.norm(vectors[start] - vectors[end])))

        pmpose_error_series.append(float(np.median(primary_errors_frame)) if primary_errors_frame else math.nan)
        foot_error_series.append(float(np.median(foot_errors_frame)) if foot_errors_frame else math.nan)
        primary_count_series.append(primary_count); foot_count_series.append(foot_count)
        frame_rows.append({
            "sequence_index": sequence_index,
            "file_name": name,
            "source_pair_index": meta.get("source_pair_index"),
            "condition": meta.get("condition"),
            "abs_host_delta_ms": meta.get("abs_host_delta_ms"),
            "pmpose_valid_lower_body_points": primary_count,
            "sapiens2_valid_foot_points": foot_count,
            "pmpose_median_reprojection_px": pmpose_error_series[-1],
            "sapiens2_median_reprojection_px": foot_error_series[-1],
        })

    point_rows = []
    for source, joints, validity, errors in (("PMPose", PRIMARY, primary_valid, primary_errors), ("Sapiens2", FOOT, foot_valid, foot_errors)):
        for joint in joints:
            accepted = sum(validity[joint])
            row = {
                "source": source,
                "joint": joint,
                "frames": len(validity[joint]),
                "accepted_frames": accepted,
                "accepted_rate": accepted / len(validity[joint]),
                "longest_invalid_run_frames": longest_invalid_run(validity[joint]),
                **{f"reprojection_{key}": value for key, value in values(errors[joint]).items()},
            }
            point_rows.append(row)
    bone_rows = [{"bone": name, **values(data)} for name, data in bone_values.items()]
    speed_rows = [{"joint": name, **values(data)} for name, data in speed_values.items()]

    def write(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    write(output / "frame_quality.csv", frame_rows)
    write(output / "joint_quality_summary.csv", point_rows)
    write(output / "bone_length_summary.csv", bone_rows)
    write(output / "joint_speed_summary.csv", speed_rows)

    x = np.arange(len(manifest))
    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    axes[0].plot(x, primary_count_series, color="#1f4e79", linewidth=1.2, label="PMPose lower body (max 6)")
    axes[0].plot(x, foot_count_series, color="#b45f06", linewidth=1.2, label="Sapiens2 foot (max 8)")
    axes[0].set_ylabel("accepted points"); axes[0].set_ylim(-0.3, 8.3); axes[0].legend(loc="lower right", fontsize=8); axes[0].grid(alpha=0.25)
    axes[1].plot(x, pmpose_error_series, color="#1f4e79", linewidth=1.0, label="PMPose median")
    axes[1].plot(x, foot_error_series, color="#b45f06", linewidth=1.0, label="Sapiens2 median")
    axes[1].axhline(10.0, color="#9e0000", linestyle="--", linewidth=0.9, label="gate")
    axes[1].set_ylabel("reprojection (px)"); axes[1].set_ylim(bottom=0); axes[1].legend(loc="upper right", fontsize=8); axes[1].grid(alpha=0.25)
    for name, data in bone_values.items():
        axes[2].plot(np.arange(len(data)), data, linewidth=0.9, label=name)
    axes[2].set_ylabel("bone length (mm)"); axes[2].set_xlabel("ordered stereo pair"); axes[2].legend(ncol=2, fontsize=8); axes[2].grid(alpha=0.25)
    figure.savefig(output / "quality_timeseries.png", dpi=180)
    plt.close(figure)

    summary = {
        "frames": len(manifest), "fps": args.fps, "coordinate_frame": "left_camera", "length_unit": "mm",
        "sources": {"pmpose": str(args.pmpose_jsonl.resolve()), "sapiens2_foot": str(args.foot_jsonl.resolve())},
        "acceptance": "2-D score threshold, positive depth and 10 px reprojection gate are inherited from the frozen stereo runs; rejected points remain missing.",
        "note": "Bone-length and frame-to-frame speed statistics are stability diagnostics for this sequence, not absolute gait-accuracy measurements.",
    }
    (output / "metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frames": len(manifest), "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
