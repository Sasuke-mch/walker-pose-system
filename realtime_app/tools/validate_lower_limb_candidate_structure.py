#!/usr/bin/env python3
"""Audit a near-distance lower-limb candidate against frozen observations.

The candidate input is a per-joint direct-fisheye-triangulation table.  This
tool does not select, modify, smooth, or fill any coordinate.  It measures
leg-segment lengths when both endpoints were observed, compares their range
with the same subject's reliable mid-distance PMPose sequence, and reports
joint-wise distance to PMPose only where PMPose is either an old baseline
observation or a temporally confirmed new observation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


JOINTS = ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
SEGMENTS = {
    "left_thigh": ("left_hip", "left_knee"),
    "right_thigh": ("right_hip", "right_knee"),
    "left_shank": ("left_knee", "left_ankle"),
    "right_shank": ("right_knee", "right_ankle"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--mid-reference-jsonl", type=Path, required=True)
    parser.add_argument("--pmpose-csv", type=Path, required=True)
    parser.add_argument("--pmpose-temporal-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-low-percentile", type=float, default=5.0)
    parser.add_argument("--reference-high-percentile", type=float, default=95.0)
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def read_per_joint(path: Path) -> dict[tuple[int, str], dict]:
    output: dict[tuple[int, str], dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["pair_id"]), row["joint"])
            if key in output:
                raise RuntimeError(f"Duplicate row {key}")
            valid = parse_bool(row["direct_valid"])
            xyz = json.loads(row["direct_xyz_left_camera_mm"]) if row["direct_xyz_left_camera_mm"] else None
            output[key] = {
                "valid": valid,
                "baseline_valid": parse_bool(row["baseline_valid"]),
                "newly_recovered": parse_bool(row["newly_recovered"]),
                "xyz": None if xyz is None or not valid else np.asarray(xyz, dtype=np.float64),
                "file_name": row["file_name"],
                "reprojection_error_px": None if not row["direct_reprojection_error_px"] else float(row["direct_reprojection_error_px"]),
            }
    return output


def read_temporal_passes(path: Path) -> set[tuple[int, str]]:
    output: set[tuple[int, str]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if parse_bool(row["within_baseline_temporal_envelope"]):
                output.add((int(row["pair_id"]), row["joint"]))
    return output


def read_mid_reference(path: Path) -> dict[tuple[int, str], np.ndarray]:
    output: dict[tuple[int, str], np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        people = row.get("persons_3d", [])
        if len(people) != 1:
            continue
        for point in people[0].get("keypoints_3d", []):
            if point.get("name") in JOINTS and point.get("valid") and point.get("xyz") is not None:
                output[(int(row["pair_id"]), point["name"])] = np.asarray(point["xyz"], dtype=np.float64)
    return output


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else None


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def render_segment_plot(reference: dict[str, list[float]], candidate: dict[str, list[float]], bounds: dict[str, tuple[float, float]], output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = list(SEGMENTS)
    fig, axis = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    y = np.arange(len(labels))
    reference_median = [float(np.median(reference[label])) for label in labels]
    lower = [bounds[label][0] for label in labels]
    upper = [bounds[label][1] for label in labels]
    candidate_median = [float(np.median(candidate[label])) if candidate[label] else np.nan for label in labels]
    axis.errorbar(reference_median, y, xerr=[np.asarray(reference_median) - np.asarray(lower), np.asarray(upper) - np.asarray(reference_median)], fmt="o", color="#3f6e99", capsize=4, label="mid PMPose P5–P95")
    axis.scatter(candidate_median, y, marker="D", s=55, color="#f1a340", label="near Sapiens2 median")
    axis.set_yticks(y, [label.replace("_", " ") for label in labels])
    axis.set_xlabel("segment length (mm)")
    axis.set_title("Observed segment-length comparison")
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not 0 <= args.reference_low_percentile < args.reference_high_percentile <= 100:
        raise ValueError("Invalid reference percentiles")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    candidate = read_per_joint(args.candidate_csv.resolve())
    pmpose = read_per_joint(args.pmpose_csv.resolve())
    temporal_passes = read_temporal_passes(args.pmpose_temporal_csv.resolve())
    reference = read_mid_reference(args.mid_reference_jsonl.resolve())
    candidate_frame_ids = sorted({frame_id for frame_id, _ in candidate})

    reference_lengths: dict[str, list[float]] = defaultdict(list)
    for frame_id in sorted({frame for frame, _ in reference}):
        for segment, (start, end) in SEGMENTS.items():
            a, b = reference.get((frame_id, start)), reference.get((frame_id, end))
            if a is not None and b is not None:
                reference_lengths[segment].append(distance(a, b))
    bounds = {
        segment: (percentile(values, args.reference_low_percentile), percentile(values, args.reference_high_percentile))
        for segment, values in reference_lengths.items()
    }
    if set(bounds) != set(SEGMENTS):
        raise RuntimeError("The mid reference has insufficient valid segment observations")

    per_frame_rows: list[dict] = []
    candidate_lengths: dict[str, list[float]] = defaultdict(list)
    segment_within: dict[str, list[bool]] = defaultdict(list)
    complete_leg = {"left": 0, "right": 0}
    for frame_id in candidate_frame_ids:
        row = {"pair_id": frame_id, "file_name": candidate[(frame_id, "left_hip")]["file_name"]}
        valid_count = 0
        for joint in JOINTS:
            valid = bool(candidate.get((frame_id, joint), {}).get("valid"))
            row[f"{joint}_valid"] = valid
            valid_count += valid
        row["candidate_valid_hka_count"] = valid_count
        for segment, (start, end) in SEGMENTS.items():
            a = candidate.get((frame_id, start), {}).get("xyz")
            b = candidate.get((frame_id, end), {}).get("xyz")
            value = distance(a, b) if a is not None and b is not None else None
            lower, upper = bounds[segment]
            within = bool(value is not None and lower <= value <= upper)
            row[f"{segment}_mm"] = value
            row[f"{segment}_within_mid_p{args.reference_low_percentile:g}_p{args.reference_high_percentile:g}"] = within
            if value is not None:
                candidate_lengths[segment].append(value)
                segment_within[segment].append(within)
        for side in ("left", "right"):
            full = all(candidate.get((frame_id, f"{side}_{name}"), {}).get("xyz") is not None for name in ("hip", "knee", "ankle"))
            row[f"{side}_complete_hka"] = full
            complete_leg[side] += full
        per_frame_rows.append(row)

    cross_rows: list[dict] = []
    cross_values: dict[str, list[float]] = defaultdict(list)
    for (frame_id, joint), candidate_point in candidate.items():
        if joint not in JOINTS or not candidate_point["valid"] or candidate_point["xyz"] is None:
            continue
        pmpose_point = pmpose.get((frame_id, joint))
        trusted_pmpose = bool(
            pmpose_point
            and pmpose_point["valid"]
            and (pmpose_point["baseline_valid"] or (frame_id, joint) in temporal_passes)
            and pmpose_point["xyz"] is not None
        )
        value = distance(candidate_point["xyz"], pmpose_point["xyz"]) if trusted_pmpose else None
        if value is not None:
            cross_values[joint].append(value)
        cross_rows.append({
            "pair_id": frame_id,
            "file_name": candidate_point["file_name"],
            "joint": joint,
            "sapiens2_valid": True,
            "pmpose_trusted_observed": trusted_pmpose,
            "distance_mm": value,
            "sapiens2_reprojection_error_px": candidate_point["reprojection_error_px"],
            "pmpose_reprojection_error_px": pmpose_point["reprojection_error_px"] if pmpose_point else None,
        })

    segment_summary = []
    for segment in SEGMENTS:
        values = candidate_lengths[segment]
        lower, upper = bounds[segment]
        segment_summary.append({
            "segment": segment,
            "mid_reference_count": len(reference_lengths[segment]),
            f"mid_p{args.reference_low_percentile:g}_mm": lower,
            f"mid_p{args.reference_high_percentile:g}_mm": upper,
            "sapiens2_observed_count": len(values),
            "sapiens2_median_mm": float(np.median(values)) if values else None,
            "sapiens2_p10_mm": percentile(values, 10),
            "sapiens2_p90_mm": percentile(values, 90),
            "within_mid_reference_count": sum(segment_within[segment]),
            "within_mid_reference_fraction": sum(segment_within[segment]) / len(values) if values else None,
        })
    cross_summary = [
        {
            "joint": joint,
            "shared_trusted_frames": len(values),
            "median_distance_mm": float(np.median(values)) if values else None,
            "p90_distance_mm": percentile(values, 90),
        }
        for joint, values in sorted(cross_values.items())
    ]

    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "sapiens2_near_per_frame_structure.csv", per_frame_rows)
    write_csv(args.output_dir / "segment_structure_summary.csv", segment_summary)
    write_csv(args.output_dir / "sapiens2_to_pmpose_trusted_joint_distance.csv", cross_rows)
    if cross_summary:
        write_csv(args.output_dir / "sapiens2_to_pmpose_trusted_joint_distance_summary.csv", cross_summary)
    render_segment_plot(reference_lengths, candidate_lengths, bounds, args.output_dir / "segment_length_comparison.png")
    metadata = {
        "candidate_csv": str(args.candidate_csv.resolve()),
        "mid_reference_jsonl": str(args.mid_reference_jsonl.resolve()),
        "pmpose_csv": str(args.pmpose_csv.resolve()),
        "pmpose_temporal_csv": str(args.pmpose_temporal_csv.resolve()),
        "coordinate_frame": "left_camera",
        "length_unit": "mm",
        "reference_range": f"mid PMPose P{args.reference_low_percentile:g}–P{args.reference_high_percentile:g}",
        "interpretation": "Segment agreement and cross-model distance are internal compatibility checks. They do not establish external 3-D accuracy and do not justify averaging or replacing PMPose coordinates.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = f"""# Sapiens2 近距离下肢候选结构检查\n\n本检查使用已保存的 Sapiens2 髋、膝、踝逐点鱼眼三角化结果。坐标不被修改；只有两个端点都已通过原有二维分数、正深度和重投影门限时才计算腿段长度。\n\n中距离 PMPose 同一受试者的 P{args.reference_low_percentile:g}–P{args.reference_high_percentile:g} 范围仅用来发现明显不相容的长度。近距离 Sapiens2 与 PMPose 的关节距离只在 PMPose 为旧流程有效点，或新增点已通过邻帧筛查时计算。两项均是内部一致性检查，不能代替三维真值，也不触发跨模型平均或替换。\n\n- `segment_structure_summary.csv`：腿段长度与中距离范围对照。\n- `sapiens2_near_per_frame_structure.csv`：逐帧髋、膝、踝可用情况和腿段长度。\n- `sapiens2_to_pmpose_trusted_joint_distance*.csv`：共同可靠观测的跨模型距离。\n- `segment_length_comparison.png`：腿段长度对照图。\n"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "frames": len(per_frame_rows), "complete_left_hka": complete_leg["left"], "complete_right_hka": complete_leg["right"], "cross_model_rows": sum(len(values) for values in cross_values.values())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
