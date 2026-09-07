#!/usr/bin/env python3
"""Screen jointwise stereo candidates against immediately adjacent observations.

The tool never changes a 3-D coordinate and never fills missing frames.  A
point rejected by the old whole-person association is merely *screened* when
the same joint has valid, old-baseline observations in both adjacent frames.
Its distance to the two-neighbour linear prediction is compared with that
joint's baseline three-frame residual distribution.  The 95th percentile of
that fixed distribution is reported as a descriptive threshold.

This distinguishes two cases that have identical single-frame reprojection
errors: an isolated geometrically plausible point, and a point compatible with
the surrounding observed trajectory.  It is not a gait model or interpolation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-joint-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quantile", type=float, default=95.0)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("No per-joint rows")
    for row in rows:
        row["pair_id"] = int(row["pair_id"])
        for key in ("baseline_valid", "direct_valid", "newly_recovered"):
            row[key] = row[key].lower() == "true"
        row["xyz"] = None if not row["direct_xyz_left_camera_mm"] else np.asarray(json.loads(row["direct_xyz_left_camera_mm"]), dtype=np.float64)
    return rows


def valid_xyz(row: dict, *, baseline_only: bool = False) -> np.ndarray | None:
    if not row["direct_valid"] or row["xyz"] is None:
        return None
    if baseline_only and not row["baseline_valid"]:
        return None
    return row["xyz"]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def render_plot(reference: dict[str, list[float]], candidate: dict[str, list[float]], thresholds: dict[str, float], output: Path) -> None:
    import matplotlib.pyplot as plt

    joints = list(reference)
    fig, axes = plt.subplots(1, len(joints), figsize=(3.1 * len(joints), 4.1), sharey=True, constrained_layout=True)
    if len(joints) == 1:
        axes = [axes]
    for axis, joint in zip(axes, joints):
        baseline_values = reference[joint]
        candidate_values = candidate[joint]
        bins = max(4, min(15, len(baseline_values) // 2 + 1))
        if baseline_values:
            axis.hist(baseline_values, bins=bins, color="#5b82a9", alpha=0.78, label="baseline 3-frame residual")
            axis.axvline(thresholds[joint], color="#c62828", linewidth=1.8, label="baseline q threshold")
        if candidate_values:
            axis.scatter(candidate_values, np.zeros(len(candidate_values)), color="#f1a340", zorder=4, label="new candidate")
        axis.set_title(joint.replace("_", " "))
        axis.set_xlabel("two-neighbour residual (mm)")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("frames")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not 50.0 <= args.quantile < 100.0:
        raise ValueError("quantile must be in [50, 100)")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    rows = read_rows(args.per_joint_csv.resolve())
    by_joint_frame: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in rows:
        if row["pair_id"] in by_joint_frame[row["joint"]]:
            raise RuntimeError(f"Duplicate joint/frame row: {row['joint']} {row['pair_id']}")
        by_joint_frame[row["joint"]][row["pair_id"]] = row

    reference: dict[str, list[float]] = defaultdict(list)
    for joint, frames in by_joint_frame.items():
        for frame_id, row in frames.items():
            before = valid_xyz(frames.get(frame_id - 1, {}), baseline_only=True) if frame_id - 1 in frames else None
            current = valid_xyz(row, baseline_only=True)
            after = valid_xyz(frames.get(frame_id + 1, {}), baseline_only=True) if frame_id + 1 in frames else None
            if before is not None and current is not None and after is not None:
                reference[joint].append(float(np.linalg.norm(current - 0.5 * (before + after))))
    thresholds = {joint: float(np.percentile(values, args.quantile)) for joint, values in reference.items() if values}

    candidate_rows: list[dict] = []
    candidate_values: dict[str, list[float]] = defaultdict(list)
    for joint, frames in by_joint_frame.items():
        for frame_id, row in frames.items():
            if not row["newly_recovered"]:
                continue
            before = valid_xyz(frames.get(frame_id - 1, {}), baseline_only=True) if frame_id - 1 in frames else None
            after = valid_xyz(frames.get(frame_id + 1, {}), baseline_only=True) if frame_id + 1 in frames else None
            current = valid_xyz(row)
            bridge_available = before is not None and after is not None and current is not None
            residual = float(np.linalg.norm(current - 0.5 * (before + after))) if bridge_available else None
            if residual is not None:
                candidate_values[joint].append(residual)
            threshold = thresholds.get(joint)
            candidate_rows.append({
                "pair_id": frame_id,
                "file_name": row["file_name"],
                "joint": joint,
                "baseline_neighbours_valid": bridge_available,
                "two_neighbour_residual_mm": residual,
                f"baseline_q{args.quantile:g}_residual_mm": threshold,
                "within_baseline_temporal_envelope": bool(residual is not None and threshold is not None and residual <= threshold),
                "single_frame_reprojection_error_px": row["direct_reprojection_error_px"],
                "direct_to_sapiens2_ankle_distance_mm": row["direct_to_sapiens2_ankle_distance_mm"],
            })
    if not candidate_rows:
        raise RuntimeError("No newly recovered points were available")

    summary_rows: list[dict] = []
    for joint in sorted(by_joint_frame):
        candidates = [row for row in candidate_rows if row["joint"] == joint]
        bridged = [row for row in candidates if row["baseline_neighbours_valid"]]
        passed = [row for row in bridged if row["within_baseline_temporal_envelope"]]
        reference_values = reference.get(joint, [])
        summary_rows.append({
            "joint": joint,
            "baseline_three_frame_reference_count": len(reference_values),
            f"baseline_q{args.quantile:g}_residual_mm": thresholds.get(joint),
            "newly_recovered_count": len(candidates),
            "newly_recovered_with_baseline_neighbours": len(bridged),
            "within_baseline_temporal_envelope": len(passed),
            "candidate_residual_median_mm": float(np.median([row["two_neighbour_residual_mm"] for row in bridged])) if bridged else None,
            "candidate_residual_p90_mm": float(np.percentile([row["two_neighbour_residual_mm"] for row in bridged], 90)) if bridged else None,
        })

    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "temporal_bridge_candidates.csv", candidate_rows)
    write_csv(args.output_dir / "temporal_bridge_summary.csv", summary_rows)
    plot_joints = [joint for joint in sorted(by_joint_frame) if reference.get(joint)]
    if plot_joints:
        render_plot({joint: reference[joint] for joint in plot_joints}, {joint: candidate_values[joint] for joint in plot_joints}, thresholds, args.output_dir / "temporal_bridge_residuals.png")
    metadata = {
        "input_per_joint_csv": str(args.per_joint_csv.resolve()),
        "coordinate_frame": "left_camera",
        "length_unit": "mm",
        "method": "No coordinate is modified. A newly retained point is evaluated only when immediately preceding and following frames have old-baseline-valid observations of the same joint.",
        "quantile": args.quantile,
        "interpretation": "The temporal envelope is a within-sequence consistency screen, not a ground-truth accuracy threshold and not a smoothing or interpolation method.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = f"""# 邻帧桥接一致性筛查\n\n输入为已完成的逐关节几何复核结果。对于原整人关联拒绝、但逐点几何通过的候选，只有其前后相邻帧中同一关节均为旧流程已通过的观测时，才计算该候选与两侧观测线性中点的距离。\n\n每个关节的参照分布来自旧流程连续三帧均有效时的相同残差，报告其 {args.quantile:g} 分位数。候选落在该分布内仅表示它没有出现超出本段可靠观测范围的瞬时跳变；本工具不修改、平滑、插值或补全任何坐标。\n\n- `temporal_bridge_candidates.csv`：每个新增候选及其邻帧状态。\n- `temporal_bridge_summary.csv`：逐关节桥接数量和筛查结果。\n- `temporal_bridge_residuals.png`：旧流程三帧残差分布与新增候选。\n"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    total_bridged = sum(row["newly_recovered_with_baseline_neighbours"] for row in summary_rows)
    total_passed = sum(row["within_baseline_temporal_envelope"] for row in summary_rows)
    print(json.dumps({"output": str(args.output_dir), "newly_recovered": len(candidate_rows), "bridged": total_bridged, "within_envelope": total_passed}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
