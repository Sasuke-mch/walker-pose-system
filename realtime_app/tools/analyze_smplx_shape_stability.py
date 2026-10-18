#!/usr/bin/env python3
"""Audit independent SMPL-X clip shapes and systematic COCO joint offsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


JOINT_NAMES = {
    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle",
}
SEGMENTS = {
    "shoulder_width": (5, 6), "hip_width": (11, 12),
    "left_upper_arm": (5, 7), "right_upper_arm": (6, 8),
    "left_forearm": (7, 9), "right_forearm": (8, 10),
    "left_thigh": (11, 13), "right_thigh": (12, 14),
    "left_shank": (13, 15), "right_shank": (14, 16),
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def body_basis(points: np.ndarray) -> np.ndarray:
    left = points[5] - points[6]
    up = 0.5 * (points[5] + points[6]) - 0.5 * (points[11] + points[12])
    left /= max(np.linalg.norm(left), 1e-9)
    up -= left * float(left @ up)
    up /= max(np.linalg.norm(up), 1e-9)
    forward = np.cross(left, up)
    forward /= max(np.linalg.norm(forward), 1e-9)
    return np.column_stack((left, up, forward))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--fits-root", type=Path, required=True)
    parser.add_argument("--prefit-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixed-fits-root", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    prefit = read_jsonl(args.prefit_jsonl)
    betas = []
    clip_rows = []
    all_offsets = {joint: [] for joint in JOINT_NAMES}
    offsets_by_clip = []
    segment_by_clip = {name: [] for name in SEGMENTS}
    all_before = []
    for clip in selection["clips"]:
        fit_dir = args.fits_root / clip["clip_id"]
        summary = json.loads((fit_dir / "summary.json").read_text(encoding="utf-8"))
        rows = read_jsonl(fit_dir / "smplx_window_joints.jsonl")
        beta = np.asarray(summary["shared_betas"], dtype=np.float64)
        betas.append(beta)
        local_lengths = {name: [] for name in SEGMENTS}
        clip_offsets = {joint: [] for joint in JOINT_NAMES}
        for row in rows:
            frame_index = int(row["frame_index"])
            fitted = np.asarray(row["smplx_coco17_joints_ground_mm"], dtype=np.float64)
            target = np.asarray(
                [prefit[frame_index]["points_ground_mm"][str(index)] for index in range(17)],
                dtype=np.float64,
            )
            basis = body_basis(target)
            for joint in JOINT_NAMES:
                residual = fitted[joint] - target[joint]
                local_residual = basis.T @ residual
                all_offsets[joint].append(local_residual)
                clip_offsets[joint].append(local_residual)
                all_before.append(float(np.linalg.norm(residual)))
            for name, (start, end) in SEGMENTS.items():
                local_lengths[name].append(float(np.linalg.norm(fitted[start] - fitted[end])))
        medians = {name: float(np.median(values)) for name, values in local_lengths.items()}
        offsets_by_clip.append({joint: np.stack(values) for joint, values in clip_offsets.items()})
        for name, value in medians.items():
            segment_by_clip[name].append(value)
        clip_rows.append({
            "clip_id": clip["clip_id"],
            "inclusive_frame_window": clip["inclusive_frame_window"],
            "shared_betas": beta.tolist(),
            "joint_residual_median_mm": summary["joint_residual_median_mm"],
            "joint_residual_p95_mm": summary["joint_residual_p95_mm"],
            "segment_lengths_median_mm": medians,
        })

    beta_array = np.stack(betas)
    offset_rows = {}
    corrected = []
    for joint, values in all_offsets.items():
        array = np.stack(values)
        median = np.median(array, axis=0)
        residuals = array - median
        corrected.extend(np.linalg.norm(residuals, axis=1).tolist())
        dispersion = np.linalg.norm(residuals, axis=1)
        offset_rows[str(joint)] = {
            "name": JOINT_NAMES[joint],
            "body_local_median_offset_mm": median.tolist(),
            "median_offset_norm_mm": float(np.linalg.norm(median)),
            "residual_dispersion_median_mm": float(np.median(dispersion)),
            "residual_dispersion_p95_mm": float(np.percentile(dispersion, 95)),
            "stable_offset_candidate": bool(
                np.linalg.norm(median) >= 10.0 and np.median(dispersion) <= 15.0
            ),
        }
    segment_rows = {}
    for name, values in segment_by_clip.items():
        array = np.asarray(values)
        segment_rows[name] = {
            "per_clip_median_mm": array.tolist(),
            "mean_mm": float(np.mean(array)),
            "range_mm": float(np.ptp(array)),
            "cv": float(np.std(array) / max(np.mean(array), 1e-9)),
        }
    beta_sd = np.std(beta_array, axis=0)
    segment_ranges = [item["range_mm"] for item in segment_rows.values()]
    leave_one_clip_out_corrected = []
    leave_one_clip_out_rows = []
    for held_out_index, clip in enumerate(selection["clips"]):
        held_out_values = []
        for joint in JOINT_NAMES:
            training = np.concatenate([
                offsets_by_clip[index][joint]
                for index in range(len(offsets_by_clip)) if index != held_out_index
            ], axis=0)
            calibration = np.median(training, axis=0)
            held_out_values.extend(
                np.linalg.norm(offsets_by_clip[held_out_index][joint] - calibration, axis=1).tolist()
            )
        leave_one_clip_out_corrected.extend(held_out_values)
        leave_one_clip_out_rows.append({
            "clip_id": clip["clip_id"],
            "corrected_residual_median_mm": float(np.median(held_out_values)),
            "corrected_residual_p95_mm": float(np.percentile(held_out_values, 95)),
        })
    fixed_shape_comparison = None
    fixed_shape_gate = True
    if args.fixed_fits_root is not None:
        fixed_clip_rows = []
        independent_medians = []
        fixed_medians = []
        for clip, independent in zip(selection["clips"], clip_rows):
            fixed_summary = json.loads(
                (args.fixed_fits_root / clip["clip_id"] / "summary.json").read_text(encoding="utf-8")
            )
            independent_value = float(independent["joint_residual_median_mm"])
            fixed_value = float(fixed_summary["joint_residual_median_mm"])
            independent_medians.append(independent_value)
            fixed_medians.append(fixed_value)
            fixed_clip_rows.append({
                "clip_id": clip["clip_id"],
                "independent_shape_residual_median_mm": independent_value,
                "fixed_shared_shape_residual_median_mm": fixed_value,
                "increase_fraction": float((fixed_value - independent_value) / max(independent_value, 1e-9)),
                "fixed_shared_shape_residual_p95_mm": float(fixed_summary["joint_residual_p95_mm"]),
            })
        independent_aggregate = float(np.median(independent_medians))
        fixed_aggregate = float(np.median(fixed_medians))
        increase = float((fixed_aggregate - independent_aggregate) / max(independent_aggregate, 1e-9))
        fixed_shape_gate = increase <= 0.10
        fixed_shape_comparison = {
            "per_clip": fixed_clip_rows,
            "independent_clip_median_of_medians_mm": independent_aggregate,
            "fixed_shared_shape_median_of_medians_mm": fixed_aggregate,
            "increase_fraction": increase,
            "gate_fixed_shape_increase_at_most_10_percent": fixed_shape_gate,
        }
    result = {
        "schema_version": "smplx_shape_and_joint_offset_audit_v1",
        "clip_count": len(clip_rows),
        "clip_results": clip_rows,
        "beta_statistics": {
            "per_dimension_mean": np.mean(beta_array, axis=0).tolist(),
            "per_dimension_std": beta_sd.tolist(),
            "maximum_dimension_std": float(np.max(beta_sd)),
            "median_dimension_std": float(np.median(beta_sd)),
            "coordinatewise_median_shared_beta": np.median(beta_array, axis=0).tolist(),
        },
        "segment_stability": segment_rows,
        "maximum_segment_length_range_mm": float(max(segment_ranges)),
        "joint_offset_audit": offset_rows,
        "raw_joint_residual_median_mm": float(np.median(all_before)),
        "raw_joint_residual_p95_mm": float(np.percentile(all_before, 95)),
        "after_body_local_median_offset_residual_median_mm": float(np.median(corrected)),
        "after_body_local_median_offset_residual_p95_mm": float(np.percentile(corrected, 95)),
        "leave_one_clip_out_offset_validation": {
            "per_held_out_clip": leave_one_clip_out_rows,
            "corrected_residual_median_mm": float(np.median(leave_one_clip_out_corrected)),
            "corrected_residual_p95_mm": float(np.percentile(leave_one_clip_out_corrected, 95)),
            "median_reduction_fraction": float(
                1.0 - np.median(leave_one_clip_out_corrected) / max(np.median(all_before), 1e-9)
            ),
        },
        "fixed_shared_shape_comparison": fixed_shape_comparison,
        "gates": {
            "maximum_segment_range_at_most_20_mm": bool(max(segment_ranges) <= 20.0),
            "median_beta_dimension_std_at_most_0_30": bool(np.median(beta_sd) <= 0.30),
            "leave_one_clip_out_offset_reduces_median_residual_by_20_percent": bool(
                np.median(leave_one_clip_out_corrected) <= 0.8 * np.median(all_before)
            ),
            "fixed_shared_shape_median_residual_increase_at_most_10_percent": bool(fixed_shape_gate),
        },
        "boundary": (
            "Cross-clip internal stability and joint-definition audit. It does not establish true body shape "
            "or physical joint-center accuracy."
        ),
    }
    result["status"] = "passed" if all(result["gates"].values()) else "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
