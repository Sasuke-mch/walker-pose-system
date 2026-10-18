#!/usr/bin/env python3
"""Diagnose raw stereo COCO geometry against fitted SMPL COCO proxies.

Triangulation is recomputed from the raw per-view 2-D observations and used
only for post-fit diagnosis. It never enters the SMPL objective.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_app.fisheye_camera import load_stereo_fisheye  # noqa: E402
from tools.fit_smpl_coco_single_frame import pixel_ray  # noqa: E402


BONES = {
    "left_upper_arm": (5, 7),
    "left_forearm": (7, 9),
    "right_upper_arm": (6, 8),
    "right_forearm": (8, 10),
    "left_thigh": (11, 13),
    "left_shin": (13, 15),
    "right_thigh": (12, 14),
    "right_shin": (14, 16),
}


def triangulate_midpoint(left: np.ndarray, right: np.ndarray, cal: object) -> tuple[np.ndarray, float]:
    ray_left = pixel_ray(left, cal.K0, cal.D0)
    ray_right_cam1 = pixel_ray(right, cal.K1, cal.D1)
    ray_right = cal.R_cam0_to_cam1.T @ ray_right_cam1
    camera_right = -cal.R_cam0_to_cam1.T @ (cal.T_cam0_to_cam1_mm / 1000.0)
    coefficients = np.linalg.lstsq(
        np.column_stack([ray_left, -ray_right]), camera_right, rcond=None
    )[0]
    point_left = coefficients[0] * ray_left
    point_right = camera_right + coefficients[1] * ray_right
    return 0.5 * (point_left + point_right), float(np.linalg.norm(point_left - point_right))


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stereo-jsonl", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--window-dir", type=Path, required=True)
    parser.add_argument("--start-pair", type=int, required=True)
    parser.add_argument("--end-pair", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = {}
    for line in args.stereo_jsonl.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[int(row["pair_id"])] = row
    cal = load_stereo_fisheye(args.calibration_dir)

    ray_gaps_mm: list[float] = []
    observed = {name: [] for name in BONES}
    predicted = {name: [] for name in BONES}
    frame_rows = []
    for pair_id in range(args.start_pair, args.end_pair + 1):
        row = rows[pair_id]
        left = np.asarray(row["left"]["persons"][0]["keypoints"], dtype=np.float64)
        right = np.asarray(row["right"]["persons"][0]["keypoints"], dtype=np.float64)
        points = []
        frame_gaps = []
        for left_point, right_point in zip(left, right):
            point, gap = triangulate_midpoint(left_point[:2], right_point[:2], cal)
            points.append(point)
            frame_gaps.append(gap * 1000.0)
        points = np.asarray(points)
        fit_path = args.window_dir / "frames" / f"frame_{pair_id:04d}" / "result.npz"
        with np.load(fit_path) as fit:
            proxy = np.asarray(fit["coco_left_m"], dtype=np.float64)
        frame_bones = {}
        for name, (a, b) in BONES.items():
            observed_mm = float(np.linalg.norm(points[a] - points[b]) * 1000.0)
            predicted_mm = float(np.linalg.norm(proxy[a] - proxy[b]) * 1000.0)
            observed[name].append(observed_mm)
            predicted[name].append(predicted_mm)
            frame_bones[name] = {
                "raw_2d_triangulated_mm": observed_mm,
                "fitted_smpl_proxy_mm": predicted_mm,
                "difference_mm": predicted_mm - observed_mm,
            }
        ray_gaps_mm.extend(frame_gaps)
        frame_rows.append({
            "pair_id": pair_id,
            "ray_gap_median_mm": float(np.median(frame_gaps)),
            "ray_gap_p95_mm": float(np.percentile(frame_gaps, 95)),
            "bones": frame_bones,
        })

    bone_summary = {}
    for name in BONES:
        differences = np.asarray(predicted[name]) - np.asarray(observed[name])
        bone_summary[name] = {
            "raw_2d_triangulated_mm": stats(observed[name]),
            "fitted_smpl_proxy_mm": stats(predicted[name]),
            "model_minus_observation_mm": stats(differences.tolist()),
        }
    report = {
        "schema_version": "smpl_coco_window_geometry_diagnostic_v1",
        "window": [args.start_pair, args.end_pair],
        "usage": "post_fit_diagnostic_only_not_anatomical_ground_truth_or_loss",
        "ray_closest_approach_gap_mm": stats(ray_gaps_mm),
        "bone_summary": bone_summary,
        "frames": frame_rows,
        "conclusion_boundary": (
            "Lengths are implied by raw PMPose pixels and calibrated stereo. They diagnose "
            "model-observation incompatibility but are not anatomical ground truth."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ray_gap": report["ray_closest_approach_gap_mm"], "bones": bone_summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
