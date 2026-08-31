#!/usr/bin/env python3
"""Triangulate named Sapiens2-308 foot landmarks in calibrated raw fisheye views.

This is an exploratory foot-3D route, not an accuracy evaluation.  It keeps
the 2-D model observations, the raw-fisheye calibration, every reprojection
residual, and local foot-shape measurements so that rejected observations are
not hidden by the final 10-pixel gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from pose_app.calibration import StereoCalibration
from pose_app.rotation import ROTATION_CHOICES, model_to_raw_point


FOOT_POINTS = {
    13: "left_ankle",
    15: "left_big_toe",
    16: "left_small_toe",
    17: "left_heel",
    14: "right_ankle",
    18: "right_big_toe",
    19: "right_small_toe",
    20: "right_heel",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--left-predictions", type=Path, required=True)
    parser.add_argument("--right-predictions", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--left-model-rotation", choices=ROTATION_CHOICES, default="ccw90")
    parser.add_argument("--right-model-rotation", choices=ROTATION_CHOICES, default="cw90")
    parser.add_argument("--keypoint-threshold", type=float, default=0.25)
    parser.add_argument("--max-reprojection-error-px", type=float, default=10.0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_predictions(path: Path) -> dict[str, dict]:
    source = json.loads(path.read_text(encoding="utf-8-sig"))
    images = source.get("images")
    if not isinstance(images, list):
        raise RuntimeError(f"No images list in {path}")
    by_name = {str(row["file_name"]): row for row in images}
    if len(by_name) != len(images):
        raise RuntimeError(f"Duplicate file_name in {path}")
    return by_name


def top_instance(image: dict) -> tuple[dict | None, int]:
    instances = image.get("instances", [])
    if not instances:
        return None, 0
    return max(instances, key=lambda item: float(item.get("bbox_score_from_yolo26x", 0.0))), len(instances)


def extract_points(
    image: dict,
    *,
    raw_size: tuple[int, int],
    rotation: str,
) -> tuple[dict[str, dict], int]:
    selected, candidates = top_instance(image)
    result: dict[str, dict] = {}
    for index, name in FOOT_POINTS.items():
        point = {"index": index, "name": name, "score": 0.0, "raw_xy": None}
        if selected is not None:
            keypoints = selected.get("keypoints308", [])
            scores = selected.get("keypoint_scores", [])
            if len(keypoints) != 308 or len(scores) != 308:
                raise RuntimeError(f"{image.get('file_name')}: expected 308 Sapiens2 points")
            x_model, y_model = [float(value) for value in keypoints[index]]
            x_raw, y_raw = model_to_raw_point(x_model, y_model, raw_size[0], raw_size[1], rotation)
            point.update({"score": float(scores[index]), "raw_xy": [x_raw, y_raw]})
        result[name] = point
    return result, candidates


def finite_vector(values: np.ndarray) -> list[float] | None:
    return [float(value) for value in values] if np.all(np.isfinite(values)) else None


def triangulate_foot_points(
    left: dict[str, dict],
    right: dict[str, dict],
    calibration: StereoCalibration,
    score_threshold: float,
    reprojection_gate: float,
) -> dict[str, dict]:
    records: dict[str, dict] = {}
    eligible: list[str] = []
    left_pixels, right_pixels = [], []
    for name in FOOT_POINTS.values():
        left_point, right_point = left[name], right[name]
        row = {
            "name": name,
            "sapiens_index": left_point["index"],
            "left_score": left_point["score"],
            "right_score": right_point["score"],
            "left_raw_xy": left_point["raw_xy"],
            "right_raw_xy": right_point["raw_xy"],
            "xyz_left_camera": None,
            "depth_left": None,
            "depth_right": None,
            "reprojection_error_left_px": None,
            "reprojection_error_right_px": None,
            "reprojection_error_mean_px": None,
            "positive_finite": False,
            "valid_at_reprojection_gate": False,
            "reason": None,
        }
        if left_point["score"] < score_threshold or right_point["score"] < score_threshold:
            row["reason"] = "low_2d_score"
        elif left_point["raw_xy"] is None or right_point["raw_xy"] is None:
            row["reason"] = "missing_2d_point"
        else:
            eligible.append(name)
            left_pixels.append(left_point["raw_xy"])
            right_pixels.append(right_point["raw_xy"])
        records[name] = row

    if not eligible:
        return records

    left_array = np.asarray(left_pixels, dtype=np.float64)
    right_array = np.asarray(right_pixels, dtype=np.float64)
    left_normalized = calibration.undistort_normalized(left_array, "left")
    right_normalized = calibration.undistort_normalized(right_array, "right")
    projection_left = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)
    projection_right = np.concatenate([calibration.R, calibration.T.reshape(3, 1)], axis=1)
    homogeneous = cv2.triangulatePoints(projection_left, projection_right, left_normalized.T, right_normalized.T)
    xyz = np.full((len(eligible), 3), np.nan, dtype=np.float64)
    usable_w = np.abs(homogeneous[3]) > 1e-12
    xyz[usable_w] = (homogeneous[:3, usable_w] / homogeneous[3, usable_w]).T
    projected_left = calibration.project_left(xyz)
    projected_right = calibration.project_right(xyz)
    right_xyz = (calibration.R @ xyz.T + calibration.T.reshape(3, 1)).T

    for local_index, name in enumerate(eligible):
        row = records[name]
        point = xyz[local_index]
        depth_left = float(point[2])
        depth_right = float(right_xyz[local_index, 2])
        error_left = float(np.linalg.norm(projected_left[local_index] - left_array[local_index]))
        error_right = float(np.linalg.norm(projected_right[local_index] - right_array[local_index]))
        error_mean = 0.5 * (error_left + error_right)
        finite = bool(
            np.all(np.isfinite(point))
            and math.isfinite(depth_left)
            and math.isfinite(depth_right)
            and math.isfinite(error_mean)
        )
        positive = finite and depth_left > 0.0 and depth_right > 0.0
        passed = positive and error_mean <= reprojection_gate
        row.update({
            "xyz_left_camera": finite_vector(point) if finite else None,
            "depth_left": depth_left if finite else None,
            "depth_right": depth_right if finite else None,
            "reprojection_error_left_px": error_left if finite else None,
            "reprojection_error_right_px": error_right if finite else None,
            "reprojection_error_mean_px": error_mean if finite else None,
            "positive_finite": positive,
            "valid_at_reprojection_gate": passed,
            "reason": None if passed else ("high_reprojection_error" if positive else "negative_or_non_finite_depth"),
        })
    return records


def distance(a: list[float], b: list[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def foot_shape_rows(pair_id: int, file_name: str, condition: str, records: dict[str, dict]) -> list[dict]:
    rows = []
    for side in ("left", "right"):
        ankle = records[f"{side}_ankle"]
        big = records[f"{side}_big_toe"]
        small = records[f"{side}_small_toe"]
        heel = records[f"{side}_heel"]
        valid = {
            name: item["xyz_left_camera"]
            for name, item in {"ankle": ankle, "big_toe": big, "small_toe": small, "heel": heel}.items()
            if item["valid_at_reprojection_gate"] and item["xyz_left_camera"] is not None
        }
        row = {
            "pair_id": pair_id,
            "file_name": file_name,
            "distance_condition": condition,
            "side": side,
            "toe_width": None,
            "ankle_to_forefoot_midpoint": None,
            "ankle_to_heel": None,
            "toe_to_heel_midpoint": None,
            "complete_foot_at_gate": False,
        }
        if {"big_toe", "small_toe"}.issubset(valid):
            row["toe_width"] = distance(valid["big_toe"], valid["small_toe"])
        if {"ankle", "big_toe", "small_toe"}.issubset(valid):
            forefoot = ((np.asarray(valid["big_toe"]) + np.asarray(valid["small_toe"])) / 2.0).tolist()
            row["ankle_to_forefoot_midpoint"] = distance(valid["ankle"], forefoot)
        if {"ankle", "heel"}.issubset(valid):
            row["ankle_to_heel"] = distance(valid["ankle"], valid["heel"])
        if {"heel", "big_toe", "small_toe"}.issubset(valid):
            forefoot = ((np.asarray(valid["big_toe"]) + np.asarray(valid["small_toe"])) / 2.0).tolist()
            row["toe_to_heel_midpoint"] = distance(valid["heel"], forefoot)
        row["complete_foot_at_gate"] = len(valid) == 4
        rows.append(row)
    return rows


def percentile(values: list[float], percentile_value: float) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=float), percentile_value)) if values else None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.keypoint_threshold <= 1.0:
        raise ValueError("--keypoint-threshold must be in [0, 1]")
    if args.max_reprojection_error_px <= 0.0:
        raise ValueError("--max-reprojection-error-px must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")

    manifest = read_csv(args.selection_manifest.resolve())
    condition_by_name = {row["file_name"]: row["condition"] for row in manifest}
    left_images = load_predictions(args.left_predictions.resolve())
    right_images = load_predictions(args.right_predictions.resolve())
    if set(left_images) != set(right_images) or set(left_images) != set(condition_by_name):
        raise RuntimeError("Selection manifest and left/right prediction file names must match exactly")
    calibration = StereoCalibration.load(args.calibration.resolve()).for_runtime_sizes((1920, 1080), (1920, 1080))

    output_dir.mkdir(parents=True)
    point_rows, shape_rows = [], []
    jsonl_path = output_dir / "foot_stereo_results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for pair_id, file_name in enumerate(sorted(condition_by_name)):
            left, left_candidates = extract_points(
                left_images[file_name], raw_size=calibration.left_image_size, rotation=args.left_model_rotation
            )
            right, right_candidates = extract_points(
                right_images[file_name], raw_size=calibration.right_image_size, rotation=args.right_model_rotation
            )
            points = triangulate_foot_points(
                left, right, calibration, args.keypoint_threshold, args.max_reprojection_error_px
            )
            condition = condition_by_name[file_name]
            shape_rows.extend(foot_shape_rows(pair_id, file_name, condition, points))
            for point in points.values():
                point_rows.append({
                    "pair_id": pair_id,
                    "file_name": file_name,
                    "distance_condition": condition,
                    **point,
                })
            payload = {
                "pair_id": pair_id,
                "file_name": file_name,
                "distance_condition": condition,
                "coordinate_frame": "left_camera",
                "length_unit": calibration.length_unit,
                "left_candidate_instances": left_candidates,
                "right_candidate_instances": right_candidates,
                "foot_points": list(points.values()),
                "foot_shape": shape_rows[-2:],
            }
            handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")

    write_csv(output_dir / "per_foot_point.csv", point_rows)
    write_csv(output_dir / "per_foot_shape.csv", shape_rows)
    summary_rows = []
    for distance_name in ["all", *sorted(set(row["distance_condition"] for row in point_rows))]:
        subset = point_rows if distance_name == "all" else [row for row in point_rows if row["distance_condition"] == distance_name]
        for name in FOOT_POINTS.values():
            rows = [row for row in subset if row["name"] == name]
            positive = [row for row in rows if row["positive_finite"]]
            accepted = [row for row in rows if row["valid_at_reprojection_gate"]]
            all_errors = [float(row["reprojection_error_mean_px"]) for row in positive]
            accepted_errors = [float(row["reprojection_error_mean_px"]) for row in accepted]
            summary_rows.append({
                "distance_condition": distance_name,
                "foot_point": name,
                "candidate_pairs": len(rows),
                "positive_finite_points": len(positive),
                "accepted_points_at_gate": len(accepted),
                "accepted_fraction_of_positive_finite": len(accepted) / len(positive) if positive else None,
                "ungated_mean_reprojection_px": float(np.mean(all_errors)) if all_errors else None,
                "ungated_median_reprojection_px": float(np.median(all_errors)) if all_errors else None,
                "ungated_p90_reprojection_px": percentile(all_errors, 90),
                "accepted_mean_reprojection_px": float(np.mean(accepted_errors)) if accepted_errors else None,
            })
    write_csv(output_dir / "foot_reprojection_summary.csv", summary_rows)

    shape_summary = []
    for distance_name in ["all", *sorted(set(row["distance_condition"] for row in shape_rows))]:
        subset = shape_rows if distance_name == "all" else [row for row in shape_rows if row["distance_condition"] == distance_name]
        for measurement in ("toe_width", "ankle_to_forefoot_midpoint", "ankle_to_heel", "toe_to_heel_midpoint"):
            values = [float(row[measurement]) for row in subset if row[measurement] is not None]
            shape_summary.append({
                "distance_condition": distance_name,
                "measurement": measurement,
                "available_feet": len(values),
                "median": float(np.median(values)) if values else None,
                "p10": percentile(values, 10),
                "p90": percentile(values, 90),
            })
    write_csv(output_dir / "foot_shape_summary.csv", shape_summary)
    metadata = {
        "input": {"selection_manifest": str(args.selection_manifest.resolve()), "left_predictions": str(args.left_predictions.resolve()), "right_predictions": str(args.right_predictions.resolve())},
        "calibration": str(args.calibration.resolve()),
        "camera_model": calibration.camera_model,
        "coordinate_frame": "left_camera",
        "length_unit": calibration.length_unit,
        "model_input_rotation": {"left": args.left_model_rotation, "right": args.right_model_rotation},
        "keypoint_threshold": args.keypoint_threshold,
        "max_reprojection_error_px": args.max_reprojection_error_px,
        "interpretation_boundary": "Sapiens2 foot points are visually checked engineering pseudo-labels, not independent 2-D truth. Reprojection consistency and local foot shape do not prove absolute 3-D accuracy or gait-contact validity.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": len(condition_by_name), "points": len(point_rows), "output": str(output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
