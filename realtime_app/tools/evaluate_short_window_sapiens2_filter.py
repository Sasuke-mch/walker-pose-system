#!/usr/bin/env python3
"""Evaluate a fixed short-window filter on observed Sapiens2 3-D runs.

Only the centre samples of fully observed five-frame runs are filtered with a
fixed binomial kernel.  Missing samples stay missing and no coordinate is
created across a gap.  Raw observations are retained beside derived values so
the effect on reprojection, local second difference, and bone-length spread can
be inspected before any filtering is used downstream.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_app.calibration import StereoCalibration
from pose_app.rotation import model_to_raw_point


HKA_INDEX = {9: "left_hip", 10: "right_hip", 11: "left_knee", 12: "right_knee", 13: "left_ankle", 14: "right_ankle"}
FOOT_INDEX = {15: "left_big_toe", 16: "left_small_toe", 17: "left_heel", 18: "right_big_toe", 19: "right_small_toe", 20: "right_heel"}
POINTS = tuple((*HKA_INDEX.values(), *FOOT_INDEX.values()))
SEGMENTS = {
    "left_thigh": ("left_hip", "left_knee"), "right_thigh": ("right_hip", "right_knee"),
    "left_shank": ("left_knee", "left_ankle"), "right_shank": ("right_knee", "right_ankle"),
}
KERNEL = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-jsonl", type=Path, required=True)
    parser.add_argument("--left-predictions", type=Path, required=True)
    parser.add_argument("--right-predictions", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_sequence(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or [row["pair_id"] for row in rows] != list(range(len(rows))):
        raise RuntimeError("Sequence must be non-empty and ordered by contiguous pair ID")
    return rows


def load_predictions(path: Path) -> dict[str, dict]:
    root = json.loads(path.read_text(encoding="utf-8-sig"))
    images = root.get("images")
    output = {row["file_name"]: row for row in images}
    if not isinstance(images, list) or len(output) != len(images):
        raise RuntimeError(f"Invalid Sapiens2 prediction file {path}")
    return output


def top_instance(row: dict) -> dict | None:
    instances = row.get("instances", [])
    return max(instances, key=lambda item: float(item.get("bbox_score_from_yolo26x", 0.0))) if instances else None


def raw_observations(image: dict, side: str) -> dict[str, np.ndarray]:
    instance = top_instance(image)
    if instance is None:
        return {}
    xy = instance.get("keypoints308", [])
    if len(xy) != 308:
        raise RuntimeError(f"{image.get('file_name')}: invalid keypoint count")
    rotation = "ccw90" if side == "left" else "cw90"
    output = {}
    for index, name in {**HKA_INDEX, **FOOT_INDEX}.items():
        x, y = model_to_raw_point(float(xy[index][0]), float(xy[index][1]), 1920, 1080, rotation)
        output[name] = np.asarray([x, y], dtype=np.float64)
    return output


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else None


def median(values: list[float]) -> float | None:
    return float(np.median(np.asarray(values, dtype=float))) if values else None


def mad(values: list[float]) -> float | None:
    if not values:
        return None
    centre = np.median(values)
    return float(np.median(np.abs(np.asarray(values) - centre)))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def point_xyz(record: dict, name: str) -> np.ndarray | None:
    group = record["sapiens2_hka"] if name in HKA_INDEX.values() else record["sapiens2_distal_foot"]
    point = group[name]
    return np.asarray(point["xyz_left_camera_mm"], dtype=np.float64) if point.get("valid") and point.get("xyz_left_camera_mm") is not None else None


def reproject_error(calibration: StereoCalibration, xyz: np.ndarray, left_xy: np.ndarray, right_xy: np.ndarray) -> float:
    left = calibration.project_left(xyz.reshape(1, 3))[0]
    right = calibration.project_right(xyz.reshape(1, 3))[0]
    return float(0.5 * (np.linalg.norm(left - left_xy) + np.linalg.norm(right - right_xy)))


def render_summary(point_summary: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["point"].replace("_", " ") for row in point_summary]
    raw = [row["raw_second_difference_median_mm"] or 0.0 for row in point_summary]
    filtered = [row["filtered_second_difference_median_mm"] or 0.0 for row in point_summary]
    reproj = [row["filtered_reprojection_increase_p90_px"] or 0.0 for row in point_summary]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.3), constrained_layout=True)
    axes[0].bar(x - 0.18, raw, 0.36, label="raw", color="#4c78a8")
    axes[0].bar(x + 0.18, filtered, 0.36, label="filtered", color="#f58518")
    axes[0].set_title("Local second-difference median")
    axes[0].set_ylabel("mm")
    axes[0].set_xticks(x, labels, rotation=55, ha="right", fontsize=8)
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, reproj, color="#54a24b")
    axes[1].set_title("P90 reprojection increase after filtering")
    axes[1].set_ylabel("px")
    axes[1].set_xticks(x, labels, rotation=55, ha="right", fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    records = load_sequence(args.sequence_jsonl.resolve())
    left_predictions = load_predictions(args.left_predictions.resolve())
    right_predictions = load_predictions(args.right_predictions.resolve())
    names = [record["file_name"] for record in records]
    if set(names) != set(left_predictions) or set(names) != set(right_predictions):
        raise RuntimeError("Sequence and prediction names do not match")
    calibration = StereoCalibration.load(args.calibration.resolve()).for_runtime_sizes((1920, 1080), (1920, 1080))
    length = len(records)
    raw: dict[str, list[np.ndarray | None]] = {name: [point_xyz(record, name) for record in records] for name in POINTS}
    filtered: dict[str, list[np.ndarray | None]] = {name: values.copy() for name, values in raw.items()}
    applied: dict[str, list[bool]] = {name: [False] * length for name in POINTS}
    for name, values in raw.items():
        for index in range(2, length - 2):
            window = values[index - 2 : index + 3]
            if all(value is not None for value in window):
                filtered[name][index] = sum(weight * value for weight, value in zip(KERNEL, window))
                applied[name][index] = True

    left_raw = {name: raw_observations(left_predictions[name], "left") for name in names}
    right_raw = {name: raw_observations(right_predictions[name], "right") for name in names}
    point_summary: list[dict] = []
    frame_records: list[dict] = []
    raw_second: dict[str, list[float]] = defaultdict(list)
    filtered_second: dict[str, list[float]] = defaultdict(list)
    for name in POINTS:
        reproj_delta: list[float] = []
        displacement: list[float] = []
        filtered_count = 0
        for index in range(1, length - 1):
            if raw[name][index - 1] is not None and raw[name][index] is not None and raw[name][index + 1] is not None:
                raw_second[name].append(float(np.linalg.norm(raw[name][index - 1] - 2 * raw[name][index] + raw[name][index + 1])))
            if applied[name][index - 1] and applied[name][index] and applied[name][index + 1]:
                filtered_second[name].append(float(np.linalg.norm(filtered[name][index - 1] - 2 * filtered[name][index] + filtered[name][index + 1])))
        for index, xyz in enumerate(raw[name]):
            if not applied[name][index] or xyz is None or name not in left_raw[names[index]] or name not in right_raw[names[index]]:
                continue
            new_xyz = filtered[name][index]
            raw_error = reproject_error(calibration, xyz, left_raw[names[index]][name], right_raw[names[index]][name])
            new_error = reproject_error(calibration, new_xyz, left_raw[names[index]][name], right_raw[names[index]][name])
            reproj_delta.append(new_error - raw_error)
            displacement.append(float(np.linalg.norm(new_xyz - xyz)))
            filtered_count += 1
        point_summary.append({
            "point": name,
            "raw_valid_frames": sum(value is not None for value in raw[name]),
            "filter_applied_frames": filtered_count,
            "median_filter_displacement_mm": median(displacement),
            "p90_filter_displacement_mm": percentile(displacement, 90),
            "raw_second_difference_median_mm": median(raw_second[name]),
            "filtered_second_difference_median_mm": median(filtered_second[name]),
            "raw_second_difference_p90_mm": percentile(raw_second[name], 90),
            "filtered_second_difference_p90_mm": percentile(filtered_second[name], 90),
            "filtered_reprojection_increase_median_px": median(reproj_delta),
            "filtered_reprojection_increase_p90_px": percentile(reproj_delta, 90),
        })
    segment_summary: list[dict] = []
    for segment, (a_name, b_name) in SEGMENTS.items():
        raw_length, filtered_length = [], []
        for index in range(length):
            if raw[a_name][index] is not None and raw[b_name][index] is not None:
                raw_length.append(float(np.linalg.norm(raw[a_name][index] - raw[b_name][index])))
            if applied[a_name][index] and applied[b_name][index] and filtered[a_name][index] is not None and filtered[b_name][index] is not None:
                filtered_length.append(float(np.linalg.norm(filtered[a_name][index] - filtered[b_name][index])))
        segment_summary.append({
            "segment": segment,
            "raw_observed_frames": len(raw_length),
            "raw_length_median_mm": median(raw_length),
            "raw_length_mad_mm": mad(raw_length),
            "filtered_overlap_frames": len(filtered_length),
            "filtered_length_median_mm": median(filtered_length),
            "filtered_length_mad_mm": mad(filtered_length),
        })
    for index, record in enumerate(records):
        points = {}
        for name in POINTS:
            xyz = raw[name][index]
            points[name] = {
                "raw_xyz_left_camera_mm": None if xyz is None else xyz.tolist(),
                "derived_filtered_xyz_left_camera_mm": None if xyz is None else filtered[name][index].tolist(),
                "filter_applied": applied[name][index],
            }
        frame_records.append({"pair_id": index, "file_name": record["file_name"], "points": points})
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "short_window_filter_sequence.jsonl").open("w", encoding="utf-8") as handle:
        for record in frame_records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    write_csv(args.output_dir / "point_filter_summary.csv", point_summary)
    write_csv(args.output_dir / "segment_filter_summary.csv", segment_summary)
    render_summary(point_summary, args.output_dir / "filter_effect_summary.png")
    metadata = {
        "sequence": str(args.sequence_jsonl.resolve()),
        "calibration": str(args.calibration.resolve()),
        "kernel": KERNEL.tolist(),
        "window_frames": 5,
        "frame_rate_assumption": 30.0,
        "missing_policy": "A filtered value is produced only for the center of a fully observed five-frame run. No gap is filled.",
        "interpretation": "Filtered values are derived display/analysis candidates. Raw observations remain authoritative; reprojection increase is measured against the same stored 2-D points and is not external accuracy.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "README.md").write_text("# Sapiens2 连续观测段短窗滤波检查\n\n采用固定五帧二项核，只处理完整连续观测段的中间帧。缺失帧不生成坐标，原始三维观测完整保留。结果表同时给出局部二阶差分、滤波位移、重新投影增加和腿段长度离散度，供判断该滤波是否值得进入后续显示或分析链。\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "frames": length, "points": len(POINTS)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
