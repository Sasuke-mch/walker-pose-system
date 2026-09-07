#!/usr/bin/env python3
"""Test whether a whole-person stereo gate discards usable lower-body points.

This experiment deliberately does not create coordinates from temporal
interpolation or from another model.  It replays already frozen left/right 2-D
predictions.  When exactly one detected person exists in both views, each
same-name lower-body joint is independently triangulated and must still pass
the existing 2-D score, positive-depth, and fish-eye reprojection gates.

The comparison answers a narrow question: did the person-level association
gate reject a frame even though one or more lower-body point pairs were
geometrically self-consistent?  It is a diagnostic and a candidate-selection
test, not an external 3-D accuracy evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_app.calibration import StereoCalibration
from pose_app.geometry_input import paired_raw_point_rejection_reason
from pose_app.rotation import ROTATION_CHOICES, raw_to_model_point
from pose_app.triangulation import (
    COCO17_NAMES,
    _epipolar_cost,
    association_cost,
    triangulate_person,
)
from tools.evaluate_offline_stereo_predictions import _records, _result


LOWER_BODY = (
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
LOWER_INDEX = {name: COCO17_NAMES.index(name) for name in LOWER_BODY}
JOINT_COLORS = {
    "left_hip": (255, 196, 0),
    "right_hip": (255, 120, 0),
    "left_knee": (0, 216, 255),
    "right_knee": (0, 170, 255),
    "left_ankle": (0, 220, 0),
    "right_ankle": (0, 150, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("pmpose", "probpose", "sapiens2"), required=True)
    parser.add_argument("--left-json", type=Path, required=True)
    parser.add_argument("--right-json", type=Path, required=True)
    parser.add_argument("--baseline-jsonl", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--left-upright-dir", type=Path, required=True)
    parser.add_argument("--right-upright-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sapiens-foot-jsonl", type=Path)
    parser.add_argument("--keypoint-threshold", type=float, default=0.25)
    parser.add_argument("--max-association-cost", type=float, default=0.05)
    parser.add_argument("--max-reprojection-error-px", type=float, default=10.0)
    parser.add_argument("--left-model-rotation", choices=ROTATION_CHOICES, default="ccw90")
    parser.add_argument("--right-model-rotation", choices=ROTATION_CHOICES, default="cw90")
    parser.add_argument("--visualization-count", type=int, default=8)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def number(values: list[float], kind: str) -> float | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return float(np.median(array) if kind == "median" else np.mean(array))


def foot_points_by_name(path: Path | None) -> dict[str, dict[str, dict]]:
    if path is None:
        return {}
    output: dict[str, dict[str, dict]] = {}
    for row in read_jsonl(path):
        output[row["file_name"]] = {point["name"]: point for point in row.get("foot_points", [])}
    return output


def baseline_points(path: Path) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for row in read_jsonl(path):
        people = row.get("persons_3d", [])
        points = {point["name"]: point for point in people[0].get("keypoints_3d", [])} if len(people) == 1 else {}
        output[row["file_name"]] = {
            "associated": len(people) == 1,
            "points": points,
        }
    return output


def epipolar_distance(left_point: list[float], right_point: list[float], calibration: StereoCalibration) -> float | None:
    if paired_raw_point_rejection_reason(
        left_point,
        right_point,
        calibration.left_image_size,
        calibration.right_image_size,
    ) is not None:
        return None
    value = _epipolar_cost(
        np.asarray([[left_point[0], left_point[1]]], dtype=np.float64),
        np.asarray([[right_point[0], right_point[1]]], dtype=np.float64),
        calibration,
    )
    return value if np.isfinite(value) else None


def draw_joint_overlay(
    image: np.ndarray,
    person,
    direct_points: dict[str, dict],
    calibration: StereoCalibration,
    side: str,
    rotation: str,
) -> np.ndarray:
    del calibration  # The coordinates were already restored to raw camera pixels.
    raw_height, raw_width = (1920, 1080)
    # The upright input image dimensions are the inverse of raw camera dimensions.
    # Resolve the real raw dimensions from the rotation mapping below when possible.
    raw_height, raw_width = image.shape[1], image.shape[0]
    canvas = image.copy()
    for name in LOWER_BODY:
        index = LOWER_INDEX[name]
        if index >= len(person.keypoints):
            continue
        x_raw, y_raw, score = person.keypoints[index]
        x, y = raw_to_model_point(x_raw, y_raw, raw_width, raw_height, rotation)
        point = direct_points.get(name, {})
        valid = bool(point.get("valid"))
        color = JOINT_COLORS[name] if valid else (80, 80, 255)
        radius = 8 if valid else 5
        cv2.circle(canvas, (round(x), round(y)), radius, color, 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{name.split('_')[0][0].upper()}{name.split('_')[1][0].upper()} {score:.2f}",
            (round(x) + 8, round(y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def add_header(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 56), (255, 255, 255), -1)
    cv2.putText(output, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2, cv2.LINE_AA)
    return output


def render_examples(
    rows: list[dict],
    output_dir: Path,
    left_dir: Path,
    right_dir: Path,
    calibration: StereoCalibration,
    left_rotation: str,
    right_rotation: str,
    limit: int,
) -> int:
    candidates = [row for row in rows if row["singleton"] and row["newly_recovered_count"] > 0]
    candidates.sort(key=lambda row: (-row["newly_recovered_count"], row["pair_id"]))
    accepted = [row for row in rows if row["baseline_associated"] and row["direct_valid_count"] > 0]
    accepted.sort(key=lambda row: (-row["direct_valid_count"], row["pair_id"]))
    picked: list[dict] = []
    for row in candidates + accepted:
        if row["file_name"] not in {item["file_name"] for item in picked}:
            picked.append(row)
        if len(picked) >= limit:
            break
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for row in picked:
        left = cv2.imread(str(left_dir / row["file_name"]))
        right = cv2.imread(str(right_dir / row["file_name"]))
        if left is None or right is None:
            continue
        left_drawn = draw_joint_overlay(left, row["left_person"], row["direct_points"], calibration, "left", left_rotation)
        right_drawn = draw_joint_overlay(right, row["right_person"], row["direct_points"], calibration, "right", right_rotation)
        scale = min(0.43, 900 / max(left_drawn.shape[0], right_drawn.shape[0]))
        left_drawn = cv2.resize(left_drawn, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        right_drawn = cv2.resize(right_drawn, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        panel = np.hstack((left_drawn, right_drawn))
        status = "baseline associated" if row["baseline_associated"] else "baseline rejected"
        text = (
            f"{row['file_name']} | {status} | cost={row['association_cost']:.4f} | "
            f"direct valid={row['direct_valid_count']} | newly retained={row['newly_recovered_count']}"
        )
        panel = add_header(panel, text)
        target = output_dir / f"{Path(row['file_name']).stem}_jointwise_overlay.png"
        if cv2.imwrite(str(target), panel):
            written += 1
    return written


def render_summary(frame_rows: list[dict], joint_rows: list[dict], output: Path, model: str, gate: float) -> None:
    import matplotlib.pyplot as plt

    costs = [row["association_cost"] for row in frame_rows if row["association_cost"] is not None]
    names = list(LOWER_BODY)
    baseline = [sum(bool(row["baseline_valid"]) for row in joint_rows if row["joint"] == name) for name in names]
    direct = [sum(bool(row["direct_valid"]) for row in joint_rows if row["joint"] == name) for name in names]
    newly = [sum(bool(row["newly_recovered"]) for row in joint_rows if row["joint"] == name) for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    axes[0].hist(costs, bins=28, color="#6e91b6", edgecolor="white")
    axes[0].axvline(gate, color="#c62828", linewidth=2, label=f"whole-person gate {gate:.2f}")
    axes[0].set_title(f"{model}: whole-person association cost")
    axes[0].set_xlabel("median normalized epipolar distance")
    axes[0].set_ylabel("frames")
    axes[0].legend()
    y = np.arange(len(names))
    axes[1].barh(y, direct, color="#80b383", label="direct jointwise valid")
    axes[1].barh(y, baseline, color="#3f6e99", label="baseline valid")
    axes[1].barh(y, newly, color="#f1a340", label="new after rejected frame")
    axes[1].set_yticks(y, [name.replace("_", " ") for name in names])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("frames")
    axes[1].set_title("Lower-body points after existing per-point gates")
    axes[1].legend(fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not 0 <= args.keypoint_threshold <= 1 or args.max_association_cost <= 0 or args.max_reprojection_error_px <= 0:
        raise ValueError("Invalid gate values")
    if args.visualization_count <= 0:
        raise ValueError("visualization-count must be positive")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    for path in (args.left_json, args.right_json, args.baseline_jsonl, args.calibration, args.left_upright_dir, args.right_upright_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    calibration = StereoCalibration.load(args.calibration.resolve()).for_runtime_sizes((1920, 1080), (1920, 1080))
    left = _records(args.left_json.resolve(), args.model, "raw_keypoint")
    right = _records(args.right_json.resolve(), args.model, "raw_keypoint")
    left_by_name = {row["name"]: row for row in left}
    right_by_name = {row["name"]: row for row in right}
    if set(left_by_name) != set(right_by_name):
        raise RuntimeError("The left/right prediction names do not match")
    baseline = baseline_points(args.baseline_jsonl.resolve())
    if set(baseline) != set(left_by_name):
        raise RuntimeError("Baseline result names do not match prediction names")
    foot_by_name = foot_points_by_name(args.sapiens_foot_jsonl.resolve() if args.sapiens_foot_jsonl else None)

    frame_rows: list[dict] = []
    joint_rows: list[dict] = []
    render_rows: list[dict] = []
    for pair_id, name in enumerate(row["name"] for row in left):
        left_result = _result(left_by_name[name], args.model, calibration.left_image_size, args.left_model_rotation, pair_id / 30.0)
        right_result = _result(right_by_name[name], args.model, calibration.right_image_size, args.right_model_rotation, pair_id / 30.0)
        singleton = len(left_result.persons) == 1 and len(right_result.persons) == 1
        baseline_row = baseline[name]
        assoc_cost: float | None = None
        common_count = 0
        direct_points: dict[str, dict] = {}
        direct_person = None
        if singleton:
            left_person, right_person = left_result.persons[0], right_result.persons[0]
            assoc_cost, common_count = association_cost(left_person, right_person, calibration, args.keypoint_threshold)
            direct_person = triangulate_person(
                left_person,
                right_person,
                calibration,
                args.keypoint_threshold,
                args.max_reprojection_error_px,
                stereo_person_id=0,
                association_cost_value=assoc_cost,
                common_keypoints=common_count,
            )
            direct_points = {point["name"]: point for point in direct_person.keypoints_3d}

        direct_valid_count = 0
        recovered_count = 0
        for joint in LOWER_BODY:
            index = LOWER_INDEX[joint]
            left_point = left_result.persons[0].keypoints[index] if singleton else [None, None, 0.0]
            right_point = right_result.persons[0].keypoints[index] if singleton else [None, None, 0.0]
            direct = direct_points.get(joint, {})
            baseline_point = baseline_row["points"].get(joint, {})
            direct_valid = bool(direct.get("valid"))
            direct_reason = (
                direct.get("reason")
                if singleton
                else "not_singleton_person_in_both_views"
            )
            baseline_valid = bool(baseline_point.get("valid"))
            newly_recovered = bool(singleton and not baseline_row["associated"] and direct_valid)
            direct_valid_count += direct_valid
            recovered_count += newly_recovered
            foot_point = foot_by_name.get(name, {}).get(joint)
            foot_valid = bool(foot_point and foot_point.get("valid_at_reprojection_gate"))
            agreement = None
            if direct_valid and foot_valid and joint in {"left_ankle", "right_ankle"}:
                agreement = float(np.linalg.norm(np.asarray(direct["xyz"], dtype=float) - np.asarray(foot_point["xyz_left_camera"], dtype=float)))
            joint_rows.append({
                "pair_id": pair_id,
                "file_name": name,
                "model": args.model,
                "singleton_person_in_both_views": singleton,
                "baseline_associated": baseline_row["associated"],
                "association_cost": assoc_cost,
                "common_keypoints": common_count if singleton else None,
                "joint": joint,
                "left_score": float(left_point[2]),
                "right_score": float(right_point[2]),
                "joint_epipolar_distance": epipolar_distance(left_point, right_point, calibration) if singleton and left_point[2] >= args.keypoint_threshold and right_point[2] >= args.keypoint_threshold else None,
                "direct_valid": direct_valid,
                "direct_reason": direct_reason,
                "direct_reprojection_error_px": direct.get("reprojection_error_mean_px"),
                "direct_xyz_left_camera_mm": json.dumps(direct.get("xyz"), ensure_ascii=False),
                "baseline_valid": baseline_valid,
                "baseline_reason": baseline_point.get("reason"),
                "newly_recovered": newly_recovered,
                "sapiens2_foot_valid": foot_valid,
                "direct_to_sapiens2_ankle_distance_mm": agreement,
            })
        frame_rows.append({
            "pair_id": pair_id,
            "file_name": name,
            "model": args.model,
            "left_person_count": len(left_result.persons),
            "right_person_count": len(right_result.persons),
            "singleton": singleton,
            "baseline_associated": baseline_row["associated"],
            "association_cost": assoc_cost,
            "common_keypoints": common_count if singleton else None,
            "direct_valid_count": direct_valid_count,
            "newly_recovered_count": recovered_count,
        })
        if singleton:
            render_rows.append({
                **frame_rows[-1],
                "left_person": left_result.persons[0],
                "right_person": right_result.persons[0],
                "direct_points": direct_points,
            })

    args.output_dir.mkdir(parents=True)
    def write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    write_csv(args.output_dir / "frame_association_diagnostic.csv", frame_rows)
    write_csv(args.output_dir / "per_joint_geometry_diagnostic.csv", joint_rows)

    summary_rows: list[dict] = []
    for joint in LOWER_BODY:
        rows = [row for row in joint_rows if row["joint"] == joint]
        recovered = [row for row in rows if row["newly_recovered"]]
        baseline_valid = [row for row in rows if row["baseline_valid"]]
        direct_valid = [row for row in rows if row["direct_valid"]]
        agreements_baseline = [row["direct_to_sapiens2_ankle_distance_mm"] for row in rows if row["baseline_associated"] and row["direct_to_sapiens2_ankle_distance_mm"] is not None]
        agreements_recovered = [row["direct_to_sapiens2_ankle_distance_mm"] for row in recovered if row["direct_to_sapiens2_ankle_distance_mm"] is not None]
        summary_rows.append({
            "joint": joint,
            "frames": len(rows),
            "baseline_valid": len(baseline_valid),
            "direct_jointwise_valid": len(direct_valid),
            "newly_recovered_after_whole_person_rejection": len(recovered),
            "median_direct_reprojection_px": number([row["direct_reprojection_error_px"] for row in direct_valid if row["direct_reprojection_error_px"] is not None], "median"),
            "p90_direct_reprojection_px": percentile([row["direct_reprojection_error_px"] for row in direct_valid if row["direct_reprojection_error_px"] is not None], 90),
            "baseline_to_sapiens2_ankle_median_mm": number(agreements_baseline, "median"),
            "recovered_to_sapiens2_ankle_median_mm": number(agreements_recovered, "median"),
            "recovered_to_sapiens2_ankle_p90_mm": percentile(agreements_recovered, 90),
        })
    write_csv(args.output_dir / "lower_body_recovery_summary.csv", summary_rows)
    render_summary(frame_rows, joint_rows, args.output_dir / "association_and_jointwise_summary.png", args.model, args.max_association_cost)
    visualizations = render_examples(
        render_rows,
        args.output_dir / "qualitative_overlays",
        args.left_upright_dir.resolve(),
        args.right_upright_dir.resolve(),
        calibration,
        args.left_model_rotation,
        args.right_model_rotation,
        args.visualization_count,
    )
    singleton_count = sum(row["singleton"] for row in frame_rows)
    rejected_singleton = sum(row["singleton"] and not row["baseline_associated"] for row in frame_rows)
    new_frames = sum(row["singleton"] and row["newly_recovered_count"] > 0 for row in frame_rows)
    metadata = {
        "model": args.model,
        "input": {"left": str(args.left_json.resolve()), "right": str(args.right_json.resolve())},
        "baseline": str(args.baseline_jsonl.resolve()),
        "calibration": str(args.calibration.resolve()),
        "camera_model": calibration.camera_model,
        "coordinate_frame": "left_camera",
        "length_unit": calibration.length_unit,
        "model_input_rotation": {"left": args.left_model_rotation, "right": args.right_model_rotation},
        "gates": {"keypoint_score": args.keypoint_threshold, "whole_person_association": args.max_association_cost, "reprojection_px": args.max_reprojection_error_px},
        "frames": len(frame_rows),
        "singleton_frames": singleton_count,
        "baseline_rejected_singleton_frames": rejected_singleton,
        "frames_with_at_least_one_newly_retained_lower_body_point": new_frames,
        "qualitative_overlay_count": visualizations,
        "interpretation": "A retained point still has paired observed 2-D coordinates and passed score, positive-depth and fish-eye reprojection gates. The result is geometric self-consistency, not external 3-D accuracy. Sapiens2 ankle distance is an inter-model compatibility diagnostic only.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = f"""# 单人条件下逐关节几何复核\n\n本试验读取已经冻结的 {args.model} 左右二维预测和既有鱼眼标定，不重新运行检测或姿态模型。\n\n整个人体关联沿用原有条件：可见同名点的归一化极线距离中位数不超过 {args.max_association_cost:.2f}。当左右各只有一名检测到的人时，额外按同名下肢关节直接重放三角化；每一点仍同时满足二维分数不低于 {args.keypoint_threshold:.2f}、左右深度为正、平均鱼眼重投影误差不超过 {args.max_reprojection_error_px:.1f} px。\n\n`newly_recovered` 只表示原整人关联未通过、但该关节本身通过上述几何条件；它不等于人工真值，也没有用时间插值或跨模型补点。`direct_to_sapiens2_ankle_distance_mm` 仅供比较两个模型在相同图像上的脚踝位置是否相容，不能当作绝对误差。\n\n- `frame_association_diagnostic.csv`：每帧整人关联状态和直接有效下肢点数。\n- `per_joint_geometry_diagnostic.csv`：逐关节分数、极线距离、重投影误差和保留状态。\n- `lower_body_recovery_summary.csv`：各下肢关节汇总。\n- `association_and_jointwise_summary.png`：关联代价分布与逐关节结果对照。\n- `qualitative_overlays/`：正向左右图上的逐关节可视化；彩色圆圈为直接通过几何门限的点，红圈为未通过点。\n"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "model": args.model, "frames": len(frame_rows), "singleton_frames": singleton_count, "baseline_rejected_singleton_frames": rejected_singleton, "newly_retained_frames": new_frames, "visualizations": visualizations}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
