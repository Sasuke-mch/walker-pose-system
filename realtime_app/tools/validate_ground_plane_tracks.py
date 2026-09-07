#!/usr/bin/env python3
"""Check whether locally reconstructed floor-plane inliers persist over time.

This follows the cross-frame ground-feature tracking step in the system
specification.  It intentionally treats the result as a validation gate: a
per-frame RANSAC plane is not allowed to become a dynamic ground reference
unless its image inliers can be tracked to the next frame and are supported by
the next frame's independently reconstructed plane.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from experiment_ground_plane_local_rectification import (
    LocalRectification,
    fit_plane_ransac,
    floor_candidate_mask,
    inverse_upright,
    load_sapiens_by_name,
    make_floor_directed_rectification,
    read_pairs,
    reconstruct_floor_candidates,
    rotate_mask_to_raw,
)
from pose_app.calibration import StereoCalibration


@dataclass
class FrameData:
    name: str
    left_local: np.ndarray
    inlier_mask: np.ndarray
    accepted: bool
    candidate_points: int
    inlier_points: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-dir", type=Path, required=True)
    parser.add_argument("--right-dir", type=Path, required=True)
    parser.add_argument("--sapiens-left-json", type=Path, required=True)
    parser.add_argument("--sapiens-right-json", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-step", type=int, default=6)
    parser.add_argument("--runtime-width", type=int, default=960)
    parser.add_argument("--runtime-height", type=int, default=540)
    parser.add_argument("--virtual-focal-px", type=float, default=330.0)
    parser.add_argument("--num-disparities", type=int, default=160)
    parser.add_argument("--lr-consistency-px", type=float, default=1.5)
    parser.add_argument("--ransac-distance-mm", type=float, default=25.0)
    parser.add_argument("--minimum-candidates", type=int, default=250)
    parser.add_argument("--minimum-inliers", type=int, default=150)
    parser.add_argument("--max-corners", type=int, default=500)
    parser.add_argument("--max-lk-error", type=float, default=20.0)
    parser.add_argument("--homography-threshold-px", type=float, default=2.0)
    return parser.parse_args()


def prepare_frame(
    index: int,
    name: str,
    left_path: Path,
    right_path: Path,
    left_sapiens: dict[str, dict],
    right_sapiens: dict[str, dict],
    rectification: LocalRectification,
    runtime_size: tuple[int, int],
    args: argparse.Namespace,
) -> FrameData:
    left_upright = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right_upright = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    if left_upright is None or right_upright is None:
        raise RuntimeError(f"cannot read {name}")
    left_raw, right_raw = inverse_upright(left_upright, right_upright)
    left_raw = cv2.resize(left_raw, runtime_size, interpolation=cv2.INTER_AREA)
    right_raw = cv2.resize(right_raw, runtime_size, interpolation=cv2.INTER_AREA)
    left_mask_raw = cv2.resize(
        rotate_mask_to_raw(floor_candidate_mask(left_upright, left_sapiens[name]), "left"),
        runtime_size, interpolation=cv2.INTER_NEAREST,
    )
    right_mask_raw = cv2.resize(
        rotate_mask_to_raw(floor_candidate_mask(right_upright, right_sapiens[name]), "right"),
        runtime_size, interpolation=cv2.INTER_NEAREST,
    )
    left_local = cv2.remap(left_raw, rectification.left_map_x, rectification.left_map_y, cv2.INTER_LINEAR)
    right_local = cv2.remap(right_raw, rectification.right_map_x, rectification.right_map_y, cv2.INTER_LINEAR)
    left_mask = cv2.remap(left_mask_raw, rectification.left_map_x, rectification.left_map_y, cv2.INTER_NEAREST)
    right_mask = cv2.remap(right_mask_raw, rectification.right_map_x, rectification.right_map_y, cv2.INTER_NEAREST)
    points, left_pixels, _, _ = reconstruct_floor_candidates(
        left_local, right_local, left_mask, right_mask, rectification,
        args.num_disparities, args.lr_consistency_px,
    )
    fit_points = points if len(points) <= 12000 else points[np.linspace(0, len(points) - 1, 12000, dtype=np.int64)]
    fit = fit_plane_ransac(fit_points, args.ransac_distance_mm, 900, 20260902 + index)
    inlier_mask = np.zeros(left_local.shape[:2], dtype=np.uint8)
    accepted = False
    inlier_count = 0
    if fit is not None:
        inliers = np.abs(points @ fit.normal + fit.offset) <= args.ransac_distance_mm
        inlier_count = int(inliers.sum())
        accepted = len(points) >= args.minimum_candidates and inlier_count >= args.minimum_inliers
        pixels = left_pixels[inliers]
        if len(pixels):
            inlier_mask[pixels[:, 1], pixels[:, 0]] = 255
            # Allow a KLT feature's small support window while requiring the
            # feature centre in the next frame to return to plane inliers.
            inlier_mask = cv2.dilate(inlier_mask, np.ones((5, 5), dtype=np.uint8))
    return FrameData(name, left_local, inlier_mask, accepted, len(points), inlier_count)


def tracked_transition(first: FrameData, second: FrameData, args: argparse.Namespace) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    row: dict[str, object] = {
        "from_file": first.name,
        "to_file": second.name,
        "from_plane_accepted": first.accepted,
        "to_plane_accepted": second.accepted,
        "from_candidate_points": first.candidate_points,
        "to_candidate_points": second.candidate_points,
        "from_plane_inliers": first.inlier_points,
        "to_plane_inliers": second.inlier_points,
        "seed_corners": 0,
        "lk_valid": 0,
        "returned_to_next_plane": 0,
        "homography_inliers": 0,
        "homography_inlier_fraction": None,
        "median_flow_px": None,
        "status": "not_evaluated",
    }
    empty = np.empty((0, 2), dtype=np.float32)
    if not first.accepted or not second.accepted:
        row["status"] = "missing_plane"
        return row, empty, empty, empty
    first_gray = cv2.cvtColor(first.left_local, cv2.COLOR_BGR2GRAY)
    second_gray = cv2.cvtColor(second.left_local, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(
        first_gray, maxCorners=args.max_corners, qualityLevel=0.01, minDistance=5,
        mask=first.inlier_mask, blockSize=7, useHarrisDetector=False,
    )
    if corners is None or len(corners) < 4:
        row["status"] = "too_few_seed_corners"
        return row, empty, empty, empty
    start = corners.reshape(-1, 2).astype(np.float32)
    tracked, status, error = cv2.calcOpticalFlowPyrLK(
        first_gray, second_gray, start.reshape(-1, 1, 2), None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    tracked = tracked.reshape(-1, 2)
    status = status.reshape(-1).astype(bool)
    error = error.reshape(-1)
    height, width = second_gray.shape
    integer = np.rint(tracked).astype(np.int32)
    inside = (
        (integer[:, 0] >= 0) & (integer[:, 0] < width) & (integer[:, 1] >= 0) & (integer[:, 1] < height)
    )
    returned = np.zeros(len(start), dtype=bool)
    valid_indices = np.where(inside)[0]
    returned[valid_indices] = second.inlier_mask[integer[valid_indices, 1], integer[valid_indices, 0]] > 0
    keep_lk = status & np.isfinite(error) & (error <= args.max_lk_error) & inside
    keep = keep_lk & returned
    start_kept, tracked_kept = start[keep], tracked[keep]
    row["seed_corners"] = int(len(start))
    row["lk_valid"] = int(keep_lk.sum())
    row["returned_to_next_plane"] = int(len(start_kept))
    if len(start_kept) < 4:
        row["status"] = "too_few_cross_plane_tracks"
        return row, start, start_kept, tracked_kept
    _, inlier_mask = cv2.findHomography(start_kept, tracked_kept, cv2.RANSAC, args.homography_threshold_px)
    homography_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    flows = np.linalg.norm(tracked_kept - start_kept, axis=1)
    row.update({
        "homography_inliers": homography_inliers,
        "homography_inlier_fraction": float(homography_inliers / len(start_kept)),
        "median_flow_px": float(np.median(flows)),
        "status": "tracked",
    })
    return row, start, start_kept, tracked_kept


def write_visualization(path: Path, first: FrameData, second: FrameData, start: np.ndarray, tracked: np.ndarray, row: dict[str, object]) -> None:
    left = first.left_local.copy()
    right = second.left_local.copy()
    for source, target in zip(start, tracked):
        x1, y1 = np.rint(source).astype(int)
        x2, y2 = np.rint(target).astype(int)
        cv2.circle(left, (x1, y1), 2, (0, 255, 0), -1)
        cv2.circle(right, (x2, y2), 2, (0, 255, 0), -1)
        cv2.line(left, (x1, y1), (x1, y1), (0, 255, 0), 1)
    sheet = np.hstack((left, right))
    label = (
        f"{first.name} -> {second.name}; cross-plane tracks={row['returned_to_next_plane']}; "
        f"H-inliers={row['homography_inliers']}; status={row['status']}"
    )
    cv2.rectangle(sheet, (0, 0), (min(1260, sheet.shape[1]), 30), (255, 255, 255), -1)
    cv2.putText(sheet, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"cannot write {path}")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    calibration_source = StereoCalibration.load(args.calibration)
    runtime_size = (args.runtime_width, args.runtime_height)
    calibration = calibration_source.for_runtime_sizes(runtime_size, runtime_size)
    pairs = read_pairs(args.left_dir, args.right_dir, args.frame_step)
    left_sapiens = load_sapiens_by_name(args.sapiens_left_json)
    right_sapiens = load_sapiens_by_name(args.sapiens_right_json)
    sample = cv2.imread(str(pairs[0][1]), cv2.IMREAD_COLOR)
    if sample is None:
        raise RuntimeError(f"cannot read {pairs[0][1]}")
    rectification = make_floor_directed_rectification(
        calibration, (sample.shape[1], sample.shape[0]), runtime_size, args.virtual_focal_px
    )
    args.output_dir.mkdir(parents=True)
    frames = [
        prepare_frame(index, name, left_path, right_path, left_sapiens, right_sapiens, rectification, runtime_size, args)
        for index, (name, left_path, right_path) in enumerate(pairs)
    ]
    rows: list[dict[str, object]] = []
    visualize = set(np.linspace(0, len(frames) - 2, min(6, len(frames) - 1), dtype=int).tolist())
    for index, (first, second) in enumerate(zip(frames[:-1], frames[1:])):
        row, _, start, tracked = tracked_transition(first, second, args)
        rows.append(row)
        if index in visualize:
            write_visualization(
                args.output_dir / "visualizations" / f"{Path(first.name).stem}_to_{Path(second.name).stem}_tracks.png",
                first, second, start, tracked, row,
            )
    with (args.output_dir / "ground_track_transitions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tracked_rows = [row for row in rows if row["status"] == "tracked"]
    metadata = {
        "experiment": "cross-frame validation of locally reconstructed floor-plane inlier tracks",
        "coordinate_frame": "local floor-directed rectification for KLT tracking; source points originate from calibrated raw-fisheye reconstruction",
        "plane_input": "left-right disparity consistent floor candidates and per-frame RANSAC from experiment_ground_plane_local_rectification",
        "track_gate": "KLT track must land inside the next independently reconstructed plane-inlier mask before homography estimation",
        "transition_count": len(rows),
        "tracked_transition_count": len(tracked_rows),
        "median_cross_plane_tracks": None if not tracked_rows else float(np.median([row["returned_to_next_plane"] for row in tracked_rows])),
        "median_homography_inlier_fraction": None if not tracked_rows else float(np.median([row["homography_inlier_fraction"] for row in tracked_rows])),
        "interpretation": "A high track return and homography support only establishes temporal consistency of the selected surface. It does not by itself prove metric floor accuracy or authorize gait-contact labels.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "# 地面候选的跨帧跟踪验证\n\n"
        "每个起始帧先独立重建局部地面候选并进行 RANSAC；KLT 仅从该平面的内点中取特征。"
        "一条轨迹必须同时通过光流检查，并落入下一帧独立平面内点掩膜，才进入单应性一致性统计。\n\n"
        "该输出用于判断平面候选是否有连续观测支持。若轨迹返回率或单应性内点率不足，平面不能用于支撑相、摆动相或脚地接触损失。\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output_dir), **metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
