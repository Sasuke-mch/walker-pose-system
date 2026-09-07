#!/usr/bin/env python3
"""Initial dense ground-plane candidate experiment for calibrated fisheye stereo.

This tool uses virtual rectification only for dense stereo matching.  It maps
each reconstructed point back to the original left-camera coordinate system
and validates the result by original-fisheye reprojection.  The fitted result
is deliberately called a lower-region dominant-plane candidate: a one-time
ground-plane calibration or IMU tracking is still required before using it as
a foot-contact constraint.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REALTIME_ROOT = PROJECT_ROOT / "realtime_app"
if str(REALTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(REALTIME_ROOT))

from pose_app.calibration import StereoCalibration


@dataclass(frozen=True)
class Rectification:
    map_left_x: np.ndarray
    map_left_y: np.ndarray
    map_right_x: np.ndarray
    map_right_y: np.ndarray
    rectification_left: np.ndarray
    projection_left: np.ndarray
    projection_right: np.ndarray


@dataclass(frozen=True)
class PlaneFit:
    normal: np.ndarray
    offset: float
    inlier_mask: np.ndarray
    median_distance_mm: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-dir", type=Path, required=True, help="upright left_ccw90 PNG directory")
    parser.add_argument("--right-dir", type=Path, required=True, help="upright right_cw90 PNG directory")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-step", type=int, default=6, help="use every Nth ordered pair")
    parser.add_argument("--runtime-width", type=int, default=960)
    parser.add_argument("--runtime-height", type=int, default=540)
    parser.add_argument("--num-disparities", type=int, default=192)
    parser.add_argument("--ransac-distance-mm", type=float, default=30.0)
    parser.add_argument("--ransac-iterations", type=int, default=700)
    parser.add_argument("--minimum-candidates", type=int, default=800)
    return parser.parse_args()


def read_image_pairs(left_dir: Path, right_dir: Path, step: int) -> list[tuple[str, Path, Path]]:
    if step < 1:
        raise ValueError("--frame-step must be at least 1")
    left_files = {path.name: path for path in left_dir.glob("pair_*.png")}
    right_files = {path.name: path for path in right_dir.glob("pair_*.png")}
    names = sorted(set(left_files) & set(right_files))
    if not names:
        raise RuntimeError("No matched pair_*.png images found")
    missing_left = sorted(set(right_files) - set(left_files))
    missing_right = sorted(set(left_files) - set(right_files))
    if missing_left or missing_right:
        raise RuntimeError(f"Input directories differ: missing_left={len(missing_left)}, missing_right={len(missing_right)}")
    return [(name, left_files[name], right_files[name]) for name in names[::step]]


def create_rectification(calibration: StereoCalibration, size: tuple[int, int]) -> Rectification:
    if calibration.camera_model != "fisheye":
        raise ValueError("This experiment is only implemented for fisheye calibration")
    width, height = size
    runtime = calibration.for_runtime_sizes(size, size)
    rect_left, rect_right, proj_left, proj_right, _ = cv2.fisheye.stereoRectify(
        runtime.left_K,
        runtime.left_D.reshape(-1, 1),
        runtime.right_K,
        runtime.right_D.reshape(-1, 1),
        (width, height),
        runtime.R,
        runtime.T.reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY,
        newImageSize=(width, height),
        balance=0.0,
        fov_scale=1.0,
    )
    map_left_x, map_left_y = cv2.fisheye.initUndistortRectifyMap(
        runtime.left_K, runtime.left_D.reshape(-1, 1), rect_left, proj_left[:, :3], (width, height), cv2.CV_32FC1
    )
    map_right_x, map_right_y = cv2.fisheye.initUndistortRectifyMap(
        runtime.right_K, runtime.right_D.reshape(-1, 1), rect_right, proj_right[:, :3], (width, height), cv2.CV_32FC1
    )
    return Rectification(map_left_x, map_left_y, map_right_x, map_right_y, rect_left, proj_left, proj_right)


def inverse_upright_rotation(left_upright: np.ndarray, right_upright: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        cv2.rotate(left_upright, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(right_upright, cv2.ROTATE_90_COUNTERCLOCKWISE),
    )


def rectify_pair(left_raw: np.ndarray, right_raw: np.ndarray, rectification: Rectification, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    width, height = size
    left = cv2.resize(left_raw, (width, height), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right_raw, (width, height), interpolation=cv2.INTER_AREA)
    return (
        cv2.remap(left, rectification.map_left_x, rectification.map_left_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT),
        cv2.remap(right, rectification.map_right_x, rectification.map_right_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT),
    )


def dense_right_to_left_disparity(left_rectified: np.ndarray, right_rectified: np.ndarray, num_disparities: int) -> np.ndarray:
    if num_disparities < 16 or num_disparities % 16:
        raise ValueError("--num-disparities must be a multiple of 16 and at least 16")
    left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    left_gray = clahe.apply(left_gray)
    right_gray = clahe.apply(right_gray)
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=7,
        P1=8 * 3 * 7 * 7,
        P2=32 * 3 * 7 * 7,
        disp12MaxDiff=2,
        uniquenessRatio=10,
        speckleWindowSize=80,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    # The calibrated projection has a positive P_right[0, 3].  Matching
    # right-to-left therefore produces d = x_right - x_left > 0.
    return matcher.compute(right_gray, left_gray).astype(np.float32) / 16.0


def reconstruct_candidates(
    disparity_right: np.ndarray,
    left_rectified: np.ndarray,
    right_rectified: np.ndarray,
    rectification: Rectification,
    calibration: StereoCalibration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = disparity_right.shape
    yy, xx_right = np.mgrid[0:height, 0:width]
    disparity = disparity_right
    fx = float(rectification.projection_left[0, 0])
    fy = float(rectification.projection_left[1, 1])
    cx = float(rectification.projection_left[0, 2])
    cy = float(rectification.projection_left[1, 2])
    baseline = abs(float(rectification.projection_right[0, 3] / fx))
    if fx <= 0.0 or fy <= 0.0 or baseline <= 0.0:
        raise RuntimeError("Invalid rectified projection matrices")
    xx_left = xx_right.astype(np.float32) - disparity
    xi_left = np.rint(xx_left).astype(np.int32)
    lower_region = yy >= int(round(height * 0.58))
    valid = (
        (disparity > 1.0)
        & np.isfinite(disparity)
        & (xi_left >= 0)
        & (xi_left < width)
        & lower_region
    )
    left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
    yy_valid, xr_valid = yy[valid], xx_right[valid]
    xl_valid = xi_left[valid]
    intensity_difference = np.abs(
        right_gray[yy_valid, xr_valid].astype(np.int16) - left_gray[yy_valid, xl_valid].astype(np.int16)
    )
    keep = intensity_difference <= 48
    yy_valid, xr_valid, xl_valid = yy_valid[keep], xr_valid[keep], xl_valid[keep]
    disp_valid = disparity[valid][keep]
    if len(disp_valid) == 0:
        return np.empty((0, 3)), np.empty((0, 2), dtype=np.int32), np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)
    depth = fx * baseline / disp_valid
    x_rectified = (xl_valid.astype(np.float64) - cx) * depth / fx
    y_rectified = (yy_valid.astype(np.float64) - cy) * depth / fy
    points_rectified = np.column_stack((x_rectified, y_rectified, depth))
    # stereoRectify gives X_rectified = R_left @ X_left_camera.
    points_left_camera = (rectification.rectification_left.T @ points_rectified.T).T
    finite = np.all(np.isfinite(points_left_camera), axis=1)
    plausible = finite & (depth > 300.0) & (depth < 6000.0)
    points_left_camera = points_left_camera[plausible]
    rectified_left_pixels = np.column_stack((xl_valid[plausible], yy_valid[plausible]))
    rectified_right_pixels = np.column_stack((xr_valid[plausible], yy_valid[plausible]))
    return points_left_camera, rectified_left_pixels, rectified_right_pixels, intensity_difference[keep][plausible]


def fit_plane_ransac(points: np.ndarray, distance_mm: float, iterations: int, seed: int) -> PlaneFit | None:
    if len(points) < 3:
        return None
    rng = np.random.default_rng(seed)
    best_mask: np.ndarray | None = None
    best_count = 0
    for _ in range(iterations):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal /= norm
        offset = -float(normal @ sample[0])
        mask = np.abs(points @ normal + offset) <= distance_mm
        count = int(mask.sum())
        if count > best_count:
            best_mask, best_count = mask, count
    if best_mask is None or best_count < 3:
        return None
    inliers = points[best_mask]
    center = np.mean(inliers, axis=0)
    _, _, vectors = np.linalg.svd(inliers - center, full_matrices=False)
    normal = vectors[-1]
    normal /= np.linalg.norm(normal)
    offset = -float(normal @ center)
    refined_mask = np.abs(points @ normal + offset) <= distance_mm
    distances = np.abs(points[refined_mask] @ normal + offset)
    return PlaneFit(normal, offset, refined_mask, float(np.median(distances)))


def original_reprojection_error(
    points_left: np.ndarray,
    rectified_left_pixels: np.ndarray,
    rectified_right_pixels: np.ndarray,
    rectification: Rectification,
    calibration: StereoCalibration,
) -> tuple[float | None, float | None]:
    if len(points_left) == 0:
        return None, None
    left_expected = np.column_stack((
        rectification.map_left_x[rectified_left_pixels[:, 1], rectified_left_pixels[:, 0]],
        rectification.map_left_y[rectified_left_pixels[:, 1], rectified_left_pixels[:, 0]],
    ))
    right_expected = np.column_stack((
        rectification.map_right_x[rectified_right_pixels[:, 1], rectified_right_pixels[:, 0]],
        rectification.map_right_y[rectified_right_pixels[:, 1], rectified_right_pixels[:, 0]],
    ))
    projected_left = calibration.project_left(points_left)
    projected_right = calibration.project_right(points_left)
    left_error = np.linalg.norm(projected_left - left_expected, axis=1)
    right_error = np.linalg.norm(projected_right - right_expected, axis=1)
    return float(np.median(left_error)), float(np.median(right_error))


def colorize_disparity(disparity: np.ndarray) -> np.ndarray:
    valid = disparity > 1.0
    canvas = np.zeros(disparity.shape, dtype=np.uint8)
    if np.any(valid):
        lo, hi = np.percentile(disparity[valid], (2, 98))
        scale = np.clip((disparity - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        canvas[valid] = np.rint(scale[valid] * 255.0).astype(np.uint8)
    return cv2.applyColorMap(canvas, cv2.COLORMAP_TURBO)


def draw_visualization(
    path: Path,
    left_raw: np.ndarray,
    left_rectified: np.ndarray,
    right_rectified: np.ndarray,
    disparity: np.ndarray,
    left_pixels: np.ndarray,
    inlier_mask: np.ndarray | None,
    rectification: Rectification,
    title: str,
) -> None:
    height, width = disparity.shape
    raw_thumb = cv2.resize(left_raw, (width, height), interpolation=cv2.INTER_AREA)
    left_panel = left_rectified.copy()
    if len(left_pixels):
        candidate = left_pixels[::max(1, len(left_pixels) // 4500)]
        left_panel[candidate[:, 1], candidate[:, 0]] = (0, 180, 255)
        raw_candidate = np.column_stack((
            np.rint(rectification.map_left_x[candidate[:, 1], candidate[:, 0]]).astype(np.int32),
            np.rint(rectification.map_left_y[candidate[:, 1], candidate[:, 0]]).astype(np.int32),
        ))
        raw_candidate[:, 0] = np.clip(raw_candidate[:, 0], 0, width - 1)
        raw_candidate[:, 1] = np.clip(raw_candidate[:, 1], 0, height - 1)
        raw_thumb[raw_candidate[:, 1], raw_candidate[:, 0]] = (0, 180, 255)
        if inlier_mask is not None:
            inliers = left_pixels[inlier_mask]
            inliers = inliers[::max(1, len(inliers) // 4500)]
            left_panel[inliers[:, 1], inliers[:, 0]] = (0, 255, 0)
            raw_inliers = np.column_stack((
                np.rint(rectification.map_left_x[inliers[:, 1], inliers[:, 0]]).astype(np.int32),
                np.rint(rectification.map_left_y[inliers[:, 1], inliers[:, 0]]).astype(np.int32),
            ))
            raw_inliers[:, 0] = np.clip(raw_inliers[:, 0], 0, width - 1)
            raw_inliers[:, 1] = np.clip(raw_inliers[:, 1], 0, height - 1)
            raw_thumb[raw_inliers[:, 1], raw_inliers[:, 0]] = (0, 255, 0)
    panels = [raw_thumb, left_panel, right_rectified, colorize_disparity(disparity)]
    labels = ["raw left (orange: candidates; green: RANSAC inliers)", "rectified left (orange: candidates; green: RANSAC inliers)", "rectified right", "right-to-left disparity"]
    labelled: list[np.ndarray] = []
    for panel, label in zip(panels, labels):
        result = panel.copy()
        cv2.rectangle(result, (0, 0), (min(width, 520), 34), (255, 255, 255), -1)
        cv2.putText(result, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1, cv2.LINE_AA)
        labelled.append(result)
    sheet = np.vstack((np.hstack((labelled[0], labelled[1])), np.hstack((labelled[2], labelled[3]))))
    cv2.rectangle(sheet, (0, 0), (min(sheet.shape[1], 900), 38), (255, 255, 255), -1)
    cv2.putText(sheet, title, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"Cannot write {path}")


def write_ply(path: Path, points: np.ndarray) -> None:
    header = "\n".join((
        "ply", "format ascii 1.0", f"element vertex {len(points)}",
        "property float x", "property float y", "property float z", "end_header",
    ))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(header + "\n")
        for point in points:
            handle.write(f"{point[0]:.5f} {point[1]:.5f} {point[2]:.5f}\n")


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cosine = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def main() -> None:
    args = parse_args()
    if args.runtime_width <= 0 or args.runtime_height <= 0:
        raise ValueError("Runtime dimensions must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    calibration = StereoCalibration.load(args.calibration)
    pairs = read_image_pairs(args.left_dir, args.right_dir, args.frame_step)
    size = (args.runtime_width, args.runtime_height)
    rectification = create_rectification(calibration, size)
    args.output_dir.mkdir(parents=True)
    visual_indices = set(np.linspace(0, len(pairs) - 1, min(6, len(pairs)), dtype=int).tolist())
    rows: list[dict[str, object]] = []
    accepted_normals: list[np.ndarray] = []
    for sequence_index, (name, left_path, right_path) in enumerate(pairs):
        left_upright = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right_upright = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left_upright is None or right_upright is None:
            raise RuntimeError(f"Cannot read {name}")
        left_raw, right_raw = inverse_upright_rotation(left_upright, right_upright)
        left_rectified, right_rectified = rectify_pair(left_raw, right_raw, rectification, size)
        disparity = dense_right_to_left_disparity(left_rectified, right_rectified, args.num_disparities)
        points, left_pixels, right_pixels, intensity_difference = reconstruct_candidates(
            disparity, left_rectified, right_rectified, rectification, calibration.for_runtime_sizes(size, size)
        )
        if len(points) > 12000:
            sample = np.linspace(0, len(points) - 1, 12000, dtype=np.int64)
            fit_points = points[sample]
            fit_indices = sample
        else:
            fit_points = points
            fit_indices = np.arange(len(points))
        fit = fit_plane_ransac(fit_points, args.ransac_distance_mm, args.ransac_iterations, seed=sequence_index + 20260831)
        full_inlier_mask: np.ndarray | None = None
        accepted = False
        if fit is not None:
            full_distance = np.abs(points @ fit.normal + fit.offset)
            full_inlier_mask = full_distance <= args.ransac_distance_mm
            accepted = len(points) >= args.minimum_candidates and int(full_inlier_mask.sum()) >= args.minimum_candidates
            if accepted:
                accepted_normals.append(fit.normal)
        inlier_points = points[full_inlier_mask] if full_inlier_mask is not None else np.empty((0, 3))
        left_error, right_error = original_reprojection_error(
            inlier_points,
            left_pixels[full_inlier_mask] if full_inlier_mask is not None else np.empty((0, 2), dtype=np.int32),
            right_pixels[full_inlier_mask] if full_inlier_mask is not None else np.empty((0, 2), dtype=np.int32),
            rectification,
            calibration.for_runtime_sizes(size, size),
        )
        row: dict[str, object] = {
            "sequence_index": sequence_index,
            "file_name": name,
            "candidate_points": int(len(points)),
            "mean_abs_match_intensity": float(np.mean(intensity_difference)) if len(intensity_difference) else None,
            "ransac_inliers": int(len(inlier_points)),
            "ransac_inlier_fraction": float(len(inlier_points) / len(points)) if len(points) else None,
            "accepted_lower_region_plane": accepted,
            "normal_x": float(fit.normal[0]) if fit is not None else None,
            "normal_y": float(fit.normal[1]) if fit is not None else None,
            "normal_z": float(fit.normal[2]) if fit is not None else None,
            "offset_mm": float(fit.offset) if fit is not None else None,
            "median_inlier_distance_mm": float(fit.median_distance_mm) if fit is not None else None,
            "median_fisheye_reprojection_left_px": left_error,
            "median_fisheye_reprojection_right_px": right_error,
        }
        rows.append(row)
        if sequence_index in visual_indices:
            draw_visualization(
                args.output_dir / "visualizations" / f"{Path(name).stem}_ground_candidate.png",
                left_raw, left_rectified, right_rectified, disparity, left_pixels, full_inlier_mask,
                rectification,
                f"{name}: lower-region dominant-plane candidate",
            )
            if len(inlier_points):
                write_ply(args.output_dir / "pointclouds" / f"{Path(name).stem}_ransac_inliers_left_camera.ply", inlier_points)
    normal_reference = np.median(np.asarray(accepted_normals), axis=0) if accepted_normals else None
    if normal_reference is not None:
        normal_reference /= np.linalg.norm(normal_reference)
        for row in rows:
            if row["accepted_lower_region_plane"]:
                normal = np.asarray([row["normal_x"], row["normal_y"], row["normal_z"]], dtype=np.float64)
                row["normal_angle_to_sequence_median_deg"] = angle_deg(normal, normal_reference)
            else:
                row["normal_angle_to_sequence_median_deg"] = None
    else:
        for row in rows:
            row["normal_angle_to_sequence_median_deg"] = None
    with (args.output_dir / "per_frame_plane.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    accepted_rows = [row for row in rows if row["accepted_lower_region_plane"]]
    metadata = {
        "experiment": "initial dense lower-region plane candidates by rectified fisheye stereo plus RANSAC",
        "input_geometry": "upright model inputs inverse-rotated to raw fisheye before geometry",
        "matching_geometry": "virtual rectification used only for dense correspondence; all 3D points transformed to left-camera coordinates",
        "coordinate_frame": "left_camera",
        "length_unit": calibration.length_unit,
        "plane_status": "candidate only; not a calibrated world-ground plane and not a foot-contact label",
        "frame_count": len(rows),
        "accepted_plane_count": len(accepted_rows),
        "median_normal": normal_reference.tolist() if normal_reference is not None else None,
        "median_normal_deviation_deg": float(np.median([row["normal_angle_to_sequence_median_deg"] for row in accepted_rows])) if accepted_rows else None,
        "parameters": {
            "frame_step": args.frame_step,
            "runtime_size": size,
            "num_disparities": args.num_disparities,
            "lower_rectified_row_fraction": 0.58,
            "photometric_difference_max": 48,
            "depth_range_mm": [300, 6000],
            "ransac_distance_mm": args.ransac_distance_mm,
            "ransac_iterations": args.ransac_iterations,
            "minimum_candidates": args.minimum_candidates,
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "# 初步地面点云与 RANSAC 平面实验\n\n"
        "本实验从左右原始鱼眼图构造密集匹配点，并在校正图下部区域以 RANSAC 拟合主平面。"
        "校正仅用于左右像素匹配；三维点已转换回原始左相机坐标，并记录原始鱼眼重投影误差。\n\n"
        "绿色 RANSAC 内点只能说明该帧下部区域存在一个支持度较高的几何平面。"
        "它尚不是已标定地面，也没有被用于脚部接触、支撑相/摆动相判断或任何模型损失。"
        "下一阶段必须验证平面法向和距离在平坦地面短窗内稳定，再以地面板或 IMU 给出世界地面参考。\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output_dir),
        "frames": len(rows),
        "accepted_plane_frames": len(accepted_rows),
        "median_normal_deviation_deg": metadata["median_normal_deviation_deg"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
