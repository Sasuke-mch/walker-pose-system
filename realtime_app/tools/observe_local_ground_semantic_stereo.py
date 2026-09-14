#!/usr/bin/env python3
"""Audit a local plane observation from externally supplied floor masks and fisheye stereo.

This is an engineering diagnostic, not a ground-truth, contact, or gait tool.
It deliberately separates a floor-identity input (one binary mask per upright
image) from stereo geometry.  A lower-image fallback exists only to exercise
the geometry path; it is *not* semantic evidence and can never emit ``direct``.

The local virtual cameras are used only to establish dense left/right matches.
All reported points and planes are in the calibrated raw left-camera frame.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REALTIME_ROOT = PROJECT_ROOT / "realtime_app"
if str(REALTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(REALTIME_ROOT))

from pose_app.calibration import StereoCalibration


SCHEMA_VERSION = "local_ground_state_v1"


@dataclass(frozen=True)
class LocalRectification:
    left_map_x: np.ndarray
    left_map_y: np.ndarray
    right_map_x: np.ndarray
    right_map_y: np.ndarray
    rotation_left: np.ndarray
    virtual_K: np.ndarray
    translation_right: np.ndarray


@dataclass(frozen=True)
class PlaneFit:
    normal: np.ndarray
    offset: float
    inlier_mask: np.ndarray
    median_distance_mm: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-dir", type=Path, required=True, help="upright left_ccw90 pair_*.png frames")
    parser.add_argument("--right-dir", type=Path, required=True, help="upright right_cw90 pair_*.png frames")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--semantic-left-dir", type=Path, help="upright binary floor masks, matching frame names")
    parser.add_argument("--semantic-right-dir", type=Path, help="upright binary floor masks, matching frame names")
    parser.add_argument(
        "--semantic-identity-evidence", choices=("provided_unvalidated", "manually_audited"),
        default="provided_unvalidated",
        help="Only manually_audited masks are eligible for a direct observation.",
    )
    parser.add_argument(
        "--fallback-candidate", choices=("none", "lower_region_unknown"), default="none",
        help="Non-semantic diagnostic fallback. It is always unavailable.",
    )
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all selected frames")
    parser.add_argument("--runtime-width", type=int, default=960)
    parser.add_argument("--runtime-height", type=int, default=540)
    parser.add_argument("--virtual-focal-px", type=float, default=330.0)
    parser.add_argument("--num-disparities", type=int, default=160)
    parser.add_argument("--lr-consistency-px", type=float, default=1.5)
    parser.add_argument("--ransac-distance-mm", type=float, default=25.0)
    parser.add_argument("--ransac-iterations", type=int, default=900)
    parser.add_argument("--minimum-candidates", type=int, default=250)
    parser.add_argument("--minimum-inliers", type=int, default=150)
    parser.add_argument("--minimum-inlier-fraction", type=float, default=0.25)
    parser.add_argument("--minimum-coverage-fraction", type=float, default=0.01)
    parser.add_argument("--maximum-median-residual-mm", type=float, default=25.0)
    parser.add_argument("--maximum-median-reprojection-px", type=float, default=5.0)
    parser.add_argument("--visualize-every", type=int, default=1, help="0 disables diagnostic images")
    return parser.parse_args()


def read_pairs(left_dir: Path, right_dir: Path, step: int, maximum: int) -> list[tuple[str, Path, Path]]:
    if step < 1:
        raise ValueError("--frame-step must be positive")
    left = {path.name: path for path in left_dir.glob("pair_*.png")}
    right = {path.name: path for path in right_dir.glob("pair_*.png")}
    if not left or set(left) != set(right):
        raise RuntimeError("left/right directories must contain the same non-empty pair_*.png set")
    selected = [(name, left[name], right[name]) for name in sorted(left)[::step]]
    return selected if maximum <= 0 else selected[:maximum]


def inverse_upright(left_upright: np.ndarray, right_upright: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return cv2.rotate(left_upright, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(right_upright, cv2.ROTATE_90_COUNTERCLOCKWISE)


def rotate_mask_to_raw(mask: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    if side == "right":
        return cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("side must be left or right")


def upright_point_to_raw(point: np.ndarray, side: str, upright_size: tuple[int, int]) -> np.ndarray:
    width, height = upright_size
    x, y = float(point[0]), float(point[1])
    if side == "left":
        return np.asarray((height - 1.0 - y, x), dtype=np.float64)
    if side == "right":
        return np.asarray((y, width - 1.0 - x), dtype=np.float64)
    raise ValueError("side must be left or right")


def mask_seed_upright(mask: np.ndarray) -> np.ndarray | None:
    yy, xx = np.nonzero(mask > 0)
    if len(xx) == 0:
        return None
    # The median is robust against isolated mask fragments and does not encode
    # a presumed image-bottom floor direction.
    return np.asarray((float(np.median(xx)), float(np.median(yy))), dtype=np.float64)


def ray_from_raw_pixel(calibration: StereoCalibration, pixel: np.ndarray, side: str) -> np.ndarray:
    normalized = calibration.undistort_normalized(pixel.reshape(1, 2), side)[0]
    ray = np.asarray((normalized[0], normalized[1], 1.0), dtype=np.float64)
    return ray / np.linalg.norm(ray)


def make_mask_directed_rectification(
    calibration: StereoCalibration,
    left_seed_raw: np.ndarray,
    right_seed_raw: np.ndarray,
    runtime_size: tuple[int, int],
    focal_px: float,
) -> LocalRectification:
    if focal_px <= 0:
        raise ValueError("virtual focal length must be positive")
    left_ray = ray_from_raw_pixel(calibration, left_seed_raw, "left")
    right_ray_in_left = calibration.R.T @ ray_from_raw_pixel(calibration, right_seed_raw, "right")
    baseline_left = -calibration.R.T @ calibration.T
    baseline_left /= np.linalg.norm(baseline_left)
    forward = left_ray + right_ray_in_left
    forward -= baseline_left * float(forward @ baseline_left)
    if np.linalg.norm(forward) < 1e-9:
        raise RuntimeError("mask seed produces a degenerate local view")
    forward /= np.linalg.norm(forward)
    vertical = np.cross(forward, baseline_left)
    vertical /= np.linalg.norm(vertical)
    rotation_left = np.stack((baseline_left, vertical, forward))
    rotation_right = rotation_left @ calibration.R.T
    translation_right = rotation_right @ calibration.T
    if abs(float(translation_right[1])) > 1e-6 or abs(float(translation_right[2])) > 1e-6:
        raise RuntimeError("local rectification did not align the stereo baseline")
    width, height = runtime_size
    virtual_K = np.asarray(((focal_px, 0.0, width / 2.0), (0.0, focal_px, height / 2.0), (0.0, 0.0, 1.0)), dtype=np.float64)
    left_map_x, left_map_y = cv2.fisheye.initUndistortRectifyMap(
        calibration.left_K, calibration.left_D.reshape(-1, 1), rotation_left, virtual_K, runtime_size, cv2.CV_32FC1
    )
    right_map_x, right_map_y = cv2.fisheye.initUndistortRectifyMap(
        calibration.right_K, calibration.right_D.reshape(-1, 1), rotation_right, virtual_K, runtime_size, cv2.CV_32FC1
    )
    return LocalRectification(left_map_x, left_map_y, right_map_x, right_map_y, rotation_left, virtual_K, translation_right)


def dense_disparity(reference: np.ndarray, partner: np.ndarray, num_disparities: int) -> np.ndarray:
    if num_disparities < 16 or num_disparities % 16:
        raise ValueError("--num-disparities must be a multiple of 16")
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    partner_gray = cv2.cvtColor(partner, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    matcher = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=num_disparities, blockSize=7,
        P1=8 * 3 * 7 * 7, P2=32 * 3 * 7 * 7, disp12MaxDiff=1,
        uniquenessRatio=12, speckleWindowSize=60, speckleRange=2,
        preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return matcher.compute(clahe.apply(reference_gray), clahe.apply(partner_gray)).astype(np.float32) / 16.0


def reconstruct_candidates(
    left: np.ndarray, right: np.ndarray, left_mask: np.ndarray, right_mask: np.ndarray,
    rectification: LocalRectification, num_disparities: int, lr_consistency_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return left-camera points and local image correspondence pixels."""
    height, width = left.shape[:2]
    tx = float(rectification.translation_right[0])
    if abs(tx) < 1e-9:
        raise RuntimeError("zero horizontal local stereo baseline")
    reference_is_left = tx < 0.0
    if reference_is_left:
        disparity = dense_disparity(left, right, num_disparities)
        reverse = dense_disparity(right, left, num_disparities)
        yy, xs = np.mgrid[0:height, 0:width]
        partner_x = np.rint(xs.astype(np.float32) - disparity).astype(np.int32)
        at_partner = reverse[yy, np.clip(partner_x, 0, width - 1)]
        valid = ((disparity > 1.0) & (partner_x >= 0) & (partner_x < width) & (left_mask > 0)
                 & (right_mask[yy, np.clip(partner_x, 0, width - 1)] > 0) & (at_partner > 1.0)
                 & (np.abs(disparity - at_partner) <= lr_consistency_px))
        left_x, right_x = xs[valid], partner_x[valid]
        disparities = disparity[valid]
    else:
        disparity = dense_disparity(right, left, num_disparities)
        reverse = dense_disparity(left, right, num_disparities)
        yy, right_x_grid = np.mgrid[0:height, 0:width]
        left_x_grid = np.rint(right_x_grid.astype(np.float32) - disparity).astype(np.int32)
        at_partner = reverse[yy, np.clip(left_x_grid, 0, width - 1)]
        valid = ((disparity > 1.0) & (left_x_grid >= 0) & (left_x_grid < width) & (right_mask > 0)
                 & (left_mask[yy, np.clip(left_x_grid, 0, width - 1)] > 0) & (at_partner > 1.0)
                 & (np.abs(disparity - at_partner) <= lr_consistency_px))
        left_x, right_x = left_x_grid[valid], right_x_grid[valid]
        disparities = disparity[valid]
    ys = yy[valid]
    if len(ys) == 0:
        return np.empty((0, 3)), np.empty((0, 2), dtype=np.int32), np.empty((0, 2), dtype=np.int32), disparity
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    photometric = np.abs(left_gray[ys, left_x].astype(np.int16) - right_gray[ys, right_x].astype(np.int16))
    keep = photometric <= 45
    left_x, right_x, ys, disparities = left_x[keep], right_x[keep], ys[keep], disparities[keep]
    depth = abs(float(rectification.virtual_K[0, 0] * tx)) / disparities
    x = (left_x.astype(np.float64) - rectification.virtual_K[0, 2]) * depth / rectification.virtual_K[0, 0]
    y = (ys.astype(np.float64) - rectification.virtual_K[1, 2]) * depth / rectification.virtual_K[1, 1]
    points_rectified = np.column_stack((x, y, depth))
    points_left = (rectification.rotation_left.T @ points_rectified.T).T
    finite = np.all(np.isfinite(points_left), axis=1) & (depth > 250.0) & (depth < 8000.0)
    return points_left[finite], np.column_stack((left_x[finite], ys[finite])).astype(np.int32), np.column_stack((right_x[finite], ys[finite])).astype(np.int32), disparity


def fit_plane_ransac(points: np.ndarray, distance_mm: float, iterations: int, seed: int) -> PlaneFit | None:
    if len(points) < 3:
        return None
    rng = np.random.default_rng(seed)
    best_mask: np.ndarray | None = None
    best_count = 0
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
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
    centre = np.mean(points[best_mask], axis=0)
    _, _, vectors = np.linalg.svd(points[best_mask] - centre, full_matrices=False)
    normal = vectors[-1]
    normal /= np.linalg.norm(normal)
    offset = -float(normal @ centre)
    mask = np.abs(points @ normal + offset) <= distance_mm
    residual = float(np.median(np.abs(points[mask] @ normal + offset))) if int(mask.sum()) else float("inf")
    return PlaneFit(normal, offset, mask, residual)


def local_coverage_fraction(pixels: np.ndarray, image_shape: tuple[int, int]) -> float:
    if len(pixels) < 3:
        return 0.0
    height, width = image_shape
    hull = cv2.convexHull(pixels.astype(np.float32))
    return float(cv2.contourArea(hull) / max(1.0, float(width * height)))


def reprojection_medians(
    points_left: np.ndarray, left_pixels: np.ndarray, right_pixels: np.ndarray,
    rectification: LocalRectification, calibration: StereoCalibration,
) -> tuple[float | None, float | None]:
    if len(points_left) == 0:
        return None, None
    left_expected = np.column_stack((rectification.left_map_x[left_pixels[:, 1], left_pixels[:, 0]], rectification.left_map_y[left_pixels[:, 1], left_pixels[:, 0]]))
    right_expected = np.column_stack((rectification.right_map_x[right_pixels[:, 1], right_pixels[:, 0]], rectification.right_map_y[right_pixels[:, 1], right_pixels[:, 0]]))
    left_error = np.linalg.norm(calibration.project_left(points_left) - left_expected, axis=1)
    right_error = np.linalg.norm(calibration.project_right(points_left) - right_expected, axis=1)
    return float(np.median(left_error)), float(np.median(right_error))


def decide_observation(
    *, semantic_source: str, semantic_identity_evidence: str, candidate_count: int, inlier_count: int,
    inlier_fraction: float | None, coverage_fraction: float, residual_mm: float | None,
    reprojection_left_px: float | None, reprojection_right_px: float | None, args: argparse.Namespace,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if semantic_source != "external_binary_mask":
        reasons.append("semantic_evidence_missing")
    elif semantic_identity_evidence != "manually_audited":
        reasons.append("semantic_identity_unvalidated")
    if candidate_count < args.minimum_candidates:
        reasons.append("insufficient_stereo_candidates")
    if inlier_count < args.minimum_inliers:
        reasons.append("insufficient_plane_inliers")
    if inlier_fraction is None or inlier_fraction < args.minimum_inlier_fraction:
        reasons.append("low_ransac_inlier_fraction")
    if coverage_fraction < args.minimum_coverage_fraction:
        reasons.append("insufficient_inlier_coverage")
    if residual_mm is None or residual_mm > args.maximum_median_residual_mm:
        reasons.append("high_plane_residual")
    if reprojection_left_px is None or reprojection_right_px is None or max(reprojection_left_px, reprojection_right_px) > args.maximum_median_reprojection_px:
        reasons.append("high_or_missing_fisheye_reprojection")
    return ("direct", []) if not reasons else ("unavailable", reasons)


def canonical_plane_toward_camera(normal: np.ndarray, offset_mm: float) -> dict[str, Any] | None:
    """Return a signed local plane suitable for downstream support auditing.

    RANSAC/SVD leaves a plane normal sign arbitrary.  Downstream walker and
    visual-inertial consumers need a stable convention: the left-camera origin
    must be on the positive side of ``n.T @ X + d = 0``.  With that convention
    the normal points toward the camera and ``d`` is its positive plane
    distance.  A candidate through the camera is not usable as ground.
    """
    unit = np.asarray(normal, dtype=np.float64)
    magnitude = float(np.linalg.norm(unit))
    if not np.all(np.isfinite(unit)) or not np.isfinite(offset_mm) or magnitude <= 1e-9:
        return None
    unit /= magnitude
    signed_offset = float(offset_mm) / magnitude
    if signed_offset < 0.0:
        unit = -unit
        signed_offset = -signed_offset
    if signed_offset <= 1e-9:
        return None
    return {
        "normal_left_camera": [float(value) for value in unit],
        "offset_mm": signed_offset,
        "normal_orientation": "toward_camera",
    }


def lower_region_unknown(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    output = np.zeros((height, width), dtype=np.uint8)
    output[int(round(0.58 * height)):, :] = 255
    return output


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"cannot read semantic mask: {path}")
    if image.shape != shape:
        image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (image > 0).astype(np.uint8) * 255


def draw_points(image: np.ndarray, points: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    output = image.copy()
    if len(points):
        sampled = points[::max(1, len(points) // 4000)]
        sampled = np.rint(sampled).astype(np.int32)
        sampled[:, 0] = np.clip(sampled[:, 0], 0, output.shape[1] - 1)
        sampled[:, 1] = np.clip(sampled[:, 1], 0, output.shape[0] - 1)
        output[sampled[:, 1], sampled[:, 0]] = color
    return output


def write_visualization(path: Path, left_raw: np.ndarray, left_local: np.ndarray, right_local: np.ndarray, left_pixels: np.ndarray, inlier_mask: np.ndarray | None, rectification: LocalRectification, title: str) -> None:
    raw_pixels = np.column_stack((rectification.left_map_x[left_pixels[:, 1], left_pixels[:, 0]], rectification.left_map_y[left_pixels[:, 1], left_pixels[:, 0]])) if len(left_pixels) else np.empty((0, 2))
    raw_panel = draw_points(left_raw, raw_pixels, (0, 170, 255))
    local_panel = draw_points(left_local, left_pixels, (0, 170, 255))
    if inlier_mask is not None:
        raw_panel = draw_points(raw_panel, raw_pixels[inlier_mask], (0, 255, 0))
        local_panel = draw_points(local_panel, left_pixels[inlier_mask], (0, 255, 0))
    raw_panel = cv2.resize(raw_panel, (960, 540), interpolation=cv2.INTER_AREA)
    panels = [raw_panel, left_local, right_local, local_panel]
    labels = ["raw left: orange candidates, green inliers", "mask-directed local left", "mask-directed local right", "local candidates / inliers"]
    rendered = []
    for panel, label in zip(panels, labels):
        current = panel.copy()
        cv2.rectangle(current, (0, 0), (min(820, current.shape[1]), 28), (255, 255, 255), -1)
        cv2.putText(current, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        rendered.append(current)
    sheet = np.vstack((np.hstack((rendered[0], rendered[1])), np.hstack((rendered[2], rendered[3]))))
    cv2.rectangle(sheet, (0, 0), (min(1520, sheet.shape[1]), 34), (255, 255, 255), -1)
    cv2.putText(sheet, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"cannot write {path}")


def write_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        for x, y, z in points:
            handle.write(f"{x:.4f} {y:.4f} {z:.4f}\n")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    has_external_masks = args.semantic_left_dir is not None or args.semantic_right_dir is not None
    if (args.semantic_left_dir is None) != (args.semantic_right_dir is None):
        raise ValueError("provide both semantic mask directories or neither")
    if not has_external_masks and args.fallback_candidate == "none":
        raise ValueError("no semantic masks: choose --fallback-candidate lower_region_unknown only for an unavailable diagnostic run")
    pairs = read_pairs(args.left_dir, args.right_dir, args.frame_step, args.max_frames)
    runtime_size = (args.runtime_width, args.runtime_height)
    source_calibration = StereoCalibration.load(args.calibration)
    calibration = source_calibration.for_runtime_sizes(runtime_size, runtime_size)
    args.output_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for index, (name, left_path, right_path) in enumerate(pairs):
        left_upright = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right_upright = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left_upright is None or right_upright is None:
            raise RuntimeError(f"cannot read pair {name}")
        shape = left_upright.shape[:2]
        if right_upright.shape[:2] != shape:
            raise RuntimeError(f"upright left/right shape mismatch for {name}")
        if has_external_masks:
            left_mask_upright = load_mask(args.semantic_left_dir / name, shape)
            right_mask_upright = load_mask(args.semantic_right_dir / name, shape)
            semantic_source = "external_binary_mask"
        else:
            left_mask_upright = lower_region_unknown(shape)
            right_mask_upright = lower_region_unknown(shape)
            semantic_source = "lower_region_unknown_nonsemantic"
        left_seed = mask_seed_upright(left_mask_upright)
        right_seed = mask_seed_upright(right_mask_upright)
        reasons: list[str] = []
        plane: dict[str, Any] | None = None
        evidence: dict[str, Any] = {
            "semantic_source": semantic_source,
            "semantic_identity_evidence": args.semantic_identity_evidence if has_external_masks else "not_available",
            "left_mask_fraction_upright": float((left_mask_upright > 0).mean()),
            "right_mask_fraction_upright": float((right_mask_upright > 0).mean()),
            "matching": "mask-directed local fisheye rectification; strict left-right disparity consistency; points transformed to left_camera",
        }
        quality: dict[str, Any] = {"candidate_points": 0, "ransac_inliers": 0, "ransac_inlier_fraction": None, "inlier_coverage_fraction": 0.0, "median_plane_residual_mm": None, "median_fisheye_reprojection_left_px": None, "median_fisheye_reprojection_right_px": None}
        visual_context: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, LocalRectification] | None = None
        if left_seed is None or right_seed is None:
            state, decision_reasons = "unavailable", ["empty_semantic_candidate_mask"]
            reasons.extend(decision_reasons)
        else:
            left_raw, right_raw = inverse_upright(left_upright, right_upright)
            raw_size = (left_raw.shape[1], left_raw.shape[0])
            left_seed_raw = upright_point_to_raw(left_seed, "left", (shape[1], shape[0])) * np.asarray((runtime_size[0] / raw_size[0], runtime_size[1] / raw_size[1]))
            right_seed_raw = upright_point_to_raw(right_seed, "right", (shape[1], shape[0])) * np.asarray((runtime_size[0] / raw_size[0], runtime_size[1] / raw_size[1]))
            rectification = make_mask_directed_rectification(calibration, left_seed_raw, right_seed_raw, runtime_size, args.virtual_focal_px)
            left_raw = cv2.resize(left_raw, runtime_size, interpolation=cv2.INTER_AREA)
            right_raw = cv2.resize(right_raw, runtime_size, interpolation=cv2.INTER_AREA)
            left_mask_raw = cv2.resize(rotate_mask_to_raw(left_mask_upright, "left"), runtime_size, interpolation=cv2.INTER_NEAREST)
            right_mask_raw = cv2.resize(rotate_mask_to_raw(right_mask_upright, "right"), runtime_size, interpolation=cv2.INTER_NEAREST)
            left_local = cv2.remap(left_raw, rectification.left_map_x, rectification.left_map_y, cv2.INTER_LINEAR)
            right_local = cv2.remap(right_raw, rectification.right_map_x, rectification.right_map_y, cv2.INTER_LINEAR)
            left_mask_local = cv2.remap(left_mask_raw, rectification.left_map_x, rectification.left_map_y, cv2.INTER_NEAREST)
            right_mask_local = cv2.remap(right_mask_raw, rectification.right_map_x, rectification.right_map_y, cv2.INTER_NEAREST)
            points, left_pixels, right_pixels, _ = reconstruct_candidates(left_local, right_local, left_mask_local, right_mask_local, rectification, args.num_disparities, args.lr_consistency_px)
            fit_points = points if len(points) <= 12000 else points[np.linspace(0, len(points) - 1, 12000, dtype=np.int64)]
            fit = fit_plane_ransac(fit_points, args.ransac_distance_mm, args.ransac_iterations, 20260909 + index)
            inlier_mask = None if fit is None else np.abs(points @ fit.normal + fit.offset) <= args.ransac_distance_mm
            inliers = points[inlier_mask] if inlier_mask is not None else np.empty((0, 3))
            left_error, right_error = reprojection_medians(inliers, left_pixels[inlier_mask] if inlier_mask is not None else np.empty((0, 2), dtype=np.int32), right_pixels[inlier_mask] if inlier_mask is not None else np.empty((0, 2), dtype=np.int32), rectification, calibration)
            fraction = None if len(points) == 0 else float(len(inliers) / len(points))
            coverage = local_coverage_fraction(left_pixels[inlier_mask], left_local.shape[:2]) if inlier_mask is not None else 0.0
            quality.update({"candidate_points": int(len(points)), "ransac_inliers": int(len(inliers)), "ransac_inlier_fraction": fraction, "inlier_coverage_fraction": coverage, "median_plane_residual_mm": None if fit is None else fit.median_distance_mm, "median_fisheye_reprojection_left_px": left_error, "median_fisheye_reprojection_right_px": right_error})
            state, decision_reasons = decide_observation(semantic_source=semantic_source, semantic_identity_evidence=evidence["semantic_identity_evidence"], candidate_count=len(points), inlier_count=len(inliers), inlier_fraction=fraction, coverage_fraction=coverage, residual_mm=None if fit is None else fit.median_distance_mm, reprojection_left_px=left_error, reprojection_right_px=right_error, args=args)
            reasons.extend(decision_reasons)
            if fit is not None:
                plane = canonical_plane_toward_camera(fit.normal, float(fit.offset))
                if state == "direct" and plane is None:
                    state = "unavailable"
                    reasons.append("invalid_signed_plane_candidate")
            visual_context = (left_raw, left_local, right_local, left_pixels, inlier_mask, rectification)
            if len(inliers):
                write_ply(args.output_dir / "pointclouds" / f"{Path(name).stem}_plane_inliers_left_camera.ply", inliers)
        pair_id = int(Path(name).stem.removeprefix("pair_"))
        unaccepted_candidate_plane = plane if state != "direct" else None
        direct_plane = plane if state == "direct" else None
        ground_state: dict[str, Any] = {
            "status": state,
            "reasons": reasons,
            "provenance": "semantic_mask_direct_stereo" if state == "direct" else None,
        }
        if state == "direct":
            if direct_plane is None:
                raise RuntimeError("direct local-ground record is missing a signed plane")
            ground_state.update({"coordinate_frame": "left_camera", "length_unit": "millimeter", "plane": direct_plane})
        row = {
            "schema_version": SCHEMA_VERSION,
            "pair_id": pair_id,
            "frame_index": pair_id,
            "frame_id": name,
            "coordinate_frame": "left_camera",
            "left_camera": "cam0",
            "length_unit": "millimeter",
            "observation_state": state,
            "reason": reasons,
            "plane": direct_plane,
            "unaccepted_candidate_plane": unaccepted_candidate_plane,
            "plane_in_left_camera": None if direct_plane is None else {
                "normal_toward_camera_unit": direct_plane["normal_left_camera"],
                "offset_mm": direct_plane["offset_mm"],
            },
            "ground_state": ground_state,
            "evidence": evidence,
            "quality": quality,
        }
        rows.append(row)
        if visual_context is not None and args.visualize_every > 0 and index % args.visualize_every == 0:
            write_visualization(args.output_dir / "visualizations" / f"{Path(name).stem}_local_ground.png", *visual_context, f"{name}: {state}; {', '.join(reasons) if reasons else 'all internal gates passed'}")
    with (args.output_dir / "local_ground_state.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    unavailable_reasons: dict[str, int] = {}
    for row in rows:
        for reason in row["reason"]:
            unavailable_reasons[reason] = unavailable_reasons.get(reason, 0) + 1
    summary = {"schema_version": SCHEMA_VERSION, "frame_count": len(rows), "direct_count": sum(row["observation_state"] == "direct" for row in rows), "unavailable_count": sum(row["observation_state"] == "unavailable" for row in rows), "unavailable_reason_counts": unavailable_reasons, "interpretation_boundary": "Internal floor-identity and calibrated stereo-geometry diagnostics only. These records are not ground truth, real ground-plane accuracy, foot contact, support phase, or gait measurements."}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata = {"schema_version": SCHEMA_VERSION, "inputs": {"left_dir": str(args.left_dir), "right_dir": str(args.right_dir), "calibration": str(args.calibration), "semantic_left_dir": None if args.semantic_left_dir is None else str(args.semantic_left_dir), "semantic_right_dir": None if args.semantic_right_dir is None else str(args.semantic_right_dir)}, "contract": {"direct_requires": ["external binary masks on both views", "semantic_identity_evidence=manually_audited", "stereo, plane, coverage, and fisheye reprojection gates"], "fallback": "lower_region_unknown_nonsemantic may exercise geometry but always reports unavailable"}, "parameters": vars(args), "interpretation_boundary": summary["interpretation_boundary"]}
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
