#!/usr/bin/env python3
"""Validate a floor-directed sparse-mask stereo plane on raw fisheye data.

This is a controlled geometry experiment derived from the dynamic-ground step
in the system specification.  It does *not* send virtual-camera images to a
pose model.  The virtual view is used only to establish local stereo
correspondence.  Every reconstructed point is returned to the calibrated raw
left-camera coordinate frame before plane fitting.

The initial global rectification experiment revealed that its common field of
view excluded the visible floor.  This tool instead directs a local stereo
view at visually verified floor rays.  Its floor candidate is deliberately
conservative: lower-image, low-saturation/medium-value pixels with the
Sapiens2 person skeleton excluded.  Therefore it is a validation input for
this indoor sequence, not an automatic general-purpose floor segmenter.
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


SAPIENS_TO_COCO17 = (0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14)
COCO_EDGES = (
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)


@dataclass(frozen=True)
class LocalRectification:
    left_map_x: np.ndarray
    left_map_y: np.ndarray
    right_map_x: np.ndarray
    right_map_y: np.ndarray
    rotation_left: np.ndarray
    rotation_right: np.ndarray
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
    parser.add_argument("--left-dir", type=Path, required=True, help="upright left_ccw90 frames")
    parser.add_argument("--right-dir", type=Path, required=True, help="upright right_cw90 frames")
    parser.add_argument("--sapiens-left-json", type=Path, required=True)
    parser.add_argument("--sapiens-right-json", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-step", type=int, default=6)
    parser.add_argument("--runtime-width", type=int, default=960)
    parser.add_argument("--runtime-height", type=int, default=540)
    parser.add_argument("--virtual-focal-px", type=float, default=330.0)
    parser.add_argument("--num-disparities", type=int, default=160)
    parser.add_argument("--ransac-distance-mm", type=float, default=25.0)
    parser.add_argument("--ransac-iterations", type=int, default=900)
    parser.add_argument("--lr-consistency-px", type=float, default=1.5)
    parser.add_argument("--minimum-candidates", type=int, default=250)
    parser.add_argument("--minimum-inliers", type=int, default=150)
    return parser.parse_args()


def read_pairs(left_dir: Path, right_dir: Path, step: int) -> list[tuple[str, Path, Path]]:
    if step < 1:
        raise ValueError("frame-step must be at least 1")
    left = {path.name: path for path in left_dir.glob("pair_*.png")}
    right = {path.name: path for path in right_dir.glob("pair_*.png")}
    if set(left) != set(right) or not left:
        raise RuntimeError("left/right input directories must contain the same non-empty pair_*.png set")
    return [(name, left[name], right[name]) for name in sorted(left)[::step]]


def load_sapiens_by_name(path: Path) -> dict[str, dict]:
    root = json.loads(path.read_text(encoding="utf-8-sig"))
    records: dict[str, dict] = {}
    for item in root.get("images", []):
        name = str(item.get("file_name", ""))
        instances = item.get("instances", [])
        if name and instances:
            records[name] = instances[0]
    if not records:
        raise RuntimeError(f"no usable Sapiens2 records in {path}")
    return records


def inverse_upright(left_upright: np.ndarray, right_upright: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        cv2.rotate(left_upright, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(right_upright, cv2.ROTATE_90_COUNTERCLOCKWISE),
    )


def rotate_mask_to_raw(mask: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    if side == "right":
        return cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("side must be left or right")


def upright_point_to_raw(point: np.ndarray, side: str, upright_width: int, upright_height: int) -> np.ndarray:
    x, y = float(point[0]), float(point[1])
    if side == "left":
        return np.asarray([upright_height - 1.0 - y, x], dtype=np.float64)
    if side == "right":
        return np.asarray([y, upright_width - 1.0 - x], dtype=np.float64)
    raise ValueError("side must be left or right")


def ray_from_raw_pixel(calibration: StereoCalibration, pixel: np.ndarray, side: str) -> np.ndarray:
    normalized = calibration.undistort_normalized(pixel.reshape(1, 2), side)[0]
    ray = np.asarray([normalized[0], normalized[1], 1.0], dtype=np.float64)
    return ray / np.linalg.norm(ray)


def make_floor_directed_rectification(
    calibration: StereoCalibration,
    upright_size: tuple[int, int],
    runtime_size: tuple[int, int],
    focal_px: float,
) -> LocalRectification:
    """Build a stereo pair aimed at the floor seen near the bottom centre.

    The normalized seed (0.46, 0.97) was visually checked on the present near
    sequence.  It represents only the local correspondence view, while the
    actual plane is always fitted after conversion to the raw camera frame.
    """
    upright_width, upright_height = upright_size
    runtime_width, runtime_height = runtime_size
    if focal_px <= 0:
        raise ValueError("virtual focal length must be positive")
    seed_upright = np.asarray([0.46 * upright_width, 0.97 * upright_height], dtype=np.float64)
    raw_left = upright_point_to_raw(seed_upright, "left", upright_width, upright_height)
    raw_right = upright_point_to_raw(seed_upright, "right", upright_width, upright_height)
    raw_left *= np.asarray([runtime_width / upright_height, runtime_height / upright_width])
    raw_right *= np.asarray([runtime_width / upright_height, runtime_height / upright_width])

    left_ray = ray_from_raw_pixel(calibration, raw_left, "left")
    right_ray_in_left = calibration.R.T @ ray_from_raw_pixel(calibration, raw_right, "right")
    baseline_left = -calibration.R.T @ calibration.T
    baseline_left /= np.linalg.norm(baseline_left)
    forward = left_ray + right_ray_in_left
    forward -= baseline_left * float(forward @ baseline_left)
    forward /= np.linalg.norm(forward)
    vertical = np.cross(forward, baseline_left)
    vertical /= np.linalg.norm(vertical)
    rotation_left = np.stack((baseline_left, vertical, forward))
    rotation_right = rotation_left @ calibration.R.T
    translation_right = rotation_right @ calibration.T
    if abs(float(translation_right[1])) > 1e-6 or abs(float(translation_right[2])) > 1e-6:
        raise RuntimeError("floor-directed rectification did not align the stereo baseline")

    virtual_K = np.asarray(
        [[focal_px, 0.0, runtime_width / 2.0], [0.0, focal_px, runtime_height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    left_map_x, left_map_y = cv2.fisheye.initUndistortRectifyMap(
        calibration.left_K, calibration.left_D.reshape(-1, 1), rotation_left, virtual_K,
        runtime_size, cv2.CV_32FC1,
    )
    right_map_x, right_map_y = cv2.fisheye.initUndistortRectifyMap(
        calibration.right_K, calibration.right_D.reshape(-1, 1), rotation_right, virtual_K,
        runtime_size, cv2.CV_32FC1,
    )
    return LocalRectification(
        left_map_x, left_map_y, right_map_x, right_map_y,
        rotation_left, rotation_right, virtual_K, translation_right,
    )


def sapiens_body_exclusion_mask(image_shape: tuple[int, int], record: dict) -> np.ndarray:
    height, width = image_shape
    coordinates = np.asarray(record.get("keypoints308", []), dtype=np.float64)
    scores = np.asarray(record.get("keypoint_scores", []), dtype=np.float64)
    if coordinates.shape != (308, 2) or scores.shape != (308,):
        raise ValueError("Sapiens2 record must contain 308 xy points and 308 scores")
    points = coordinates[list(SAPIENS_TO_COCO17)]
    valid = (scores[list(SAPIENS_TO_COCO17)] >= 0.25) & np.all(np.isfinite(points), axis=1)
    mask = np.zeros((height, width), dtype=np.uint8)
    for first, second in COCO_EDGES:
        if valid[first] and valid[second]:
            cv2.line(mask, tuple(np.rint(points[first]).astype(int)), tuple(np.rint(points[second]).astype(int)), 255, 160)
    for point, is_valid in zip(points, valid):
        if is_valid:
            cv2.circle(mask, tuple(np.rint(point).astype(int)), 110, 255, -1)
    torso_indices = np.asarray((5, 6, 11, 12), dtype=np.int32)
    torso = points[torso_indices]
    torso_valid = valid[torso_indices]
    if int(torso_valid.sum()) >= 3:
        cv2.fillConvexPoly(mask, np.rint(torso[torso_valid]).astype(np.int32), 255)
    return mask


def floor_candidate_mask(upright: np.ndarray, sapiens_record: dict) -> np.ndarray:
    height, width = upright.shape[:2]
    hsv = cv2.cvtColor(upright, cv2.COLOR_BGR2HSV)
    # The current test floor is neutral gray.  Keep this explicitly local to
    # the validation sequence; a deployed system requires a floor segmenter.
    color = cv2.inRange(hsv, (0, 0, 30), (180, 70, 205)) > 0
    lower = np.zeros((height, width), dtype=bool)
    lower[int(round(0.58 * height)):] = True
    body = sapiens_body_exclusion_mask((height, width), sapiens_record) > 0
    return (color & lower & ~body).astype(np.uint8) * 255


def dense_disparity(reference: np.ndarray, partner: np.ndarray, num_disparities: int) -> np.ndarray:
    if num_disparities < 16 or num_disparities % 16:
        raise ValueError("num-disparities must be a positive multiple of 16")
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    partner_gray = cv2.cvtColor(partner, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=7,
        P1=8 * 3 * 7 * 7,
        P2=32 * 3 * 7 * 7,
        disp12MaxDiff=1,
        uniquenessRatio=12,
        speckleWindowSize=60,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return matcher.compute(clahe.apply(ref_gray), clahe.apply(partner_gray)).astype(np.float32) / 16.0


def reconstruct_floor_candidates(
    left: np.ndarray,
    right: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    rectification: LocalRectification,
    num_disparities: int,
    lr_consistency_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = left.shape[:2]
    tx = float(rectification.translation_right[0])
    if abs(tx) < 1e-9:
        raise RuntimeError("zero horizontal local stereo baseline")
    # P2 has negative x translation in the usual left-reference convention.
    # For an uncommon opposite sign, reverse reference direction explicitly.
    reference_is_left = tx < 0.0
    if reference_is_left:
        disparity = dense_disparity(left, right, num_disparities)
        reverse_disparity = dense_disparity(right, left, num_disparities)
        yy, xx_left = np.mgrid[0:height, 0:width]
        xx_right = np.rint(xx_left.astype(np.float32) - disparity).astype(np.int32)
        reverse_at_match = reverse_disparity[yy, np.clip(xx_right, 0, width - 1)]
        valid = (
            (disparity > 1.0) & (xx_right >= 0) & (xx_right < width)
            & (left_mask > 0) & (right_mask[yy, np.clip(xx_right, 0, width - 1)] > 0)
            & (reverse_at_match > 1.0) & (np.abs(disparity - reverse_at_match) <= lr_consistency_px)
        )
        xs = xx_left[valid]
        ys = yy[valid]
        partner_x = xx_right[valid]
    else:
        disparity = dense_disparity(right, left, num_disparities)
        reverse_disparity = dense_disparity(left, right, num_disparities)
        yy, xx_right = np.mgrid[0:height, 0:width]
        xx_left = np.rint(xx_right.astype(np.float32) - disparity).astype(np.int32)
        reverse_at_match = reverse_disparity[yy, np.clip(xx_left, 0, width - 1)]
        valid = (
            (disparity > 1.0) & (xx_left >= 0) & (xx_left < width)
            & (right_mask > 0) & (left_mask[yy, np.clip(xx_left, 0, width - 1)] > 0)
            & (reverse_at_match > 1.0) & (np.abs(disparity - reverse_at_match) <= lr_consistency_px)
        )
        xs = xx_left[valid]
        ys = yy[valid]
        partner_x = xx_right[valid]
    if len(xs) == 0:
        return np.empty((0, 3)), np.empty((0, 2), dtype=np.int32), np.empty((0, 2), dtype=np.int32), disparity

    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    difference = np.abs(left_gray[ys, xs].astype(np.int16) - right_gray[ys, partner_x].astype(np.int16))
    keep = difference <= 45
    xs, ys, partner_x = xs[keep], ys[keep], partner_x[keep]
    disparity_values = disparity[valid][keep]
    depth = abs(float(rectification.virtual_K[0, 0] * tx)) / disparity_values
    x = (xs.astype(np.float64) - rectification.virtual_K[0, 2]) * depth / rectification.virtual_K[0, 0]
    y = (ys.astype(np.float64) - rectification.virtual_K[1, 2]) * depth / rectification.virtual_K[1, 1]
    points_rectified = np.column_stack((x, y, depth))
    points_left = (rectification.rotation_left.T @ points_rectified.T).T
    finite = np.all(np.isfinite(points_left), axis=1) & (depth > 250.0) & (depth < 8000.0)
    return (
        points_left[finite],
        np.column_stack((xs[finite], ys[finite])).astype(np.int32),
        np.column_stack((partner_x[finite], ys[finite])).astype(np.int32),
        disparity,
    )


def fit_plane_ransac(points: np.ndarray, distance_mm: float, iterations: int, seed: int) -> PlaneFit | None:
    if len(points) < 3:
        return None
    rng = np.random.default_rng(seed)
    best: np.ndarray | None = None
    best_count = 0
    for _ in range(iterations):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal /= norm
        offset = -float(normal @ sample[0])
        inliers = np.abs(points @ normal + offset) <= distance_mm
        if int(inliers.sum()) > best_count:
            best, best_count = inliers, int(inliers.sum())
    if best is None:
        return None
    centre = np.mean(points[best], axis=0)
    _, _, vectors = np.linalg.svd(points[best] - centre, full_matrices=False)
    normal = vectors[-1]
    normal /= np.linalg.norm(normal)
    offset = -float(normal @ centre)
    inliers = np.abs(points @ normal + offset) <= distance_mm
    return PlaneFit(normal, offset, inliers, float(np.median(np.abs(points[inliers] @ normal + offset))))


def raw_pixels_from_local(local_pixels: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    if len(local_pixels) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.column_stack((map_x[local_pixels[:, 1], local_pixels[:, 0]], map_y[local_pixels[:, 1], local_pixels[:, 0]]))


def raw_to_upright(raw_pixels: np.ndarray, side: str, raw_size: tuple[int, int]) -> np.ndarray:
    raw_width, raw_height = raw_size
    if side == "left":
        return np.column_stack((raw_pixels[:, 1], raw_width - 1.0 - raw_pixels[:, 0]))
    if side == "right":
        return np.column_stack((raw_height - 1.0 - raw_pixels[:, 1], raw_pixels[:, 0]))
    raise ValueError("side must be left or right")


def draw_points(image: np.ndarray, points: np.ndarray, color: tuple[int, int, int], maximum: int = 3500) -> np.ndarray:
    output = image.copy()
    if len(points):
        subset = points[::max(1, len(points) // maximum)]
        valid = np.all(np.isfinite(subset), axis=1)
        subset = np.rint(subset[valid]).astype(np.int32)
        subset[:, 0] = np.clip(subset[:, 0], 0, image.shape[1] - 1)
        subset[:, 1] = np.clip(subset[:, 1], 0, image.shape[0] - 1)
        output[subset[:, 1], subset[:, 0]] = color
    return output


def colorize_disparity(disparity: np.ndarray) -> np.ndarray:
    valid = disparity > 1.0
    mono = np.zeros(disparity.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(disparity[valid], (2, 98))
        mono[valid] = np.rint(255.0 * np.clip((disparity[valid] - low) / max(high - low, 1e-6), 0.0, 1.0)).astype(np.uint8)
    return cv2.applyColorMap(mono, cv2.COLORMAP_TURBO)


def write_visualization(
    path: Path,
    left_upright: np.ndarray,
    left_local: np.ndarray,
    right_local: np.ndarray,
    disparity: np.ndarray,
    left_local_pixels: np.ndarray,
    inlier_mask: np.ndarray | None,
    rectification: LocalRectification,
    title: str,
) -> None:
    raw_pixels = raw_pixels_from_local(left_local_pixels, rectification.left_map_x, rectification.left_map_y)
    raw_pixels *= np.asarray(
        [left_upright.shape[0] / left_local.shape[1], left_upright.shape[1] / left_local.shape[0]],
        dtype=np.float64,
    )
    upright_pixels = raw_to_upright(raw_pixels, "left", (left_upright.shape[0], left_upright.shape[1]))
    raw_panel = draw_points(left_upright, upright_pixels, (0, 170, 255))
    local_panel = draw_points(left_local, left_local_pixels, (0, 170, 255))
    if inlier_mask is not None:
        raw_panel = draw_points(raw_panel, upright_pixels[inlier_mask], (0, 255, 0))
        local_panel = draw_points(local_panel, left_local_pixels[inlier_mask], (0, 255, 0))
    target_size = (960, 540)
    raw_panel = cv2.resize(raw_panel, target_size, interpolation=cv2.INTER_AREA)
    panels = [raw_panel, local_panel, right_local, colorize_disparity(disparity)]
    labels = [
        "upright raw left (orange: floor candidates; green: plane inliers)",
        "floor-directed rectified left",
        "floor-directed rectified right",
        "local stereo disparity",
    ]
    decorated = []
    for panel, label in zip(panels, labels):
        view = panel.copy()
        cv2.rectangle(view, (0, 0), (min(740, view.shape[1]), 30), (255, 255, 255), -1)
        cv2.putText(view, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        decorated.append(view)
    sheet = np.vstack((np.hstack((decorated[0], decorated[1])), np.hstack((decorated[2], decorated[3]))))
    cv2.rectangle(sheet, (0, 0), (min(1100, sheet.shape[1]), 34), (255, 255, 255), -1)
    cv2.putText(sheet, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"cannot write {path}")


def write_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        for x, y, z in points:
            handle.write(f"{x:.5f} {y:.5f} {z:.5f}\n")


def angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(float(first @ second)), -1.0, 1.0))))


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
    sample_upright = cv2.imread(str(pairs[0][1]), cv2.IMREAD_COLOR)
    if sample_upright is None:
        raise RuntimeError(f"cannot read {pairs[0][1]}")
    upright_size = (sample_upright.shape[1], sample_upright.shape[0])
    rectification = make_floor_directed_rectification(calibration, upright_size, runtime_size, args.virtual_focal_px)
    args.output_dir.mkdir(parents=True)
    visual_indices = set(np.linspace(0, len(pairs) - 1, min(6, len(pairs)), dtype=int).tolist())
    rows: list[dict[str, object]] = []
    normals: list[np.ndarray] = []

    for index, (name, left_path, right_path) in enumerate(pairs):
        if name not in left_sapiens or name not in right_sapiens:
            raise RuntimeError(f"missing Sapiens2 record for {name}")
        left_upright = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right_upright = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left_upright is None or right_upright is None:
            raise RuntimeError(f"cannot read {name}")
        left_raw, right_raw = inverse_upright(left_upright, right_upright)
        left_raw = cv2.resize(left_raw, runtime_size, interpolation=cv2.INTER_AREA)
        right_raw = cv2.resize(right_raw, runtime_size, interpolation=cv2.INTER_AREA)
        left_mask_raw = cv2.resize(
            rotate_mask_to_raw(floor_candidate_mask(left_upright, left_sapiens[name]), "left"), runtime_size,
            interpolation=cv2.INTER_NEAREST,
        )
        right_mask_raw = cv2.resize(
            rotate_mask_to_raw(floor_candidate_mask(right_upright, right_sapiens[name]), "right"), runtime_size,
            interpolation=cv2.INTER_NEAREST,
        )
        left_local = cv2.remap(left_raw, rectification.left_map_x, rectification.left_map_y, cv2.INTER_LINEAR)
        right_local = cv2.remap(right_raw, rectification.right_map_x, rectification.right_map_y, cv2.INTER_LINEAR)
        left_mask = cv2.remap(left_mask_raw, rectification.left_map_x, rectification.left_map_y, cv2.INTER_NEAREST)
        right_mask = cv2.remap(right_mask_raw, rectification.right_map_x, rectification.right_map_y, cv2.INTER_NEAREST)
        points, left_pixels, _, disparity = reconstruct_floor_candidates(
            left_local, right_local, left_mask, right_mask, rectification, args.num_disparities,
            args.lr_consistency_px,
        )
        if len(points) > 12000:
            selected = np.linspace(0, len(points) - 1, 12000, dtype=np.int64)
            fit = fit_plane_ransac(points[selected], args.ransac_distance_mm, args.ransac_iterations, 20260901 + index)
        else:
            fit = fit_plane_ransac(points, args.ransac_distance_mm, args.ransac_iterations, 20260901 + index)
        full_inliers: np.ndarray | None = None
        if fit is not None:
            full_inliers = np.abs(points @ fit.normal + fit.offset) <= args.ransac_distance_mm
        accepted = (
            fit is not None
            and len(points) >= args.minimum_candidates
            and full_inliers is not None
            and int(full_inliers.sum()) >= args.minimum_inliers
        )
        if accepted:
            normals.append(fit.normal)
        inliers = points[full_inliers] if full_inliers is not None else np.empty((0, 3))
        rows.append({
            "sequence_index": index,
            "file_name": name,
            "local_left_floor_mask_fraction": float((left_mask > 0).mean()),
            "local_right_floor_mask_fraction": float((right_mask > 0).mean()),
            "candidate_points": int(len(points)),
            "ransac_inliers": int(len(inliers)),
            "ransac_inlier_fraction": float(len(inliers) / len(points)) if len(points) else None,
            "accepted_floor_plane_candidate": bool(accepted),
            "normal_x": float(fit.normal[0]) if fit is not None else None,
            "normal_y": float(fit.normal[1]) if fit is not None else None,
            "normal_z": float(fit.normal[2]) if fit is not None else None,
            "offset_mm": float(fit.offset) if fit is not None else None,
            "median_inlier_distance_mm": float(fit.median_distance_mm) if fit is not None else None,
        })
        if index in visual_indices:
            write_visualization(
                args.output_dir / "visualizations" / f"{Path(name).stem}_floor_directed_candidate.png",
                left_upright, left_local, right_local, disparity, left_pixels, full_inliers, rectification,
                f"{name}: floor-directed candidate plane",
            )
            if len(inliers):
                write_ply(args.output_dir / "pointclouds" / f"{Path(name).stem}_floor_plane_inliers_left_camera.ply", inliers)

    reference = np.median(np.asarray(normals), axis=0) if normals else None
    if reference is not None:
        reference /= np.linalg.norm(reference)
    for row in rows:
        if reference is not None and row["accepted_floor_plane_candidate"]:
            normal = np.asarray([row["normal_x"], row["normal_y"], row["normal_z"]], dtype=np.float64)
            row["normal_angle_to_sequence_median_deg"] = angle_degrees(normal, reference)
        else:
            row["normal_angle_to_sequence_median_deg"] = None
    with (args.output_dir / "per_frame_floor_plane.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    accepted = [row for row in rows if row["accepted_floor_plane_candidate"]]
    metadata = {
        "experiment": "floor-directed local rectification plus lower-region color/person-exclusion candidate and RANSAC",
        "input": "upright pose-model images inverse-rotated to raw fisheye before local stereo correspondence",
        "coordinate_frame": "left_camera",
        "length_unit": calibration.length_unit,
        "pose_model": "not run; existing Sapiens2 keypoints used only to exclude the observed person from floor candidates",
        "local_rectification": "used only for stereo matching; fitted points are transformed back to raw left-camera coordinates",
        "floor_candidate_status": "sequence-specific validation input, not a general semantic floor segmentation result",
        "frame_count": len(rows),
        "accepted_plane_count": len(accepted),
        "median_normal": None if reference is None else reference.tolist(),
        "median_normal_deviation_deg": None if not accepted else float(np.median([row["normal_angle_to_sequence_median_deg"] for row in accepted])),
        "parameters": {
            "frame_step": args.frame_step,
            "runtime_size": runtime_size,
            "virtual_focal_px": args.virtual_focal_px,
            "floor_ray_seed_upright_normalized": [0.46, 0.97],
            "hsv_floor_candidate": {"s_max": 70, "v_min": 30, "v_max": 205, "lower_fraction": 0.58},
            "ransac_distance_mm": args.ransac_distance_mm,
            "ransac_iterations": args.ransac_iterations,
            "left_right_disparity_consistency_px": args.lr_consistency_px,
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "# 地面朝向局部校正与 RANSAC 平面验证\n\n"
        "本实验按系统说明书的地面候选、双目对应、三角测量和鲁棒平面拟合顺序执行。"
        "局部校正仅用于建立地面附近的左右对应，未作为姿态模型输入；三维点和 RANSAC 平面均回到原始左相机坐标。\n\n"
        "候选掩膜由本序列的低饱和中亮度下部区域和 Sapiens2 人体骨架排除构成，必须逐张检查绿色平面内点是否仍局限在真实地面。"
        "只有该检查通过，后续才能将平面跟踪用于脚部高度、支撑相和摆动相。\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output_dir), "frames": len(rows), "accepted": len(accepted),
        "median_normal_deviation_deg": metadata["median_normal_deviation_deg"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
