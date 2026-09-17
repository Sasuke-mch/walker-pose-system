#!/usr/bin/env python3
"""Build causal wrist-guided but image-derived stereo handle-axis candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from pose_app.calibration import StereoCalibration  # noqa: E402
from pose_app.rotation import model_to_raw_point  # noqa: E402


WRISTS = {"left": 9, "right": 10}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def prediction_rows(path: Path) -> dict[str, dict]:
    data = read_json(path)
    return {str(row["file_name"]): row for row in data["images"]}


def wrist(row: dict, index: int) -> np.ndarray | None:
    people = row.get("keypoints") or []
    if not people or len(people[0]) <= index:
        return None
    point = np.asarray(people[0][index], dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)) or point[2] < 0.25:
        return None
    return point[:2]


def point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    denominator = float(delta @ delta)
    if denominator <= 1e-9:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip((point - start) @ delta / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * delta)))


def dark_support(gray: np.ndarray, start: np.ndarray, end: np.ndarray, threshold: int) -> float:
    count = max(12, int(np.linalg.norm(end - start)))
    xs = np.rint(np.linspace(start[0], end[0], count)).astype(int)
    ys = np.rint(np.linspace(start[1], end[1], count)).astype(int)
    inside = (xs >= 0) & (xs < gray.shape[1]) & (ys >= 0) & (ys < gray.shape[0])
    if not np.any(inside):
        return 0.0
    return float(np.mean(gray[ys[inside], xs[inside]] <= threshold))


def detect_handle_line(
    image: np.ndarray, wrist_px: np.ndarray, *, roi_radius: int, dark_threshold: int,
    minimum_length_px: float, maximum_wrist_distance_px: float,
) -> tuple[np.ndarray | None, dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x, y = float(wrist_px[0]), float(wrist_px[1])
    x0, x1 = max(0, int(x) - roi_radius), min(gray.shape[1], int(x) + roi_radius + 1)
    y0, y1 = max(0, int(y) - roi_radius), min(gray.shape[0], int(y) + roi_radius + 1)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None, {"status": "unavailable", "reason": "wrist_roi_outside_image"}
    roi = gray[y0:y1, x0:x1]
    edges = cv2.Canny(roi, 35, 105, apertureSize=3)
    raw = cv2.HoughLinesP(
        edges, 1.0, np.pi / 180.0, threshold=18,
        minLineLength=max(12, int(minimum_length_px)), maxLineGap=12,
    )
    if raw is None:
        return None, {"status": "unavailable", "reason": "no_local_line_segment"}
    candidates = []
    for values in raw[:, 0, :]:
        start = np.asarray((values[0] + x0, values[1] + y0), dtype=np.float64)
        end = np.asarray((values[2] + x0, values[3] + y0), dtype=np.float64)
        length = float(np.linalg.norm(end - start))
        if length < minimum_length_px:
            continue
        vertical_fraction = abs(float(end[1] - start[1])) / max(length, 1e-9)
        if vertical_fraction < 0.52:
            continue
        distance = point_segment_distance(wrist_px, start, end)
        if distance > maximum_wrist_distance_px:
            continue
        support = dark_support(gray, start, end, dark_threshold)
        if support < 0.58:
            continue
        score = 1.8 * support + 0.012 * length + 0.6 * vertical_fraction - 0.018 * distance
        candidates.append((score, start, end, length, distance, support, vertical_fraction))
    if not candidates:
        return None, {"status": "unavailable", "reason": "no_dark_vertical_line_near_wrist"}
    score, start, end, length, distance, support, vertical_fraction = max(candidates, key=lambda item: item[0])
    return np.vstack((start, end)), {
        "status": "candidate", "score": score, "length_px": length,
        "wrist_to_line_distance_px": distance, "dark_support_fraction": support,
        "vertical_fraction": vertical_fraction, "roi_bounds_xyxy": [x0, y0, x1, y1],
        "candidate_count": len(candidates),
    }


def sample_segment(segment: np.ndarray, count: int = 25) -> np.ndarray:
    fractions = np.linspace(0.08, 0.92, count)[:, None]
    return segment[0] + fractions * (segment[1] - segment[0])


def upright_to_raw(points: np.ndarray, side: str) -> np.ndarray:
    rotation = "ccw90" if side == "left" else "cw90"
    return np.asarray([
        model_to_raw_point(float(point[0]), float(point[1]), 1920, 1080, rotation)
        for point in points
    ], dtype=np.float64)


def triangulate_pairs(
    left_upright: np.ndarray, right_upright: np.ndarray, calibration: StereoCalibration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_raw = upright_to_raw(left_upright, "left")
    right_raw = upright_to_raw(right_upright, "right")
    left_norm = calibration.undistort_normalized(left_raw, "left")
    right_norm = calibration.undistort_normalized(right_raw, "right")
    p_left = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
    p_right = np.concatenate((calibration.R, calibration.T.reshape(3, 1)), axis=1)
    homogeneous = cv2.triangulatePoints(p_left, p_right, left_norm.T, right_norm.T)
    xyz = (homogeneous[:3] / homogeneous[3]).T
    right_xyz = (calibration.R @ xyz.T + calibration.T.reshape(3, 1)).T
    left_error = np.linalg.norm(calibration.project_left(xyz) - left_raw, axis=1)
    right_error = np.linalg.norm(calibration.project_right(xyz) - right_raw, axis=1)
    mean_error = 0.5 * (left_error + right_error)
    positive = np.all(np.isfinite(xyz), axis=1) & (xyz[:, 2] > 0) & (right_xyz[:, 2] > 0)
    return xyz, mean_error, positive


def reconstruct_axis(
    left_segment: np.ndarray, right_segment: np.ndarray, calibration: StereoCalibration,
    maximum_reprojection_error_px: float,
) -> dict:
    left_samples = sample_segment(left_segment)
    right_samples = sample_segment(right_segment)
    tiled_left = np.repeat(left_samples, len(right_samples), axis=0)
    tiled_right = np.tile(right_samples, (len(left_samples), 1))
    xyz, errors, positive = triangulate_pairs(tiled_left, tiled_right, calibration)
    error_matrix = np.where(positive, errors, np.inf).reshape(len(left_samples), len(right_samples))
    chosen_right = np.argmin(error_matrix, axis=1)
    chosen_error = error_matrix[np.arange(len(left_samples)), chosen_right]
    chosen_xyz = xyz.reshape(len(left_samples), len(right_samples), 3)[np.arange(len(left_samples)), chosen_right]
    valid = np.isfinite(chosen_error) & (chosen_error <= maximum_reprojection_error_px)
    if int(valid.sum()) < 6:
        return {
            "status": "unavailable", "reason": "insufficient_epipolar_line_support",
            "matched_sample_count": int(valid.sum()),
        }
    points = chosen_xyz[valid]
    errors_kept = chosen_error[valid]
    center = np.median(points, axis=0)
    centered = points - center
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    projection = centered @ direction
    low, high = np.percentile(projection, [10.0, 90.0])
    start, end = center + low * direction, center + high * direction
    length = float(np.linalg.norm(end - start))
    residual = np.linalg.norm(centered - projection[:, None] * direction, axis=1)
    if not 20.0 <= length <= 350.0:
        return {
            "status": "rejected", "reason": "implausible_handle_axis_length",
            "length_mm": length, "matched_sample_count": int(valid.sum()),
        }
    return {
        "status": "candidate", "source": "bilateral_dark_line_stereo_geometry",
        "start_left_camera_mm": start.tolist(), "end_left_camera_mm": end.tolist(),
        "length_mm": length, "matched_sample_count": int(valid.sum()),
        "reprojection_error_median_px": float(np.median(errors_kept)),
        "reprojection_error_p95_px": float(np.percentile(errors_kept, 95)),
        "line_residual_median_mm": float(np.median(residual)),
        "principal_variance_fraction": float(singular[0] ** 2 / max(np.sum(singular ** 2), 1e-12)),
        "causal": True, "uses_future_frames": False,
        "identity": "wrist_guided_image_handle_candidate",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-dir", type=Path, required=True)
    parser.add_argument("--right-dir", type=Path, required=True)
    parser.add_argument("--left-predictions", type=Path, required=True)
    parser.add_argument("--right-predictions", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roi-radius", type=int, default=115)
    parser.add_argument("--dark-threshold", type=int, default=82)
    parser.add_argument("--minimum-line-length-px", type=float, default=28.0)
    parser.add_argument("--maximum-wrist-distance-px", type=float, default=52.0)
    parser.add_argument("--maximum-reprojection-error-px", type=float, default=8.0)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    left_rows = prediction_rows(args.left_predictions)
    right_rows = prediction_rows(args.right_predictions)
    calibration = StereoCalibration.load(args.calibration)
    images = sorted(args.left_dir.glob("pair_*.png"))
    statuses = {side: Counter() for side in WRISTS}
    records = []
    preview_ids = {0, 75, 135, 205, 278, 360, 435, 447}
    preview_dir = output / "previews"
    preview_dir.mkdir()
    for pair_id, left_path in enumerate(images):
        name = left_path.name
        right_path = args.right_dir / name
        left_image = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right_image = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        record = {"pair_id": pair_id, "file_name": name, "handles": {}}
        annotated = [left_image.copy(), right_image.copy()]
        for body_side, index in WRISTS.items():
            left_wrist = wrist(left_rows[name], index)
            right_wrist = wrist(right_rows[name], index)
            if left_wrist is None or right_wrist is None:
                result = {"status": "unavailable", "reason": "bilateral_wrist_roi_missing"}
            else:
                left_line, left_audit = detect_handle_line(
                    left_image, left_wrist, roi_radius=args.roi_radius, dark_threshold=args.dark_threshold,
                    minimum_length_px=args.minimum_line_length_px,
                    maximum_wrist_distance_px=args.maximum_wrist_distance_px,
                )
                right_line, right_audit = detect_handle_line(
                    right_image, right_wrist, roi_radius=args.roi_radius, dark_threshold=args.dark_threshold,
                    minimum_length_px=args.minimum_line_length_px,
                    maximum_wrist_distance_px=args.maximum_wrist_distance_px,
                )
                if left_line is None or right_line is None:
                    result = {
                        "status": "unavailable", "reason": "bilateral_handle_line_required",
                        "left_image_audit": left_audit, "right_image_audit": right_audit,
                    }
                else:
                    result = reconstruct_axis(left_line, right_line, calibration, args.maximum_reprojection_error_px)
                    result.update({
                        "left_upright_line_px": left_line.tolist(), "right_upright_line_px": right_line.tolist(),
                        "left_image_audit": left_audit, "right_image_audit": right_audit,
                    })
                    for canvas, line, wrist_px in ((annotated[0], left_line, left_wrist), (annotated[1], right_line, right_wrist)):
                        color = (255, 180, 0) if body_side == "left" else (180, 0, 255)
                        cv2.line(canvas, tuple(np.rint(line[0]).astype(int)), tuple(np.rint(line[1]).astype(int)), color, 4)
                        cv2.circle(canvas, tuple(np.rint(wrist_px).astype(int)), 7, color, -1)
            statuses[body_side][result["status"]] += 1
            record["handles"][body_side] = result
        records.append(record)
        if pair_id in preview_ids:
            joined = np.hstack([cv2.resize(image, (540, 960)) for image in annotated])
            cv2.imwrite(str(preview_dir / f"pair_{pair_id:04d}.jpg"), joined)
    records_path = output / "stereo_handle_axes.jsonl"
    records_path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records), encoding="utf-8")
    summary = {
        "schema_version": "wrist_guided_image_derived_stereo_handle_axes_v1",
        "frame_count": len(records), "status_counts": {side: dict(counts) for side, counts in statuses.items()},
        "causal": True, "uses_future_frames": False,
        "geometry_source": "bilateral current-frame dark line pixels near each wrist, calibrated fisheye triangulation",
        "wrist_role": "ROI selection only; handle pixels and 3-D line are image-derived",
        "records": str(records_path.resolve()),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
