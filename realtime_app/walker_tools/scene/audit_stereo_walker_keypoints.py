#!/usr/bin/env python3
"""Audit paired LabelMe walker landmarks in calibrated fisheye stereo.

This utility measures geometric compatibility only.  A low residual says that
the two marked pixels can be explained by one 3-D point under the supplied
calibration; it does not prove that they identify the same physical feature.
"""

from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
_tool_prepare_imports()

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np


TOOLS_ROOT = (Path(__file__).resolve().parents[2] / "tools")
REALTIME_ROOT = TOOLS_ROOT.parent
if str(REALTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(REALTIME_ROOT))

from pose_app.calibration import StereoCalibration  # noqa: E402
from pose_app.rotation import model_to_raw_point  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-label", type=Path, required=True)
    parser.add_argument("--right-label", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-annotation-reprojection-px", type=float, default=10.0)
    return parser.parse_args()


def load_points(path: Path) -> tuple[dict[str, np.ndarray], tuple[int, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    width, height = int(data["imageWidth"]), int(data["imageHeight"])
    result: dict[str, np.ndarray] = {}
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "point":
            continue
        label = str(shape.get("label", "")).strip()
        points = shape.get("points", [])
        if not label or len(points) != 1:
            continue
        point = np.asarray(points[0], dtype=np.float64)
        if point.shape == (2,) and np.all(np.isfinite(point)):
            result[label] = point
    return result, (width, height)


def ray_closest_gap(
    left_ray: np.ndarray, right_ray_left: np.ndarray, right_center_left: np.ndarray
) -> tuple[float, float, float]:
    system = np.column_stack((left_ray, -right_ray_left))
    scales = np.linalg.lstsq(system, right_center_left, rcond=None)[0]
    left_point = scales[0] * left_ray
    right_point = right_center_left + scales[1] * right_ray_left
    return float(np.linalg.norm(left_point - right_point)), float(scales[0]), float(scales[1])


def audit_pair(
    label: str,
    left_upright: np.ndarray,
    right_upright: np.ndarray,
    calibration: StereoCalibration,
    maximum_error_px: float,
) -> dict:
    left_raw = np.asarray(
        model_to_raw_point(float(left_upright[0]), float(left_upright[1]), 1920, 1080, "ccw90"),
        dtype=np.float64,
    )
    right_raw = np.asarray(
        model_to_raw_point(float(right_upright[0]), float(right_upright[1]), 1920, 1080, "cw90"),
        dtype=np.float64,
    )
    left_normalized = calibration.undistort_normalized(left_raw[None, :], "left")[0]
    right_normalized = calibration.undistort_normalized(right_raw[None, :], "right")[0]
    left_ray = np.r_[left_normalized, 1.0]
    left_ray /= np.linalg.norm(left_ray)
    right_ray = np.r_[right_normalized, 1.0]
    right_ray /= np.linalg.norm(right_ray)
    right_ray_left = calibration.R.T @ right_ray
    right_center_left = -calibration.R.T @ calibration.T
    ray_gap_mm, left_scale, right_scale = ray_closest_gap(
        left_ray, right_ray_left, right_center_left
    )
    p_left = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
    p_right = np.concatenate((calibration.R, calibration.T.reshape(3, 1)), axis=1)
    homogeneous = cv2.triangulatePoints(
        p_left, p_right, left_normalized.reshape(2, 1), right_normalized.reshape(2, 1)
    )
    xyz = homogeneous[:3, 0] / homogeneous[3, 0]
    projected_left = calibration.project_left(xyz[None, :])[0]
    projected_right = calibration.project_right(xyz[None, :])[0]
    error_left = float(np.linalg.norm(projected_left - left_raw))
    error_right = float(np.linalg.norm(projected_right - right_raw))
    mean_error = 0.5 * (error_left + error_right)
    depth_right = float((calibration.R @ xyz + calibration.T)[2])
    positive = bool(np.all(np.isfinite(xyz)) and xyz[2] > 0 and depth_right > 0)
    if not positive:
        state = "negative_or_nonfinite_geometry"
    elif mean_error <= maximum_error_px:
        state = "within_declared_annotation_tolerance"
    else:
        state = "geometry_conflict_beyond_declared_tolerance"
    return {
        "label": label,
        "state": state,
        "left_upright_px": left_upright.tolist(),
        "right_upright_px": right_upright.tolist(),
        "left_raw_fisheye_px": left_raw.tolist(),
        "right_raw_fisheye_px": right_raw.tolist(),
        "xyz_left_camera_mm": xyz.tolist(),
        "depth_left_mm": float(xyz[2]),
        "depth_right_mm": depth_right,
        "ray_gap_mm": ray_gap_mm,
        "left_ray_distance_mm": left_scale,
        "right_ray_distance_mm": right_scale,
        "reprojection_error_left_px": error_left,
        "reprojection_error_right_px": error_right,
        "mean_reprojection_error_px": mean_error,
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    if not math.isfinite(args.maximum_annotation_reprojection_px) or args.maximum_annotation_reprojection_px <= 0:
        raise ValueError("maximum annotation reprojection tolerance must be positive")
    left, left_size = load_points(args.left_label)
    right, right_size = load_points(args.right_label)
    if left_size != (1080, 1920) or right_size != (1080, 1920):
        raise ValueError("this audit expects 1080x1920 upright labels derived from 1920x1080 raw images")
    calibration = StereoCalibration.load(args.calibration)
    common = sorted(set(left) & set(right))
    rows = [audit_pair(name, left[name], right[name], calibration, args.maximum_annotation_reprojection_px) for name in common]
    states: dict[str, int] = {}
    for row in rows:
        states[row["state"]] = states.get(row["state"], 0) + 1
    result = {
        "schema_version": "stereo_walker_keypoint_audit_v1",
        "inputs": {
            "left_label": str(args.left_label.resolve()),
            "right_label": str(args.right_label.resolve()),
            "calibration": str(args.calibration.resolve()),
        },
        "paired_label_count": len(common),
        "left_only_labels": sorted(set(left) - set(right)),
        "right_only_labels": sorted(set(right) - set(left)),
        "maximum_annotation_reprojection_px": args.maximum_annotation_reprojection_px,
        "state_counts": states,
        "pairs": rows,
        "interpretation_boundary": (
            "The tolerance accounts for manual pixel placement. Low residual means calibrated geometric "
            "compatibility only; it neither proves physical landmark identity nor measures calibration accuracy."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("paired_label_count", "state_counts", "maximum_annotation_reprojection_px")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
