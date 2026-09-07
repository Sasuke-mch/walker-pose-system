#!/usr/bin/env python3
"""Render an ideal upright skeleton and exact stereo ray intersections.

The stereo rig keeps the calibrated relative R/T.  The body is deliberately a
schematic COCO torso/limb model, so this tool explains central-ray geometry
without presenting an unverified reconstructed person as ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_app.calibration import StereoCalibration


EDGES = (
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--joints",
        nargs="+",
        default=("right_hip", "right_knee", "right_ankle"),
    )
    return parser.parse_args()


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero vector")
    return vector / norm


def build_display_frame(calibration: StereoCalibration) -> tuple[np.ndarray, np.ndarray]:
    """Return raw-left to display rotation and a display translation.

    Display X follows the physical baseline, display Y follows the mean optical
    direction after removing its baseline component, and display Z completes a
    right-handed frame.  Z is consequently a schematic upright axis, not a
    measured gravity axis.
    """

    center_right_raw = -calibration.R.T @ calibration.T
    x_axis_raw = unit(center_right_raw)
    left_forward_raw = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right_forward_raw = calibration.R.T @ left_forward_raw
    mean_forward_raw = unit(left_forward_raw + right_forward_raw)
    y_axis_raw = unit(mean_forward_raw - float(mean_forward_raw @ x_axis_raw) * x_axis_raw)
    z_axis_raw = unit(np.cross(x_axis_raw, y_axis_raw))
    y_axis_raw = unit(np.cross(z_axis_raw, x_axis_raw))
    rotation = np.vstack((x_axis_raw, y_axis_raw, z_axis_raw))

    midpoint_raw = 0.5 * center_right_raw
    target_midpoint_display = np.asarray([0.0, 0.0, -220.0], dtype=np.float64)
    translation = target_midpoint_display - rotation @ midpoint_raw
    return rotation, translation


def schematic_skeleton() -> dict[str, np.ndarray]:
    """A front-facing, upright, millimetre-scale body in the display frame."""

    depth = 1250.0
    return {
        "left_shoulder": np.asarray([-205.0, depth, 520.0]),
        "right_shoulder": np.asarray([205.0, depth, 520.0]),
        "left_elbow": np.asarray([-350.0, depth - 10.0, 235.0]),
        "right_elbow": np.asarray([350.0, depth - 10.0, 235.0]),
        "left_wrist": np.asarray([-390.0, depth - 20.0, -65.0]),
        "right_wrist": np.asarray([390.0, depth - 20.0, -65.0]),
        "left_hip": np.asarray([-145.0, depth, 0.0]),
        "right_hip": np.asarray([145.0, depth, 0.0]),
        "left_knee": np.asarray([-135.0, depth + 5.0, -410.0]),
        "right_knee": np.asarray([135.0, depth + 5.0, -410.0]),
        "left_ankle": np.asarray([-125.0, depth + 15.0, -825.0]),
        "right_ankle": np.asarray([125.0, depth + 15.0, -825.0]),
    }


def view_vector(elevation_deg: float, azimuth_deg: float) -> np.ndarray:
    elevation = math.radians(elevation_deg)
    azimuth = math.radians(azimuth_deg)
    return np.asarray(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ],
        dtype=np.float64,
    )


def projected_fraction(vector: np.ndarray, view: np.ndarray) -> float:
    vector = unit(vector)
    return math.sqrt(max(0.0, 1.0 - float(vector @ view) ** 2))


def projected_angle_sine(left: np.ndarray, right: np.ndarray, view: np.ndarray) -> float:
    left_projected = left - float(left @ view) * view
    right_projected = right - float(right @ view) * view
    denominator = float(np.linalg.norm(left_projected) * np.linalg.norm(right_projected))
    if denominator <= 1e-12:
        return 0.0
    return abs(float(view @ np.cross(left_projected, right_projected))) / denominator


def angular_distance_degrees(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip(abs(first @ second), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def choose_view(
    left_direction: np.ndarray,
    right_direction: np.ndarray,
    baseline: np.ndarray,
    prior_views: list[np.ndarray],
) -> tuple[float, float, np.ndarray, dict[str, float]]:
    best: tuple[float, float, float, float, float, float, np.ndarray] | None = None
    for elevation_deg in range(-36, 49, 3):
        for azimuth_deg in range(-180, 180, 3):
            view = view_vector(elevation_deg, azimuth_deg)
            ray_visibility = min(
                projected_fraction(left_direction, view),
                projected_fraction(right_direction, view),
            )
            baseline_visibility = projected_fraction(baseline, view)
            crossing = projected_angle_sine(left_direction, right_direction, view)
            separation = (
                min(angular_distance_degrees(view, previous) for previous in prior_views)
                if prior_views
                else 90.0
            )
            diversity = min(1.0, separation / 24.0)
            score = (
                ray_visibility**1.4
                * baseline_visibility
                * (0.20 + 0.80 * crossing)
                * (0.45 + 0.55 * diversity)
            )
            candidate = (
                score,
                diversity,
                crossing,
                ray_visibility,
                float(elevation_deg),
                float(azimuth_deg),
                view,
            )
            if best is None or candidate[:6] > best[:6]:
                best = candidate
    if best is None:
        raise RuntimeError("No valid viewing direction found")
    return best[4], best[5], best[6], {
        "score": best[0],
        "minimum_angular_separation_deg": (
            min(angular_distance_degrees(best[6], previous) for previous in prior_views)
            if prior_views
            else 90.0
        ),
        "projected_ray_angle_sine": best[2],
        "minimum_ray_projection_fraction": best[3],
        "baseline_projection_fraction": projected_fraction(baseline, best[6]),
    }


def draw_camera(axis, center: np.ndarray, camera_to_display: np.ndarray, label: str) -> None:
    ex, ey, ez = (camera_to_display[:, index] for index in range(3))
    body_center = center - 16.0 * ez
    half_x, half_y, half_z = 26.0, 19.0, 19.0
    corners = np.asarray(
        [
            body_center + sx * half_x * ex + sy * half_y * ey + sz * half_z * ez
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
    )
    edges = (
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    )
    for a, b in edges:
        axis.plot(*zip(corners[a], corners[b]), color="#303030", linewidth=0.72)
    angles = np.linspace(0.0, 2.0 * np.pi, 72)
    lens_center = center + 6.0 * ez
    ring = lens_center[:, None] + 19.0 * (
        ex[:, None] * np.cos(angles) + ey[:, None] * np.sin(angles)
    )
    axis.plot(ring[0], ring[1], ring[2], color="#111111", linewidth=0.78)
    axis.text(center[0], center[1], center[2] + 32.0, label, fontsize=8, weight="bold")


def neighbour_names(focus: str) -> set[str]:
    names = {focus}
    for first, second in EDGES:
        if first == focus:
            names.add(second)
        if second == focus:
            names.add(first)
    return names


def set_focus_limits(
    axis,
    focus: np.ndarray,
    centers: tuple[np.ndarray, np.ndarray],
    skeleton: dict[str, np.ndarray],
    focus_name: str,
) -> None:
    context = [focus, *centers]
    context.extend(skeleton[name] for name in neighbour_names(focus_name))
    values = np.asarray(context)
    minimum = values.min(axis=0)
    maximum = values.max(axis=0)
    span = maximum - minimum
    padding = np.maximum(np.asarray([90.0, 95.0, 90.0]), 0.09 * span)
    minimum -= padding
    maximum += padding
    axis.set_xlim(minimum[0], maximum[0])
    axis.set_ylim(minimum[1], maximum[1])
    axis.set_zlim(minimum[2], maximum[2])
    axis.set_box_aspect(maximum - minimum)


def main() -> None:
    args = parse_args()
    calibration = StereoCalibration.load(args.calibration)
    if calibration.camera_model != "fisheye":
        raise ValueError("This figure is specifically for the calibrated fisheye rig")

    rotation, translation = build_display_frame(calibration)
    to_display = lambda point: rotation @ np.asarray(point, dtype=np.float64) + translation
    to_raw_left = lambda point: rotation.T @ (np.asarray(point, dtype=np.float64) - translation)

    center_left = to_display(np.zeros(3, dtype=np.float64))
    center_right = to_display(-calibration.R.T @ calibration.T)
    left_camera_to_display = rotation
    right_camera_to_display = rotation @ calibration.R.T
    skeleton = schematic_skeleton()
    unknown = sorted(set(args.joints) - set(skeleton))
    if unknown:
        raise ValueError(f"Unknown joints: {unknown}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_views: list[np.ndarray] = []
    records: list[dict] = []
    baseline = center_right - center_left

    for focus_name in args.joints:
        focus = skeleton[focus_name]
        direction_left = unit(focus - center_left)
        direction_right = unit(focus - center_right)
        elevation, azimuth, view, view_metrics = choose_view(
            direction_left, direction_right, baseline, prior_views
        )
        prior_views.append(view)

        raw_focus = to_raw_left(focus)
        left_pixel = calibration.project_left(raw_focus.reshape(1, 3))[0]
        right_pixel = calibration.project_right(raw_focus.reshape(1, 3))[0]
        left_normalized = calibration.undistort_normalized(left_pixel.reshape(1, 2), "left")[0]
        right_normalized = calibration.undistort_normalized(right_pixel.reshape(1, 2), "right")[0]
        recovered_left_raw = unit(np.asarray([*left_normalized, 1.0]))
        recovered_right_raw = unit(calibration.R.T @ np.asarray([*right_normalized, 1.0]))
        expected_left_raw = unit(raw_focus)
        expected_right_raw = unit(raw_focus - (-calibration.R.T @ calibration.T))
        left_roundtrip_deg = math.degrees(
            math.acos(float(np.clip(recovered_left_raw @ expected_left_raw, -1.0, 1.0)))
        )
        right_roundtrip_deg = math.degrees(
            math.acos(float(np.clip(recovered_right_raw @ expected_right_raw, -1.0, 1.0)))
        )

        figure = plt.figure(figsize=(9.4, 8.4), dpi=220, facecolor="white")
        axis = figure.add_axes([0.01, 0.015, 0.98, 0.945], projection="3d")
        axis.set_proj_type("ortho")
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_axis_off()

        for first, second in EDGES:
            p1, p2 = skeleton[first], skeleton[second]
            axis.plot(*zip(p1, p2), color="#2468c9", linewidth=0.82, alpha=0.98)
        for name, point in skeleton.items():
            if name == focus_name:
                axis.scatter(
                    *point, s=20, c="#e53935", edgecolors="white",
                    linewidths=0.38, depthshade=False, zorder=8,
                )
            else:
                axis.scatter(
                    *point, s=7, c="#2468c9", edgecolors="white",
                    linewidths=0.18, depthshade=False,
                )

        draw_camera(axis, center_left, left_camera_to_display, "L")
        draw_camera(axis, center_right, right_camera_to_display, "R")
        extension = 1.075
        left_end = center_left + extension * (focus - center_left)
        right_end = center_right + extension * (focus - center_right)
        axis.plot(*zip(center_left, left_end), color="#00a6c8", linewidth=0.82)
        axis.plot(*zip(center_right, right_end), color="#d13c89", linewidth=0.82)
        axis.scatter(
            *focus, s=20, c="#e53935", edgecolors="white",
            linewidths=0.38, depthshade=False, zorder=9,
        )
        set_focus_limits(axis, focus, (center_left, center_right), skeleton, focus_name)
        figure.text(
            0.035, 0.965, focus_name.replace("_", " "),
            ha="left", va="top", fontsize=10, weight="semibold", color="#202020",
        )

        png_path = output_dir / f"ideal_upright_{focus_name}_stereo_rays.png"
        svg_path = output_dir / f"ideal_upright_{focus_name}_stereo_rays.svg"
        figure.savefig(png_path, facecolor="white")
        figure.savefig(svg_path, facecolor="white")
        plt.close(figure)

        intersection_left = center_left + float(np.linalg.norm(focus - center_left)) * direction_left
        intersection_right = center_right + float(np.linalg.norm(focus - center_right)) * direction_right
        records.append(
            {
                "joint": focus_name,
                "x_mm": float(focus[0]),
                "y_mm": float(focus[1]),
                "z_mm": float(focus[2]),
                "view_elevation_deg": elevation,
                "view_azimuth_deg": azimuth,
                "ray_intersection_residual_mm": float(np.linalg.norm(intersection_left - intersection_right)),
                "left_pixel_x": float(left_pixel[0]),
                "left_pixel_y": float(left_pixel[1]),
                "right_pixel_x": float(right_pixel[0]),
                "right_pixel_y": float(right_pixel[1]),
                "left_projection_roundtrip_deg": left_roundtrip_deg,
                "right_projection_roundtrip_deg": right_roundtrip_deg,
                **view_metrics,
                "png": str(png_path),
                "svg": str(svg_path),
            }
        )

    csv_path = output_dir / "ideal_upright_stereo_rays.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    vertical = np.asarray([0.0, 0.0, 1.0])
    left_forward = unit(left_camera_to_display[:, 2])
    right_forward = unit(right_camera_to_display[:, 2])
    angle_to_vertical = lambda direction: math.degrees(
        math.acos(float(np.clip(abs(direction @ vertical), -1.0, 1.0)))
    )
    metadata = {
        "figure_type": "calibration-constrained ideal geometry schematic",
        "calibration": str(args.calibration.resolve()),
        "camera_model": calibration.camera_model,
        "length_unit": calibration.length_unit,
        "baseline_mm": float(np.linalg.norm(center_right - center_left)),
        "display_frame": {
            "x": "calibrated left-to-right camera baseline",
            "y": "mean calibrated optical direction with baseline component removed",
            "z": "right-handed schematic upright axis; not measured gravity",
        },
        "left_optical_axis_angle_to_schematic_vertical_deg": angle_to_vertical(left_forward),
        "right_optical_axis_angle_to_schematic_vertical_deg": angle_to_vertical(right_forward),
        "style": {
            "skeleton": "blue #2468c9, 0.82 pt",
            "focus_joint": "red #e53935, 20 pt^2",
            "other_joints": "blue #2468c9, 7 pt^2",
            "left_ray": "cyan #00a6c8, 0.82 pt",
            "right_ray": "magenta #d13c89, 0.82 pt",
            "png_dpi": 220,
        },
        "records": records,
        "interpretation_boundary": (
            "The body is an ideal upright schematic. Relative camera R/T and fisheye projection are "
            "calibration-constrained, but display Z is not a gravity calibration. Exact ray intersection "
            "is constructed for explanation and is not evidence of real 2D or 3D accuracy."
        ),
    }
    metadata_path = output_dir / "ideal_upright_stereo_rays.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
