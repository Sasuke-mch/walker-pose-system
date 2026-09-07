#!/usr/bin/env python3
"""Render two calibrated fisheye cameras and one saved joint's observation rays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_app.calibration import StereoCalibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--estimates-jsonl", type=Path, required=True)
    parser.add_argument("--pair-id", type=int, required=True)
    parser.add_argument("--joint", required=True)
    parser.add_argument("--output-image", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    return parser.parse_args()


def load_record(path: Path, pair_id: int) -> dict:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row["pair_id"]) == pair_id:
            return row
    raise ValueError(f"pair_id {pair_id} not found in {path}")


def normalized_ray(calibration: StereoCalibration, pixel: np.ndarray, side: str) -> np.ndarray:
    normalized = calibration.undistort_normalized(pixel.reshape(1, 2), side)[0]
    direction = np.asarray([normalized[0], normalized[1], 1.0], dtype=np.float64)
    if side == "right":
        direction = calibration.R.T @ direction
    return direction / np.linalg.norm(direction)


def closest_points_on_rays(
    center_a: np.ndarray,
    direction_a: np.ndarray,
    center_b: np.ndarray,
    direction_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    offset = center_a - center_b
    a = float(direction_a @ direction_a)
    b = float(direction_a @ direction_b)
    c = float(direction_b @ direction_b)
    d = float(direction_a @ offset)
    e = float(direction_b @ offset)
    denominator = a * c - b * b
    if abs(denominator) <= 1e-12:
        raise ValueError("Observation rays are parallel or numerically degenerate")
    distance_a = (b * e - c * d) / denominator
    distance_b = (a * e - b * d) / denominator
    return (
        center_a + distance_a * direction_a,
        center_b + distance_b * direction_b,
        distance_a,
        distance_b,
    )


def draw_camera(
    axis,
    center: np.ndarray,
    rotation_camera_to_left: np.ndarray,
    label: str,
    scale: float = 75.0,
) -> None:
    colors = ("#ff5252", "#55dd66", "#4d8dff")
    for index, color in enumerate(colors):
        direction = rotation_camera_to_left[:, index]
        endpoint = center + scale * direction
        axis.plot(
            [center[0], endpoint[0]],
            [center[1], endpoint[1]],
            [center[2], endpoint[2]],
            color=color,
            linewidth=2.0,
        )
    # A small ring normal to the optical axis marks the fisheye lens plane.
    ex = rotation_camera_to_left[:, 0]
    ey = rotation_camera_to_left[:, 1]
    ez = rotation_camera_to_left[:, 2]
    lens_center = center + 18.0 * ez
    angles = np.linspace(0.0, 2.0 * np.pi, 80)
    ring = lens_center[:, None] + 23.0 * (ex[:, None] * np.cos(angles) + ey[:, None] * np.sin(angles))
    axis.plot(ring[0], ring[1], ring[2], color="#e8e8e8", linewidth=1.3)
    axis.scatter(*center, marker="s", s=85, c="#ffffff", edgecolors="#111111", linewidths=0.8)
    axis.text(*(center + np.asarray([10.0, 10.0, 10.0])), label, color="#111111", fontsize=10, weight="bold")


def set_equal_3d_limits(axis, points: np.ndarray, padding: float = 80.0) -> None:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    radius = 0.5 * float(np.max(maximum - minimum)) + padding
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def draw_projection(
    axis,
    centers: tuple[np.ndarray, np.ndarray],
    directions: tuple[np.ndarray, np.ndarray],
    ray_lengths: tuple[float, float],
    closest: tuple[np.ndarray, np.ndarray],
    dlt: np.ndarray,
    dimensions: tuple[int, int],
    labels: tuple[str, str],
) -> None:
    colors = ("#00a8cc", "#d43f8d")
    for center, direction, length, color in zip(centers, directions, ray_lengths, colors):
        endpoint = center + length * direction
        axis.plot(
            [center[dimensions[0]], endpoint[dimensions[0]]],
            [center[dimensions[1]], endpoint[dimensions[1]]],
            color=color,
            linewidth=2.0,
        )
        axis.scatter(center[dimensions[0]], center[dimensions[1]], marker="s", s=45, c=color)
    axis.plot(
        [closest[0][dimensions[0]], closest[1][dimensions[0]]],
        [closest[0][dimensions[1]], closest[1][dimensions[1]]],
        color="#ffb000",
        linewidth=3.0,
    )
    axis.scatter(
        [closest[0][dimensions[0]], closest[1][dimensions[0]]],
        [closest[0][dimensions[1]], closest[1][dimensions[1]]],
        s=38,
        c=["#00a8cc", "#d43f8d"],
        edgecolors="black",
    )
    axis.scatter(dlt[dimensions[0]], dlt[dimensions[1]], marker="*", s=120, c="#111111")
    axis.set_xlabel(labels[0])
    axis.set_ylabel(labels[1])
    axis.grid(True, alpha=0.35)
    axis.set_aspect("equal", adjustable="datalim")


def main() -> None:
    args = parse_args()
    calibration = StereoCalibration.load(args.calibration)
    record = load_record(args.estimates_jsonl, args.pair_id)
    candidates = {
        str(item["name"]): item for item in record["keypoints_3d_estimated"]
    }
    if args.joint not in candidates:
        raise ValueError(f"Unknown joint {args.joint!r}; available: {sorted(candidates)}")
    joint = candidates[args.joint]
    direct = joint.get("direct_stereo") or {}
    if direct.get("raw_xyz") is None:
        raise ValueError(f"{args.joint} at pair {args.pair_id} has no direct stereo candidate")

    left_pixel = np.asarray(joint["left_2d"][:2], dtype=np.float64)
    right_pixel = np.asarray(joint["right_2d"][:2], dtype=np.float64)
    center_left = np.zeros(3, dtype=np.float64)
    center_right = -calibration.R.T @ calibration.T
    direction_left = normalized_ray(calibration, left_pixel, "left")
    direction_right = normalized_ray(calibration, right_pixel, "right")
    closest_left, closest_right, distance_left, distance_right = closest_points_on_rays(
        center_left, direction_left, center_right, direction_right
    )
    midpoint = 0.5 * (closest_left + closest_right)
    gap = float(np.linalg.norm(closest_left - closest_right))
    dlt = np.asarray(direct["raw_xyz"], dtype=np.float64)
    dlt_to_midpoint = float(np.linalg.norm(dlt - midpoint))
    ray_length_left = max(distance_left * 1.22, np.linalg.norm(dlt - center_left) * 1.18)
    ray_length_right = max(distance_right * 1.22, np.linalg.norm(dlt - center_right) * 1.18)

    figure = plt.figure(figsize=(18, 11), dpi=140, facecolor="white")
    grid = figure.add_gridspec(2, 3, width_ratios=(1.25, 1.25, 1.0), hspace=0.22, wspace=0.22)
    axis_3d = figure.add_subplot(grid[:, :2], projection="3d")
    axis_xz = figure.add_subplot(grid[0, 2])
    axis_yz = figure.add_subplot(grid[1, 2])

    draw_camera(axis_3d, center_left, np.eye(3), "LEFT fisheye / C_L")
    draw_camera(axis_3d, center_right, calibration.R.T, "RIGHT fisheye / C_R")
    endpoint_left = center_left + ray_length_left * direction_left
    endpoint_right = center_right + ray_length_right * direction_right
    axis_3d.plot(
        [center_left[0], endpoint_left[0]],
        [center_left[1], endpoint_left[1]],
        [center_left[2], endpoint_left[2]],
        color="#00a8cc",
        linewidth=3.0,
        label="left observed ray",
    )
    axis_3d.plot(
        [center_right[0], endpoint_right[0]],
        [center_right[1], endpoint_right[1]],
        [center_right[2], endpoint_right[2]],
        color="#d43f8d",
        linewidth=3.0,
        label="right observed ray",
    )
    axis_3d.scatter(*closest_left, s=75, c="#00a8cc", edgecolors="black", linewidths=0.7)
    axis_3d.scatter(*closest_right, s=75, c="#d43f8d", edgecolors="black", linewidths=0.7)
    axis_3d.plot(
        [closest_left[0], closest_right[0]],
        [closest_left[1], closest_right[1]],
        [closest_left[2], closest_right[2]],
        color="#ffb000",
        linewidth=5.0,
        label=f"closest gap = {gap:.1f} mm",
    )
    axis_3d.scatter(*dlt, marker="*", s=230, c="#111111", edgecolors="white", linewidths=0.8, label="DLT compromise point")
    axis_3d.scatter(*midpoint, marker="x", s=90, c="#ffb000", linewidths=2.5, label="closest-pair midpoint")

    all_points = np.vstack((center_left, center_right, endpoint_left, endpoint_right, closest_left, closest_right, dlt))
    set_equal_3d_limits(axis_3d, all_points)
    axis_3d.set_xlabel("Left-camera X (mm)")
    axis_3d.set_ylabel("Left-camera Y (mm)")
    axis_3d.set_zlabel("Left-camera Z / optical depth (mm)")
    axis_3d.set_title("Calibrated fisheye observation rays in the LEFT camera optical frame", fontsize=14, pad=18)
    axis_3d.view_init(elev=20, azim=-58)
    axis_3d.set_proj_type("persp", focal_length=0.9)
    axis_3d.grid(True, alpha=0.4)
    axis_3d.legend(loc="upper left", fontsize=9)

    draw_projection(
        axis_xz,
        (center_left, center_right),
        (direction_left, direction_right),
        (ray_length_left, ray_length_right),
        (closest_left, closest_right),
        dlt,
        (0, 2),
        ("Left-camera X (mm)", "Left-camera Z (mm)"),
    )
    axis_xz.set_title("X-Z projection")
    draw_projection(
        axis_yz,
        (center_left, center_right),
        (direction_left, direction_right),
        (ray_length_left, ray_length_right),
        (closest_left, closest_right),
        dlt,
        (1, 2),
        ("Left-camera Y (mm)", "Left-camera Z (mm)"),
    )
    axis_yz.set_title("Y-Z projection")

    model_label = str(record.get("model", "unknown"))
    mean_reprojection = direct.get("reprojection_error_mean_px")
    figure.suptitle(
        f"{model_label} | pair {args.pair_id:04d} | {args.joint}: the two rays are skew, not intersecting",
        fontsize=17,
        weight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.025,
        (
            f"Observed pixels: left=({left_pixel[0]:.1f}, {left_pixel[1]:.1f}), "
            f"right=({right_pixel[0]:.1f}, {right_pixel[1]:.1f}) | "
            f"ray gap={gap:.2f} mm | mean fisheye reprojection={float(mean_reprojection):.2f} px | "
            f"DLT-to-gap-midpoint={dlt_to_midpoint:.2f} mm"
        ),
        ha="center",
        fontsize=11,
    )
    figure.text(
        0.5,
        0.006,
        "Camera-axis colors: X red, Y green, Z blue. The yellow segment is the shortest separation; it is not a measured body segment.",
        ha="center",
        fontsize=9,
        color="#444444",
    )

    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_image, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    metadata = {
        "model": record.get("model"),
        "pair_id": args.pair_id,
        "file_name": record.get("file_name"),
        "joint": args.joint,
        "coordinate_frame": "left_camera_optical",
        "length_unit": calibration.length_unit,
        "left_camera_center": center_left.tolist(),
        "right_camera_center_in_left_frame": center_right.tolist(),
        "baseline_mm": calibration.baseline,
        "left_observed_pixel": left_pixel.tolist(),
        "right_observed_pixel": right_pixel.tolist(),
        "left_ray_direction_in_left_frame": direction_left.tolist(),
        "right_ray_direction_in_left_frame": direction_right.tolist(),
        "closest_point_on_left_ray": closest_left.tolist(),
        "closest_point_on_right_ray": closest_right.tolist(),
        "closest_ray_gap_mm": gap,
        "closest_pair_midpoint": midpoint.tolist(),
        "dlt_point": dlt.tolist(),
        "dlt_to_closest_pair_midpoint_mm": dlt_to_midpoint,
        "reprojection_error_left_px": direct.get("reprojection_error_left_px"),
        "reprojection_error_right_px": direct.get("reprojection_error_right_px"),
        "reprojection_error_mean_px": mean_reprojection,
        "interpretation": "The rays are calibrated central rays derived from fisheye pixels. A nonzero closest gap means there is no exact 3-D intersection; DLT is a compromise, not ground truth.",
    }
    args.output_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
