#!/usr/bin/env python3
"""Render an upright body skeleton, both fisheye cameras, and joint rays."""

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


BODY_EDGES = (
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
)
BODY_INDICES = tuple(range(5, 17))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--estimates-jsonl", type=Path, required=True)
    parser.add_argument("--pair-id", type=int, required=True)
    parser.add_argument("--joints", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_record(path: Path, pair_id: int) -> dict:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if int(row["pair_id"]) == pair_id:
                return row
    raise ValueError(f"pair_id {pair_id} not found: {path}")


def rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = float(np.clip(source @ target, -1.0, 1.0))
    sine = float(np.linalg.norm(cross))
    if sine <= 1e-12:
        if dot > 0:
            return np.eye(3)
        helper = np.asarray([1.0, 0.0, 0.0])
        if abs(float(source @ helper)) > 0.9:
            helper = np.asarray([0.0, 1.0, 0.0])
        axis = np.cross(source, helper)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    axis = cross / sine
    skew = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    angle = math.atan2(sine, dot)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def body_alignment(points: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shoulder_mid = 0.5 * (points[5] + points[6])
    hip_mid = 0.5 * (points[11] + points[12])
    ankle_mid = 0.5 * (points[15] + points[16])
    body_up_left = shoulder_mid - ankle_mid
    body_up_left /= np.linalg.norm(body_up_left)
    upright_rotation = rotation_between(body_up_left, np.asarray([0.0, 0.0, 1.0]))

    lateral_after = upright_rotation @ (points[12] - points[11])
    yaw = math.atan2(float(lateral_after[1]), float(lateral_after[0]))
    cosine, sine = math.cos(-yaw), math.sin(-yaw)
    yaw_rotation = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rotation = yaw_rotation @ upright_rotation
    return rotation, hip_mid, body_up_left


def normalized_ray(calibration: StereoCalibration, pixel: np.ndarray, side: str) -> np.ndarray:
    normalized = calibration.undistort_normalized(pixel.reshape(1, 2), side)[0]
    direction = np.asarray([normalized[0], normalized[1], 1.0], dtype=np.float64)
    if side == "right":
        direction = calibration.R.T @ direction
    return direction / np.linalg.norm(direction)


def closest_points(
    center_left: np.ndarray,
    direction_left: np.ndarray,
    center_right: np.ndarray,
    direction_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    offset = center_left - center_right
    a = float(direction_left @ direction_left)
    b = float(direction_left @ direction_right)
    c = float(direction_right @ direction_right)
    d = float(direction_left @ offset)
    e = float(direction_right @ offset)
    denominator = a * c - b * b
    if abs(denominator) <= 1e-12:
        raise ValueError("Degenerate ray pair")
    left_distance = (b * e - c * d) / denominator
    right_distance = (a * e - b * d) / denominator
    return (
        center_left + left_distance * direction_left,
        center_right + right_distance * direction_right,
        left_distance,
        right_distance,
    )


def draw_camera(axis, center: np.ndarray, rotation_camera_to_display: np.ndarray, label: str) -> None:
    ex, ey, ez = (rotation_camera_to_display[:, index] for index in range(3))
    body_center = center - 18.0 * ez
    half_x, half_y, half_z = 24.0, 17.0, 18.0
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corners.append(body_center + sx * half_x * ex + sy * half_y * ey + sz * half_z * ez)
    corners = np.asarray(corners)
    edges = (
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    )
    for a, b in edges:
        axis.plot(
            [corners[a, 0], corners[b, 0]],
            [corners[a, 1], corners[b, 1]],
            [corners[a, 2], corners[b, 2]],
            color="#333333",
            linewidth=0.9,
        )
    angles = np.linspace(0.0, 2.0 * np.pi, 64)
    lens_center = center + 8.0 * ez
    ring = lens_center[:, None] + 18.0 * (ex[:, None] * np.cos(angles) + ey[:, None] * np.sin(angles))
    axis.plot(ring[0], ring[1], ring[2], color="#111111", linewidth=0.9)
    optical_end = center + 70.0 * ez
    axis.plot(
        [center[0], optical_end[0]],
        [center[1], optical_end[1]],
        [center[2], optical_end[2]],
        color="#555555",
        linewidth=0.9,
    )
    axis.text(center[0], center[1], center[2] + 28.0, label, fontsize=10, weight="bold")


def set_common_limits(axis, bounds: tuple[np.ndarray, np.ndarray]) -> None:
    minimum, maximum = bounds
    center = 0.5 * (minimum + maximum)
    spans = maximum - minimum
    radius_xy = 0.58 * max(float(spans[0]), float(spans[1]))
    radius_z = 0.58 * float(spans[2])
    radius_xy = max(radius_xy, 360.0)
    radius_z = max(radius_z, 520.0)
    axis.set_xlim(center[0] - radius_xy, center[0] + radius_xy)
    axis.set_ylim(center[1] - radius_xy, center[1] + radius_xy)
    axis.set_zlim(center[2] - radius_z, center[2] + radius_z)
    axis.set_box_aspect((2 * radius_xy, 2 * radius_xy, 2 * radius_z))


def optimal_view(
    direction_left: np.ndarray,
    direction_right: np.ndarray,
    closest_left: np.ndarray,
    closest_right: np.ndarray,
) -> tuple[float, float, dict[str, float]]:
    gap = closest_left - closest_right
    gap /= np.linalg.norm(gap)
    best: tuple[float, float, float, float, float, float] | None = None
    for elevation_deg in range(-40, 56, 4):
        elevation = math.radians(elevation_deg)
        for azimuth_deg in range(-180, 180, 4):
            azimuth = math.radians(azimuth_deg)
            view = np.asarray(
                [
                    math.cos(elevation) * math.cos(azimuth),
                    math.cos(elevation) * math.sin(azimuth),
                    math.sin(elevation),
                ],
                dtype=np.float64,
            )
            gap_visibility = math.sqrt(max(0.0, 1.0 - float(view @ gap) ** 2))
            left_visibility = math.sqrt(max(0.0, 1.0 - float(view @ direction_left) ** 2))
            right_visibility = math.sqrt(max(0.0, 1.0 - float(view @ direction_right) ** 2))
            projected_left = direction_left - float(direction_left @ view) * view
            projected_right = direction_right - float(direction_right @ view) * view
            denominator = np.linalg.norm(projected_left) * np.linalg.norm(projected_right)
            if denominator <= 1e-9:
                continue
            projected_sine = abs(float(view @ np.cross(projected_left, projected_right))) / denominator
            ray_visibility = min(left_visibility, right_visibility)
            score = gap_visibility**3 * ray_visibility**1.5 * (0.25 + 0.75 * projected_sine)
            candidate = (
                score,
                gap_visibility,
                ray_visibility,
                projected_sine,
                float(elevation_deg),
                float(azimuth_deg),
            )
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None:
        raise RuntimeError("Could not find a non-degenerate viewing direction")
    return best[4], best[5], {
        "score": best[0],
        "gap_projection_fraction": best[1],
        "minimum_ray_projection_fraction": best[2],
        "projected_ray_angle_sine": best[3],
    }


def draw_ray_geometry(
    axis,
    ray: dict,
    center_left: np.ndarray,
    center_right: np.ndarray,
    line_width: float,
    show_cameras_to_endpoints: bool,
    local_radius_mm: float | None = None,
) -> None:
    if show_cameras_to_endpoints:
        left_start, left_end = center_left, ray["ray_end_left"]
        right_start, right_end = center_right, ray["ray_end_right"]
    else:
        radius = local_radius_mm if local_radius_mm is not None else ray["zoom_radius_mm"]
        left_start = ray["closest_left"] - radius * ray["left_direction"]
        left_end = ray["closest_left"] + radius * ray["left_direction"]
        right_start = ray["closest_right"] - radius * ray["right_direction"]
        right_end = ray["closest_right"] + radius * ray["right_direction"]
    axis.plot(
        [left_start[0], left_end[0]],
        [left_start[1], left_end[1]],
        [left_start[2], left_end[2]],
        color="#00a6c8", linewidth=line_width,
    )
    axis.plot(
        [right_start[0], right_end[0]],
        [right_start[1], right_end[1]],
        [right_start[2], right_end[2]],
        color="#d13c89", linewidth=line_width,
    )
    axis.plot(
        [ray["closest_left"][0], ray["closest_right"][0]],
        [ray["closest_left"][1], ray["closest_right"][1]],
        [ray["closest_left"][2], ray["closest_right"][2]],
        color="#f0a000", linewidth=max(0.75, line_width),
    )
    axis.scatter(
        *ray["closest_left"], s=8, facecolors="none",
        edgecolors="#00a6c8", linewidths=0.6,
    )
    axis.scatter(
        *ray["closest_right"], s=8, facecolors="none",
        edgecolors="#d13c89", linewidths=0.6,
    )


def main() -> None:
    args = parse_args()
    calibration = StereoCalibration.load(args.calibration)
    record = load_record(args.estimates_jsonl, args.pair_id)
    raw_items = {int(item["index"]): item for item in record["keypoints_3d_estimated"]}
    name_to_index = {str(item["name"]): int(item["index"]) for item in raw_items.values()}
    for index in BODY_INDICES:
        if not raw_items[index].get("has_estimate"):
            raise ValueError(f"Body point {raw_items[index]['name']} is unavailable at pair {args.pair_id}")
    for name in args.joints:
        if name not in name_to_index:
            raise ValueError(f"Unknown joint {name!r}")
        if name_to_index[name] not in BODY_INDICES:
            raise ValueError(f"Selected joint {name!r} is outside the rendered body set")

    points_left = {
        index: np.asarray(raw_items[index]["xyz"], dtype=np.float64)
        for index in BODY_INDICES
    }
    align_rotation, body_origin_left, body_up_left = body_alignment(points_left)

    def transform_point(point: np.ndarray) -> np.ndarray:
        return align_rotation @ (point - body_origin_left)

    def transform_direction(direction: np.ndarray) -> np.ndarray:
        result = align_rotation @ direction
        return result / np.linalg.norm(result)

    points_display = {index: transform_point(point) for index, point in points_left.items()}
    center_left_raw = np.zeros(3, dtype=np.float64)
    center_right_raw = -calibration.R.T @ calibration.T
    center_left = transform_point(center_left_raw)
    center_right = transform_point(center_right_raw)
    left_camera_rotation = align_rotation
    right_camera_rotation = align_rotation @ calibration.R.T

    geometry: dict[str, dict] = {}
    bound_points = [*points_display.values(), center_left, center_right]
    for name in args.joints:
        item = raw_items[name_to_index[name]]
        direct = item.get("direct_stereo") or {}
        if direct.get("raw_xyz") is None:
            raise ValueError(f"Selected joint {name!r} has no direct stereo candidate")
        left_pixel = np.asarray(item["left_2d"][:2], dtype=np.float64)
        right_pixel = np.asarray(item["right_2d"][:2], dtype=np.float64)
        direction_left_raw = normalized_ray(calibration, left_pixel, "left")
        direction_right_raw = normalized_ray(calibration, right_pixel, "right")
        closest_left_raw, closest_right_raw, left_distance, right_distance = closest_points(
            center_left_raw,
            direction_left_raw,
            center_right_raw,
            direction_right_raw,
        )
        direction_left = transform_direction(direction_left_raw)
        direction_right = transform_direction(direction_right_raw)
        closest_left = transform_point(closest_left_raw)
        closest_right = transform_point(closest_right_raw)
        ray_end_left = center_left + 1.12 * left_distance * direction_left
        ray_end_right = center_right + 1.12 * right_distance * direction_right
        bound_points.extend((ray_end_left, ray_end_right))
        geometry[name] = {
            "left_pixel": left_pixel,
            "right_pixel": right_pixel,
            "left_direction": direction_left,
            "right_direction": direction_right,
            "closest_left": closest_left,
            "closest_right": closest_right,
            "gap_mm": float(np.linalg.norm(closest_left - closest_right)),
            "ray_end_left": ray_end_left,
            "ray_end_right": ray_end_right,
            "reprojection_mean_px": float(direct["reprojection_error_mean_px"]),
        }

    for name in args.joints:
        ray = geometry[name]
        elevation, azimuth, view_metrics = optimal_view(
            ray["left_direction"],
            ray["right_direction"],
            ray["closest_left"],
            ray["closest_right"],
        )
        ray["view_elevation_deg"] = elevation
        ray["view_azimuth_deg"] = azimuth
        ray["view_metrics"] = view_metrics
        ray["zoom_radius_mm"] = max(38.0, 1.65 * ray["gap_mm"] + 20.0)

    bound_array = np.asarray(bound_points)
    bounds = (bound_array.min(axis=0), bound_array.max(axis=0))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    for name in args.joints:
        focus_index = name_to_index[name]
        ray = geometry[name]
        figure = plt.figure(figsize=(9, 9), dpi=200, facecolor="white")
        axis = figure.add_axes([0.015, 0.015, 0.97, 0.93], projection="3d")
        axis.set_proj_type("ortho")
        axis.view_init(
            elev=ray["view_elevation_deg"],
            azim=ray["view_azimuth_deg"],
        )
        axis.set_axis_off()

        for a, b in BODY_EDGES:
            pa, pb = points_display[a], points_display[b]
            axis.plot(
                [pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                color="#2468c9", linewidth=1.0,
            )
        for index in BODY_INDICES:
            point = points_display[index]
            if index == focus_index:
                axis.scatter(*point, s=18, c="#e53935", edgecolors="white", linewidths=0.45, depthshade=False)
            else:
                axis.scatter(*point, s=8, c="#2468c9", edgecolors="white", linewidths=0.2, depthshade=False)

        focus = points_display[focus_index]
        closest_midpoint = 0.5 * (ray["closest_left"] + ray["closest_right"])
        local_center = 0.5 * (
            focus
            + closest_midpoint
        )
        focus_extent = max(
            float(np.linalg.norm(focus - ray["closest_left"])),
            float(np.linalg.norm(focus - ray["closest_right"])),
        )
        local_radius = max(12.0, 2.4 * ray["gap_mm"], 1.8 * focus_extent)
        ray["local_display_radius_mm"] = local_radius
        draw_ray_geometry(
            axis,
            ray,
            center_left,
            center_right,
            0.8,
            False,
            local_radius_mm=1.25 * local_radius,
        )
        axis.set_xlim(local_center[0] - local_radius, local_center[0] + local_radius)
        axis.set_ylim(local_center[1] - local_radius, local_center[1] + local_radius)
        axis.set_zlim(local_center[2] - local_radius, local_center[2] + local_radius)
        axis.set_box_aspect((1.0, 1.0, 1.0))
        figure.text(
            0.04,
            0.965,
            f"{name.replace('_', ' ')}   |   closest approach {ray['gap_mm']:.1f} mm",
            fontsize=11,
            weight="semibold",
            ha="left",
            va="top",
        )
        figure.text(
            0.04,
            0.025,
            "cyan / magenta: left / right ray    orange: shortest connector",
            fontsize=7,
            color="#454545",
            ha="left",
            va="bottom",
        )

        output = args.output_dir / f"pmpose_pair{args.pair_id:04d}_{name}_optimized_zoom_v4.png"
        figure.savefig(output, facecolor="white")
        plt.close(figure)
        outputs.append(str(output.resolve()))

    upright_check = align_rotation @ body_up_left
    left_optical_display = transform_direction(np.asarray([0.0, 0.0, 1.0]))
    right_optical_display = transform_direction(calibration.R.T @ np.asarray([0.0, 0.0, 1.0]))
    vertical = np.asarray([0.0, 0.0, 1.0])

    def unsigned_angle(direction: np.ndarray) -> float:
        return math.degrees(math.acos(float(np.clip(abs(direction @ vertical), -1.0, 1.0))))

    metadata = {
        "model": record.get("model"),
        "pair_id": args.pair_id,
        "coordinate_frame": "body-aligned display frame derived by rigidly rotating the left-camera optical frame",
        "origin": "midpoint of left/right hips",
        "z_axis_definition": "ankle-midpoint to shoulder-midpoint body axis",
        "x_axis_definition": "left-to-right hip direction after removing the body-axis component",
        "body_axis_after_alignment": upright_check.tolist(),
        "left_camera_center": center_left.tolist(),
        "right_camera_center": center_right.tolist(),
        "left_optical_axis_angle_to_body_vertical_deg": unsigned_angle(left_optical_display),
        "right_optical_axis_angle_to_body_vertical_deg": unsigned_angle(right_optical_display),
        "joints": {
            name: {
                "ray_gap_mm": geometry[name]["gap_mm"],
                "reprojection_error_mean_px": geometry[name]["reprojection_mean_px"],
                "left_observed_pixel": geometry[name]["left_pixel"].tolist(),
                "right_observed_pixel": geometry[name]["right_pixel"].tolist(),
                "view_elevation_deg": geometry[name]["view_elevation_deg"],
                "view_azimuth_deg": geometry[name]["view_azimuth_deg"],
                "view_metrics": geometry[name]["view_metrics"],
                "zoom_radius_mm": geometry[name]["zoom_radius_mm"],
                "local_display_radius_mm": geometry[name]["local_display_radius_mm"],
            }
            for name in args.joints
        },
        "outputs": outputs,
        "interpretation": "All cameras, skeleton points, and rays received one common rigid rotation. Each joint uses an independently optimized viewing direction and a same-view local zoom. Geometry is unchanged; Z is an explicitly body-aligned display axis, not a calibrated gravity axis.",
    }
    (args.output_dir / f"pmpose_pair{args.pair_id:04d}_upright_camera_rays.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_path = args.output_dir / f"pmpose_pair{args.pair_id:04d}_optimized_views.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "joint",
                "view_elevation_deg",
                "view_azimuth_deg",
                "ray_gap_mm",
                "reprojection_error_mean_px",
                "gap_projection_fraction",
                "minimum_ray_projection_fraction",
                "projected_ray_angle_sine",
                "zoom_radius_mm",
                "local_display_radius_mm",
            ),
        )
        writer.writeheader()
        for name in args.joints:
            ray = geometry[name]
            writer.writerow(
                {
                    "joint": name,
                    "view_elevation_deg": ray["view_elevation_deg"],
                    "view_azimuth_deg": ray["view_azimuth_deg"],
                    "ray_gap_mm": ray["gap_mm"],
                    "reprojection_error_mean_px": ray["reprojection_mean_px"],
                    "gap_projection_fraction": ray["view_metrics"]["gap_projection_fraction"],
                    "minimum_ray_projection_fraction": ray["view_metrics"]["minimum_ray_projection_fraction"],
                    "projected_ray_angle_sine": ray["view_metrics"]["projected_ray_angle_sine"],
                    "zoom_radius_mm": ray["zoom_radius_mm"],
                    "local_display_radius_mm": ray["local_display_radius_mm"],
                }
            )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
