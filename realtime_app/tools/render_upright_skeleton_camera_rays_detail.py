#!/usr/bin/env python3
"""Render the full-context pair-0123 ray figures with a legible main-panel closest pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_app.calibration import StereoCalibration
from tools.render_upright_skeleton_camera_rays import (
    BODY_EDGES,
    BODY_INDICES,
    body_alignment,
    closest_points,
    draw_camera,
    normalized_ray,
    optimal_view,
    set_common_limits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--estimates-jsonl", type=Path, required=True)
    parser.add_argument("--pair-id", type=int, required=True)
    parser.add_argument("--joints", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-version", default="v4")
    return parser.parse_args()


def load_record(path: Path, pair_id: int) -> dict:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if int(record["pair_id"]) == pair_id:
                return record
    raise ValueError(f"pair_id {pair_id} not found: {path}")


def draw_rays(
    axis,
    ray: dict,
    center_left: np.ndarray,
    center_right: np.ndarray,
    *,
    local: bool,
    closest_marker_size: float,
    closest_marker_linewidth: float,
    connector_linewidth: float,
    show_closest_markers: bool,
) -> None:
    if local:
        radius = ray["zoom_radius_mm"]
        left_start = ray["closest_left"] - radius * ray["left_direction"]
        left_end = ray["closest_left"] + radius * ray["left_direction"]
        right_start = ray["closest_right"] - radius * ray["right_direction"]
        right_end = ray["closest_right"] + radius * ray["right_direction"]
        ray_linewidth = 1.0
    else:
        left_start, left_end = center_left, ray["ray_end_left"]
        right_start, right_end = center_right, ray["ray_end_right"]
        ray_linewidth = 1.05

    axis.plot(
        [left_start[0], left_end[0]],
        [left_start[1], left_end[1]],
        [left_start[2], left_end[2]],
        color="#00a6c8",
        linewidth=ray_linewidth,
    )
    axis.plot(
        [right_start[0], right_end[0]],
        [right_start[1], right_end[1]],
        [right_start[2], right_end[2]],
        color="#d13c89",
        linewidth=ray_linewidth,
    )
    axis.plot(
        [ray["closest_left"][0], ray["closest_right"][0]],
        [ray["closest_left"][1], ray["closest_right"][1]],
        [ray["closest_left"][2], ray["closest_right"][2]],
        color="#f0a000",
        linewidth=connector_linewidth,
    )
    if show_closest_markers:
        axis.scatter(
            *ray["closest_left"],
            s=closest_marker_size,
            facecolors="none" if local else "white",
            edgecolors="#00a6c8",
            linewidths=closest_marker_linewidth,
            depthshade=False,
        )
        axis.scatter(
            *ray["closest_right"],
            s=closest_marker_size,
            facecolors="none" if local else "white",
            edgecolors="#d13c89",
            linewidths=closest_marker_linewidth,
            depthshade=False,
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
        if name not in name_to_index or name_to_index[name] not in BODY_INDICES:
            raise ValueError(f"Unknown rendered body joint {name!r}")

    points_left = {
        index: np.asarray(raw_items[index]["xyz"], dtype=np.float64)
        for index in BODY_INDICES
    }
    align_rotation, body_origin_left, body_up_left = body_alignment(points_left)

    def transform_point(point: np.ndarray) -> np.ndarray:
        return align_rotation @ (np.asarray(point, dtype=np.float64) - body_origin_left)

    def transform_direction(direction: np.ndarray) -> np.ndarray:
        result = align_rotation @ np.asarray(direction, dtype=np.float64)
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
        elevation, azimuth, metrics = optimal_view(
            direction_left,
            direction_right,
            closest_left,
            closest_right,
        )
        focus = points_display[name_to_index[name]]
        closest_midpoint = 0.5 * (closest_left + closest_right)
        geometry[name] = {
            "left_pixel": left_pixel,
            "right_pixel": right_pixel,
            "left_direction": direction_left,
            "right_direction": direction_right,
            "closest_left": closest_left,
            "closest_right": closest_right,
            "closest_midpoint": closest_midpoint,
            "gap_mm": float(np.linalg.norm(closest_left - closest_right)),
            "dlt_to_midpoint_mm": float(np.linalg.norm(focus - closest_midpoint)),
            "dlt_to_left_closest_mm": float(np.linalg.norm(focus - closest_left)),
            "dlt_to_right_closest_mm": float(np.linalg.norm(focus - closest_right)),
            "ray_end_left": ray_end_left,
            "ray_end_right": ray_end_right,
            "reprojection_mean_px": float(direct["reprojection_error_mean_px"]),
            "view_elevation_deg": elevation,
            "view_azimuth_deg": azimuth,
            "view_metrics": metrics,
            "zoom_radius_mm": max(38.0, 1.65 * float(np.linalg.norm(closest_left - closest_right)) + 20.0),
        }

    bound_array = np.asarray(bound_points)
    bounds = (bound_array.min(axis=0), bound_array.max(axis=0))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    for name in args.joints:
        focus_index = name_to_index[name]
        focus = points_display[focus_index]
        ray = geometry[name]
        figure = plt.figure(figsize=(14, 9.5), dpi=200, facecolor="white")
        grid = figure.add_gridspec(
            1,
            2,
            width_ratios=(3.25, 1.05),
            left=0.015,
            right=0.985,
            bottom=0.055,
            top=0.915,
            wspace=-0.02,
        )
        axis = figure.add_subplot(grid[0, 0], projection="3d")
        axis.set_proj_type("persp", focal_length=0.9)
        axis.view_init(elev=ray["view_elevation_deg"], azim=ray["view_azimuth_deg"])
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("X (mm)", labelpad=5)
        axis.set_ylabel("Y (mm)", labelpad=5)
        axis.set_zlabel("Body Z (mm)", labelpad=6)
        axis.tick_params(labelsize=8, pad=1)
        set_common_limits(axis, bounds)

        for first, second in BODY_EDGES:
            point_first, point_second = points_display[first], points_display[second]
            axis.plot(
                [point_first[0], point_second[0]],
                [point_first[1], point_second[1]],
                [point_first[2], point_second[2]],
                color="#2468c9",
                linewidth=1.25,
            )
        for index in BODY_INDICES:
            if index == focus_index:
                continue
            point = points_display[index]
            axis.scatter(
                *point,
                s=31,
                c="#2468c9",
                edgecolors="white",
                linewidths=0.45,
                depthshade=False,
            )

        draw_camera(axis, center_left, left_camera_rotation, "L")
        draw_camera(axis, center_right, right_camera_rotation, "R")
        draw_rays(
            axis,
            ray,
            center_left,
            center_right,
            local=False,
            closest_marker_size=44,
            closest_marker_linewidth=1.05,
            connector_linewidth=1.45,
            show_closest_markers=False,
        )
        axis.scatter(
            *focus,
            s=18,
            c="#e53935",
            edgecolors="none",
            depthshade=False,
        )

        figure.text(
            0.025,
            0.955,
            f"PMPose | pair {args.pair_id:04d} | {name.replace('_', ' ')}",
            fontsize=15,
            weight="semibold",
            ha="left",
            va="top",
        )
        figure.text(
            0.025,
            0.025,
            "blue: skeleton   red: selected joint   cyan/magenta: left/right ray   orange: shortest connector",
            fontsize=8,
            color="#454545",
            ha="left",
            va="bottom",
        )

        inset = figure.add_subplot(grid[0, 1], projection="3d", facecolor="white")
        inset.set_proj_type("ortho")
        inset.view_init(elev=ray["view_elevation_deg"], azim=ray["view_azimuth_deg"])
        draw_rays(
            inset,
            ray,
            center_left,
            center_right,
            local=True,
            closest_marker_size=16,
            closest_marker_linewidth=0.75,
            connector_linewidth=1.0,
            show_closest_markers=False,
        )
        inset.scatter(
            *focus,
            s=18,
            c="#e53935",
            edgecolors="none",
            depthshade=False,
        )
        inset.set_xlim(ray["closest_midpoint"][0] - ray["zoom_radius_mm"], ray["closest_midpoint"][0] + ray["zoom_radius_mm"])
        inset.set_ylim(ray["closest_midpoint"][1] - ray["zoom_radius_mm"], ray["closest_midpoint"][1] + ray["zoom_radius_mm"])
        inset.set_zlim(ray["closest_midpoint"][2] - ray["zoom_radius_mm"], ray["closest_midpoint"][2] + ray["zoom_radius_mm"])
        inset.set_box_aspect((1.0, 1.0, 1.0))
        inset.set_axis_off()
        inset.set_title(f"Closest approach\n{ray['gap_mm']:.1f} mm", fontsize=10, pad=8)

        output = args.output_dir / f"pmpose_pair{args.pair_id:04d}_{name}_optimized_view_{args.output_version}.png"
        figure.savefig(output, facecolor="white")
        plt.close(figure)
        outputs.append(str(output.resolve()))

    metadata = {
        "model": record.get("model"),
        "pair_id": args.pair_id,
        "inputs": {
            "calibration": str(args.calibration.resolve()),
            "estimates_jsonl": str(args.estimates_jsonl.resolve()),
            "reference_full_context_figures": [
                str((args.output_dir / f"pmpose_pair{args.pair_id:04d}_{name}_optimized_view_v3.png").resolve())
                for name in args.joints
            ],
        },
        "geometry": {
            name: {
                "ray_gap_mm": ray["gap_mm"],
                "dlt_to_closest_pair_midpoint_mm": ray["dlt_to_midpoint_mm"],
                "dlt_to_left_ray_closest_point_mm": ray["dlt_to_left_closest_mm"],
                "dlt_to_right_ray_closest_point_mm": ray["dlt_to_right_closest_mm"],
                "reprojection_error_mean_px": ray["reprojection_mean_px"],
                "view_elevation_deg": ray["view_elevation_deg"],
                "view_azimuth_deg": ray["view_azimuth_deg"],
            }
            for name, ray in geometry.items()
        },
        "render_change": {
            "kept": "pair, body-aligned coordinates, camera geometry, joint coordinates, ray geometry, v3 layout, viewing directions, axes, skeleton, cameras, and right local inset",
            "intersection_style": "both panels remove all closest-point markers on the cyan and magenta rays; the selected DLT joint is the only intersection marker, a solid 18 pt^2 red dot; the orange shortest connector remains",
        },
        "outputs": outputs,
        "interpretation": "The closest-point pair and DLT point are rendered from the saved raw fisheye observations and calibration. They are a geometry-consistency diagnostic, not a 3-D accuracy or ground-truth claim.",
    }
    metadata_path = args.output_dir / f"pmpose_pair{args.pair_id:04d}_optimized_view_{args.output_version}_detail.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
