#!/usr/bin/env python3
"""Render strict-gated stereo body points and observed joint rays."""

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
from tools.render_upright_skeleton_camera_rays import (
    BODY_EDGES,
    BODY_INDICES,
    body_alignment,
    closest_points,
    draw_camera,
    normalized_ray,
    set_common_limits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--strict-results-jsonl", type=Path, required=True)
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
    raise ValueError(f"pair_id {pair_id} not found in {path}")


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


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
    return math.degrees(math.acos(float(np.clip(abs(first @ second), -1.0, 1.0))))


def choose_distinct_view(
    direction_left: np.ndarray,
    direction_right: np.ndarray,
    gap_vector: np.ndarray,
    prior_views: list[np.ndarray],
) -> tuple[float, float, np.ndarray, dict[str, float]]:
    gap_direction = unit(gap_vector)
    best = None
    for elevation_deg in range(-42, 55, 3):
        for azimuth_deg in range(-180, 180, 3):
            view = view_vector(elevation_deg, azimuth_deg)
            gap_visibility = projected_fraction(gap_direction, view)
            ray_visibility = min(
                projected_fraction(direction_left, view),
                projected_fraction(direction_right, view),
            )
            crossing = projected_angle_sine(direction_left, direction_right, view)
            separation = (
                min(angular_distance_degrees(view, previous) for previous in prior_views)
                if prior_views
                else 90.0
            )
            diversity = min(1.0, separation / 25.0)
            score = (
                gap_visibility**2.4
                * ray_visibility**1.2
                * (0.18 + 0.82 * crossing)
                * (0.42 + 0.58 * diversity)
            )
            candidate = (
                score,
                diversity,
                gap_visibility,
                crossing,
                ray_visibility,
                float(elevation_deg),
                float(azimuth_deg),
                view,
            )
            if best is None or candidate[:7] > best[:7]:
                best = candidate
    if best is None:
        raise RuntimeError("No usable view found")
    return best[5], best[6], best[7], {
        "score": best[0],
        "minimum_angular_separation_deg": (
            min(angular_distance_degrees(best[7], previous) for previous in prior_views)
            if prior_views
            else 90.0
        ),
        "gap_projection_fraction": best[2],
        "projected_ray_angle_sine": best[3],
        "minimum_ray_projection_fraction": best[4],
    }


def draw_skeleton(axis, points: dict[int, np.ndarray], focus_index: int, *, local: bool) -> None:
    for first, second in BODY_EDGES:
        p1, p2 = points[first], points[second]
        axis.plot(
            [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            color="#2468c9", linewidth=0.72 if local else 0.64,
        )
    for index in BODY_INDICES:
        point = points[index]
        if index == focus_index:
            axis.scatter(
                *point, s=16 if local else 12, c="#e53935", edgecolors="white",
                linewidths=0.36, depthshade=False, zorder=10,
            )
        else:
            axis.scatter(
                *point, s=5 if local else 4, c="#2468c9", edgecolors="white",
                linewidths=0.14, depthshade=False,
            )


def draw_rays(axis, ray: dict, center_left: np.ndarray, center_right: np.ndarray, *, local: bool) -> None:
    if local:
        radius = ray["local_radius_mm"] * 1.35
        left_start = ray["closest_left"] - radius * ray["direction_left"]
        left_end = ray["closest_left"] + radius * ray["direction_left"]
        right_start = ray["closest_right"] - radius * ray["direction_right"]
        right_end = ray["closest_right"] + radius * ray["direction_right"]
        linewidth = 0.78
    else:
        left_start, left_end = center_left, ray["ray_end_left"]
        right_start, right_end = center_right, ray["ray_end_right"]
        linewidth = 0.68
    axis.plot(*zip(left_start, left_end), color="#00a6c8", linewidth=linewidth)
    axis.plot(*zip(right_start, right_end), color="#d13c89", linewidth=linewidth)
    axis.plot(
        *zip(ray["closest_left"], ray["closest_right"]),
        color="#f0a000", linewidth=0.68,
    )


def main() -> None:
    args = parse_args()
    calibration = StereoCalibration.load(args.calibration)
    record = load_record(args.strict_results_jsonl, args.pair_id)
    if len(record.get("persons_3d", [])) != 1:
        raise ValueError("Selected frame must contain exactly one accepted stereo person")
    person = record["persons_3d"][0]
    keypoints = {int(item["index"]): item for item in person["keypoints_3d"]}
    names = {str(item["name"]): int(item["index"]) for item in keypoints.values()}
    invalid_body = [keypoints[index]["name"] for index in BODY_INDICES if not keypoints[index]["valid"]]
    if invalid_body:
        raise ValueError(f"All rendered body points must pass strict gates: {invalid_body}")
    for name in args.joints:
        if name not in names or names[name] not in BODY_INDICES:
            raise ValueError(f"Unknown rendered body joint {name!r}")
        if not keypoints[names[name]]["valid"]:
            raise ValueError(f"Focus joint {name!r} did not pass strict gates")

    left_person = record["left"]["persons"][int(person["left_person_id"])]
    right_person = record["right"]["persons"][int(person["right_person_id"])]
    left_2d = np.asarray(left_person["keypoints"], dtype=np.float64)
    right_2d = np.asarray(right_person["keypoints"], dtype=np.float64)
    points_raw = {
        index: np.asarray(keypoints[index]["xyz"], dtype=np.float64)
        for index in BODY_INDICES
    }
    align_rotation, body_origin, body_up = body_alignment(points_raw)
    transform_point = lambda point: align_rotation @ (np.asarray(point) - body_origin)
    transform_direction = lambda direction: unit(align_rotation @ np.asarray(direction))
    points = {index: transform_point(point) for index, point in points_raw.items()}
    center_left_raw = np.zeros(3, dtype=np.float64)
    center_right_raw = -calibration.R.T @ calibration.T
    center_left = transform_point(center_left_raw)
    center_right = transform_point(center_right_raw)

    geometry = {}
    bound_points = [*points.values(), center_left, center_right]
    prior_views: list[np.ndarray] = []
    for name in args.joints:
        index = names[name]
        direction_left_raw = normalized_ray(calibration, left_2d[index, :2], "left")
        direction_right_raw = normalized_ray(calibration, right_2d[index, :2], "right")
        closest_left_raw, closest_right_raw, distance_left, distance_right = closest_points(
            center_left_raw, direction_left_raw, center_right_raw, direction_right_raw
        )
        direction_left = transform_direction(direction_left_raw)
        direction_right = transform_direction(direction_right_raw)
        closest_left = transform_point(closest_left_raw)
        closest_right = transform_point(closest_right_raw)
        focus = points[index]
        closest_midpoint = 0.5 * (closest_left + closest_right)
        gap_mm = float(np.linalg.norm(closest_left - closest_right))
        focus_offset_mm = float(np.linalg.norm(focus - closest_midpoint))
        local_radius_mm = max(6.0, 3.0 * gap_mm, 3.0 * focus_offset_mm)
        ray_end_left = center_left + 1.06 * distance_left * direction_left
        ray_end_right = center_right + 1.06 * distance_right * direction_right
        elevation, azimuth, view, metrics = choose_distinct_view(
            direction_left,
            direction_right,
            closest_left - closest_right,
            prior_views,
        )
        prior_views.append(view)
        geometry[name] = {
            "index": index,
            "focus": focus,
            "direction_left": direction_left,
            "direction_right": direction_right,
            "closest_left": closest_left,
            "closest_right": closest_right,
            "closest_midpoint": closest_midpoint,
            "gap_mm": gap_mm,
            "focus_offset_mm": focus_offset_mm,
            "local_radius_mm": local_radius_mm,
            "ray_end_left": ray_end_left,
            "ray_end_right": ray_end_right,
            "reprojection_error_mean_px": float(keypoints[index]["reprojection_error_mean_px"]),
            "view_elevation_deg": elevation,
            "view_azimuth_deg": azimuth,
            "view_metrics": metrics,
            "left_pixel": left_2d[index, :2].tolist(),
            "right_pixel": right_2d[index, :2].tolist(),
        }
        bound_points.extend((ray_end_left, ray_end_right))

    bound_array = np.asarray(bound_points)
    bounds = (bound_array.min(axis=0), bound_array.max(axis=0))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for name in args.joints:
        ray = geometry[name]
        figure = plt.figure(figsize=(13.0, 8.0), dpi=200, facecolor="white")
        context = figure.add_axes([0.01, 0.035, 0.42, 0.90], projection="3d")
        zoom = figure.add_axes([0.41, 0.035, 0.58, 0.90], projection="3d")
        for axis in (context, zoom):
            axis.set_proj_type("ortho")
            axis.view_init(elev=ray["view_elevation_deg"], azim=ray["view_azimuth_deg"])
            axis.set_axis_off()

        draw_skeleton(context, points, ray["index"], local=False)
        draw_camera(context, center_left, align_rotation, "L")
        draw_camera(context, center_right, align_rotation @ calibration.R.T, "R")
        draw_rays(context, ray, center_left, center_right, local=False)
        set_common_limits(context, bounds)

        draw_rays(zoom, ray, center_left, center_right, local=True)
        zoom.scatter(
            *ray["focus"], s=16, c="#e53935", edgecolors="white",
            linewidths=0.36, depthshade=False, zorder=10,
        )
        local_center = 0.5 * (ray["focus"] + ray["closest_midpoint"])
        radius = ray["local_radius_mm"]
        zoom.set_xlim(local_center[0] - radius, local_center[0] + radius)
        zoom.set_ylim(local_center[1] - radius, local_center[1] + radius)
        zoom.set_zlim(local_center[2] - radius, local_center[2] + radius)
        zoom.set_box_aspect((1.0, 1.0, 1.0))

        figure.text(
            0.025, 0.965, name.replace("_", " "),
            ha="left", va="top", fontsize=10.5, weight="semibold", color="#202020",
        )
        figure.text(
            0.985, 0.025,
            f"ray gap {ray['gap_mm']:.2f} mm   |   reprojection {ray['reprojection_error_mean_px']:.2f} px",
            ha="right", va="bottom", fontsize=7.2, color="#4b4b4b",
        )
        png = output_dir / f"pmpose_mid_pair{args.pair_id:04d}_{name}_validated_view.png"
        svg = output_dir / f"pmpose_mid_pair{args.pair_id:04d}_{name}_validated_view.svg"
        figure.savefig(png, facecolor="white")
        figure.savefig(svg, facecolor="white")
        plt.close(figure)
        records.append(
            {
                "joint": name,
                "pair_id": args.pair_id,
                "reprojection_error_mean_px": ray["reprojection_error_mean_px"],
                "ray_gap_mm": ray["gap_mm"],
                "dlt_to_closest_midpoint_mm": ray["focus_offset_mm"],
                "view_elevation_deg": ray["view_elevation_deg"],
                "view_azimuth_deg": ray["view_azimuth_deg"],
                "local_radius_mm": ray["local_radius_mm"],
                **ray["view_metrics"],
                "left_pixel_x": ray["left_pixel"][0],
                "left_pixel_y": ray["left_pixel"][1],
                "right_pixel_x": ray["right_pixel"][0],
                "right_pixel_y": ray["right_pixel"][1],
                "png": str(png),
                "svg": str(svg),
            }
        )

    csv_path = output_dir / f"pmpose_mid_pair{args.pair_id:04d}_validated_views.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    metadata = {
        "model": record["left"]["model_name"],
        "sequence": "mid",
        "pair_id": args.pair_id,
        "strict_results_jsonl": str(args.strict_results_jsonl.resolve()),
        "calibration": str(args.calibration.resolve()),
        "accepted_stereo_people": len(record["persons_3d"]),
        "association_cost": float(person["association_cost"]),
        "valid_keypoints": int(person["valid_keypoints"]),
        "valid_rendered_body_points": len(BODY_INDICES),
        "person_mean_reprojection_error_px": float(person["mean_reprojection_error_px"]),
        "body_axis_after_alignment": (align_rotation @ body_up).tolist(),
        "display_z_boundary": "body-aligned axis from ankle midpoint to shoulder midpoint; not gravity",
        "records": records,
        "interpretation_boundary": (
            "Every rendered body point passed the existing strict stereo gates. Rays come from saved raw "
            "fisheye pixels. Small ray gaps are observation-consistency diagnostics, not external 3D accuracy."
        ),
    }
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
