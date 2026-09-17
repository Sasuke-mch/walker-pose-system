#!/usr/bin/env python3
"""Render unsmoothed ground-frame skeletons with partial handle proxies and distance audits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from pose_app.gait_interaction_candidates import point_segment_distance_mm  # noqa: E402


EDGES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
    (3, 5), (4, 6),
)
SIDES = {"left": {"wrist": 9, "ankle": 15}, "right": {"wrist": 10, "ankle": 16}}
STAGE2 = "stage2_feet_static_walker_moving"
STAGE_NAMES = {
    "stage1_walker_static_human_moving": "STAGE 1  HUMAN MOVING / WALKER STATIC",
    STAGE2: "STAGE 2  FEET STATIC / WALKER MOVING",
    "transition": "TRANSITION",
    "warming_up": "WARMING UP",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def joint_rows(row: dict) -> dict[int, dict]:
    people = row.get("persons_3d") or []
    if len(people) != 1:
        return {}
    return {int(point["index"]): point for point in people[0].get("keypoints_3d", [])}


def fit_wrist_supported_handle_proxy(strict_rows: list[dict], dynamic_rows: list[dict], side: str) -> dict:
    index = SIDES[side]["wrist"]
    points, errors, pair_ids = [], [], []
    for stereo, dynamic in zip(strict_rows, dynamic_rows):
        if dynamic.get("stage") != STAGE2:
            continue
        point = joint_rows(stereo).get(index)
        if point is None or not bool(point.get("valid")):
            continue
        xyz = np.asarray(point.get("xyz"), dtype=np.float64)
        error = float(point.get("reprojection_error_mean_px", float("nan")))
        if xyz.shape == (3,) and np.all(np.isfinite(xyz)) and np.isfinite(error):
            points.append(xyz)
            errors.append(error)
            pair_ids.append(int(stereo["pair_id"]))
    if len(points) < 12:
        raise RuntimeError(f"insufficient strict Stage-2 {side} wrists for a handle proxy")
    values = np.asarray(points)
    median = np.median(values, axis=0)
    radial = np.linalg.norm(values - median, axis=1)
    radial_mad = float(np.median(np.abs(radial - np.median(radial))))
    cutoff = max(15.0, float(np.median(radial) + 3.5 * 1.4826 * radial_mad))
    inliers = radial <= cutoff
    kept = values[inliers]
    centered = kept - np.median(kept, axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    projection = centered @ direction
    low, high = np.percentile(projection, [5.0, 95.0])
    center = np.median(kept, axis=0)
    start, end = center + low * direction, center + high * direction
    return {
        "side": side,
        "source": "robust_stage2_strict_wrist_cluster_in_left_camera",
        "geometry_status": "wrist_supported_partial_handle_proxy_not_independent_reconstruction",
        "start_left_camera_mm": start.tolist(),
        "end_left_camera_mm": end.tolist(),
        "observed_proxy_length_mm": float(np.linalg.norm(end - start)),
        "sample_count": len(values),
        "inlier_count": int(np.count_nonzero(inliers)),
        "radial_inlier_cutoff_mm": cutoff,
        "principal_variance_fraction": float(singular[0] ** 2 / np.sum(singular ** 2)),
        "source_pair_id_min_max": [min(pair_ids), max(pair_ids)],
        "source_wrist_reprojection_error_median_px": float(np.median(errors)),
        "source_wrist_reprojection_error_p95_px": float(np.percentile(errors, 95)),
        "contact_evidence_eligible": False,
        "contact_blocker": "handle proxy is derived from wrist locations and lacks independent walker-pixel identity",
    }


def transform(point: np.ndarray, record: dict) -> np.ndarray:
    rotation = np.asarray(record["rotation_world_from_left_camera"], dtype=np.float64)
    translation = np.asarray(record["translation_world_from_left_camera_mm"], dtype=np.float64)
    return rotation @ np.asarray(point, dtype=np.float64) + translation


def error_color(error: float | None) -> str:
    if error is None or not np.isfinite(error):
        return "#777777"
    if error <= 5.0:
        return "#009E73"
    if error <= 10.0:
        return "#E69F00"
    return "#D55E00"


def choose_azimuth(camera_xyz: np.ndarray) -> float:
    movement = camera_xyz[-1, :2] - camera_xyz[0, :2]
    return float(np.degrees(np.arctan2(movement[1], movement[0])) - 35.0) if np.linalg.norm(movement) else -58.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic-ground-jsonl", type=Path, required=True)
    parser.add_argument("--strict-stereo-jsonl", type=Path, required=True)
    parser.add_argument("--visual-stereo-jsonl", type=Path, required=True)
    parser.add_argument("--handle-axes-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--near-handle-axis-mm", type=float, default=60.0)
    parser.add_argument("--near-ground-ankle-mm", type=float, default=80.0)
    parser.add_argument("--maximum-audit-reprojection-error-px", type=float, default=10.0)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    dynamic_rows = read_jsonl(args.dynamic_ground_jsonl)
    strict_rows = read_jsonl(args.strict_stereo_jsonl)
    visual_rows = read_jsonl(args.visual_stereo_jsonl)
    handle_axis_rows = read_jsonl(args.handle_axes_jsonl) if args.handle_axes_jsonl else None
    if not (len(dynamic_rows) == len(strict_rows) == len(visual_rows)):
        raise ValueError("dynamic, strict and visual streams must have identical lengths")
    if handle_axis_rows is not None and len(handle_axis_rows) != len(dynamic_rows):
        raise ValueError("handle axis stream must align one-to-one with dynamic records")
    for dynamic, strict, visual in zip(dynamic_rows, strict_rows, visual_rows):
        if len({int(dynamic["pair_id"]), int(strict["pair_id"]), int(visual["pair_id"])}) != 1:
            raise ValueError("unaligned pair streams")

    proxies = None
    if handle_axis_rows is None:
        proxies = {
            side: fit_wrist_supported_handle_proxy(strict_rows, dynamic_rows, side)
            for side in ("left", "right")
        }
        handle_source_record = {
            "schema_version": "wrist_supported_partial_handle_proxy_v1",
            "coordinate_frame": "left_camera", "length_unit": "millimeter",
            "handles": list(proxies.values()),
            "interpretation": (
                "Partial display proxy inferred from robust Stage-2 strict wrist clusters. "
                "It is not independent walker reconstruction and cannot validate hand contact."
            ),
        }
    else:
        handle_source_record = {
            "schema_version": "image_derived_stereo_handle_axes_reference_v1",
            "source_jsonl": str(args.handle_axes_jsonl.resolve()),
            "coordinate_frame": "per-frame left_camera", "length_unit": "millimeter",
            "interpretation": (
                "Each displayed axis comes from current-frame bilateral dark handle-line pixels and calibrated "
                "stereo geometry. Wrists select the image ROI but do not define the 3-D axis."
            ),
        }
    (output / "partial_handle_proxy.json").write_text(
        json.dumps(handle_source_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    visual_points, errors_by_frame, handles_by_frame, audits = [], [], [], []
    values: dict[str, dict[str, list[float]]] = {
        side: defaultdict(list) for side in ("left", "right")
    }
    states: dict[str, Counter] = {side: Counter() for side in ("left", "right")}
    handle_iterator = handle_axis_rows if handle_axis_rows is not None else [None] * len(dynamic_rows)
    transformed_from_camera_count = 0
    for dynamic, strict, visual, handle_axis_row in zip(dynamic_rows, strict_rows, visual_rows, handle_iterator):
        raw = dynamic.get("raw_points_ground_mm") or dynamic.get("points_ground_mm") or {}
        strict_joint_rows = joint_rows(strict)
        visual_joint_rows = joint_rows(visual)
        dynamic_provenance = dynamic.get("point_provenance") or {}
        if raw:
            points = {int(index): np.asarray(value, dtype=np.float64) for index, value in raw.items()}
        else:
            camera_points = {}
            for index, item in visual_joint_rows.items():
                xyz = np.asarray(item.get("xyz", []), dtype=np.float64)
                if xyz.shape == (3,) and np.all(np.isfinite(xyz)):
                    camera_points[index] = xyz
            for index, item in strict_joint_rows.items():
                xyz = np.asarray(item.get("xyz", []), dtype=np.float64)
                if bool(item.get("valid")) and xyz.shape == (3,) and np.all(np.isfinite(xyz)):
                    camera_points[index] = xyz
            points = {index: transform(point, dynamic) for index, point in camera_points.items()}
            transformed_from_camera_count += 1
        if not dynamic_provenance:
            dynamic_provenance = {}
            for index in sorted(points):
                strict_item = strict_joint_rows.get(index)
                visual_item = visual_joint_rows.get(index)
                strict_valid = bool(strict_item and strict_item.get("valid"))
                selected = strict_item if strict_valid else visual_item
                dynamic_provenance[str(index)] = {
                    "source": "strict_stereo" if strict_valid else "force_all_stereo_display",
                    "strict_valid": strict_valid,
                    "strict_rejection_reason": None if strict_item is None else strict_item.get("reason"),
                    "quality_flags": [] if selected is None else list(selected.get("quality_flags") or []),
                    "reprojection_error_mean_px": (
                        None if selected is None else selected.get("reprojection_error_mean_px")
                    ),
                    "used_for_pose": bool(index in (15, 16) and strict_valid and dynamic["stage"] == STAGE2),
                }
        errors = {}
        for index, point in visual_joint_rows.items():
            provenance = dynamic_provenance.get(str(index)) or {}
            value = provenance.get("reprojection_error_mean_px", point.get("reprojection_error_mean_px"))
            if value is not None:
                errors[index] = float(value)
        world_handles = {}
        if handle_axis_row is not None:
            if int(handle_axis_row["pair_id"]) != int(dynamic["pair_id"]):
                raise ValueError("misaligned handle axis pair id")
            for side in ("left", "right"):
                candidate = (handle_axis_row.get("handles") or {}).get(side) or {}
                if candidate.get("status") != "candidate":
                    continue
                start = transform(np.asarray(candidate["start_left_camera_mm"]), dynamic)
                end = transform(np.asarray(candidate["end_left_camera_mm"]), dynamic)
                world_handles[side] = {"start": start, "end": end, "evidence": candidate}
        else:
            for side, proxy in proxies.items():
                start = transform(np.asarray(proxy["start_left_camera_mm"]), dynamic)
                end = transform(np.asarray(proxy["end_left_camera_mm"]), dynamic)
                world_handles[side] = {"start": start, "end": end, "evidence": proxy}
        interaction = {"pair_id": int(dynamic["pair_id"]), "stage": dynamic["stage"], "hands": {}, "feet": {}}
        for side in ("left", "right"):
            wrist_index = SIDES[side]["wrist"]
            wrist = points.get(wrist_index)
            reprojection = errors.get(wrist_index)
            if side not in world_handles:
                hand = {"status": "unavailable", "reason": "current_frame_handle_axis_unavailable"}
                states[side]["handle_axis_unavailable"] += 1
            elif wrist is None:
                hand = {"status": "unavailable", "reason": "wrist_3d_missing"}
            else:
                distance, closest, fraction = point_segment_distance_mm(
                    wrist, world_handles[side]["start"], world_handles[side]["end"]
                )
                quality_ok = reprojection is not None and reprojection <= args.maximum_audit_reprojection_error_px
                state = "near_handle_axis_candidate" if quality_ok and distance <= args.near_handle_axis_mm else "not_supported"
                hand = {
                    "status": "diagnostic_candidate", "state": state,
                    "wrist_to_handle_axis_distance_mm": distance,
                    "closest_proxy_point_world_mm": closest.tolist(),
                    "closest_segment_fraction": fraction,
                    "wrist_reprojection_error_mean_px": reprojection,
                    "quality_gate_passed": quality_ok,
                    "handle_geometry_source": world_handles[side]["evidence"].get("source"),
                    "contact_conclusion": (
                        "visual_proximity_candidate" if handle_axis_row is not None else "unavailable_not_independent"
                    ),
                    "contact_blocker": (
                        "no tactile/load sensing; visual proximity is not physical contact truth"
                        if handle_axis_row is not None else
                        "wrist-derived handle proxy; no independent bilateral handle-pixel support or tactile/load sensing"
                    ),
                }
                values[side]["hand_distance"].append(distance)
                states[side][state] += 1
            interaction["hands"][side] = hand

            ankle_index = SIDES[side]["ankle"]
            ankle = points.get(ankle_index)
            ankle_error = errors.get(ankle_index)
            if ankle is None:
                foot = {"status": "unavailable", "reason": "ankle_3d_missing"}
            else:
                height = float(ankle[2])
                quality_ok = ankle_error is not None and ankle_error <= args.maximum_audit_reprojection_error_px
                state = "near_ground_ankle_candidate" if quality_ok and abs(height) <= args.near_ground_ankle_mm else "not_supported"
                foot = {
                    "status": "diagnostic_candidate", "state": state,
                    "ankle_signed_height_above_ground_mm": height,
                    "ankle_reprojection_error_mean_px": ankle_error,
                    "quality_gate_passed": quality_ok,
                    "contact_conclusion": "unavailable_ankle_is_not_shoe_sole",
                    "circular_dependency": dynamic["stage"] == STAGE2,
                    "circular_dependency_reason": (
                        "Stage-2 camera XYZ uses stationary strict feet" if dynamic["stage"] == STAGE2 else None
                    ),
                }
                values[side]["ankle_height"].append(height)
                states[side][f"ground_{state}"] += 1
            interaction["feet"][side] = foot
        interaction["per_keypoint_reprojection_error_mean_px"] = {
            str(index): value for index, value in errors.items()
        }
        interaction["per_keypoint_provenance"] = dynamic_provenance
        interaction["interpretation"] = (
            "Current-frame unsmoothed visual geometry. "
            + (
                "Hand distances use bilateral current-frame image-derived handle axes and are visual proximity candidates only. "
                if handle_axis_row is not None else
                "Hand distances use a wrist-supported proxy and are not independent contact evidence. "
            )
            + "Ankle height is not shoe-sole contact; Stage-2 height is circular with the foot constraint."
        )
        visual_points.append(points)
        errors_by_frame.append(errors)
        handles_by_frame.append(world_handles)
        audits.append(interaction)

    audit_path = output / "interaction_distance_records.jsonl"
    audit_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in audits), encoding="utf-8"
    )
    all_geometry = [point for frame in visual_points for point in frame.values()]
    all_geometry += [
        handle[key]
        for frame in handles_by_frame for handle in frame.values() for key in ("start", "end")
    ]
    cloud = np.asarray(all_geometry)
    xlim = tuple((np.percentile(cloud[:, 0], [0.5, 99.5]) + [-220, 220]).tolist())
    ylim = tuple((np.percentile(cloud[:, 1], [0.5, 99.5]) + [-220, 220]).tolist())
    zlim = (min(-80.0, float(np.percentile(cloud[:, 2], 0.5) - 100)), max(1600.0, float(np.percentile(cloud[:, 2], 99.5) + 140)))
    camera_xyz = np.asarray([row["translation_world_from_left_camera_mm"] for row in dynamic_rows])
    azimuth = choose_azimuth(camera_xyz)

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#92989D", "axes.linewidth": 0.7})
    figure = plt.figure(figsize=(8.0, 6.0), dpi=120, facecolor="white")
    # Matplotlib's automatic 3D depth sorting can place a translucent ground
    # surface or an axis pane in front of joints even when the joint has a
    # larger world Z value.  The audit video must keep measured human geometry
    # readable, so use explicit painter layers instead of computed 3D z-order.
    axis = figure.add_subplot(111, projection="3d", computed_zorder=False)
    figure.subplots_adjust(left=0.02, right=0.96, bottom=0.055, top=0.93)
    video_path = output / "raw_visual_human_partial_handles_ground.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (960, 720))
    if not writer.isOpened():
        raise RuntimeError("failed to open video writer")
    started = time.perf_counter()
    try:
        for frame_index, (dynamic, points, errors, handles, audit) in enumerate(
            zip(dynamic_rows, visual_points, errors_by_frame, handles_by_frame, audits)
        ):
            axis.cla()
            axis.set_axisbelow(True)
            gx, gy = np.linspace(*xlim, 13), np.linspace(*ylim, 13)
            xx, yy = np.meshgrid(gx, gy)
            axis.plot_surface(
                xx, yy, np.zeros_like(xx), color="#E7EAED", alpha=0.72,
                linewidth=0, shade=False, zorder=0,
            )
            for value in gx:
                axis.plot(
                    [value, value], ylim, [0, 0], color="#AEB5BB",
                    linewidth=0.55, alpha=0.7, linestyle="-", zorder=1,
                )
            for value in gy:
                axis.plot(
                    xlim, [value, value], [0, 0], color="#AEB5BB",
                    linewidth=0.55, alpha=0.7, linestyle="-", zorder=1,
                )
            axis.plot(
                [xlim[0], xlim[1], xlim[1], xlim[0], xlim[0]],
                [ylim[0], ylim[0], ylim[1], ylim[1], ylim[0]],
                [0, 0, 0, 0, 0],
                color="#858D94", linewidth=0.9, linestyle="-", zorder=2,
            )
            for start_index, end_index in EDGES:
                if start_index in points and end_index in points:
                    segment = np.vstack((points[start_index], points[end_index]))
                    axis.plot(
                        *segment.T, color="#252525", linewidth=2.5,
                        linestyle="-", solid_capstyle="round", zorder=100,
                    )
            for index, point in points.items():
                axis.scatter(
                    *point, s=34, color=error_color(errors.get(index)),
                    edgecolors="#202020", linewidths=0.6, depthshade=False, zorder=110,
                )
            for side, color in (("left", "#00A6D6"), ("right", "#CC79A7")):
                if side not in handles:
                    continue
                hand = audit["hands"][side]
                if handle_axis_rows is not None and hand.get("state") != "near_handle_axis_candidate":
                    continue
                handle = handles[side]
                segment = np.vstack((handle["start"], handle["end"]))
                axis.plot(
                    *segment.T, color=color, linewidth=7.0, linestyle="-",
                    solid_capstyle="round", zorder=80,
                )
                wrist = points.get(SIDES[side]["wrist"])
                closest = hand.get("closest_proxy_point_world_mm")
                if wrist is not None and closest is not None:
                    connector = np.vstack((wrist, np.asarray(closest)))
                    axis.plot(
                        *connector.T, color=color, linewidth=1.2,
                        linestyle="-", alpha=0.75, zorder=85,
                    )
            # Reuse the former ankle ground projections and camera projection.
            history_start = max(0, frame_index - 119)
            for joint_index, color in ((15, "#0072B2"), (16, "#D55E00")):
                ankle_trail = [frame[joint_index] for frame in visual_points[history_start : frame_index + 1] if joint_index in frame]
                if ankle_trail:
                    ankle_trail = np.asarray(ankle_trail)
                    axis.plot(
                        ankle_trail[:, 0], ankle_trail[:, 1], np.full(len(ankle_trail), 7.0),
                        color=color, linewidth=2.0, linestyle="-", alpha=0.72, zorder=20,
                    )
                    current_ankle = points.get(joint_index)
                    if current_ankle is not None:
                        axis.plot(
                            [current_ankle[0], current_ankle[0]], [current_ankle[1], current_ankle[1]],
                            [0.0, current_ankle[2]], color=color, linewidth=0.8,
                            linestyle="-", alpha=0.48, zorder=25,
                        )
                        axis.scatter(
                            current_ankle[0], current_ankle[1], 7.0,
                            color=color, s=20, depthshade=False, zorder=30,
                        )
            camera_history = camera_xyz[: frame_index + 1]
            if frame_index >= 1:
                axis.plot(
                    *camera_history.T, color="#2E8B57", linewidth=1.5,
                    linestyle="-", alpha=0.65, zorder=35,
                )
                axis.plot(
                    camera_history[:, 0], camera_history[:, 1], np.full(len(camera_history), 7.0),
                    color="#2E8B57", linewidth=1.4, linestyle="-", alpha=0.55, zorder=30,
                )
            current_camera = camera_xyz[frame_index]
            axis.plot(
                [current_camera[0], current_camera[0]], [current_camera[1], current_camera[1]],
                [0.0, current_camera[2]], color="#2E8B57", linewidth=0.8,
                linestyle="-", alpha=0.48, zorder=35,
            )
            axis.scatter(
                *current_camera, marker="^", color="#2E8B57",
                s=45, depthshade=False, zorder=40,
            )
            axis.scatter(
                current_camera[0], current_camera[1], 7.0, marker="^",
                color="#2E8B57", s=24, depthshade=False, zorder=35,
            )
            axis.set(xlim=xlim, ylim=ylim, zlim=zlim, xlabel="Ground X (mm)", ylabel="Ground Y (mm)", zlabel="Height (mm)")
            # Reuse the reference video's native complete-corner coordinate
            # style: two vertical panes plus the single explicit z=0 ground.
            axis.xaxis.pane.set_visible(True)
            axis.yaxis.pane.set_visible(True)
            axis.zaxis.pane.set_visible(False)
            axis.xaxis.pane.set_facecolor((0.97, 0.97, 0.97, 0.45))
            axis.yaxis.pane.set_facecolor((0.97, 0.97, 0.97, 0.45))
            axis.grid(True, linewidth=0.5, alpha=0.45)
            axis.tick_params(labelsize=8, pad=1)
            axis.view_init(elev=23, azim=azimuth)
            axis.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
            handle_title = "image-derived handle axes" if handle_axis_rows is not None else "partial handle proxies"
            axis.set_title(f"Raw current-frame geometry + {handle_title} | pair {dynamic['pair_id']:04d}", fontsize=11.5, pad=13)
            axis.text2D(0.02, 0.965, STAGE_NAMES.get(dynamic["stage"], dynamic["stage"]), transform=axis.transAxes, fontsize=9, fontweight="bold")
            left_hand = audit["hands"]["left"]
            right_hand = audit["hands"]["right"]
            left_foot = audit["feet"]["left"]
            right_foot = audit["feet"]["right"]
            def value(record, key):
                item = record.get(key)
                return "NA" if item is None else f"{item:.1f}"
            axis.text2D(
                0.02, 0.925,
                f"wrist-to-handle-axis L/R: {value(left_hand, 'wrist_to_handle_axis_distance_mm')} / {value(right_hand, 'wrist_to_handle_axis_distance_mm')} mm",
                transform=axis.transAxes, fontsize=8.3, color="#555555",
            )
            axis.text2D(
                0.02, 0.890,
                f"raw ankle height L/R: {value(left_foot, 'ankle_signed_height_above_ground_mm')} / {value(right_foot, 'ankle_signed_height_above_ground_mm')} mm",
                transform=axis.transAxes, fontsize=8.3, color="#555555",
            )
            provenance_rows = audit.get("per_keypoint_provenance") or {}
            strict_count = sum(item.get("source") == "strict_stereo" for item in provenance_rows.values())
            display_count = sum(item.get("source") == "force_all_stereo_display" for item in provenance_rows.values())
            axis.text2D(
                0.02, 0.855,
                f"point source: strict {strict_count} / display-only {display_count}",
                transform=axis.transAxes, fontsize=8.0, color="#555555",
            )
            axis.text2D(
                0.98, 0.915,
                "single z=0 ground + side coordinates\n"
                "ankle projections: blue / orange\n"
                "camera + ground projection: green\n"
                "point error: green <=5, amber <=10, red >10 px",
                transform=axis.transAxes, ha="right", va="top", fontsize=6.9, color="#555555",
            )
            figure.canvas.draw()
            rgba = np.asarray(figure.canvas.buffer_rgba())
            writer.write(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR))
    finally:
        writer.release()
        plt.close(figure)

    summary_values = {}
    for side in ("left", "right"):
        hand = np.asarray(values[side]["hand_distance"], dtype=np.float64)
        ankle = np.asarray(values[side]["ankle_height"], dtype=np.float64)
        summary_values[side] = {
            "wrist_to_handle_axis_distance_mm": {
                "count": len(hand),
                "median": float(np.median(hand)) if len(hand) else None,
                "p95": float(np.percentile(hand, 95)) if len(hand) else None,
            },
            "raw_ankle_signed_height_mm": {
                "count": len(ankle), "median": float(np.median(ankle)),
                "p05_p95": [float(np.percentile(ankle, 5)), float(np.percentile(ankle, 95))],
            },
            "state_counts": dict(states[side]),
        }
    summary = {
        "schema_version": "raw_visual_handle_interaction_audit_v2",
        "frame_count": len(dynamic_rows), "fps": args.fps, "display_smoothing": "none",
        "line_style": "solid_only", "ground_display": "reference_style_single_z0_ground_with_native_two_wall_corner",
        "projection_display": "camera_3d_and_ground_projection_plus_left_right_ankle_ground_trails",
        "human_geometry": (
            "current_frame_visual_stereo_transformed_by_exact_realtime_ground_pose"
            if transformed_from_camera_count == len(dynamic_rows) else
            "current_frame_raw_visual_stereo_transformed_to_ground"
        ),
        "keypoint_quality_encoding": "per-keypoint reprojection error retained in JSONL and point color",
        "keypoint_provenance": "strict/display source, strict rejection reason, quality flags and pose-use flag retained per point when supplied",
        "partial_handle_geometry": (
            "current-frame bilateral image-derived stereo axis" if handle_axis_rows is not None
            else "wrist-supported proxy; not independent walker reconstruction"
        ),
        "distance_summary": summary_values,
        "hand_contact_conclusion": (
            "visual proximity candidates available on frames with bilateral image-derived handle axes; tactile/load truth unavailable"
            if handle_axis_rows is not None else
            "unavailable: proxy is wrist-derived and no independent bilateral handle-pixel/tactile support exists"
        ),
        "ground_contact_conclusion": "unavailable: COCO ankle is not shoe sole; Stage-2 camera pose uses foot anchors",
        "video": str(video_path.resolve()), "interaction_records": str(audit_path.resolve()),
        "render_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
