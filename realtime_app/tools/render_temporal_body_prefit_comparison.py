#!/usr/bin/env python3
"""Render raw observations beside the contact-aware temporal body prefit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COCO_EDGES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 1),
    (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
)


STAGE_LABELS = {
    "stage1_walker_static_human_moving": "STAGE 1  HUMAN MOVING / WALKER STATIC",
    "stage2_feet_static_walker_moving": "STAGE 2  FEET STATIC / WALKER MOVING",
    "transition": "TRANSITION",
    "warming_up": "WARMING UP",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def transform(point: np.ndarray, frame: dict) -> np.ndarray:
    rotation = np.asarray(frame["rotation_world_from_left"], dtype=np.float64)
    translation = np.asarray(frame["translation_world_from_left_mm"], dtype=np.float64)
    return rotation @ point + translation


def draw_scene(
    axis,
    points: np.ndarray,
    walker: dict[str, np.ndarray],
    model: dict,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
    azimuth: float,
    title: str,
    raw_ghost: np.ndarray | None = None,
    contacts: dict | None = None,
) -> None:
    axis.cla()
    gx, gy = np.linspace(*xlim, 11), np.linspace(*ylim, 11)
    for value in gx:
        axis.plot([value, value], ylim, [0, 0], color="#C5CCD1", linewidth=0.45, alpha=0.7)
    for value in gy:
        axis.plot(xlim, [value, value], [0, 0], color="#C5CCD1", linewidth=0.45, alpha=0.7)
    if raw_ghost is not None:
        for start, end in COCO_EDGES:
            if np.isfinite(raw_ghost[[start, end]]).all():
                segment = raw_ghost[[start, end]]
                axis.plot(*segment.T, color="#A8ADB2", linewidth=1.0, alpha=0.35)
    for start, end in COCO_EDGES:
        if np.isfinite(points[[start, end]]).all():
            segment = points[[start, end]]
            axis.plot(*segment.T, color="#202428", linewidth=2.3, solid_capstyle="round")
    axis.scatter(
        points[:, 0], points[:, 1], points[:, 2], s=22,
        color="#0072B2" if raw_ghost is not None else "#666B70",
        edgecolors="#202428", linewidths=0.35, depthshade=False,
    )
    handle_edges = {tuple(edge) for edge in model["handle_segments"].values()}
    for start_name, end_name in model["edges"]:
        segment = np.vstack((walker[start_name], walker[end_name]))
        is_handle = (start_name, end_name) in handle_edges or (end_name, start_name) in handle_edges
        axis.plot(
            *segment.T, color="#CC79A7" if is_handle else "#00A6D6",
            linewidth=4.5 if is_handle else 2.5, solid_capstyle="round", alpha=0.85,
        )
    if contacts:
        for side, joint, color in (("left", 15, "#009E73"), ("right", 16, "#D55E00")):
            if contacts["foot_contact_regularizer_active"][side]:
                axis.scatter(*points[joint], s=70, facecolors="none", edgecolors=color, linewidths=1.5, depthshade=False)
        for side, joint, color in (("left", 9, "#009E73"), ("right", 10, "#7B61A8")):
            if contacts["hand_contact_regularizer_active"][side]:
                axis.scatter(*points[joint], s=70, facecolors="none", edgecolors=color, linewidths=1.5, depthshade=False)
    axis.set(xlim=xlim, ylim=ylim, zlim=zlim, xlabel="Ground X (mm)", ylabel="Ground Y (mm)", zlabel="Height (mm)")
    axis.view_init(elev=23, azim=azimuth)
    axis.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    axis.tick_params(labelsize=7, pad=1)
    axis.set_title(title, fontsize=10, pad=9)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-json", type=Path, required=True)
    parser.add_argument("--prefit-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    motion = json.loads(args.motion_json.read_text(encoding="utf-8-sig"))
    fitted_rows = read_jsonl(args.prefit_jsonl)
    if len(motion["frames"]) != len(fitted_rows):
        raise ValueError("motion and prefit streams must align")
    raw = np.asarray([
        [point if point is not None else [np.nan, np.nan, np.nan] for point in frame["points_mm"]]
        for frame in motion["frames"]
    ], dtype=np.float64)
    fitted = np.asarray([
        [row["points_ground_mm"][str(joint)] for joint in range(17)]
        for row in fitted_rows
    ], dtype=np.float64)
    finite = np.vstack((raw[np.isfinite(raw).all(axis=2)], fitted.reshape(-1, 3)))
    xlim = tuple((np.percentile(finite[:, 0], [0.5, 99.5]) + [-180, 180]).tolist())
    ylim = tuple((np.percentile(finite[:, 1], [0.5, 99.5]) + [-180, 180]).tolist())
    zlim = (min(-60.0, float(np.percentile(finite[:, 2], 0.5) - 80)), max(1550.0, float(np.percentile(finite[:, 2], 99.5) + 100)))
    camera_path = np.asarray([frame["translation_world_from_left_mm"] for frame in motion["frames"]])
    direction = camera_path[-1, :2] - camera_path[0, :2]
    azimuth = float(np.degrees(np.arctan2(direction[1], direction[0])) - 35.0)

    figure = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="white")
    left_axis = figure.add_subplot(121, projection="3d", computed_zorder=False)
    right_axis = figure.add_subplot(122, projection="3d", computed_zorder=False)
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.05, top=0.90, wspace=0.02)
    video = output / "raw_vs_temporal_body_prefit.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError("failed to open video writer")
    try:
        nodes_left = {name: np.asarray(point) for name, point in motion["walker"]["nodes_left_camera_mm"].items()}
        for index, (frame, fitted_row) in enumerate(zip(motion["frames"], fitted_rows)):
            walker = {name: transform(point, frame) for name, point in nodes_left.items()}
            draw_scene(left_axis, raw[index], walker, motion["walker"], xlim, ylim, zlim, azimuth, "Raw current-frame stereo observations")
            draw_scene(
                right_axis, fitted[index], walker, motion["walker"], xlim, ylim, zlim, azimuth,
                "Fixed-bone temporal prefit (raw ghost in gray)", raw_ghost=raw[index], contacts=fitted_row,
            )
            figure.suptitle(
                f"{STAGE_LABELS.get(frame['stage'], frame['stage'])}  |  pair {frame['pair']:04d}  |  circles = active soft contact regularizer",
                fontsize=11, y=0.965,
            )
            figure.canvas.draw()
            rgba = np.asarray(figure.canvas.buffer_rgba())
            writer.write(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR))
    finally:
        writer.release()
        plt.close(figure)
    summary = {
        "frame_count": len(fitted_rows), "fps": args.fps, "resolution": [1280, 720],
        "video": str(video), "display_smoothing": "none beyond the explicit temporal body prefit algorithm",
        "interpretation": "Left is raw; right is the optimization result with raw observations retained as a gray audit ghost.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
