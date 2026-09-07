#!/usr/bin/env python3
"""Compose frozen C3 left/right 2-D pose panels with complete 3-D estimates.

The two 2-D inputs are existing qualitative videos.  Their right halves are
the saved-prediction PMPose panels.  The 3-D panel is rendered from the
corresponding complete-estimation JSONL without changing any coordinates or
estimate-source labels.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


EDGES = (
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 1), (0, 2), (1, 3),
    (2, 4), (0, 5), (0, 6),
)

SOURCE_COLORS = {
    "stereo_raw": (55, 220, 55),
    "single_view_temporal": (0, 170, 255),
    "temporal": (255, 210, 40),
    "unavailable": (115, 115, 115),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-2d-video", type=Path, required=True)
    parser.add_argument("--right-2d-video", type=Path, required=True)
    parser.add_argument("--complete-3d-jsonl", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--keyframe-sheet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--display-scale-px-per-mm", type=float, default=0.64)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    pair_ids = [int(row["pair_id"]) for row in rows]
    if pair_ids != list(range(len(rows))):
        raise ValueError(f"pair_id must be continuous 0..N-1: {path}")
    return rows


def source_group(source: str) -> str:
    if source == "stereo_raw":
        return "stereo_raw"
    if source.startswith("single_view_temporal_"):
        return "single_view_temporal"
    if source.startswith("temporal_"):
        return "temporal"
    return "unavailable"


def outlined_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def crop_pose_panel(frame: np.ndarray, pair_id: int, side: str) -> np.ndarray:
    height, width = frame.shape[:2]
    panel = frame[:, width // 2:].copy()
    outlined_text(panel, f"{side.upper()} | pair {pair_id:03d}", (12, height - 18), 0.62)
    return panel


def pelvis_center(points: list[dict]) -> np.ndarray:
    hips = [points[index].get("xyz") for index in (11, 12) if points[index].get("has_estimate")]
    if hips:
        return np.mean(np.asarray(hips, dtype=np.float64), axis=0)
    available = [item["xyz"] for item in points if item.get("has_estimate")]
    return np.mean(np.asarray(available, dtype=np.float64), axis=0) if available else np.zeros(3, dtype=np.float64)


def render_3d_panel(record: dict, width: int, height: int, scale: float) -> tuple[np.ndarray, Counter]:
    points = record["keypoints_3d_estimated"]
    model_label = {"pmpose": "PMPose", "probpose": "ProbPose"}.get(
        str(record.get("model", "")).lower(), str(record.get("model", "pose")).title()
    )
    counts = Counter(source_group(item.get("estimate_source", "")) for item in points)
    pelvis = pelvis_center(points)
    # Reorder the left-camera axes only for display: Y=lateral, Z=depth,
    # X=upright.  Labels retain the original coordinate names.
    rel: dict[int, np.ndarray] = {}
    visible: dict[int, bool] = {}
    lateral_limit, depth_limit, upright_limit = 520.0, 620.0, 920.0
    for item in points:
        index = int(item["index"])
        if not item.get("has_estimate"):
            continue
        xyz = np.asarray(item["xyz"], dtype=np.float64) - pelvis
        rel[index] = xyz
        visible[index] = bool(
            abs(xyz[1]) <= lateral_limit
            and abs(xyz[2]) <= depth_limit
            and abs(xyz[0]) <= upright_limit
        )

    off_view = sum(bool(item.get("has_estimate")) and not visible.get(int(item["index"]), False) for item in points)
    counts["off_view"] = off_view

    def rgb(group: str) -> tuple[float, float, float]:
        b, g, r = SOURCE_COLORS[group]
        return r / 255.0, g / 255.0, b / 255.0

    figure = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100, facecolor="#111111")
    axis_3d = figure.add_subplot(111, projection="3d")
    axis_3d.set_position([-0.06, 0.12, 1.12, 0.72])
    axis_3d.set_facecolor("#111111")
    axis_3d.tick_params(colors="#c8c8c8", labelsize=8)

    axis_3d.set_proj_type("persp", focal_length=0.85)
    axis_3d.view_init(elev=17, azim=-53)
    axis_3d.set_xlim(-lateral_limit, lateral_limit)
    axis_3d.set_ylim(-depth_limit, depth_limit)
    axis_3d.set_zlim(-upright_limit, upright_limit)
    axis_3d.set_box_aspect((2 * lateral_limit, 2 * depth_limit, 2 * upright_limit))
    axis_3d.set_xlabel("Y lateral (mm)", color="#dddddd", fontsize=8, labelpad=2)
    axis_3d.set_ylabel("Z depth (mm)", color="#dddddd", fontsize=8, labelpad=2)
    axis_3d.set_zlabel("X upright (mm)", color="#dddddd", fontsize=8, labelpad=2)
    axis_3d.grid(True, color="#555555", alpha=0.55)
    axis_3d.xaxis.pane.set_facecolor((0.08, 0.08, 0.08, 1.0))
    axis_3d.yaxis.pane.set_facecolor((0.08, 0.08, 0.08, 1.0))
    axis_3d.zaxis.pane.set_facecolor((0.08, 0.08, 0.08, 1.0))

    for a, b in EDGES:
        if not visible.get(a) or not visible.get(b):
            continue
        ga = source_group(points[a].get("estimate_source", ""))
        gb = source_group(points[b].get("estimate_source", ""))
        group = "stereo_raw" if ga == gb == "stereo_raw" else ("temporal" if "temporal" in (ga, gb) else "single_view_temporal")
        pa, pb = rel[a], rel[b]
        axis_3d.plot([pa[1], pb[1]], [pa[2], pb[2]], [pa[0], pb[0]], color=rgb(group), linewidth=2.2)

    for item in points:
        index = int(item["index"])
        if not visible.get(index):
            continue
        group = source_group(item.get("estimate_source", ""))
        xyz = rel[index]
        axis_3d.scatter(xyz[1], xyz[2], xyz[0], s=28, c=[rgb(group)], edgecolors="white", linewidths=0.45, depthshade=True)

    figure.suptitle(f"{model_label} complete 3-D | pair {record['pair_id']:03d} | fixed camera", color="white", fontsize=11, y=0.975)
    figure.text(0.5, 0.925, "Perspective oblique view | left-camera frame | pelvis-centered", ha="center", color="#cccccc", fontsize=8)
    figure.text(
        0.5,
        0.895,
        f"direct {counts['stereo_raw']} | single+time {counts['single_view_temporal']} | time {counts['temporal']} | unavailable {counts['unavailable']} | off-view {off_view}",
        ha="center",
        color="#cccccc",
        fontsize=7.2,
    )
    handles = [
        Line2D([0], [0], marker="o", color=rgb("stereo_raw"), label="direct stereo", markersize=5),
        Line2D([0], [0], marker="o", color=rgb("single_view_temporal"), label="single-view + time", markersize=5),
        Line2D([0], [0], marker="o", color=rgb("temporal"), label="time only", markersize=5),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, labelcolor="#dddddd", fontsize=7, bbox_to_anchor=(0.5, 0.012))
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    canvas = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    plt.close(figure)
    if canvas.shape[:2] != (height, width):
        canvas = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
    return canvas, counts


def save_keyframe_sheet(path: Path, frames: list[np.ndarray], pair_ids: list[int]) -> None:
    thumbs = []
    for frame, pair_id in zip(frames, pair_ids):
        target_width = 900
        target_height = round(frame.shape[0] * target_width / frame.shape[1])
        thumb = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        outlined_text(thumb, f"pair {pair_id:03d}", (12, target_height - 14), 0.6)
        thumbs.append(thumb)
    rows = []
    for index in range(0, len(thumbs), 2):
        row = thumbs[index:index + 2]
        if len(row) == 1:
            row.append(np.zeros_like(row[0]))
        rows.append(np.hstack(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.vstack(rows)):
        raise RuntimeError(f"Cannot write keyframe sheet: {path}")


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.complete_3d_jsonl)
    left_cap = cv2.VideoCapture(str(args.left_2d_video))
    right_cap = cv2.VideoCapture(str(args.right_2d_video))
    if not left_cap.isOpened() or not right_cap.isOpened():
        raise RuntimeError("Cannot open one or both 2-D videos")
    left_frames = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    right_frames = int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if left_frames != right_frames or left_frames != len(records):
        raise ValueError(f"Frame mismatch: left={left_frames}, right={right_frames}, 3d={len(records)}")
    source_fps = float(left_cap.get(cv2.CAP_PROP_FPS)) or 15.0
    fps = args.fps or source_fps

    ok_left, left_first = left_cap.read()
    ok_right, right_first = right_cap.read()
    if not ok_left or not ok_right:
        raise RuntimeError("Cannot read first 2-D video frame")
    left_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    right_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    panel_width = left_first.shape[1] - left_first.shape[1] // 2
    panel_height = left_first.shape[0]
    output_size = (panel_width * 3, panel_height)
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {args.output_video}")

    keyframe_ids = sorted({0, len(records) // 5, 2 * len(records) // 5, 3 * len(records) // 5, 4 * len(records) // 5, len(records) - 1})
    keyframes: list[np.ndarray] = []
    aggregate = Counter()
    try:
        for pair_id, record in enumerate(records):
            ok_left, left_frame = left_cap.read()
            ok_right, right_frame = right_cap.read()
            if not ok_left or not ok_right:
                raise RuntimeError(f"Cannot read 2-D frame {pair_id}")
            left_panel = crop_pose_panel(left_frame, pair_id, "left")
            right_panel = crop_pose_panel(right_frame, pair_id, "right")
            panel_3d, counts = render_3d_panel(record, panel_width, panel_height, args.display_scale_px_per_mm)
            aggregate.update(counts)
            composed = np.hstack((left_panel, right_panel, panel_3d))
            writer.write(composed)
            if pair_id in keyframe_ids:
                keyframes.append(composed)
    finally:
        writer.release()
        left_cap.release()
        right_cap.release()

    save_keyframe_sheet(args.keyframe_sheet, keyframes, keyframe_ids)
    manifest = {
        "scope": f"saved C3 {records[0].get('model', 'pose')} 2-D visualization plus saved complete 3-D estimates; no model rerun",
        "model": records[0].get("model"),
        "left_2d_video": str(args.left_2d_video.resolve()),
        "right_2d_video": str(args.right_2d_video.resolve()),
        "complete_3d_jsonl": str(args.complete_3d_jsonl.resolve()),
        "output_video": str(args.output_video.resolve()),
        "keyframe_sheet": str(args.keyframe_sheet.resolve()),
        "frames": len(records),
        "fps": fps,
        "duration_sec": len(records) / fps,
        "resolution": {"width": output_size[0], "height": output_size[1]},
        "3d_coordinate_frame": "left_camera",
        "3d_length_unit": "mm",
        "3d_view": "fixed perspective oblique view only; pelvis-centered; no side-view inset",
        "3d_source_colors_bgr": SOURCE_COLORS,
        "aggregate_frame_point_counts": dict(aggregate),
        "interpretation": "Estimate provenance is shown explicitly. The video is not a 2-D or 3-D accuracy validation.",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
