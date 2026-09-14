"""Render people_1 / people_1_near upright left/right 2-D + 3-D skeleton.

The saved stereo JSONL ``pair_id`` runs 0..N-1 in "sequence" order, while the
raw capture AVIs contain every captured frame (including frames that never
entered a stereo pair).  This script maps each ``pair_id`` to the correct AVI
frame through the input-selection manifest and the capture frame CSVs, draws
the 2-D skeleton on the raw fisheye frame, then rotates the whole panel (frame
+ skeleton together) to the upright convention (left ``ccw90``, right
``cw90``) so left and right views sit upright side by side.  The 3-D panel is
drawn in the style of ``render_c3_stereo_2d_complete_3d_video.py`` (dark
background, perspective oblique view, reference grid, labelled axes,
colour-coded joints, header counts).

No model is re-run and no 2-D/3-D coordinate, association or triangulation
threshold is modified.  When fed the ``*_ungated_stereo`` JSONL every finite,
in-bounds triangulated point is shown, with points carrying a
``high_reprojection_error`` quality flag drawn in orange and reported in the
header so the "complete" skeleton is not mistaken for a strict-geometry claim.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401
from matplotlib.lines import Line2D  # noqa: E402

# Allow importing the sibling visualization module whose 2-D drawing logic we
# reuse verbatim.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_people1_2d_3d_video as rv  # noqa: E402

# COCO-17 adjacency (same as the sibling module).
EDGES = rv.EDGES

# BGR colours for the 3-D panel.
QUALITY_COLORS_BGR = {
    "clean": (55, 220, 55),      # direct stereo, no quality flag
    "flagged": (0, 170, 255),    # high_reprojection_error flagged
    "fill": (0, 0, 255),         # temporal/single-view time-interpolated fill
    "unavailable": (115, 115, 115),
}


def rgb(key: str) -> tuple[float, float, float]:
    b, g, r = QUALITY_COLORS_BGR[key]
    return r / 255.0, g / 255.0, b / 255.0


def _rotate_code(rotation: str) -> int | None:
    return {
        "none": None,
        "cw90": cv2.ROTATE_90_CLOCKWISE,
        "ccw90": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "180": cv2.ROTATE_180,
    }[rotation]


def rotate_upright(panel: np.ndarray, rotation: str) -> np.ndarray:
    """Rotate a raw 2-D panel (frame + drawn skeleton) to the upright view."""
    code = _rotate_code(rotation)
    return panel if code is None else cv2.rotate(panel, code)


def load_frame_index_map(csv_path: Path) -> dict[int, int]:
    """Return {frame_id: avi_frame_index} from a capture ``*_frames.csv``."""
    mapping: dict[int, int] = {}
    with csv_path.open("r", encoding="utf-8") as fh:
        fh.readline()  # header
        for index, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            frame_id = int(line.split(",")[0])
            mapping[frame_id] = index
    return mapping


def load_pair_frame_ids(manifest_path: Path) -> list[tuple[int, int]]:
    """Return ``(left_frame_id, right_frame_id)`` per sequence index, in order."""
    pairs: list[tuple[int, int]] = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        left_col = header.index("left_frame_id")
        right_col = header.index("right_frame_id")
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            pairs.append((int(parts[left_col]), int(parts[right_col])))
    return pairs


def flagged_count(record: dict) -> int:
    """Number of output 3-D points carrying a non-empty quality flag."""
    total = 0
    for person in record.get("persons_3d", []):
        for kp in person.get("keypoints_3d", []):
            if kp.get("quality_flags"):
                total += 1
    return total


def render_3d_panel_clear(record: dict, width: int, height: int) -> np.ndarray:
    """Render a clear 3-D skeleton from the strict/ungated ``keypoints_3d`` list.

    Pelvis-centred, left-camera frame, display axes reordered to
    (lateral, depth, upright), dark background, perspective oblique view, grid
    and colour-coded joints (green = no quality flag, orange = flagged
    high-reprojection).  Returns a BGR uint8 array sized ``height x width``.
    """
    persons_3d = record.get("persons_3d", [])
    points: list[dict] = []
    if persons_3d:
        for kp in persons_3d[0].get("keypoints_3d", []):
            if kp.get("valid") and kp.get("xyz") is not None:
                points.append({
                    "index": int(kp["index"]),
                    "xyz": np.asarray(kp["xyz"], dtype=np.float64),
                    "flagged": bool(kp.get("quality_flags")),
                    "fill": bool(kp.get("_fill", False)),
                })
    by_index = {p["index"]: p for p in points}

    # Pelvis centre (left hip=11, right hip=12), else mean of available points.
    pelvis = None
    if 11 in by_index and 12 in by_index:
        pelvis = (by_index[11]["xyz"] + by_index[12]["xyz"]) / 2.0
    elif points:
        pelvis = np.mean([p["xyz"] for p in points], axis=0)
    for p in points:
        p["rel"] = (p["xyz"] - pelvis) if pelvis is not None else p["xyz"]

    lateral_limit, depth_limit, upright_limit = 520.0, 620.0, 920.0
    visible = {}
    for p in points:
        r = p["rel"]
        visible[p["index"]] = bool(
            abs(r[1]) <= lateral_limit and abs(r[2]) <= depth_limit and abs(r[0]) <= upright_limit
        )
    off_view = sum(1 for p in points if not visible.get(p["index"], False))

    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100, facecolor="#111111")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([-0.06, 0.12, 1.12, 0.72])
    ax.set_facecolor("#111111")
    ax.tick_params(colors="#c8c8c8", labelsize=8)
    ax.set_proj_type("persp", focal_length=0.85)
    ax.view_init(elev=17, azim=-53)
    ax.set_xlim(-lateral_limit, lateral_limit)
    ax.set_ylim(-depth_limit, depth_limit)
    ax.set_zlim(-upright_limit, upright_limit)
    ax.set_box_aspect((2 * lateral_limit, 2 * depth_limit, 2 * upright_limit))
    ax.set_xlabel("Y lateral (mm)", color="#dddddd", fontsize=8, labelpad=2)
    ax.set_ylabel("Z depth (mm)", color="#dddddd", fontsize=8, labelpad=2)
    ax.set_zlabel("X upright (mm)", color="#dddddd", fontsize=8, labelpad=2)
    ax.grid(True, color="#555555", alpha=0.55)
    ax.xaxis.pane.set_facecolor((0.08, 0.08, 0.08, 1.0))
    ax.yaxis.pane.set_facecolor((0.08, 0.08, 0.08, 1.0))
    ax.zaxis.pane.set_facecolor((0.08, 0.08, 0.08, 1.0))

    def group_of(idx: int) -> str:
        p = by_index[idx]
        if p["fill"]:
            return "fill"
        return "flagged" if p["flagged"] else "clean"

    for a, b in EDGES:
        if a not in visible or b not in visible or not visible[a] or not visible[b]:
            continue
        ga, gb = group_of(a), group_of(b)
        if "fill" in (ga, gb):
            group = "fill"
        elif ga == "flagged" or gb == "flagged":
            group = "flagged"
        else:
            group = "clean"
        pa, pb = by_index[a]["rel"], by_index[b]["rel"]
        ax.plot([pa[1], pb[1]], [pa[2], pb[2]], [pa[0], pb[0]], color=rgb(group), linewidth=2.2)

    for p in points:
        idx = p["index"]
        if not visible.get(idx):
            continue
        group = group_of(idx)
        rel = p["rel"]
        ax.scatter(rel[1], rel[2], rel[0], s=28, c=[rgb(group)], edgecolors="white",
                   linewidths=0.45, depthshade=True)

    left_model = str(record.get("left", {}).get("model_name", "")).upper()
    model = left_model if left_model else "POSE"
    n_clean = sum(1 for p in points if not p["flagged"] and not p["fill"])
    n_flagged = sum(1 for p in points if p["flagged"] and not p["fill"])
    n_fill = sum(1 for p in points if p["fill"])
    fig.suptitle(f"{model} 3-D | pair {record['pair_id']:03d} | pelvis-centred | fixed camera",
                 color="white", fontsize=11, y=0.975)
    fig.text(0.5, 0.925, "Perspective oblique view | left-camera frame (mm) | lateral / depth / upright",
             ha="center", color="#cccccc", fontsize=8)
    fig.text(0.5, 0.895,
             f"visible {len(visible) - off_view}/{max(len(points), 1)} | clean {n_clean} | "
             f"high-reproj {n_flagged} | fill {n_fill} | off-view {off_view}",
             ha="center", color="#cccccc", fontsize=7.4)

    handles = [
        Line2D([0], [0], marker="o", color=rgb("clean"), label="direct stereo", markersize=5),
        Line2D([0], [0], marker="o", color=rgb("flagged"), label="high-reprojection", markersize=5),
    ]
    if n_fill:
        handles.append(Line2D([0], [0], marker="o", color=rgb("fill"), label="time fill", markersize=5))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               labelcolor="#dddddd", fontsize=7, bbox_to_anchor=(0.5, 0.012))
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    canvas = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    if canvas.shape[:2] != (height, width):
        canvas = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
    return canvas


def _merge_fills(
    records: list[dict],
    fills: list[dict],
) -> list[dict]:
    """Merge temporal estimates into geometry records for rendering only.

    Direct finite, positive-depth joints stay as-is.  Missing joints are filled
    from the temporal/single-view estimate and flagged with ``_fill=True`` so the
    renderer draws them red.  No geometry file is modified.
    """
    import copy

    merged_records = []
    for record, fill_row in zip(records, fills):
        new_record = copy.deepcopy(record)
        estimates = {
            int(point["index"]): point
            for point in fill_row.get("keypoints_3d_estimated", [])
        }
        person_blocks = new_record.get("persons_3d") or [{}]
        if not new_record.get("persons_3d"):
            new_record["persons_3d"] = [{
                "left_person_id": 0,
                "right_person_id": 0,
                "valid_keypoints": 0,
                "mean_reprojection_error_px": None,
                "keypoints_3d": [],
            }]
            person_blocks = new_record["persons_3d"]
        existing = {
            int(point.get("index", -1)): point
            for point in person_blocks[0].get("keypoints_3d", [])
        }
        merged = []
        for index in range(17):
            keypoint = existing.get(index)
            keep_direct = (
                keypoint is not None
                and keypoint.get("valid")
                and keypoint.get("xyz") is not None
                and not any(
                    flag == "negative_or_zero_depth"
                    for flag in keypoint.get("quality_flags", [])
                )
            )
            if keep_direct:
                point = dict(keypoint)
                point["_fill"] = False
                merged.append(point)
                continue
            estimate = estimates.get(index)
            if (
                estimate is not None
                and estimate.get("has_estimate")
                and estimate.get("xyz") is not None
                and estimate.get("estimate_source") != "unavailable_no_stereo_anchor"
            ):
                merged.append({
                    "index": index,
                    "name": estimate.get("name", keypoint.get("name", "") if keypoint else ""),
                    "valid": True,
                    "xyz": [float(v) for v in estimate["xyz"]],
                    "score": 0.0,
                    "left_score": None,
                    "right_score": None,
                    "depth_left": None,
                    "depth_right": None,
                    "reprojection_error_left_px": None,
                    "reprojection_error_right_px": None,
                    "reprojection_error_mean_px": None,
                    "reason": None,
                    "quality_flags": [],
                    "estimate_source": estimate.get("estimate_source"),
                    "temporal_anchor_pair_ids": estimate.get("temporal_anchor_pair_ids", []),
                    "_fill": True,
                })
            elif keypoint is not None:
                point = dict(keypoint)
                point["_fill"] = False
                merged.append(point)
        person_blocks[0]["keypoints_3d"] = merged
        person_blocks[0]["valid_keypoints"] = sum(
            1 for point in merged if point.get("valid")
        )
        merged_records.append(new_record)
    return merged_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-video", required=True, type=Path)
    parser.add_argument("--right-video", required=True, type=Path)
    parser.add_argument("--geometry-jsonl", required=True, type=Path)
    parser.add_argument(
        "--fill-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional complete-3D-estimates JSONL (estimate_complete_stereo_3d.py). "
            "When given, joints missing from the geometry JSONL are filled with the "
            "temporal estimate and rendered in red (fill) with red edges."
        ),
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--left-frames-csv", required=True, type=Path)
    parser.add_argument("--right-frames-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--keypoint-threshold", type=float, default=0.25)
    parser.add_argument("--left-rotation", default="ccw90")
    parser.add_argument("--right-rotation", default="cw90")
    parser.add_argument("--col-width", type=int, default=576)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    records = rv.load_records(args.geometry_jsonl)
    frame_ids = load_pair_frame_ids(args.manifest)
    if len(records) != len(frame_ids):
        raise ValueError(
            f"geometry JSONL ({len(records)} records) and manifest "
            f"({len(frame_ids)} rows) disagree"
        )
    if args.fill_jsonl is not None:
        fills = rv.load_records(args.fill_jsonl)
        if len(fills) != len(records):
            raise ValueError(
                f"fill JSONL ({len(fills)} records) and geometry JSONL "
                f"({len(records)} records) disagree"
            )
        records = _merge_fills(records, fills)
    left_map = load_frame_index_map(args.left_frames_csv)
    right_map = load_frame_index_map(args.right_frames_csv)

    count = len(records)
    if args.limit is not None:
        count = min(count, args.limit)

    left_cap = cv2.VideoCapture(str(args.left_video))
    right_cap = cv2.VideoCapture(str(args.right_video))
    if not left_cap.isOpened() or not right_cap.isOpened():
        raise RuntimeError("Cannot open one or both raw stereo videos")

    fps = args.fps or (left_cap.get(cv2.CAP_PROP_FPS) or 30.0)

    l0, r0 = frame_ids[0]
    left_cap.set(cv2.CAP_PROP_POS_FRAMES, left_map[l0])
    right_cap.set(cv2.CAP_PROP_POS_FRAMES, right_map[r0])
    ok_l, frame0_l = left_cap.read()
    ok_r, frame0_r = right_cap.read()
    if not ok_l or not ok_r:
        raise RuntimeError("Cannot read first mapped frame")

    col_w = args.col_width
    # Upright frame aspect (raw 1920x1080 -> upright 1080x1920 -> h = w*1920/1080).
    col_h = int(round(col_w * 1920 / 1080))
    header_h = 40
    canvas_w = col_w * 3
    canvas_h = header_h + col_h
    left_x = 0
    right_x = col_w
    d3_x = 2 * col_w

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (canvas_w, canvas_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create MP4: {args.output}")

    try:
        for idx in range(count):
            left_frame_id, right_frame_id = frame_ids[idx]
            left_cap.set(cv2.CAP_PROP_POS_FRAMES, left_map[left_frame_id])
            right_cap.set(cv2.CAP_PROP_POS_FRAMES, right_map[right_frame_id])
            ok_l, left = left_cap.read()
            ok_r, right = right_cap.read()
            if not ok_l or not ok_r:
                raise RuntimeError(
                    f"Cannot read pair {idx} (left_frame_id={left_frame_id}, "
                    f"right_frame_id={right_frame_id})"
                )

            record = records[idx]

            left_2d = rv.draw_side(left, record, "left", args.keypoint_threshold)
            right_2d = rv.draw_side(right, record, "right", args.keypoint_threshold)
            left_2d = rotate_upright(left_2d, args.left_rotation)
            right_2d = rotate_upright(right_2d, args.right_rotation)
            left_2d = cv2.resize(left_2d, (col_w, col_h), interpolation=cv2.INTER_AREA)
            right_2d = cv2.resize(right_2d, (col_w, col_h), interpolation=cv2.INTER_AREA)

            d3 = render_3d_panel_clear(record, col_w, col_h)

            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            persons_3d = record.get("persons_3d", [])
            flagged = flagged_count(record)
            if persons_3d:
                p3 = persons_3d[0]
                valid = p3.get("valid_keypoints", 0)
                reproj = p3.get("mean_reprojection_error_px")
                reproj_text = f"{reproj:.2f}px" if reproj is not None else "n/a"
                header = (
                    f"pair {record['pair_id']:03d}  valid_3D={valid}/17  "
                    f"reproj={reproj_text}  flagged_high_reproj={flagged}"
                )
            else:
                header = f"pair {record['pair_id']:03d}  association_failed"
            cv2.putText(canvas, header, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "LEFT 2D (upright)", (left_x + 12, header_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(canvas, "RIGHT 2D (upright)", (right_x + 12, header_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(canvas, "3D skeleton (mm)", (d3_x + 12, header_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

            canvas[header_h:canvas_h, left_x:left_x + col_w] = left_2d
            canvas[header_h:canvas_h, right_x:right_x + col_w] = right_2d
            canvas[header_h:canvas_h, d3_x:d3_x + col_w] = d3

            writer.write(canvas)
            if (idx + 1) % 50 == 0:
                print(f"  rendered {idx + 1}/{count}", flush=True)
    finally:
        writer.release()
        left_cap.release()
        right_cap.release()

    print(f"Wrote {args.output}  ({count} frames, {fps} fps, {canvas_w}x{canvas_h})")


if __name__ == "__main__":
    main()
