"""Render people_1 C3 PMPose: left/right 2D skeleton + 3D skeleton in one video.

Reuses drawing logic from visualize_offline_stereo_models.py (2D) and
render_c3_stereo_2d_complete_3d_video.py (3D panel), adapted to the
strict-stereo JSONL format (keypoints_3d as a list with valid/xyz/reason).

Output: one MP4 with three panels (left 2D, right 2D, 3D).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

# COCO-17 edges (same as visualize_offline_stereo_models.py)
EDGES = (
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 1), (0, 2), (1, 3),
    (2, 4), (0, 5), (0, 6),
)

COLORS = {
    "valid": (40, 220, 40),
    "high_reprojection_error": (0, 165, 255),
    "out_of_raw_image_bounds": (255, 0, 255),
    "low_2d_score": (130, 130, 130),
    "negative_depth": (255, 0, 255),
    "association_failed": (255, 255, 0),
    "other": (30, 30, 230),
}


def load_records(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def clip_point(x, y, width, height):
    return (
        int(round(np.clip(x, 0, width - 1))),
        int(round(np.clip(y, 0, height - 1))),
    )


def statuses_for_person(record, side, person_id):
    for stereo_person in record.get("persons_3d", []):
        if stereo_person.get(f"{side}_person_id") != person_id:
            continue
        return {
            int(point["index"]): (
                "valid" if point.get("valid") else point.get("reason") or "other"
            )
            for point in stereo_person.get("keypoints_3d", [])
        }
    return {}


def draw_person(image, person, statuses, threshold, associated):
    height, width = image.shape[:2]
    bbox = person.get("bbox")
    if bbox and len(bbox) == 4:
        x1, y1 = clip_point(bbox[0], bbox[1], width, height)
        x2, y2 = clip_point(bbox[2], bbox[3], width, height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)

    keypoints = person.get("keypoints", [])
    visible = []
    for point in keypoints[:17]:
        if len(point) < 3:
            visible.append(False)
            continue
        x, y, score = map(float, point[:3])
        visible.append(score >= threshold and 0 <= x < width and 0 <= y < height)

    for a, b in EDGES:
        if a >= len(visible) or b >= len(visible) or not visible[a] or not visible[b]:
            continue
        sa = statuses.get(a, "association_failed" if not associated else "other")
        sb = statuses.get(b, "association_failed" if not associated else "other")
        color = COLORS["valid"] if sa == sb == "valid" else COLORS["high_reprojection_error"]
        pa = clip_point(keypoints[a][0], keypoints[a][1], width, height)
        pb = clip_point(keypoints[b][0], keypoints[b][1], width, height)
        cv2.line(image, pa, pb, color, 2, cv2.LINE_AA)

    for index, point in enumerate(keypoints[:17]):
        if len(point) < 3:
            continue
        x, y, score = map(float, point[:3])
        if not (0 <= x < width and 0 <= y < height):
            continue
        if score < threshold:
            color, radius = COLORS["low_2d_score"], 2
        else:
            state = statuses.get(index, "association_failed" if not associated else "other")
            color, radius = COLORS.get(state, COLORS["other"]), 5
        cv2.circle(image, clip_point(x, y, width, height), radius, color, -1, cv2.LINE_AA)


def draw_side(image, record, side, threshold):
    painted = image.copy()
    associated_ids = {
        item[f"{side}_person_id"]
        for item in record.get("persons_3d", [])
        if f"{side}_person_id" in item
    }
    for person in record.get(side, {}).get("persons", []):
        person_id = person.get("person_id")
        draw_person(
            painted, person,
            statuses_for_person(record, side, person_id),
            threshold, person_id in associated_ids,
        )
    return painted


def render_3d_panel(record, width, height):
    """Render 3D skeleton from strict-stereo keypoints_3d list.

    Pelvis-centered, equal aspect, fixed view. Returns BGR uint8 array.
    """
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    persons_3d = record.get("persons_3d", [])
    valid_pts = {}
    if persons_3d:
        for kp in persons_3d[0].get("keypoints_3d", []):
            if kp.get("valid") and kp.get("xyz") is not None:
                valid_pts[int(kp["index"])] = np.array(kp["xyz"], dtype=float)

    # pelvis center (left_hip=11, right_hip=12)
    pelvis = None
    if 11 in valid_pts and 12 in valid_pts:
        pelvis = (valid_pts[11] + valid_pts[12]) / 2.0
    elif valid_pts:
        pelvis = np.mean(list(valid_pts.values()), axis=0)

    if pelvis is not None:
        centered = {i: p - pelvis for i, p in valid_pts.items()}
    else:
        centered = {}

    # draw edges
    for a, b in EDGES:
        if a in centered and b in centered:
            pa, pb = centered[a], centered[b]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                    color="#28dc28", linewidth=2.0)

    # draw joints
    if centered:
        xs = [p[0] for p in centered.values()]
        ys = [p[1] for p in centered.values()]
        zs = [p[2] for p in centered.values()]
        ax.scatter(xs, ys, zs, color="#28dc28", s=30, depthshade=False)

    # fixed view and equal aspect
    ax.view_init(elev=17, azim=-53)
    ax.set_xlabel("X (mm)", fontsize=8)
    ax.set_ylabel("Y (mm)", fontsize=8)
    ax.set_zlabel("Z (mm)", fontsize=8)
    ax.tick_params(labelsize=7)

    # equal aspect with a fixed range so skeleton doesn't jump
    rng = 1200.0
    ax.set_xlim([-rng, rng])
    ax.set_ylim([-rng, rng])
    ax.set_zlim([-rng, rng])
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    valid_count = len(valid_pts)
    ax.set_title(f"3D skeleton  valid={valid_count}/17", fontsize=10)

    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    bgr = cv2.cvtColor(buf[:, :, :3], cv2.COLOR_RGB2BGR)
    return bgr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-video", required=True, type=Path)
    parser.add_argument("--right-video", required=True, type=Path)
    parser.add_argument("--geometry-jsonl", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--keypoint-threshold", type=float, default=0.25)
    parser.add_argument("--panel-width", type=int, default=960)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    records = load_records(args.geometry_jsonl)
    count = len(records)
    if args.limit is not None:
        count = min(count, args.limit)

    left_cap = cv2.VideoCapture(str(args.left_video))
    right_cap = cv2.VideoCapture(str(args.right_video))
    if not left_cap.isOpened() or not right_cap.isOpened():
        raise RuntimeError("Cannot open one or both raw stereo videos")

    fps = left_cap.get(cv2.CAP_PROP_FPS) or 30.0

    # first frame to determine sizes
    ok_l, frame0_l = left_cap.read()
    ok_r, frame0_r = right_cap.read()
    if not ok_l or not ok_r:
        raise RuntimeError("Cannot read first frame")
    left_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    right_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    src_h, src_w = frame0_l.shape[:2]
    pw = args.panel_width
    ph = int(round(pw * src_h / src_w))
    # 3D panel same height as 2D panels, width = 2*pw (full bottom row)
    d3_w = pw * 2
    d3_h = ph

    canvas_w = pw * 2
    canvas_h = ph + d3_h + 40  # header + 2D row + 3D row

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (canvas_w, canvas_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create MP4: {args.output}")

    try:
        for idx in range(count):
            ok_l, left = left_cap.read()
            ok_r, right = right_cap.read()
            if not ok_l or not ok_r:
                raise RuntimeError(f"Cannot read pair {idx}")

            record = records[idx]

            # 2D panels
            left_2d = draw_side(left, record, "left", args.keypoint_threshold)
            right_2d = draw_side(right, record, "right", args.keypoint_threshold)
            left_2d = cv2.resize(left_2d, (pw, ph), interpolation=cv2.INTER_AREA)
            right_2d = cv2.resize(right_2d, (pw, ph), interpolation=cv2.INTER_AREA)

            # 3D panel
            d3 = render_3d_panel(record, d3_w, d3_h)
            d3 = cv2.resize(d3, (d3_w, d3_h), interpolation=cv2.INTER_AREA)

            # compose
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            # header
            persons_3d = record.get("persons_3d", [])
            if persons_3d:
                p3 = persons_3d[0]
                valid = p3.get("valid_keypoints", 0)
                cost = p3.get("association_cost", 0)
                reproj = p3.get("mean_reprojection_error_px")
                reproj_text = f"{reproj:.2f}px" if reproj is not None else "n/a"
                header = (f"pair {record['pair_id']:03d}  valid_3D={valid}/17  "
                          f"assoc_cost={cost:.4f}  reproj={reproj_text}")
            else:
                header = f"pair {record['pair_id']:03d}  association_failed"
            cv2.putText(canvas, header, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "LEFT 2D", (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(canvas, "RIGHT 2D", (pw + 16, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (180, 180, 180), 1, cv2.LINE_AA)

            y0 = 40
            canvas[y0:y0 + ph, :pw] = left_2d
            canvas[y0:y0 + ph, pw:] = right_2d
            y1 = y0 + ph
            canvas[y1:y1 + d3_h, :] = d3

            writer.write(canvas)
            if (idx + 1) % 50 == 0:
                print(f"  rendered {idx + 1}/{count}")
    finally:
        writer.release()
        left_cap.release()
        right_cap.release()

    print(f"Wrote {args.output}  ({count} frames, {fps} fps, {canvas_w}x{canvas_h})")


if __name__ == "__main__":
    main()
