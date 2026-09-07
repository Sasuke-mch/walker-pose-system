#!/usr/bin/env python3
"""Measure whether adjacent-frame optical flow can support 2-D pose tracking.

Sapiens2 observations on two adjacent upright images are used only as a
repeatability reference.  The experiment tracks each visible source landmark
with pyramidal Lucas-Kanade flow and compares the tracked position to the
stored next-frame observation.  It neither changes pose predictions nor fills
missing 3-D data.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


LANDMARKS = {
    9: "left_hip", 10: "right_hip", 11: "left_knee", 12: "right_knee",
    13: "left_ankle", 14: "right_ankle", 15: "left_big_toe", 16: "left_small_toe",
    17: "left_heel", 18: "right_big_toe", 19: "right_small_toe", 20: "right_heel",
}
COLORS = {
    "left_hip": (255, 196, 0), "right_hip": (255, 140, 0),
    "left_knee": (0, 220, 255), "right_knee": (0, 165, 255),
    "left_ankle": (0, 220, 0), "right_ankle": (0, 150, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-predictions", type=Path, required=True)
    parser.add_argument("--right-predictions", type=Path, required=True)
    parser.add_argument("--left-upright-dir", type=Path, required=True)
    parser.add_argument("--right-upright-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--fb-threshold-px", type=float, default=1.5)
    parser.add_argument("--visualization-count", type=int, default=6)
    return parser.parse_args()


def load_predictions(path: Path) -> dict[str, dict]:
    root = json.loads(path.read_text(encoding="utf-8-sig"))
    images = root.get("images")
    if not isinstance(images, list):
        raise RuntimeError(f"No images list in {path}")
    output = {row["file_name"]: row for row in images}
    if len(output) != len(images):
        raise RuntimeError(f"Duplicate file name in {path}")
    return output


def top_instance(row: dict) -> dict | None:
    instances = row.get("instances", [])
    return max(instances, key=lambda item: float(item.get("bbox_score_from_yolo26x", 0.0))) if instances else None


def landmark_points(row: dict, threshold: float) -> dict[str, tuple[np.ndarray, float]]:
    instance = top_instance(row)
    if instance is None:
        return {}
    xy = instance.get("keypoints308", [])
    scores = instance.get("keypoint_scores", [])
    if len(xy) != 308 or len(scores) != 308:
        raise RuntimeError(f"{row.get('file_name')}: invalid Sapiens2 point count")
    output = {}
    for index, name in LANDMARKS.items():
        score = float(scores[index])
        if score >= threshold:
            output[name] = (np.asarray(xy[index], dtype=np.float32), score)
    return output


def read_gray(directory: Path, name: str) -> np.ndarray:
    image = cv2.imread(str(directory / name), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot read {directory / name}")
    return image


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else None


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def draw_track_pair(directory: Path, transition: int, names: list[str], rows: list[dict], output: Path) -> bool:
    before = cv2.imread(str(directory / names[transition]))
    after = cv2.imread(str(directory / names[transition + 1]))
    if before is None or after is None:
        return False
    before = before.copy(); after = after.copy()
    scale = 0.46
    before = cv2.resize(before, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    after = cv2.resize(after, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for row in rows:
        if row["transition"] != transition or not row["track_ok"]:
            continue
        name = row["landmark"]
        color = COLORS.get(name, (220, 220, 220))
        start = (round(float(row["start_x"]) * scale), round(float(row["start_y"]) * scale))
        predicted = (round(float(row["tracked_x"]) * scale), round(float(row["tracked_y"]) * scale))
        observed = (round(float(row["observed_next_x"]) * scale), round(float(row["observed_next_y"]) * scale))
        cv2.circle(before, start, 5, color, -1, cv2.LINE_AA)
        cv2.arrowedLine(after, observed, predicted, color, 2, cv2.LINE_AA, tipLength=0.25)
        cv2.circle(after, observed, 6, color, 1, cv2.LINE_AA)
        cv2.circle(after, predicted, 3, color, -1, cv2.LINE_AA)
        cv2.putText(after, name.replace("_", " "), (observed[0] + 6, observed[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)
    panel = np.hstack((before, after))
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 44), (255, 255, 255), -1)
    cv2.putText(panel, f"{names[transition]} -> {names[transition + 1]} | circle: next Sapiens2 | dot: LK prediction | arrow: observed to prediction", (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return cv2.imwrite(str(output), panel)


def render_summary(rows: list[dict], output: Path, fb_threshold: float) -> None:
    import matplotlib.pyplot as plt

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["track_ok"]:
            groups[f"{row['view']}:{row['landmark']}"] .append(row)
    labels = list(groups)
    median_error = [float(np.median([float(row["next_observation_error_px"]) for row in groups[label]])) for label in labels]
    p90_error = [percentile([float(row["next_observation_error_px"]) for row in groups[label]], 90) for label in labels]
    fig, axis = plt.subplots(figsize=(max(11, len(labels) * 0.55), 5.0), constrained_layout=True)
    x = np.arange(len(labels))
    axis.bar(x - 0.19, median_error, width=0.38, label="median LK-to-next-observation", color="#4c78a8")
    axis.bar(x + 0.19, p90_error, width=0.38, label="P90", color="#f58518")
    axis.axhline(10.0, color="#c62828", linewidth=1.5, label="10 px geometric reference")
    axis.axhline(fb_threshold, color="#54a24b", linewidth=1.5, linestyle="--", label=f"FB check {fb_threshold:g} px")
    axis.set_xticks(x, [label.replace(":", "\n") for label in labels], rotation=0, fontsize=8)
    axis.set_ylabel("pixels")
    axis.set_title("Adjacent-frame LK consistency against stored Sapiens2 observations")
    axis.legend(ncol=2, fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def evaluate_view(view: str, predictions: dict[str, dict], directory: Path, threshold: float, fb_threshold: float) -> tuple[list[dict], list[str]]:
    names = sorted(predictions)
    rows: list[dict] = []
    for transition, (name, next_name) in enumerate(zip(names, names[1:])):
        start_points = landmark_points(predictions[name], threshold)
        next_points = landmark_points(predictions[next_name], threshold)
        gray = read_gray(directory, name)
        next_gray = read_gray(directory, next_name)
        common = [landmark for landmark in LANDMARKS.values() if landmark in start_points and landmark in next_points]
        if common:
            initial = np.asarray([start_points[name][0] for name in common], dtype=np.float32).reshape(-1, 1, 2)
            tracked, status, _ = cv2.calcOpticalFlowPyrLK(gray, next_gray, initial, None, winSize=(31, 31), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(next_gray, gray, tracked, None, winSize=(31, 31), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        else:
            tracked = status = backward = backward_status = None
        for local_index, landmark in enumerate(common):
            start, start_score = start_points[landmark]
            observed_next, next_score = next_points[landmark]
            track_ok = bool(status[local_index, 0]) if status is not None else False
            back_ok = bool(backward_status[local_index, 0]) if backward_status is not None else False
            predicted = tracked[local_index, 0] if track_ok else np.asarray([np.nan, np.nan], dtype=np.float32)
            fb_error = float(np.linalg.norm(backward[local_index, 0] - start)) if track_ok and back_ok else None
            next_error = float(np.linalg.norm(predicted - observed_next)) if track_ok else None
            rows.append({
                "view": view,
                "transition": transition,
                "file_name": name,
                "next_file_name": next_name,
                "landmark": landmark,
                "start_score": float(start_score),
                "next_score": float(next_score),
                "start_x": float(start[0]), "start_y": float(start[1]),
                "observed_next_x": float(observed_next[0]), "observed_next_y": float(observed_next[1]),
                "tracked_x": float(predicted[0]) if track_ok else None, "tracked_y": float(predicted[1]) if track_ok else None,
                "track_ok": track_ok,
                "forward_backward_ok": bool(fb_error is not None and fb_error <= fb_threshold),
                "forward_backward_error_px": fb_error,
                "next_observation_error_px": next_error,
            })
    return rows, names


def main() -> int:
    args = parse_args()
    if not 0 <= args.score_threshold <= 1 or args.fb_threshold_px <= 0 or args.visualization_count <= 0:
        raise ValueError("Invalid thresholds or visualization count")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    left = load_predictions(args.left_predictions.resolve())
    right = load_predictions(args.right_predictions.resolve())
    if set(left) != set(right):
        raise RuntimeError("Left/right prediction names do not match")
    left_rows, names = evaluate_view("left", left, args.left_upright_dir.resolve(), args.score_threshold, args.fb_threshold_px)
    right_rows, _ = evaluate_view("right", right, args.right_upright_dir.resolve(), args.score_threshold, args.fb_threshold_px)
    rows = left_rows + right_rows
    if not rows:
        raise RuntimeError("No two-frame tracks could be evaluated")
    summary = []
    for view in ("left", "right"):
        for landmark in LANDMARKS.values():
            selected = [row for row in rows if row["view"] == view and row["landmark"] == landmark]
            tracked = [row for row in selected if row["track_ok"]]
            errors = [float(row["next_observation_error_px"]) for row in tracked if row["next_observation_error_px"] is not None]
            fb = [float(row["forward_backward_error_px"]) for row in tracked if row["forward_backward_error_px"] is not None]
            summary.append({
                "view": view,
                "landmark": landmark,
                "visible_two_frame_opportunities": len(selected),
                "lk_returned": len(tracked),
                "lk_return_rate": len(tracked) / len(selected) if selected else None,
                "fb_within_threshold": sum(row["forward_backward_ok"] for row in tracked),
                "fb_median_px": float(np.median(fb)) if fb else None,
                "observation_error_median_px": float(np.median(errors)) if errors else None,
                "observation_error_p90_px": percentile(errors, 90),
                "observation_error_within_5px": sum(value <= 5.0 for value in errors),
                "observation_error_within_10px": sum(value <= 10.0 for value in errors),
            })
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "per_track.csv", rows)
    write_csv(args.output_dir / "tracking_summary.csv", summary)
    render_summary(rows, args.output_dir / "tracking_error_summary.png", args.fb_threshold_px)
    candidates = sorted({int(row["transition"]) for row in rows if row["track_ok"]}, key=lambda index: sum(float(row["next_observation_error_px"]) for row in rows if int(row["transition"]) == index and row["track_ok"]), reverse=True)
    selected = candidates[: args.visualization_count]
    visual_root = args.output_dir / "qualitative_tracks"
    visual_root.mkdir()
    written = 0
    for transition in selected:
        written += int(draw_track_pair(args.left_upright_dir.resolve(), transition, names, left_rows, visual_root / f"left_pair_{transition:04d}_to_{transition + 1:04d}.png"))
        written += int(draw_track_pair(args.right_upright_dir.resolve(), transition, names, right_rows, visual_root / f"right_pair_{transition:04d}_to_{transition + 1:04d}.png"))
    metadata = {
        "input": {"left_predictions": str(args.left_predictions.resolve()), "right_predictions": str(args.right_predictions.resolve())},
        "orientation": {"left": "upright ccw90", "right": "upright cw90"},
        "score_threshold": args.score_threshold,
        "forward_backward_threshold_px": args.fb_threshold_px,
        "interpretation": "Stored next-frame Sapiens2 points are a temporal repeatability reference, not manual 2-D ground truth. This evaluation does not alter any pose or 3-D result.",
        "qualitative_tracks": written,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = """# Sapiens2 相邻帧光流可行性检查

每个可见关节从当前正向图像用 Lucas-Kanade 光流跟踪到下一帧，再同下一帧已保存的 Sapiens2 观察比较。箭头图中空心圆是下一帧 Sapiens2 点，实心点是光流预测，二者间箭头长度即比较误差。该误差只反映两种观测的时间一致性，不是人工真值误差。

本检查用于决定短时二维跟踪是否值得作为未来缺失点的候选来源；没有在此处填补任何关节或生成新的三维坐标。
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "tracks": len(rows), "qualitative_images": written}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
