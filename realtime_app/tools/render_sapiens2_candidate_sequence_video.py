#!/usr/bin/env python3
"""Render the observed Sapiens2 lower-foot candidate sequence on both views.

Every drawn point is an already stored 3-D observation projected back through
the project fisheye calibration.  Missing points remain absent.  This is a
qualitative inspection video, not a new 2-D inference or a 3-D smoothing step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_app.calibration import StereoCalibration
from pose_app.rotation import raw_to_model_point


HKA = ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
DISTAL = ("left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel")
ABBR = {
    "left_hip": "LH", "right_hip": "RH", "left_knee": "LK", "right_knee": "RK",
    "left_ankle": "LA", "right_ankle": "RA", "left_big_toe": "LBT", "left_small_toe": "LST",
    "left_heel": "LHE", "right_big_toe": "RBT", "right_small_toe": "RST", "right_heel": "RHE",
}
STATUS_COLOR = {
    "baseline_geometric_valid": (0, 190, 0),
    "temporally_confirmed": (255, 210, 0),
    "geometric_valid_temporal_unknown": (0, 160, 255),
    "geometric_valid": (255, 90, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-jsonl", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--left-upright-dir", type=Path, required=True)
    parser.add_argument("--right-upright-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--output-height", type=int, default=900)
    return parser.parse_args()


def load_sequence(path: Path) -> list[dict]:
    output = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not output:
        raise RuntimeError("The sequence file is empty")
    expected = list(range(len(output)))
    if [int(row["pair_id"]) for row in output] != expected:
        raise RuntimeError("Sequence pair IDs must be contiguous and ordered")
    return output


def draw_label(canvas: np.ndarray, point: tuple[float, float], name: str, color: tuple[int, int, int]) -> None:
    x, y = round(point[0]), round(point[1])
    if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
        cv2.circle(canvas, (x, y), 6, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 9, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, ABBR[name], (x + 9, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 2, cv2.LINE_AA)


def project_upright(calibration: StereoCalibration, xyz: list[float], side: str) -> tuple[float, float]:
    raw = calibration.project_left(np.asarray([xyz], dtype=np.float64))[0] if side == "left" else calibration.project_right(np.asarray([xyz], dtype=np.float64))[0]
    rotation = "ccw90" if side == "left" else "cw90"
    return raw_to_model_point(float(raw[0]), float(raw[1]), 1920, 1080, rotation)


def overlay_view(image: np.ndarray, record: dict, calibration: StereoCalibration, side: str) -> np.ndarray:
    canvas = image.copy()
    groups = (record["sapiens2_hka"], record["sapiens2_distal_foot"])
    projected: dict[str, tuple[float, float]] = {}
    for group in groups:
        for name, point in group.items():
            if not point.get("valid") or point.get("xyz_left_camera_mm") is None:
                continue
            projected[name] = project_upright(calibration, point["xyz_left_camera_mm"], side)
    for chain in (("left_hip", "left_knee", "left_ankle"), ("right_hip", "right_knee", "right_ankle")):
        for a, b in zip(chain, chain[1:]):
            if a in projected and b in projected:
                cv2.line(canvas, tuple(map(round, projected[a])), tuple(map(round, projected[b])), (230, 230, 230), 2, cv2.LINE_AA)
    for group in groups:
        for name, point in group.items():
            if name not in projected:
                continue
            draw_label(canvas, projected[name], name, STATUS_COLOR.get(point.get("status"), (220, 220, 220)))
    return canvas


def header(panel: np.ndarray, record: dict) -> np.ndarray:
    quality = record["quality"]
    output = panel.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 86), (255, 255, 255), -1)
    line1 = f"{record['file_name']} | HKA {quality['hka_valid_count']}/6 | knee-ankle {quality['knee_ankle_valid_count']}/4 | forefoot {quality['forefoot_valid_count']}/4"
    line2 = "green: baseline geometry | cyan: temporally confirmed | orange: geometry, time unknown | blue: distal foot geometry"
    cv2.putText(output, line1, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(output, line2, (16, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)
    return output


def contact_sheet(panels: list[np.ndarray], output: Path) -> None:
    thumbs = []
    for panel in panels:
        factor = 240 / panel.shape[0]
        thumbs.append(cv2.resize(panel, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA))
    columns = 3
    rows = (len(thumbs) + columns - 1) // columns
    cell_width = max(image.shape[1] for image in thumbs)
    sheet = np.full((rows * 240, columns * cell_width, 3), 245, dtype=np.uint8)
    for index, thumb in enumerate(thumbs):
        row, col = divmod(index, columns)
        x = col * cell_width + (cell_width - thumb.shape[1]) // 2
        sheet[row * 240 : row * 240 + thumb.shape[0], x : x + thumb.shape[1]] = thumb
    cv2.imwrite(str(output), sheet)


def main() -> int:
    args = parse_args()
    if args.fps <= 0 or args.output_height < 360:
        raise ValueError("fps must be positive and output-height must be at least 360")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    records = load_sequence(args.sequence_jsonl.resolve())
    calibration = StereoCalibration.load(args.calibration.resolve()).for_runtime_sizes((1920, 1080), (1920, 1080))
    args.output_dir.mkdir(parents=True)
    first = cv2.imread(str(args.left_upright_dir.resolve() / records[0]["file_name"]))
    if first is None:
        raise RuntimeError("Cannot read the first left image")
    factor = args.output_height / first.shape[0]
    panel_width = round(first.shape[1] * factor) * 2
    panel_height = args.output_height
    video_path = args.output_dir / "near_sapiens2_lower_foot_candidate_review.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (panel_width, panel_height))
    if not writer.isOpened():
        raise RuntimeError("Cannot open MP4 writer")
    sheet_panels: list[np.ndarray] = []
    try:
        sheet_indices = set(round(i * (len(records) - 1) / 8) for i in range(9))
        for index, record in enumerate(records):
            left = cv2.imread(str(args.left_upright_dir.resolve() / record["file_name"]))
            right = cv2.imread(str(args.right_upright_dir.resolve() / record["file_name"]))
            if left is None or right is None:
                raise RuntimeError(f"Cannot read image pair {record['file_name']}")
            panel = np.hstack((overlay_view(left, record, calibration, "left"), overlay_view(right, record, calibration, "right")))
            panel = header(panel, record)
            panel = cv2.resize(panel, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
            writer.write(panel)
            if index in sheet_indices:
                sheet_panels.append(panel)
    finally:
        writer.release()
    contact_sheet(sheet_panels, args.output_dir / "near_sapiens2_candidate_contact_sheet.jpg")
    metadata = {
        "sequence": str(args.sequence_jsonl.resolve()),
        "calibration": str(args.calibration.resolve()),
        "frames": len(records),
        "fps": args.fps,
        "orientation": {"left": "ccw90 upright", "right": "cw90 upright"},
        "interpretation": "Points are existing candidate 3-D observations reprojected to the two original fisheye views. The video did not run a model, smooth a path, interpolate a point, or infer ground contact.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "README.md").write_text("# Sapiens2 下肢与足部候选连续可视化\n\n视频将已通过几何门限的候选三维点反投影回左右正向鱼眼图。未通过点不会绘制；视频不重跑模型、不过滤轨迹或补点。颜色含义见每帧标题栏。\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "video": str(video_path), "frames": len(records), "fps": args.fps}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
