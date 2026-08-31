#!/usr/bin/env python3
"""Render an auditable continuous-ROI qualitative review video.

Each output frame has three aligned panels from one upright camera view:
the YOLO/raw-versus-continuous-ROI overlay, PMPose 2-D rendering, and Sapiens2
rendering.  The headers state the already-computed stereo acceptance status;
this renderer does not invoke a model or alter a 3-D observation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


LOWER_NAMES = {
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--box-visualization-dir", type=Path, required=True)
    parser.add_argument("--pmpose-visualization-dir", type=Path, required=True)
    parser.add_argument("--sapiens-visualization-dir", type=Path, required=True)
    parser.add_argument("--pmpose-geometry", type=Path, required=True)
    parser.add_argument("--foot-geometry", type=Path, required=True)
    parser.add_argument("--side-label", required=True, choices=("left", "right"))
    parser.add_argument("--model-panel", choices=("both", "pmpose", "sapiens2"), default="both")
    parser.add_argument("--pose-model-label", default="PMPose")
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--keyframe-sheet", type=Path, required=True)
    parser.add_argument("--display-fps", type=float, default=15.0)
    parser.add_argument("--panel-height", type=int, default=1080)
    return parser.parse_args()


def read_jsonl(path: Path) -> dict[int, dict]:
    result: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[int(row["pair_id"])] = row
    return result


def outline_text(image: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.54) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def render_panel(image: np.ndarray, title: str, detail: str, target_height: int) -> np.ndarray:
    if image is None:
        raise RuntimeError("Missing visualization image")
    scale = target_height / image.shape[0]
    width = max(1, round(image.shape[1] * scale))
    body = cv2.resize(image, (width, target_height), interpolation=cv2.INTER_AREA)
    header = np.zeros((58, width, 3), dtype=np.uint8)
    outline_text(header, title, (12, 23), 0.57)
    outline_text(header, detail, (12, 48), 0.48)
    return np.vstack((header, body))


def pmpose_detail(record: dict | None) -> str:
    if not record or not record.get("persons_3d"):
        return "stereo association: no | valid lower body: 0/6"
    person = record["persons_3d"][0]
    valid = sum(
        bool(point.get("valid"))
        for point in person.get("keypoints_3d", [])
        if point.get("name") in LOWER_NAMES
    )
    residual = person.get("mean_reprojection_error_px")
    residual_text = "n/a" if residual is None else f"{float(residual):.1f}px"
    return f"stereo association: yes | valid lower body: {valid}/6 | mean: {residual_text}"


def foot_detail(record: dict | None) -> str:
    if not record:
        return "valid foot points: 0/8"
    points = record.get("foot_points", [])
    valid = sum(bool(point.get("valid_at_reprojection_gate")) for point in points)
    return f"stereo accepted foot points: {valid}/8"


def compose_frame(
    pair_id: int,
    name: str,
    args: argparse.Namespace,
    pmpose: dict[int, dict],
    feet: dict[int, dict],
) -> np.ndarray:
    stem = Path(name).stem
    box = cv2.imread(str(args.box_visualization_dir / f"{stem}.jpg"))
    if box is None:
        box = cv2.imread(str(args.input_dir / name))
        if box is None:
            raise RuntimeError(f"Cannot read source frame: {args.input_dir / name}")
        cv2.putText(box, "NO YOLO PERSON DETECTION: no pose ROI generated", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(box, "NO YOLO PERSON DETECTION: no pose ROI generated", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
    pmpose_image = cv2.imread(str(args.pmpose_visualization_dir / f"{stem}.jpg"))
    sapiens = cv2.imread(str(args.sapiens_visualization_dir / f"{stem}.jpg"))
    panels = [render_panel(box, f"{args.side_label.upper()} view: continuous pose ROI", "red: raw YOLO box | green: actual pose ROI", args.panel_height)]
    if args.model_panel in ("both", "pmpose"):
        panels.append(render_panel(pmpose_image, f"{args.pose_model_label} 2-D pose", pmpose_detail(pmpose.get(pair_id)), args.panel_height))
    if args.model_panel in ("both", "sapiens2"):
        panels.append(render_panel(sapiens, "Sapiens2 2-D pose and feet", foot_detail(feet.get(pair_id)), args.panel_height))
    return np.hstack(panels)


def save_keyframe_sheet(path: Path, frames: list[np.ndarray], indices: list[int]) -> None:
    thumbs = []
    for frame, index in zip(frames, indices):
        thumb_height = 420
        thumb_width = round(frame.shape[1] * thumb_height / frame.shape[0])
        thumb = cv2.resize(frame, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, f"pair {index:03d}", (16, thumb_height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(thumb, f"pair {index:03d}", (16, thumb_height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(thumb)
    rows = [np.hstack(thumbs[i:i + 2]) for i in range(0, len(thumbs), 2)]
    if len(rows) > 1 and rows[-1].shape[1] < rows[0].shape[1]:
        rows[-1] = cv2.copyMakeBorder(rows[-1], 0, 0, 0, rows[0].shape[1] - rows[-1].shape[1], cv2.BORDER_CONSTANT)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.vstack(rows)):
        raise RuntimeError(f"Cannot write {path}")


def main() -> None:
    args = parse_args()
    image_names = sorted(path.name for path in args.input_dir.glob("*.png"))
    if not image_names:
        raise RuntimeError(f"No PNG frames in {args.input_dir}")
    pmpose = read_jsonl(args.pmpose_geometry)
    feet = read_jsonl(args.foot_geometry)
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    sample = compose_frame(0, image_names[0], args, pmpose, feet)
    writer = cv2.VideoWriter(
        str(args.output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.display_fps,
        (sample.shape[1], sample.shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {args.output_video}")
    keyframe_indices = sorted({0, len(image_names) // 5, 2 * len(image_names) // 5, 3 * len(image_names) // 5, 4 * len(image_names) // 5, len(image_names) - 1})
    keyframes: list[np.ndarray] = []
    try:
        for pair_id, name in enumerate(image_names):
            frame = sample if pair_id == 0 else compose_frame(pair_id, name, args, pmpose, feet)
            writer.write(frame)
            if pair_id in keyframe_indices:
                keyframes.append(frame)
    finally:
        writer.release()
    save_keyframe_sheet(args.keyframe_sheet, keyframes, keyframe_indices)
    print(json.dumps({
        "side": args.side_label,
        "model_panel": args.model_panel,
        "frames": len(image_names),
        "display_fps": args.display_fps,
        "duration_sec": len(image_names) / args.display_fps,
        "video": str(args.output_video),
        "keyframe_sheet": str(args.keyframe_sheet),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
