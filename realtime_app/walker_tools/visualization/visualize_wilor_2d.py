#!/usr/bin/env python3
"""Render WiLoR candidate boxes and 21-point projections in raw pixels."""
from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
_tool_prepare_imports()

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def restore_raw(image: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "left_ccw90":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if orientation == "right_cw90":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def draw_record(image: np.ndarray, record: dict) -> None:
    h, w = image.shape[:2]
    box = np.asarray(record["bbox_xyxy"], dtype=float).reshape(4)
    side = str(record.get("side", "?"))
    ok = bool(record.get("raw_pixel_bounds_ok", False))
    color = (50, 190, 50) if ok else (0, 90, 255)
    p0 = tuple(np.round(box[:2]).astype(int))
    p1 = tuple(np.round(box[2:]).astype(int))
    cv2.rectangle(image, p0, p1, color, 2, cv2.LINE_AA)
    pts = np.asarray(record["keypoints_2d_raw_fisheye"], dtype=float)
    for j, (x, y) in enumerate(pts):
        inside = 0 <= x < w and 0 <= y < h and np.isfinite(x) and np.isfinite(y)
        pc = (50, 210, 50) if inside else (0, 0, 255)
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(image, (int(round(x)), int(round(y))), 5, pc, -1, cv2.LINE_AA)
            cv2.putText(image, str(j), (int(round(x)) + 5, int(round(y)) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, pc, 1, cv2.LINE_AA)
    confidence = record.get("detector_confidence")
    conf_text = "n/a" if confidence is None else f"{float(confidence):.2f}"
    label = f"{side} cand={record.get('candidate_index')} bounds={ok} conf={conf_text}"
    cv2.putText(image, label, (max(5, p0[0]), max(22, p0[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--orientation", choices=("left_ccw90", "right_cw90", "raw"), required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 100, 200, 300, 400])
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    rows = {int(json.loads(line)["frame_index"]): json.loads(line) for line in args.jsonl.open(encoding="utf-8")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for frame in args.frames:
        row = rows.get(frame)
        if row is None:
            continue
        source = Path(row["image"])
        image = restore_raw(cv2.imread(str(source)), args.orientation)
        for record in row["records"]:
            draw_record(image, record)
        cv2.putText(image, f"frame={frame} raw={image.shape[1]}x{image.shape[0]}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(image, f"frame={frame} raw={image.shape[1]}x{image.shape[0]}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
        out = args.output_dir / f"frame_{frame:04d}.png"
        cv2.imwrite(str(out), image)
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
