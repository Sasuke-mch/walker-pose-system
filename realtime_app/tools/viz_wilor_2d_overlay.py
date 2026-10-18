#!/usr/bin/env python3
"""Overlay WiLoR 2D hand keypoints on raw-fisheye images.

Green = point inside the raw image; red = point outside the raw image.
Only reads inputs and writes new PNGs; never modifies existing results.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "research_records/engineering_validation/G20260927_wilor_smplh_full448_v1"
INP = ROOT / "research_records/engineering_validation/V20260908_people0_1_2_pmpose_c3_chain/people_1/input_448pairs"
OUT = EXP / "wilor_2d_overlay"

EDGES = [(0, 1), (1, 2), (2, 3), (3, 4),
         (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16),
         (0, 17), (17, 18), (18, 19), (19, 20)]

RAW_W, RAW_H = 1920, 1080


def to_raw(image: np.ndarray, orientation: str) -> np.ndarray:
    # Mirror of run_wilor_sequence.process_image: the stored image rotated
    # back to the raw (w=1920, h=1080) frame in which the saved 2D points live.
    if orientation == "left_ccw90":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if orientation == "right_cw90":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def render(jsonl: Path, orientation: str, frames: list[int]) -> None:
    rows = {}
    with jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if int(row["frame_index"]) in frames:
                rows[int(row["frame_index"])] = row
    for fi in frames:
        row = rows[fi]
        src = cv2.imread(str(Path(row["image"])))
        if src is None:
            print(f"missing image for frame {fi}", file=sys.stderr)
            continue
        raw = to_raw(src, orientation)
        h, w = raw.shape[:2]
        for cand in row["records"]:
            pts = np.asarray(cand["keypoints_2d_raw_fisheye"], np.float32)
            inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
            color_side = (255, 255, 0) if cand.get("side") == "left" else (255, 0, 255)
            for a, b in EDGES:
                pa, pb = pts[a], pts[b]
                if np.isfinite(pa).all() and np.isfinite(pb).all():
                    clip_a = (int(np.clip(pa[0], 0, w - 1)), int(np.clip(pa[1], 0, h - 1)))
                    clip_b = (int(np.clip(pb[0], 0, w - 1)), int(np.clip(pb[1], 0, h - 1)))
                    cv2.line(raw, clip_a, clip_b, (200, 200, 200), 1, cv2.LINE_AA)
            for k in range(21):
                if not np.isfinite(pts[k]).all():
                    continue
                col = (0, 255, 0) if inside[k] else (0, 0, 255)
                cx, cy = int(np.clip(pts[k][0], 0, w - 1)), int(np.clip(pts[k][1], 0, h - 1))
                cv2.circle(raw, (cx, cy), 6, col, -1, cv2.LINE_AA)
                cv2.circle(raw, (cx, cy), 6, (0, 0, 0), 1, cv2.LINE_AA)
            x0 = int(np.clip(pts[0][0], 0, w - 1)) if np.isfinite(pts[0]).all() else 20
            y0 = int(np.clip(pts[0][1], 0, h - 1)) if np.isfinite(pts[0]).all() else 30
            cv2.putText(raw, f"{cand.get('side')}#{cand.get('candidate_index')}",
                        (x0 + 10, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_side, 2, cv2.LINE_AA)
        cv2.putText(raw, f"frame {fi} {orientation} raw_fisheye  green=in  red=out",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        out = OUT / f"{orientation}_frame_{fi:04d}.png"
        cv2.imwrite(str(out), raw)
        print(f"wrote {out}")


def main() -> None:
    frames = [int(x) for x in sys.argv[1:]] or [0, 1, 2]
    OUT.mkdir(parents=True, exist_ok=True)
    render(EXP / "wilor_left.jsonl", "left_ccw90", frames)
    render(EXP / "wilor_right.jsonl", "right_cw90", frames)


if __name__ == "__main__":
    main()
