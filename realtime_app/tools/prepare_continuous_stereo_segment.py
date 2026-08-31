#!/usr/bin/env python3
"""Extract an ordered upright stereo segment from a recorded capture session.

The left stream is rotated counter-clockwise and the right stream clockwise,
matching the frozen pose-inference convention.  A contact sheet is emitted for
both preview and extracted segments so selection remains visually auditable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument("--condition", required=True, choices=("far", "mid", "near"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--pair-count", type=int, default=0, help="0 extracts no frames and writes a preview only")
    parser.add_argument("--preview-count", type=int, default=18)
    return parser.parse_args()


def read_pairs(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No pairs in {path}")
    return rows


def frame_positions(path: Path) -> dict[int, int]:
    """Map capture frame IDs to their zero-based AVI positions.

    Capture IDs may start from a nonzero value after stream warmup and therefore
    cannot be passed directly to CAP_PROP_POS_FRAMES.
    """
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {int(row["frame_id"]): position for position, row in enumerate(rows)}
    if not mapping or len(mapping) != len(rows):
        raise RuntimeError(f"Invalid frame-id mapping in {path}")
    return mapping


def read_frame(capture: cv2.VideoCapture, index: int, label: str) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read {label} frame {index}")
    return frame


def label(frame: np.ndarray, text: str) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (390, 52), (255, 255, 255), -1)
    cv2.putText(result, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2, cv2.LINE_AA)
    return result


def make_sheet(panels: list[np.ndarray], output: Path, columns: int = 3) -> None:
    if not panels:
        return
    thumb_h = 270
    thumbs = [
        cv2.resize(panel, (round(panel.shape[1] * thumb_h / panel.shape[0]), thumb_h), interpolation=cv2.INTER_AREA)
        for panel in panels
    ]
    cell_w = max(panel.shape[1] for panel in thumbs)
    rows = math.ceil(len(thumbs) / columns)
    sheet = np.full((rows * thumb_h, columns * cell_w, 3), 245, dtype=np.uint8)
    for index, panel in enumerate(thumbs):
        row, col = divmod(index, columns)
        x = col * cell_w + (cell_w - panel.shape[1]) // 2
        sheet[row * thumb_h : row * thumb_h + panel.shape[0], x : x + panel.shape[1]] = panel
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"Cannot write {output}")


def upright_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return cv2.rotate(left, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.rotate(right, cv2.ROTATE_90_CLOCKWISE)


def main() -> int:
    args = parse_args()
    session = args.session_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    pairs = read_pairs(session / "stereo_pairs.csv")
    left_positions = frame_positions(session / "left_frames.csv")
    right_positions = frame_positions(session / "right_frames.csv")
    if args.start_pair < 0 or args.start_pair >= len(pairs):
        raise ValueError("--start-pair is outside stereo_pairs.csv")
    if args.pair_count < 0 or args.preview_count <= 0:
        raise ValueError("pair-count must be nonnegative and preview-count positive")
    if args.pair_count and args.start_pair + args.pair_count > len(pairs):
        raise ValueError("Requested segment exceeds stereo_pairs.csv")

    left_cap = cv2.VideoCapture(str(session / "left_capture.avi"))
    right_cap = cv2.VideoCapture(str(session / "right_capture.avi"))
    if not left_cap.isOpened() or not right_cap.isOpened():
        raise RuntimeError("Cannot open one or both capture AVI files")
    try:
        output.mkdir(parents=True)
        preview_indexes = sorted({round(i * (len(pairs) - 1) / max(1, args.preview_count - 1)) for i in range(args.preview_count)})
        preview_panels = []
        for pair_index in preview_indexes:
            row = pairs[pair_index]
            left, right = upright_pair(
                read_frame(left_cap, left_positions[int(row["left_frame_id"])], "left"),
                read_frame(right_cap, right_positions[int(row["right_frame_id"])], "right"),
            )
            preview_panels.append(label(np.hstack((left, right)), f"pair {pair_index}"))
        make_sheet(preview_panels, output / "timeline_preview.jpg")

        manifest = []
        if args.pair_count:
            left_dir, right_dir = output / "left_ccw90", output / "right_cw90"
            left_dir.mkdir(); right_dir.mkdir()
            selected = []
            for relative_index, pair_index in enumerate(range(args.start_pair, args.start_pair + args.pair_count)):
                row = pairs[pair_index]
                file_name = f"pair_{relative_index:04d}.png"
                left_position = left_positions[int(row["left_frame_id"])]
                right_position = right_positions[int(row["right_frame_id"])]
                selected.append((relative_index, pair_index, file_name, left_position, right_position, row))
                manifest.append({
                    "sequence_index": relative_index,
                    "source_pair_index": pair_index,
                    "file_name": file_name,
                    "condition": args.condition,
                    "left_frame_id": row["left_frame_id"],
                    "right_frame_id": row["right_frame_id"],
                    "abs_host_delta_ms": row["abs_host_delta_ms"],
                })

            left_targets = {item[3]: item[2] for item in selected}
            right_targets = {item[4]: item[2] for item in selected}
            left_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            right_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for frame_position in range(max(left_targets) + 1):
                ok, frame = left_cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Cannot sequentially read left frame {frame_position}")
                if frame_position in left_targets:
                    upright = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    if not cv2.imwrite(str(left_dir / left_targets[frame_position]), upright):
                        raise RuntimeError(f"Cannot write left frame {frame_position}")
            for frame_position in range(max(right_targets) + 1):
                ok, frame = right_cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Cannot sequentially read right frame {frame_position}")
                if frame_position in right_targets:
                    upright = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    if not cv2.imwrite(str(right_dir / right_targets[frame_position]), upright):
                        raise RuntimeError(f"Cannot write right frame {frame_position}")

            segment_panels = []
            for relative_index, pair_index, file_name, *_ in selected:
                if relative_index % max(1, args.pair_count // 18) == 0 or relative_index == args.pair_count - 1:
                    left = cv2.imread(str(left_dir / file_name))
                    right = cv2.imread(str(right_dir / file_name))
                    if left is None or right is None:
                        raise RuntimeError(f"Cannot reread extracted pair {pair_index}")
                    segment_panels.append(label(np.hstack((left, right)), f"source pair {pair_index}"))
            with (output / "selection_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
                writer.writeheader(); writer.writerows(manifest)
            make_sheet(segment_panels, output / "segment_preview.jpg")

        metadata = {
            "source_session": str(session),
            "condition": args.condition,
            "pair_count": args.pair_count,
            "start_pair": args.start_pair,
            "model_input_rotation": {"left": "ccw90", "right": "cw90"},
            "timeline_preview_pairs": preview_indexes,
            "selection": "ordered, contiguous stereo_pairs.csv rows",
        }
        (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output), "preview_pairs": len(preview_indexes), "extracted_pairs": len(manifest)}, ensure_ascii=False))
    finally:
        left_cap.release(); right_cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
