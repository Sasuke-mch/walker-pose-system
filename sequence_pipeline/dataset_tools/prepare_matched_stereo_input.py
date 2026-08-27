from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROTATIONS = {
    "none": None,
    "cw90": cv2.ROTATE_90_CLOCKWISE,
    "ccw90": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180": cv2.ROTATE_180,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export timestamp-matched stereo capture pairs as model-ready PNG inputs."
    )
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--max-pair-delta-ms", type=float, default=10.0)
    parser.add_argument("--left-rotation", choices=ROTATIONS, default="cw90")
    parser.add_argument("--right-rotation", choices=ROTATIONS, default="ccw90")
    parser.add_argument("--preview-samples", type=int, default=6)
    return parser.parse_args()


def rotate(image, mode):
    operation = ROTATIONS[mode]
    return image if operation is None else cv2.rotate(image, operation)


def read_frame(capture, frame_id, side):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Cannot read {side} frame {frame_id}")
    return frame


def choose_evenly(rows, count):
    if count > len(rows):
        raise RuntimeError(f"Need {count} eligible pairs, only have {len(rows)}")
    positions = np.linspace(0, len(rows) - 1, count, dtype=int)
    if len(set(map(int, positions))) != count:
        raise RuntimeError("Uniform sampling produced duplicate positions")
    return [rows[int(position)] for position in positions]


def make_preview(path, samples, left_rotation, right_rotation):
    cells = []
    panel_width, panel_height = 240, 427

    for item in samples:
        left = cv2.resize(
            rotate(item["left"], left_rotation),
            (panel_width, panel_height),
            interpolation=cv2.INTER_AREA,
        )
        right = cv2.resize(
            rotate(item["right"], right_rotation),
            (panel_width, panel_height),
            interpolation=cv2.INTER_AREA,
        )

        cell = np.zeros((panel_height + 42, panel_width * 2, 3), dtype=np.uint8)
        cell[42:, :panel_width] = left
        cell[42:, panel_width:] = right
        text = (
            f"selected={item['selected_index']:02d} "
            f"orig_pair={item['pair_id']} "
            f"dt={item['abs_host_delta_ms']:.2f}ms"
        )
        cv2.putText(
            cell, text, (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA,
        )
        cells.append(cell)

    columns = 3
    blank = np.zeros_like(cells[0])
    while len(cells) % columns:
        cells.append(blank.copy())

    rows = [np.hstack(cells[index:index + columns]) for index in range(0, len(cells), columns)]
    sheet = np.vstack(rows)
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"Cannot write preview: {path}")


def main():
    args = parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.max_pair_delta_ms <= 0:
        raise ValueError("--max-pair-delta-ms must be positive")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Output root is non-empty: {args.output_root}")

    pairs_csv = args.session_dir / "stereo_pairs.csv"
    left_video = args.session_dir / "left_capture.avi"
    right_video = args.session_dir / "right_capture.avi"

    for path in (pairs_csv, left_video, right_video):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required capture artifact: {path}")

    with pairs_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    required_columns = {
        "pair_id",
        "left_frame_id",
        "right_frame_id",
        "left_host_return_timestamp_ns",
        "right_host_return_timestamp_ns",
        "signed_host_delta_ms_right_minus_left",
        "abs_host_delta_ms",
    }
    if not rows or not required_columns.issubset(rows[0]):
        raise RuntimeError(
            f"Unexpected stereo_pairs.csv schema. Required={sorted(required_columns)}, "
            f"actual={list(rows[0]) if rows else 'no rows'}"
        )

    eligible = [
        row for row in rows
        if float(row["abs_host_delta_ms"]) <= args.max_pair_delta_ms
    ]
    selected = choose_evenly(eligible, args.count)

    args.output_root.mkdir(parents=True, exist_ok=True)
    left_dir = args.output_root / f"left_{args.left_rotation}"
    right_dir = args.output_root / f"right_{args.right_rotation}"
    raw_dir = args.output_root / "raw_selected_pairs"
    left_dir.mkdir()
    right_dir.mkdir()
    raw_dir.mkdir()

    left_cap = cv2.VideoCapture(str(left_video))
    right_cap = cv2.VideoCapture(str(right_video))
    if not left_cap.isOpened() or not right_cap.isOpened():
        raise RuntimeError("Cannot open one or both source AVI files")

    source_fps = left_cap.get(cv2.CAP_PROP_FPS) or 30.0
    first_timestamp_ns = None
    selected_metadata = []
    preview_positions = set(np.linspace(0, args.count - 1, args.preview_samples, dtype=int).tolist())
    preview_items = []

    left_writer = None
    right_writer = None

    try:
        for selected_index, row in enumerate(selected):
            left_id = int(row["left_frame_id"])
            right_id = int(row["right_frame_id"])
            left_raw = read_frame(left_cap, left_id, "left")
            right_raw = read_frame(right_cap, right_id, "right")

            if left_raw.shape != right_raw.shape:
                raise RuntimeError(
                    f"Raw frame shape mismatch at selected pair {selected_index}: "
                    f"left={left_raw.shape}, right={right_raw.shape}"
                )

            left_model = rotate(left_raw, args.left_rotation)
            right_model = rotate(right_raw, args.right_rotation)

            if selected_index == 0:
                height, width = left_raw.shape[:2]
                timestamps = [
                    int(item["left_host_return_timestamp_ns"])
                    for item in selected
                ]
                duration_sec = max((max(timestamps) - min(timestamps)) / 1e9, 1.0)
                selected_fps = max((args.count - 1) / duration_sec, 1.0)
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                left_writer = cv2.VideoWriter(
                    str(raw_dir / "left_selected.avi"), fourcc, selected_fps, (width, height)
                )
                right_writer = cv2.VideoWriter(
                    str(raw_dir / "right_selected.avi"), fourcc, selected_fps, (width, height)
                )
                if not left_writer.isOpened() or not right_writer.isOpened():
                    raise RuntimeError("Cannot create selected-pair AVI files")

            output_name = f"pair_{selected_index:03d}.png"
            if not cv2.imwrite(str(left_dir / output_name), left_model):
                raise RuntimeError(f"Cannot write left image: {output_name}")
            if not cv2.imwrite(str(right_dir / output_name), right_model):
                raise RuntimeError(f"Cannot write right image: {output_name}")

            left_writer.write(left_raw)
            right_writer.write(right_raw)

            pair_timestamp_ns = (
                int(row["left_host_return_timestamp_ns"])
                + int(row["right_host_return_timestamp_ns"])
            ) // 2
            if first_timestamp_ns is None:
                first_timestamp_ns = pair_timestamp_ns

            item = {
                "selected_index": selected_index,
                "file_name": output_name,
                "pair_id": int(row["pair_id"]),
                "left_frame_id": left_id,
                "right_frame_id": right_id,
                "left_host_return_timestamp_ns": int(row["left_host_return_timestamp_ns"]),
                "right_host_return_timestamp_ns": int(row["right_host_return_timestamp_ns"]),
                "pair_timestamp_sec": (pair_timestamp_ns - first_timestamp_ns) / 1e9,
                "signed_host_delta_ms_right_minus_left": float(
                    row["signed_host_delta_ms_right_minus_left"]
                ),
                "abs_host_delta_ms": float(row["abs_host_delta_ms"]),
            }
            selected_metadata.append(item)

            if selected_index in preview_positions:
                preview_items.append(
                    {
                        **item,
                        "left": left_raw,
                        "right": right_raw,
                    }
                )
    finally:
        left_cap.release()
        right_cap.release()
        if left_writer is not None:
            left_writer.release()
        if right_writer is not None:
            right_writer.release()

    manifest_path = args.output_root / "selection_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_metadata[0]))
        writer.writeheader()
        writer.writerows(selected_metadata)

    make_preview(
        args.output_root / "model_input_preflight.jpg",
        preview_items,
        args.left_rotation,
        args.right_rotation,
    )

    summary = {
        "source_session_dir": str(args.session_dir),
        "source_left_video": str(left_video),
        "source_right_video": str(right_video),
        "source_pairs_csv": str(pairs_csv),
        "total_recorded_pairs": len(rows),
        "eligible_pairs": len(eligible),
        "selected_pairs": len(selected_metadata),
        "max_pair_delta_ms": args.max_pair_delta_ms,
        "left_model_rotation": args.left_rotation,
        "right_model_rotation": args.right_rotation,
        "source_capture_fps": source_fps,
        "left_input_dir": str(left_dir),
        "right_input_dir": str(right_dir),
        "selected_raw_pair_dir": str(raw_dir),
        "selection_manifest": str(manifest_path),
        "preflight_image": str(args.output_root / "model_input_preflight.jpg"),
    }
    (args.output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"left images: {len(list(left_dir.glob('*.png')))}")
    print(f"right images: {len(list(right_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()