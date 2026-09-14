#!/usr/bin/env python3
"""Capture full-resolution stereo ChArUco pairs for one fixed floor frame."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
APP_DIR = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(APP_DIR))

from capture_stereo_charuco import (  # noqa: E402
    detect_charuco,
    draw_detection,
    make_board,
    put_text,
    resolve_camera_selection,
)
from pose_app.stereo_camera import StereoCameraConfig, StereoCameraSource  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-registry", type=Path)
    parser.add_argument("--left-camera", type=int)
    parser.add_argument("--right-camera", type=int)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--backend", choices=["msmf", "dshow", "auto"], default="auto")
    parser.add_argument("--max-pair-delta-ms", type=float, default=25.0)
    parser.add_argument(
        "--min-corners-per-view",
        "--min-common-corners",
        dest="min_corners_per_view",
        type=int,
        default=12,
        help=(
            "Minimum ChArUco corners detected independently in each view. "
            "The legacy --min-common-corners spelling is kept as an alias."
        ),
    )
    parser.add_argument("--target-count", type=int, default=20)
    parser.add_argument("--board-thickness-mm", type=float, default=12.0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=APP_DIR / "calibration" / "captures" / "static_ground",
    )
    args = parser.parse_args()
    if args.target_count < 1:
        parser.error("--target-count must be positive")
    if args.min_corners_per_view < 4:
        parser.error("--min-corners-per-view must be at least 4")
    if args.board_thickness_mm < 0:
        parser.error("--board-thickness-mm must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    left_index, right_index, backend, resolved, selection_mode = resolve_camera_selection(args)
    _, detector = make_board()
    session = args.output_root / f"session_{datetime.now():%Y%m%d_%H%M%S}"
    left_dir, right_dir = session / "cam0", session / "cam1"
    left_dir.mkdir(parents=True, exist_ok=False)
    right_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema_version": "static_ground_charuco_capture_v1",
        "created_local": datetime.now().isoformat(timespec="seconds"),
        "purpose": "offline fixed ground frame",
        "board": {
            "type": "ChArUco",
            "dictionary": "DICT_4X4_50",
            "squares_x": 8,
            "squares_y": 6,
            "square_mm": 30.0,
            "marker_mm": 22.0,
            "thickness_mm": args.board_thickness_mm,
            "detected_surface": "printed top surface",
        },
        "capture": {
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "backend": backend,
            "target_count": args.target_count,
            "max_pair_delta_ms": args.max_pair_delta_ms,
            "min_corners_per_view": args.min_corners_per_view,
        },
        "camera_selection": {
            "mode": selection_mode,
            "left_index": left_index,
            "right_index": right_index,
            "registry": resolved.to_dict() if resolved is not None else None,
        },
        "scene_requirement": (
            "The board lies flat with printed face upward. The walker and both cameras remain "
            "stationary until the walking sequence has been captured."
        ),
        "timestamp_note": "Host read-return timestamps are not sensor exposure timestamps.",
    }
    (session / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    source = StereoCameraSource(
        StereoCameraConfig(
            left_id=left_index,
            right_id=right_index,
            width=args.width,
            height=args.height,
            fps=args.fps,
            backend=backend,
            max_pair_delta_ms=args.max_pair_delta_ms,
            queue_size=8,
        )
    )
    csv_path = session / "pairs.csv"
    saved = 0
    print(f"Output: {session}")
    print("Board: printed face upward, flat on floor; thickness = " f"{args.board_thickness_mm:.1f} mm")
    print("Do not move the walker or cameras before the walking sequence is captured.")
    print("S / SPACE / ENTER = save one pair; Q/ESC = stop")
    print(
        "A pair is ready when each view independently detects at least "
        f"{args.min_corners_per_view} ChArUco corners."
    )
    try:
        source.start()
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "saved_pair_id", "source_pair_id", "left_frame_id", "right_frame_id",
                "left_host_timestamp_ns", "right_host_timestamp_ns", "abs_host_delta_ms",
                "left_charuco_corners", "right_charuco_corners", "common_charuco_corners",
                "common_ids", "left_file", "right_file",
            ])
            last_pair = time.monotonic()
            while saved < args.target_count:
                pair = source.read(timeout_sec=1.0)
                if pair is None:
                    if time.monotonic() - last_pair > 30.0:
                        raise RuntimeError("No paired frames for 30 seconds")
                    continue
                last_pair = time.monotonic()
                left, right = pair.left.image, pair.right.image
                expected = (args.height, args.width)
                if left.shape[:2] != expected or right.shape[:2] != expected:
                    raise RuntimeError(
                        f"Incomplete frame: expected={expected}, left={left.shape[:2]}, right={right.shape[:2]}"
                    )
                dl, dr = detect_charuco(detector, left), detect_charuco(detector, right)
                common = sorted(dl["ids"].intersection(dr["ids"]))
                left_count = len(dl["ids"])
                right_count = len(dr["ids"])
                ready = (
                    left_count >= args.min_corners_per_view
                    and right_count >= args.min_corners_per_view
                )
                vl, vr = draw_detection(left, dl), draw_detection(right, dr)
                scale = min(700.0 / args.width, 500.0 / args.height, 1.0)
                if scale < 1.0:
                    size = (round(args.width * scale), round(args.height * scale))
                    vl, vr = cv2.resize(vl, size), cv2.resize(vr, size)
                view = np.hstack([vl, vr])
                color = (0, 255, 0) if ready else (0, 165, 255)
                put_text(
                    view,
                    f"left={left_count} right={right_count} common={len(common)}  "
                    f"saved={saved}/{args.target_count}",
                    view.shape[0] - 40,
                    color,
                )
                put_text(
                    view,
                    "READY: S/SPACE/ENTER to save"
                    if ready
                    else f"Need >= {args.min_corners_per_view} corners in EACH view",
                    view.shape[0] - 12,
                    color,
                )
                cv2.imshow("Offline static-ground ChArUco capture", view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key not in (ord("s"), ord("S"), ord(" "), 10, 13):
                    continue
                if not ready:
                    print(
                        "Not saved: need "
                        f">={args.min_corners_per_view} corners in each view; "
                        f"left={left_count}, right={right_count}, common={len(common)}"
                    )
                    continue
                left_name = f"pair_{saved:03d}_cam0.png"
                right_name = f"pair_{saved:03d}_cam1.png"
                left_path, right_path = left_dir / left_name, right_dir / right_name
                if not cv2.imwrite(str(left_path), left) or not cv2.imwrite(str(right_path), right):
                    raise RuntimeError(f"Failed to save pair {saved}")
                writer.writerow([
                    saved, pair.pair_id, pair.left.frame_id, pair.right.frame_id,
                    pair.left.host_return_timestamp_ns, pair.right.host_return_timestamp_ns,
                    f"{pair.abs_host_delta_ms:.6f}", len(dl["ids"]), len(dr["ids"]), len(common),
                    " ".join(map(str, common)), str(left_path.relative_to(session)), str(right_path.relative_to(session)),
                ])
                handle.flush()
                print(f"Saved {saved:03d}: common={len(common)}, |dt_host|={pair.abs_host_delta_ms:.2f} ms")
                saved += 1
    finally:
        source.close()
        cv2.destroyAllWindows()
    print(f"Saved {saved} pairs to {session}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
