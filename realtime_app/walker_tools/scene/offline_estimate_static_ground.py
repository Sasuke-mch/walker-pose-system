#!/usr/bin/env python3
"""Estimate one fixed floor frame from a static stereo ChArUco capture."""

from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
_tool_prepare_imports()

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np


APP_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP_DIR))

from pose_app.calibration import StereoCalibration  # noqa: E402
from pose_app.static_ground_reference import (  # noqa: E402
    build_reference,
    detect_charuco,
    estimate_board_pose,
    make_charuco_board,
    right_pose_in_left_frame,
    rotation_angle_deg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-session", type=Path, required=True)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=APP_DIR / "calibration" / "results" / "stereo_fisheye.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--board-thickness-mm", type=float, default=12.0)
    parser.add_argument("--min-corners", type=int, default=12)
    parser.add_argument("--min-accepted-pairs", type=int, default=8)
    parser.add_argument("--max-reprojection-rmse-px", type=float, default=1.5)
    parser.add_argument("--max-stereo-origin-difference-mm", type=float, default=20.0)
    parser.add_argument("--max-stereo-rotation-difference-deg", type=float, default=3.0)
    parser.add_argument("--max-repeatability-origin-p95-mm", type=float, default=8.0)
    parser.add_argument("--max-repeatability-normal-p95-deg", type=float, default=1.5)
    return parser.parse_args()


def load_capture_rows(session: Path) -> tuple[dict, list[dict[str, str]]]:
    metadata = json.loads((session / "metadata.json").read_text(encoding="utf-8-sig"))
    with (session / "pairs.csv").open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("capture session contains no saved pairs")
    return metadata, rows


def image_path(session: Path, row: dict[str, str], side: str) -> Path:
    keys = ("left_file", "cam0_file") if side == "left" else ("right_file", "cam1_file")
    for key in keys:
        value = row.get(key)
        if value:
            return session / Path(value.replace("\\", "/"))
    raise ValueError(f"pairs.csv has no {side} image path")


def main() -> int:
    args = parse_args()
    session = args.capture_session.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    metadata, rows = load_capture_rows(session)
    board_spec = metadata.get("board") or {
        "dictionary": "DICT_4X4_50", "squares_x": 8, "squares_y": 6,
        "square_mm": 30.0, "marker_mm": 22.0,
    }
    board = make_charuco_board(board_spec)
    calibration = StereoCalibration.load(args.calibration)
    accepted: list[tuple[np.ndarray, np.ndarray]] = []
    pair_results: list[dict] = []
    rejection_counts: dict[str, int] = {}

    for row in rows:
        pair_id = int(row.get("saved_pair_id", len(pair_results)))
        reasons: list[str] = []
        left_image = cv2.imread(str(image_path(session, row, "left")), cv2.IMREAD_COLOR)
        right_image = cv2.imread(str(image_path(session, row, "right")), cv2.IMREAD_COLOR)
        if left_image is None or right_image is None:
            raise FileNotFoundError(f"pair {pair_id}: failed to read image")
        runtime = calibration.for_runtime_sizes(
            (left_image.shape[1], left_image.shape[0]),
            (right_image.shape[1], right_image.shape[0]),
        )
        left_corners, left_ids = detect_charuco(left_image, board)
        right_corners, right_ids = detect_charuco(right_image, board)
        if len(left_ids) < args.min_corners:
            reasons.append("too_few_left_corners")
        if len(right_ids) < args.min_corners:
            reasons.append("too_few_right_corners")
        left_pose = right_pose = None
        if not reasons:
            left_pose = estimate_board_pose(
                left_corners, left_ids, board, runtime.left_K, runtime.left_D, runtime.camera_model
            )
            right_pose = estimate_board_pose(
                right_corners, right_ids, board, runtime.right_K, runtime.right_D, runtime.camera_model
            )
            if left_pose["reprojection_rmse_px"] > args.max_reprojection_rmse_px:
                reasons.append("left_reprojection_too_large")
            if right_pose["reprojection_rmse_px"] > args.max_reprojection_rmse_px:
                reasons.append("right_reprojection_too_large")
        origin_difference = rotation_difference = None
        left_rotation = left_translation = None
        if left_pose is not None and right_pose is not None:
            left_rotation = np.asarray(left_pose["rotation_camera_from_board"], dtype=np.float64)
            left_translation = np.asarray(left_pose["translation_camera_from_board_mm"], dtype=np.float64)
            right_as_left_rotation, right_as_left_translation = right_pose_in_left_frame(
                np.asarray(right_pose["rotation_camera_from_board"]),
                np.asarray(right_pose["translation_camera_from_board_mm"]),
                runtime.R,
                runtime.T,
            )
            origin_difference = float(np.linalg.norm(left_translation - right_as_left_translation))
            rotation_difference = rotation_angle_deg(left_rotation, right_as_left_rotation)
            if origin_difference > args.max_stereo_origin_difference_mm:
                reasons.append("left_right_board_origin_difference_too_large")
            if rotation_difference > args.max_stereo_rotation_difference_deg:
                reasons.append("left_right_board_rotation_difference_too_large")
        if not reasons and left_rotation is not None and left_translation is not None:
            accepted.append((left_rotation, left_translation))
        for reason in reasons:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        pair_results.append({
            "pair_id": pair_id,
            "accepted": not reasons,
            "reasons": reasons,
            "left_pose": left_pose,
            "right_pose": right_pose,
            "left_right_origin_difference_mm": origin_difference,
            "left_right_rotation_difference_deg": rotation_difference,
        })

    with (output / "per_pair.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for result in pair_results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    if len(accepted) < args.min_accepted_pairs:
        summary = {
            "status": "rejected",
            "reason": "too_few_accepted_pairs",
            "input_pairs": len(rows),
            "accepted_pairs": len(accepted),
            "rejection_counts": rejection_counts,
        }
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False))
        return 2

    reference = build_reference(
        accepted,
        source_capture_session=str(session),
        per_pair_results=pair_results,
        board_spec=board_spec,
        board_thickness_mm=args.board_thickness_mm,
    )
    repeatability = reference["repeatability"]
    repeatability_failures = []
    if repeatability["origin_spread_p95_mm"] > args.max_repeatability_origin_p95_mm:
        repeatability_failures.append("board_origin_repeatability_too_large")
    if repeatability["normal_angle_p95_deg"] > args.max_repeatability_normal_p95_deg:
        repeatability_failures.append("board_normal_repeatability_too_large")
    if repeatability_failures:
        reference["status"] = "rejected"
        reference["reasons"] = repeatability_failures
        filename = "rejected_ground_reference.json"
        return_code = 2
    else:
        filename = "ground_reference.json"
        return_code = 0
    (output / filename).write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "status": reference["status"],
        "input_pairs": len(rows),
        "accepted_pairs": len(accepted),
        "rejection_counts": rejection_counts,
        "board_thickness_mm": args.board_thickness_mm,
        "camera_height_mm": reference["plane_left_camera"]["camera_height_mm"],
        "repeatability": repeatability,
        "output": filename,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
