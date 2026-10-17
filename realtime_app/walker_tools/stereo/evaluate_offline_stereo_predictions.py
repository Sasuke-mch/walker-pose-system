"""Replay sequence_pipeline raw predictions through Walker stereo geometry."""
from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
from walker_tools._compat import APP_ROOT as _tool_app_root
_tool_prepare_imports()

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

ROOT = _tool_app_root

from pose_app.calibration import StereoCalibration
from pose_app.geometry_input import count_out_of_raw_image_bounds
from pose_app.rotation import ROTATION_CHOICES
from pose_app.triangulation import triangulate_matches

from pose_app.stereo_prediction_records import records as _records, result as _result


def main():
    parser = argparse.ArgumentParser(
        description="Offline saved-prediction stereo geometry replay."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=("yolo26x_pose", "pmpose", "probpose", "bboxmaskpose", "sapiens2"),
    )
    parser.add_argument("--left-json", required=True, type=Path)
    parser.add_argument("--right-json", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--left-model-rotation", choices=ROTATION_CHOICES, default="ccw90")
    parser.add_argument("--right-model-rotation", choices=ROTATION_CHOICES, default="cw90")
    parser.add_argument("--keypoint-threshold", type=float, default=0.25)
    parser.add_argument("--max-association-cost", type=float, default=0.05)
    parser.add_argument("--max-reprojection-error-px", type=float, default=10.0)
    parser.add_argument(
        "--triangulation-mode",
        choices=("strict", "ungated", "force_all"),
        default="strict",
        help=(
            "strict requires positive depth and the reprojection threshold; "
            "ungated retains every finite in-bounds triangulation after the "
            "2-D score and person-association checks, recording depth and "
            "reprojection issues as quality_flags instead of rejecting points. "
            "force_all additionally forces the best left/right person pairing "
            "per frame (no association gate) and ignores the 2-D score gate, "
            "so every finitely-triangulable in-bounds joint is produced for a "
            "continuous skeleton; quality flags and errors are statistics only."
        ),
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=1,
        help="Maximum stereo person matches retained per frame (default: 1).",
    )
    parser.add_argument(
        "--topdown-score-source",
        choices=("raw_keypoint", "presence"),
        default="raw_keypoint",
    )
    args = parser.parse_args()

    if args.fps <= 0 or not 0 <= args.keypoint_threshold <= 1:
        parser.error("fps must be positive and keypoint threshold must be in [0, 1].")
    if args.max_matches <= 0:
        parser.error("max-matches must be positive.")

    started = time.perf_counter()
    args.left_json = args.left_json.resolve()
    args.right_json = args.right_json.resolve()
    args.calibration = args.calibration.resolve()
    args.output_dir = args.output_dir.resolve()

    cwd = Path.cwd()
    try:
        os.chdir(ROOT)
        source_calibration = StereoCalibration.load(args.calibration)
    finally:
        os.chdir(cwd)

    calibration = source_calibration.for_runtime_sizes(
        source_calibration.left_image_size,
        source_calibration.right_image_size,
    )

    left = _records(args.left_json, args.model, args.topdown_score_source)
    right = _records(args.right_json, args.model, args.topdown_score_source)
    left_by_name = {item["name"]: item for item in left}
    right_by_name = {item["name"]: item for item in right}

    if set(left_by_name) != set(right_by_name):
        raise RuntimeError("Left/right prediction file names do not match exactly.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "offline_stereo_results.jsonl"

    valid = 0
    valid_by_name = Counter()
    reasons = Counter()
    quality_flags = Counter()
    costs, common = [], []
    matched_pairs, matched_persons = 0, 0
    pair_outcomes = Counter()
    rejected_out_of_bounds_keypoints = 0

    with results_path.open("w", encoding="utf-8") as handle:
        for pair_id, name in enumerate(item["name"] for item in left):
            timestamp = pair_id / args.fps

            left_raw = _result(
                left_by_name[name],
                args.model,
                calibration.left_image_size,
                args.left_model_rotation,
                timestamp,
            )
            right_raw = _result(
                right_by_name[name],
                args.model,
                calibration.right_image_size,
                args.right_model_rotation,
                timestamp,
            )
            left_rejected = count_out_of_raw_image_bounds(
                left_raw.persons,
                calibration.left_image_size,
            )
            right_rejected = count_out_of_raw_image_bounds(
                right_raw.persons,
                calibration.right_image_size,
            )
            rejected_out_of_bounds_keypoints += left_rejected + right_rejected

            force_all = args.triangulation_mode == "force_all"
            accept_all = args.triangulation_mode in ("ungated", "force_all")
            keypoint_threshold_eff = 0.0 if force_all else args.keypoint_threshold
            association_cost_eff = float("inf") if force_all else args.max_association_cost

            people3d = triangulate_matches(
                left_raw.persons,
                right_raw.persons,
                calibration,
                keypoint_threshold_eff,
                association_cost_eff,
                args.max_reprojection_error_px,
                max_matches=args.max_matches,
                accept_all_finite_triangulations=accept_all,
            )

            matched_pairs += bool(people3d)
            matched_persons += len(people3d)
            if people3d:
                pair_outcomes["matched"] += 1
            elif not left_raw.persons or not right_raw.persons:
                pair_outcomes["no_target_person"] += 1
            else:
                pair_outcomes["association_failed"] += 1

            for person in people3d:
                valid += person.valid_keypoints
                costs.append(person.association_cost)
                common.append(person.common_keypoints)

                for point in person.keypoints_3d:
                    if point["valid"]:
                        valid_by_name[point["name"]] += 1
                    else:
                        reasons[point["reason"]] += 1
                    for flag in point.get("quality_flags", []):
                        quality_flags[flag] += 1

            payload = {
                "pair_id": pair_id,
                "file_name": name,
                "pair_timestamp_sec": timestamp,
                "left_frame_id": left_raw.source_frame_id,
                "right_frame_id": right_raw.source_frame_id,
                "timestamp_skew_ms": 0.0,
                "timestamp_type": "sequence_file_index_over_fps",
                "coordinate_frame": "left_camera",
                "length_unit": calibration.length_unit,
                "max_matches": args.max_matches,
                "out_of_raw_image_bounds_policy": "reject",
                "geometry_rejected_out_of_raw_bounds_keypoints": {
                    "left": left_rejected,
                    "right": right_rejected,
                },
                "pair_outcome": (
                    "matched"
                    if people3d
                    else (
                        "no_target_person"
                        if not left_raw.persons or not right_raw.persons
                        else "association_failed"
                    )
                ),
                "left": left_raw.to_dict(),
                "right": right_raw.to_dict(),
                "persons_3d": [person.to_dict() for person in people3d],
            }
            handle.write(
                json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
            )

    count = len(left)
    summary = {
        "source": f"offline-predictions:{args.model}",
        "processed_pairs": count,
        "wall_time_sec": time.perf_counter() - started,
        "model": args.model,
        "left_prediction_json": str(args.left_json),
        "right_prediction_json": str(args.right_json),
        "calibration_file": str(args.calibration),
        "camera_model": calibration.camera_model,
        "baseline": calibration.baseline,
        "length_unit": calibration.length_unit,
        "coordinate_frame": "left_camera",
        "keypoint_threshold": args.keypoint_threshold,
        "max_association_cost": args.max_association_cost,
        "max_reprojection_error_px": args.max_reprojection_error_px,
        "triangulation_mode": args.triangulation_mode,
        "ungated_definition": (
            "all finite, in-bounds triangulations after 2-D score and association checks; "
            "depth/reprojection conditions are diagnostic quality flags only"
            if args.triangulation_mode == "ungated"
            else None
        ),
        "force_all_definition": (
            "forced best-pair association (no 0.05 gate) and no 2-D score gate; every "
            "finitely-triangulable in-bounds joint is output with depth/reprojection "
            "errors recorded as diagnostic quality_flags; out-of-bounds/non-finite 2-D "
            "points remain missing because no image observation exists to triangulate."
            if args.triangulation_mode == "force_all"
            else None
        ),
        "max_matches": args.max_matches,
        "out_of_raw_image_bounds_policy": "reject",
        "geometry_rejected_out_of_raw_bounds_keypoints": rejected_out_of_bounds_keypoints,
        "model_input_rotation": {
            "left": args.left_model_rotation,
            "right": args.right_model_rotation,
            "result_coordinate_space": "raw_camera_pixels_after_inverse_rotation",
        },
        "topdown_score_source": args.topdown_score_source,
        "matched_pairs": matched_pairs,
        "matched_person_pairs": matched_persons,
        "pair_outcomes": dict(pair_outcomes),
        "mean_association_cost": sum(costs) / len(costs) if costs else None,
        "mean_common_keypoints_per_matched_person": (
            sum(common) / len(common) if common else None
        ),
        "total_valid_3d_keypoints": valid,
        "mean_valid_3d_keypoints_per_pair": valid / count,
        "valid_3d_keypoints_by_name": dict(valid_by_name),
        "matched_keypoint_rejection_reasons": dict(reasons),
        "triangulated_quality_flags": dict(quality_flags),
        "results_jsonl": str(results_path),
        "annotated_video": None,
        "accuracy_note": (
            "Geometric self-consistency only; no human ground truth is used."
        ),
    }

    summary_path = args.output_dir / "offline_stereo_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
