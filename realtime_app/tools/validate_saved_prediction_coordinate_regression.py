"""Validate saved top-down predictions before raw-fisheye stereo replay.

This is deliberately a coordinate-contract check, not a pose or 3-D accuracy
evaluation.  It verifies that every saved upright-model keypoint can be
inverse-rotated to the frozen raw-fisheye image and forward-rotated back
without numerical drift.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_app.rotation import (  # noqa: E402
    ROTATION_CHOICES,
    model_image_size,
    model_to_raw_point,
    raw_to_model_point,
)


def _load_cases(args: argparse.Namespace) -> list[tuple[str, str, str, Path]]:
    return [
        ("C3", "left", args.left_model_rotation, args.c3_left_json.resolve()),
        ("C3", "right", args.right_model_rotation, args.c3_right_json.resolve()),
        ("D1", "left", args.left_model_rotation, args.d1_left_json.resolve()),
        ("D1", "right", args.right_model_rotation, args.d1_right_json.resolve()),
    ]


def _is_inclusive_pixel(value: float, upper: int) -> bool:
    return 0.0 <= value <= float(upper - 1)


def _check_case(
    condition: str,
    side: str,
    rotation: str,
    path: Path,
    raw_width: int,
    raw_height: int,
) -> tuple[dict, set[str], list[dict]]:
    root = json.loads(path.read_text(encoding="utf-8-sig"))
    images = root.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError(f"{path}: expected a non-empty images list.")

    model_width, model_height = model_image_size(raw_width, raw_height, rotation)
    names: set[str] = set()
    violations: list[dict] = []
    person_count = 0
    keypoint_count = 0
    out_of_bounds_model_keypoints = 0
    geometry_rejected_out_of_bounds_keypoints = 0
    max_round_trip_error = 0.0

    def violation(file_name: str, location: str, reason: str) -> None:
        violations.append(
            {
                "condition": condition,
                "side": side,
                "file_name": file_name,
                "location": location,
                "reason": reason,
            }
        )

    for frame_index, image in enumerate(images):
        file_name = image.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            raise ValueError(f"{path}: image {frame_index} has no file_name.")
        if file_name in names:
            violation(file_name, "frame", "duplicate_file_name")
        names.add(file_name)

        boxes = image.get("output_bboxes")
        keypoint_sets = image.get("keypoints")
        if not isinstance(boxes, list) or not isinstance(keypoint_sets, list):
            violation(file_name, "frame", "missing_output_boxes_or_keypoints")
            continue
        if len(boxes) != len(keypoint_sets):
            violation(file_name, "frame", "misaligned_output_boxes_and_keypoints")
            continue

        for person_index, (bbox, points) in enumerate(zip(boxes, keypoint_sets)):
            person_count += 1
            bbox_location = f"person[{person_index}].bbox"
            if not isinstance(bbox, list) or len(bbox) < 4:
                violation(file_name, bbox_location, "invalid_bbox")
            else:
                values = [float(value) for value in bbox[:4]]
                if not all(math.isfinite(value) for value in values):
                    violation(file_name, bbox_location, "non_finite_bbox")
                elif (
                    values[0] > values[2]
                    or values[1] > values[3]
                    or not _is_inclusive_pixel(values[0], model_width)
                    or not _is_inclusive_pixel(values[2], model_width)
                    or not _is_inclusive_pixel(values[1], model_height)
                    or not _is_inclusive_pixel(values[3], model_height)
                ):
                    violation(file_name, bbox_location, "bbox_outside_upright_model_bounds")

            if not isinstance(points, list) or not points:
                violation(file_name, f"person[{person_index}]", "missing_keypoints")
                continue
            for point_index, point in enumerate(points):
                keypoint_count += 1
                point_location = f"person[{person_index}].keypoint[{point_index}]"
                if not isinstance(point, list) or len(point) < 3:
                    violation(file_name, point_location, "invalid_keypoint")
                    continue
                x_model, y_model, score = [float(value) for value in point[:3]]
                if not all(math.isfinite(value) for value in (x_model, y_model, score)):
                    violation(file_name, point_location, "non_finite_keypoint")
                    continue
                model_in_bounds = (
                    _is_inclusive_pixel(x_model, model_width)
                    and _is_inclusive_pixel(y_model, model_height)
                )
                if not model_in_bounds:
                    out_of_bounds_model_keypoints += 1
                x_raw, y_raw = model_to_raw_point(
                    x_model, y_model, raw_width, raw_height, rotation
                )
                raw_in_bounds = (
                    _is_inclusive_pixel(x_raw, raw_width)
                    and _is_inclusive_pixel(y_raw, raw_height)
                )
                if not raw_in_bounds:
                    # The replay rejects this observation from association and
                    # triangulation without clipping it. Retain it here as an
                    # auditable diagnostic rather than a map failure.
                    geometry_rejected_out_of_bounds_keypoints += 1
                x_roundtrip, y_roundtrip = raw_to_model_point(
                    x_raw, y_raw, raw_width, raw_height, rotation
                )
                max_round_trip_error = max(
                    max_round_trip_error,
                    abs(x_roundtrip - x_model),
                    abs(y_roundtrip - y_model),
                )

    return (
        {
            "condition": condition,
            "side": side,
            "prediction_json": str(path),
            "rotation": rotation,
            "model_image_size": [model_width, model_height],
            "raw_fisheye_size": [raw_width, raw_height],
            "frames": len(images),
            "persons": person_count,
            "keypoints": keypoint_count,
            "out_of_bounds_upright_model_keypoints": out_of_bounds_model_keypoints,
            "geometry_rejected_out_of_raw_bounds_keypoints": (
                geometry_rejected_out_of_bounds_keypoints
            ),
            "max_point_round_trip_error_px": max_round_trip_error,
            "violations": len(violations),
        },
        names,
        violations,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Coordinate regression for saved C3/D1 top-down predictions."
    )
    parser.add_argument("--c3-left-json", required=True, type=Path)
    parser.add_argument("--c3-right-json", required=True, type=Path)
    parser.add_argument("--d1-left-json", required=True, type=Path)
    parser.add_argument("--d1-right-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--raw-width", type=int, default=1920)
    parser.add_argument("--raw-height", type=int, default=1080)
    parser.add_argument(
        "--left-model-rotation", choices=ROTATION_CHOICES, default="ccw90"
    )
    parser.add_argument(
        "--right-model-rotation", choices=ROTATION_CHOICES, default="cw90"
    )
    args = parser.parse_args()
    if args.raw_width <= 0 or args.raw_height <= 0:
        parser.error("raw-width and raw-height must be positive.")

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    name_sets: dict[str, set[str]] = {}
    violations: list[dict] = []
    for condition, side, rotation, path in _load_cases(args):
        summary, names, case_violations = _check_case(
            condition,
            side,
            rotation,
            path,
            args.raw_width,
            args.raw_height,
        )
        summaries.append(summary)
        name_sets[f"{condition}_{side}"] = names
        violations.extend(case_violations)

    reference_names = name_sets["C3_left"]
    name_sets_match = {
        label: names == reference_names for label, names in name_sets.items()
    }
    if not all(name_sets_match.values()):
        for label, matches in name_sets_match.items():
            if not matches:
                violations.append(
                    {
                        "condition": label.split("_", maxsplit=1)[0],
                        "side": label.split("_", maxsplit=1)[1],
                        "file_name": "",
                        "location": "dataset",
                        "reason": "file_name_set_differs_from_c3_left",
                    }
                )

    passed = not violations and all(
        summary["max_point_round_trip_error_px"] <= 1e-9 for summary in summaries
    )
    output = {
        "purpose": (
            "Coordinate-contract regression only: saved upright-model predictions "
            "must inverse-rotate to the frozen raw-fisheye pixel system before "
            "stereo geometry."
        ),
        "raw_fisheye_size": [args.raw_width, args.raw_height],
        "frozen_rotations": {
            "left": args.left_model_rotation,
            "right": args.right_model_rotation,
        },
        "file_name_sets_match": name_sets_match,
        "case_summaries": summaries,
        "violation_count": len(violations),
        "passed": passed,
        "interpretation_boundary": (
            "Passing this regression establishes only coordinate mapping and "
            "saved-prediction integrity. It does not validate 2-D keypoint "
            "accuracy, person identity, calibration, or 3-D accuracy."
        ),
    }
    (args.output_dir / "coordinate_regression_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    with (args.output_dir / "coordinate_regression_cases.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (args.output_dir / "coordinate_regression_violations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["condition", "side", "file_name", "location", "reason"],
        )
        writer.writeheader()
        writer.writerows(violations)
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
    if not passed:
        raise SystemExit("Coordinate regression failed; do not start geometry replay.")


if __name__ == "__main__":
    main()
