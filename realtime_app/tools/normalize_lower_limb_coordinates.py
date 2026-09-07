#!/usr/bin/env python3
"""Apply one locked left-camera-to-physical-frame transform to a T1 trajectory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_app.fixed_coordinate import (  # noqa: E402
    load_coordinate_transform,
    summarize_static_reference,
    summarize_transformed_trajectory,
    transform_trajectory_records,
)
from pose_app.lower_limb_trajectory import LOWER_LIMB_JOINTS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--static-reference",
        type=Path,
        help=(
            "Optional JSONL static-reference observations. Each row needs reference_id, "
            "xyz_left_camera_mm, and expected_xyz_target_mm."
        ),
    )
    parser.add_argument(
        "--allow-test-transform",
        action="store_true",
        help="Permit a transform explicitly marked status=test_only for software-path validation only.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_long_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "pair_id",
        "pair_timestamp_sec",
        "joint_index",
        "joint_name",
        "observed_3d",
        "x_left_camera_mm",
        "y_left_camera_mm",
        "z_left_camera_mm",
        "x_target_mm",
        "y_target_mm",
        "z_target_mm",
        "target_coordinate_status",
        "target_coordinate_reason",
        "source_reason",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for _, name in LOWER_LIMB_JOINTS:
                point = row["points"][name]
                source = point["xyz_left_camera_mm"] or [None, None, None]
                target = point["xyz_target_mm"] or [None, None, None]
                writer.writerow(
                    {
                        "pair_id": row["pair_id"],
                        "pair_timestamp_sec": row["pair_timestamp_sec"],
                        "joint_index": point["index"],
                        "joint_name": name,
                        "observed_3d": point["observed_3d"],
                        "x_left_camera_mm": source[0],
                        "y_left_camera_mm": source[1],
                        "z_left_camera_mm": source[2],
                        "x_target_mm": target[0],
                        "y_target_mm": target[1],
                        "z_target_mm": target[2],
                        "target_coordinate_status": point["target_coordinate_status"],
                        "target_coordinate_reason": point["target_coordinate_reason"],
                        "source_reason": point["reason"],
                    }
                )


def main() -> int:
    args = parse_args()
    trajectory_path = args.trajectory.resolve()
    transform_path = args.transform.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")

    transform = load_coordinate_transform(
        transform_path,
        allow_test_transform=args.allow_test_transform,
    )
    rows = transform_trajectory_records(load_jsonl(trajectory_path), transform)
    summary = summarize_transformed_trajectory(rows, transform)
    static_summary = None
    if args.static_reference is not None:
        static_summary = summarize_static_reference(
            load_jsonl(args.static_reference.resolve()), transform
        )
        summary["static_reference_stability"] = static_summary

    output_dir.mkdir(parents=True)
    write_jsonl(output_dir / "lower_limb_trajectory_target_frame.jsonl", rows)
    write_long_csv(output_dir / "lower_limb_trajectory_target_frame.csv", rows)
    (output_dir / "coordinate_normalization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "input_trajectory": str(trajectory_path),
        "transform": str(transform_path),
        "transform_status": transform.status,
        "static_reference": str(args.static_reference.resolve()) if args.static_reference else None,
        "source_observation_policy": "direct_accepted_stereo_only",
        "coordinate_policy": (
            "preserve source xyz_left_camera_mm and add xyz_target_mm using one locked "
            "proper rigid transform; no point correction, smoothing, or interpolation"
        ),
        "interpretation_boundary": (
            "Target-frame coordinates are not step length, step width, foot height, "
            "gait event, or external accuracy results."
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output_dir), **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
