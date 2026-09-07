#!/usr/bin/env python3
"""Export the direct lower-limb 3-D trajectory contract from stereo JSONL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_app.lower_limb_trajectory import (  # noqa: E402
    LOWER_LIMB_JOINTS,
    normalize_stereo_records,
    summarize_trajectory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return records


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_long_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "pair_id",
        "pair_timestamp_sec",
        "left_frame_id",
        "right_frame_id",
        "timestamp_skew_ms",
        "frame_status",
        "joint_index",
        "joint_name",
        "observed_3d",
        "x_left_camera_mm",
        "y_left_camera_mm",
        "z_left_camera_mm",
        "score",
        "left_score",
        "right_score",
        "reprojection_error_left_px",
        "reprojection_error_right_px",
        "reprojection_error_mean_px",
        "reason",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for _, name in LOWER_LIMB_JOINTS:
                point = row["points"][name]
                xyz = point["xyz_left_camera_mm"] or [None, None, None]
                writer.writerow(
                    {
                        "pair_id": row["pair_id"],
                        "pair_timestamp_sec": row["pair_timestamp_sec"],
                        "left_frame_id": row["left_frame_id"],
                        "right_frame_id": row["right_frame_id"],
                        "timestamp_skew_ms": row["timestamp_skew_ms"],
                        "frame_status": row["frame_status"],
                        "joint_index": point["index"],
                        "joint_name": name,
                        "observed_3d": point["observed_3d"],
                        "x_left_camera_mm": xyz[0],
                        "y_left_camera_mm": xyz[1],
                        "z_left_camera_mm": xyz[2],
                        "score": point["score"],
                        "left_score": point["left_score"],
                        "right_score": point["right_score"],
                        "reprojection_error_left_px": point[
                            "reprojection_error_left_px"
                        ],
                        "reprojection_error_right_px": point[
                            "reprojection_error_right_px"
                        ],
                        "reprojection_error_mean_px": point[
                            "reprojection_error_mean_px"
                        ],
                        "reason": point["reason"],
                    }
                )


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")

    normalized = normalize_stereo_records(load_jsonl(input_path))
    summary = summarize_trajectory(normalized)
    output_dir.mkdir(parents=True)
    write_jsonl(output_dir / "lower_limb_trajectory.jsonl", normalized)
    write_long_csv(output_dir / "lower_limb_trajectory_long.csv", normalized)
    (output_dir / "trajectory_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "input": str(input_path),
        "outputs": {
            "trajectory_jsonl": str(output_dir / "lower_limb_trajectory.jsonl"),
            "trajectory_long_csv": str(output_dir / "lower_limb_trajectory_long.csv"),
            "summary": str(output_dir / "trajectory_summary.json"),
        },
        "joint_set": [name for _, name in LOWER_LIMB_JOINTS],
        "selection_policy": (
            "accept exactly one existing persons_3d entry; never choose among multiple"
        ),
        "coordinate_policy": "preserve accepted left-camera millimeter coordinates",
        "missing_policy": "preserve missing and rejected observations; no gap filling",
        "interpretation_boundary": (
            "Interface bring-up only; not temporal smoothing, gait-event inference, "
            "or external 3-D accuracy validation."
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_dir), **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
