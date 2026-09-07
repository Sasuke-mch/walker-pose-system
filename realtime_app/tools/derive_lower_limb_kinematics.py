#!/usr/bin/env python3
"""Derive non-ground-referenced lower-limb kinematics from the T1 trajectory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_app.lower_limb_kinematics import (  # noqa: E402
    ANGLE_METRICS,
    DISTANCE_METRICS,
    derive_kinematics,
    summarize_kinematics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def write_csv(path: Path, rows: list[dict]) -> None:
    metric_names = DISTANCE_METRICS + ANGLE_METRICS
    fields = ["pair_id", "pair_timestamp_sec", "source_frame_status"]
    for name in metric_names:
        fields.extend((name, f"{name}_available", f"{name}_reason"))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {
                "pair_id": row["pair_id"],
                "pair_timestamp_sec": row["pair_timestamp_sec"],
                "source_frame_status": row["source_frame_status"],
            }
            for name in metric_names:
                metric = row["metrics"][name]
                output[name] = metric["value"]
                output[f"{name}_available"] = metric["available"]
                output[f"{name}_reason"] = metric["reason"]
            writer.writerow(output)


def render(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    pair_ids = [row["pair_id"] for row in rows]
    left = [row["metrics"]["left_knee_angle_deg"]["value"] for row in rows]
    right = [row["metrics"]["right_knee_angle_deg"]["value"] for row in rows]
    ankles = [row["metrics"]["ankle_separation_mm"]["value"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    axes[0].plot(pair_ids, left, label="left knee", color="#4c78a8")
    axes[0].plot(pair_ids, right, label="right knee", color="#f58518")
    axes[0].set_ylabel("angle (deg)")
    axes[0].set_title("Direct-observation lower-limb kinematics")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(pair_ids, ankles, label="ankle separation", color="#54a24b")
    axes[1].set_xlabel("pair id")
    axes[1].set_ylabel("distance (mm)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    trajectory_path = args.trajectory.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")

    rows = derive_kinematics(load_jsonl(trajectory_path))
    summary = summarize_kinematics(rows)
    output_dir.mkdir(parents=True)
    write_jsonl(output_dir / "lower_limb_kinematics.jsonl", rows)
    write_csv(output_dir / "lower_limb_kinematics.csv", rows)
    render(rows, output_dir / "lower_limb_kinematics.png")
    (output_dir / "kinematics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "input_trajectory": str(trajectory_path),
        "source_observation_policy": "direct_accepted_stereo_only",
        "coordinate_policy": "preserve left-camera coordinates; no ground transform",
        "missing_policy": "metric unavailable unless every required source joint is observed",
        "interpretation_boundary": (
            "Frame-local relative geometry only; no smoothing, interpolation, gait "
            "event, ground contact, stride or step-width claim."
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
