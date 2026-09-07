#!/usr/bin/env python3
"""Summarize full-image PMPose/ProbPose agreement and timing artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmpose-evaluation", type=Path, required=True)
    parser.add_argument("--probpose-evaluation", type=Path, required=True)
    parser.add_argument("--pmpose-left-timing", type=Path, required=True)
    parser.add_argument("--pmpose-right-timing", type=Path, required=True)
    parser.add_argument("--probpose-left-timing", type=Path, required=True)
    parser.add_argument("--probpose-right-timing", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def truth(value: str) -> bool:
    return value.strip().lower() == "true"


def pck(errors: list[float], threshold: float) -> float | None:
    return float(np.mean(np.asarray(errors) <= threshold)) if errors else None


def summarize_agreement(label: str, evaluation: Path) -> dict:
    data = rows(evaluation / "per_keypoint_relative_error_all_reference.csv")
    reference = [row for row in data if truth(row["reference_usable"])]
    compared = [row for row in reference if truth(row["model_usable"])]
    errors = [float(row["error_px"]) for row in compared if row["error_px"] != ""]
    if len(errors) != len(compared):
        raise RuntimeError(f"{label}: compared row without an error")
    return {
        "model": label,
        "reference_usable_keypoints": len(reference),
        "compared_keypoints": len(compared),
        "relative_coverage": len(compared) / len(reference) if reference else None,
        "mean_relative_error_px": float(np.mean(errors)) if errors else None,
        "median_relative_error_px": float(np.median(errors)) if errors else None,
        "p90_relative_error_px": float(np.percentile(errors, 90)) if errors else None,
        "relative_pck_25px": pck(errors, 25.0),
        "relative_pck_50px": pck(errors, 50.0),
    }


def summarize_speed(label: str, timing_paths: list[Path]) -> dict:
    times: list[float] = []
    for path in timing_paths:
        for row in rows(path):
            times.append(float(row["elapsed_ms"]))
    if not times:
        raise RuntimeError(f"{label}: no timing records")
    return {
        "model": label,
        "measured_calls": len(times),
        "mean_ms_per_image": float(np.mean(times)),
        "median_ms_per_image": float(np.median(times)),
        "p95_ms_per_image": float(np.percentile(times, 95)),
        "pose_only_fps": float(1000.0 / np.mean(times)),
        "timing_scope": "GPU-synchronized top-down inference including image decode and preprocessing; excludes model loading and prompt-cache construction.",
    }


def write_csv(path: Path, data: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    agreement = [
        summarize_agreement("PMPose-b fixed full-image prompt", args.pmpose_evaluation.resolve()),
        summarize_agreement("ProbPose-s fixed full-image prompt", args.probpose_evaluation.resolve()),
    ]
    speed = [
        summarize_speed("PMPose-b fixed full-image prompt", [args.pmpose_left_timing.resolve(), args.pmpose_right_timing.resolve()]),
        summarize_speed("ProbPose-s fixed full-image prompt", [args.probpose_left_timing.resolve(), args.probpose_right_timing.resolve()]),
    ]
    output.mkdir(parents=True)
    write_csv(output / "relative_agreement_summary.csv", agreement)
    write_csv(output / "speed_summary.csv", speed)
    metadata = {
        "input_mode": "fixed_full_image_prompt_no_detector",
        "models": [item["model"] for item in agreement],
        "agreement_boundary": "All errors are agreement with the saved Sapiens2 engineering reference, not independent 2-D accuracy or 3-D accuracy.",
        "speed_boundary": speed[0]["timing_scope"],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "agreement": agreement, "speed": speed}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
