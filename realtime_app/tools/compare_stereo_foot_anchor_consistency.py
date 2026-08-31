#!/usr/bin/env python3
"""Measure whether PMPose and Sapiens2 place the same ankle similarly in 3-D.

The comparison is a compatibility diagnostic before mixing PMPose leg points
with Sapiens2 toe/heel points.  It is not a 3-D accuracy measure because both
tracks originate from learned 2-D observations on the same images.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ANKLES = ("left_ankle", "right_ankle")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stat(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean_mm": None, "median_mm": None, "p90_mm": None}
    data = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "mean_mm": float(np.mean(data)),
        "median_mm": float(np.median(data)),
        "p90_mm": float(np.percentile(data, 90)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmpose-results", type=Path, required=True)
    parser.add_argument("--sapiens-foot-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    pmpose_by_name = {row["file_name"]: row for row in read_jsonl(args.pmpose_results.resolve())}
    foot_rows = read_jsonl(args.sapiens_foot_results.resolve())
    if set(pmpose_by_name) != {row["file_name"] for row in foot_rows}:
        raise RuntimeError("PMPose and Sapiens2 result file names must match exactly")

    comparisons = []
    for foot_result in foot_rows:
        pmpose = pmpose_by_name[foot_result["file_name"]]
        people = pmpose.get("persons_3d", [])
        if len(people) != 1:
            continue
        pmpose_points = {point["name"]: point for point in people[0]["keypoints_3d"]}
        foot_points = {point["name"]: point for point in foot_result["foot_points"]}
        for ankle in ANKLES:
            primary = pmpose_points[ankle]
            foot = foot_points[ankle]
            if not primary["valid"] or not foot["valid_at_reprojection_gate"]:
                continue
            p_xyz = np.asarray(primary["xyz"], dtype=float)
            s_xyz = np.asarray(foot["xyz_left_camera"], dtype=float)
            comparisons.append({
                "pair_id": foot_result["pair_id"],
                "file_name": foot_result["file_name"],
                "distance_condition": foot_result["distance_condition"],
                "joint": ankle,
                "pmpose_xyz_left_camera": json.dumps(primary["xyz"]),
                "sapiens2_xyz_left_camera": json.dumps(foot["xyz_left_camera"]),
                "distance_between_models_mm": float(np.linalg.norm(p_xyz - s_xyz)),
                "pmpose_reprojection_error_px": primary["reprojection_error_mean_px"],
                "sapiens2_reprojection_error_px": foot["reprojection_error_mean_px"],
            })

    output.mkdir(parents=True)
    with (output / "per_pair_ankle_difference.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    summary = []
    distances = ["all", *sorted({row["distance_condition"] for row in comparisons})]
    for distance_name in distances:
        for joint in ANKLES:
            values = [
                row["distance_between_models_mm"] for row in comparisons
                if row["joint"] == joint and (distance_name == "all" or row["distance_condition"] == distance_name)
            ]
            summary.append({"distance_condition": distance_name, "joint": joint, **stat(values)})
    with (output / "ankle_difference_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    metadata = {
        "pmpose_results": str(args.pmpose_results.resolve()),
        "sapiens_foot_results": str(args.sapiens_foot_results.resolve()),
        "unit": "mm",
        "interpretation_boundary": "Inter-model ankle distance is a compatibility diagnostic, not external accuracy. Do not merge the tracks by averaging or replacement until an explicit anchor policy is selected.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"comparisons": len(comparisons), "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
