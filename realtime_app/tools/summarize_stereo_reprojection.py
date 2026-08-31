#!/usr/bin/env python3
"""Summarize calibrated-stereo reprojection residuals by condition and distance.

The tool keeps the ungated finite, positive-depth residual distribution separate
from the <= gate accepted distribution.  This prevents a low accepted-point
mean from being mistaken for an improvement when a condition merely rejects
more points.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


LOWER_BODY = {"left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"}


def parse_condition(value: str) -> tuple[str, Path]:
    fields = value.split("|", 1)
    if len(fields) != 2 or not fields[0] or not fields[1]:
        raise ValueError("--condition must be NAME|RESULTS_JSONL")
    return fields[0], Path(fields[1])


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else None


def stat_record(rows: list[dict], label: str, condition: str, distance: str, gate: float) -> dict:
    associated_pairs = {row["pair_id"] for row in rows if row["associated"]}
    candidates = [row for row in rows if row["positive_finite"]]
    accepted = [row for row in candidates if row["accepted"]]
    residuals = [row["error"] for row in candidates]
    accepted_residuals = [row["error"] for row in accepted]
    return {
        "condition": condition,
        "distance_condition": distance,
        "joint_subset": label,
        "processed_pairs": len({row["pair_id"] for row in rows}),
        "associated_pairs": len(associated_pairs),
        "positive_finite_points": len(candidates),
        "accepted_points_at_gate": len(accepted),
        "accepted_fraction_of_positive_finite": len(accepted) / len(candidates) if candidates else None,
        "gate_px": gate,
        "ungated_mean_reprojection_px": float(np.mean(residuals)) if residuals else None,
        "ungated_median_reprojection_px": float(np.median(residuals)) if residuals else None,
        "ungated_p90_reprojection_px": percentile(residuals, 90),
        "accepted_mean_reprojection_px": float(np.mean(accepted_residuals)) if accepted_residuals else None,
        "accepted_median_reprojection_px": float(np.median(accepted_residuals)) if accepted_residuals else None,
        "accepted_p90_reprojection_px": percentile(accepted_residuals, 90),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gate-px", type=float, default=10.0)
    parser.add_argument("--condition", action="append", required=True, metavar="NAME|RESULTS_JSONL")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.gate_px <= 0:
        raise ValueError("--gate-px must be positive")
    with args.selection_manifest.open(encoding="utf-8-sig", newline="") as handle:
        selection = list(csv.DictReader(handle))
    condition_by_name = {row["file_name"]: row["condition"] for row in selection}
    if not condition_by_name:
        raise RuntimeError("Selection manifest has no rows")

    summary, detail = [], []
    source_paths = {}
    for name, path in (parse_condition(item) for item in args.condition):
        path = path.resolve()
        source_paths[name] = str(path)
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            file_name = item["file_name"]
            if file_name not in condition_by_name:
                raise RuntimeError(f"{name}: {file_name} missing from selection manifest")
            associated = bool(item.get("persons_3d"))
            points = item["persons_3d"][0].get("keypoints_3d", []) if associated else []
            for point in points:
                error = point.get("reprojection_error_mean_px")
                depth_left, depth_right = point.get("depth_left"), point.get("depth_right")
                positive_finite = (
                    error is not None and depth_left is not None and depth_right is not None
                    and np.isfinite(float(error)) and float(depth_left) > 0 and float(depth_right) > 0
                )
                rows.append({
                    "pair_id": int(item["pair_id"]), "file_name": file_name,
                    "distance": condition_by_name[file_name], "joint": point["name"],
                    "associated": associated, "positive_finite": positive_finite,
                    "accepted": bool(positive_finite and float(error) <= args.gate_px),
                    "error": None if error is None else float(error),
                })
            # Retain association state even if all joints were below score threshold.
            if not points:
                rows.append({"pair_id": int(item["pair_id"]), "file_name": file_name,
                             "distance": condition_by_name[file_name], "joint": "__none__",
                             "associated": associated, "positive_finite": False,
                             "accepted": False, "error": None})
        for distance in ("all", "far_3m", "mid_2m", "near_1p3m"):
            subset = rows if distance == "all" else [row for row in rows if row["distance"] == distance]
            for label, allowed in (("all_joints", None), ("lower_body_6", LOWER_BODY)):
                current = subset if allowed is None else [row for row in subset if row["joint"] in allowed]
                summary.append(stat_record(current, label, name, distance, args.gate_px))
        detail.extend(dict(row, condition=name) for row in rows)

    args.output_dir.mkdir(parents=True)
    fields = list(summary[0])
    with (args.output_dir / "reprojection_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(summary)
    detail_fields = list(detail[0])
    with (args.output_dir / "per_keypoint_reprojection.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields); writer.writeheader(); writer.writerows(detail)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "selection_manifest": str(args.selection_manifest.resolve()), "conditions": source_paths,
        "gate_px": args.gate_px,
        "interpretation": "Ungated residuals include every associated, finite, positive-depth point. Accepted residuals are restricted to residual <= gate. Neither is 3-D accuracy without ground truth.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"conditions": len(source_paths), "summary_rows": len(summary), "output": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
