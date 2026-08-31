#!/usr/bin/env python3
"""Compare PMPose and Sapiens2 ankle choices for a prospective hybrid leg.

Only frames where PMPose knee/ankle and the Sapiens2 ankle all pass their
respective reprojection gates enter the paired comparison.  Bone-length spread
is a structural stability diagnostic, not an absolute-accuracy claim.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean_mm": None, "median_mm": None, "mad_mm": None, "p10_mm": None, "p90_mm": None}
    data = np.asarray(values, dtype=float)
    median = float(np.median(data))
    return {
        "count": len(values),
        "mean_mm": float(np.mean(data)),
        "median_mm": median,
        "mad_mm": float(np.median(np.abs(data - median))),
        "p10_mm": float(np.percentile(data, 10)),
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
    sapiens = read_jsonl(args.sapiens_foot_results.resolve())
    rows = []
    for foot_result in sapiens:
        pmpose = pmpose_by_name.get(foot_result["file_name"])
        if pmpose is None or len(pmpose.get("persons_3d", [])) != 1:
            continue
        primary = {point["name"]: point for point in pmpose["persons_3d"][0]["keypoints_3d"]}
        feet = {point["name"]: point for point in foot_result["foot_points"]}
        for side in ("left", "right"):
            knee, ankle = primary[f"{side}_knee"], primary[f"{side}_ankle"]
            s_ankle = feet[f"{side}_ankle"]
            if not knee["valid"] or not ankle["valid"] or not s_ankle["valid_at_reprojection_gate"]:
                continue
            knee_xyz = np.asarray(knee["xyz"], dtype=float)
            pmpose_ankle = np.asarray(ankle["xyz"], dtype=float)
            sapiens_ankle = np.asarray(s_ankle["xyz_left_camera"], dtype=float)
            pm_length = float(np.linalg.norm(knee_xyz - pmpose_ankle))
            hybrid_length = float(np.linalg.norm(knee_xyz - sapiens_ankle))
            rows.append({
                "pair_id": foot_result["pair_id"],
                "file_name": foot_result["file_name"],
                "distance_condition": foot_result["distance_condition"],
                "side": side,
                "pmpose_knee_to_ankle_mm": pm_length,
                "hybrid_knee_to_sapiens_ankle_mm": hybrid_length,
                "hybrid_minus_pmpose_mm": hybrid_length - pm_length,
                "pmpose_knee_reprojection_px": knee["reprojection_error_mean_px"],
                "pmpose_ankle_reprojection_px": ankle["reprojection_error_mean_px"],
                "sapiens_ankle_reprojection_px": s_ankle["reprojection_error_mean_px"],
            })

    output.mkdir(parents=True)
    with (output / "per_pair_lower_leg_length.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = []
    for distance in ["all", *sorted({row["distance_condition"] for row in rows})]:
        for side in ("left", "right"):
            subset = [row for row in rows if row["side"] == side and (distance == "all" or row["distance_condition"] == distance)]
            for source, field in (("PMPose ankle", "pmpose_knee_to_ankle_mm"), ("Sapiens2 ankle", "hybrid_knee_to_sapiens_ankle_mm")):
                summary.append({"distance_condition": distance, "side": side, "ankle_source": source, **summarize([row[field] for row in subset])})
    with (output / "lower_leg_structure_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    metadata = {
        "pmpose_results": str(args.pmpose_results.resolve()),
        "sapiens_foot_results": str(args.sapiens_foot_results.resolve()),
        "unit": "mm",
        "interpretation_boundary": "A smaller lower-leg MAD supports consistency only. It does not establish which ankle is anatomically more accurate, and it does not authorize averaging model outputs.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"paired_lower_legs": len(rows), "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
