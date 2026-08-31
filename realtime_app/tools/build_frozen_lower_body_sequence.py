#!/usr/bin/env python3
"""Publish a lower-body sequence without repairing unreliable observations.

The output keeps PMPose hip/knee/ankle observations as the primary chain and
keeps Sapiens2 foot observations as separate candidates.  An invalid point is
stored as invalid with its original rejection reason; this program never
interpolates, averages, or substitutes coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


PRIMARY_NAMES = (
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
)
FOOT_NAMES = (
    "left_ankle", "left_big_toe", "left_small_toe", "left_heel",
    "right_ankle", "right_big_toe", "right_small_toe", "right_heel",
)


def load_jsonl(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[int(row["pair_id"])] = row
    return rows


def point_from_pmpose(point: dict | None) -> dict:
    if point is None:
        return {"valid": False, "xyz": None, "reason": "missing_model_output"}
    valid = bool(point.get("valid"))
    return {
        "valid": valid,
        "xyz": point.get("xyz") if valid else None,
        "score": point.get("score"),
        "left_score": point.get("left_score"),
        "right_score": point.get("right_score"),
        "reprojection_error_mean_px": point.get("reprojection_error_mean_px"),
        "reason": point.get("reason"),
    }


def point_from_foot(point: dict | None) -> dict:
    if point is None:
        return {"valid": False, "xyz": None, "reason": "missing_model_output"}
    valid = bool(point.get("valid_at_reprojection_gate"))
    return {
        "valid": valid,
        "xyz": point.get("xyz_left_camera") if valid else None,
        "left_score": point.get("left_score"),
        "right_score": point.get("right_score"),
        "reprojection_error_mean_px": point.get("reprojection_error_mean_px"),
        "reason": point.get("reason"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pmpose-jsonl", type=Path, required=True)
    parser.add_argument("--foot-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest: list[dict] = []
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    pmpose_rows = load_jsonl(args.pmpose_jsonl)
    foot_rows = load_jsonl(args.foot_jsonl)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Counter] = defaultdict(Counter)
    records: list[dict] = []
    for source in manifest:
        pair_id = int(source["sequence_index"])
        pmpose = pmpose_rows.get(pair_id, {})
        feet = foot_rows.get(pair_id, {})
        persons = pmpose.get("persons_3d", [])
        pmpose_points = {point["name"]: point for point in persons[0].get("keypoints_3d", [])} if persons else {}
        foot_points = {point["name"]: point for point in feet.get("foot_points", [])}

        primary = {name: point_from_pmpose(pmpose_points.get(name)) for name in PRIMARY_NAMES}
        foot_candidates = {name: point_from_foot(foot_points.get(name)) for name in FOOT_NAMES}
        for name, point in {**primary, **{f"sapiens2_{key}": value for key, value in foot_candidates.items()}}.items():
            summary[name]["valid" if point["valid"] else "invalid"] += 1
            if not point["valid"]:
                summary[name][f"reason:{point.get('reason') or 'unspecified'}"] += 1

        left_complete = all(foot_candidates[name]["valid"] for name in ("left_big_toe", "left_small_toe", "left_heel"))
        right_complete = all(foot_candidates[name]["valid"] for name in ("right_big_toe", "right_small_toe", "right_heel"))
        records.append({
            "sequence_index": pair_id,
            "source_pair_index": int(source["source_pair_index"]),
            "file_name": source["file_name"],
            "condition": source["condition"],
            "coordinate_frame": "left_camera",
            "length_unit": "mm",
            "primary_lower_body_pmpose": primary,
            "foot_candidates_sapiens2": foot_candidates,
            "quality": {
                "pmpose_primary_valid_count": sum(point["valid"] for point in primary.values()),
                "sapiens2_foot_valid_count": sum(point["valid"] for point in foot_candidates.values()),
                "left_foot_complete": left_complete,
                "right_foot_complete": right_complete,
            },
        })

    sequence_path = args.output_dir / "frozen_lower_body_sequence.jsonl"
    with sequence_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metadata = {
        "frame_count": len(records),
        "primary_source": "PMpose hip/knee/ankle",
        "foot_candidate_source": "Sapiens2 ankle/toe/heel",
        "coordinate_frame": "left_camera",
        "length_unit": "mm",
        "missing_data_policy": "keep invalid observations missing; no interpolation, averaging, or cross-model ankle replacement",
        "point_summary": {key: dict(value) for key, value in summary.items()},
    }
    with (args.output_dir / "sequence_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"sequence": str(sequence_path), "frames": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
