#!/usr/bin/env python3
"""Join strict Sapiens2 HKA and inherited-association distal-foot observations.

This is an offline observation archive, not a contact detector or gait module.
It accepts only one already-decided stereo person per frame, preserves missing
points, and performs no interpolation, filtering, or cross-model fusion.
"""

from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
_tool_prepare_imports()

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


HKA = ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
DISTAL = ("left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-stereo-jsonl", type=Path, required=True)
    parser.add_argument("--foot-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            file_name = str(row.get("file_name", "")).strip()
            if not file_name or file_name in rows:
                raise RuntimeError(f"{path}:{line_number}: missing or duplicate file_name")
            rows[file_name] = row
    return rows


def longest_run(values: list[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def expected_association(strict_row: dict) -> dict:
    persons = strict_row.get("persons_3d", [])
    if len(persons) == 0:
        return {"status": "no_accepted_stereo_person", "left_person_id": None, "right_person_id": None}
    if len(persons) != 1:
        raise RuntimeError(f"{strict_row.get('file_name')}: expected at most one strict stereo person")
    person = persons[0]
    return {
        "status": "accepted_upstream_association",
        "left_person_id": int(person["left_person_id"]),
        "right_person_id": int(person["right_person_id"]),
    }


def point_record(point: dict | None, *, coordinate_key: str) -> dict:
    valid = bool(point and point.get("valid" if coordinate_key == "xyz" else "valid_at_reprojection_gate"))
    return {
        "valid": valid,
        "xyz_left_camera_mm": point.get(coordinate_key) if valid else None,
        "left_score": point.get("left_score") if point else None,
        "right_score": point.get("right_score") if point else None,
        "reprojection_error_mean_px": point.get("reprojection_error_mean_px") if point else None,
        "reason": point.get("reason") if point else "missing_model_output",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["point"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    strict = load_jsonl(args.strict_stereo_jsonl.resolve())
    foot = load_jsonl(args.foot_jsonl.resolve())
    if set(strict) != set(foot):
        raise RuntimeError("Strict stereo and distal-foot inputs must have identical file names")

    records: list[dict] = []
    validity: dict[str, list[bool]] = {name: [] for name in (*HKA, *DISTAL)}
    for file_name in sorted(strict):
        strict_row, foot_row = strict[file_name], foot[file_name]
        if int(strict_row["pair_id"]) != int(foot_row["pair_id"]):
            raise RuntimeError(f"{file_name}: strict stereo and foot pair IDs differ")
        association = expected_association(strict_row)
        inherited = foot_row.get("association", {})
        if any(inherited.get(key) != value for key, value in association.items()):
            raise RuntimeError(f"{file_name}: foot association does not exactly inherit strict stereo")

        keypoints = {}
        persons = strict_row.get("persons_3d", [])
        if persons:
            keypoints = {point["name"]: point for point in persons[0].get("keypoints_3d", [])}
        foot_points = {point["name"]: point for point in foot_row.get("foot_points", [])}
        hka = {name: point_record(keypoints.get(name), coordinate_key="xyz") for name in HKA}
        distal = {name: point_record(foot_points.get(name), coordinate_key="xyz_left_camera") for name in DISTAL}
        for name, point in {**hka, **distal}.items():
            validity[name].append(point["valid"])
        quality = {
            "hka_valid_count": sum(point["valid"] for point in hka.values()),
            "distal_valid_count": sum(point["valid"] for point in distal.values()),
            "left_knee_ankle_forefoot_complete": all(hka[name]["valid"] for name in ("left_knee", "left_ankle")) and all(distal[name]["valid"] for name in ("left_big_toe", "left_small_toe")),
            "right_knee_ankle_forefoot_complete": all(hka[name]["valid"] for name in ("right_knee", "right_ankle")) and all(distal[name]["valid"] for name in ("right_big_toe", "right_small_toe")),
        }
        records.append({
            "pair_id": int(strict_row["pair_id"]), "file_name": file_name,
            "coordinate_frame": strict_row.get("coordinate_frame"), "length_unit": strict_row.get("length_unit"),
            "association": association, "sapiens2_hka": hka,
            "sapiens2_distal_foot": distal, "quality": quality,
        })

    output_dir.mkdir(parents=True)
    with (output_dir / "sapiens2_lower_foot_observations.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    point_summary = [{
        "point": name, "valid_frames": sum(values), "valid_fraction": sum(values) / len(values),
        "longest_contiguous_valid_run": longest_run(values),
    } for name, values in validity.items()]
    frame_summary = [{"pair_id": row["pair_id"], "file_name": row["file_name"], **row["quality"]} for row in records]
    write_csv(output_dir / "point_continuity_summary.csv", point_summary)
    write_csv(output_dir / "per_frame_quality.csv", frame_summary)
    metadata = {
        "strict_stereo_source": str(args.strict_stereo_jsonl.resolve()),
        "distal_foot_source": str(args.foot_jsonl.resolve()),
        "frames": len(records), "coordinate_frame": "left_camera", "length_unit": "mm",
        "policy": "Strict Sapiens2 HKA and distal feet must inherit the same accepted person association. No interpolation, smoothing, fusion, contact decision, or gait output.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_dir), "frames": len(records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
