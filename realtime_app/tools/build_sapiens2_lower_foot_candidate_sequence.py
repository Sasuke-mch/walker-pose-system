#!/usr/bin/env python3
"""Publish a Sapiens2 lower-limb/foot observation sequence without fusion.

Hip, knee and ankle observations originate from the single-person, per-joint
fisheye replay.  Toe and heel observations originate from the existing named
Sapiens2 foot replay.  The tool never averages them with PMPose, interpolates
missing frames, or alters coordinates.  Each point carries an explicit status
so later work can use only observations whose evidence it requires.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


HKA = ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
DISTAL = ("left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hka-csv", type=Path, required=True)
    parser.add_argument("--temporal-csv", type=Path, required=True)
    parser.add_argument("--foot-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def truth(value: str) -> bool:
    return str(value).strip().lower() == "true"


def read_hka(path: Path) -> dict[int, dict[str, dict]]:
    output: dict[int, dict[str, dict]] = defaultdict(dict)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            joint = row["joint"]
            if joint not in HKA:
                continue
            pair_id = int(row["pair_id"])
            if joint in output[pair_id]:
                raise RuntimeError(f"Duplicate HKA row {pair_id}/{joint}")
            valid = truth(row["direct_valid"])
            output[pair_id][joint] = {
                "valid": valid,
                "xyz_left_camera_mm": json.loads(row["direct_xyz_left_camera_mm"]) if valid and row["direct_xyz_left_camera_mm"] else None,
                "left_score": None if not row["left_score"] else float(row["left_score"]),
                "right_score": None if not row["right_score"] else float(row["right_score"]),
                "reprojection_error_mean_px": None if not row["direct_reprojection_error_px"] else float(row["direct_reprojection_error_px"]),
                "reason": row["direct_reason"] or None,
                "baseline_associated": truth(row["baseline_associated"]),
                "baseline_valid": truth(row["baseline_valid"]),
                "newly_recovered": truth(row["newly_recovered"]),
                "file_name": row["file_name"],
            }
    return output


def read_temporal(path: Path) -> set[tuple[int, str]]:
    passed: set[tuple[int, str]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if truth(row["within_baseline_temporal_envelope"]):
                passed.add((int(row["pair_id"]), row["joint"]))
    return passed


def read_foot(path: Path) -> dict[int, dict[str, dict]]:
    output: dict[int, dict[str, dict]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            pair_id = int(row["pair_id"])
            if pair_id in output:
                raise RuntimeError(f"Duplicate foot frame {pair_id}")
            output[pair_id] = {point["name"]: point for point in row.get("foot_points", [])}
    return output


def longest_run(values: list[bool]) -> int:
    best = current = 0
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    hka = read_hka(args.hka_csv.resolve())
    temporal = read_temporal(args.temporal_csv.resolve())
    feet = read_foot(args.foot_jsonl.resolve())
    frame_ids = sorted(hka)
    if frame_ids != list(range(len(frame_ids))) or set(frame_ids) != set(feet):
        raise RuntimeError("HKA and foot inputs must contain the same contiguous frame IDs")

    point_status: dict[str, list[bool]] = {name: [] for name in (*HKA, *DISTAL)}
    status_counts: dict[str, Counter] = defaultdict(Counter)
    quality_rows: list[dict] = []
    records: list[dict] = []
    for frame_id in frame_ids:
        hka_output: dict[str, dict] = {}
        for name in HKA:
            source = hka[frame_id][name]
            valid = source["valid"]
            if not valid:
                status = "missing"
            elif source["baseline_valid"]:
                status = "baseline_geometric_valid"
            elif (frame_id, name) in temporal:
                status = "temporally_confirmed"
            else:
                status = "geometric_valid_temporal_unknown"
            hka_output[name] = {
                "valid": valid,
                "status": status,
                "xyz_left_camera_mm": source["xyz_left_camera_mm"],
                "left_score": source["left_score"],
                "right_score": source["right_score"],
                "reprojection_error_mean_px": source["reprojection_error_mean_px"],
                "reason": source["reason"],
            }
            point_status[name].append(valid)
            status_counts[name][status] += 1
        distal_output: dict[str, dict] = {}
        for name in DISTAL:
            source = feet[frame_id].get(name)
            valid = bool(source and source.get("valid_at_reprojection_gate"))
            distal_output[name] = {
                "valid": valid,
                "status": "geometric_valid" if valid else "missing",
                "xyz_left_camera_mm": source.get("xyz_left_camera") if valid else None,
                "left_score": source.get("left_score") if source else None,
                "right_score": source.get("right_score") if source else None,
                "reprojection_error_mean_px": source.get("reprojection_error_mean_px") if source else None,
                "reason": source.get("reason") if source else "missing_model_output",
            }
            point_status[name].append(valid)
            status_counts[name][distal_output[name]["status"]] += 1
        quality = {
            "hka_valid_count": sum(point["valid"] for point in hka_output.values()),
            "knee_ankle_valid_count": sum(hka_output[name]["valid"] for name in ("left_knee", "right_knee", "left_ankle", "right_ankle")),
            "forefoot_valid_count": sum(distal_output[name]["valid"] for name in ("left_big_toe", "left_small_toe", "right_big_toe", "right_small_toe")),
            "left_knee_ankle_forefoot_complete": all(hka_output[name]["valid"] for name in ("left_knee", "left_ankle")) and all(distal_output[name]["valid"] for name in ("left_big_toe", "left_small_toe")),
            "right_knee_ankle_forefoot_complete": all(hka_output[name]["valid"] for name in ("right_knee", "right_ankle")) and all(distal_output[name]["valid"] for name in ("right_big_toe", "right_small_toe")),
        }
        record = {
            "pair_id": frame_id,
            "file_name": hka[frame_id]["left_hip"]["file_name"],
            "coordinate_frame": "left_camera",
            "length_unit": "mm",
            "sapiens2_hka": hka_output,
            "sapiens2_distal_foot": distal_output,
            "quality": quality,
        }
        records.append(record)
        quality_rows.append({"pair_id": frame_id, "file_name": record["file_name"], **quality})

    point_summary = []
    for name, values in point_status.items():
        point_summary.append({
            "point": name,
            "geometric_valid_frames": sum(values),
            "geometric_valid_rate": sum(values) / len(values),
            "longest_contiguous_geometric_valid_run": longest_run(values),
            **{f"status_{key}": value for key, value in sorted(status_counts[name].items())},
        })
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "sapiens2_lower_foot_candidate_sequence.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    write_csv(args.output_dir / "per_frame_quality.csv", quality_rows)
    write_csv(args.output_dir / "point_continuity_summary.csv", point_summary)
    metadata = {
        "hka_source": str(args.hka_csv.resolve()),
        "temporal_screen_source": str(args.temporal_csv.resolve()),
        "distal_foot_source": str(args.foot_jsonl.resolve()),
        "coordinate_frame": "left_camera",
        "length_unit": "mm",
        "policy": "No interpolation, smoothing, cross-model averaging, or PMPose coordinate replacement. A temporally_confirmed state means only that an already geometric-valid Sapiens2 point lies within the old-baseline adjacent-frame envelope.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = """# Sapiens2 近距离下肢与足部候选序列

该序列将已经通过鱼眼几何门限的 Sapiens2 髋、膝、踝和远端足部点按时间顺序保存。髋、膝、踝的 `temporally_confirmed` 仅表示该点前后均有旧流程观测且未出现异常跳变；没有这类邻帧证据的点仍保留为 `geometric_valid_temporal_unknown`，不被删除或补点。脚趾与脚跟保留独立的逐点几何状态。

`sapiens2_lower_foot_candidate_sequence.jsonl` 是后续下肢子链和步态候选的输入；`per_frame_quality.csv` 与 `point_continuity_summary.csv` 用于筛选连续片段。该文件不是 PMPose/Sapiens2 融合结果，不能直接用作地面接触或完整髋—膝—踝步态真值。
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "frames": len(records), "left_subchain_complete": sum(row["left_knee_ankle_forefoot_complete"] for row in quality_rows), "right_subchain_complete": sum(row["right_knee_ankle_forefoot_complete"] for row in quality_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
