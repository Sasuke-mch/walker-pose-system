#!/usr/bin/env python3
"""Audit a saved strict stereo run against a frozen protocol without changing it."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_app.calibration import StereoCalibration  # noqa: E402
from pose_app.stereo_quality_audit import audit_saved_stereo_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--results-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--frozen-protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(not row.get("file_name") for row in rows):
        raise ValueError(f"Manifest has no usable file_name rows: {path}")
    if any(row.get("abs_host_delta_ms") in (None, "") for row in rows):
        raise ValueError(f"Manifest lacks abs_host_delta_ms host-pairing metadata: {path}")
    return rows


def validate_protocol(protocol: dict[str, Any], summary: dict[str, Any], records: list[dict[str, Any]], calibration_path: Path, manifest_path: Path, manifest_rows: list[dict[str, str]]) -> dict[str, Any]:
    """Fail closed when a run differs from the explicitly frozen controls."""
    expected = protocol.get("frozen_controls")
    if not isinstance(expected, dict):
        raise ValueError("Frozen protocol must contain frozen_controls")
    checks = {
        "model": summary.get("model") == expected.get("model"),
        "keypoint_threshold": summary.get("keypoint_threshold") == expected.get("keypoint_threshold"),
        "max_association_cost": summary.get("max_association_cost") == expected.get("max_association_cost"),
        "max_reprojection_error_px": summary.get("max_reprojection_error_px") == expected.get("max_reprojection_error_px"),
        "triangulation_mode": summary.get("triangulation_mode") == expected.get("triangulation_mode"),
        "max_matches": summary.get("max_matches") == expected.get("max_matches"),
        "left_rotation": (summary.get("model_input_rotation") or {}).get("left") == expected.get("left_rotation"),
        "right_rotation": (summary.get("model_input_rotation") or {}).get("right") == expected.get("right_rotation"),
        "raw_bounds_policy": summary.get("out_of_raw_image_bounds_policy") == expected.get("out_of_raw_image_bounds_policy"),
        "calibration_path": Path(str(summary.get("calibration_file"))).resolve() == calibration_path.resolve(),
        "manifest_count": len(manifest_rows) == len(records),
        "manifest_file_order": [row["file_name"] for row in manifest_rows] == [str(record["file_name"]) for record in records],
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError(f"Frozen protocol mismatch: {', '.join(failures)}")
    return {"protocol_id": protocol.get("protocol_id"), "status": "passed", "checks": checks, "manifest": str(manifest_path.resolve()), "records": len(records)}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    protocol = load_json(args.frozen_protocol.resolve())
    summary = load_json(args.summary_json.resolve())
    records = load_records(args.results_jsonl.resolve())
    manifest_rows = load_manifest(args.manifest.resolve())
    verification = validate_protocol(protocol, summary, records, args.calibration.resolve(), args.manifest.resolve(), manifest_rows)
    calibration = StereoCalibration.load(args.calibration.resolve())
    audit = audit_saved_stereo_records(
        records,
        calibration=calibration,
        keypoint_threshold=float(protocol["frozen_controls"]["keypoint_threshold"]),
        manifest_by_file_name={row["file_name"]: row for row in manifest_rows},
    )
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "frame_quality.csv", audit["frame_rows"])
    write_csv(args.output_dir / "joint_quality_summary.csv", audit["joint_rows"])
    write_csv(args.output_dir / "rejection_reason_summary.csv", audit["rejection_rows"])
    write_csv(args.output_dir / "bone_stability_summary.csv", audit["bone_rows"])
    (args.output_dir / "frozen_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "protocol_verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "run_quality_summary.json").write_text(json.dumps(audit["summary"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), **audit["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
