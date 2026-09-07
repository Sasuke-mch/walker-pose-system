#!/usr/bin/env python3
"""Derive conservative non-contact T4 candidates from T2 kinematics JSONL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_app.gait_candidates import KINEMATIC_SIGNALS, derive_gait_candidates  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinematics", type=Path, required=True)
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


def write_per_frame_csv(path: Path, rows: list[dict]) -> None:
    fields = ["pair_id", "pair_timestamp_sec", "source_frame_status"]
    for signal in KINEMATIC_SIGNALS:
        fields.extend((signal, f"{signal}_available", f"{signal}_reason"))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {
                "pair_id": row["pair_id"],
                "pair_timestamp_sec": row.get("pair_timestamp_sec"),
                "source_frame_status": row.get("source_frame_status"),
            }
            for signal in KINEMATIC_SIGNALS:
                metric = row["metrics"][signal]
                output[signal] = metric["value"]
                output[f"{signal}_available"] = metric["available"]
                output[f"{signal}_reason"] = metric["reason"]
            writer.writerow(output)


def write_events_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "candidate_id", "candidate_type", "signal", "turning_point", "pair_id",
        "pair_timestamp_sec", "value", "unit", "support_pair_ids", "source_observation_policy",
        "interpretation",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["support_pair_ids"] = ";".join(str(value) for value in row["support_pair_ids"])
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    input_path = args.kinematics.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")

    output = derive_gait_candidates(load_jsonl(input_path))
    output_dir.mkdir(parents=True)
    write_per_frame_csv(output_dir / "t4_source_signals.csv", output["records"])
    write_jsonl(output_dir / "gait_event_candidates.jsonl", output["events"])
    write_events_csv(output_dir / "gait_event_candidates.csv", output["events"])
    (output_dir / "gait_parameter_candidates.json").write_text(
        json.dumps(output["summary"]["gait_parameter_candidates"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "gait_candidate_summary.json").write_text(
        json.dumps(output["summary"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "input_kinematics": str(input_path),
        "source_observation_policy": "direct_accepted_stereo_only",
        "event_policy": "strict three-adjacent-frame raw local extrema; no smoothing or gap bridging",
        "contact_policy": "no heel strike, toe-off, support, swing, or gait-cycle labels",
        "parameter_policy": "step, width, height, cycle duration, and cadence remain unavailable until their prerequisites are independently met",
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output_dir), **output["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
