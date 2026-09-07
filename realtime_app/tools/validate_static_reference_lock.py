#!/usr/bin/env python3
"""Create auditable T3 lock evidence from static-reference observations and field criteria."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_app.fixed_coordinate import RigidTransform, assess_static_reference_lock  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument(
        "--static-reference",
        type=Path,
        required=True,
        help=(
            "JSONL source records. Every row needs sample_id, reference_id, capture_session, "
            "xyz_left_camera_mm, and expected_xyz_target_mm."
        ),
    )
    parser.add_argument(
        "--criteria",
        type=Path,
        required=True,
        help="JSON object defining the field-selected static-reference acceptance criteria.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    args = parse_args()
    transform_path = args.transform.resolve()
    source_path = args.static_reference.resolve()
    criteria_path = args.criteria.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing static-reference evidence: {output_path}")

    transform = RigidTransform.from_mapping(json.loads(transform_path.read_text(encoding="utf-8")))
    if transform.status != "measured_locked":
        raise ValueError("static-reference lock evidence requires a transform marked measured_locked")
    evidence = assess_static_reference_lock(
        load_jsonl(source_path),
        transform,
        json.loads(criteria_path.read_text(encoding="utf-8")),
    )
    evidence["static_reference_input"] = str(source_path)
    evidence["criteria_input"] = str(criteria_path)
    evidence["transform_input"] = str(transform_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_path), "status": evidence["status"]}, ensure_ascii=False))
    return 0 if evidence["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
