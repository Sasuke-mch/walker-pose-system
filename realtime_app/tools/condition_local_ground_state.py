#!/usr/bin/env python3
"""Condition explicit local-ground observations with walker and IMU interfaces.

This is a downstream-only JSONL tool.  It never reconstructs a plane, changes
stereo points, promotes an old RANSAC candidate to a direct ground observation,
or writes a fixed world coordinate frame.  Route-1 must provide an explicit
``ground_state.status=direct`` record before the walker condition can be
evaluated.  With no IMU hardware records, it writes ``unavailable`` rather
than fabricating visual-inertial improvement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pose_app.local_ground_state import (  # noqa: E402
    GROUND_STATE_SCHEMA,
    ImuRelativeRotation,
    LocalGroundPlane,
    PropagationCriteria,
    SupportCriteria,
    VisualRelativePose,
    WalkerSupportTemplate,
    audit_walker_support,
    propagate_plane_with_visual_imu,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-jsonl", type=Path, required=True, help="Route-1 explicit local-ground JSONL.")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="New conditioned JSONL; must not already exist.")
    parser.add_argument("--walker-template", type=Path, help="Walker support template JSON; optional.")
    parser.add_argument("--support-criteria", type=Path, help="Declared support/lift condition criteria JSON.")
    parser.add_argument("--visual-motion-jsonl", type=Path, help="Accepted static-background visual relative-pose JSONL.")
    parser.add_argument("--imu-motion-jsonl", type=Path, help="Verified-timing IMU rotation JSONL; no acceleration height fields.")
    parser.add_argument("--propagation-criteria", type=Path, help="Declared visual-inertial propagation criteria JSON.")
    parser.add_argument(
        "--allow-test-walker-template",
        action="store_true",
        help="Permit parsing a test_only template, which still cannot yield a support or lift condition.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.resolve().read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        records.append(value)
    if not records:
        raise ValueError(f"JSONL source is empty: {path}")
    return records


def indexed_by_to_pair(records: list[dict[str, Any]], label: str) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for record in records:
        try:
            pair_id = int(record["to_pair_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{label} JSONL record requires integer to_pair_id") from exc
        if pair_id in indexed:
            raise ValueError(f"duplicate {label} to_pair_id: {pair_id}")
        indexed[pair_id] = record
    return indexed


def ground_pair_id(record: dict[str, Any]) -> int:
    try:
        return int(record["pair_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("ground JSONL record requires integer pair_id") from exc


def ground_timestamp(record: dict[str, Any]) -> float | None:
    value = record.get("pair_timestamp_sec")
    if value is None:
        return None
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("pair_timestamp_sec must be finite when supplied") from exc
    if not timestamp == timestamp or timestamp in {float("inf"), float("-inf")}:
        raise ValueError("pair_timestamp_sec must be finite when supplied")
    return timestamp


def unavailable_visual_inertial(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reasons": [reason],
        "interpretation": (
            "No visual-inertial local-plane propagation is claimed. IMU acceleration is not integrated "
            "into height or translation."
        ),
    }


def main() -> int:
    args = parse_args()
    output_path = args.output_jsonl.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite conditioned local-ground output: {output_path}")
    if (args.walker_template is None) != (args.support_criteria is None):
        raise ValueError("--walker-template and --support-criteria must be supplied together")
    vi_paths = (args.visual_motion_jsonl, args.imu_motion_jsonl, args.propagation_criteria)
    if any(path is not None for path in vi_paths) and not all(path is not None for path in vi_paths):
        raise ValueError("visual-inertial propagation requires visual motion, IMU motion, and propagation criteria together")

    grounds = load_jsonl(args.ground_jsonl)
    previous_pair_id: int | None = None
    for record in grounds:
        pair_id = ground_pair_id(record)
        if previous_pair_id is not None and pair_id <= previous_pair_id:
            raise ValueError("ground pair_id values must be strictly increasing")
        previous_pair_id = pair_id
        ground_timestamp(record)

    template = (
        WalkerSupportTemplate.from_mapping(load_json(args.walker_template), allow_test_template=args.allow_test_walker_template)
        if args.walker_template is not None
        else None
    )
    support_criteria = SupportCriteria.from_mapping(load_json(args.support_criteria)) if args.support_criteria else None
    visual_by_target = (
        indexed_by_to_pair(load_jsonl(args.visual_motion_jsonl), "visual motion") if args.visual_motion_jsonl else {}
    )
    imu_by_target = indexed_by_to_pair(load_jsonl(args.imu_motion_jsonl), "IMU motion") if args.imu_motion_jsonl else {}
    propagation_criteria = (
        PropagationCriteria.from_mapping(load_json(args.propagation_criteria)) if args.propagation_criteria else None
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_counts = {"direct": 0, "propagated": 0, "unavailable": 0}
    support_counts = {"support-compatible": 0, "lift-candidate": 0, "contact_unconfirmed": 0}
    previous_direct: LocalGroundPlane | None = None
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        for source in grounds:
            pair_id = ground_pair_id(source)
            source_timestamp = ground_timestamp(source)
            direct_parse_error: str | None = None
            try:
                direct_ground = LocalGroundPlane.from_mapping(source)
            except ValueError as exc:
                direct_ground = LocalGroundPlane.unavailable("invalid_ground_state_contract")
                direct_parse_error = str(exc)

            if direct_ground.status == "direct":
                selected_ground = direct_ground
                visual_inertial = unavailable_visual_inertial("current_direct_ground_observation_preferred")
                previous_direct = direct_ground
            elif previous_direct is None:
                selected_ground = LocalGroundPlane.unavailable("no_previous_direct_ground_anchor")
                visual_inertial = unavailable_visual_inertial("no_previous_direct_ground_anchor")
            elif propagation_criteria is None:
                selected_ground = LocalGroundPlane.unavailable("visual_inertial_inputs_not_supplied")
                visual_inertial = unavailable_visual_inertial("visual_inertial_inputs_not_supplied")
            else:
                visual_record = visual_by_target.get(pair_id)
                imu_record = imu_by_target.get(pair_id)
                try:
                    visual = None if visual_record is None else VisualRelativePose.from_mapping(visual_record)
                    imu = None if imu_record is None else ImuRelativeRotation.from_mapping(imu_record)
                    selected_ground, visual_inertial = propagate_plane_with_visual_imu(
                        previous_direct, visual, imu, propagation_criteria
                    )
                except ValueError as exc:
                    selected_ground = LocalGroundPlane.unavailable("invalid_visual_inertial_input")
                    visual_inertial = unavailable_visual_inertial("invalid_visual_inertial_input")
                    visual_inertial["input_error"] = str(exc)

            support = audit_walker_support(direct_ground, template, support_criteria)
            result = {
                "schema": GROUND_STATE_SCHEMA,
                "pair_id": pair_id,
                "pair_timestamp_sec": source_timestamp,
                "coordinate_frame": "left_camera",
                "length_unit": "millimeter",
                "ground_state": selected_ground.as_mapping(),
                "direct_ground_observation": direct_ground.as_mapping(),
                "walker_support": support,
                "visual_inertial_plane": visual_inertial,
                "input_lineage": {
                    "ground_jsonl": str(args.ground_jsonl.resolve()),
                    "walker_template": None if args.walker_template is None else str(args.walker_template.resolve()),
                    "visual_motion_jsonl": None if args.visual_motion_jsonl is None else str(args.visual_motion_jsonl.resolve()),
                    "imu_motion_jsonl": None if args.imu_motion_jsonl is None else str(args.imu_motion_jsonl.resolve()),
                },
                "interpretation": (
                    "Downstream local-ground conditioning only. It does not alter semantic masks, stereo matching, "
                    "triangulation, camera calibration, or direct-pose observations. It does not establish contact."
                ),
            }
            if direct_parse_error is not None:
                result["ground_input_contract_error"] = direct_parse_error
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            status_counts[selected_ground.status] += 1
            support_counts[support["status"]] += 1
    print(
        json.dumps(
            {
                "output_jsonl": str(output_path),
                "frames": len(grounds),
                "ground_state_counts": status_counts,
                "walker_support_counts": support_counts,
                "interpretation": "Counts are interface/condition states, not ground, contact, gait, or accuracy metrics.",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
