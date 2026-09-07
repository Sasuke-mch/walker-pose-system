"""One-directory T1--T4 post-processing for a completed stereo JSONL run."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .fixed_coordinate import (
    RigidTransform,
    summarize_transformed_trajectory,
    transform_trajectory_records,
)
from .gait_candidates import KINEMATIC_SIGNALS, derive_gait_candidates
from .lower_limb_kinematics import (
    ANGLE_METRICS,
    DISTANCE_METRICS,
    derive_kinematics,
    summarize_kinematics,
)
from .lower_limb_trajectory import LOWER_LIMB_JOINTS, normalize_stereo_records, summarize_trajectory


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _write_trajectory_csv(path: Path, rows: list[dict[str, Any]], *, target: bool = False) -> None:
    fields = [
        "pair_id", "pair_timestamp_sec", "joint_index", "joint_name", "observed_3d",
        "x_left_camera_mm", "y_left_camera_mm", "z_left_camera_mm",
    ]
    if target:
        fields.extend(("x_target_mm", "y_target_mm", "z_target_mm", "target_coordinate_status", "target_coordinate_reason"))
    fields.append("reason")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for _, name in LOWER_LIMB_JOINTS:
                point = row["points"][name]
                source = point["xyz_left_camera_mm"] or [None, None, None]
                output: dict[str, Any] = {
                    "pair_id": row["pair_id"],
                    "pair_timestamp_sec": row.get("pair_timestamp_sec"),
                    "joint_index": point["index"],
                    "joint_name": name,
                    "observed_3d": point["observed_3d"],
                    "x_left_camera_mm": source[0],
                    "y_left_camera_mm": source[1],
                    "z_left_camera_mm": source[2],
                    "reason": point["reason"],
                }
                if target:
                    target_xyz = point["xyz_target_mm"] or [None, None, None]
                    output.update(
                        {
                            "x_target_mm": target_xyz[0],
                            "y_target_mm": target_xyz[1],
                            "z_target_mm": target_xyz[2],
                            "target_coordinate_status": point["target_coordinate_status"],
                            "target_coordinate_reason": point["target_coordinate_reason"],
                        }
                    )
                writer.writerow(output)


def _write_kinematics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    metrics = DISTANCE_METRICS + ANGLE_METRICS
    fields = ["pair_id", "pair_timestamp_sec", "source_frame_status"]
    for metric in metrics:
        fields.extend((metric, f"{metric}_available", f"{metric}_reason"))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output: dict[str, Any] = {
                "pair_id": row["pair_id"],
                "pair_timestamp_sec": row.get("pair_timestamp_sec"),
                "source_frame_status": row.get("source_frame_status"),
            }
            for metric in metrics:
                value = row["metrics"][metric]
                output[metric] = value["value"]
                output[f"{metric}_available"] = value["available"]
                output[f"{metric}_reason"] = value["reason"]
            writer.writerow(output)


def _write_event_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "candidate_id", "candidate_type", "signal", "turning_point", "pair_id",
        "pair_timestamp_sec", "value", "unit", "support_pair_ids", "source_observation_policy",
        "interpretation",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in rows:
            output = dict(event)
            output["support_pair_ids"] = ";".join(str(pair_id) for pair_id in event["support_pair_ids"])
            writer.writerow(output)


def _write_t4_source_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def build_lower_limb_pipeline(
    stereo_results_path: str | Path,
    output_dir: str | Path,
    *,
    coordinate_transform_path: str | Path | None = None,
    allow_test_coordinate_transform: bool = False,
) -> dict[str, Any]:
    """Build all T1--T4 outputs below one new directory without changing stereo output."""

    source_path = Path(stereo_results_path).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing lower-limb output: {destination}")

    trajectory = normalize_stereo_records(_load_jsonl(source_path))
    kinematics = derive_kinematics(trajectory)
    gait = derive_gait_candidates(kinematics)

    destination.mkdir(parents=True)
    t1_dir = destination / "t1_trajectory"
    t2_dir = destination / "t2_kinematics"
    t3_dir = destination / "t3_coordinates"
    t4_dir = destination / "t4_noncontact_candidates"
    for directory in (t1_dir, t2_dir, t3_dir, t4_dir):
        directory.mkdir()

    _write_jsonl(t1_dir / "lower_limb_trajectory.jsonl", trajectory)
    _write_trajectory_csv(t1_dir / "lower_limb_trajectory.csv", trajectory)
    t1_summary = summarize_trajectory(trajectory)
    _write_json(t1_dir / "trajectory_summary.json", t1_summary)

    _write_jsonl(t2_dir / "lower_limb_kinematics.jsonl", kinematics)
    _write_kinematics_csv(t2_dir / "lower_limb_kinematics.csv", kinematics)
    t2_summary = summarize_kinematics(kinematics)
    _write_json(t2_dir / "kinematics_summary.json", t2_summary)

    t3_summary: dict[str, Any]
    if coordinate_transform_path is None:
        t3_summary = {
            "status": "not_configured",
            "reason": "no_coordinate_transform_supplied",
            "interpretation": (
                "Source left-camera trajectory is retained in T1. No target-frame coordinates, "
                "ground parameters, or physical-coordinate claim are produced."
            ),
        }
    else:
        transform_path = Path(coordinate_transform_path).resolve()
        transform = RigidTransform.from_mapping(
            json.loads(transform_path.read_text(encoding="utf-8")),
            allow_test_transform=allow_test_coordinate_transform,
        )
        transformed = transform_trajectory_records(trajectory, transform)
        _write_jsonl(t3_dir / "lower_limb_trajectory_target_frame.jsonl", transformed)
        _write_trajectory_csv(t3_dir / "lower_limb_trajectory_target_frame.csv", transformed, target=True)
        t3_summary = summarize_transformed_trajectory(transformed, transform)
        t3_summary["status"] = "test_only" if transform.status == "test_only" else "measured_locked"
        t3_summary["transform_file"] = str(transform_path)
    _write_json(t3_dir / "coordinate_normalization_summary.json", t3_summary)

    _write_t4_source_csv(t4_dir / "t4_source_signals.csv", gait["records"])
    _write_jsonl(t4_dir / "gait_event_candidates.jsonl", gait["events"])
    _write_event_csv(t4_dir / "gait_event_candidates.csv", gait["events"])
    _write_json(t4_dir / "gait_parameter_candidates.json", gait["summary"]["gait_parameter_candidates"])
    _write_json(t4_dir / "gait_candidate_summary.json", gait["summary"])

    pipeline_summary = {
        "status": "complete",
        "input_stereo_results": str(source_path),
        "frames": len(trajectory),
        "stages": {
            "t1_trajectory": {"status": "complete", "summary": t1_summary},
            "t2_kinematics": {"status": "complete", "summary": t2_summary},
            "t3_coordinates": t3_summary,
            "t4_noncontact_candidates": {"status": "complete", "summary": gait["summary"]},
        },
        "interpretation": (
            "End-to-end post-processing integration only. T3 and T4 status fields remain "
            "authoritative; no gait parameter or accuracy claim is added by this wrapper."
        ),
    }
    _write_json(destination / "pipeline_summary.json", pipeline_summary)
    _write_json(
        destination / "pipeline_metadata.json",
        {
            "input_stereo_results": str(source_path),
            "coordinate_transform": (
                str(Path(coordinate_transform_path).resolve())
                if coordinate_transform_path is not None
                else None
            ),
            "allow_test_coordinate_transform": allow_test_coordinate_transform,
            "source_observation_policy": "direct_accepted_stereo_only",
            "write_policy": "new output directory only; source stereo JSONL is read-only",
            "interpretation_boundary": (
                "T5 integrates existing T1-T4 contracts only. Coordinate and gait status "
                "remain controlled by T3/T4 prerequisites."
            ),
        },
    )
    return pipeline_summary
