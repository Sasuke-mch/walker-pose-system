#!/usr/bin/env python3
"""Evaluate predicted planes against independently direct target planes.

For every requested horizon, the target direct plane is *only* read after the
source-plane propagation.  It is never used to compose motion or to correct a
prediction.  Hence this reports internal agreement, not ground-truth accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pose_app.local_plane_propagation import (  # noqa: E402
    LocalPlane, RelativePose, compose_relative_poses, parse_direct_plane, parse_relative_pose, plane_agreement,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-planes", type=Path, required=True)
    parser.add_argument("--relative-poses", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 6, 9])
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    if any(value < 1 for value in args.horizons):
        raise ValueError("horizons must be positive")
    direct_by_index: dict[int, LocalPlane] = {}
    for record in read_jsonl(args.direct_planes):
        plane = parse_direct_plane(record)
        if plane is not None:
            index = int(record["frame_index"])
            if index in direct_by_index:
                raise ValueError(f"duplicate direct plane frame {index}")
            direct_by_index[index] = plane
    poses: dict[int, RelativePose] = {}
    for raw in read_jsonl(args.relative_poses):
        pose = parse_relative_pose(raw)
        if pose is not None:
            poses[pose.from_frame_index] = pose
    rows: list[dict[str, Any]] = []
    for horizon in sorted(set(args.horizons)):
        for source_index, source_plane in sorted(direct_by_index.items()):
            target_index = source_index + horizon
            target_plane = direct_by_index.get(target_index)
            if target_plane is None:
                continue
            edges = [poses.get(index) for index in range(source_index, target_index)]
            if any(edge is None for edge in edges):
                rows.append({"source_frame_index": source_index, "target_frame_index": target_index,
                             "horizon_frames": horizon, "status": "unavailable", "failure_reason": "missing_or_rejected_static_background_relative_pose",
                             "normal_angle_deg": None, "camera_plane_distance_delta": None})
                continue
            rotation, translation = compose_relative_poses(edge for edge in edges if edge is not None)
            prediction = LocalPlane(rotation @ source_plane.normal, source_plane.offset - float((rotation @ source_plane.normal) @ translation))
            rows.append({"source_frame_index": source_index, "target_frame_index": target_index,
                         "horizon_frames": horizon, "status": "compared", "failure_reason": None,
                         **plane_agreement(prediction, target_plane)})
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "propagation_agreement_per_pair.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["source_frame_index", "target_frame_index", "horizon_frames", "status", "failure_reason", "normal_angle_deg", "camera_plane_distance_delta"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    summary_rows = []
    for horizon in sorted(set(args.horizons)):
        compared = [row for row in rows if row["horizon_frames"] == horizon and row["status"] == "compared"]
        angles = [float(row["normal_angle_deg"]) for row in compared]
        distances = [float(row["camera_plane_distance_delta"]) for row in compared]
        summary_rows.append({"horizon_frames": horizon, "eligible_direct_pairs": sum(row["horizon_frames"] == horizon for row in rows),
                             "compared": len(compared), "median_normal_angle_deg": percentile(angles, 50), "p95_normal_angle_deg": percentile(angles, 95),
                             "median_camera_plane_distance_delta": percentile(distances, 50), "p95_camera_plane_distance_delta": percentile(distances, 95)})
    with (args.output_dir / "propagation_agreement_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader(); writer.writerows(summary_rows)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "direct_target_rule": "Target direct plane is evaluation-only and is excluded from the prediction path.",
        "feature_domain": "static_background_excluding_person_walker_ground",
        "interpretation": "Agreement is a propagation-versus-independent-direct consistency diagnostic, not true ground-plane accuracy.",
        "horizons_frames": sorted(set(args.horizons)),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "summary": summary_rows}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
