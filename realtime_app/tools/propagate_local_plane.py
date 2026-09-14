#!/usr/bin/env python3
"""Propagate local left-camera ground planes through accepted static-background VO.

This never estimates a plane from RGB data.  It only consumes a direct-plane
JSONL produced by the independent ground-observation route and a relative-pose
JSONL whose records explicitly certify static-background-only 3-D support.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pose_app.local_plane_propagation import parse_relative_pose, sequential_propagation  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-planes", type=Path, required=True)
    parser.add_argument("--relative-poses", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-propagation-frames", type=int, default=9)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    direct = read_jsonl(args.direct_planes)
    raw_poses = read_jsonl(args.relative_poses)
    poses = {}
    rejected = 0
    for raw in raw_poses:
        pose = parse_relative_pose(raw)
        if pose is None:
            rejected += 1
            continue
        if pose.from_frame_index in poses:
            raise ValueError(f"duplicate accepted relative pose from frame {pose.from_frame_index}")
        poses[pose.from_frame_index] = pose
    rows = sequential_propagation(direct, poses, args.max_propagation_frames)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "local_ground_state.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    summary = {
        "coordinate_frame": "per-frame left-camera frame; no world/fixed floor frame",
        "direct_plane_source": str(args.direct_planes.resolve()),
        "relative_pose_source": str(args.relative_poses.resolve()),
        "relative_pose_required_feature_domain": "static_background_excluding_person_walker_ground",
        "relative_pose_rejected_by_contract": rejected,
        "max_propagation_frames": args.max_propagation_frames,
        "local_ground_state_version": "local_ground_state_v1",
        "state_counts": {state: sum(row["observation_state"] == state for row in rows) for state in ("direct", "propagated", "unavailable")},
        "interpretation": "Propagated planes are geometric predictions. Their agreement with an independently direct target plane is internal consistency, not physical ground accuracy or contact evidence.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
