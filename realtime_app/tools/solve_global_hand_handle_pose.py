"""Solve an all-frame fixed hand-to-walker prior from an SMPL-H result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pose_app.global_hand_handle_pose import (
    estimate_global_hand_handle_pose,
    load_static_handle_ends,
    validate_walker_pose,
    transform_ground_to_walker,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--walker-model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--handle-ends-ground", type=Path, default=None,
                    help="optional npz/npy/json with (N,2,2,3) metre endpoints")
    ap.add_argument("--walker-poses", type=Path, default=None,
                    help="npz with rotation_ground_from_walker and translation_ground_from_walker_m")
    ap.add_argument("--allow-static-diagnostic", action="store_true",
                    help="explicitly allow static fallback for diagnosing a missing walker trajectory")
    args = ap.parse_args()
    z = np.load(args.result, allow_pickle=False)
    n = int(z["hand_points_left"].shape[0])
    if "left_palm_ground_m" not in z or "right_palm_ground_m" not in z:
        raise ValueError("result must contain left_palm_ground_m/right_palm_ground_m")
    if args.walker_poses:
        if args.handle_ends_ground:
            raise ValueError("provide --walker-poses or --handle-ends-ground, not both")
        pose = np.load(args.walker_poses, allow_pickle=False)
        R = pose["rotation_ground_from_walker"]
        t = pose["translation_ground_from_walker_m"]
        validate_walker_pose(R, t, n)
        static = load_static_handle_ends(args.walker_model, 1)[0]
        palm = {s: transform_ground_to_walker(z[f"{s}_palm_ground_m"], R, t) for s in ("left", "right")}
        ends = np.repeat(static[None], n, axis=0)
        source = str(args.walker_poses)
        relative_from_pose = True
    elif args.handle_ends_ground:
        q = args.handle_ends_ground
        if q.suffix == ".npz":
            d = np.load(q, allow_pickle=False)
            ends = d["handle_ends_ground_m"]
        elif q.suffix == ".npy":
            ends = np.load(q, allow_pickle=False)
        else:
            ends = np.asarray(json.loads(q.read_text(encoding="utf-8")), float)
        source = str(q)
        palm = {s: z[f"{s}_palm_ground_m"] for s in ("left", "right")}
        relative_from_pose = False
    else:
        if not args.allow_static_diagnostic:
            raise ValueError("hand pose must be solved relative to a per-frame walker trajectory; provide --handle-ends-ground")
        ends = load_static_handle_ends(args.walker_model, n)
        source = "static_walker_model_nodes_initial_ground_mm_repeated"
        palm = {s: z[f"{s}_palm_ground_m"] for s in ("left", "right")}
        relative_from_pose = False
    if ends.shape != (n, 2, 2, 3):
        raise ValueError(f"handle endpoints must be (N,2,2,3), got {ends.shape}")
    out = {
        "schema_version": "global_hand_handle_pose_v1",
        "status": "engineering_candidate" if (args.handle_ends_ground or args.walker_poses) else "invalid_static_world_fallback",
        "assumption": "hand_static_relative_to_walker_for_entire_video",
        "source_result": str(args.result.resolve()),
        "handle_source": source,
        "handle_trajectory_external_truth": bool(args.handle_ends_ground or args.walker_poses),
        "walker_pose_source": source if args.walker_poses else None,
        "relative_frame": "walker_rigid_frame_per_frame",
        "hands": {},
    }
    for side, j in (("left", 0), ("right", 1)):
        out["hands"][side] = estimate_global_hand_handle_pose(
            palm[side], ends[:, j])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
