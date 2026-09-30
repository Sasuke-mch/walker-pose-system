"""Solve an all-frame fixed hand-to-walker prior from an SMPL-H result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pose_app.global_hand_handle_pose import (
    estimate_global_hand_handle_pose,
    load_static_handle_ends,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--walker-model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--handle-ends-ground", type=Path, default=None,
                    help="optional npz/npy/json with (N,2,2,3) metre endpoints")
    args = ap.parse_args()
    z = np.load(args.result, allow_pickle=False)
    n = int(z["hand_points_left"].shape[0])
    if "left_palm_ground_m" not in z or "right_palm_ground_m" not in z:
        raise ValueError("result must contain left_palm_ground_m/right_palm_ground_m")
    if args.handle_ends_ground:
        q = args.handle_ends_ground
        if q.suffix == ".npz":
            d = np.load(q, allow_pickle=False)
            ends = d["handle_ends_ground_m"]
        elif q.suffix == ".npy":
            ends = np.load(q, allow_pickle=False)
        else:
            ends = np.asarray(json.loads(q.read_text(encoding="utf-8")), float)
        source = str(q)
    else:
        ends = load_static_handle_ends(args.walker_model, n)
        source = "static_walker_model_nodes_initial_ground_mm_repeated"
    if ends.shape != (n, 2, 2, 3):
        raise ValueError(f"handle endpoints must be (N,2,2,3), got {ends.shape}")
    out = {
        "schema_version": "global_hand_handle_pose_v1",
        "status": "engineering_candidate",
        "assumption": "hand_static_relative_to_walker_for_entire_video",
        "source_result": str(args.result.resolve()),
        "handle_source": source,
        "handle_trajectory_external_truth": bool(args.handle_ends_ground),
        "hands": {},
    }
    for side, j in (("left", 0), ("right", 1)):
        out["hands"][side] = estimate_global_hand_handle_pose(
            z[f"{side}_palm_ground_m"], ends[:, j])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
