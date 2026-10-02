"""Solve an all-frame fixed hand-to-walker prior from an SMPL-H result."""
from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
_tool_prepare_imports()

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
    ap.add_argument("--camera-rigid", action="store_true", help="use fixed camera<-walker installation; bypass world coordinates")
    ap.add_argument("--vertex-sets", type=Path)
    ap.add_argument("--handle-ends-ground", type=Path, default=None,
                    help="optional npz/npy/json with (N,2,2,3) metre endpoints")
    ap.add_argument("--walker-poses", type=Path, default=None,
                    help="npz with rotation_ground_from_walker and translation_ground_from_walker_m")
    ap.add_argument("--allow-static-diagnostic", action="store_true",
                    help="explicitly allow static fallback for diagnosing a missing walker trajectory")
    args = ap.parse_args()
    z = np.load(args.result, allow_pickle=False)
    n = int(z["hand_points_left"].shape[0])
    if not args.camera_rigid and ("left_palm_ground_m" not in z or "right_palm_ground_m" not in z):
        raise ValueError("result must contain left_palm_ground_m/right_palm_ground_m")
    if args.camera_rigid:
        if args.walker_poses or args.handle_ends_ground or args.vertex_sets is None:
            raise ValueError("camera-rigid requires vertex-sets and no world trajectory inputs")
        model = json.loads(args.walker_model.read_text(encoding="utf-8"))
        R = np.asarray(model["rotation_left_camera_from_walker"], float)
        t = np.asarray(model["translation_left_camera_from_walker_mm"], float) / 1000
        R = np.repeat(R[None], n, axis=0); t = np.repeat(t[None], n, axis=0)
        validate_walker_pose(R, t, n)
        sets = json.loads(args.vertex_sets.read_text(encoding="utf-8"))["sets"]
        palm = {}
        palm_indices = {}
        for s in ("left", "right"):
            idx = np.asarray(sets[f"{s}_palm_surface_candidate"]["palm_fingers"], int)
            palm_indices[s] = idx.tolist()
            palm[s] = transform_ground_to_walker(z["vertices"][:, idx], R, t)
        ends = load_static_handle_ends(args.walker_model, n)
        source = "fixed_camera_from_walker_installation_model"
        relative_from_pose = True
    elif args.walker_poses:
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
        ends = load_static_handle_ends(args.walker_model, n, "nodes_initial_ground_mm")
        source = "static_walker_model_nodes_initial_ground_mm_repeated"
        palm = {s: z[f"{s}_palm_ground_m"] for s in ("left", "right")}
        relative_from_pose = False
    if ends.shape != (n, 2, 2, 3):
        raise ValueError(f"handle endpoints must be (N,2,2,3), got {ends.shape}")
    out = {
        "schema_version": "global_hand_handle_pose_v1",
        "status": "engineering_candidate" if (args.camera_rigid or args.handle_ends_ground or args.walker_poses) else "invalid_static_world_fallback",
        "assumption": "hand_static_relative_to_walker_for_entire_video",
        "source_result": str(args.result.resolve()),
        "handle_source": source,
        "handle_trajectory_external_truth": False,
        "geometry_source": "camera_rigid_mount_assumption" if args.camera_rigid else "supplied_engineering_trajectory",
        "walker_pose_source": source if args.walker_poses else None,
        "relative_frame": "walker_rigid_frame_per_frame",
        "hands": {},
    }
    for side, j in (("left", 0), ("right", 1)):
        out["hands"][side] = estimate_global_hand_handle_pose(
            palm[side], ends[:, j])
        pts = palm[side]; a = ends[:, j, 0]; b = ends[:, j, 1]
        axis = b-a
        u = np.sum((pts-a[:,None])*axis[:,None],axis=-1)/np.sum(axis*axis,axis=-1)[:,None]
        nearest = a[:,None]+np.clip(u,0,1)[...,None]*axis[:,None]
        distance = np.linalg.norm(pts-nearest,axis=-1)-0.016
        centers = pts.mean(axis=1)
        delta = np.linalg.norm(centers-np.median(centers,axis=0),axis=-1)
        out["hands"][side]["diagnostics_all_frames"] = {
            "centroid_variation_median_mm": float(np.median(delta)*1000),
            "centroid_variation_p95_mm": float(np.percentile(delta,95)*1000),
            "nearest_surface_gap_median_mm": float(np.median(distance.min(axis=1))*1000),
            "nearest_surface_gap_p95_mm": float(np.percentile(distance.min(axis=1),95)*1000),
            "penetrating_vertex_fraction": float(np.mean(distance<0)),
        }
        if args.camera_rigid:
            # Fixed correspondences preserve the entire hand surface, rather
            # than constraining just a single nearest point or PCA centroid.
            shared = np.median(pts, axis=0)
            out["hands"][side]["vertex_indices"] = palm_indices[side]
            out["hands"][side]["shared_surface_walker_m"] = shared.tolist()
            av, bv = a[0], b[0]; ab = bv-av
            sv = np.clip(((shared-av)@ab)/(ab@ab),0,1)
            gap = np.linalg.norm(shared-(av+sv[:,None]*ab),axis=-1)-.016
            out["hands"][side]["shared_surface_diagnostics"] = {
                "nearest_gap_mm": float(gap.min()*1000),
                "penetrating_fraction": float(np.mean(gap<0)),
                "penetration_p95_mm": float(np.percentile(np.maximum(-gap,0),95)*1000),
                "max_penetration_mm": float(np.maximum(-gap,0).max()*1000),
                "half_sequence_center_difference_mm": float(np.linalg.norm(np.median(centers[:n//2],axis=0)-np.median(centers[n//2:],axis=0))*1000),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
