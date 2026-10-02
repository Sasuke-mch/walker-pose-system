#!/usr/bin/env python3
"""Audit every full-body frame against the constructed-grasp geometry gate.

No smoothing, substitution or acceptance based only on wrist locations.
Triangle screen reuses the existing algorithm and its explicit limitations.
"""
from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
_tool_prepare_imports()

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "realtime_app"))
from pose_app.smpl_coco_observation import _install_legacy_smpl_pickle_compatibility
from walker_tools.hands.audit_fixed_mano_grip import capsule
from pose_app.constructed_grasp_refinement import screened_triangle_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--walker-model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError("refuse existing audit")
    _install_legacy_smpl_pickle_compatibility()
    with (ROOT / "third_party/WiLoR/mano_data/models/SMPLH_male.pkl").open("rb") as f:
        asset = pickle.load(f, encoding="latin1")
    z = np.load(args.result, allow_pickle=False)
    walker = json.loads(args.walker_model.read_text(encoding="utf-8"))
    R = np.asarray(walker["rotation_left_camera_from_walker"])
    t = np.asarray(walker["translation_left_camera_from_walker_mm"]) / 1000
    v = (z["vertices"] - t) @ R
    dom = np.asarray(asset["weights"]).argmax(1)
    rest, restj = np.asarray(asset["v_template"]), np.asarray(asset["J"])
    records = []
    for side, start, wrist in (("left", 22, 20), ("right", 37, 21)):
        use = ((dom >= start) & (dom < start + 15)) | ((dom == wrist) & (np.linalg.norm(rest - restj[wrist], axis=1) < .08))
        idx = np.flatnonzero(use)
        local = np.full(6890, -1, int)
        local[idx] = np.arange(len(idx))
        faces = local[z["faces"][use[z["faces"]].all(1)]]
        groups = [np.flatnonzero((dom[idx] >= start + 3 * i + 1) & (dom[idx] < start + 3 * i + 3)) for i in range(5)]
        center = .5 * (restj[wrist] + restj[[start, start + 3, start + 6, start + 9]].mean(0))
        palm = np.flatnonzero((dom[idx] == wrist) & (np.linalg.norm(rest[idx] - center, axis=1) < .03))
        a, b = [np.asarray(walker["nodes_walker_mm"][key]) / 1000 for key in walker["handle_segments"][side]]
        axis = (b - a) / np.linalg.norm(b - a)
        for frame, pair in enumerate(z["pair_id"]):
            points = v[frame, idx]
            gaps = capsule(points, a, b, .016)
            finger_gap = [float(np.abs(gaps[g]).min() * 1000) for g in groups]
            palm_gap = float(np.abs(gaps[palm]).min() * 1000)
            chosen = np.asarray([points[g[np.abs(gaps[g]).argmin()]] for g in groups])
            radial = chosen - a - ((chosen - a) @ axis)[:, None] * axis
            radial /= np.linalg.norm(radial, axis=1, keepdims=True).clip(1e-8)
            other = radial[:4].mean(0)
            dot = float(radial[4] @ other / max(np.linalg.norm(other), 1e-8))
            crossings = screened_triangle_pairs(points, faces)
            penetration = float(max(0., -gaps.min()) * 1000)
            reasons = []
            if crossings:
                reasons.append("screened_triangle_crossings")
            if penetration > 3:
                reasons.append("handle_penetration_exceeds_3mm")
            if max(finger_gap) > 5:
                reasons.append("finger_region_gap_exceeds_5mm")
            if palm_gap > 5:
                reasons.append("palm_gap_exceeds_5mm")
            if dot > .2:
                reasons.append("thumb_opposition_proxy_failed")
            records.append({"pair_id": int(pair), "hand": side,
                "passed_geometry_proxy": not reasons, "reject_reasons": reasons,
                "finger_region_gap_mm": finger_gap, "palm_gap_mm": palm_gap,
                "thumb_dot": dot, "max_vertex_penetration_mm": penetration,
                "triangle_crossings": crossings})
        print(side, "audited", len(z["pair_id"]), "frames", flush=True)
    summary = {s: {"passed": sum(r["passed_geometry_proxy"] for r in records if r["hand"] == s),
        "total": len(z["pair_id"]), "max_penetration_mm": max(r["max_vertex_penetration_mm"] for r in records if r["hand"] == s)} for s in ("left", "right")}
    args.output.write_text(json.dumps({"status": "engineering_surface_proxy_audit", "summary": summary,
        "source": str(args.result.resolve()), "records": records,
        "limits": "16mm assumed handle; discrete surface regions; noncoplanar triangle screen excludes tangency/coplanar overlap; not force closure or measured contact"}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
