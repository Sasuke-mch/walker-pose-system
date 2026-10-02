#!/usr/bin/env python3
"""Build SMPL 6890 surface-vertex candidate sets for sole/palm contact.

Reads only the official male SMPL pkl (v_template, shapedirs, weights, f,
kintree_table). Joint semantics are verified against the kinematic tree and
rest-pose joint positions; no vertex index is copied from old experiments.
All outputs are candidate sets pending audit, never validated contact maps.
"""
from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
from walker_tools._compat import APP_ROOT as _tool_app_root
_tool_prepare_imports()

import argparse
import json
import sys
from pathlib import Path

APP_ROOT = _tool_app_root

import numpy as np

from pose_app.smpl_coco_observation import _install_legacy_smpl_pickle_compatibility

EXPECTED_SEMANTICS = {
    "left_ankle": 7, "right_ankle": 8, "left_foot": 10, "right_foot": 11,
    # SMPL-H appends the complete left hand chain (22..36) before the
    # complete right hand chain (37..51).  Index 23 is the first left-finger
    # joint, not the right-hand root.
    "left_wrist": 20, "right_wrist": 21, "left_hand": 22, "right_hand": 37,
}
PARENT = {
    7: 4, 8: 5, 10: 7, 11: 8, 20: 18, 21: 19, 22: 20, 37: 21,
}


def load_smpl_arrays(model_path: Path) -> dict:
    _install_legacy_smpl_pickle_compatibility()
    import pickle

    with open(model_path, "rb") as fh:
        data = pickle.load(fh, encoding="latin1")
    out = {
        "v_template": np.asarray(data["v_template"], dtype=np.float64),
        "shapedirs": np.asarray(data["shapedirs"], dtype=np.float64),
        "weights": np.asarray(data["weights"], dtype=np.float64),
        "faces": np.asarray(data["f"], dtype=np.int64),
        "kintree_table": np.asarray(data["kintree_table"], dtype=np.int64),
        "joints": np.asarray(data["J"], dtype=np.float64),
    }
    v, w, f = out["v_template"], out["weights"], out["faces"]
    if v.shape != (6890, 3):
        raise ValueError(f"v_template shape {v.shape} != (6890,3)")
    if f.shape[1] != 3 or f.min() < 0 or f.max() > 6889:
        raise ValueError("faces are not valid SMPL 6890 triangles")
    # SMPL-H keeps the 24 body joints and appends 28 MANO joints.  The
    # contact candidates below only use the body/wrist/hand-root columns, but
    # the source model must therefore be accepted with its full 52-column
    # skinning matrix.
    if w.shape[0] != 6890 or w.shape[1] < 24:
        raise ValueError(f"weights shape {w.shape} is not a SMPL-H-compatible matrix")
    if not (np.isfinite(v).all() and np.isfinite(w).all()):
        raise ValueError("non-finite vertices or weights")
    if np.any(w < -1e-9):
        raise ValueError("negative skinning weights")
    if not np.allclose(w.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("skinning weights do not sum to one")
    return out


def verify_semantics(kintree_table: np.ndarray, joints: np.ndarray) -> dict:
    """Check the 8 contact joints against parents and rest-pose positions."""
    kt = np.asarray(kintree_table)
    if kt.ndim != 2 or kt.shape[0] != 2 or kt.shape[1] < 24:
        raise ValueError(f"kintree_table shape {kt.shape} is not SMPL-H-compatible")
    parent_of = {int(child): int(parent) for parent, child in zip(kt[0].tolist(), kt[1].tolist())}
    checks = {}
    for name, index in EXPECTED_SEMANTICS.items():
        ok_parent = parent_of.get(index) == PARENT[index]
        pos = joints[index]
        if "ankle" in name or "foot" in name:
            ok_pos = pos[1] < -0.9
        else:
            ok_pos = abs(pos[1] - 0.2) < 0.15
        side_ok = (pos[0] > 0) if name.startswith("left") else (pos[0] < 0)
        checks[name] = {"index": index, "parent_ok": bool(ok_parent),
                        "position_ok": bool(ok_pos), "side_ok": bool(side_ok)}
    if not all(c["parent_ok"] and c["position_ok"] and c["side_ok"] for c in checks.values()):
        raise ValueError(f"joint semantics failed kinematic-tree check: {checks}")
    return checks


def adjacency_list(faces: np.ndarray) -> list[set]:
    adj: list[set] = [set() for _ in range(6890)]
    for tri in faces:
        a, b, c = (int(x) for x in tri)
        adj[a] |= {b, c}
        adj[b] |= {a, c}
        adj[c] |= {a, b}
    return adj


def check_compact(indices: np.ndarray, adj: list[set], name: str) -> dict:
    chosen = set(int(i) for i in indices)
    lonely = sum(1 for i in chosen if not (adj[i] & chosen))
    return {"count": len(chosen), "lonely_vertices": lonely,
            "passed": bool(len(chosen) > 0 and lonely == 0)}


def build_foot_sets(v: np.ndarray, w: np.ndarray, joints: np.ndarray,
                    adj: list[set], side: str) -> tuple[dict, dict]:
    ankle, foot = (7, 10) if side == "left" else (8, 11)
    influence = w[:, ankle] + w[:, foot]
    cand = np.nonzero(influence > 0.25)[0]
    ankle_y = joints[ankle, 1]
    cand = cand[v[cand, 1] < ankle_y + 0.05]
    if len(cand) < 20:
        raise ValueError(f"{side} foot candidate set too small: {len(cand)}")
    if np.any(v[cand, 1] > joints[4 if side == "left" else 5, 1]):
        raise ValueError(f"{side} foot set reaches above the knee")
    sole_level = float(np.percentile(v[cand, 1], 20))
    sole = cand[v[cand, 1] <= sole_level]
    z = v[sole, 2]
    lo, hi = float(np.percentile(z, 33)), float(np.percentile(z, 67))
    subsets = {
        "heel": sorted(int(i) for i in sole[z <= lo]),
        "ball": sorted(int(i) for i in sole[(z > lo) & (z <= hi)]),
        "toe": sorted(int(i) for i in sole[z > hi]),
    }
    if any(len(s) == 0 for s in subsets.values()):
        raise ValueError(f"{side} sole subset is empty")
    audit = {"influence_threshold": 0.25, "sole_band_percentile": 20,
             "sole_level_y": sole_level, "z_terciles": [lo, hi],
             "joint_source": [ankle, foot],
             "adjacency": {k: check_compact(np.asarray(vv), adj, k) for k, vv in subsets.items()}}
    if not all(a["passed"] for a in audit["adjacency"].values()):
        raise ValueError(f"{side} sole adjacency check failed")
    return subsets, audit


def build_hand_sets(v: np.ndarray, w: np.ndarray, joints: np.ndarray,
                    adj: list[set], side: str) -> tuple[dict, dict]:
    # The SMPL-H right hand root is joint 37.  Using 23 here silently selects
    # the first joint of the left index-finger chain and produces a tiny,
    # anatomically mixed right-hand surface set.
    wrist, hand = (20, 22) if side == "left" else (21, 37)
    dominant = np.argmax(w, axis=1)
    cand = np.nonzero((dominant == wrist) | (dominant == hand))[0]
    center = joints[hand]
    cand = cand[np.linalg.norm(v[cand] - center, axis=1) < 0.30]
    if len(cand) < 50:
        raise ValueError(f"{side} hand candidate set too small: {len(cand)}")
    elbow = joints[18 if side == "left" else 19]
    if np.any(np.linalg.norm(v[cand] - elbow, axis=1) < 0.05):
        raise ValueError(f"{side} hand set reaches the elbow")
    palm = sorted(int(i) for i in cand)
    audit = {"dominant_joints": [wrist, hand], "spatial_radius_m": 0.30,
             "joint_source": [wrist, hand],
             "adjacency": {"palm_fingers": check_compact(np.asarray(palm), adj, "palm")}}
    if not audit["adjacency"]["palm_fingers"]["passed"]:
        raise ValueError(f"{side} hand adjacency check failed")
    return {"palm_fingers": palm}, audit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refuse to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    arrays = load_smpl_arrays(args.model)
    v, w, faces = arrays["v_template"], arrays["weights"], arrays["faces"]
    semantics = verify_semantics(arrays["kintree_table"], arrays["joints"])
    adj = adjacency_list(faces)

    sets, audits = {}, {}
    for side in ("left", "right"):
        sole, sole_audit = build_foot_sets(v, w, arrays["joints"], adj, side)
        sets[f"{side}_sole_surface_candidate"] = sole
        audits[f"{side}_sole_surface_candidate"] = sole_audit
        palm, palm_audit = build_hand_sets(v, w, arrays["joints"], adj, side)
        sets[f"{side}_palm_surface_candidate"] = palm
        audits[f"{side}_palm_surface_candidate"] = palm_audit

    (out / "contact_vertex_sets.json").write_text(json.dumps({
        "schema_version": "smpl_contact_vertex_sets_v1",
        "model_topology": "male_SMPL_6890",
        "semantics": EXPECTED_SEMANTICS,
        "sets": sets,
    }, indent=2) + "\n", encoding="utf-8")
    counts = {k: ({s: len(vv) for s, vv in v.items()} if isinstance(v, dict) else len(v))
              for k, v in sets.items()}
    (out / "set_audit.json").write_text(json.dumps({
        "candidate_only": True,
        "model_file": str(args.model.resolve()),
        "semantic_checks": semantics,
        "counts": counts,
        "per_set": audits,
        "index_range": "[0,6889]",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": counts, "status": "candidate_only"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
