#!/usr/bin/env python3
"""Read-only geometry/coordinate audit of the v5 surface contact setup.

Recomputes ground-frame sole heights, handle capsule geometry, surface set
index semantics and the Stage C->D change from already-saved v5 results.
Never reruns fitting and never modifies any v1-v5 result, label, scene,
surface-set or topology input.

The high sole penetration fraction here is an engineering-consistency finding
between the current ground transform and the SMPL surface; it must not be
reported as a physical touch-down failure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research_records/engineering_validation/G20260924_smpl_vposer_shared_beta_v1"
HANDLE_RADIUS_M = 0.016
SOLE_PARTS = ("heel", "ball", "toe")


def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64).ravel()
    return {
        "min": float(np.min(x)), "p05": float(np.percentile(x, 5)),
        "median": float(np.median(x)), "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)), "finite": bool(np.isfinite(x).all()),
    }


def load_ground(v, scene, sl):
    R = np.asarray(scene["rotation_ground_from_left"], dtype=np.float64)[sl]
    T = np.asarray(scene["translation_ground_from_left_mm"], dtype=np.float64)[sl] / 1000.0
    return np.einsum("nij,nvj->nvi", R, v) + T[:, None, :]


def sole_indices(sets, side):
    return np.concatenate([np.asarray(sets["sets"][f"{side}_sole_surface_candidate"][p], dtype=np.int64)
                           for p in SOLE_PARTS])


def palm_indices(sets, side):
    return np.asarray(sets["sets"][f"{side}_palm_surface_candidate"]["palm_fingers"], dtype=np.int64)


def capsule_endpoints(topology, scene, sl):
    nodes = {k: np.asarray(v, dtype=np.float64) / 1000.0 for k, v in topology["nodes_left_camera_mm"].items()}
    R = np.asarray(scene["rotation_ground_from_left"], dtype=np.float64)[sl]
    T = np.asarray(scene["translation_ground_from_left_mm"], dtype=np.float64)[sl] / 1000.0
    out = {}
    for side, (na, nb) in {"left": ("front_right_top", "rear_right_top"),
                           "right": ("front_left_top", "rear_left_top")}.items():
        a = np.einsum("nij,j->ni", R, nodes[na]) + T
        b = np.einsum("nij,j->ni", R, nodes[nb]) + T
        out[side] = (a, b)
    return out


def capsule_centerline_dist(pts, a, b):
    ab = b - a
    u = (((pts - a[:, None, :]) * ab[:, None, :]).sum(-1, keepdims=True)
         / np.maximum((ab[:, None, :] ** 2).sum(-1, keepdims=True), 1e-8)).clip(0.0, 1.0)
    return np.linalg.norm(pts - (a[:, None, :] + u * ab[:, None, :]), axis=-1)


def foot_block(vg, sole_l, sole_r) -> dict:
    out = {}
    for side, idx in (("left", sole_l), ("right", sole_r)):
        z = vg[:, idx, 2]
        out[side] = {
            "stats_m": _stats(z),
            "stats_mm": _stats(z * 1000.0),
            "fraction_z_negative": float((z < 0).mean()),
            "fraction_z_below_10mm": float((z < -0.010).mean()),
            "fraction_z_below_25mm": float((z < -0.025).mean()),
            "per_frame_softmin_m": [float(-0.005 * (np.log(np.exp(-z[t] / 0.005).sum())
                                        - np.log(float(idx.size)))) for t in range(z.shape[0])],
            "per_frame_negative_fraction": [float((z[t] < 0).mean()) for t in range(z.shape[0])],
        }
    return out


def hand_block(vg, caps, palm_l, palm_r) -> dict:
    out = {}
    for side, idx in (("left", palm_l), ("right", palm_r)):
        a, b = caps[side]
        d = capsule_centerline_dist(vg[:, idx, :], a, b)
        resid = d - HANDLE_RADIUS_M
        length = np.linalg.norm(b - a, axis=1)
        out[side] = {
            "capsule_length_m": _stats(length),
            "capsule_endpoints_finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
            "endpoint_height_m": {"a": _stats(a[:, 2]), "b": _stats(b[:, 2])},
            "distance_m": _stats(d),
            "residual_m": _stats(resid),
            "residual_mm": _stats(resid * 1000.0),
            "fraction_residual_negative": float((resid < 0).mean()),
        }
    return out


def set_semantics(vg, sets) -> dict:
    l_sole, r_sole = sole_indices(sets, "left"), sole_indices(sets, "right")
    l_palm, r_palm = palm_indices(sets, "left"), palm_indices(sets, "right")
    all_idx = np.concatenate([l_sole, r_sole, l_palm, r_palm])
    return {
        "left_sole_count": int(l_sole.size), "right_sole_count": int(r_sole.size),
        "left_palm_count": int(l_palm.size), "right_palm_count": int(r_palm.size),
        "index_min": int(all_idx.min()), "index_max": int(all_idx.max()),
        "index_range_ok": bool(all_idx.min() >= 0 and all_idx.max() <= 6889),
        "sole_left_right_identical": bool(l_sole.size == r_sole.size and np.array_equal(np.sort(l_sole), np.sort(r_sole))),
        "palm_left_right_identical": bool(l_palm.size == r_palm.size and np.array_equal(np.sort(l_palm), np.sort(r_palm))),
        "left_sole_centroid_x_mean": float(vg[:, l_sole, 0].mean()),
        "right_sole_centroid_x_mean": float(vg[:, r_sole, 0].mean()),
        "left_palm_centroid_x_mean": float(vg[:, l_palm, 0].mean()),
        "right_palm_centroid_x_mean": float(vg[:, r_palm, 0].mean()),
    }


def stage_change(prev, cur) -> dict:
    d = np.asarray(cur, dtype=np.float64) - np.asarray(prev, dtype=np.float64)
    return {"max_abs_m": float(np.abs(d).max()), "l2_m": float(np.linalg.norm(d))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path,
                    default=BASE / "surface_contact_window60_90_v5_geometry_audit.json")
    ap.add_argument("--start", type=int, default=60)
    ap.add_argument("--end", type=int, default=90)
    args = ap.parse_args()
    sl = slice(args.start, args.end + 1)
    window = [args.start, args.end]
    scene = np.load(BASE / "scene_stage_ground_v3_pair_replay/scene_transforms.npz", allow_pickle=True)
    sets = json.loads((BASE / "surface_contact_sets_v1/contact_vertex_sets.json").read_text(encoding="utf-8"))
    topology = json.loads((ROOT / "research_records/engineering_validation/G20260918_offline_walker_frame_evidence_v1"
                           "/coarse_complete_model_v2_camera_rail/coarse_walker_model.json").read_text(encoding="utf-8-sig"))
    caps = capsule_endpoints(topology, scene, sl)
    sole_l, sole_r = sole_indices(sets, "left"), sole_indices(sets, "right")
    palm_l, palm_r = palm_indices(sets, "left"), palm_indices(sets, "right")

    sources = {
        "control_stage_c": BASE / "surface_contact_window60_90_v5_stage_d_no_contact/result_stage_c_joint.npz",
        "control_stage_d": BASE / "surface_contact_window60_90_v5_stage_d_no_contact/result_stage_d_no_contact_control.npz",
        "both_a010_stage_c": BASE / "surface_contact_window60_90_v5_both_a010/result_stage_c_joint.npz",
        "both_a010_stage_d": BASE / "surface_contact_window60_90_v5_both_a010/result_stage_d_surface_foot_hand.npz",
    }
    data = {name: np.load(path, allow_pickle=True) for name, path in sources.items()}
    for name, d in data.items():
        v = np.asarray(d["vertices"])
        if v.shape != (31, 6890, 3):
            raise ValueError(f"{name}: vertices shape {v.shape} != (31,6890,3)")

    vg = {name: load_ground(np.asarray(d["vertices"], dtype=np.float64), scene, sl) for name, d in data.items()}
    report = {"inputs": {k: str(v.resolve()) for k, v in sources.items()},
              "scene": str((BASE / "scene_stage_ground_v3_pair_replay/scene_transforms.npz").resolve()),
              "surface_sets": str((BASE / "surface_contact_sets_v1/contact_vertex_sets.json").resolve()),
              "handle_radius_assumption_m": HANDLE_RADIUS_M, "window": window}
    report["set_semantics"] = set_semantics(vg["control_stage_c"], sets)
    for name in data:
        report[name] = {"foot": foot_block(vg[name], sole_l, sole_r), "hand": hand_block(vg[name], caps, palm_l, palm_r)}
    report["stage_c_to_d_change"] = {
        "control": {"vertices": stage_change(data["control_stage_c"]["vertices"], data["control_stage_d"]["vertices"]),
                    "sole_zleft": _stats(vg["control_stage_d"][:, sole_l, 2] - vg["control_stage_c"][:, sole_l, 2]),
                    "sole_zright": _stats(vg["control_stage_d"][:, sole_r, 2] - vg["control_stage_c"][:, sole_r, 2])},
        "both_a010": {"vertices": stage_change(data["both_a010_stage_c"]["vertices"], data["both_a010_stage_d"]["vertices"]),
                      "sole_zleft": _stats(vg["both_a010_stage_d"][:, sole_l, 2] - vg["both_a010_stage_c"][:, sole_l, 2]),
                      "sole_zright": _stats(vg["both_a010_stage_d"][:, sole_r, 2] - vg["both_a010_stage_c"][:, sole_r, 2])},
    }
    report["coordinate_frame_note"] = ("sole z_G < 0 fraction near 1.0 is an engineering-consistency "
                                       "finding between the current ground transform and the SMPL surface, "
                                       "not a physical touch-down failure.")

    ctrl_c = report["control_stage_c"]["foot"]
    foot_blocked = (ctrl_c["left"]["fraction_z_negative"] > 0.90
                    and ctrl_c["right"]["fraction_z_negative"] > 0.90)
    ctrl_h = report["control_stage_c"]["hand"]
    hand_median_m = max(ctrl_h["left"]["residual_m"]["median"], ctrl_h["right"]["residual_m"]["median"])
    hand_neg = max(ctrl_h["left"]["fraction_residual_negative"], ctrl_h["right"]["fraction_residual_negative"])
    hand_suspect = hand_median_m > 0.040 and hand_neg < 0.10
    semantics_ok = report["set_semantics"]["index_range_ok"] and not report["set_semantics"]["sole_left_right_identical"]
    coord_status = "pass" if semantics_ok else "suspect"

    report["status"] = "completed_geometry_audit"
    report["engineering_validation_only"] = True
    report["foot_geometry_status"] = "blocked" if foot_blocked else "pass"
    report["hand_geometry_status"] = "suspect" if hand_suspect else "pass"
    report["coordinate_frame_status"] = coord_status
    report["weight_tuning_allowed"] = False
    report["next_action"] = ("geometry_fix" if (foot_blocked or hand_suspect or coord_status != "pass")
                             else "gradient_audit")
    report["physical_touch_validated"] = False
    report["load_bearing_validated"] = False
    report["true_3d_accuracy_validated"] = False

    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "foot_geometry_status", "hand_geometry_status",
                                             "coordinate_frame_status", "weight_tuning_allowed", "next_action")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
