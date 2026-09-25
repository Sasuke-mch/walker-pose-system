#!/usr/bin/env python3
"""Sole-mapping audit for the v5 surface-contact foot geometry (read-only).

Compares units, scene-transform direction, sole-vs-ankle anatomy and set
integrity using only saved v5 control outputs, current scene/set inputs and
the official male SMPL template. Never refits, never rewrites history.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research_records/engineering_validation/G20260924_smpl_vposer_shared_beta_v1"
CTRL = BASE / "surface_contact_window60_90_v5_stage_d_no_contact"
OUT = BASE / "surface_contact_window60_90_v5_sole_mapping_audit.json"

LEFT, RIGHT = "left", "right"
ANKLE = {LEFT: 15, RIGHT: 16}
PELVIS = (11, 12)
SOLE_PARTS = ("heel", "ball", "toe")


def stats(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    return {"shape": list(a.shape), "min": float(np.nanmin(a)), "max": float(np.nanmax(a)),
            "median": float(np.nanmedian(a)), "finite": bool(np.isfinite(a).all())}


def load_smpl_template():
    sys.path.insert(0, str(ROOT / "realtime_app"))
    from pose_app.smpl_coco_observation import _install_legacy_smpl_pickle_compatibility
    _install_legacy_smpl_pickle_compatibility()
    import pickle
    with open(ROOT / "models/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl", "rb") as fh:
        d = pickle.load(fh, encoding="latin1")
    return np.asarray(d["v_template"], dtype=np.float64), np.asarray(d["J"], dtype=np.float64)


def main() -> int:
    c = np.load(CTRL / "result_stage_c_joint.npz", allow_pickle=True)
    d = np.load(CTRL / "result_stage_d_no_contact_control.npz", allow_pickle=True)
    scene = np.load(BASE / "scene_stage_ground_v3_pair_replay/scene_transforms.npz", allow_pickle=True)
    sets = json.loads((BASE / "surface_contact_sets_v1/contact_vertex_sets.json").read_text(encoding="utf-8"))
    v = np.asarray(c["vertices"], dtype=np.float64)          # meters, SMPL forward
    p = np.asarray(c["predicted_coco"], dtype=np.float64)    # meters, regressed
    raw_mm = np.asarray(c["raw_triangulated_points"], dtype=np.float64)  # millimeters
    R = np.asarray(scene["rotation_ground_from_left"], dtype=np.float64)[60:91]
    Tmm = np.asarray(scene["translation_ground_from_left_mm"], dtype=np.float64)[60:91]
    idx = {s: np.concatenate([np.asarray(sets["sets"][f"{s}_sole_surface_candidate"][part])
                              for part in SOLE_PARTS]) for s in (LEFT, RIGHT)}

    # 1. Unit audit. Entry-code citations: fit_vposer_shared_beta.py:136
    # (triangulation mm -> m via /1000.0), :187 (scene translation mm -> m),
    # :194 (stereo baseline mm -> m), :216 (walker nodes mm -> m).
    unit = {
        "vertices": stats(v) | {"unit": "m", "basis": "SMPL forward output; median |x| ~0.3 inconsistent with mm"},
        "predicted_coco": stats(p) | {"unit": "m", "basis": "J_regressor_coco @ vertices, same frame as vertices"},
        "raw_triangulated_points": stats(raw_mm) | {"unit": "mm", "basis": "entry :136 converts with /1000.0 before use"},
        "translation_ground_from_left_mm": stats(Tmm) | {"unit": "mm", "basis": "entry :187 converts with /1000.0 before use"},
    }
    unit_status = "pass" if (abs(np.nanmedian(np.abs(v))) < 10 and abs(np.nanmedian(np.abs(p))) < 10
                             and abs(np.nanmedian(np.abs(raw_mm))) > 10 and abs(np.nanmedian(np.abs(Tmm))) > 10) else "blocked"

    def ground(x_m: np.ndarray) -> np.ndarray:
        return np.einsum("nij,nvj->nvi", R, x_m) + (Tmm / 1000.0)[:, None, :]

    def inverse(x_m: np.ndarray) -> np.ndarray:
        return np.einsum("nij,nvj->nvi", R.transpose(0, 2, 1), x_m - (Tmm / 1000.0)[:, None, :])

    # 2. Transform closed loop on model COCO and raw observations (raw -> m).
    raw_m = raw_mm / 1000.0
    pg, rg = ground(p), ground(raw_m)
    pi, ri = inverse(p), inverse(raw_m)
    tl = {}
    for name, arr in (("forward_model", pg), ("forward_raw", rg), ("inverse_model", pi), ("inverse_raw", ri)):
        tl[name] = {s: {"ankle_z_mm": float(np.median(arr[:, ANKLE[s], 2])) * 1000.0,
                        "ankle_xy_mm": (np.median(arr[:, ANKLE[s], :2], axis=0) * 1000.0).tolist()}
                    for s in (LEFT, RIGHT)}
        tl[name]["pelvis_z_mm"] = float(np.median(arr[:, list(PELVIS), :].mean(axis=1)[:, 2])) * 1000.0
    with np.errstate(invalid="ignore"):
        err_fwd = np.linalg.norm(pg - rg, axis=-1) * 1000.0
        err_inv = np.linalg.norm(pi - ri, axis=-1) * 1000.0
    m_acc = np.asarray(c["accepted_mask"], dtype=bool) & np.isfinite(err_fwd) & np.isfinite(err_inv)
    tl["model_obs_3d_forward_mm"] = {"median": float(np.median(err_fwd[m_acc])), "p95": float(np.percentile(err_fwd[m_acc], 95))}
    tl["model_obs_3d_inverse_mm"] = {"median": float(np.median(err_inv[m_acc])), "p95": float(np.percentile(err_inv[m_acc], 95))}
    # The same rigid map applied to model and observations preserves their
    # difference exactly; near-identical residuals mean the direction test is
    # uninformative on its own and must be reported ambiguous.
    gap = abs(tl["model_obs_3d_forward_mm"]["median"] - tl["model_obs_3d_inverse_mm"]["median"])
    if gap < 1e-6:
        transform_direction_status = "ambiguous"
    else:
        transform_direction_status = ("forward_supported" if tl["model_obs_3d_forward_mm"]["median"] < tl["model_obs_3d_inverse_mm"]["median"]
                                      else "inverse_supported")

    # 3. Sole-vs-ankle anatomy in the ground frame, Stage C and zero pose.
    tpl_v, tpl_J = load_smpl_template()
    vg_all = ground(v)
    pg_all = ground(p)
    anatomy = {}
    for s in (LEFT, RIGHT):
        sole = vg_all[:, idx[s], :]
        ankle = pg_all[:, [ANKLE[s]], :]
        vec = sole - ankle
        dz = vec[..., 2] * 1000.0
        c_xy = np.linalg.norm(sole[..., :2].mean(axis=1) - ankle[:, 0, :2], axis=-1) * 1000.0
        # Body left/right is defined in the left-camera frame (left = +x);
        # the ground-frame x axis is arbitrary, so side is checked in camera.
        same_side = bool(np.sign(v[:, idx[s], 0].mean()) == np.sign(p[:, ANKLE[s], 0].mean()))
        sole0 = tpl_v[idx[s]]
        dz0 = (sole0[:, 1] - tpl_J[7 if s == LEFT else 8, 1]) * 1000.0
        anatomy[s] = {
            "stage_c_dz_mm": {"median": float(np.median(dz)), "p05": float(np.percentile(dz, 5)),
                              "p95": float(np.percentile(dz, 95))},
            "stage_c_centroid_xy_mm": float(np.median(c_xy)),
            "same_side_as_ankle": same_side,
            "zero_pose_dz_mm": {"median": float(np.median(dz0)), "p05": float(np.percentile(dz0, 5)),
                                "p95": float(np.percentile(dz0, 95))},
            "zero_pose_lowest60_overlap": int(np.isin(np.argsort(tpl_v[:, 1])[:60], idx[s]).sum()),
        }

    def side_status(s: str) -> str:
        a = anatomy[s]
        if not a["same_side_as_ankle"]:
            return "blocked"
        drift = abs(a["stage_c_dz_mm"]["median"] - a["zero_pose_dz_mm"]["median"])
        if drift > 60.0:
            return "suspect"
        return "pass"

    left_status, right_status = side_status(LEFT), side_status(RIGHT)

    # 4. Set integrity.
    low60 = set(np.argsort(tpl_v[:, 1])[:60].tolist())
    integrity = {}
    for s in (LEFT, RIGHT):
        parts = {part: np.asarray(sets["sets"][f"{s}_sole_surface_candidate"][part]).tolist() for part in SOLE_PARTS}
        all_idx = [i for part in SOLE_PARTS for i in parts[part]]
        integrity[s] = {"counts": {k: len(v) for k, v in parts.items()},
                        "duplicates": len(all_idx) - len(set(all_idx)),
                        "low60_overlap": len(low60 & set(all_idx)),
                        "x_range": [float(tpl_v[np.asarray(all_idx), 0].min()),
                                    float(tpl_v[np.asarray(all_idx), 0].max())]}
    mirror_gap = abs(abs(np.mean(tpl_v[idx[LEFT], 0])) - abs(np.mean(tpl_v[idx[RIGHT], 0])))
    integrity["mirror_mean_absx_gap_m"] = float(mirror_gap)
    integrity["mirror_ok"] = bool(mirror_gap < 0.05 and integrity[LEFT]["duplicates"] == 0
                                  and integrity[RIGHT]["duplicates"] == 0)

    # 5. Verdict gates. No code fix is supported by this audit, so no rerun.
    if left_status == "blocked" and right_status == "blocked":
        fix, nxt = "surface_set_fix", "apply_single_fix_then_v6"
    elif left_status == "blocked" or right_status == "blocked":
        fix, nxt = "none", "stop"
    elif left_status == "suspect" or right_status == "suspect":
        fix, nxt = "none", "stop"
    elif unit_status == "pass":
        # Mappings and units reasonable: pose-stage question, gradient audit
        # next. Weight tuning stays forbidden (fit_rerun_allowed=false).
        fix, nxt = ("pose_only" if transform_direction_status == "forward_supported" else "none"), "audit_gradient"
    else:
        fix, nxt = "none", "stop"

    out = {"status": "completed_sole_mapping_audit", "engineering_validation_only": True,
           "unit_audit": unit, "unit_status": unit_status,
           "transform_closed_loop": tl, "transform_direction_status": transform_direction_status,
           "sole_ankle_anatomy": anatomy,
           "left_sole_mapping_status": left_status, "right_sole_mapping_status": right_status,
           "set_integrity": integrity,
           "single_supported_fix": fix, "fit_rerun_allowed": False, "next_action": nxt,
           "physical_touch_validated": False, "true_3d_accuracy_validated": False,
           "inputs": {"stage_c": str((CTRL / 'result_stage_c_joint.npz').resolve()),
                      "stage_d": str((CTRL / 'result_stage_d_no_contact_control.npz').resolve()),
                      "scene": str((BASE / 'scene_stage_ground_v3_pair_replay/scene_transforms.npz').resolve()),
                      "sets": str((BASE / 'surface_contact_sets_v1/contact_vertex_sets.json').resolve())}}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("unit_status", "transform_direction_status",
           "left_sole_mapping_status", "right_sole_mapping_status",
           "single_supported_fix", "fit_rerun_allowed", "next_action")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
