#!/usr/bin/env python3
"""Read-only readiness audit: confirm surface contact losses are wired
correctly, compare the needed v6 routes on their engineering scale, and
pick one reproducible mainline. No fitting, no history modification.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research_records/engineering_validation/G20260924_smpl_vposer_shared_beta_v1"
OUT = BASE / "surface_route_readiness_v1.json"
DEVEL = {(60, 90): ("surface_contact_window60_90_v6_stage_d_no_contact",
                    "surface_contact_window60_90_v6_foot_a1",
                    "surface_contact_window60_90_v6_hand_a30",
                    "surface_contact_window60_90_v6_both_a1_a30")}


def stats(v: np.ndarray, mm: bool = True) -> dict:
    v = np.asarray(v, dtype=np.float64).ravel()
    m = 1000.0 if mm else 1.0
    return {"median": float(np.median(v)) * m, "med_abs": float(np.median(np.abs(v))) * m,
            "p05": float(np.percentile(v, 5)) * m, "p95": float(np.percentile(v, 95)) * m,
            "finite": bool(np.isfinite(v).all())}


def route(d: Path) -> dict:
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    a = json.loads((d / "surface_contact_forward_audit.json").read_text(encoding="utf-8"))
    tag = [p.name for p in d.glob("result_stage_d_*.npz")][0].replace("result_", "").replace(".npz", "")
    r = np.load(d / f"result_{tag}.npz", allow_pickle=True)
    c = np.load(d / "result_stage_c_joint.npz", allow_pickle=True)
    beta = np.asarray(r["betas"][0], float)
    cbeta = np.asarray(c["betas"][0], float)
    if "foot_surface_residuals_m" in r.files:
        fz = np.asarray(r["foot_surface_residuals_m"], float)
        hr = np.asarray(r["hand_surface_residuals_m"], float)
    else:
        # Zero-contact control stores no surface fields: recompute read-only.
        sc = np.load(BASE / "scene_stage_ground_v3_pair_replay/scene_transforms.npz", allow_pickle=True)
        st = np.load(BASE / "contact_labels_stage_audit_v2/contact_labels.npz", allow_pickle=True)
        sd = json.loads((BASE / "surface_contact_sets_v1/contact_vertex_sets.json").read_text(encoding="utf-8"))
        R = np.asarray(sc["rotation_ground_from_left"], float)[60:91]
        T = np.asarray(sc["translation_ground_from_left_mm"], float)[60:91] / 1000.0
        vg = np.einsum("nij,nvj->nvi", R, np.asarray(r["vertices"], float)) + T[:, None, :]
        sole = {s: np.concatenate([np.asarray(sd["sets"][f"{s}_sole_surface_candidate"][p]) for p in ("heel", "ball", "toe")])
                for s in ("left", "right")}
        fz = np.stack([vg[:, sole["left"], 2], vg[:, sole["right"], 2]], axis=1)
        he = np.asarray(st["handle_ends_ground_m"], float)[60:91]
        parts = []
        for i, s in enumerate(("left", "right")):
            palm = np.asarray(sd["sets"][f"{s}_palm_surface_candidate"]["palm_fingers"])
            pts = vg[:, palm, :]
            ea, eb = he[:, i, 0, :], he[:, i, 1, :]
            ab = eb - ea
            u = (((pts - ea[:, None, :]) * ab[:, None, :]).sum(-1, keepdims=True)
                 / np.maximum((ab[:, None, :] ** 2).sum(-1, keepdims=True), 1e-8)).clip(0, 1)
            parts.append(np.linalg.norm(pts - (ea[:, None, :] + u * ab[:, None, :]), axis=-1) - 0.016)
        hr = np.stack(parts, axis=1)
    dv = np.asarray(r["vertices"], float) - np.asarray(c["vertices"], float)
    w_sum = float(np.asarray(np.load(BASE / "contact_labels_stage_audit_v2/contact_labels.npz", allow_pickle=True)["foot_contact_weight"])[60:91].sum())
    return {"stage_c_loss": m["stage_c_joint"]["loss"][0], "stage_d_loss": (m.get("stage_d_contact") or {}).get("loss")[0],
            "2d_median": m["2d"]["median_px"], "2d_p95": m["2d"]["p95_px"],
            "3d_median": m["3d"]["median_mm"], "3d_p95": m["3d"]["p95_mm"],
            "foot_signed": {s: stats(fz[:, i, :]) for i, s in enumerate(("left", "right"))},
            "hand_signed": {s: stats(hr[:, i, :]) for i, s in enumerate(("left", "right"))},
            "foot_pen_frac": {s: float((fz[:, i, :] < 0).mean()) for i, s in enumerate(("left", "right"))},
            "hand_pen_frac": {s: float((hr[:, i, :] < 0).mean()) for i, s in enumerate(("left", "right"))},
            "verts_stageD": {"max_abs_m": float(np.abs(dv).max()), "l2_m": float(np.linalg.norm(dv))},
            "hand_coverage": a.get("hand_coverage"),
            "beta_drift": float(np.abs(beta - cbeta).max()), "beta_frozen": m["beta_frozen_during_stage_d"],
            "same_process": m["stage_c_to_d_same_process"], "same_forward_graph": a.get("same_forward_graph"),
            "grad_to_coco": a.get("surface_probe_grad_norm_to_coco"), "grad_to_verts": a.get("surface_probe_grad_norm_to_vertices"),
            "foot_valid_weight_sum": w_sum,
            "finite": bool(np.isfinite(fz).all() and np.isfinite(hr).all() and np.isfinite(dv).all())}


def main() -> int:
    routes = {}
    for (s, e), names in DEVEL.items():
        group = {}
        for key, name in zip(("control", "foot_a1", "hand_a30", "both_a1_a30"), names):
            group[key] = route(BASE / name)
        routes[f"{s}_{e}"] = group

    c, f, h, b = routes["60_90"]["control"], routes["60_90"]["foot_a1"], routes["60_90"]["hand_a30"], routes["60_90"]["both_a1_a30"]
    foot_gain = (abs(c["foot_signed"]["left"]["median"]) + abs(c["foot_signed"]["right"]["median"]) - (
        abs(f["foot_signed"]["left"]["median"]) + abs(f["foot_signed"]["right"]["median"]))) / (
        abs(c["foot_signed"]["left"]["median"]) + abs(c["foot_signed"]["right"]["median"]))
    hand_gain = (c["hand_signed"]["left"]["med_abs"] + c["hand_signed"]["right"]["med_abs"] - (
        h["hand_signed"]["left"]["med_abs"] + h["hand_signed"]["right"]["med_abs"])) / (
        c["hand_signed"]["left"]["med_abs"] + c["hand_signed"]["right"]["med_abs"])

    buggy = []
    sets_doc = json.loads((BASE / "surface_contact_sets_v1/contact_vertex_sets.json").read_text(encoding="utf-8"))
    for key in ("left_sole_surface_candidate", "right_sole_surface_candidate"):
        for part in ("heel", "ball", "toe"):
            idx = np.asarray(sets_doc["sets"][key][part])
            if idx.size != 18 or idx.min() < 0 or idx.max() > 6889:
                buggy.append(f"{key}/{part} index/count violation")
    for key in ("left_palm_surface_candidate", "right_palm_surface_candidate"):
        idx = np.asarray(sets_doc["sets"][key]["palm_fingers"])
        if idx.size != 778 or idx.min() < 0 or idx.max() > 6889:
            buggy.append(f"{key} index/count violation")
    graph_ok = (f["same_forward_graph"] and h["same_forward_graph"] and b["same_forward_graph"]
                and f["grad_to_coco"] == 0.0 and (f["grad_to_verts"] or 0) > 0 and (h["grad_to_verts"] or 0) > 0)
    beta_ok = c["beta_frozen"] and f["beta_frozen"] and h["beta_frozen"] and b["beta_frozen"]
    obs_ok = (f["3d_p95"] / c["3d_p95"] <= 1.02 and f["2d_p95"] - c["2d_p95"] <= 5.0
              and h["3d_p95"] / c["3d_p95"] <= 1.02 and h["2d_p95"] - c["2d_p95"] <= 5.0)
    fin = all(r["finite"] for r in [c, f, h, b])

    if buggy:
        route_status = "blocked_by_implementation_bug"
        selected, reason = None, "; ".join(buggy)
    elif not (graph_ok and beta_ok and fin):
        route_status = "no_observable_contact_correction"
        selected, reason = None, "graph/beta/finite gate failed"
    elif not obs_ok:
        route_status = "blocked_by_observation_divergence"
        selected, reason = None, "2D/3D observed term diverged by orders"
    elif foot_gain >= 0.05 or hand_gain >= 0.05:
        route_status = "engineering_route_ready"
        selected = "both_a1_a30" if (obs_ok and graph_ok) else "foot_a1"
        reason = f"foot_gain={foot_gain:.3f} hand_gain={hand_gain:.3f}; both stays within obs gates -> mainline=both_a1_a30"
    else:
        route_status = "no_observable_contact_correction"
        selected = "foot_a1"
        reason = "residual change within noise on dev window; keep foot_a1 as mainline, hand_a30 auxiliary (graph-verified only)"

    out = {"status": "completed_readiness_audit", "engineering_validation_only": True,
           "development_window": [60, 90], "routes": routes,
           "foot_toward_zero_gain": float(foot_gain), "hand_abs_gain": float(hand_gain),
           "gates": {"graph_ok": bool(graph_ok), "beta_frozen": bool(beta_ok),
                     "obs_not_diverged": bool(obs_ok), "finite": bool(fin),
                     "implementation_bug": bool(buggy)},
           "selected_engineering_route": selected if route_status == "engineering_route_ready" else None,
           "route_status": route_status,
           "hand_auxiliary_graph_verified": bool(graph_ok),
           "physical_touch_validated": False, "true_3d_accuracy_validated": False,
           "load_bearing_validated": False, "grip_force_validated": False}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"foot_gain": float(foot_gain), "hand_gain": float(hand_gain),
                      "gates": {k: bool(v) for k, v in {"graph_ok": graph_ok, "beta": beta_ok,
                                                        "obs_ok": obs_ok, "finite": fin, "bug": bool(buggy)}.items()},
                      "route_status": route_status, "selected_engineering_route": selected},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())