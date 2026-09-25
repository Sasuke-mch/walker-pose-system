#!/usr/bin/env python3
"""Summarize the v6 foot_a1 independent-window validation (read-only)."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research_records/engineering_validation/G20260924_smpl_vposer_shared_beta_v1"
WINDOWS = [(129, 159), (278, 308), (373, 403)]
OUT = BASE / "surface_contact_v6_independent_validation.json"


def foot_stats(d: Path, tag: str) -> dict:
    r = np.load(d / f"result_{tag}.npz", allow_pickle=True)
    if "foot_surface_residuals_m" in r:
        z = np.asarray(r["foot_surface_residuals_m"], float)
    else:
        v = np.asarray(r["vertices"], float)
        sc = np.load(BASE / "scene_stage_ground_v3_pair_replay/scene_transforms.npz", allow_pickle=True)
        st = np.load(BASE / "contact_labels_stage_audit_v2/contact_labels.npz", allow_pickle=True)
        sets = json.loads((BASE / "surface_contact_sets_v1/contact_vertex_sets.json").read_text(encoding="utf-8"))
        s, e = int(str(d.name).split("_")[2]), int(str(d.name).split("_")[3])
        R = np.asarray(sc["rotation_ground_from_left"], float)[s:e + 1]
        T = np.asarray(sc["translation_ground_from_left_mm"], float)[s:e + 1] / 1000.0
        vg = np.einsum("nij,nvj->nvi", R, v) + T[:, None, :]
        sole = {x: np.concatenate([np.asarray(sets["sets"][f"{x}_sole_surface_candidate"][p]) for p in ("heel", "ball", "toe")])
                for x in ("left", "right")}
        z = np.stack([vg[:, sole["left"], 2], vg[:, sole["right"], 2]], axis=1)
    out = {}
    for i, s in enumerate(("left", "right")):
        v = (z[:, i, :] * 1000.0).ravel()
        out[s] = {"signed_median_mm": float(np.median(v)),
                  "median_abs_mm": float(np.median(np.abs(v))),
                  "p05_mm": float(np.percentile(v, 5)), "p95_mm": float(np.percentile(v, 95)),
                  "frac_z_neg": float((v < 0).mean()),
                  "frac_z_lt_25mm": float((v < -25).mean()),
                  "finite": bool(np.isfinite(v).all())}
    return out


def main() -> int:
    windows = {}
    for s, e in WINDOWS:
        w = {}
        for kind in ("control", "foot_a1"):
            d = BASE / f"v6_validation_{s}_{e}_{kind}"
            try:
                m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
                a = json.loads((d / "surface_contact_forward_audit.json").read_text(encoding="utf-8"))
            except FileNotFoundError:
                w[kind] = {"completed": False, "reason": "run did not finish; no metrics.json"}
                continue
            tag = [p.name for p in d.glob("result_stage_d_*.npz")][0].replace("result_", "").replace(".npz", "")
            r = np.load(d / f"result_{tag}.npz", allow_pickle=True)
            c = np.load(d / "result_stage_c_joint.npz", allow_pickle=True)
            dv = np.asarray(r["vertices"], float) - np.asarray(c["vertices"], float)
            w[kind] = {"completed": True,
                       "loss_c": m["stage_c_joint"]["loss"], "loss_d": (m.get("stage_d_contact") or {}).get("loss"),
                       "reproj_median": m["2d"]["median_px"], "reproj_p95": m["2d"]["p95_px"],
                       "reproj_l": m["2d"]["left_p95_px"], "reproj_r": m["2d"]["right_p95_px"],
                       "tri_median": m["3d"]["median_mm"], "tri_p95": m["3d"]["p95_mm"],
                       "foot": foot_stats(d, tag),
                       "verts_max_abs_m": float(np.abs(dv).max()), "verts_l2_m": float(np.linalg.norm(dv)),
                       "beta_drift": float(np.abs(np.asarray(r["betas"][0], float) - np.asarray(c["betas"][0], float)).max()),
                       "beta_frozen": m["beta_frozen_during_stage_d"],
                       "same_process": m["stage_c_to_d_same_process"],
                       "same_forward_graph": a.get("same_forward_graph"),
                       "grad_to_coco": a.get("surface_probe_grad_norm_to_coco"),
                       "grad_to_verts": a.get("surface_probe_grad_norm_to_vertices"),
                       "hand_mode": a.get("hand_contact_mode")}
        c, f = w.get("control", {}), w.get("foot_a1", {})
        if c.get("completed") and f.get("completed"):
            cm = abs(c["foot"]["left"]["signed_median_mm"]) + abs(c["foot"]["right"]["signed_median_mm"])
            fm = abs(f["foot"]["left"]["signed_median_mm"]) + abs(f["foot"]["right"]["signed_median_mm"])
            gain = (cm - fm) / cm
            gates = {
                "3d_p95_within_2pct": (f["tri_p95"] - c["tri_p95"]) / c["tri_p95"] <= 0.02,
                "2d_p95_within_5px": (f["reproj_p95"] - c["reproj_p95"]) <= 5.0,
                "foot_abs_gain_5pct": gain >= 0.05,
                "pen_within_1pp": all((f["foot"][x]["frac_z_neg"] - c["foot"][x]["frac_z_neg"]) <= 0.01 for x in ("left", "right")),
                "beta_frozen": f["beta_frozen"],
                "graph": bool(f["same_forward_graph"]) and f["grad_to_coco"] == 0.0 and (f["grad_to_verts"] or 0) > 0,
                "finite": all(v for k in ("foot",) for x in ("left", "right") for v in [f[k][x]["finite"], c[k][x]["finite"]]),
            }
            w["gain"] = gain
            w["gates"] = {k: bool(v) for k, v in gates.items()}
            w["passed"] = bool(all(gates.values()))
        else:
            w["passed"] = False
            w["reason"] = "incomplete pair"
        windows[f"{s}_{e}"] = w
    ok = all(w.get("passed", False) for w in windows.values())
    out = {"status": "v6_independent_validation", "engineering_validation_only": True,
           "config": {"foot": 1.0, "hand": 0.0, "obs3d": 1.0, "obs2d": 0.20,
                      "stage_d_3d": 0.70, "stage_d_2d": 0.020, "contact_steps": 80},
           "windows": windows,
           "selected_candidate": "foot_a1" if ok else None,
           "stop_reason": None if ok else "foot_a1_not_stable_across_independent_windows",
           "engineering_validation_candidate": "foot_a1" if ok else None,
           "physical_touch_validated": False, "true_3d_accuracy_validated": False,
           "load_bearing_validated": False, "grip_force_validated": False}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {"gain": v.get("gain"), "passed": v.get("passed"), "reason": v.get("reason")} for k, v in windows.items()},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
