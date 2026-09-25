#!/usr/bin/env python3
"""Summarize the two Stage C gradient audits (read-only, no fitting)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research_records/engineering_validation/G20260924_smpl_vposer_shared_beta_v1"
SURF_RUN = BASE / "surface_contact_window60_90_v5_gradient_audit_surface_run"
NOSURF_RUN = BASE / "surface_contact_window60_90_v5_gradient_audit_nosurface_run"
SURF_JSON = BASE / "surface_contact_window60_90_v5_gradient_audit_surface.json"
NOSURF_JSON = BASE / "surface_contact_window60_90_v5_gradient_audit_nosurface.json"


def load(run: Path) -> dict:
    return json.loads((run / "gradient_audit.json").read_text(encoding="utf-8"))


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


def main() -> int:
    s, n = load(SURF_RUN), load(NOSURF_RUN)
    for doc in (s, n):
        assert doc["stage_d_executed"] is False and doc["beta_optimized_in_audit"] is False
        assert all(t["finite"] for t in doc["terms"].values())
    consistent = all(
        rel(s["terms"][t]["gradient_l2"][p], n["terms"][t]["gradient_l2"][p]) <= 1e-6
        for t in ("obs3d", "obs2d") for p in ("latent", "root", "transl")
    ) and all(
        rel(s["terms"][t]["term_value"], n["terms"][t]["term_value"]) <= 1e-6
        for t in ("obs3d", "obs2d")
    )
    if not consistent:
        raise SystemExit("Stage C diverged between audits; stop")
    obs = s["terms"]["obs3d"]["gradient_l2"]["all"] + s["terms"]["obs2d"]["gradient_l2"]["all"]
    foot = s["terms"]["foot"]["gradient_l2"]["all"]
    hand = s["terms"]["hand"]["gradient_l2"]["all"]
    surf = foot + hand
    ratio = surf / max(obs, 1e-12)
    broken = (foot == 0.0 and hand == 0.0)
    if broken:
        gstatus, nxt = "surface_graph_broken", "fix_graph"
    elif ratio < 1e-3:
        gstatus, nxt = "surface_gradient_negligible", "calibrate_contact_coefficient"
    elif ratio < 0.1:
        gstatus, nxt = "surface_gradient_weak_but_active", "calibrate_contact_coefficient"
    elif ratio <= 1.0:
        gstatus, nxt = "surface_gradient_comparable", "calibrate_contact_coefficient"
    else:
        gstatus, nxt = "surface_gradient_dominant", "stop"
    summary = {
        "status": "completed_stage_c_gradient_audit",
        "engineering_validation_only": True,
        "stage_d_executed": False,
        "beta_optimized_in_audit": False,
        "surface_terms": {t: s["terms"][t] for t in ("foot", "hand")},
        "observation_terms": {t: s["terms"][t] for t in ("obs3d", "obs2d")},
        "surface_to_observation_gradient_ratio": ratio,
        "stage_c_consistent_between_audits": True,
        "surface_graph_status": "surface_graph_broken" if broken else "pass",
        "surface_gradient_status": gstatus,
        "weight_selection_allowed": False,
        "next_action": nxt,
        "physical_touch_validated": False,
        "true_3d_accuracy_validated": False,
    }
    SURF_JSON.write_text(json.dumps(s, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    NOSURF_JSON.write_text(json.dumps(n, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (BASE / "surface_contact_window60_90_v5_gradient_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"term_values": {t: s["terms"][t]["term_value"] for t in s["terms"]},
                      "grad_l2_all": {t: s["terms"][t]["gradient_l2"]["all"] for t in s["terms"]},
                      "surface_to_observation_gradient_ratio": ratio,
                      "consistent": True,
                      "surface_graph_status": summary["surface_graph_status"],
                      "surface_gradient_status": gstatus,
                      "next_action": nxt}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
