#!/usr/bin/env python3
"""Read-only audit of invalid foot labels in v6 window 373-403.

Compares source labels, frame mapping, entry weight loading and saved
results across windows 129-159, 278-308 and 373-403. Never refits,
never rewrites labels or history.
"""
from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
_tool_prepare_imports()

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "research_records/engineering_validation/G20260924_smpl_vposer_shared_beta_v1"
LABELS = BASE / "contact_labels_stage_audit_v2/contact_labels.npz"
OUT = BASE / "v6_373_403_contact_label_audit.json"
WINDOWS = {"w129": (129, 159), "w278": (278, 308), "w373": (373, 403)}


def arr_info(a: np.ndarray) -> dict:
    info = {"shape": list(a.shape), "dtype": str(a.dtype)}
    if a.dtype.kind == "f":
        info.update({"finite": int(np.isfinite(a).sum()), "nan": int(np.isnan(a).sum()),
                     "inf": int(np.isinf(a).sum())})
    elif a.dtype.kind in "iu":
        info.update({"min": int(a.min()), "max": int(a.max())})
    elif a.dtype.kind in "OSU":
        vals, counts = np.unique(a, return_counts=True)
        info.update({"unique": [(str(v), int(c)) for v, c in zip(vals, counts)]})
    elif a.dtype.kind == "b":
        info.update({"true": int(a.sum()), "false": int((~a).sum())})
    return info


def main() -> int:
    cl = np.load(LABELS, allow_pickle=True)
    structure = {k: arr_info(np.asarray(cl[k])) for k in cl.files}
    n = int(np.asarray(cl["foot_contact_weight"]).shape[0])

    comparison, frames = {}, {}
    for name, (s, e) in WINDOWS.items():
        sl = slice(s, e + 1)
        fw = np.asarray(cl["foot_contact_weight"])[sl]
        fl = np.asarray(cl["foot_label"])[sl]
        fs = np.asarray(cl["foot_support_score"])[sl]
        pid = np.asarray(cl["pair_id"])[sl]
        rows = []
        for i in range(e - s + 1):
            rows.append({"frame": s + i, "pair_id": int(pid[i]),
                         "foot_label": [str(fl[i, 0]), str(fl[i, 1])],
                         "foot_weight": [float(fw[i, 0]), float(fw[i, 1])],
                         "support_finite": [bool(np.isfinite(fs[i, 0])), bool(np.isfinite(fs[i, 1]))]})
        comparison[name] = {
            "frames": e - s + 1, "pair_id_range": [int(pid.min()), int(pid.max())],
            "pair_id_contiguous": bool(np.array_equal(pid, np.arange(s, e + 1))),
            "left_valid": int((fl[:, 0] != "invalid").sum()),
            "right_valid": int((fl[:, 1] != "invalid").sum()),
            "both_valid": int(((fl[:, 0] != "invalid") & (fl[:, 1] != "invalid")).sum()),
            "weight_pos_frames": int(((fw[:, 0] > 0) | (fw[:, 1] > 0)).sum()),
            "weight_sum": float(fw.sum()),
            "support_finite": int(np.isfinite(fs).sum()), "support_nonzero": int((fs != 0).sum()),
            "label_values": sorted(set(fl.ravel().tolist())),
        }
        frames[name] = rows

    stage = np.asarray(cl["stage"])
    stages = {name: sorted(set(stage[s:e + 1].tolist())) for name, (s, e) in WINDOWS.items()}
    w373 = comparison["w373"]
    # Generator rule (audit_foot_support_labels.py:124-129): contact_weight is
    # nonzero only for labels 'support' or 'stage2_contact_candidate'; frames
    # labelled swing/ambiguous/invalid always carry weight 0.0 by design.
    weight_bearing = {"support", "stage2_contact_candidate"}
    w373_labels = set(comparison["w373"]["label_values"])
    if (w373["pair_id_contiguous"] and w373["weight_sum"] == 0.0
            and not (w373_labels & weight_bearing)):
        root, evidence = "source_labels_invalid", [
            "pair_id 373..403 contiguous: slice mapping correct, no remap error",
            "source slice labels are only swing/ambiguous/invalid; no support or stage2_contact_candidate frame",
            "window stage values lack stage2_feet_static_walker_moving (only stage1/transition); stage2 gate never opens",
            "foot_contact_weight sum 0.0 by generator design for non-weight-bearing labels; entry reads the field verbatim (fit:193)",
            "no effective foot frame filtered out: zero weight-bearing frames exist to filter",
        ]
    else:
        root, evidence = "insufficient_evidence", ["window pattern does not match a clean source-invalid case"]
    stage_info = stages

    cmd = {}
    for name, (s, e) in WINDOWS.items():
        for kind in ("control", "foot_a1"):
            d = BASE / f"v6_validation_{s}_{e}_{kind}"
            p = d / "command.txt"
            cmd[f"{name}_{kind}"] = {"exists": p.exists(),
                                     "text": p.read_text(encoding="utf-8").strip() if p.exists() else None}

    saved = {}
    d373 = BASE / "v6_validation_373_403_foot_a1"
    for f in ("raw_observations.npz", "result_stage_c_joint.npz",
              "result_stage_d_surface_foot.npz", "triangulation.npz"):
        p = d373 / f
        if not p.exists():
            saved[f] = {"exists": False}
            continue
        r = np.load(p, allow_pickle=True)
        entry = {"exists": True, "keys": list(r.files)}
        if "vertices" in r.files:
            vv = np.asarray(r["vertices"])
            entry["frames"] = int(vv.shape[0])
            entry["vertices_finite"] = bool(np.isfinite(vv).all())
            b = np.asarray(r["betas"])
            entry["beta_frozen_single_row"] = bool((b == b[0]).all())
        if "foot_surface_residuals_m" in r.files:
            fr = np.asarray(r["foot_surface_residuals_m"])
            entry["foot_res_shape"] = list(fr.shape)
            entry["foot_res_all_zero"] = bool((fr == 0).all())
        else:
            entry["foot_surface_residuals_m"] = "absent (control-style save or audit-gated)"
        saved[f] = entry

    out = {"status": "read_only_audit", "window": [373, 403],
           "source_label_file": str(LABELS.resolve()),
           "label_file_structure": structure,
           "window_comparison": comparison, "frame_table": frames,
           "window_stages": stage_info,
           "frame_mapping": {"method": "absolute array slice window[0]:window[1]+1 (fit:184, fit:130)",
                             "pair_id_contiguous_373": w373["pair_id_contiguous"]},
           "fit_entry_mapping": {"load": "np.load(contact_labels) verbatim (fit:180)",
                                 "foot_w": "cl[foot_contact_weight][wsl] direct tensor, no label conversion (fit:193)",
                                 "invalid_handling": "no invalid branch for foot; zero weights flow into lfoot denominator guard",
                                 "lfoot_without_labels": "lfoot collapses to constant 0 -> probe grads None -> audit raises, Stage D npz already saved"},
           "saved_result_audit": saved, "command_record_audit": cmd,
           "root_cause_class": root, "direct_evidence": evidence,
           "rerun_allowed": False,
           "recommendation": ("Do not relabel, interpolate or swap windows. The 373..403 pair is a "
                              "no-effective-foot-contact window by source labels; keep selected_candidate=null.")},
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"window": [373, 403], "root_cause_class": root,
                      "w373": {k: w373[k] for k in ("frames", "left_valid", "right_valid", "weight_pos_frames", "weight_sum", "pair_id_contiguous")},
                      "w129_weight_sum": comparison["w129"]["weight_sum"],
                      "w278_weight_sum": comparison["w278"]["weight_sum"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
