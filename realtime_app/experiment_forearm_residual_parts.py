"""Exact endpoint residual decomposition of frozen body fits, without fitting."""
import argparse
import json
from pathlib import Path

import numpy as np
from compare_capture_arm_bias import RECORDS, stats

FITS = {
    "old_pmpose_wrist_on": RECORDS / "G20261003_smplh_ablation_v1/pmpose_baseline/fit/result.npz",
    "old_pmpose_wrist_off": RECORDS / "G20261003_smplh_ablation_v1/pmpose_no_wrist/fit/result.npz",
    "old_sapiens_wrist_on": RECORDS / "G20261002_sapiens2_body_mainline448_v2/smplh_cold_start_v2_joint300/fit/result.npz",
    "new_pmpose_wrist_off": RECORDS / "G20261003_repositioned360_manual_v1/pmpose/fit/result.npz",
}


def decompose(model, observed):
    """Endpoints (...,2,3) in mm, exact unweighted SSE identity."""
    model, observed = np.asarray(model, float), np.asarray(observed, float)
    bm = model[..., 1, :] - model[..., 0, :]
    bo = observed[..., 1, :] - observed[..., 0, :]
    lm, lo = np.linalg.norm(bm, axis=-1), np.linalg.norm(bo, axis=-1)
    center = 2 * np.sum((model.mean(axis=-2) - observed.mean(axis=-2)) ** 2, axis=-1)
    length = .5 * (lm - lo) ** 2
    # Dot-product form is defined even for a zero-length bone.
    direction = lm * lo - np.sum(bm * bo, axis=-1)
    actual = np.sum((model - observed) ** 2, axis=(-1, -2))
    return {"center_mm2": center, "length_mm2": length, "direction_mm2": direction, "total_mm2": actual}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError("refuse_existing_output")
    args.output.mkdir(parents=True)
    protocol = {
        "inputs": {k: str(v) for k, v in FITS.items()},
        "code": "realtime_app/experiment_forearm_residual_parts.py",
        "variable": "No new candidate; exact decomposition of frozen elbow+wrist squared residual",
        "formula": "SSE=2*||midpoint difference||^2 + 0.5*(length difference)^2 + Lm*Lo*(1-cos angle)",
        "controls": "Original predicted_coco, raw_triangulated_points and body_accepted; both endpoints required",
        "success": "Identity numerical closure on all finite valid pairs; report all components and unavailable counts",
        "stop": "Frame mismatch or identity failure abort, no fitting change",
        "limits": "Unweighted endpoint algebra, not decomposition of robust fitting objective or causal proof. Only old wrist-on/off is controlled A/B."
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    summaries = {}
    for name, path in FITS.items():
        with np.load(path) as data:
            model = data["predicted_coco"].astype(float) * 1000
            observed = data["raw_triangulated_points"].astype(float)
            mask = data["body_accepted"]
            if not np.array_equal(data["pair_id"], np.arange(len(model))):
                raise ValueError("frame_mismatch")
        rows = []
        summaries[name] = {}
        for hand, e, w in (("left", 7, 9), ("right", 8, 10)):
            good = mask[:, e] & mask[:, w] & np.isfinite(model[:, [e, w]]).all(axis=(1, 2)) & np.isfinite(observed[:, [e, w]]).all(axis=(1, 2))
            values = decompose(model[good][:, [e, w]], observed[good][:, [e, w]])
            closure = values["center_mm2"] + values["length_mm2"] + values["direction_mm2"] - values["total_mm2"]
            np.testing.assert_allclose(closure, 0, atol=1e-6)
            total = float(values["total_mm2"].sum())
            summaries[name][hand] = {
                "frames": len(model), "used": int(good.sum()), "unavailable": int((~good).sum()),
                "max_closure_mm2": float(np.max(np.abs(closure))) if len(closure) else None,
                "parts": {k: {"stats": stats(v), "fraction_of_sum_sse": float(v.sum() / total) if total > 0 else None} for k, v in values.items()},
            }
            for fid in range(len(model)):
                row = {"frame": fid, "hand": hand, "state": "available" if good[fid] else "unavailable"}
                if good[fid]:
                    row.update({k: float(v) for k, v in decompose(model[fid, [e, w]], observed[fid, [e, w]]).items()})
                rows.append(row)
        (args.output / (name + ".json")).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
