"""Analytic body-data-only translation counterfactual on frozen fits."""
import argparse
import json
from pathlib import Path

import numpy as np
from compare_capture_arm_bias import stats
from experiment_forearm_residual_parts import FITS


def optimal_translation(residual, weights):
    """residual is model-observation in mm, weights already include validity."""
    weights = np.asarray(weights, float)
    safe = np.where(weights[..., None] > 0, residual, 0.)
    denominator = weights.sum(axis=1)
    delta = -np.sum(safe * weights[..., None], axis=1) / np.maximum(denominator[:, None], 1e-12)
    return delta, denominator > 0


def weighted_rms(residual, weights):
    good = weights > 0
    return float(np.sqrt(np.sum(np.sum(residual[good] ** 2, axis=-1) * weights[good]) / weights[good].sum())) if good.any() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError("refuse_existing_output")
    args.output.mkdir(parents=True)
    protocol = {
        "inputs": {k: str(v) for k, v in FITS.items()},
        "code": "realtime_app/experiment_body_translation_audit.py",
        "variable": "Analytic per-frame rigid translation minimizing same weighted body observation SSE",
        "formula": "delta=-sum_j(w_j*(model_j-observed_j))/sum_j(w_j)",
        "controls": "Frozen shape, root, pose, detector, mask and quality. No coordinates exported or changed",
        "success": "Nonincrease of data SSE; show whether residual center shift is globally correctable",
        "stop": "Nonfinite accepted observations or negative weights abort; zero-weight frames explicit",
        "limits": "Body-data term only. Wrist-on cases have additional objective; no physical geometry/contact claim",
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    summary = {}
    for name, path in FITS.items():
        with np.load(path) as data:
            residual = data["predicted_coco"].astype(float) * 1000 - data["raw_triangulated_points"]
            weights = data["body_quality"].astype(float).copy()
            weights[~data["body_accepted"]] = 0
            if np.any(weights < 0) or not np.isfinite(weights).all() or not np.isfinite(residual[weights > 0]).all():
                raise ValueError("invalid_frozen_observations")
        delta, valid = optimal_translation(residual, weights)
        corrected = residual + delta[:, None, :]
        before, after = weighted_rms(residual, weights), weighted_rms(corrected, weights)
        if after > before + 1e-8:
            raise AssertionError("translation_objective_increased")
        summary[name] = {"translation_norm_mm": stats(np.linalg.norm(delta[valid], axis=1)),
            "zero_weight_frames": int((~valid).sum()), "weighted_body_rms_mm": [before, after],
            "groups": {label: {"before_mm": stats(np.linalg.norm(residual[:, joints], axis=-1)[weights[:, joints] > 0]),
                              "after_mm": stats(np.linalg.norm(corrected[:, joints], axis=-1)[weights[:, joints] > 0])}
                       for label, joints in (("arms", [7, 8, 9, 10]), ("legs", [13, 14, 15, 16]), ("shoulders", [5, 6]))}}
        (args.output / (name + ".json")).write_text(json.dumps({"delta_mm": delta.tolist(), "valid": valid.tolist()}, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
