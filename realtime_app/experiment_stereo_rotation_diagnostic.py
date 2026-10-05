"""Held-out epipolar consistency of bounded rotation perturbation, never calibration."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from compare_capture_arm_bias import read, stats
from experiment_arm_triangulation import CAPTURES
from experiment_fast_arm_assignment import normalized_derivatives, normalized_rays
from pose_app.body_observations import raw_side, raw_triangulate
from pose_app.fisheye_camera import load_stereo_fisheye
from scipy.optimize import least_squares


def residual(rotation_vector, rays_l, rays_r, jac_l, jac_r, R, t):
    tx, ty, tz = t
    skew = np.array([[0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0]])
    correction = cv2.Rodrigues(np.asarray(rotation_vector, float))[0]
    essential = skew @ correction @ R
    el, er = rays_l @ essential.T, rays_r @ essential
    error = np.sum(rays_r * el, axis=-1)
    gl = np.einsum("...i,...ij->...j", er, jac_l)
    gr = np.einsum("...i,...ij->...j", el, jac_r)
    return error / np.sqrt(np.maximum(np.sum(gl * gl + gr * gr, axis=-1), 1e-20))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError("refuse_existing_output")
    args.output.mkdir(parents=True)
    protocol = {
        "inputs": {k: str(v / "protocol.json") for k, v in CAPTURES.items()},
        "code": "realtime_app/experiment_stereo_rotation_diagnostic.py",
        "variable": "One constant three-axis stereo rotation perturbation bounded +/-1 degree per axis, translation fixed",
        "train": "first half frames, shoulders and hips 5,6,11,12 only, frozen original positive-depth/body mask",
        "test": "second half torso and both halves held-out elbows/wrists, separately reported",
        "solver": "least_squares Huber f_scale3px max_nfev50, start zero, no hyperparameter search",
        "success": "All four capture/model diagnostics report effects and fail states; calibration files never written",
        "stop": "Input mismatch abort, unsuccessful solver not promoted",
        "limits": "May absorb consistent detector bias or wrong correspondence; human landmarks are not independent calibration evidence. No metric accuracy claim. Time halves not independent subjects."
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    summaries = {}
    for capture, source in CAPTURES.items():
        p = read(source / "protocol.json")
        for model in ("PMPose", "Sapiens2"):
            inputs = p["inputs"][model]
            views = [raw_side(Path(inputs[c + "_raw"]), c) for c in ("left", "right")]
            n = p.get("sequence_frames", 448)
            if any(sorted(v) != list(range(n)) for v in views):
                raise ValueError("frame_mismatch")
            left, right = [np.stack([v[i] for i in range(n)]) for v in views]
            cal = load_stereo_fisheye(inputs["calibration_dir"])
            mask = raw_triangulate(left, right, cal)[6]
            l, r = normalized_rays(left[..., :2], cal.K0, cal.D0), normalized_rays(right[..., :2], cal.K1, cal.D1)
            jl, jr = normalized_derivatives(left[..., :2], cal.K0, cal.D0), normalized_derivatives(right[..., :2], cal.K1, cal.D1)
            training = np.zeros_like(mask)
            training[:n // 2, [5, 6, 11, 12]] = True
            training &= mask
            def values(x, l=l, r=r, jl=jl, jr=jr, cal=cal):
                return residual(x, l, r, jl, jr, cal.R_cam0_to_cam1, cal.T_cam0_to_cam1_mm)
            def objective(x, training=training, values=values):
                return values(x)[training]
            result = least_squares(objective, np.zeros(3), bounds=(-np.pi / 180, np.pi / 180),
                                   loss="huber", f_scale=3., max_nfev=50)
            before, after = values(np.zeros(3)), values(result.x)
            key = f"{capture}_{model}"
            groups = {}
            for half, sl in (("train_time", slice(0, n // 2)), ("test_time", slice(n // 2, n))):
                for group, js in (("torso", [5, 6, 11, 12]), ("elbows", [7, 8]), ("wrists", [9, 10])):
                    active = mask[sl][:, js]
                    b, a = np.abs(before[sl][:, js])[active], np.abs(after[sl][:, js])[active]
                    groups[f"{half}_{group}"] = {"before_px": stats(b), "after_px": stats(a), "improved": int(np.sum(a < b)),
                                                 "unavailable": int(np.sum(~active))}
            summaries[key] = {"rotation_delta_deg": np.degrees(result.x).tolist(), "solver_success": bool(result.success),
                              "nfev": result.nfev, "active_bounds": result.active_mask.tolist(), "groups": groups}
            np.savez_compressed(args.output / (key + ".npz"), signed_before_px=before, signed_after_px=after, original_body_mask=mask)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
