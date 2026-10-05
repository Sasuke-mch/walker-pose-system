"""Vectorized raw-fisheye Sampson diagnostic for two arm correspondence hypotheses."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from compare_capture_arm_bias import RECORDS, read, stats
from experiment_arm_triangulation import CAPTURES
from experiment_stereo_arm_assignment import prefer_crossed
from pose_app.body_observations import raw_side
from pose_app.fisheye_camera import load_stereo_fisheye


def normalized_rays(pixels, K, D):
    normalized = cv2.fisheye.undistortPoints(np.ascontiguousarray(pixels).reshape(-1, 1, 2), K, D).reshape(*pixels.shape[:-1], 2)
    return np.concatenate((normalized, np.ones((*pixels.shape[:-1], 1))), axis=-1)


def normalized_derivatives(pixels, K, D):
    derivatives = []
    for axis in range(2):
        shift = np.eye(2)[axis] * .001
        derivatives.append((normalized_rays(pixels + shift, K, D) - normalized_rays(pixels - shift, K, D)) / .002)
    return np.stack(derivatives, axis=-1)


def diagnostic(left, right, cal):
    l = normalized_rays(left, cal.K0, cal.D0)
    r = normalized_rays(right, cal.K1, cal.D1)
    dl = normalized_derivatives(left, cal.K0, cal.D0)
    dr = normalized_derivatives(right, cal.K1, cal.D1)
    tx, ty, tz = cal.T_cam0_to_cam1_mm
    skew = np.array([[0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0]])
    essential = skew @ cal.R_cam0_to_cam1
    el = l @ essential.T
    er = r @ essential
    value = np.sum(r * el, axis=-1)
    gradient_l = np.einsum("...i,...ij->...j", er, dl)
    gradient_r = np.einsum("...i,...ij->...j", el, dr)
    den = np.sum(gradient_l ** 2 + gradient_r ** 2, axis=-1)
    rms = np.sqrt(value ** 2 / np.maximum(den, 1e-20) / 2)
    p0 = np.column_stack((np.eye(3), np.zeros(3)))
    p1 = np.column_stack((cal.R_cam0_to_cam1, cal.T_cam0_to_cam1_mm))
    h = cv2.triangulatePoints(p0, p1, l[..., :2].reshape(-1, 2).T, r[..., :2].reshape(-1, 2).T)
    with np.errstate(divide="ignore", invalid="ignore"):
        xyz = (h[:3] / h[3]).T.reshape(*left.shape[:-1], 3)
    right_xyz = xyz @ cal.R_cam0_to_cam1.T + cal.T_cam0_to_cam1_mm
    ray1 = r @ cal.R_cam0_to_cam1
    cosine = np.sum(l * ray1, axis=-1) / (np.linalg.norm(l, axis=-1) * np.linalg.norm(ray1, axis=-1))
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    bounds = ((left >= 0) & (left < [cal.image_width, cal.image_height]) &
              (right >= 0) & (right < [cal.image_width, cal.image_height])).all(axis=-1)
    valid = bounds & np.isfinite(xyz).all(axis=-1) & (xyz[..., 2] > 0) & (right_xyz[..., 2] > 0) & (den > 1e-20) & (angle >= 1)
    return rms, valid & (rms <= 10)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError("refuse_existing_output")
    args.output.mkdir(parents=True)
    protocol = {
        "inputs": {k: str(v / "protocol.json") for k, v in CAPTURES.items()},
        "code": "realtime_app/experiment_fast_arm_assignment.py",
        "variable": "Replace nonlinear cost with raw-pixel Jacobian Sampson approximation and DLT depth",
        "pixel_derivative_step": .001, "rule": "same conservative 4 passes / +2 passes / 0.75 residual ratio",
        "success": "Report decision agreement against nonlinear diagnostic and all original manual slots, runtime",
        "limits": "RMS<=10 differs from each-view<=10; diagnostic only, not strict mainline gate. Local approximation can fail for large errors. Batch amortized timing is not streaming latency",
        "stop": "Frame mismatch abort; no point relabeling in production, no smoothing",
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    report = {}
    for capture, source in CAPTURES.items():
        p = read(source / "protocol.json")
        for model in ("PMPose", "Sapiens2"):
            v = [raw_side(Path(p["inputs"][model][c + "_raw"]), c) for c in ("left", "right")]
            n = p.get("sequence_frames", 448)
            if any(sorted(x) != list(range(n)) for x in v):
                raise ValueError("frame_mismatch")
            left, right = [np.stack([x[i][5:11, :2] for i in range(n)]) for x in v]
            cal = load_stereo_fisheye(p["inputs"][model]["calibration_dir"])
            start = time.perf_counter()
            same_error, same_pass = diagnostic(left, right, cal)
            crossed_error, crossed_pass = diagnostic(left, right[:, [1, 0, 3, 2, 5, 4]], cal)
            rows = []
            for i in range(n):
                same = {"pass": int(same_pass[i].sum()), "mean_px": float(same_error[i].mean())}
                crossed = {"pass": int(crossed_pass[i].sum()), "mean_px": float(crossed_error[i].mean())}
                rows.append({"frame": i, "same": same, "crossed": crossed, "prefer_crossed": prefer_crossed(same, crossed)})
            elapsed = (time.perf_counter() - start) * 1000 / n
            key = f"{capture}_{model}"
            exact = read(RECORDS / "G20261005_stereo_arm_assignment_v1" / (key + "_frames.json"))
            report[key] = {"batch_amortized_ms": elapsed, "frames": n,
                           "flagged": sum(r["prefer_crossed"] for r in rows),
                           "same_as_nonlinear": sum(a["prefer_crossed"] == b["prefer_crossed"] for a, b in zip(rows, exact)),
                           "additional_frames": [a["frame"] for a, b in zip(rows, exact) if a["prefer_crossed"] and not b["prefer_crossed"]],
                           "missed_frames": [a["frame"] for a, b in zip(rows, exact) if not a["prefer_crossed"] and b["prefer_crossed"]]}
            timing = []
            for i in range(n):
                start = time.perf_counter()
                e0, m0 = diagnostic(left[i:i + 1], right[i:i + 1], cal)
                e1, m1 = diagnostic(left[i:i + 1], right[i:i + 1, [1, 0, 3, 2, 5, 4]], cal)
                timing.append((time.perf_counter() - start) * 1000)
                np.testing.assert_allclose(e0[0], same_error[i], atol=1e-8)
                np.testing.assert_allclose(e1[0], crossed_error[i], atol=1e-8)
                np.testing.assert_array_equal(m0[0], same_pass[i])
                np.testing.assert_array_equal(m1[0], crossed_pass[i])
            report[key]["two_hypothesis_stream_ms"] = stats(timing)
            manual = []
            for row in read(source / "points.json"):
                if row["camera"] != "right" or row["methods"][model]["state"] != "compared":
                    continue
                fid = row["frame_index"]
                j = 4 if row["hand"] == "left" else 5
                m = np.asarray(row["manual"]["raw_px"])
                selected = 9 - j if rows[fid]["prefer_crossed"] else j
                manual.append({"frame": fid, "hand": row["hand"], "switched": rows[fid]["prefer_crossed"],
                               "before_px": float(np.linalg.norm(right[fid, j] - m)),
                               "after_px": float(np.linalg.norm(right[fid, selected] - m))})
            report[key]["manual"] = manual
            (args.output / (key + ".json")).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
