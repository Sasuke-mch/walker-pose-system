"""Local pixel-to-3D sensitivity under fixed calibration; not calibrated uncertainty."""
import argparse
import json
from pathlib import Path

import numpy as np
from compare_capture_arm_bias import RECORDS, read, stats
from experiment_arm_triangulation import CAPTURES, JOINTS
from pose_app.fisheye_camera import fisheye_project_numpy, load_stereo_fisheye


def projections(x, cal):
    return np.concatenate((fisheye_project_numpy(x[None], cal.K0, cal.D0)[0],
        fisheye_project_numpy((cal.R_cam0_to_cam1 @ x + cal.T_cam0_to_cam1_mm)[None], cal.K1, cal.D1)[0]))


def sensitivity(x, cal, step_mm=0.01):
    x = np.asarray(x, float)
    if not np.isfinite(x).all() or x[2] <= 0 or (cal.R_cam0_to_cam1 @ x + cal.T_cam0_to_cam1_mm)[2] <= 0:
        return None
    J = np.column_stack([(projections(x + np.eye(3)[i] * step_mm, cal)
                          - projections(x - np.eye(3)[i] * step_mm, cal)) / (2 * step_mm)
                         for i in range(3)])
    _, singular, vt = np.linalg.svd(J, full_matrices=False)
    if singular[-1] < 1e-10:
        return None
    covariance = (vt.T * (1 / singular ** 2)) @ vt
    return covariance, float(singular[0] / singular[-1])


def length_sigma(a, b, covariance_a, covariance_b):
    delta = np.asarray(a) - np.asarray(b)
    length = float(np.linalg.norm(delta))
    if length < 1e-8:
        return None
    axis = delta / length
    return float(np.sqrt(axis @ (covariance_a + covariance_b) @ axis))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError("refuse_existing_output")
    args.output.mkdir(parents=True)
    protocol = {
        "input": str(RECORDS / "G20261005_arm_triangulation_v1"),
        "calibration_sources": {k: str(v / "protocol.json") for k, v in CAPTURES.items()},
        "code": "realtime_app/experiment_arm_sensitivity.py",
        "variable": "Finite-difference projection Jacobian at unchanged nonlinear point, step=0.01mm",
        "assumptions": "Fixed exact calibration, independent isotropic 1px per-coordinate noise, local linearization",
        "formula": "Cov=(J^T J)^-1; length std=sqrt(u^T(C_elbow+C_wrist)u)",
        "success": "Positive-depth slots with per-point condition/sensitivity and paired bone propagation",
        "stop": "Nonfinite, negative depth, rank deficient retain explicit unavailable state",
        "limits": "Not learned or calibrated uncertainty, ignores correlations/calibration/semantic errors and bias. No confidence interval or accuracy claim."
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    summaries = {}
    for capture, source in CAPTURES.items():
        p = read(source / "protocol.json")
        for model in ("PMPose", "Sapiens2"):
            cal = load_stereo_fisheye(p["inputs"][model]["calibration_dir"])
            key = f"{capture}_{model}"
            rows = read(RECORDS / "G20261005_arm_triangulation_v1" / (key + ".json"))
            records, covariances = [], {}
            for r in rows:
                result = sensitivity(r["refined_mm"], cal) if "refined_mm" in r else None
                row = {"frame": r["frame"], "joint": r["joint"], "state": "unavailable"}
                if result is not None:
                    cov, condition = result
                    covariances[(r["frame"], r["joint"])] = (np.asarray(r["refined_mm"]), cov)
                    row.update(state="available", condition=condition, xyz_rms_mm_per_px=float(np.sqrt(np.trace(cov))),
                               worst_axis_mm_per_px=float(np.sqrt(np.linalg.eigvalsh(cov)[-1])),
                               common_gate=r["common_diagnostic_pass"])
                records.append(row)
            bones = []
            for fid in range(p.get("sequence_frames", 448)):
                for hand in ("left", "right"):
                    a = covariances.get((fid, hand + "_elbow"))
                    b = covariances.get((fid, hand + "_wrist"))
                    bones.append({"frame": fid, "hand": hand,
                                  "length_std_mm_per_px": None if a is None or b is None else length_sigma(a[0], b[0], a[1], b[1])})
            summaries[key] = {"unavailable": sum(r["state"] == "unavailable" for r in records),
                "joints": {j: {"xyz_rms_mm_per_px": stats([r["xyz_rms_mm_per_px"] for r in records if r["joint"] == j and r["state"] == "available"]),
                    "condition": stats([r["condition"] for r in records if r["joint"] == j and r["state"] == "available"])} for j in JOINTS},
                "forearms": {h: stats([r["length_std_mm_per_px"] for r in bones if r["hand"] == h and r["length_std_mm_per_px"] is not None]) for h in ("left", "right")}}
            (args.output / (key + ".json")).write_text(json.dumps({"points": records, "bones": bones}, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
