"""Read-only DLT versus existing raw-fisheye nonlinear arm triangulation.

This experiment never exports fitter inputs or changes production acceptance.
The manual-reference solver is reused as a numerical candidate, not as truth.
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from compare_capture_arm_bias import RECORDS, read, stats
from pose_app.body_observations import raw_side, raw_triangulate
from pose_app.fisheye_camera import load_stereo_fisheye
from walker_tools.hands.solve_annotated_wrist_targets import solve_point

JOINTS = {"left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7,
          "right_elbow": 8, "left_wrist": 9, "right_wrist": 10}
CAPTURES = {
    "old448": RECORDS / "G20261003_wrist_detector_manual_v2",
    "new360": RECORDS / "G20261003_repositioned360_manual_v1/comparison",
}


def geometry_reasons(x, errors, angle, cal):
    """Same numeric diagnostic limits for both estimates; not the live gate."""
    reasons = []
    if not np.isfinite(x).all() or not np.isfinite(errors).all():
        reasons.append("nonfinite")
    if x[2] <= 0 or (cal.R_cam0_to_cam1 @ x + cal.T_cam0_to_cam1_mm)[2] <= 0:
        reasons.append("nonpositive_depth")
    if np.any(np.asarray(errors) > 10):
        reasons.append("reprojection_over_10px")
    if not np.isfinite(angle) or angle < 1:
        reasons.append("ray_angle_under_1deg")
    return reasons


def evaluate(left, right, cal):
    baseline = raw_triangulate(left, right, cal)
    xyz, el, er, _, _, _, accepted, reason = baseline[:8]
    refined = np.full_like(xyz, np.nan)
    common = np.zeros(accepted.shape, bool)
    records = []
    for frame in range(len(left)):
        for name, joint in JOINTS.items():
            row = {"frame": frame, "joint": name, "baseline_body_mask": bool(accepted[frame, joint]),
                   "baseline_body_reason": reason[frame, joint], "state": "unavailable"}
            records.append(row)
            if not np.isfinite(xyz[frame, joint]).all():
                continue
            start = time.perf_counter()
            try:
                solved = solve_point(left[frame, joint, :2], right[frame, joint, :2], cal)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                row.update(state="solver_exception", exception=str(exc))
                continue
            row["solver_ms"] = (time.perf_counter() - start) * 1000
            row["candidate_solver"] = solved
            if "wrist_left_camera_m" not in solved:
                continue
            candidate = np.asarray(solved["wrist_left_camera_m"]) * 1000
            refined[frame, joint] = candidate
            errors = [float(el[frame, joint]), float(er[frame, joint])]
            reasons = geometry_reasons(xyz[frame, joint], errors, solved["ray_angle_deg"], cal)
            good = solved["state"] == "accepted_manual_reference_geometry"
            common[frame, joint] = not reasons and good
            row.update(state="evaluated", dlt_mm=xyz[frame, joint].tolist(),
                       refined_mm=candidate.tolist(), dlt_errors_px=errors,
                       dlt_diagnostic_reasons=reasons, common_diagnostic_pass=bool(common[frame, joint]),
                       displacement_mm=float(np.linalg.norm(candidate - xyz[frame, joint])),
                       pixel_sse_change=float(np.sum(np.square(solved["reprojection_error_px"]))
                                              - np.sum(np.square(errors))))
    summary = {"total_slots": len(records), "states": dict(Counter(r["state"] for r in records)),
               "candidate_reasons": dict(Counter(reason for r in records
                   for reason in r.get("candidate_solver", {}).get("reasons", []))), "joints": {}}
    for name in JOINTS:
        rows = [r for r in records if r["joint"] == name and r["state"] == "evaluated"]
        common_rows = [r for r in rows if r["common_diagnostic_pass"]]
        summary["joints"][name] = {
            "evaluated": len(rows), "common_pass": len(common_rows),
            "dlt_pass": sum(not r["dlt_diagnostic_reasons"] for r in rows),
            "refined_pass": sum(r["candidate_solver"]["state"] == "accepted_manual_reference_geometry" for r in rows),
            "displacement_all_mm": stats([r["displacement_mm"] for r in rows]),
            "displacement_common_mm": stats([r["displacement_mm"] for r in common_rows]),
            "sse_change_px2": stats([r["pixel_sse_change"] for r in rows]),
            "solver_ms": stats([r["solver_ms"] for r in rows]),
        }
    summary["forearms"] = {}
    for hand, elbow, wrist in (("left", 7, 9), ("right", 8, 10)):
        # Fixed original body mask, and separately common stricter diagnostic support.
        for support, mask in (("original_body", accepted[:, elbow] & accepted[:, wrist]),
                              ("common_diagnostic", common[:, elbow] & common[:, wrist])):
            valid_candidate = np.isfinite(refined[:, [elbow, wrist]]).all(axis=(1, 2))
            mask = mask & valid_candidate
            a = np.linalg.norm(xyz[mask, elbow] - xyz[mask, wrist], axis=1)
            b = np.linalg.norm(refined[mask, elbow] - refined[mask, wrist], axis=1)
            summary["forearms"][hand + "_" + support] = {
                "dlt_mm": stats(a), "refined_mm": stats(b), "paired_change_mm": stats(b - a),
                "missing_candidate_count": int(np.sum(~valid_candidate)),
            }
    return summary, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refuse_existing_output")
    args.output.mkdir(parents=True)
    protocols = {label: read(path / "protocol.json") for label, path in CAPTURES.items()}
    protocol = {
        "question": "Does raw-fisheye nonlinear triangulation materially explain arm length mismatch?",
        "inputs": protocols, "frames": {"old448": 448, "new360": 360},
        "variable": "DLT versus existing solve_point nonlinear pixel least squares, max_nfev=100",
        "controls": "Identical detections/calibration, no label use, six arm joints, all frames retained",
        "code": ["realtime_app/experiment_arm_triangulation.py", "realtime_app/pose_app/body_observations.py",
                 "realtime_app/walker_tools/hands/solve_annotated_wrist_targets.py"],
        "diagnostic_gate": "positive depths; each view <=10px; ray angle >=1deg; solver success for candidate",
        "success": "Numerically verified paired metrics and timing; no automatic mainline promotion",
        "stop": "Missing/mismatched frames abort; rejected/nonfinite/negative depth retained",
        "limits": "No independent elbow/3D truth. Lower pixel error is objective optimization, not accuracy.",
        "output": str(args.output.resolve()),
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    summaries = {}
    for label, p in protocols.items():
        for method in ("PMPose", "Sapiens2"):
            inputs = p["inputs"][method]
            views = [raw_side(Path(inputs[side + "_raw"]), side) for side in ("left", "right")]
            n = protocol["frames"][label]
            if any(sorted(view) != list(range(n)) for view in views):
                raise ValueError("frame_mismatch")
            arrays = [np.stack([view[i] for i in range(n)]) for view in views]
            summary, records = evaluate(*arrays, load_stereo_fisheye(inputs["calibration_dir"]))
            key = label + "_" + method
            summaries[key] = summary
            (args.output / (key + ".json")).write_text(json.dumps(records, indent=2), encoding="utf-8")
            (args.output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
            print(key, "finished", flush=True)


if __name__ == "__main__":
    main()
