"""Isolated two-hypothesis stereo arm correspondence experiment, no export to fit."""
import argparse
import json
from pathlib import Path

import numpy as np
from compare_capture_arm_bias import RECORDS, read, stats
from experiment_arm_triangulation import CAPTURES, evaluate
from pose_app.body_observations import raw_side
from pose_app.fisheye_camera import load_stereo_fisheye

SWAP = {5: 6, 6: 5, 7: 8, 8: 7, 9: 10, 10: 9}


def swap_arm(points):
    out = points.copy()
    for destination, source in SWAP.items():
        out[:, destination] = points[:, source]
    return out


def score(rows):
    costs = []
    passed = 0
    for row in rows:
        solved = row.get("candidate_solver", {})
        passed += solved.get("state") == "accepted_manual_reference_geometry"
        errors = solved.get("reprojection_error_px")
        if errors is not None and np.isfinite(errors).all():
            costs.append(float(np.mean(errors)))
    return {"pass": passed, "mean_px": float(np.mean(costs)) if len(costs) == 6 else None}


def prefer_crossed(same, crossed):
    return (same["mean_px"] is not None and crossed["mean_px"] is not None
            and crossed["pass"] >= 4 and crossed["pass"] >= same["pass"] + 2
            and crossed["mean_px"] < 0.75 * same["mean_px"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError("refuse_existing_output")
    args.output.mkdir(parents=True)
    protocol = {
        "inputs": {k: str(p / "protocol.json") for k, p in CAPTURES.items()},
        "same_source": str(RECORDS / "G20261005_arm_triangulation_v1"),
        "code": "realtime_app/experiment_stereo_arm_assignment.py",
        "variable": "Unchanged pairing versus swapping right-view shoulder/elbow/wrist labels jointly",
        "rule": "candidate pass>=4/6 and >=same+2, mean pixel error <0.75*same; otherwise unchanged",
        "controls": "Same nonlinear solve and gates for both; no other detector, labels or future used",
        "success": "Compare full geometry and original sparse label disagreement, preserve failures",
        "stop": "No missing frames permitted. No fitted-body export or production config change",
        "limits": "Stereo cannot tell which camera is mislabeled, or resolve simultaneous bilateral swaps. Right-only is a hypothesis, not identity truth."
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    summaries = {}
    for capture, source in CAPTURES.items():
        p = read(source / "protocol.json")
        labels = read(source / "points.json")
        for model in ("PMPose", "Sapiens2"):
            inputs = p["inputs"][model]
            views = [raw_side(Path(inputs[c + "_raw"]), c) for c in ("left", "right")]
            n = p.get("sequence_frames", 448)
            if any(sorted(v) != list(range(n)) for v in views):
                raise ValueError("frame_mismatch")
            left, right = [np.stack([v[i] for i in range(n)]) for v in views]
            key = f"{capture}_{model}"
            original = read(RECORDS / "G20261005_arm_triangulation_v1" / (key + ".json"))
            if len(original) != 6 * n:
                raise ValueError("original_size_mismatch")
            _, crossed = evaluate(left, swap_arm(right), load_stereo_fisheye(inputs["calibration_dir"]))
            frames = []
            for fid in range(n):
                old = score(original[fid * 6:(fid + 1) * 6])
                new = score(crossed[fid * 6:(fid + 1) * 6])
                frames.append({"frame": fid, "same": old, "crossed": new,
                               "prefer_crossed": prefer_crossed(old, new)})
            manual = []
            for row in labels:
                if row["camera"] != "right" or row["methods"][model]["state"] != "compared":
                    continue
                fid = row["frame_index"]
                joint = 9 if row["hand"] == "left" else 10
                m = np.asarray(row["manual"]["raw_px"])
                new_joint = SWAP[joint] if frames[fid]["prefer_crossed"] else joint
                manual.append({"frame": fid, "hand": row["hand"], "switched": frames[fid]["prefer_crossed"],
                               "original_px": float(np.linalg.norm(right[fid, joint, :2] - m)),
                               "hypothesis_px": float(np.linalg.norm(right[fid, new_joint, :2] - m))})
            summaries[key] = {
                "frames": n, "prefer_crossed": sum(f["prefer_crossed"] for f in frames),
                "original_pass_slots": sum(f["same"]["pass"] for f in frames),
                "hypothesis_pass_slots": sum(f["crossed" if f["prefer_crossed"] else "same"]["pass"] for f in frames),
                "manual_original_px": stats([r["original_px"] for r in manual]),
                "manual_hypothesis_px": stats([r["hypothesis_px"] for r in manual]),
                "manual": manual,
            }
            (args.output / (key + "_frames.json")).write_text(json.dumps(frames, indent=2), encoding="utf-8")
            (args.output / (key + "_crossed.json")).write_text(json.dumps(crossed, indent=2), encoding="utf-8")
            (args.output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
            print(key, "finished", flush=True)


if __name__ == "__main__":
    main()
