"""Single-detector causal arm warnings; never reject, swap or fill observations."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from compare_capture_arm_bias import read, stats
from experiment_arm_triangulation import CAPTURES
from pose_app.body_observations import raw_side


def features(current, previous=None):
    arm = np.asarray(current, float)[[7, 8, 9, 10], :2]
    if not np.isfinite(arm).all():
        return {"state": "invalid_pixels"}
    lengths = np.linalg.norm(arm[:2] - arm[2:], axis=1)
    scale = float(np.median(lengths))
    if scale < 1:
        return {"state": "degenerate_scale"}
    out = {"state": "available", "forearm_scale_px": scale,
           "wrist_separation_ratio": float(np.linalg.norm(arm[2] - arm[3]) / scale),
           "jump_ratio": None, "crossed_assignment_cost_ratio": None}
    if previous is not None:
        old = np.asarray(previous, float)[[7, 8, 9, 10], :2]
        if np.isfinite(old).all():
            old_scale = float(np.median(np.linalg.norm(old[:2] - old[2:], axis=1)))
            if old_scale >= 1:
                own = np.linalg.norm(arm[2:] - old[2:], axis=1)
                crossed = np.linalg.norm(arm[2:] - old[2:][::-1], axis=1)
                out["jump_ratio"] = float(own.max() / old_scale)
                out["crossed_assignment_cost_ratio"] = float(crossed.sum() / max(own.sum(), 1e-8))
    return out


def flagged(row, separation, jump):
    return row["state"] == "available" and (
        row["wrist_separation_ratio"] < separation
        or (row["jump_ratio"] is not None and row["jump_ratio"] > jump))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError("refuse_existing_output")
    args.output.mkdir(parents=True)
    protocol = {
        "inputs": {k: str(v / "protocol.json") for k, v in CAPTURES.items()},
        "code": "realtime_app/experiment_arm_semantics.py",
        "variable": "diagnostic-only same-model wrist separation and one-frame jump",
        "separation_ratios": [0.1, 0.2, 0.3], "jump_ratios": [0.5, 1.0, 2.0],
        "success": "Report all-frame flags and sparse-label disagreement without choosing tuned thresholds",
        "stop": "frame mismatch abort; missing/invalid states retained",
        "controls": "No future, smoothing, other model, manual input or image inference used for features",
        "limits": "Close hands and rapid real motion may be legitimate; flags are not semantic error truth",
        "reference": "All original sparse labels retained including known anomalous labels; 100px report-only split",
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    summary, all_rows = {}, []
    for capture, source in CAPTURES.items():
        p = read(source / "protocol.json")
        labels = read(source / "points.json")
        for model in ("PMPose", "Sapiens2"):
            for camera in ("left", "right"):
                observations = raw_side(Path(p["inputs"][model][camera + "_raw"]), camera)
                n = p.get("sequence_frames", 448)
                if sorted(observations) != list(range(n)):
                    raise ValueError("frame_mismatch")
                start = time.perf_counter()
                rows = [dict(features(observations[i], observations.get(i - 1)), frame=i,
                             capture=capture, model=model, camera=camera) for i in range(n)]
                elapsed = (time.perf_counter() - start) * 1000 / n
                all_rows.extend(rows)
                manual = [(r["frame_index"], r["methods"][model]["error_px"]) for r in labels
                          if r["camera"] == camera and r["methods"][model]["state"] == "compared"]
                variants = []
                for separation in protocol["separation_ratios"]:
                    for jump in protocol["jump_ratios"]:
                        mask = [flagged(r, separation, jump) for r in rows]
                        variants.append({"separation": separation, "jump": jump,
                            "flagged_frames": sum(mask), "total_frames": n,
                            "manual_flagged_px": stats([err for fid, err in manual if mask[fid]]),
                            "manual_unflagged_px": stats([err for fid, err in manual if not mask[fid]]),
                            "over100_flagged": sum(err > 100 and mask[fid] for fid, err in manual),
                            "over100_total": sum(err > 100 for _, err in manual)})
                key = f"{capture}_{model}_{camera}"
                summary[key] = {"feature_mean_ms": elapsed, "variants": variants,
                                "invalid_frames": sum(r["state"] != "available" for r in rows)}
    (args.output / "records.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
