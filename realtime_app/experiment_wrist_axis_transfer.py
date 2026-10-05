"""Leave-one-capture-out wrist-axis correction; never fit on the test capture."""
import argparse
import json
from pathlib import Path

import numpy as np
from compare_capture_arm_bias import read, stats
from experiment_arm_triangulation import CAPTURES
from pose_app.body_observations import raw_side


def axis_target(wrist, elbow, manual):
    axis = np.asarray(wrist, float) - np.asarray(elbow, float)
    length = float(np.linalg.norm(axis))
    if length < 1 or not np.isfinite(axis).all():
        return None
    projection = float((np.asarray(manual) - wrist) @ (axis / length))
    return projection, projection / length


def correct(wrist, elbow, coefficient, mode):
    axis = np.asarray(wrist, float) - np.asarray(elbow, float)
    length = float(np.linalg.norm(axis))
    if length < 1 or not np.isfinite(axis).all():
        return np.asarray(wrist, float).copy()
    return np.asarray(wrist) + coefficient * (axis / length if mode == "pixels" else axis)


def load_rows(source, model):
    p = read(source / "protocol.json")
    views = {c: raw_side(Path(p["inputs"][model][c + "_raw"]), c) for c in ("left", "right")}
    rows = []
    for r in read(source / "points.json"):
        if r["methods"][model]["state"] != "compared":
            continue
        joints = views[r["camera"]][r["frame_index"]]
        elbow, wrist = (7, 9) if r["hand"] == "left" else (8, 10)
        manual = np.asarray(r["manual"]["raw_px"])
        target = axis_target(joints[wrist, :2], joints[elbow, :2], manual)
        rows.append({"frame": r["frame_index"], "camera": r["camera"], "hand": r["hand"],
                     "wrist": joints[wrist, :2], "elbow": joints[elbow, :2], "manual": manual,
                     "target": target})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError("refuse_existing_output")
    args.output.mkdir(parents=True)
    protocol = {
        "inputs": {k: str(v / "protocol.json") for k, v in CAPTURES.items()},
        "code": "realtime_app/experiment_wrist_axis_transfer.py",
        "variable": "One scalar along own predicted elbow-to-wrist axis, pixels or forearm fraction",
        "training": "Median signed 1D projections across every available training manual slot; robust L1 location",
        "test": "Opposite capture only, both directions; no hyperparameter or grouping fitted on test labels",
        "controls": "Original raw detections and original anomalous labels retained; no stereo swaps applied",
        "success": "Report all errors and camera-hand groups including failures; do not select method by test means",
        "stop": "No valid training axis abort; degenerate test axes use unchanged point and explicit flag",
        "limits": "Small repeated samples, manual operational wrist definition, not independent anatomical truth; group tables descriptive only",
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    summary = {}
    records = []
    for model in ("PMPose", "Sapiens2"):
        captures = {k: load_rows(v, model) for k, v in CAPTURES.items()}
        for train, test in (("old448", "new360"), ("new360", "old448")):
            training = [r["target"] for r in captures[train] if r["target"] is not None]
            if not training:
                raise ValueError("no_training_axis")
            for mode, index in (("pixels", 0), ("fraction", 1)):
                coefficient = float(np.median([r[index] for r in training]))
                key = f"{model}_{train}_to_{test}_{mode}"
                rows = []
                for r in captures[test]:
                    candidate = correct(r["wrist"], r["elbow"], coefficient, mode)
                    rows.append({"experiment": key, "frame": r["frame"], "camera": r["camera"],
                        "hand": r["hand"], "degenerate_axis": r["target"] is None,
                        "original_px": float(np.linalg.norm(r["wrist"] - r["manual"])),
                        "candidate_px": float(np.linalg.norm(candidate - r["manual"])),
                        "candidate_raw": candidate.tolist(),
                        "out_of_bounds": not (0 <= candidate[0] < 1920 and 0 <= candidate[1] < 1080)})
                groups = {"all": rows}
                groups.update({f"{c}_{h}": [r for r in rows if r["camera"] == c and r["hand"] == h]
                               for c in ("left", "right") for h in ("left", "right")})
                summary[key] = {"coefficient": coefficient, "training_slots": len(training),
                    "groups": {g: {"original": stats([r["original_px"] for r in rr]),
                        "candidate": stats([r["candidate_px"] for r in rr]),
                        "paired_change": stats([r["candidate_px"] - r["original_px"] for r in rr]),
                        "improved": sum(r["candidate_px"] < r["original_px"] for r in rr),
                        "out_of_bounds": sum(r["out_of_bounds"] for r in rr)} for g, rr in groups.items()}}
                records.extend(rows)
    for name, obj in (("summary", summary), ("records", records)):
        (args.output / (name + ".json")).write_text(json.dumps(obj, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
