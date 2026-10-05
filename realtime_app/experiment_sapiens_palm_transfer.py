"""Same-inference palm-direction wrist correction with capture-held-out coefficient."""
import argparse
import json
import runpy
from pathlib import Path

import numpy as np
from compare_capture_arm_bias import RECORDS, ROOT, read, stats
from experiment_arm_triangulation import CAPTURES
from pose_app.body_observations import raw_side

SOURCES = {"old448": RECORDS / "G20261002_sapiens2_body_mainline448_v2",
           "new360": RECORDS / "G20261003_repositioned360_cold_smplh_v2"}


def to_raw(points, camera):
    p = np.asarray(points, float)
    return np.column_stack((1919 - p[:, 1], p[:, 0])) if camera == "left" else np.column_stack((p[:, 1], 1079 - p[:, 0]))


def coefficient_target(wrist, palm, manual):
    direction = palm - wrist
    length2 = float(direction @ direction)
    return None if length2 < 1 or not np.isfinite(length2) else float((manual - wrist) @ direction / length2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError("refuse_existing_output")
    args.output.mkdir(parents=True)
    mapping_path = ROOT / "third_party/sapiens2/repo/sapiens/pose/configs/_base_/keypoints308.py"
    mapping = runpy.run_path(str(mapping_path))["dataset_info"]["keypoint_info"]
    names = {v["name"]: k for k, v in mapping.items()}
    ids = {hand: [names[f"{hand}_{finger}_third_joint"] for finger in ("forefinger", "middle_finger", "ring_finger", "pinky_finger")]
           for hand in ("left", "right")}
    wrists = {h: names[h + "_wrist"] for h in ("left", "right")}
    protocol = {"sources": {k: str(v) for k, v in SOURCES.items()}, "labels": {k: str(v / "points.json") for k, v in CAPTURES.items()},
        "mapping": str(mapping_path), "palm_indices": ids, "wrist_indices": wrists,
        "code": "realtime_app/experiment_sapiens_palm_transfer.py",
        "variable": "Direction from original wrist to mean of four proximal finger landmarks, one scalar coefficient",
        "training": "Median signed projected coefficient on all original other-capture labels; no test tuning or additional inference",
        "controls": "Raw full308 first instance mapped by file_name sequence index and raw-fisheye inverse rotation; exact compare COCO source wrists",
        "success": "Report both transfer directions, every original slot, camera/hand groups, no silent exclusions",
        "stop": "Missing frame or mapping mismatch abort; degenerate palm keeps original with unavailable flag",
        "limits": "Proximal landmarks are model-derived anatomical labels; coefficient is operational pixel alignment, not 3D truth or semantic repair"}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    capture_rows = {}
    for capture, source in SOURCES.items():
        frames = {}
        p = read(CAPTURES[capture] / "protocol.json")
        for camera in ("left", "right"):
            coco = raw_side(Path(p["inputs"]["Sapiens2"][camera + "_raw"]), camera)
            with (source / f"{camera}_sapiens_raw.jsonl").open(encoding="utf-8-sig") as stream:
                for line in stream:
                    row = json.loads(line)
                    fid = int(Path(row["file_name"]).stem.rsplit("_", 1)[-1])
                    if (camera, fid) in frames:
                        raise ValueError("duplicate_frame")
                    instance = row["instances"][0]
                    pixels = to_raw(instance["keypoints308"], camera)
                    if pixels.shape != (308, 2):
                        raise ValueError("mapping_shape")
                    np.testing.assert_allclose(pixels[[wrists["left"], wrists["right"]]], coco[fid][[9, 10], :2], atol=1e-5)
                    frames[camera, fid] = pixels
            if sorted(fid for c, fid in frames if c == camera) != sorted(coco):
                raise ValueError("frame_mismatch")
        slots = []
        for label in read(CAPTURES[capture] / "points.json"):
            if label["methods"]["Sapiens2"]["state"] != "compared":
                continue
            c, f, h = label["camera"], label["frame_index"], label["hand"]
            pixel = frames[c, f]
            wrist = pixel[wrists[h]]
            palm = pixel[ids[h]].mean(axis=0)
            manual = np.asarray(label["manual"]["raw_px"])
            slots.append({"camera": c, "frame": f, "hand": h, "wrist": wrist, "palm": palm, "manual": manual,
                          "target": coefficient_target(wrist, palm, manual)})
        capture_rows[capture] = slots
    summary, records = {}, []
    for train, test in (("old448", "new360"), ("new360", "old448")):
        coeffs = [r["target"] for r in capture_rows[train] if r["target"] is not None]
        if not coeffs:
            raise ValueError("no_training_support")
        coefficient = float(np.median(coeffs))
        key = f"{train}_to_{test}"
        rows = []
        for r in capture_rows[test]:
            candidate = r["wrist"] if r["target"] is None else r["wrist"] + coefficient * (r["palm"] - r["wrist"])
            rows.append({"experiment": key, "frame": r["frame"], "camera": r["camera"], "hand": r["hand"],
                         "before_px": float(np.linalg.norm(r["wrist"] - r["manual"])),
                         "after_px": float(np.linalg.norm(candidate - r["manual"])),
                         "candidate_raw_px": candidate.tolist(), "degenerate": r["target"] is None,
                         "out_of_bounds": not (0 <= candidate[0] < 1920 and 0 <= candidate[1] < 1080)})
        groups = {"all": rows, **{f"{c}_{h}": [r for r in rows if r["camera"] == c and r["hand"] == h]
                                  for c in ("left", "right") for h in ("left", "right")}}
        summary[key] = {"coefficient": coefficient, "training_slots": len(coeffs),
                       "groups": {g: {"before": stats([r["before_px"] for r in group]), "after": stats([r["after_px"] for r in group]),
                                      "improved": sum(r["after_px"] < r["before_px"] for r in group)} for g, group in groups.items()}}
        records.extend(rows)
    for name, obj in (("summary", summary), ("records", records)):
        (args.output / (name + ".json")).write_text(json.dumps(obj, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
