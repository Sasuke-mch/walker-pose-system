"""Recompute original first-instance strict stereo observations; no gate changes."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "realtime_app"))
from pose_app.calibration import StereoCalibration
from pose_app.triangulation import triangulate_matches
from walker_tools.stereo.evaluate_offline_stereo_predictions import _records, _result


def main():
    target = Path(__file__).resolve().parent
    source = ROOT / "research_records/engineering_validation/V20260908_people0_1_2_pmpose_c3_chain/people_1/c3_predictions"
    predictions = {
        side: _records(source / side / "pmpose/raw_predictions.json", "pmpose", "score")
        for side in ("left", "right")
    }
    calibration_path = ROOT / "realtime_app/calibration/results/stereo_fisheye.json"
    calibration = StereoCalibration.load(calibration_path).for_runtime_sizes((1920, 1080), (1920, 1080))
    computed = []
    assert len(predictions["left"]) == len(predictions["right"]) == 448
    for index, (left, right) in enumerate(zip(predictions["left"], predictions["right"])):
        assert left["id"] == right["id"] == index
        assert left["name"] == right["name"] == f"pair_{index:04d}.png"
        results = []
        for frame, rotation in ((left, "ccw90"), (right, "cw90")):
            # Preserve the first detector instance only, matching the fitter.
            frame = dict(frame, persons=frame["persons"][:1])
            assert len(frame["persons"]) == 1 and frame["persons"][0].person_id == 0
            results.append(_result(frame, "pmpose", (1920, 1080), rotation, index / 30.0))
        people = triangulate_matches(results[0].persons, results[1].persons, calibration,
                                     .25, .05, 10., max_matches=1)
        assert all((p.left_person_id, p.right_person_id) == (0, 0) for p in people)
        computed.append({"index": index, "original_pair_id": index,
                         "pair_outcome": "associated" if people else "association_rejected",
                         "persons_3d": [person.to_dict() for person in people]})
    out = target / "first_instance_strict.jsonl"
    out.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in computed), encoding="utf-8")
    old_path = target.parent / "first_frame_sapiens_20261015/visualization/first_instance_strict.jsonl"
    previous = [json.loads(line) for line in old_path.read_text(encoding="utf-8").splitlines()]
    assert len(previous) == len(computed)
    changed_masks, changed_reasons, changed_presence = [], [], []
    max_difference = 0.
    for old, new in zip(previous, computed):
        i = old["index"]
        if len(old["persons_3d"]) != len(new["persons_3d"]):
            changed_presence.append(i)
            continue
        for a, b in zip(old["persons_3d"], new["persons_3d"]):
            for x, y in zip(a["keypoints_3d"], b["keypoints_3d"]):
                if x["valid"] != y["valid"]:
                    changed_masks.append([i, x["index"]])
                if x["reason"] != y["reason"]:
                    changed_reasons.append([i, x["index"]])
                if (x["xyz"] is None) != (y["xyz"] is None):
                    changed_presence.append([i, x["index"]])
                elif x["xyz"] is not None:
                    max_difference = max(max_difference, float(np.max(np.abs(np.asarray(x["xyz"]) - y["xyz"]))))
    summary = {
        "sources": {side: str(source / side / "pmpose/raw_predictions.json") for side in predictions},
        "calibration": str(calibration_path), "previous": str(old_path), "output": str(out),
        "implementation": ["walker_tools.stereo.evaluate_offline_stereo_predictions._records/_result", "pose_app.triangulation.triangulate_matches/TriangulatedPerson.to_dict"],
        "parameters": {"keypoint_threshold": .25, "max_association_cost": .05, "max_reprojection_error_px": 10., "max_matches": 1, "left_instance": 0, "right_instance": 0, "score_source": "raw point third channel"},
        "frames": len(computed), "pair_outcomes": dict(Counter(r["pair_outcome"] for r in computed)),
        "accepted_points": sum(p["valid_keypoints"] for r in computed for p in r["persons_3d"]),
        "comparison": {"whole_records_equal": previous == computed, "changed_valid": changed_masks,
                       "changed_reasons": changed_reasons, "changed_presence": changed_presence,
                       "max_abs_xyz_difference_mm": max_difference},
        "scene_replayed": False,
    }
    (target / "strict_recompute_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
