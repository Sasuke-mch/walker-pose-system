"""Independent full-sequence audit of bounded local-arm research candidates."""
import argparse
import json
from pathlib import Path

import numpy as np
from compare_capture_arm_bias import RECORDS, read, stats
from experiment_forearm_residual_parts import FITS
from experiment_local_arm_pose import BODY_INDICES


def second_difference(points):
    return np.linalg.norm(np.diff(points, n=2, axis=0), axis=-1) * 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", type=Path, required=True)
    args = ap.parse_args()
    destination = args.candidate / "independent_audit.json"
    if destination.exists():
        raise ValueError("refuse_existing_audit")
    sets_path = RECORDS / "G20260930_smplh_contact_set_fix_v1_run3/contact_vertex_sets.json"
    sets = read(sets_path)["sets"]
    report = {"surface_sets": str(sets_path), "limits": "Surface displacement is not contact validation; second differences in mm/frame^2, no assumed fps", "cases": {}}
    for name in read(args.candidate / "summary.json"):
        with np.load(FITS[name]) as source, np.load(args.candidate / (name + ".npz")) as candidate:
            np.testing.assert_array_equal(source["pair_id"], candidate["pair_id"])
            np.testing.assert_array_equal(source["faces"], candidate["faces"])
            unchanged = {key: bool(np.array_equal(source[key], candidate[key])) for key in
                         ("betas", "global_orient", "transl", "left_hand_pose", "right_hand_pose")}
            a = source["body_pose"].reshape(-1, 21, 3)
            b = candidate["body_pose"].reshape(-1, 21, 3)
            frozen = [i for i in range(21) if i not in BODY_INDICES]
            unchanged["other_body_pose"] = bool(np.array_equal(a[:, frozen], b[:, frozen]))
            if not all(unchanged.values()):
                raise ValueError("frozen_parameter_changed")
            delta = b[:, BODY_INDICES] - a[:, BODY_INDICES]
            coco0, coco1 = source["predicted_coco"], candidate["predicted_coco"]
            residual0 = np.linalg.norm(coco0 * 1000 - source["raw_triangulated_points"], axis=-1)
            residual1 = np.linalg.norm(coco1 * 1000 - source["raw_triangulated_points"], axis=-1)
            row = {"unchanged": unchanged, "max_component_delta_rad": float(np.abs(delta).max()),
                   "near_box_limit_fraction": float(np.mean(np.abs(delta) >= .1499)),
                   "joints": {}, "surfaces": {}}
            for label, joints in (("arms", [7, 8, 9, 10]), ("legs", [13, 14, 15, 16])):
                mask = source["body_accepted"][:, joints]
                change = (residual1[:, joints] - residual0[:, joints])[mask]
                row["joints"][label] = {
                    "residual_change_mm": stats(change), "worse_over_1mm": int(np.sum(change > 1)),
                    "second_difference_before": stats(second_difference(coco0[:, joints])),
                    "second_difference_after": stats(second_difference(coco1[:, joints]))}
            for side in ("left", "right"):
                for part in ("sole", "palm"):
                    indices = sorted({index for group in sets[f"{side}_{part}_surface_candidate"].values()
                                      for index in group})
                    movement = np.linalg.norm(candidate["vertices"][:, indices] - source["vertices"][:, indices], axis=-1) * 1000
                    row["surfaces"][side + "_" + part + "_movement_mm"] = stats(movement)
            row["decision"] = "research_candidate_not_accepted"
            report["cases"][name] = row
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
