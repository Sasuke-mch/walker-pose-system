"""Audit paired prior ablations without requiring or fabricating wrist truth."""
import argparse
import json
from pathlib import Path

import numpy as np
from audit_local_arm_pose import second_difference
from compare_capture_arm_bias import RECORDS, read, stats
from run_smplh_ablations import modified_options


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, required=True)
    args = ap.parse_args()
    destination = args.experiment / "evaluation.json"
    if destination.exists():
        raise ValueError("refuse_existing_evaluation")
    plan = read(args.experiment / "plan.json")
    sets_path = RECORDS / "G20260930_smplh_contact_set_fix_v1_run3/contact_vertex_sets.json"
    sets = read(sets_path)["sets"]
    from walker_tools.body.fit_smplh_wilor_sequence import build_parser

    report = {"limits": "Same observations not independent truth; surface displacement not contact; motion units mm/frame^2; no fps assumed",
              "surface_sets": str(sets_path), "cases": {}}
    for case in plan["cases"]:
        root = args.experiment / "runs" / case["id"]
        if not (root / "execution.json").exists():
            report["cases"][case["id"]] = {"state": "incomplete_or_failed"}
            continue
        baseline = Path(plan["control_results"][case["detector"]])
        command = read(Path(plan["source_commands"][case["detector"]]))
        options = build_parser().parse_args(modified_options(command, {}, "baseline-placeholder"))
        base_options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(options).items()}
        candidate_options = read(root / "resolved_options.json")
        differences = {k: [base_options[k], v] for k, v in candidate_options.items() if base_options[k] != v}
        if set(differences) != {"output_dir", "vposer_prior_weight"}:
            raise ValueError("unexpected_option_difference")
        with np.load(baseline.parent / "cold_initialization.npz") as a, np.load(root / "fit/cold_initialization.npz") as b:
            parity = {k: bool(np.array_equal(a[k], b[k], equal_nan=True)) for k in a.files if a[k].dtype.kind in "fibu"}
        if not all(parity.values()):
            raise ValueError("initialization_changed")
        with np.load(baseline) as a, np.load(root / "fit/result.npz") as b:
            for key in ("pair_id", "body_accepted", "body_quality", "raw_triangulated_points"):
                np.testing.assert_array_equal(a[key], b[key])
            mask = a["body_accepted"]
            weights = a["body_quality"][mask]
            rows = {}
            for name, data in (("baseline", a), ("candidate", b)):
                errors = np.linalg.norm(data["predicted_coco"] * 1000 - a["raw_triangulated_points"], axis=-1)
                rows[name] = {"weighted_body_rms_mm": float(np.sqrt(np.sum(weights * errors[mask] ** 2) / weights.sum())),
                              "beta_norm": float(np.linalg.norm(data["betas"])),
                              "latent_mean_square": float(np.mean(data["vposer_latent"] ** 2)),
                              "groups": {g: {"residual_mm": stats(errors[:, js][mask[:, js]]),
                                             "second_difference": stats(second_difference(data["predicted_coco"][:, js]))}
                                         for g, js in (("arms", [7, 8, 9, 10]), ("legs", [13, 14, 15, 16]))}}
            movement = {}
            for side in ("left", "right"):
                for part in ("sole", "palm"):
                    ids = sorted({i for group in sets[f"{side}_{part}_surface_candidate"].values() for i in group})
                    movement[f"{side}_{part}"] = stats(np.linalg.norm(b["vertices"][:, ids] - a["vertices"][:, ids], axis=-1) * 1000)
            report["cases"][case["id"]] = {"state": "completed_not_promoted", "options_changed": differences,
                                           "initialization_parity": parity, "metrics": rows, "surface_movement_mm": movement,
                                           "execution": read(root / "execution.json")}
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
