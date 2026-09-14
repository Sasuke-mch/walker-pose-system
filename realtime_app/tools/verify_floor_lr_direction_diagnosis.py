#!/usr/bin/env python3
"""Independently re-verify the prior floor-mask left/right direction diagnosis.

The prior stage reported that the strict local-ground tool's left/right
consistency gate reads a reverse disparity field whose search direction cannot
represent this calibration's correspondence, and that correcting that direction
changes the accepted set.  This tool re-checks those claims directly from the
archived artifacts and from a fresh matcher run, instead of trusting the earlier
summary files.

It performs no new experiment and writes no ground state: it only reports
whether each archived claim holds.

Checks
------
1. frozen baseline: row count, states, rejection reasons
2. control arm: the recorded control values are compared against the *frozen
   baseline jsonl itself*, not against the values the control tool recorded
3. diagnostic: per-frame exact reproduction flags, funnel totals recomputed from
   the per-frame stages, single-condition counterfactual totals
4. gate controls: frozen-direction pass rate versus its own nulls, and versus the
   direction-corrected field, recomputed from the per-frame records
5. matcher direction: a synthetic shift control plus a cross-implementation check
   that the corrected field equals a mirrored-image computation on real data
6. one frame recomputed through the frozen tool's own candidate function
7. cross-artifact check: the diagnostic's corrected-direction probe count equals
   the independent control experiment's candidate count
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


TOOLS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_ROOT.parents[1]
REALTIME_ROOT = PROJECT_ROOT / "realtime_app"
for candidate in (str(REALTIME_ROOT), str(TOOLS_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from pose_app.calibration import StereoCalibration  # noqa: E402
import observe_local_ground_semantic_stereo as strict  # noqa: E402
import diagnose_floor_mask_stereo_correspondence as diagnostic  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def close(first: Any, second: Any, tolerance: float = 1e-9) -> bool:
    if first is None or second is None:
        return first == second
    return abs(float(first) - float(second)) <= tolerance * max(1.0, abs(float(first)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prior-experiment-dir", type=Path, required=True,
                        help="G20260911_floor_stereo_correspondence_diagnosis_v1")
    parser.add_argument("--left-dir", type=Path, required=True)
    parser.add_argument("--right-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--semantic-left-dir", type=Path, required=True)
    parser.add_argument("--semantic-right-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON report path (inside the new experiment dir)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prior = args.prior_experiment_dir
    frozen_dir = prior.parent / "G20260911_floor_semantic_backend_comparison_v1" / "stereo_manual_floor_masks_12pairs_v1"
    diagnostic_dir = prior / "readonly_diagnostic_12pairs_v4_final"
    experiment_dir = prior / "controlled_lr_direction_12pairs_v2_final"

    report: dict[str, Any] = {"checks": {}, "problems": []}

    def record(name: str, passed: bool, detail: dict[str, Any]) -> None:
        report["checks"][name] = {"passed": bool(passed), **detail}
        if not passed:
            report["problems"].append(name)

    # 1. frozen baseline ------------------------------------------------------
    frozen_rows = {int(row["pair_id"]): row for row in read_jsonl(frozen_dir / "local_ground_state.jsonl")}
    frozen_direct = sorted(pid for pid, row in frozen_rows.items() if row["observation_state"] == "direct")
    reason_counts: dict[str, int] = {}
    for row in frozen_rows.values():
        for reason in row["reason"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    record("frozen_baseline", len(frozen_rows) == 12 and frozen_direct == [240, 320], {
        "rows": len(frozen_rows),
        "direct_pair_ids": frozen_direct,
        "reason_counts": reason_counts,
    })

    # 2. control arm compared against the frozen baseline itself -------------
    control_items = json.loads((experiment_dir / "control_arm_reproduction.json").read_text(encoding="utf-8"))["items"]
    mismatches: list[dict[str, Any]] = []
    for item in control_items:
        pid = int(item["pair_id"])
        baseline_quality = frozen_rows[pid]["quality"]
        for key, payload in item["fields"].items():
            if not close(payload["control_arm"], baseline_quality[key]):
                mismatches.append({"pair_id": pid, "field": key,
                                   "baseline": baseline_quality[key], "control_arm": payload["control_arm"]})
        if item["control_arm_state"] != frozen_rows[pid]["observation_state"]:
            mismatches.append({"pair_id": pid, "field": "state",
                               "baseline": frozen_rows[pid]["observation_state"],
                               "control_arm": item["control_arm_state"]})
        if sorted(item["control_arm_reasons"]) != sorted(frozen_rows[pid]["reason"]):
            mismatches.append({"pair_id": pid, "field": "reasons",
                               "baseline": frozen_rows[pid]["reason"],
                               "control_arm": item["control_arm_reasons"]})
    record("control_arm_matches_frozen_baseline", not mismatches and len(control_items) == 12, {
        "items": len(control_items), "mismatches": mismatches,
    })

    # 3. diagnostic internal consistency ------------------------------------
    diagnostic_rows = {int(row["pair_id"]): row for row in read_jsonl(diagnostic_dir / "correspondence_diagnostic.jsonl")}
    summary = json.loads((diagnostic_dir / "summary.json").read_text(encoding="utf-8"))
    exact_failures = [pid for pid, row in diagnostic_rows.items()
                      if not row["frozen_baseline_reproduction"]["verification"]["exact"]]
    reason_failures = [pid for pid, row in diagnostic_rows.items()
                       if not row["frozen_gate_attribution"]["reason_matches_frozen_baseline"]]
    recomputed_totals: dict[str, int] = {}
    recomputed_drop_one: dict[str, int] = {}
    for row in diagnostic_rows.values():
        for entry in row["funnel"]["stages"]:
            recomputed_totals[entry["stage"]] = recomputed_totals.get(entry["stage"], 0) + entry["cumulative_pixels"]
        for key, value in row["funnel"]["drop_one_condition_candidate_counts"].items():
            recomputed_drop_one[key] = recomputed_drop_one.get(key, 0) + value
    totals_match = recomputed_totals == summary["funnel_stage_totals_across_frames"]
    drop_one_match = recomputed_drop_one == summary["drop_one_condition_candidate_totals"]
    record("diagnostic_reproduction_and_totals",
           not exact_failures and not reason_failures and totals_match and drop_one_match,
           {"exact_failures": exact_failures, "reason_failures": reason_failures,
            "recomputed_totals": recomputed_totals, "totals_match_summary": totals_match,
            "drop_one_match_summary": drop_one_match})

    # 4. gate controls ------------------------------------------------------
    frozen_below_null = 0
    corrected_above_frozen = 0
    per_frame: list[dict[str, Any]] = []
    for pid, row in sorted(diagnostic_rows.items()):
        gate = row["consistency_gate_controls"]
        observed = gate["observed_pass_rate_frozen_reverse_field_at_partner"]
        nulls = [gate["null_pass_rate_frozen_reverse_field_partner_shifted_17px"],
                 gate["null_pass_rate_frozen_reverse_field_other_row_5px"]]
        corrected = gate["observed_pass_rate_direction_corrected_field_at_partner"]
        if observed is not None and observed <= max(value for value in nulls if value is not None):
            frozen_below_null += 1
        if observed is not None and corrected is not None and corrected > observed:
            corrected_above_frozen += 1
        per_frame.append({"pair_id": pid, "frozen": observed, "nulls": nulls, "corrected": corrected})
    record("gate_controls", frozen_below_null >= 9 and corrected_above_frozen == 12, {
        "frames_where_frozen_at_or_below_its_nulls": frozen_below_null,
        "frames_where_corrected_exceeds_frozen": corrected_above_frozen,
        "per_frame": per_frame,
    })

    # 5. matcher direction, synthetic plus a mirrored cross-implementation ----
    rng = np.random.default_rng(11)
    left = rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)
    left = cv2.GaussianBlur(left, (5, 5), 0)
    shift = 120
    right = np.zeros_like(left)
    right[:, : 960 - shift] = left[:, shift:]
    right[:, 960 - shift:] = left[:, :shift]
    forward_synthetic = strict.dense_disparity(left, right, 160)
    reverse_synthetic = strict.dense_disparity(right, left, 160)
    corrected_synthetic = diagnostic.dense_disparity_right_direction(right, left, 160)
    band = slice(300, 400)
    synthetic = {
        "forward_median": float(np.median(forward_synthetic[band, 300:600])),
        "frozen_reverse_median": float(np.median(reverse_synthetic[band, 180:480])),
        "frozen_reverse_valid_fraction": float((reverse_synthetic[band, 180:480] > 1.0).mean()),
        "corrected_reverse_median": float(np.median(corrected_synthetic[band, 180:480])),
        "corrected_reverse_valid_fraction": float((corrected_synthetic[band, 180:480] > 1.0).mean()),
    }

    pair_name = "pair_0240.png"
    left_upright = cv2.imread(str(args.left_dir / pair_name), cv2.IMREAD_COLOR)
    right_upright = cv2.imread(str(args.right_dir / pair_name), cv2.IMREAD_COLOR)
    shape_upright = left_upright.shape[:2]
    left_mask_upright = strict.load_mask(args.semantic_left_dir / pair_name, shape_upright)
    right_mask_upright = strict.load_mask(args.semantic_right_dir / pair_name, shape_upright)
    calibration = StereoCalibration.load(args.calibration).for_runtime_sizes((960, 540), (960, 540))
    left_raw, right_raw = strict.inverse_upright(left_upright, right_upright)
    raw_size = (left_raw.shape[1], left_raw.shape[0])
    left_seed = strict.mask_seed_upright(left_mask_upright)
    right_seed = strict.mask_seed_upright(right_mask_upright)
    left_seed_raw = strict.upright_point_to_raw(left_seed, "left", (shape_upright[1], shape_upright[0])) * np.asarray(
        (960 / raw_size[0], 540 / raw_size[1]))
    right_seed_raw = strict.upright_point_to_raw(right_seed, "right", (shape_upright[1], shape_upright[0])) * np.asarray(
        (960 / raw_size[0], 540 / raw_size[1]))
    rectification = strict.make_mask_directed_rectification(calibration, left_seed_raw, right_seed_raw, (960, 540), 330.0)
    left_local = cv2.remap(cv2.resize(left_raw, (960, 540), interpolation=cv2.INTER_AREA),
                           rectification.left_map_x, rectification.left_map_y, cv2.INTER_LINEAR)
    right_local = cv2.remap(cv2.resize(right_raw, (960, 540), interpolation=cv2.INTER_AREA),
                            rectification.right_map_x, rectification.right_map_y, cv2.INTER_LINEAR)
    left_mask_local = cv2.remap(cv2.resize(strict.rotate_mask_to_raw(left_mask_upright, "left"), (960, 540), interpolation=cv2.INTER_NEAREST),
                                rectification.left_map_x, rectification.left_map_y, cv2.INTER_NEAREST)
    right_mask_local = cv2.remap(cv2.resize(strict.rotate_mask_to_raw(right_mask_upright, "right"), (960, 540), interpolation=cv2.INTER_NEAREST),
                                 rectification.right_map_x, rectification.right_map_y, cv2.INTER_NEAREST)
    corrected_real = diagnostic.dense_disparity_right_direction(right_local, left_local, 160)
    mirrored_real = cv2.flip(strict.dense_disparity(cv2.flip(right_local, 1), cv2.flip(left_local, 1), 160), 1)
    both_valid = (corrected_real > 1.0) & (mirrored_real > 1.0)
    agreement = float(np.median(np.abs(corrected_real[both_valid] - mirrored_real[both_valid]))) if both_valid.any() else None
    record("matcher_direction", synthetic["forward_median"] > 100 and synthetic["corrected_reverse_median"] > 100
           and synthetic["frozen_reverse_valid_fraction"] < 0.4 and agreement is not None and agreement <= 1.0,
           {**synthetic, "mirrored_cross_check_pixels": int(both_valid.sum()),
            "mirrored_cross_check_median_abs_diff_px": agreement})

    # 6. frozen candidate function on real data ------------------------------
    _, baseline_left_pixels, _, _ = strict.reconstruct_candidates(
        left_local, right_local, left_mask_local, right_mask_local, rectification, 160, 1.5
    )
    record("frozen_candidate_function", len(baseline_left_pixels) == frozen_rows[240]["quality"]["candidate_points"], {
        "recomputed_candidates": int(len(baseline_left_pixels)),
        "frozen_candidates": frozen_rows[240]["quality"]["candidate_points"],
    })

    # 7. cross-artifact consistency between the two independent runs ---------
    experiment_rows = {int(row["pair_id"]): row for row in read_jsonl(experiment_dir / "local_ground_state.jsonl")}
    cross: list[dict[str, Any]] = []
    for pid in sorted(diagnostic_rows):
        probe = diagnostic_rows[pid]["consistency_gate_controls"]["direction_corrected_candidate_pixels_probe"]
        actual = experiment_rows[pid]["quality"]["candidate_points"]
        cross.append({"pair_id": pid, "diagnostic_probe": probe, "experiment_arm": actual, "match": probe == actual})
    record("cross_artifact_candidate_counts", all(item["match"] for item in cross), {"per_frame": cross})

    report["conclusion"] = (
        "All archived claims of the prior stage re-verified from the artifacts"
        if not report["problems"] else
        "Re-verification found problems: " + ", ".join(report["problems"])
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "problems": report["problems"],
                      "conclusion": report["conclusion"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
