#!/usr/bin/env python3
"""Estimate the frozen shared skeleton bone lengths from six quality clips.

Input: the ground-transformed 3D points of the six frozen clips (the offline
prefit used for cold-start / comparison only) plus the clip selection. Output:
frozen_skeleton_shape.json with per-clip lengths, frozen medians, per-bone
range and CV, and the frozen-bone gate verdict.

Gate (frozen before this run):
  - every bone length finite and positive;
  - unilateral limb bone CV <= 0.10;
  - left/right asymmetry |L-R| / mean(L,R) <= 0.15 for the paired bones.
If the gate fails this script exits non-zero and writes no usable shape file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from pose_app.smplx_fitting import read_jsonl  # noqa: E402
from pose_app.skeleton_factor_graph import DEFAULT_BONE_NAMES, estimate_bone_lengths_mm  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefit-jsonl", type=Path, required=True)
    parser.add_argument("--clip-selection-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    prefit = read_jsonl(args.prefit_jsonl)
    selection = json.loads(args.clip_selection_json.read_text(encoding="utf-8"))
    clips = selection["clips"] if isinstance(selection, dict) and "clips" in selection else selection
    source_windows = []
    per_clip_lengths = []
    for clip in clips:
        start, end = (int(clip["inclusive_frame_window"][0]), int(clip["inclusive_frame_window"][1])
                      if "inclusive_frame_window" in clip else (int(clip["start"]), int(clip["end"])))
        source_windows.append([start, end])
        points = np.asarray([
            [row["points_ground_mm"][str(joint)] for joint in range(5, 17)]
            for row in prefit[start:end + 1]
        ], dtype=np.float64)
        if not np.all(np.isfinite(points)):
            raise ValueError(f"clip {start}-{end} contains non-finite points")
        per_clip_lengths.append(estimate_bone_lengths_mm(points))

    lengths = np.asarray(per_clip_lengths, dtype=np.float64)  # (6, 11) mm
    frozen_median = np.median(lengths, axis=0)
    per_bone_range = np.ptp(lengths, axis=0)
    per_bone_cv = np.std(lengths, axis=0, ddof=0) / np.clip(frozen_median, 1e-9, None)

    paired = {
        "upper_arm": ("upper_arm_left", "upper_arm_right"),
        "forearm": ("forearm_left", "forearm_right"),
        "thigh": ("thigh_left", "thigh_right"),
        "shank": ("shank_left", "shank_right"),
    }
    name_to_index = {name: index for index, name in enumerate(DEFAULT_BONE_NAMES)}
    asymmetry = {}
    for pair_name, (left_name, right_name) in paired.items():
        li, ri = name_to_index[left_name], name_to_index[right_name]
        lr_mean = 0.5 * (frozen_median[li] + frozen_median[ri])
        asymmetry[pair_name] = float(abs(frozen_median[li] - frozen_median[ri]) / max(lr_mean, 1e-9))

    unilateral = [name for name in DEFAULT_BONE_NAMES if "left" in name or "right" in name]
    cv_gate = {name: float(per_bone_cv[name_to_index[name]]) for name in unilateral}
    finite_positive = bool(np.all(np.isfinite(frozen_median)) and np.all(frozen_median > 0.0))
    cv_ok = all(value <= 0.10 for value in cv_gate.values())
    asym_ok = all(value <= 0.15 for value in asymmetry.values())
    gate = {
        "all_bones_finite_positive": finite_positive,
        "unilateral_limb_cv_at_most_0_10": cv_ok,
        "left_right_asymmetry_at_most_0_15": asym_ok,
        "passed": bool(finite_positive and cv_ok and asym_ok),
    }

    record = {
        "schema_version": "frozen_skeleton_shape_v1",
        "bone_names": DEFAULT_BONE_NAMES,
        "per_clip_lengths_mm": [row.round(3).tolist() for row in lengths],
        "frozen_median_lengths_mm": frozen_median.round(3).tolist(),
        "per_bone_range_mm": per_bone_range.round(3).tolist(),
        "per_bone_cv": per_bone_cv.round(4).tolist(),
        "left_right_asymmetry": {name: round(value, 4) for name, value in asymmetry.items()},
        "source_windows": source_windows,
        "estimation_rule": (
            "per-clip bone lengths from ground-transformed 3D points of the six frozen clips; "
            "frozen value = coordinate-wise median across clips; runtime never optimises bones."
        ),
        "boundary": (
            "Bone lengths are internal geometry for the kinematic skeleton, derived from the "
            "offline 3D stream used only for cold start and comparison. They are not an "
            "independent anthropometric ground truth."
        ),
        "gate": gate,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    if not gate["passed"]:
        args.output_json.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False, indent=2))
        print("FROZEN BONE GATE FAILED", flush=True)
        return 1
    args.output_json.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print("FROZEN BONE GATE PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
