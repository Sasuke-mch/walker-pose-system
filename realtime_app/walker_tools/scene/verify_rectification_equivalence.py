#!/usr/bin/env python3
"""Prove that the learned-stereo benchmark rectifies exactly like the frozen tool.

The learned-stereo benchmark needs the rectified local views *without* running
SGBM, so it carries its own copy of the rectification block of
``benchmark_ground_walker_reconstruction.stereo_candidates``.  A copy is only
acceptable if it is provably identical, so this read-only check re-executes the
frozen function on the same frames and compares, field by field:

* the four remap grids of the mask-directed rectification,
* the left/right virtual intrinsics, rotation and translation,
* the rectified left/right local images,
* the rectified left/right semantic masks.

The frozen function also computes an SGBM disparity as a side effect; that array
is discarded here.  This script produces no SGBM comparison number and is not a
re-run of the task-01 benchmark: it only certifies code equality of the
rectification stage.

Usage::

    python realtime_app/tools/verify_rectification_equivalence.py \
        --left-dir ... --right-dir ... --left-label-dir ... --right-label-dir ... \
        --calibration ... --frozen-baseline-dir ... --auto-floor-left-dir ... --auto-floor-right-dir ...
"""

from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
_tool_prepare_imports()

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


TOOLS_ROOT = (Path(__file__).resolve().parents[2] / "tools")
PROJECT_ROOT = TOOLS_ROOT.parents[1]
REALTIME_ROOT = PROJECT_ROOT / "realtime_app"
for candidate in (str(REALTIME_ROOT), str(TOOLS_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from pose_app.calibration import StereoCalibration  # noqa: E402
import walker_tools.scene.benchmark_ground_walker_reconstruction as frozen  # noqa: E402
import walker_tools.stereo.benchmark_learned_stereo_replacements as learned  # noqa: E402
import walker_tools.scene.observe_local_ground_semantic_stereo_lr_direction_control as corrected  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--left-dir", type=Path, required=True)
    parser.add_argument("--right-dir", type=Path, required=True)
    parser.add_argument("--left-label-dir", type=Path, required=True)
    parser.add_argument("--right-label-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--frozen-baseline-dir", type=Path, required=True)
    parser.add_argument("--auto-floor-left-dir", type=Path)
    parser.add_argument("--auto-floor-right-dir", type=Path)
    parser.add_argument("--frames", default="pair_0000.png,pair_0160.png,pair_0440.png")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def compare(name: str, first: Any, second: Any) -> dict[str, Any]:
    if isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
        if first.shape != second.shape:
            return {"field": name, "equal": False, "reason": f"shape {first.shape} vs {second.shape}"}
        difference = np.abs(first.astype(np.float64) - second.astype(np.float64))
        return {
            "field": name,
            "equal": bool(np.array_equal(first, second)),
            "max_abs_difference": float(difference.max()) if difference.size else 0.0,
            "shape": list(first.shape),
            "dtype": str(first.dtype),
        }
    if isinstance(first, (float, int, np.floating, np.integer)) and isinstance(second, (float, int, np.floating, np.integer)):
        return {"field": name, "equal": bool(float(first) == float(second)),
                "max_abs_difference": abs(float(first) - float(second))}
    return {"field": name, "equal": bool(np.array_equal(np.asarray(first), np.asarray(second)))}


def main() -> int:
    args = parse_args()
    parameters, _ = corrected.load_frozen(args.frozen_baseline_dir)
    runtime_size = (int(parameters["runtime_width"]), int(parameters["runtime_height"]))
    calibration = StereoCalibration.load(args.calibration).for_runtime_sizes(runtime_size, runtime_size)
    report: dict[str, Any] = {
        "purpose": "code-equality certificate for the rectification stage copied into the learned-stereo benchmark",
        "note": (
            "the frozen function also computes an SGBM disparity as a side effect; it is discarded here and no "
            "SGBM comparison number is produced or used"
        ),
        "frames": [],
    }
    all_equal = True
    for name in [item.strip() for item in args.frames.split(",") if item.strip()]:
        left = cv2.imread(str(args.left_dir / name), cv2.IMREAD_COLOR)
        right = cv2.imread(str(args.right_dir / name), cv2.IMREAD_COLOR)
        if left is None or right is None:
            raise RuntimeError(f"cannot read {name}")
        left_masks = frozen.load_manual_masks(args.left_label_dir / Path(name).with_suffix(".json"), left.shape[:2])
        right_masks = frozen.load_manual_masks(args.right_label_dir / Path(name).with_suffix(".json"), right.shape[:2])
        sources = [("manual_floor", left_masks["floor_eligible"], right_masks["floor_eligible"])]
        if args.auto_floor_left_dir is not None and args.auto_floor_right_dir is not None:
            sources.append((
                "mask2former_floor",
                frozen.load_binary(args.auto_floor_left_dir / name, left.shape[:2]),
                frozen.load_binary(args.auto_floor_right_dir / name, right.shape[:2]),
            ))
        for source_name, left_mask, right_mask in sources:
            frozen_candidate = frozen.stereo_candidates(
                left, right, left_mask, right_mask, calibration, parameters
            )
            learned_views = learned.build_rectified_views(
                left, right, left_mask, right_mask, calibration, parameters
            )
            checks: list[dict[str, Any]] = []
            if frozen_candidate["status"] != "candidate" or learned_views["status"] != "candidate":
                checks.append({"field": "status", "equal": False,
                               "reason": f"{frozen_candidate['status']} vs {learned_views['status']}"})
            else:
                frozen_rect = frozen_candidate["rectification"]
                learned_rect = learned_views["rectification"]
                for field in ("left_map_x", "left_map_y", "right_map_x", "right_map_y",
                              "virtual_K", "rotation_left", "translation_right"):
                    checks.append(compare(
                        f"rectification.{field}", getattr(frozen_rect, field), getattr(learned_rect, field)
                    ))
                for field in ("left_local", "right_local", "left_mask_local", "right_mask_local"):
                    checks.append(compare(field, frozen_candidate[field], learned_views[field]))
                checks.append(compare(
                    "left_mask_for_matching", frozen_candidate["left_mask_for_matching"],
                    learned_views["left_mask_for_matching"],
                ))
                checks.append(compare(
                    "right_mask_for_matching", frozen_candidate["right_mask_for_matching"],
                    learned_views["right_mask_for_matching"],
                ))
                checks.append(compare(
                    "matching_funnel.left_mask_pixels",
                    frozen_candidate["matching_funnel"]["left_mask_pixels"],
                    int((learned_views["left_mask_for_matching"] > 0).sum()),
                ))
            equal = all(item["equal"] for item in checks)
            all_equal = all_equal and equal
            report["frames"].append({
                "frame_id": name, "semantic_source": source_name, "equal": equal, "checks": checks,
            })
    report["all_fields_equal"] = bool(all_equal)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "all_fields_equal": report["all_fields_equal"],
        "checks": sum(len(item["checks"]) for item in report["frames"]),
        "failed_checks": [
            {"frame": item["frame_id"], "source": item["semantic_source"], "field": check["field"],
             "reason": check.get("reason"), "max_abs_difference": check.get("max_abs_difference")}
            for item in report["frames"] for check in item["checks"] if not check["equal"]
        ],
    }, ensure_ascii=False))
    return 0 if all_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
