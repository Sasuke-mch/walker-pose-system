#!/usr/bin/env python3
"""Run walker 3-D representations on paired candidate masks.

This is intentionally a candidate-only pressure test.  It is useful when both
views have masks but those masks have not passed an independent semantic audit.
The output cannot be promoted to walker geometry, support, or hand contact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


TOOLS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_ROOT.parents[1]
REALTIME_ROOT = PROJECT_ROOT / "realtime_app"
for candidate in (str(REALTIME_ROOT), str(TOOLS_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from pose_app.calibration import StereoCalibration  # noqa: E402
from pose_app.scene_geometry_variants import (  # noqa: E402
    camera_attached_temporal_consensus,
    reconstruct_walker_components,
)
from benchmark_ground_walker_reconstruction import load_binary, stereo_candidates  # noqa: E402
import observe_local_ground_semantic_stereo_lr_direction_control as corrected  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-dir", type=Path, required=True)
    parser.add_argument("--right-dir", type=Path, required=True)
    parser.add_argument("--left-mask-dir", type=Path, required=True)
    parser.add_argument("--right-mask-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--frozen-baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-dilation-px", type=int, default=6)
    parser.add_argument("--maximum-pairs", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    left_names = {path.name for path in args.left_mask_dir.glob("pair_*.png")}
    right_names = {path.name for path in args.right_mask_dir.glob("pair_*.png")}
    names = sorted(left_names & right_names)
    if args.maximum_pairs > 0:
        names = names[: args.maximum_pairs]
    if not names:
        raise RuntimeError("no paired candidate masks")
    parameters, _ = corrected.load_frozen(args.frozen_baseline_dir)
    runtime_size = (int(parameters["runtime_width"]), int(parameters["runtime_height"]))
    calibration = StereoCalibration.load(args.calibration).for_runtime_sizes(runtime_size, runtime_size)
    args.output_dir.mkdir(parents=True)
    records = []
    point_sets = []
    for name in names:
        left = cv2.imread(str(args.left_dir / name), cv2.IMREAD_COLOR)
        right = cv2.imread(str(args.right_dir / name), cv2.IMREAD_COLOR)
        if left is None or right is None:
            raise RuntimeError(f"cannot read stereo image {name}")
        left_mask = load_binary(args.left_mask_dir / name, left.shape[:2])
        right_mask = load_binary(args.right_mask_dir / name, right.shape[:2])
        candidate = stereo_candidates(
            left, right, left_mask, right_mask, calibration, parameters,
            mask_dilation_px=args.mask_dilation_px,
        )
        points = candidate.get("points", np.empty((0, 3), dtype=np.float64))
        point_sets.append(points)
        if len(points):
            model = reconstruct_walker_components(
                points, candidate["left_pixels"], candidate["left_mask_for_matching"],
                minimum_component_pixels=30, minimum_component_points=10,
            )
        else:
            model = {"status": "unavailable", "reasons": candidate.get("reasons", ["no_stereo_candidates"])}
        records.append({
            "frame_id": name,
            "semantic_identity_evidence": "provided_unvalidated",
            "mask_dilation_px": args.mask_dilation_px,
            "matching_funnel": candidate.get("matching_funnel", {}),
            "stereo_candidate_count": int(len(points)),
            "component_model": model,
            "interpretation": "candidate-only pressure test; not verified walker geometry",
        })
    temporal = camera_attached_temporal_consensus(
        point_sets, voxel_size_mm=20.0, minimum_frame_support=max(2, min(4, len(records) // 3))
    )
    with (args.output_dir / "frame_records.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "schema_version": "walker_candidate_reconstruction_benchmark_v1",
        "pair_count": len(records),
        "frames_with_stereo_candidates": sum(record["stereo_candidate_count"] > 0 for record in records),
        "frames_with_component_model": sum(record["component_model"]["status"] == "candidate" for record in records),
        "candidate_count_median": float(np.median([record["stereo_candidate_count"] for record in records])),
        "camera_attached_temporal_consensus": temporal,
        "interpretation_boundary": (
            "Input masks are unvalidated candidates. Passing stereo or temporal gates does not establish "
            "walker identity, metric accuracy, rigidity, support, lift, or hand contact."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "run_metadata.json").write_text(json.dumps({
        "inputs": {key: str(getattr(args, key).resolve()) for key in (
            "left_dir", "right_dir", "left_mask_dir", "right_mask_dir", "calibration", "frozen_baseline_dir"
        )},
        "parameters": {"mask_dilation_px": args.mask_dilation_px, "frozen_stereo": parameters},
        "pair_names": names,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
