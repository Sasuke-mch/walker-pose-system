#!/usr/bin/env python3
"""Evaluate windowed walk/stationary candidates from strict PMPose stereo ankles."""

from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
from walker_tools._compat import APP_ROOT as _tool_app_root
_tool_prepare_imports()

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = _tool_app_root

from pose_app.gait_interaction_candidates import GaitCriteria, classify_walking_window  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-stereo-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-frames", type=int, default=15)
    parser.add_argument("--stride-frames", type=int, default=5)
    parser.add_argument("--nominal-fps", type=float, default=30.0)
    return parser.parse_args()


def ankle(person: dict, name: str) -> tuple[np.ndarray, bool]:
    for keypoint in person.get("keypoints_3d", []):
        if keypoint.get("name") == name:
            xyz = np.asarray(keypoint.get("xyz", [np.nan] * 3), dtype=np.float64)
            return xyz, bool(keypoint.get("valid")) and xyz.shape == (3,) and bool(np.all(np.isfinite(xyz)))
    return np.full(3, np.nan), False


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    if args.window_frames < 3 or args.stride_frames < 1 or args.nominal_fps <= 0:
        raise ValueError("window/stride/fps must be positive")
    rows = [json.loads(line) for line in args.strict_stereo_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    pair_ids, left, right, valid = [], [], [], []
    for row in rows:
        pair_ids.append(int(row["pair_id"]))
        persons = row.get("persons_3d", [])
        if persons:
            left_xyz, left_valid = ankle(persons[0], "left_ankle")
            right_xyz, right_valid = ankle(persons[0], "right_ankle")
        else:
            left_xyz = right_xyz = np.full(3, np.nan)
            left_valid = right_valid = False
        left.append(left_xyz)
        right.append(right_xyz)
        valid.append(left_valid and right_valid)
    left_array, right_array, valid_array = np.asarray(left), np.asarray(right), np.asarray(valid)
    criteria = GaitCriteria(minimum_window_frames=args.window_frames)
    outputs = []
    for start in range(0, max(0, len(rows) - args.window_frames + 1), args.stride_frames):
        stop = start + args.window_frames
        result = classify_walking_window(
            np.arange(start, stop, dtype=np.float64) / args.nominal_fps,
            left_array[start:stop], right_array[start:stop], valid_array[start:stop], criteria,
        )
        outputs.append({
            "start_pair_id": pair_ids[start], "end_pair_id": pair_ids[stop - 1],
            "window_start_row": start, "window_stop_row_exclusive": stop,
            **result,
        })
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "window_candidates.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for output in outputs:
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for output in outputs:
        counts[output["state"]] = counts.get(output["state"], 0) + 1
    summary = {
        "schema_version": "walking_state_candidate_evaluation_v1",
        "input_frames": len(rows),
        "both_ankles_strict_valid_frames": int(valid_array.sum()),
        "both_ankles_strict_valid_fraction": float(valid_array.mean()) if len(valid_array) else 0.0,
        "window_count": len(outputs),
        "state_counts": counts,
        "criteria": criteria.__dict__,
        "nominal_time_note": (
            "nominal frame-index time only; it is not camera exposure synchronization evidence and is not used "
            "to infer a physical contact event"
        ),
        "interpretation_boundary": (
            "Walk/stationary candidates from strict PMPose ankle motion relative to the walker-mounted camera. "
            "No ground contact, step, clinical gait event, or accuracy claim."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
