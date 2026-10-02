#!/usr/bin/env python3
"""Estimate relative left-camera pose from pre-masked static stereo 3-D correspondences.

The input does not permit human, walker, or fitted-ground correspondences.
It is the auditable RANSAC back end of the static-background VO route; a later
feature front end must emit the documented correspondence JSONL.
"""

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
from typing import Any

ROOT = _tool_app_root
from pose_app.static_background_vo import StaticCorrespondences, relative_pose_record  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-correspondences", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--ransac-distance-threshold-mm", type=float, required=True)
    parser.add_argument("--ransac-iterations", type=int, default=400)
    parser.add_argument("--minimum-inliers", type=int, required=True)
    parser.add_argument("--random-seed", type=int, default=20260909)
    args = parser.parse_args()
    if args.output_jsonl.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_jsonl}")
    rows = []
    for raw in read_jsonl(args.static_correspondences):
        item = StaticCorrespondences.from_mapping(raw)
        rows.append(relative_pose_record(item, distance_threshold_mm=args.ransac_distance_threshold_mm,
                                         iterations=args.ransac_iterations, random_seed=args.random_seed + item.from_frame_index,
                                         minimum_inliers=args.minimum_inliers))
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"output": str(args.output_jsonl), "accepted": sum(row["status"] == "accepted" for row in rows), "total": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
