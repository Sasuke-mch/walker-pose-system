#!/usr/bin/env python3
"""Build T1--T4 together from one completed stereo_results.jsonl file."""

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


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pose_app.lower_limb_pipeline import build_lower_limb_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stereo-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coordinate-transform", type=Path)
    parser.add_argument("--allow-test-coordinate-transform", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_lower_limb_pipeline(
        args.stereo_results,
        args.output_dir,
        coordinate_transform_path=args.coordinate_transform,
        allow_test_coordinate_transform=args.allow_test_coordinate_transform,
    )
    print(json.dumps({"output": str(args.output_dir.resolve()), **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
