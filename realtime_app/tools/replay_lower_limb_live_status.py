#!/usr/bin/env python3
"""Replay completed stereo JSONL through the online lower-limb status contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_app.lower_limb_live_status import LowerLimbLiveStatusWriter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stereo-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coordinate-transform", type=Path)
    parser.add_argument("--allow-test-coordinate-transform", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.stereo_results.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing live-status output: {output_dir}")
    if not source.is_file():
        raise FileNotFoundError(f"stereo results JSONL not found: {source}")

    output_dir.mkdir(parents=True)
    writer = LowerLimbLiveStatusWriter(
        output_dir / "lower_limb_live_status.jsonl",
        coordinate_transform_path=args.coordinate_transform,
        allow_test_coordinate_transform=args.allow_test_coordinate_transform,
    )
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    writer.consume(json.loads(line))
        summary = writer.close(completed=True)
    except Exception:
        writer.close(completed=False)
        raise

    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "input_stereo_results": str(source),
                "coordinate_transform": (
                    str(args.coordinate_transform.resolve())
                    if args.coordinate_transform is not None
                    else None
                ),
                "allow_test_coordinate_transform": args.allow_test_coordinate_transform,
                "source_observation_policy": "direct_accepted_stereo_only",
                "write_policy": "new output directory only; source stereo JSONL is read-only",
                "interpretation_boundary": (
                    "Offline replay exercises the same online tail contract used by run_stereo; "
                    "it does not run 2-D inference or re-triangulate data."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_dir), **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
