#!/usr/bin/env python3
"""Extract a contiguous, provenance-preserving subset of saved detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-pair", type=int, required=True)
    parser.add_argument("--end-pair", type=int, required=True)
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    selected = [
        row for row in source["images"]
        if args.start_pair <= int(row["frame_index"]) <= args.end_pair
    ]
    expected = list(range(args.start_pair, args.end_pair + 1))
    actual = [int(row["frame_index"]) for row in selected]
    if actual != expected:
        raise ValueError(f"window is incomplete or unordered: expected {expected}, got {actual}")
    for row in selected:
        expected_name = f"pair_{int(row['frame_index']):04d}.png"
        if row["file_name"] != expected_name:
            raise ValueError(f"unexpected file mapping: {row['file_name']} != {expected_name}")
        if len(row.get("detections", [])) != 1:
            raise ValueError(f"pair {row['frame_index']} does not have exactly one frozen ROI")

    output = dict(source)
    output["images"] = selected
    output["window_extraction"] = {
        "source": str(args.input.resolve()),
        "start_pair": args.start_pair,
        "end_pair": args.end_pair,
        "count": len(selected),
        "selection": "contiguous frame_index filter; detections unchanged",
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["window_extraction"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
