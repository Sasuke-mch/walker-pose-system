#!/usr/bin/env python3
"""Create a contiguous selection manifest without copying source images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-selection", type=Path, required=True)
    parser.add_argument("--output-selection", type=Path, required=True)
    parser.add_argument("--start-pair", type=int, required=True)
    parser.add_argument("--end-pair", type=int, required=True)
    args = parser.parse_args()

    source_manifest = args.input_selection / "selection_manifest.csv"
    with source_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_name = {row["file_name"]: row for row in rows}
    names = [f"pair_{pair_id:04d}.png" for pair_id in range(args.start_pair, args.end_pair + 1)]
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"selection is missing requested frames: {missing}")
    selected = [by_name[name] for name in names]
    if args.output_selection.exists():
        raise FileExistsError(args.output_selection)
    args.output_selection.mkdir(parents=True)
    with (args.output_selection / "selection_manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    metadata = {
        "source_selection": str(args.input_selection.resolve()),
        "start_pair": args.start_pair,
        "end_pair": args.end_pair,
        "count": len(selected),
        "selection": "contiguous file_name filter; source rows unchanged",
    }
    (args.output_selection / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
