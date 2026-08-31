#!/usr/bin/env python3
"""Rotate a paired raw-fisheye image directory for upright pose-model input.

The source files remain unchanged.  The script creates an explicit manifest so
that model coordinates can later be inverse-mapped with the same rotations.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


ROTATIONS = {
    "none": None,
    "cw90": cv2.ROTATE_90_CLOCKWISE,
    "ccw90": cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rotation", choices=ROTATIONS, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    sources = sorted(args.input_dir.glob("*.png"))
    if not sources:
        raise RuntimeError(f"No PNG images in {args.input_dir}")
    args.output_dir.mkdir(parents=True)
    rows = []
    for source in sources:
        image = cv2.imread(str(source))
        if image is None:
            raise RuntimeError(f"Cannot read {source}")
        operation = ROTATIONS[args.rotation]
        rotated = image if operation is None else cv2.rotate(image, operation)
        target = args.output_dir / source.name
        if not cv2.imwrite(str(target), rotated):
            raise RuntimeError(f"Cannot write {target}")
        rows.append({
            "file_name": source.name,
            "source_width": image.shape[1], "source_height": image.shape[0],
            "model_width": rotated.shape[1], "model_height": rotated.shape[0],
            "rotation": args.rotation,
        })
    with (args.output_dir / "rotation_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print({"images": len(rows), "rotation": args.rotation, "output": str(args.output_dir)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
