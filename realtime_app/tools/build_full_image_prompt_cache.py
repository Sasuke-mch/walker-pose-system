#!/usr/bin/env python3
"""Build fixed whole-image prompts for a detector-free top-down pose experiment.

The resulting JSON deliberately resembles a generic region-input cache because
the two third-party top-down runners require an input region.  Every record is
checked to contain exactly one region that equals the full input image; no
detector is run or read by this tool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--container-image-dir",
        required=True,
        help="Mounted container directory used in each generated image_path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output = args.output_json.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    container_dir = args.container_image_dir.rstrip("/")
    if not container_dir.startswith("/"):
        raise ValueError("--container-image-dir must be an absolute container path")

    images = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise RuntimeError(f"No supported images in {input_dir}")

    records = []
    for index, path in enumerate(images):
        with Image.open(path) as image:
            width, height = image.size
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid image size: {path}")
        records.append({
            "image_id": index,
            "file_name": path.name,
            "image_path": f"{container_dir}/{path.name}",
            "width": width,
            "height": height,
            "prompt_kind": "fixed_full_image",
            "regions": [{
                "region_xyxy": [0.0, 0.0, float(width), float(height)],
                "score": 1.0,
                "prompt_kind": "fixed_full_image",
            }],
            # Compatibility payload for existing top-down runners.  This is
            # copied from regions above; it is not a detection result.
            "detections": [{
                "bbox_xyxy": [0.0, 0.0, float(width), float(height)],
                "score": 1.0,
                "class_name": "fixed_full_image_prompt",
                "prompt_kind": "fixed_full_image",
            }],
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_mode": "fixed_full_image_prompt_no_detector",
        "input_directory": str(input_dir),
        "contract": "One region per image, exactly [0, 0, image_width, image_height].",
        "images": records,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "images": len(records), "input_mode": payload["input_mode"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
