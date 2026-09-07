#!/usr/bin/env python3
"""Rewrite a legacy PMPose result as a detector-free full-image prompt result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--prompt-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    return parser.parse_args()


def as_regions(prompt: dict) -> list[list[float]]:
    if prompt.get("prompt_kind") != "fixed_full_image":
        raise ValueError(f"{prompt.get('file_name')}: expected fixed_full_image prompt")
    width, height = float(prompt["width"]), float(prompt["height"])
    regions = prompt.get("regions")
    if not isinstance(regions, list) or len(regions) != 1:
        raise ValueError(f"{prompt.get('file_name')}: expected one fixed full-image region")
    region = [float(value) for value in regions[0].get("region_xyxy", [])]
    if region != [0.0, 0.0, width, height]:
        raise ValueError(f"{prompt.get('file_name')}: region is not the complete image")
    return [region]


def main() -> int:
    args = parse_args()
    source_path, prompt_path, output = (args.source_json.resolve(), args.prompt_json.resolve(), args.output_json.resolve())
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    source = json.loads(source_path.read_text(encoding="utf-8-sig"))
    prompts = json.loads(prompt_path.read_text(encoding="utf-8-sig"))
    source_images = source.get("images")
    prompt_images = prompts.get("images")
    if not isinstance(source_images, list) or not isinstance(prompt_images, list):
        raise ValueError("Both files must contain an images list")
    prompt_by_name = {item["file_name"]: item for item in prompt_images}
    if len(prompt_by_name) != len(prompt_images) or {item["file_name"] for item in source_images} != set(prompt_by_name):
        raise RuntimeError("Prediction and full-image prompt file names do not match")

    output_images = []
    for source_image in source_images:
        prompt = prompt_by_name[source_image["file_name"]]
        output_images.append({
            "image_id": source_image.get("image_id"),
            "file_name": source_image["file_name"],
            "image_path": source_image["image_path"],
            "input_regions_xyxy": as_regions(prompt),
            "input_region_scores": [1.0],
            "output_bboxes": source_image.get("output_bboxes", []),
            "keypoints": source_image.get("keypoints", []),
            "presence": source_image.get("presence", []),
            "visibility": source_image.get("visibility", []),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "method": args.model_name,
        "input_mode": "fixed_full_image_prompt_no_detector",
        "source_prediction": str(source_path),
        "prompt_source": str(prompt_path),
        "images": output_images,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "images": len(output_images)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
