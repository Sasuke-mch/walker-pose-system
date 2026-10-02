#!/usr/bin/env python3
"""Generate unvalidated indoor-floor mask candidates with ADE20K SegFormer.

This is a candidate generator only.  Its masks must be visually/independently
audited before they are supplied to the direct local-ground tool with
``--semantic-identity-evidence manually_audited``.  It never calls a mask a
ground observation and it never modifies pose or stereo-geometry results.
"""

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
from typing import Any

import cv2
import numpy as np


DEFAULT_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"


def find_floor_class_id(id2label: dict[Any, Any]) -> int:
    matches = [int(index) for index, label in id2label.items() if str(label).strip().casefold() == "floor"]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one ADE20K floor label, found {matches}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="upright pair_*.png images")
    parser.add_argument("--output-dir", type=Path, required=True, help="new binary-mask output directory")
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames")
    return parser.parse_args()


def resolve_device(requested: str, torch: Any) -> str:
    if requested == "auto":
        return "cuda" if bool(torch.cuda.is_available()) else "cpu"
    if requested == "cuda" and not bool(torch.cuda.is_available()):
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return requested


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite floor-mask output: {args.output_dir}")
    if args.batch_size < 1 or args.max_frames < 0:
        raise ValueError("batch size must be positive and max frames non-negative")
    paths = sorted(args.input_dir.glob("pair_*.png"))
    if args.max_frames:
        paths = paths[:args.max_frames]
    if not paths:
        raise ValueError(f"No pair_*.png images found in {args.input_dir}")

    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    except ImportError as exc:
        raise RuntimeError(
            "SegFormer candidate masks require optional packages. Install with: python -m pip install transformers"
        ) from exc

    device = resolve_device(args.device, torch)
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_id).to(device).eval()
    floor_class_id = find_floor_class_id(dict(model.config.id2label))
    args.output_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start:start + args.batch_size]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        encoded = processor(images=images, return_tensors="pt")
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
        for path, image, image_logits in zip(batch_paths, images, logits):
            width, height = image.size
            resized = torch.nn.functional.interpolate(
                image_logits.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
            )[0]
            labels = resized.argmax(dim=0).detach().cpu().numpy()
            mask = (labels == floor_class_id).astype(np.uint8) * 255
            output = args.output_dir / path.name
            if not cv2.imwrite(str(output), mask):
                raise RuntimeError(f"Cannot write mask: {output}")
            rows.append({"frame_file": path.name, "floor_class_id": floor_class_id, "floor_fraction": float((mask > 0).mean())})
    metadata = {
        "kind": "unvalidated_semantic_floor_candidate",
        "model_id": args.model_id,
        "model_label": str(model.config.id2label[floor_class_id]),
        "floor_class_id": floor_class_id,
        "device": device,
        "input_dir": str(args.input_dir.resolve()),
        "frames": len(rows),
        "interpretation_boundary": (
            "ADE20K model output is an unvalidated semantic candidate in a non-pinhole walker fisheye domain. "
            "It is not a manually audited ground mask and cannot directly authorize a local ground plane."
        ),
        "per_frame": rows,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_dir), "frames": len(rows), **metadata}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
