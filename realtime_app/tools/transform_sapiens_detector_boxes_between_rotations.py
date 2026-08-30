"""Transform selected Sapiens detector boxes between image rotations.

This creates a matched-content detector JSON for a top-down pose control.  The
highest-score box from the source Sapiens run is restored to raw camera pixels
and then mapped into the target model-input rotation.  No new detector is run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from pose_app.rotation import model_image_size, model_to_raw_point, raw_to_model_point


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sapiens-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--source-rotation", required=True)
    parser.add_argument("--target-rotation", required=True)
    parser.add_argument("--raw-width", type=int, default=1920)
    parser.add_argument("--raw-height", type=int, default=1080)
    return parser.parse_args()


def transform_bbox(box: list[float], raw_width: int, raw_height: int, source_rotation: str, target_rotation: str) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    source_corners = ((x1, y1), (x1, y2), (x2, y1), (x2, y2))
    raw_corners = [model_to_raw_point(x, y, raw_width, raw_height, source_rotation) for x, y in source_corners]
    target_corners = [raw_to_model_point(x, y, raw_width, raw_height, target_rotation) for x, y in raw_corners]
    width, height = model_image_size(raw_width, raw_height, target_rotation)
    xs = [min(max(point[0], 0.0), width - 1.0) for point in target_corners]
    ys = [min(max(point[1], 0.0), height - 1.0) for point in target_corners]
    return [min(xs), min(ys), max(xs), max(ys)]


def main() -> int:
    args = parse_args()
    source = json.loads(args.source_sapiens_json.resolve().read_text(encoding="utf-8-sig"))
    images = source.get("images")
    if not isinstance(images, list):
        raise RuntimeError("Source Sapiens JSON has no images list")
    target_width, target_height = model_image_size(args.raw_width, args.raw_height, args.target_rotation)
    output_images = []
    for image_index, image in enumerate(images):
        instances = image.get("instances", [])
        detections = []
        if instances:
            selected = max(instances, key=lambda item: float(item["bbox_score_from_yolo26x"]))
            detections.append({
                "class_id": 0,
                "class_name": "person",
                "bbox_xyxy": transform_bbox(
                    selected["bbox_xyxy_from_yolo26x"],
                    args.raw_width,
                    args.raw_height,
                    args.source_rotation,
                    args.target_rotation,
                ),
                "score": float(selected["bbox_score_from_yolo26x"]),
            })
        output_images.append({
            "image_id": image.get("image_id", image_index),
            "frame_index": image_index,
            "file_name": image["file_name"],
            "image_path": f"/workspace/input/{image['file_name']}",
            "width": target_width,
            "height": target_height,
            "detections": detections,
        })
    payload = {
        "schema_version": "1.0",
        "detector": "YOLO26x historical highest-score box transformed through raw fisheye coordinates",
        "matched_crop_control": {
            "source_sapiens_json": str(args.source_sapiens_json.resolve()),
            "source_rotation": args.source_rotation,
            "target_rotation": args.target_rotation,
            "selection": "highest bbox score only",
        },
        "images": output_images,
    }
    output_path = args.output_json.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "images": len(output_images), "detections": sum(bool(row["detections"]) for row in output_images)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
