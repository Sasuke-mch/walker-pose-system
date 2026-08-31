#!/usr/bin/env python3
"""Run YOLO26x person detection and write an auditable per-image JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--batch", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images = sorted(args.image_dir.glob("*.png"))
    if not images:
        raise RuntimeError(f"No PNG images in {args.image_dir}")
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    model = YOLO(str(args.weights))
    results = model.predict(
        source=[str(path) for path in images], device=args.device, imgsz=args.imgsz,
        conf=args.conf, iou=args.iou, batch=args.batch, verbose=False,
    )
    records = []
    for index, (path, result) in enumerate(zip(images, results)):
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Cannot read {path}")
        detections = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            for box, score, class_id in zip(xyxy, scores, classes):
                # COCO class 0 is person. Explicitly filter instead of trusting rank.
                if int(class_id) != 0:
                    continue
                detections.append({
                    "class_id": 0,
                    "class_name": "person",
                    "bbox_xyxy": [float(value) for value in box.tolist()],
                    "score": float(score),
                })
        detections.sort(key=lambda item: item["score"], reverse=True)
        records.append({
            "image_id": index,
            "frame_index": index,
            "file_name": path.name,
            "image_path": str(path),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "detections": detections,
        })
    payload = {
        "schema_version": "yolo26x_person_detection_v1",
        "detector": "YOLO26x",
        "weights": str(args.weights),
        "settings": {"imgsz": args.imgsz, "conf": args.conf, "iou": args.iou, "batch": args.batch},
        "images": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"images": len(records), "person_detected": sum(bool(x["detections"]) for x in records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
