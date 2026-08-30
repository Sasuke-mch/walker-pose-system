"""Run YOLO pose in a process isolated from OpenCV-based fisheye audit code."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("images", nargs="+", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"Missing weights: {args.weights}")
    for image in args.images:
        if not image.is_file():
            raise FileNotFoundError(f"Missing input image: {image}")
    model = YOLO(str(args.weights))
    results = model.predict(
        source=[str(image) for image in args.images],
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        verbose=False,
    )
    if torch.cuda.is_available() and str(args.device).lower() not in {"cpu", "mps"}:
        torch.cuda.synchronize()
    payload = {
        "execution_scope": "isolated_yolo_process_without_project_opencv_audit_import",
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "views": [],
    }
    for result in results:
        if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
            payload["views"].append({"boxes_xyxy": [], "box_scores": [], "keypoints_xy": [], "keypoint_scores": []})
            continue
        points = result.keypoints.xy.detach().cpu().numpy()
        scores = (
            result.keypoints.conf.detach().cpu().numpy()
            if result.keypoints.conf is not None
            else None
        )
        payload["views"].append(
            {
                "boxes_xyxy": result.boxes.xyxy.detach().cpu().numpy().astype(float).tolist(),
                "box_scores": result.boxes.conf.detach().cpu().numpy().astype(float).tolist(),
                "keypoints_xy": points.astype(float).tolist(),
                "keypoint_scores": (np.ones(points.shape[:2], dtype=float).tolist() if scores is None else scores.astype(float).tolist()),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
