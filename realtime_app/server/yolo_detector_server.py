from __future__ import annotations
import argparse
import time
from typing import Any
import numpy as np
import torch
from ultralytics import YOLO
from common_http import run_server


def sync(device: str) -> None:
    if torch.cuda.is_available() and str(device).lower() not in {"cpu", "mps"}:
        torch.cuda.synchronize()


class Adapter:
    model_name = "YOLO26x detector"

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model = YOLO(args.weights)
        self.model.predict(
            source=np.zeros((640, 640, 3), dtype=np.uint8),
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            classes=[0],
            verbose=False,
        )
        sync(args.device)

    def health(self) -> dict[str, Any]:
        return {
            "weights": self.args.weights,
            "device": self.args.device,
            "conf": self.args.conf,
            "iou": self.args.iou,
        }

    def infer(self, image: np.ndarray, payload: dict[str, Any]) -> dict[str, Any]:
        sync(self.args.device)
        started = time.perf_counter()
        results = self.model.predict(
            source=image,
            device=self.args.device,
            imgsz=self.args.imgsz,
            conf=self.args.conf,
            iou=self.args.iou,
            max_det=self.args.max_det,
            classes=[0],
            verbose=False,
        )
        sync(self.args.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        detections: list[dict[str, Any]] = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.detach().cpu().numpy()
            scores = results[0].boxes.conf.detach().cpu().numpy()
            for box, score in zip(boxes, scores):
                detections.append(
                    {
                        "bbox": [float(v) for v in box[:4]],
                        "score": float(score),
                        "class_id": 0,
                        "class_name": "person",
                    }
                )
        height, width = image.shape[:2]
        return {
            "model": self.model_name,
            "source_frame_id": payload.get("source_frame_id"),
            "image_width": width,
            "image_height": height,
            "model_ms": elapsed_ms,
            "detections": detections,
        }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18081)
    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    run_server(Adapter(args), args.host, args.port)
