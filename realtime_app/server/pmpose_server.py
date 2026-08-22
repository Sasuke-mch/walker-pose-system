from __future__ import annotations
import argparse
import time
from typing import Any
import numpy as np
import torch
from pmpose import PMPose
from common_http import run_server


def sync(device: str) -> None:
    if torch.cuda.is_available() and "cuda" in str(device).lower():
        torch.cuda.synchronize()


def parse_detections(payload: dict[str, Any], width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    boxes: list[list[float]] = []
    scores: list[float] = []
    raw = payload.get("detections", [])
    if not isinstance(raw, list):
        raw = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        box = item.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        x1 = max(0.0, min(float(width - 1), x1))
        y1 = max(0.0, min(float(height - 1), y1))
        x2 = max(x1 + 1.0, min(float(width), x2))
        y2 = max(y1 + 1.0, min(float(height), y2))
        if x2 - x1 < 2.0 or y2 - y1 < 2.0:
            continue
        boxes.append([x1, y1, x2, y2])
        scores.append(float(item.get("score", 1.0)))
    return (
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        np.asarray(scores, dtype=np.float32),
    )


def rectangle_masks(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    masks = np.zeros((len(boxes), height, width), dtype=np.uint8)
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))
        masks[index, y1:y2, x1:x2] = 1
    return masks


def unpack_prediction(result: Any) -> np.ndarray:
    if isinstance(result, (tuple, list)):
        keypoints = result[0]
    elif isinstance(result, dict):
        keypoints = result.get("keypoints")
    else:
        keypoints = result
    if hasattr(keypoints, "detach"):
        keypoints = keypoints.detach().cpu().numpy()
    array = np.asarray(keypoints)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[-1] < 3:
        raise RuntimeError(f"PMPose关键点形状异常：{array.shape}")
    return array.astype(np.float32, copy=False)


class Adapter:
    model_name = "YOLO26x + PMPose-b"

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model_name = f"YOLO26x + {args.variant}"
        self.model = PMPose(
            device=args.device,
            variant=args.variant,
            from_pretrained=True,
        )
        # Warm up one valid person crop so startup cost is not counted in frame 1.
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        boxes = np.asarray([[120, 40, 520, 620]], dtype=np.float32)
        masks = rectangle_masks(boxes, 640, 640)
        with torch.no_grad():
            self.model.predict(
                image=dummy,
                bboxes=boxes,
                masks=masks,
                return_probmaps=False,
            )
        sync(args.device)

    def health(self) -> dict[str, Any]:
        return {
            "device": self.args.device,
            "variant": self.args.variant,
            "pose_threshold": self.args.pose_thr,
            "keypoint_score_threshold": self.args.keypoint_score_thr,
            "mask_mode": self.args.mask_mode,
        }

    def infer(self, image: np.ndarray, payload: dict[str, Any]) -> dict[str, Any]:
        height, width = image.shape[:2]
        boxes, bbox_scores = parse_detections(payload, width, height)
        if len(boxes) == 0:
            return {
                "model": self.model_name,
                "source_frame_id": payload.get("source_frame_id"),
                "image_width": width,
                "image_height": height,
                "model_ms": 0.0,
                "persons": [],
            }
        masks = rectangle_masks(boxes, width, height)
        sync(self.args.device)
        started = time.perf_counter()
        with torch.no_grad():
            raw = self.model.predict(
                image=image,
                bboxes=boxes,
                masks=masks,
                return_probmaps=False,
            )
        sync(self.args.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        keypoints = unpack_prediction(raw)
        count = min(len(boxes), len(keypoints), len(bbox_scores))
        persons: list[dict[str, Any]] = []
        for index in range(count):
            current = keypoints[index, :17, :3]
            scores = current[:, 2]
            valid = scores > self.args.keypoint_score_thr
            mean_score = float(np.mean(scores[valid])) if np.any(valid) else 0.0
            pose_score = float(bbox_scores[index]) * mean_score
            if pose_score < self.args.pose_thr:
                continue
            persons.append(
                {
                    "person_id": len(persons),
                    "bbox": boxes[index].astype(float).tolist(),
                    "bbox_score": float(bbox_scores[index]),
                    "pose_score": pose_score,
                    "keypoints": current.astype(float).tolist(),
                }
            )
        return {
            "model": self.model_name,
            "source_frame_id": payload.get("source_frame_id"),
            "image_width": width,
            "image_height": height,
            "model_ms": elapsed_ms,
            "persons": persons,
        }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--variant", default="PMPose-b")
    parser.add_argument("--pose-thr", type=float, default=0.30)
    parser.add_argument("--keypoint-score-thr", type=float, default=0.20)
    parser.add_argument("--mask-mode", default="bbox")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18082)
    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    run_server(Adapter(args), args.host, args.port)
