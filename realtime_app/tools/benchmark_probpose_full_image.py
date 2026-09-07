#!/usr/bin/env python3
"""Benchmark ProbPose with one fixed full-image region per image.

This script runs inside the existing ProbPose container.  It intentionally
accepts only a cache made by build_full_image_prompt_cache.py and validates
that every input region is the complete image.  It never runs or reads a
person detector.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-json", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--timing-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--max-images", type=int, default=None, help="Optional smoke-test limit; never use for the final experiment.")
    return parser.parse_args()


def sync(device: str) -> None:
    if "cuda" in device.lower() and torch.cuda.is_available():
        torch.cuda.synchronize()


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def squeeze_first(value: Any) -> np.ndarray:
    array = to_numpy(value)
    return array[0] if array.ndim > 0 and array.shape[0] == 1 else array


def field(prediction: Any, name: str, default: Any) -> Any:
    return getattr(prediction, name) if hasattr(prediction, name) else default


def load_prompts(path: Path) -> list[dict]:
    root = json.loads(path.read_text(encoding="utf-8"))
    if root.get("input_mode") != "fixed_full_image_prompt_no_detector":
        raise ValueError("Prompt cache does not declare fixed_full_image_prompt_no_detector")
    entries = root.get("images")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Prompt cache must contain a non-empty images list")
    names = set()
    for entry in entries:
        name = entry.get("file_name")
        if not isinstance(name, str) or name in names:
            raise ValueError("Prompt cache has missing or duplicate file_name")
        names.add(name)
        if entry.get("prompt_kind") != "fixed_full_image":
            raise ValueError(f"{name}: prompt_kind is not fixed_full_image")
        width, height = float(entry["width"]), float(entry["height"])
        regions = entry.get("regions")
        if not isinstance(regions, list) or len(regions) != 1:
            raise ValueError(f"{name}: expected exactly one region")
        region = [float(value) for value in regions[0].get("region_xyxy", [])]
        if region != [0.0, 0.0, width, height]:
            raise ValueError(f"{name}: region is not the complete image")
        image_path = entry.get("image_path")
        if not isinstance(image_path, str) or not Path(image_path).is_file():
            raise FileNotFoundError(f"{name}: cannot access mounted image_path {image_path!r}")
    return entries


def infer(model: Any, entry: dict, device: str) -> tuple[list[Any], float]:
    region = np.asarray(entry["regions"][0]["region_xyxy"], dtype=np.float32).reshape(1, 4)
    sync(device)
    started = time.perf_counter()
    with torch.no_grad():
        results = inference_topdown(model, entry["image_path"], bboxes=region, bbox_format="xyxy")
    sync(device)
    return results, (time.perf_counter() - started) * 1000.0


def serialize_instance(result: Any, input_region: list[float]) -> dict:
    prediction = result.pred_instances
    keypoints = squeeze_first(field(prediction, "keypoints", np.zeros((17, 2), dtype=np.float32)))
    scores = squeeze_first(field(prediction, "keypoint_scores", np.ones((17,), dtype=np.float32)))
    output_boxes = squeeze_first(field(prediction, "bboxes", np.asarray(input_region, dtype=np.float32)))
    output_box = np.asarray(output_boxes).reshape(-1)
    if output_box.size < 4:
        output_box = np.asarray(input_region, dtype=np.float32)
    points = []
    for index in range(min(17, len(keypoints))):
        score = float(scores[index]) if index < len(scores) else 1.0
        points.append([float(keypoints[index][0]), float(keypoints[index][1]), score])
    if len(points) != 17:
        raise RuntimeError("ProbPose did not return 17 COCO keypoints")
    return {
        "input_region_xyxy": [float(value) for value in input_region],
        "input_region_score": 1.0,
        "output_bbox_xyxy": output_box[:4].astype(float).tolist(),
        "keypoints_coco17": points,
    }


def p95(values: list[float]) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), 95))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("--warmup must be non-negative and --repeat must be positive")
    output_paths = (args.output_json.resolve(), args.timing_csv.resolve(), args.summary_csv.resolve())
    if any(path.exists() for path in output_paths):
        raise FileExistsError("Refusing to overwrite an existing output file")
    if "cuda" in args.device.lower() and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    entries = load_prompts(args.prompt_json.resolve())
    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError("--max-images must be positive when supplied")
        entries = entries[:args.max_images]
    register_all_modules()
    model = init_model(args.config, args.checkpoint, device=args.device)
    if "cuda" in args.device.lower():
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()

    for warm_round in range(args.warmup):
        for entry in entries:
            infer(model, entry, args.device)
        print(f"Warmup {warm_round + 1}/{args.warmup} done")

    rows: list[dict] = []
    output_images: list[dict] = []
    for round_id in range(1, args.repeat + 1):
        for image_index, entry in enumerate(entries):
            results, elapsed_ms = infer(model, entry, args.device)
            rows.append({
                "round": round_id,
                "image_index": image_index,
                "file_name": entry["file_name"],
                "elapsed_ms": round(elapsed_ms, 3),
                "input_region_count": 1,
                "output_instance_count": len(results),
            })
            if round_id == 1:
                input_region = [float(value) for value in entry["regions"][0]["region_xyxy"]]
                output_images.append({
                    "image_id": entry["image_id"],
                    "file_name": entry["file_name"],
                    "image_path": entry["image_path"],
                    "input_mode": "fixed_full_image_prompt_no_detector",
                    "num_input_regions": 1,
                    "num_output_instances": len(results),
                    "elapsed_ms_correctness_run": elapsed_ms,
                    "instances": [serialize_instance(result, input_region) for result in results],
                })
        print(f"Measured round {round_id}/{args.repeat} done")

    times = [float(row["elapsed_ms"]) for row in rows]
    summary = {
        "method": "ProbPose-s fixed full-image prompt",
        "input_mode": "fixed_full_image_prompt_no_detector",
        "images": len(entries),
        "warmup_rounds": args.warmup,
        "measured_rounds": args.repeat,
        "measured_calls": len(rows),
        "mean_ms_per_image": round(statistics.mean(times), 3),
        "median_ms_per_image": round(statistics.median(times), 3),
        "p95_ms_per_image": round(p95(times), 3),
        "pose_only_fps": round(1000.0 / statistics.mean(times), 3),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if "cuda" in args.device.lower() else "cpu",
        "peak_cuda_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 3) if "cuda" in args.device.lower() else 0.0,
        "timing_scope": "GPU-synchronized inference_topdown call including image decode and model preprocessing; excludes model loading and prompt-cache construction.",
    }
    write_csv(args.timing_csv.resolve(), rows)
    write_csv(args.summary_csv.resolve(), [summary])
    args.output_json.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output_json.resolve().write_text(json.dumps({
        "method": summary["method"],
        "input_mode": summary["input_mode"],
        "prompt_source": str(args.prompt_json.resolve()),
        "timing_scope": summary["timing_scope"],
        "images": output_images,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
