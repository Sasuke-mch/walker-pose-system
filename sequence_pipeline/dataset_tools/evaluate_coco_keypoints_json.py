#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from xtcocotools.coco import COCO
    from xtcocotools.cocoeval import COCOeval
except ImportError:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an existing COCO keypoint result JSON with standard COCO metrics."
    )
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--summary-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    annotation = Path(args.annotation)
    predictions = Path(args.predictions)
    metrics_json = Path(args.metrics_json)
    summary_csv = Path(args.summary_csv)

    if not annotation.is_file():
        raise FileNotFoundError(f"Annotation not found: {annotation}")
    if not predictions.is_file():
        raise FileNotFoundError(f"Predictions not found: {predictions}")

    with predictions.open("r", encoding="utf-8-sig") as f:
        prediction_records = json.load(f)

    if not isinstance(prediction_records, list):
        raise ValueError("Prediction JSON must be a list of COCO result records.")

    coco_gt = COCO(str(annotation))
    coco_dt = coco_gt.loadRes(str(predictions))

    evaluator = COCOeval(coco_gt, coco_dt, "keypoints")
    evaluator.params.imgIds = sorted(coco_gt.getImgIds())

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = [float(x) for x in evaluator.stats]
    names = [
        "AP",
        "AP50",
        "AP75",
        "APM",
        "APL",
        "AR",
        "AR50",
        "AR75",
        "ARM",
        "ARL",
    ]
    metrics = dict(zip(names, stats))
    metrics.update(
        {
            "images": len(evaluator.params.imgIds),
            "prediction_instances": len(prediction_records),
            "max_dets": int(evaluator.params.maxDets[-1]),
        }
    )

    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    print("===== STANDARD COCO KEYPOINT EVALUATION =====")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Metrics JSON: {metrics_json}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
