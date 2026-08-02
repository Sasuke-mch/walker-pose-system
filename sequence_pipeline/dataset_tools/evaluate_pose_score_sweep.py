from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all post-OKS-NMS pose predictions at multiple final pose-score "
            "thresholds without COCO's standard maxDets=20 truncation."
        )
    )
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument(
        "--pose-score-thresholds",
        nargs="+",
        type=float,
        default=[0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
    )
    parser.add_argument(
        "--oks-thresholds", nargs="+", type=float, default=[0.50, 0.75]
    )
    parser.add_argument("--category-name", default="person")
    return parser.parse_args()


def safe_div(a: int | float, b: int | float) -> float:
    return float(a / b) if b else 0.0


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows were generated.")
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_threshold(
    gt: COCO,
    all_img_ids: list[int],
    category_id: int,
    predictions: list[dict[str, Any]],
    pose_score_threshold: float,
    oks_thresholds: list[float],
) -> list[dict[str, Any]]:
    retained = [
        item
        for item in predictions
        if float(item.get("score", 0.0)) >= pose_score_threshold
    ]
    counts = Counter(int(item["image_id"]) for item in retained)
    max_per_image = max(counts.values(), default=0)
    max_dets = max(1, max_per_image)

    totals = {
        float(oks): {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "ignored_predictions": 0,
            "ignored_gt": 0,
        }
        for oks in oks_thresholds
    }

    if retained:
        quiet = io.StringIO()
        with contextlib.redirect_stdout(quiet):
            detections = gt.loadRes(retained)
            evaluator = COCOeval(gt, detections, "keypoints")
            evaluator.params.imgIds = all_img_ids
            evaluator.params.catIds = [category_id]
            evaluator.params.iouThrs = np.asarray(oks_thresholds, dtype=np.float64)
            evaluator.params.maxDets = [max_dets]
            evaluator.params.areaRng = [[0.0, 1.0e10]]
            evaluator.params.areaRngLbl = ["all"]
            evaluator.evaluate()

        eval_by_image = {
            int(item["image_id"]): item
            for item in evaluator.evalImgs
            if item is not None
        }

        for image_id in all_img_ids:
            result = eval_by_image.get(image_id)
            if result is None:
                annotation_ids = gt.getAnnIds(
                    imgIds=[image_id], catIds=[category_id]
                )
                annotations = gt.loadAnns(annotation_ids)
                valid_gt = sum(
                    not bool(annotation.get("ignore", 0))
                    and not bool(annotation.get("iscrowd", 0))
                    and int(annotation.get("num_keypoints", 0)) > 0
                    for annotation in annotations
                )
                for oks in oks_thresholds:
                    totals[float(oks)]["fp"] += counts.get(image_id, 0)
                    totals[float(oks)]["fn"] += valid_gt
                continue

            dt_matches = np.asarray(result["dtMatches"])
            gt_matches = np.asarray(result["gtMatches"])
            dt_ignore = np.asarray(result["dtIgnore"], dtype=bool)
            gt_ignore = np.asarray(result["gtIgnore"], dtype=bool)

            for index, oks in enumerate(oks_thresholds):
                valid_dt = ~dt_ignore[index]
                valid_gt = ~gt_ignore
                matched_dt = dt_matches[index] > 0
                matched_gt = gt_matches[index] > 0

                totals[float(oks)]["tp"] += int(
                    np.count_nonzero(matched_dt & valid_dt)
                )
                totals[float(oks)]["fp"] += int(
                    np.count_nonzero((~matched_dt) & valid_dt)
                )
                totals[float(oks)]["fn"] += int(
                    np.count_nonzero((~matched_gt) & valid_gt)
                )
                totals[float(oks)]["ignored_predictions"] += int(
                    np.count_nonzero(~valid_dt)
                )
                totals[float(oks)]["ignored_gt"] += int(
                    np.count_nonzero(~valid_gt)
                )
    else:
        valid_gt_total = 0
        ignored_gt_total = 0
        for image_id in all_img_ids:
            annotation_ids = gt.getAnnIds(imgIds=[image_id], catIds=[category_id])
            for annotation in gt.loadAnns(annotation_ids):
                ignored = (
                    bool(annotation.get("ignore", 0))
                    or bool(annotation.get("iscrowd", 0))
                    or int(annotation.get("num_keypoints", 0)) == 0
                )
                ignored_gt_total += int(ignored)
                valid_gt_total += int(not ignored)
        for oks in oks_thresholds:
            totals[float(oks)]["fn"] = valid_gt_total
            totals[float(oks)]["ignored_gt"] = ignored_gt_total

    rows: list[dict[str, Any]] = []
    images_over_20 = sum(count > 20 for count in counts.values())
    predictions_beyond_20 = sum(max(0, count - 20) for count in counts.values())

    for oks in oks_thresholds:
        metrics = totals[float(oks)]
        tp = int(metrics["tp"])
        fp = int(metrics["fp"])
        fn = int(metrics["fn"])
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        rows.append(
            {
                "pose_score_threshold": pose_score_threshold,
                "oks_threshold": float(oks),
                "images": len(all_img_ids),
                "gt_instances": tp + fn,
                "retained_predictions": len(retained),
                "mean_predictions_per_image": safe_div(
                    len(retained), len(all_img_ids)
                ),
                "max_predictions_per_image": max_per_image,
                "images_over_20_predictions": images_over_20,
                "predictions_beyond_20": predictions_beyond_20,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fp_per_image": safe_div(fp, len(all_img_ids)),
                "fn_per_image": safe_div(fn, len(all_img_ids)),
                "ignored_predictions": int(metrics["ignored_predictions"]),
                "ignored_gt": int(metrics["ignored_gt"]),
                "max_dets_used": max_dets,
            }
        )

    return rows


def main() -> None:
    args = parse_args()
    prediction_path = Path(args.predictions)
    predictions = json.loads(prediction_path.read_text(encoding="utf-8-sig"))
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("Prediction JSON must be a non-empty list.")

    gt = COCO(args.annotation)
    image_ids = sorted(map(int, gt.getImgIds()))
    image_id_set = set(image_ids)

    unknown_image_ids = sorted(
        {
            int(item["image_id"])
            for item in predictions
            if int(item["image_id"]) not in image_id_set
        }
    )
    if unknown_image_ids:
        raise ValueError(f"Unknown image IDs: {unknown_image_ids[:10]}")

    category_ids = gt.getCatIds(catNms=[args.category_name])
    if len(category_ids) != 1:
        raise ValueError(
            f"Expected one category named {args.category_name}, got {category_ids}"
        )
    category_id = int(category_ids[0])
    predictions = [
        item
        for item in predictions
        if int(item.get("category_id", category_id)) == category_id
    ]

    thresholds = sorted(set(float(value) for value in args.pose_score_thresholds))
    oks_thresholds = sorted(set(float(value) for value in args.oks_thresholds))

    all_rows: list[dict[str, Any]] = []
    print("===== POSE-SCORE THRESHOLD SWEEP =====")
    print(f"Images: {len(image_ids)}")
    print(f"Input post-OKS-NMS predictions: {len(predictions)}")

    for threshold in thresholds:
        rows = evaluate_threshold(
            gt=gt,
            all_img_ids=image_ids,
            category_id=category_id,
            predictions=predictions,
            pose_score_threshold=threshold,
            oks_thresholds=oks_thresholds,
        )
        all_rows.extend(rows)
        for row in rows:
            print(
                f"pose_score>={threshold:.3f} | OKS={row['oks_threshold']:.2f} | "
                f"kept={row['retained_predictions']} TP={row['tp']} FP={row['fp']} "
                f"FN={row['fn']} P={row['precision']:.6f} "
                f"R={row['recall']:.6f} F1={row['f1']:.6f} "
                f"FP/image={row['fp_per_image']:.6f}"
            )

    write_csv(args.summary_csv, all_rows)
    output = {
        "annotation": args.annotation,
        "predictions": args.predictions,
        "evaluation_scope": (
            "All post-OKS-NMS predictions remaining after each final pose-score "
            "threshold; maxDets dynamically covers every retained prediction."
        ),
        "pose_score_thresholds": thresholds,
        "oks_thresholds": oks_thresholds,
        "rows": all_rows,
    }
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Summary CSV: {args.summary_csv}")
    print(f"Summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
