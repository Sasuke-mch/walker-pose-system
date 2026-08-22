import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = map(float, box)
    return [x, y, x + width, y + height]


def box_iou(
    first: list[float],
    second: list[float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)
    intersection = intersection_width * intersection_height

    first_area = max(
        0.0,
        first[2] - first[0],
    ) * max(
        0.0,
        first[3] - first[1],
    )

    second_area = max(
        0.0,
        second[2] - second[0],
    ) * max(
        0.0,
        second[3] - second[1],
    )

    union = first_area + second_area - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def load_ground_truth(
    path: Path,
) -> dict[str, list[list[float]]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        coco = json.load(file)

    file_name_by_id = {
        int(image["id"]): image["file_name"]
        for image in coco.get("images", [])
    }

    boxes_by_file: dict[str, list[list[float]]] = defaultdict(list)

    for annotation in coco.get("annotations", []):
        image_id = int(annotation["image_id"])
        file_name = file_name_by_id.get(image_id)

        if file_name is None:
            continue

        bbox = annotation.get("bbox", [])

        if len(bbox) != 4:
            continue

        boxes_by_file[file_name].append(
            xywh_to_xyxy(bbox)
        )

    return dict(boxes_by_file)


def load_detections(
    path: Path,
) -> dict[str, list[dict[str, Any]]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        data = json.load(file)

    if isinstance(data, dict):
        records = data.get(
            "images",
            data.get(
                "items",
                data.get("records", []),
            ),
        )
    elif isinstance(data, list):
        records = data
    else:
        raise TypeError(
            "Unsupported detection JSON structure"
        )

    output: dict[str, list[dict[str, Any]]] = {}

    for record in records:
        file_name = (
            record.get("file_name")
            or record.get("filename")
            or Path(record.get("image_path", "")).name
        )

        if not file_name:
            continue

        raw_detections = record.get(
            "detections",
            record.get(
                "instances",
                record.get("predictions", []),
            ),
        )

        parsed: list[dict[str, Any]] = []

        for detection in raw_detections:
            if not isinstance(detection, dict):
                continue

            bbox = (
                detection.get("bbox_xyxy")
                or detection.get("box_xyxy")
                or detection.get("bbox")
                or detection.get("box")
            )

            if bbox is None or len(bbox) != 4:
                continue

            score = detection.get(
                "score",
                detection.get(
                    "confidence",
                    detection.get("conf", 1.0),
                ),
            )

            parsed.append(
                {
                    "bbox": list(map(float, bbox)),
                    "score": float(score),
                }
            )

        output[file_name] = parsed

    return output


def match_image(
    ground_truth: list[list[float]],
    detections: list[dict[str, Any]],
    score_threshold: float,
    iou_threshold: float,
) -> tuple[int, int, int]:
    selected = [
        detection
        for detection in detections
        if detection["score"] >= score_threshold
    ]

    selected.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    matched_gt: set[int] = set()
    true_positives = 0

    for detection in selected:
        best_index = -1
        best_iou = 0.0

        for gt_index, gt_box in enumerate(ground_truth):
            if gt_index in matched_gt:
                continue

            overlap = box_iou(
                detection["bbox"],
                gt_box,
            )

            if overlap > best_iou:
                best_iou = overlap
                best_index = gt_index

        if (
            best_index >= 0
            and best_iou >= iou_threshold
        ):
            matched_gt.add(best_index)
            true_positives += 1

    false_positives = len(selected) - true_positives
    false_negatives = len(ground_truth) - true_positives

    return (
        true_positives,
        false_positives,
        false_negatives,
    )


def safe_ratio(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze YOLO detection confidence thresholds "
            "against COCO ground-truth boxes."
        )
    )
    parser.add_argument(
        "--detections",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.40,
            0.50,
        ],
    )

    args = parser.parse_args()

    gt_by_file = load_ground_truth(
        args.annotation
    )
    detections_by_file = load_detections(
        args.detections
    )

    file_names = sorted(gt_by_file)

    total_gt = sum(
        len(gt_by_file[file_name])
        for file_name in file_names
    )

    print("=" * 78)
    print("Images:", len(file_names))
    print("GT persons:", total_gt)
    print(
        "IoU matching threshold:",
        args.iou_threshold,
    )
    print()

    count_header = (
        f"{'file_name':<15}"
        f"{'GT':>5}"
    )

    for threshold in args.thresholds:
        count_header += f"{threshold:>8.2f}"

    print("Detection counts per image")
    print(count_header)
    print("-" * len(count_header))

    for file_name in file_names:
        row = (
            f"{file_name:<15}"
            f"{len(gt_by_file[file_name]):>5}"
        )

        detections = detections_by_file.get(
            file_name,
            [],
        )

        for threshold in args.thresholds:
            count = sum(
                detection["score"] >= threshold
                for detection in detections
            )
            row += f"{count:>8}"

        print(row)

    print()
    print(
        f"{'threshold':>10}"
        f"{'detections':>12}"
        f"{'TP':>7}"
        f"{'FP':>7}"
        f"{'FN':>7}"
        f"{'precision':>12}"
        f"{'recall':>10}"
        f"{'F1':>10}"
    )
    print("-" * 85)

    rows = []

    for threshold in args.thresholds:
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for file_name in file_names:
            tp, fp, fn = match_image(
                ground_truth=gt_by_file[file_name],
                detections=detections_by_file.get(
                    file_name,
                    [],
                ),
                score_threshold=threshold,
                iou_threshold=args.iou_threshold,
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = safe_ratio(
            total_tp,
            total_tp + total_fp,
        )
        recall = safe_ratio(
            total_tp,
            total_tp + total_fn,
        )
        f1 = safe_ratio(
            2 * precision * recall,
            precision + recall,
        )

        detection_count = total_tp + total_fp

        row = {
            "threshold": threshold,
            "detections": detection_count,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        rows.append(row)

        print(
            f"{threshold:>10.2f}"
            f"{detection_count:>12}"
            f"{total_tp:>7}"
            f"{total_fp:>7}"
            f"{total_fn:>7}"
            f"{precision:>12.3f}"
            f"{recall:>10.3f}"
            f"{f1:>10.3f}"
        )

    args.output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Saved:", args.output_csv)
    print()
    print(
        "Note: this is a detector diagnostic using bbox IoU, "
        "not the final keypoint AP evaluation."
    )


if __name__ == "__main__":
    main()
