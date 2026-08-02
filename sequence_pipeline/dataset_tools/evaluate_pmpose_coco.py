import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from mmpose.evaluation.functional import oks_nms
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval


STAT_NAMES = [
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


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        return json.load(file)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def xyxy_to_xywh(box) -> list[float]:
    if len(box) != 4:
        raise ValueError(
            f"Expected four bbox values, found {len(box)}"
        )

    x1, y1, x2, y2 = map(float, box)

    return [
        x1,
        y1,
        max(0.0, x2 - x1),
        max(0.0, y2 - y1),
    ]


def find_person_category_id(annotation) -> int:
    categories = annotation.get(
        "categories",
        [],
    )

    for category in categories:
        if str(category.get("name", "")).lower() == "person":
            return int(category["id"])

    if len(categories) == 1:
        return int(categories[0]["id"])

    raise ValueError(
        "Could not determine the person category ID"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert PMPose raw predictions to COCO17, "
            "apply MMPose-style rescoring and OKS NMS, "
            "then evaluate COCO keypoint AP."
        )
    )

    parser.add_argument(
        "--raw-predictions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--detector-conf",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--expected-images",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--expected-instances",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--keypoint-score-thr",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--oks-nms-thr",
        type=float,
        default=0.9,
    )

    args = parser.parse_args()

    if not args.raw_predictions.is_file():
        raise FileNotFoundError(
            args.raw_predictions
        )

    if not args.annotation.is_file():
        raise FileNotFoundError(
            args.annotation
        )

    raw = load_json(
        args.raw_predictions
    )

    annotation = load_json(
        args.annotation
    )

    raw_images = raw.get(
        "images",
        [],
    )

    annotation_images = annotation.get(
        "images",
        [],
    )

    if not isinstance(raw_images, list):
        raise TypeError(
            "raw_predictions.json has no images list"
        )

    image_by_filename = {}

    for image in annotation_images:
        file_name = image.get("file_name")

        if not file_name:
            raise ValueError(
                "Annotation image has no file_name"
            )

        if file_name in image_by_filename:
            raise ValueError(
                f"Duplicate annotation file_name: {file_name}"
            )

        image_by_filename[file_name] = image

    person_category_id = (
        find_person_category_id(annotation)
    )

    errors = []
    grouped_instances = defaultdict(list)

    total_instances_before_nms = 0
    zero_instance_images = 0
    mapped_image_ids = set()

    for record_index, record in enumerate(
        raw_images
    ):
        file_name = record.get("file_name")

        if file_name not in image_by_filename:
            errors.append(
                f"Prediction file not in annotation: "
                f"{file_name}"
            )
            continue

        original_image_id = int(
            image_by_filename[file_name]["id"]
        )

        mapped_image_ids.add(
            original_image_id
        )

        boxes = record.get(
            "boxes_from_yolo26x",
            [],
        )

        bbox_scores = record.get(
            "bbox_scores_from_yolo26x",
            [],
        )

        output_bboxes = record.get(
            "output_bboxes",
            [],
        )

        keypoints = record.get(
            "keypoints",
            [],
        )

        presence = record.get(
            "presence",
            [],
        )

        visibility = record.get(
            "visibility",
            [],
        )

        instance_count = len(keypoints)

        if instance_count == 0:
            zero_instance_images += 1

        lengths = {
            "boxes": len(boxes),
            "bbox_scores": len(bbox_scores),
            "output_bboxes": len(output_bboxes),
            "keypoints": len(keypoints),
            "presence": len(presence),
            "visibility": len(visibility),
        }

        if len(set(lengths.values())) != 1:
            errors.append(
                f"{file_name}: inconsistent instance "
                f"counts {lengths}"
            )
            continue

        total_instances_before_nms += (
            instance_count
        )

        for person_index in range(
            instance_count
        ):
            keypoint_array = np.asarray(
                keypoints[person_index],
                dtype=np.float64,
            )

            if keypoint_array.shape != (23, 3):
                errors.append(
                    f"{file_name}, person "
                    f"{person_index}: keypoint shape "
                    f"{keypoint_array.shape}"
                )
                continue

            if not np.all(
                np.isfinite(keypoint_array)
            ):
                errors.append(
                    f"{file_name}, person "
                    f"{person_index}: non-finite "
                    "keypoint values"
                )
                continue

            coco_keypoints = (
                keypoint_array[:17]
            )

            coordinates = (
                coco_keypoints[:, :2]
            )

            keypoint_scores = (
                coco_keypoints[:, 2]
            )

            presence_array = np.asarray(
                presence[person_index],
                dtype=np.float64,
            ).reshape(-1)

            if presence_array.shape[0] < 17:
                errors.append(
                    f"{file_name}, person "
                    f"{person_index}: presence "
                    f"length {presence_array.shape[0]}"
                )
                continue

            coco_probabilities = (
                presence_array[:17]
            )

            if not np.all(
                np.isfinite(coco_probabilities)
            ):
                errors.append(
                    f"{file_name}, person "
                    f"{person_index}: non-finite "
                    "presence values"
                )
                continue

            bbox_score = float(
                bbox_scores[person_index]
            )

            valid_scores = (
                keypoint_scores[
                    keypoint_scores
                    > args.keypoint_score_thr
                ]
            )

            if valid_scores.size:
                mean_keypoint_score = float(
                    valid_scores.mean()
                )
            else:
                mean_keypoint_score = 0.0

            instance_score = (
                bbox_score
                * mean_keypoint_score
            )

            selected_box = (
                output_bboxes[person_index]
                if output_bboxes
                else boxes[person_index]
            )

            bbox_xywh = xyxy_to_xywh(
                selected_box
            )

            area = float(
                bbox_xywh[2]
                * bbox_xywh[3]
            )

            if area <= 0:
                x_span = float(
                    coordinates[:, 0].max()
                    - coordinates[:, 0].min()
                )

                y_span = float(
                    coordinates[:, 1].max()
                    - coordinates[:, 1].min()
                )

                area = max(
                    x_span * y_span,
                    1.0,
                )

            keypoints_for_evaluation = (
                np.concatenate(
                    [
                        coordinates,
                        coco_probabilities[
                            :, None
                        ],
                    ],
                    axis=1,
                )
            )

            grouped_instances[
                original_image_id
            ].append(
                {
                    "image_id": (
                        original_image_id
                    ),
                    "category_id": (
                        person_category_id
                    ),
                    "keypoints": (
                        keypoints_for_evaluation
                    ),
                    "bbox": np.asarray(
                        bbox_xywh,
                        dtype=np.float64,
                    ),
                    "area": area,
                    "score": (
                        instance_score
                    ),
                    "bbox_score": (
                        bbox_score
                    ),
                    "mean_keypoint_score": (
                        mean_keypoint_score
                    ),
                }
            )

    if (
        args.expected_images is not None
        and len(raw_images)
        != args.expected_images
    ):
        errors.append(
            f"Expected {args.expected_images} "
            f"image records, found "
            f"{len(raw_images)}"
        )

    if (
        args.expected_instances is not None
        and total_instances_before_nms
        != args.expected_instances
    ):
        errors.append(
            f"Expected {args.expected_instances} "
            f"instances, found "
            f"{total_instances_before_nms}"
        )

    if errors:
        print("PREDICTION CONVERSION FAILED")

        for error in errors[:50]:
            print("-", error)

        if len(errors) > 50:
            print(
                f"... and {len(errors) - 50} "
                "more errors"
            )

        raise SystemExit(1)

    coco_results = []
    total_instances_after_nms = 0
    suppressed_instances = 0

    for image_id in sorted(
        grouped_instances
    ):
        instances = grouped_instances[
            image_id
        ]

        keep_indices = oks_nms(
            instances,
            thr=args.oks_nms_thr,
        )

        keep_indices = [
            int(index)
            for index in keep_indices
        ]

        total_instances_after_nms += len(
            keep_indices
        )

        suppressed_instances += (
            len(instances)
            - len(keep_indices)
        )

        for index in keep_indices:
            instance = instances[index]

            flattened_keypoints = (
                instance["keypoints"]
                .reshape(-1)
                .tolist()
            )

            coco_results.append(
                {
                    "image_id": int(
                        instance["image_id"]
                    ),
                    "category_id": int(
                        instance[
                            "category_id"
                        ]
                    ),
                    "keypoints": (
                        flattened_keypoints
                    ),
                    "score": float(
                        instance["score"]
                    ),
                    "bbox": (
                        instance["bbox"]
                        .tolist()
                    ),
                }
            )

    save_json(
        args.output_json,
        coco_results,
    )

    print("Raw image records:", len(raw_images))
    print(
        "Mapped original image IDs:",
        len(mapped_image_ids),
    )
    print(
        "Instances before OKS NMS:",
        total_instances_before_nms,
    )
    print(
        "Instances after OKS NMS:",
        total_instances_after_nms,
    )
    print(
        "Suppressed by OKS NMS:",
        suppressed_instances,
    )
    print(
        "Zero-instance images:",
        zero_instance_images,
    )
    print(
        "COCO prediction JSON:",
        args.output_json,
    )

    coco_gt = COCO(
        str(args.annotation)
    )

    coco_dt = coco_gt.loadRes(
        str(args.output_json)
    )

    evaluator = COCOeval(
        coco_gt,
        coco_dt,
        "keypoints",
    )

    evaluator.params.imgIds = sorted(
        coco_gt.getImgIds()
    )

    evaluator.params.catIds = [
        person_category_id
    ]

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    metric_values = {
        name: float(value)
        for name, value in zip(
            STAT_NAMES,
            evaluator.stats,
        )
    }

    metrics_output = {
        "method": raw.get(
            "method",
            "YOLO26x + PMPose-b",
        ),
        "detector_confidence": (
            args.detector_conf
        ),
        "keypoint_mapping": (
            "PMPose indices 0-16 to COCO17"
        ),
        "instance_score_mode": (
            "bbox_score * mean(keypoint_score "
            "where keypoint_score > threshold)"
        ),
        "keypoint_score_threshold": (
            args.keypoint_score_thr
        ),
        "keypoint_value_field": (
            "PMPose presence probability"
        ),
        "oks_nms_threshold": (
            args.oks_nms_thr
        ),
        "image_records": len(
            raw_images
        ),
        "instances_before_oks_nms": (
            total_instances_before_nms
        ),
        "instances_after_oks_nms": (
            total_instances_after_nms
        ),
        "suppressed_by_oks_nms": (
            suppressed_instances
        ),
        "zero_instance_images": (
            zero_instance_images
        ),
        "metrics": metric_values,
        "raw_predictions": str(
            args.raw_predictions
        ),
        "annotation": str(
            args.annotation
        ),
        "coco_predictions": str(
            args.output_json
        ),
    }

    save_json(
        args.metrics_json,
        metrics_output,
    )

    args.summary_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_row = {
        "detector_confidence": (
            args.detector_conf
        ),
        "images": len(raw_images),
        "instances_before_oks_nms": (
            total_instances_before_nms
        ),
        "instances_after_oks_nms": (
            total_instances_after_nms
        ),
        "suppressed_by_oks_nms": (
            suppressed_instances
        ),
        "zero_instance_images": (
            zero_instance_images
        ),
        **metric_values,
    }

    with args.summary_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                summary_row.keys()
            ),
        )

        writer.writeheader()
        writer.writerow(
            summary_row
        )

    print()
    print("Metrics JSON:", args.metrics_json)
    print("Summary CSV:", args.summary_csv)
    print()
    print("KEYPOINT EVALUATION PASSED")


if __name__ == "__main__":
    main()
