from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from mmpose.evaluation.functional import oks_nms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw YOLO26x + ProbPose predictions to "
            "official-style COCO17 keypoint predictions."
        )
    )

    parser.add_argument("--raw-json", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--output-coco", required=True)
    parser.add_argument("--output-pre-nms", required=True)
    parser.add_argument("--output-post-nms", required=True)
    parser.add_argument("--summary-json", required=True)

    parser.add_argument(
        "--keypoint-score-thr",
        "--keypoint-prob-thr",
        dest="keypoint_score_thr",
        type=float,
        default=0.20,
        help=(
            "A keypoint participates in instance rescoring when "
            "keypoint_scores is greater than this value."
        ),
    )

    parser.add_argument(
        "--oks-nms-thr",
        type=float,
        default=0.90,
    )

    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


def bbox_xyxy_to_xywh(
    bbox_xyxy: list[float],
) -> tuple[list[float], float]:
    bbox = np.asarray(bbox_xyxy, dtype=np.float64).reshape(-1)

    if bbox.size < 4:
        raise ValueError(
            f"Expected an xyxy bbox with four values, got {bbox_xyxy}"
        )

    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height

    return [x1, y1, width, height], area


def validate_vector(
    value: Any,
    field_name: str,
    expected_length: int = 17,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)

    if array.size != expected_length:
        raise ValueError(
            f"{field_name} must contain {expected_length} values, "
            f"but got {array.size}"
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{field_name} contains NaN or infinite values"
        )

    return array


def validate_keypoints(
    value: Any,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)

    if array.shape != (17, 3):
        raise ValueError(
            "keypoints_coco17 must have shape (17, 3), "
            f"but got {array.shape}"
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            "keypoints_coco17 contains NaN or infinite values"
        )

    return array


def main() -> None:
    args = parse_args()

    raw_path = Path(args.raw_json)
    annotation_path = Path(args.annotation)

    output_coco_path = Path(args.output_coco)
    output_pre_nms_path = Path(args.output_pre_nms)
    output_post_nms_path = Path(args.output_post_nms)
    summary_path = Path(args.summary_json)

    raw = load_json(raw_path)
    annotation = load_json(annotation_path)

    annotation_images = annotation.get("images")

    if not isinstance(annotation_images, list):
        raise ValueError(
            "Annotation JSON does not contain an images list"
        )

    file_name_to_image_id: dict[str, int] = {}

    for image in annotation_images:
        file_name = Path(str(image["file_name"])).name
        image_id = int(image["id"])

        if file_name in file_name_to_image_id:
            raise ValueError(
                f"Duplicate file_name in annotation: {file_name}"
            )

        file_name_to_image_id[file_name] = image_id

    raw_images = raw.get("images")

    if not isinstance(raw_images, list):
        raise ValueError(
            "Raw ProbPose JSON does not contain an images list"
        )

    nms_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    pre_nms_records: list[dict[str, Any]] = []

    missing_file_names: list[str] = []
    image_id_mismatches: list[dict[str, Any]] = []

    total_declared_input_boxes = 0
    total_declared_outputs = 0
    total_raw_instances = 0

    for raw_image_index, image in enumerate(raw_images):
        file_name = Path(str(image["file_name"])).name

        if file_name not in file_name_to_image_id:
            missing_file_names.append(file_name)
            continue

        true_image_id = file_name_to_image_id[file_name]
        raw_image_id = image.get("image_id")

        if raw_image_id != true_image_id:
            image_id_mismatches.append({
                "file_name": file_name,
                "raw_image_id": raw_image_id,
                "true_image_id": true_image_id,
            })

        instances = image.get("instances", [])

        if not isinstance(instances, list):
            raise ValueError(
                f"instances is not a list for {file_name}"
            )

        total_declared_input_boxes += int(
            image.get("num_input_boxes", len(instances))
        )
        total_declared_outputs += int(
            image.get("num_output_instances", len(instances))
        )
        total_raw_instances += len(instances)

        for instance_index, instance in enumerate(instances):
            raw_keypoints = validate_keypoints(
                instance["keypoints_coco17"]
            )

            coordinates = raw_keypoints[:, :2]

            keypoint_scores = validate_vector(
                instance["keypoint_scores"],
                "keypoint_scores",
            )

            keypoint_probs = validate_vector(
                instance["keypoints_probs"],
                "keypoints_probs",
            )

            keypoints_visible = validate_vector(
                instance.get(
                    "keypoints_visible",
                    np.ones(17, dtype=np.float64),
                ),
                "keypoints_visible",
            )

            bbox_score = float(
                instance["bbox_score_from_yolo26x"]
            )

            bbox_xyxy = instance.get(
                "output_bbox_xyxy",
                instance["bbox_xyxy_from_yolo26x"],
            )

            bbox_xywh, bbox_area = bbox_xyxy_to_xywh(
                bbox_xyxy
            )

            official_nms_area = instance.get(
                "official_nms_area"
            )

            if official_nms_area is not None:
                candidate_area = float(
                    official_nms_area
                )
            else:
                candidate_area = float(
                    bbox_area
                )

            if (
                np.isfinite(candidate_area)
                and candidate_area > 0.0
            ):
                area = candidate_area
                area_source = (
                    "official_bbox_scales"
                    if official_nms_area is not None
                    else "bbox_xyxy"
                )
            else:
                area = float(bbox_area)
                area_source = "bbox_xyxy_fallback"

            valid_mask = (
                keypoint_scores > args.keypoint_score_thr
            )
            valid_num = int(valid_mask.sum())

            if valid_num > 0:
                mean_keypoint_score = float(
                    keypoint_scores[valid_mask].mean()
                )
            else:
                mean_keypoint_score = 0.0

            pose_score = float(
                bbox_score * mean_keypoint_score
            )

            # ProbPose official CocoMetric exports [x, y, probability].
            coco_keypoints_array = np.concatenate(
                [
                    coordinates,
                    keypoint_probs[:, None],
                ],
                axis=1,
            )

            coco_keypoints_flat = (
                coco_keypoints_array
                .reshape(-1)
                .astype(float)
                .tolist()
            )

            record = {
                "image_id": true_image_id,
                "category_id": 1,
                "file_name": file_name,
                "raw_image_id": raw_image_id,
                "raw_image_index": raw_image_index,
                "raw_instance_index": instance_index,
                "bbox": bbox_xywh,
                "bbox_xyxy": [
                    float(value)
                    for value in np.asarray(
                        bbox_xyxy,
                        dtype=np.float64,
                    ).reshape(-1)[:4]
                ],
                "bbox_area": float(bbox_area),
                "area": float(area),
                "area_source": area_source,
                "bbox_score": bbox_score,
                "keypoints": coco_keypoints_flat,
                "keypoint_scores": (
                    keypoint_scores
                    .astype(float)
                    .tolist()
                ),
                "keypoint_probs": (
                    keypoint_probs
                    .astype(float)
                    .tolist()
                ),
                "keypoints_visible": (
                    keypoints_visible
                    .astype(float)
                    .tolist()
                ),
                "num_valid_keypoints_for_score": valid_num,
                "mean_valid_keypoint_score": (
                    mean_keypoint_score
                ),
                "score": pose_score,
            }

            pre_nms_records.append(record)

            nms_groups[true_image_id].append({
                "score": pose_score,
                "keypoints": coco_keypoints_array,
                "area": float(area),
                "record": record,
            })

    if missing_file_names:
        raise ValueError(
            "The following prediction images were not found in "
            f"the annotation: {missing_file_names}"
        )

    post_nms_records: list[dict[str, Any]] = []
    post_nms_counts_by_image: dict[int, int] = {}

    for image_id, candidates in nms_groups.items():
        keep_indices = oks_nms(
            candidates,
            args.oks_nms_thr,
        )

        kept_records = [
            candidates[int(index)]["record"]
            for index in keep_indices
        ]

        post_nms_records.extend(kept_records)
        post_nms_counts_by_image[image_id] = len(
            kept_records
        )

    coco_predictions = []

    for record in post_nms_records:
        coco_predictions.append({
            "image_id": record["image_id"],
            "category_id": record["category_id"],
            "keypoints": record["keypoints"],
            "score": record["score"],
            "bbox": record["bbox"],
            "visibility": record["keypoints_visible"],
        })

    predictions_per_image = [
        post_nms_counts_by_image.get(
            file_name_to_image_id[
                Path(str(image["file_name"])).name
            ],
            0,
        )
        for image in raw_images
    ]

    images_over_20 = sum(
        count > 20
        for count in predictions_per_image
    )

    predictions_beyond_20 = sum(
        max(0, count - 20)
        for count in predictions_per_image
    )

    scores = np.asarray(
        [
            record["score"]
            for record in post_nms_records
        ],
        dtype=np.float64,
    )

    summary = {
        "raw_json": str(raw_path),
        "annotation": str(annotation_path),
        "raw_image_records": len(raw_images),
        "annotation_image_records": len(annotation_images),
        "declared_input_boxes": total_declared_input_boxes,
        "declared_output_instances": total_declared_outputs,
        "raw_instances_before_nms": total_raw_instances,
        "valid_instances_before_nms": len(
            pre_nms_records
        ),
        "official_area_instances_before_nms": sum(
            record["area_source"]
            == "official_bbox_scales"
            for record in pre_nms_records
        ),
        "bbox_fallback_area_instances_before_nms": sum(
            record["area_source"]
            != "official_bbox_scales"
            for record in pre_nms_records
        ),
        "instances_after_oks_nms": len(
            post_nms_records
        ),
        "instances_removed_by_oks_nms": (
            len(pre_nms_records)
            - len(post_nms_records)
        ),
        "image_id_mismatch_count": len(
            image_id_mismatches
        ),
        "image_id_mismatches": image_id_mismatches,
        "missing_file_names": missing_file_names,
        "keypoint_score_threshold": (
            args.keypoint_score_thr
        ),
        "oks_nms_threshold": args.oks_nms_thr,
        "instance_score_formula": (
            "bbox_score_from_yolo26x * mean("
            "keypoint_scores where keypoint_scores > "
            f"{args.keypoint_score_thr})"
        ),
        "coco_keypoint_third_value": "keypoints_probs",
        "zero_prediction_images_after_nms": sum(
            count == 0
            for count in predictions_per_image
        ),
        "max_predictions_per_image_after_nms": (
            max(predictions_per_image)
            if predictions_per_image
            else 0
        ),
        "images_over_20_predictions_after_nms": (
            images_over_20
        ),
        "predictions_beyond_top20_after_nms": (
            predictions_beyond_20
        ),
        "score_min_after_nms": (
            float(scores.min())
            if scores.size
            else None
        ),
        "score_mean_after_nms": (
            float(scores.mean())
            if scores.size
            else None
        ),
        "score_max_after_nms": (
            float(scores.max())
            if scores.size
            else None
        ),
    }

    save_json(
        output_pre_nms_path,
        pre_nms_records,
    )
    save_json(
        output_post_nms_path,
        post_nms_records,
    )
    save_json(
        output_coco_path,
        coco_predictions,
    )
    save_json(
        summary_path,
        summary,
    )

    print("=" * 80)
    print("ProbPose conversion completed")
    print("Raw images:", len(raw_images))
    print("Input boxes:", total_declared_input_boxes)
    print("Raw instances:", total_raw_instances)
    print("Before OKS NMS:", len(pre_nms_records))
    print("After OKS NMS:", len(post_nms_records))
    print(
        "Removed by OKS NMS:",
        len(pre_nms_records) - len(post_nms_records),
    )
    print(
        "Corrected image_id records:",
        len(image_id_mismatches),
    )
    print(
        "Keypoint score threshold:",
        args.keypoint_score_thr,
    )
    print(
        "OKS NMS threshold:",
        args.oks_nms_thr,
    )
    print("Saved COCO JSON:", output_coco_path)
    print(
        "Saved pre-NMS records:",
        output_pre_nms_path,
    )
    print(
        "Saved post-NMS records:",
        output_post_nms_path,
    )
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
