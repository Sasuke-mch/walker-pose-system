import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax, ay, aw, ah = map(float, box_a)
    bx, by, bw, bh = map(float, box_b)

    ax2 = ax + max(0.0, aw)
    ay2 = ay + max(0.0, ah)
    bx2 = bx + max(0.0, bw)
    by2 = by + max(0.0, bh)

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def calculate_image_metrics(
    annotations: list[dict[str, Any]],
) -> dict[str, float]:
    person_count = len(annotations)

    boxes = [
        annotation.get("bbox", [0, 0, 0, 0])
        for annotation in annotations
    ]

    pairwise_ious: list[float] = []

    for left_index in range(len(boxes)):
        for right_index in range(left_index + 1, len(boxes)):
            pairwise_ious.append(
                bbox_iou(
                    boxes[left_index],
                    boxes[right_index],
                )
            )

    mean_iou = (
        sum(pairwise_ious) / len(pairwise_ious)
        if pairwise_ious
        else 0.0
    )

    max_iou = max(pairwise_ious, default=0.0)

    total_keypoints = 0
    labeled_keypoints = 0
    visible_keypoints = 0
    occluded_keypoints = 0
    missing_keypoints = 0

    for annotation in annotations:
        keypoints = annotation.get("keypoints", [])

        for index in range(2, len(keypoints), 3):
            visibility = int(keypoints[index])
            total_keypoints += 1

            if visibility > 0:
                labeled_keypoints += 1

            if visibility == 2:
                visible_keypoints += 1
            elif visibility == 1:
                occluded_keypoints += 1
            else:
                missing_keypoints += 1

    if labeled_keypoints > 0:
        occluded_ratio = occluded_keypoints / labeled_keypoints
    else:
        occluded_ratio = 1.0

    if total_keypoints > 0:
        missing_ratio = missing_keypoints / total_keypoints
        visible_ratio = visible_keypoints / total_keypoints
    else:
        missing_ratio = 1.0
        visible_ratio = 0.0

    # This score is only for selecting a difficult debugging subset.
    # It is not an evaluation metric.
    difficulty_score = (
        3.0 * math.log1p(person_count)
        + 4.0 * mean_iou
        + 2.0 * max_iou
        + 2.0 * occluded_ratio
        + 1.5 * missing_ratio
    )

    return {
        "person_count": float(person_count),
        "mean_bbox_iou": mean_iou,
        "max_bbox_iou": max_iou,
        "visible_ratio": visible_ratio,
        "occluded_ratio": occluded_ratio,
        "missing_ratio": missing_ratio,
        "difficulty_score": difficulty_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic difficult subset from "
            "OCHuman-Pose validation annotations."
        )
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
    )
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be greater than zero")

    if not args.annotation.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: {args.annotation}"
        )

    with args.annotation.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        coco = json.load(file)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    if not images:
        raise ValueError("No images found in annotation file")

    annotations_by_image: dict[int, list[dict[str, Any]]] = (
        defaultdict(list)
    )

    for annotation in annotations:
        image_id = int(annotation["image_id"])
        annotations_by_image[image_id].append(annotation)

    ranked_images: list[dict[str, Any]] = []

    for image in images:
        image_id = int(image["id"])
        image_annotations = annotations_by_image.get(image_id, [])

        metrics = calculate_image_metrics(image_annotations)

        ranked_images.append(
            {
                "image_id": image_id,
                "file_name": image["file_name"],
                "width": image.get("width"),
                "height": image.get("height"),
                **metrics,
            }
        )

    ranked_images.sort(
        key=lambda item: (
            item["difficulty_score"],
            item["person_count"],
            item["max_bbox_iou"],
            -item["image_id"],
        ),
        reverse=True,
    )

    selected = ranked_images[: min(args.count, len(ranked_images))]
    selected_ids = {
        int(item["image_id"])
        for item in selected
    }

    subset_images = [
        image
        for image in images
        if int(image["id"]) in selected_ids
    ]

    subset_annotations = [
        annotation
        for annotation in annotations
        if int(annotation["image_id"]) in selected_ids
    ]

    subset_coco: dict[str, Any] = {}

    for key in ("info", "licenses"):
        if key in coco:
            subset_coco[key] = coco[key]

    subset_coco["images"] = subset_images
    subset_coco["annotations"] = subset_annotations
    subset_coco["categories"] = coco.get("categories", [])

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    annotation_output = (
        args.output_dir / "val_hard50_annotations.json"
    )
    manifest_output = (
        args.output_dir / "val_hard50_manifest.json"
    )
    filenames_output = (
        args.output_dir / "val_hard50_filenames.txt"
    )

    with annotation_output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            subset_coco,
            file,
            ensure_ascii=False,
            indent=2,
        )

    manifest = {
        "source_annotation": str(args.annotation),
        "selection_purpose": (
            "Difficult debugging subset only; "
            "not the final benchmark."
        ),
        "selection_count": len(selected),
        "ranking_features": [
            "person_count",
            "mean_bbox_iou",
            "max_bbox_iou",
            "occluded_ratio",
            "missing_ratio",
        ],
        "selected_images": [
            {
                "rank": rank,
                **item,
            }
            for rank, item in enumerate(selected, start=1)
        ],
    }

    with manifest_output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    with filenames_output.open(
        "w",
        encoding="utf-8",
    ) as file:
        for item in selected:
            file.write(f'{item["file_name"]}\n')

    print("Subset creation completed.")
    print(f"Selected images: {len(subset_images)}")
    print(f"Selected annotations: {len(subset_annotations)}")
    print(f"Annotation output: {annotation_output}")
    print(f"Manifest output: {manifest_output}")
    print(f"Filename list: {filenames_output}")

    print()
    print("Top 10 difficult images:")

    for rank, item in enumerate(selected[:10], start=1):
        print(
            f'{rank:02d}. {item["file_name"]} | '
            f'persons={int(item["person_count"])} | '
            f'mean_iou={item["mean_bbox_iou"]:.3f} | '
            f'max_iou={item["max_bbox_iou"]:.3f} | '
            f'occluded={item["occluded_ratio"]:.3f} | '
            f'missing={item["missing_ratio"]:.3f} | '
            f'score={item["difficulty_score"]:.3f}'
        )


if __name__ == "__main__":
    main()
