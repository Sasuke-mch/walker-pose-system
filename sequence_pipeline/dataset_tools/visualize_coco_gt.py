import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_image(path: Path) -> np.ndarray:
    """Read an image from a Windows path, including Chinese paths."""
    data = np.fromfile(str(path), dtype=np.uint8)

    if data.size == 0:
        raise RuntimeError(f"Cannot read image bytes: {path}")

    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"Cannot decode image: {path}")

    return image


def write_image(path: Path, image: np.ndarray) -> None:
    """Write an image reliably to a Windows path."""
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()

    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".jpg"

    success, encoded = cv2.imencode(suffix, image)

    if not success:
        raise RuntimeError(f"Cannot encode image: {path}")

    encoded.tofile(str(path))


def draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float = 0.5,
) -> None:
    thickness = 1

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )

    top = max(0, y - text_height - 6)

    cv2.rectangle(
        image,
        (x, top),
        (x + text_width + 6, y + baseline + 2),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        image,
        text,
        (x + 3, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_person(
    image: np.ndarray,
    annotation: dict[str, Any],
    skeleton: list[tuple[int, int]],
    person_number: int,
) -> None:
    height, width = image.shape[:2]
    base_size = min(height, width)

    point_radius = max(3, round(base_size / 250))
    line_thickness = max(2, round(base_size / 450))
    box_thickness = max(2, round(base_size / 400))

    bbox = annotation.get("bbox", [])

    if len(bbox) == 4:
        x, y, box_width, box_height = map(float, bbox)

        x1 = max(0, round(x))
        y1 = max(0, round(y))
        x2 = min(width - 1, round(x + box_width))
        y2 = min(height - 1, round(y + box_height))

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            box_thickness,
        )

        draw_label(
            image,
            f"GT person {person_number}",
            x1,
            max(18, y1),
        )

    flat_keypoints = annotation.get("keypoints", [])

    if len(flat_keypoints) != 51:
        return

    keypoints: list[tuple[float, float, int]] = []

    for index in range(0, 51, 3):
        point_x = float(flat_keypoints[index])
        point_y = float(flat_keypoints[index + 1])
        visibility = int(flat_keypoints[index + 2])

        keypoints.append(
            (point_x, point_y, visibility)
        )

    # Draw skeleton before drawing points.
    for first_index, second_index in skeleton:
        if first_index >= len(keypoints) or second_index >= len(keypoints):
            continue

        x1, y1, visibility1 = keypoints[first_index]
        x2, y2, visibility2 = keypoints[second_index]

        if visibility1 <= 0 or visibility2 <= 0:
            continue

        cv2.line(
            image,
            (round(x1), round(y1)),
            (round(x2), round(y2)),
            (255, 0, 255),
            line_thickness,
            cv2.LINE_AA,
        )

    for point_x, point_y, visibility in keypoints:
        if visibility <= 0:
            continue

        # COCO visibility:
        # 1 = labeled but occluded
        # 2 = labeled and visible
        color = (
            (0, 165, 255)
            if visibility == 1
            else (0, 255, 0)
        )

        center = (
            round(point_x),
            round(point_y),
        )

        cv2.circle(
            image,
            center,
            point_radius,
            color,
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            image,
            center,
            point_radius,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize COCO ground-truth keypoints."
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="0 means all images.",
    )

    args = parser.parse_args()

    if not args.images.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {args.images}"
        )

    if not args.annotation.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: {args.annotation}"
        )

    with args.annotation.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        coco = json.load(file)

    images = sorted(
        coco.get("images", []),
        key=lambda item: item["file_name"],
    )

    if args.max_images > 0:
        images = images[: args.max_images]

    annotations_by_image: dict[int, list[dict[str, Any]]] = (
        defaultdict(list)
    )

    for annotation in coco.get("annotations", []):
        image_id = int(annotation["image_id"])
        annotations_by_image[image_id].append(annotation)

    categories = coco.get("categories", [])

    if not categories:
        raise ValueError("No categories found in COCO annotation")

    category = categories[0]
    keypoint_names = category.get("keypoints", [])
    raw_skeleton = category.get("skeleton", [])

    if len(keypoint_names) != 17:
        raise ValueError(
            f"Expected 17 keypoints, found {len(keypoint_names)}"
        )

    # COCO skeleton indices are 1-based.
    skeleton = [
        (
            int(edge[0]) - 1,
            int(edge[1]) - 1,
        )
        for edge in raw_skeleton
        if len(edge) == 2
    ]

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_count = 0
    missing_count = 0

    for index, image_record in enumerate(images, start=1):
        image_id = int(image_record["id"])
        file_name = image_record["file_name"]

        input_path = args.images / file_name
        output_path = args.output / file_name

        if not input_path.is_file():
            print(
                f"[{index}/{len(images)}] MISSING: {file_name}"
            )
            missing_count += 1
            continue

        image = read_image(input_path)

        person_annotations = annotations_by_image.get(
            image_id,
            [],
        )

        for person_number, annotation in enumerate(
            person_annotations,
            start=1,
        ):
            draw_person(
                image=image,
                annotation=annotation,
                skeleton=skeleton,
                person_number=person_number,
            )

        draw_label(
            image,
            (
                f"GT | image_id={image_id} | "
                f"persons={len(person_annotations)}"
            ),
            10,
            25,
            0.55,
        )

        write_image(output_path, image)

        print(
            f"[{index}/{len(images)}] "
            f"{file_name}: persons={len(person_annotations)}"
        )

        saved_count += 1

    print()
    print("GT visualization completed.")
    print("Saved:", saved_count)
    print("Missing:", missing_count)
    print("Output:", args.output)

    if missing_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
