#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2


COCO17_EDGES = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize top-down pose predictions."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--pred-json", required=True)
    parser.add_argument("--vis-dir", required=True)
    parser.add_argument("--score-thr", type=float, default=0.20)
    parser.add_argument("--line-thickness", type=int, default=2)
    parser.add_argument("--point-radius", type=int, default=4)
    return parser.parse_args()


def first_value(data: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float))


def scalar_float(
    value: Any,
    default: float | None = None,
) -> float | None:
    """Convert numbers, singleton lists and score dictionaries to float."""

    while isinstance(value, (list, tuple)):
        if not value:
            return default
        value = value[0]

    if isinstance(value, dict):
        for key in (
            "score",
            "confidence",
            "probability",
            "value",
            "data",
        ):
            if key in value:
                return scalar_float(value[key], default)

        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_frames(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if not isinstance(payload, dict):
        raise ValueError("Prediction JSON must contain an object or list.")

    for key in (
        "frames",
        "images",
        "results",
        "predictions",
        "items",
        "data",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    raise ValueError(
        "Cannot find frame list. Top-level keys: "
        + ", ".join(payload.keys())
    )


def get_file_name(frame: dict[str, Any]) -> str:
    value = first_value(
        frame,
        [
            "file_name",
            "filename",
            "image_name",
            "image_file",
            "image_path",
            "source_image",
            "path",
            "image",
        ],
    )

    if value is None:
        raise ValueError(
            "Cannot find image filename. Frame keys: "
            + ", ".join(frame.keys())
        )

    return Path(str(value)).name


def normalize_bbox(value: Any) -> list[float] | None:
    if value is None:
        return None

    if isinstance(value, dict):
        if all(k in value for k in ("x1", "y1", "x2", "y2")):
            return [
                float(value["x1"]),
                float(value["y1"]),
                float(value["x2"]),
                float(value["y2"]),
            ]

        if all(k in value for k in ("x", "y", "w", "h")):
            x = float(value["x"])
            y = float(value["y"])
            w = float(value["w"])
            h = float(value["h"])
            return [x, y, x + w, y + h]

        for key in ("bbox_xyxy", "bbox", "box", "data"):
            if key in value:
                return normalize_bbox(value[key])

    if isinstance(value, list) and len(value) >= 4:
        return [float(v) for v in value[:4]]

    return None


def keypoint_depth(value: Any) -> int:
    depth = 0
    current = value

    while isinstance(current, list) and current:
        depth += 1
        current = current[0]

    return depth


def split_frame_level_instances(
    frame: dict[str, Any],
) -> list[dict[str, Any]]:
    source = frame.get("pred_instances")

    if not isinstance(source, dict):
        source = frame

    keypoints = first_value(
        source,
        [
            "keypoints",
            "keypoints_xy",
            "pred_keypoints",
            "pose_keypoints",
        ],
    )

    if not isinstance(keypoints, list) or not keypoints:
        return []

    depth = keypoint_depth(keypoints)

    if depth == 2:
        keypoints = [keypoints]
    elif depth < 2:
        return []

    scores = first_value(
        source,
        [
            "keypoint_scores",
            "keypoints_score",
            "scores",
            "keypoint_confidence",
        ],
    )

    visibility = first_value(
        source,
        [
            "keypoint_visibility",
            "visibility",
            "visible",
        ],
    )

    presence = first_value(
        source,
        [
            "keypoint_presence",
            "presence",
        ],
    )

    boxes = first_value(
        source,
        [
            "bboxes",
            "boxes",
            "bbox_xyxy",
            "bbox",
        ],
    )

    if boxes is not None and (
        not isinstance(boxes, list)
        or not boxes
        or is_number(boxes[0])
    ):
        boxes = [boxes]

    instances = []

    for person_index, person_keypoints in enumerate(keypoints):
        instance: dict[str, Any] = {
            "person_id": person_index,
            "keypoints": person_keypoints,
        }

        if (
            isinstance(scores, list)
            and scores
            and isinstance(scores[0], list)
            and person_index < len(scores)
        ):
            instance["keypoint_scores"] = scores[person_index]
        elif isinstance(scores, list):
            instance["keypoint_scores"] = scores

        if (
            isinstance(visibility, list)
            and visibility
            and isinstance(visibility[0], list)
            and person_index < len(visibility)
        ):
            instance["keypoint_visibility"] = visibility[person_index]

        if (
            isinstance(presence, list)
            and presence
            and isinstance(presence[0], list)
            and person_index < len(presence)
        ):
            instance["keypoint_presence"] = presence[person_index]

        if isinstance(boxes, list) and person_index < len(boxes):
            instance["bbox_xyxy"] = boxes[person_index]

        instances.append(instance)

    return instances


def get_instances(frame: dict[str, Any]) -> list[dict[str, Any]]:
    for key in (
        "instances",
        "people",
        "persons",
        "outputs",
        "detections",
    ):
        value = frame.get(key)
        if isinstance(value, list):
            return [
                item
                for item in value
                if isinstance(item, dict)
            ]

    predictions = frame.get("predictions")
    if isinstance(predictions, list):
        return [
            item
            for item in predictions
            if isinstance(item, dict)
        ]

    return split_frame_level_instances(frame)


def normalize_keypoints(
    instance: dict[str, Any],
) -> list[tuple[float, float, float]]:
    raw = first_value(
        instance,
        [
            "keypoints",
            "keypoints_xy",
            "pred_keypoints",
            "pose_keypoints",
        ],
    )

    if isinstance(raw, dict):
        raw = first_value(
            raw,
            ["data", "keypoints", "coordinates", "xy"],
        )

    if not isinstance(raw, list):
        return []

    scores = first_value(
        instance,
        [
            "keypoint_scores",
            "keypoints_score",
            "scores",
            "keypoint_confidence",
        ],
    )

    visibility = first_value(
        instance,
        [
            "keypoint_visibility",
            "visibility",
            "visible",
        ],
    )

    presence = first_value(
        instance,
        [
            "keypoint_presence",
            "presence",
        ],
    )

    if raw and is_number(raw[0]):
        if len(raw) % 3 == 0:
            raw = [
                raw[index:index + 3]
                for index in range(0, len(raw), 3)
            ]
        elif len(raw) % 2 == 0:
            raw = [
                raw[index:index + 2]
                for index in range(0, len(raw), 2)
            ]

    output: list[tuple[float, float, float]] = []

    for index, point in enumerate(raw):
        if isinstance(point, dict):
            x = first_value(point, ["x", "X"])
            y = first_value(point, ["y", "Y"])
            score = first_value(
                point,
                ["score", "confidence", "probability"],
            )
        elif isinstance(point, list) and len(point) >= 2:
            x = point[0]
            y = point[1]
            score = point[2] if len(point) >= 3 else None
        else:
            continue

        if x is None or y is None:
            continue

        if score is None and isinstance(scores, list):
            if index < len(scores):
                score = scalar_float(scores[index])

        if score is None and isinstance(visibility, list):
            if index < len(visibility):
                score = scalar_float(visibility[index])

        score = scalar_float(score)

        if (
            isinstance(presence, list)
            and index < len(presence)
        ):
            p = scalar_float(presence[index])

            if p is not None:
                if score is None:
                    score = p
                else:
                    score = min(score, p)

        if score is None:
            score = 1.0

        output.append(
            (float(x), float(y), float(score))
        )

    return output


def draw_instance(
    image,
    instance: dict[str, Any],
    person_index: int,
    score_thr: float,
    line_thickness: int,
    point_radius: int,
) -> None:
    bbox = normalize_bbox(
        first_value(
            instance,
            [
                "bbox_xyxy",
                "bbox",
                "box",
                "bboxes",
                "detection_box",
            ],
        )
    )

    if bbox is not None:
        x1, y1, x2, y2 = [
            int(round(value))
            for value in bbox
        ]

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            line_thickness,
        )

        cv2.putText(
            image,
            f"person {person_index}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    keypoints = normalize_keypoints(instance)

    for start, end in COCO17_EDGES:
        if start >= len(keypoints) or end >= len(keypoints):
            continue

        x1, y1, score1 = keypoints[start]
        x2, y2, score2 = keypoints[end]

        if score1 < score_thr or score2 < score_thr:
            continue

        cv2.line(
            image,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            (0, 255, 0),
            line_thickness,
            cv2.LINE_AA,
        )

    for x, y, score in keypoints:
        if score < score_thr:
            continue

        cv2.circle(
            image,
            (int(round(x)), int(round(y))),
            point_radius,
            (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    pred_path = Path(args.pred_json)
    vis_dir = Path(args.vis_dir)

    payload = json.loads(
        pred_path.read_text(encoding="utf-8-sig")
    )

    frames = get_frames(payload)

    vis_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0

    for frame_index, frame in enumerate(frames):
        try:
            file_name = get_file_name(frame)
        except ValueError as exc:
            print(
                f"[SKIP {frame_index}] {exc}"
            )
            skipped += 1
            continue

        image_path = input_dir / file_name
        image = cv2.imread(str(image_path))

        if image is None:
            print(
                f"[SKIP {frame_index}] cannot read: "
                f"{image_path}"
            )
            skipped += 1
            continue

        instances = get_instances(frame)

        for person_index, instance in enumerate(instances):
            draw_instance(
                image=image,
                instance=instance,
                person_index=person_index,
                score_thr=args.score_thr,
                line_thickness=args.line_thickness,
                point_radius=args.point_radius,
            )

        output_path = vis_dir / (
            Path(file_name).stem + ".jpg"
        )

        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(
                f"Failed to save visualization: {output_path}"
            )

        print(
            f"[{frame_index + 1}/{len(frames)}] "
            f"{file_name}: instances={len(instances)}"
        )
        saved += 1

    print("Visualization complete.")
    print("Saved:", saved)
    print("Skipped:", skipped)
    print("Directory:", vis_dir)


if __name__ == "__main__":
    main()
