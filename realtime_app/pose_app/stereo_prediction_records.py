"""Convert saved detection prediction records into raw-fisheye InferenceResults.

This is the reusable input conversion shared by the strict stereo evaluation
CLI, the mainline export, and the PHC/viewer consumers. It parses saved
prediction JSON (``yolo26x_pose``, ``pmpose``/``bboxmaskpose``, ``probpose`` or
``sapiens2``) into :class:`~pose_app.schema.PersonPose` lists and wraps them in
an :class:`~pose_app.schema.InferenceResult`, then restores model-input
coordinates back to the original raw fisheye pixel convention.

No thresholds, association or triangulation live here; the strict gates stay in
``pose_app.triangulation``. Imports only pose_app modules.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from pose_app.rotation import model_image_size, restore_model_result_to_raw
from pose_app.schema import InferenceResult, PersonPose

COCO17 = 17
SAPIENS_TO_COCO17 = (0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14)


def _points(value):
    if not isinstance(value, list) or len(value) < COCO17:
        raise ValueError("Expected at least 17 [x, y, score] keypoints.")
    result = []
    for point in value[:COCO17]:
        if not isinstance(point, (list, tuple)) or len(point) < 3:
            raise ValueError("Invalid [x, y, score] keypoint.")
        result.append([float(point[0]), float(point[1]), float(point[2])])
    return result


def _person(person_id, bbox, bbox_score, points):
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        raise ValueError("Invalid bbox.")
    if not all(math.isfinite(v) for point in points for v in point):
        raise ValueError("Prediction contains non-finite keypoint values.")
    return PersonPose(
        person_id=int(person_id),
        bbox=[float(v) for v in bbox[:4]],
        bbox_score=float(bbox_score),
        pose_score=sum(point[2] for point in points) / COCO17,
        keypoints=points,
    )


def _match_input_region_score(output_box, input_boxes, input_scores, used):
    for index, box in enumerate(input_boxes):
        if index in used:
            continue
        if len(box) >= 4 and all(
            abs(float(a) - float(b)) <= 1e-6
            for a, b in zip(output_box[:4], box[:4])
        ):
            used.add(index)
            return float(input_scores[index])
    raise ValueError("An output bbox cannot be matched to an input prompt region.")


def _topdown_record(record, model, score_source):
    boxes = record["output_bboxes"]
    raw_points = record["keypoints"]
    input_boxes = record.get("input_regions_xyxy", record.get("boxes_from_yolo26x"))
    input_scores = record.get("input_region_scores", record.get("bbox_scores_from_yolo26x"))
    if input_boxes is None or input_scores is None:
        raise ValueError(f"{model}: missing input prompt regions or their scores")
    if len(boxes) != len(raw_points) or len(input_boxes) != len(input_scores):
        raise ValueError(f"{model}: misaligned fields for {record.get('file_name')}")

    used, persons = set(), []
    for index, (box, value) in enumerate(zip(boxes, raw_points)):
        points = _points(value)
        if score_source == "presence":
            presence = record["presence"][index]
            if len(presence) < COCO17:
                raise ValueError("Invalid presence field.")
            for joint in range(COCO17):
                points[joint][2] = float(presence[joint][0])

        persons.append(
            _person(
                index,
                box,
                _match_input_region_score(box, input_boxes, input_scores, used),
                points,
            )
        )
    return persons


def records(path, model, topdown_score_source):
    root = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    raw_records = root["frames"] if model == "yolo26x_pose" else root["images"]
    output = []

    for fallback, record in enumerate(raw_records):
        file_name = record["file_name"]
        frame_id = int(record.get("frame_index", record.get("image_id", fallback)))

        if model == "yolo26x_pose":
            persons = [
                _person(
                    item.get("person_id", index),
                    item["bbox_xyxy"],
                    item.get("bbox_score", 1.0),
                    _points(item["keypoints"]),
                )
                for index, item in enumerate(record.get("instances", []))
            ]

        elif model in {"pmpose", "bboxmaskpose"}:
            persons = _topdown_record(record, model, topdown_score_source)

        elif model == "probpose":
            persons = [
                _person(
                    index,
                    item.get("output_bbox_xyxy", item.get("input_region_xyxy", item.get("bbox_xyxy_from_yolo26x"))),
                    item.get(
                        "input_region_score",
                        item.get("bbox_score_from_yolo26x",
                            item.get("output_bbox_score", 1.0),
                        ),
                    ),
                    _points(item["keypoints_coco17"]),
                )
                for index, item in enumerate(record.get("instances", []))
            ]

        else:  # sapiens2
            persons = []
            for index, item in enumerate(record.get("instances", [])):
                xy, score = item["keypoints308"], item["keypoint_scores"]
                if len(xy) != 308 or len(score) != 308:
                    raise ValueError("Sapiens2 record must contain 308 keypoints and scores.")

                points = [
                    [float(xy[src][0]), float(xy[src][1]), float(score[src])]
                    for src in SAPIENS_TO_COCO17
                ]
                persons.append(
                    _person(
                        index,
                        item["bbox_xyxy_from_yolo26x"],
                        item.get("bbox_score_from_yolo26x", 1.0),
                        points,
                    )
                )

        elapsed = record.get("elapsed_ms_correctness_run")
        output.append(
            {
                "id": frame_id,
                "name": file_name,
                "elapsed": None if elapsed is None else float(elapsed),
                "persons": persons,
            }
        )

    if not output or len({item["name"] for item in output}) != len(output):
        raise ValueError(f"{model}: empty or duplicate file_name records.")
    return output


def result(frame, model, raw_size, rotation, timestamp):
    elapsed = frame["elapsed"]
    rotated_size = model_image_size(*raw_size, rotation)
    model_result = InferenceResult(
        source_frame_id=frame["id"],
        source_timestamp_sec=timestamp,
        image_width=rotated_size[0],
        image_height=rotated_size[1],
        model_name=model,
        model_ms=elapsed or 0.0,
        roundtrip_ms=0.0,
        persons=frame["persons"],
        stage_times_ms=(
            {} if elapsed is None
            else {"saved_prediction_elapsed_ms": elapsed}
        ),
    )
    return restore_model_result_to_raw(
        model_result,
        raw_width=raw_size[0],
        raw_height=raw_size[1],
        rotation=rotation,
    )
