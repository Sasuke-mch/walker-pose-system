"""Replay sequence_pipeline raw predictions through Walker stereo geometry."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_app.calibration import StereoCalibration
from pose_app.rotation import ROTATION_CHOICES, model_image_size, restore_model_result_to_raw
from pose_app.schema import InferenceResult, PersonPose
from pose_app.triangulation import triangulate_matches

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


def _match_yolo_score(output_box, input_boxes, input_scores, used):
    for index, box in enumerate(input_boxes):
        if index in used:
            continue
        if len(box) >= 4 and all(
            abs(float(a) - float(b)) <= 1e-6
            for a, b in zip(output_box[:4], box[:4])
        ):
            used.add(index)
            return float(input_scores[index])
    raise ValueError("An output bbox cannot be matched to its YOLO input bbox.")


def _topdown_record(record, model, score_source):
    boxes = record["output_bboxes"]
    raw_points = record["keypoints"]
    input_boxes = record["boxes_from_yolo26x"]
    input_scores = record["bbox_scores_from_yolo26x"]
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
                _match_yolo_score(box, input_boxes, input_scores, used),
                points,
            )
        )
    return persons


def _records(path, model, topdown_score_source):
    root = json.loads(path.read_text(encoding="utf-8-sig"))
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
                    item.get("output_bbox_xyxy", item["bbox_xyxy_from_yolo26x"]),
                    item.get(
                        "bbox_score_from_yolo26x",
                        item.get("output_bbox_score", 1.0),
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


def _result(frame, model, raw_size, rotation, timestamp):
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


def main():
    parser = argparse.ArgumentParser(
        description="Offline saved-prediction stereo geometry replay."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=("yolo26x_pose", "pmpose", "probpose", "bboxmaskpose", "sapiens2"),
    )
    parser.add_argument("--left-json", required=True, type=Path)
    parser.add_argument("--right-json", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--left-model-rotation", choices=ROTATION_CHOICES, default="ccw90")
    parser.add_argument("--right-model-rotation", choices=ROTATION_CHOICES, default="cw90")
    parser.add_argument("--keypoint-threshold", type=float, default=0.25)
    parser.add_argument("--max-association-cost", type=float, default=0.05)
    parser.add_argument("--max-reprojection-error-px", type=float, default=10.0)
    parser.add_argument(
        "--topdown-score-source",
        choices=("raw_keypoint", "presence"),
        default="raw_keypoint",
    )
    args = parser.parse_args()

    if args.fps <= 0 or not 0 <= args.keypoint_threshold <= 1:
        parser.error("fps must be positive and keypoint threshold must be in [0, 1].")

    started = time.perf_counter()
    args.left_json = args.left_json.resolve()
    args.right_json = args.right_json.resolve()
    args.calibration = args.calibration.resolve()
    args.output_dir = args.output_dir.resolve()

    cwd = Path.cwd()
    try:
        os.chdir(ROOT)
        source_calibration = StereoCalibration.load(args.calibration)
    finally:
        os.chdir(cwd)

    calibration = source_calibration.for_runtime_sizes(
        source_calibration.left_image_size,
        source_calibration.right_image_size,
    )

    left = _records(args.left_json, args.model, args.topdown_score_source)
    right = _records(args.right_json, args.model, args.topdown_score_source)
    left_by_name = {item["name"]: item for item in left}
    right_by_name = {item["name"]: item for item in right}

    if set(left_by_name) != set(right_by_name):
        raise RuntimeError("Left/right prediction file names do not match exactly.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "offline_stereo_results.jsonl"

    valid = 0
    valid_by_name = Counter()
    reasons = Counter()
    costs, common = [], []
    matched_pairs, matched_persons = 0, 0

    with results_path.open("w", encoding="utf-8") as handle:
        for pair_id, name in enumerate(item["name"] for item in left):
            timestamp = pair_id / args.fps

            left_raw = _result(
                left_by_name[name],
                args.model,
                calibration.left_image_size,
                args.left_model_rotation,
                timestamp,
            )
            right_raw = _result(
                right_by_name[name],
                args.model,
                calibration.right_image_size,
                args.right_model_rotation,
                timestamp,
            )

            people3d = triangulate_matches(
                left_raw.persons,
                right_raw.persons,
                calibration,
                args.keypoint_threshold,
                args.max_association_cost,
                args.max_reprojection_error_px,
            )

            matched_pairs += bool(people3d)
            matched_persons += len(people3d)

            for person in people3d:
                valid += person.valid_keypoints
                costs.append(person.association_cost)
                common.append(person.common_keypoints)

                for point in person.keypoints_3d:
                    if point["valid"]:
                        valid_by_name[point["name"]] += 1
                    else:
                        reasons[point["reason"]] += 1

            payload = {
                "pair_id": pair_id,
                "file_name": name,
                "pair_timestamp_sec": timestamp,
                "left_frame_id": left_raw.source_frame_id,
                "right_frame_id": right_raw.source_frame_id,
                "timestamp_skew_ms": 0.0,
                "timestamp_type": "sequence_file_index_over_fps",
                "coordinate_frame": "left_camera",
                "length_unit": calibration.length_unit,
                "left": left_raw.to_dict(),
                "right": right_raw.to_dict(),
                "persons_3d": [person.to_dict() for person in people3d],
            }
            handle.write(
                json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
            )

    count = len(left)
    summary = {
        "source": f"offline-predictions:{args.model}",
        "processed_pairs": count,
        "wall_time_sec": time.perf_counter() - started,
        "model": args.model,
        "left_prediction_json": str(args.left_json),
        "right_prediction_json": str(args.right_json),
        "calibration_file": str(args.calibration),
        "camera_model": calibration.camera_model,
        "baseline": calibration.baseline,
        "length_unit": calibration.length_unit,
        "coordinate_frame": "left_camera",
        "keypoint_threshold": args.keypoint_threshold,
        "max_association_cost": args.max_association_cost,
        "max_reprojection_error_px": args.max_reprojection_error_px,
        "model_input_rotation": {
            "left": args.left_model_rotation,
            "right": args.right_model_rotation,
            "result_coordinate_space": "raw_camera_pixels_after_inverse_rotation",
        },
        "topdown_score_source": args.topdown_score_source,
        "matched_pairs": matched_pairs,
        "matched_person_pairs": matched_persons,
        "mean_association_cost": sum(costs) / len(costs) if costs else None,
        "mean_common_keypoints_per_matched_person": (
            sum(common) / len(common) if common else None
        ),
        "total_valid_3d_keypoints": valid,
        "mean_valid_3d_keypoints_per_pair": valid / count,
        "valid_3d_keypoints_by_name": dict(valid_by_name),
        "matched_keypoint_rejection_reasons": dict(reasons),
        "results_jsonl": str(results_path),
        "annotated_video": None,
        "accuracy_note": (
            "Geometric self-consistency only; no human ground truth is used."
        ),
    }

    summary_path = args.output_dir / "offline_stereo_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
