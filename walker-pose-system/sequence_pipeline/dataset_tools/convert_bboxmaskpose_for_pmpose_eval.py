import argparse
import json
from pathlib import Path

import numpy as np


FINAL_KEYPOINTS = 17
PMP_COMPAT_KEYPOINTS = 23


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert BBoxMaskPose raw predictions into "
            "the raw format accepted by "
            "evaluate_pmpose_coco.py."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--expected-images",
        type=int,
        default=2500,
    )

    parser.add_argument(
        "--expected-instances",
        type=int,
        default=11067,
    )

    return parser.parse_args()


def find_exact_input_index(
    output_box,
    input_boxes,
    used_indices,
):
    output_box = np.asarray(
        output_box,
        dtype=np.float64,
    ).reshape(4)

    for index, input_box in enumerate(input_boxes):
        if index in used_indices:
            continue

        input_box = np.asarray(
            input_box,
            dtype=np.float64,
        ).reshape(4)

        if np.array_equal(
            output_box,
            input_box,
        ):
            return index

    # 理论上不会进入这里；保留一个极小误差兜底。
    for index, input_box in enumerate(input_boxes):
        if index in used_indices:
            continue

        input_box = np.asarray(
            input_box,
            dtype=np.float64,
        ).reshape(4)

        if np.allclose(
            output_box,
            input_box,
            rtol=0.0,
            atol=1e-6,
        ):
            return index

    return None


def convert_record(record):
    input_boxes = np.asarray(
        record.get("boxes_from_yolo26x") or [],
        dtype=np.float64,
    ).reshape(-1, 4)

    input_scores = np.asarray(
        record.get("bbox_scores_from_yolo26x") or [],
        dtype=np.float64,
    ).reshape(-1)

    output_boxes = np.asarray(
        record.get("output_bboxes") or [],
        dtype=np.float64,
    ).reshape(-1, 4)

    keypoints = np.asarray(
        record.get("keypoints") or [],
        dtype=np.float64,
    )

    presence = np.asarray(
        record.get("presence") or [],
        dtype=np.float64,
    )

    visibility = np.asarray(
        record.get("visibility") or [],
        dtype=np.float64,
    )

    file_name = record.get(
        "file_name",
        "<unknown>",
    )

    if len(input_boxes) != len(input_scores):
        raise RuntimeError(
            f"{file_name}: input box/score mismatch: "
            f"{len(input_boxes)} vs {len(input_scores)}"
        )

    final_counts = {
        len(output_boxes),
        len(keypoints),
        len(presence),
        len(visibility),
    }

    if len(final_counts) != 1:
        raise RuntimeError(
            f"{file_name}: final instance fields "
            "are not aligned"
        )

    output_count = len(output_boxes)

    if output_count == 0:
        keypoints = np.zeros(
            (0, FINAL_KEYPOINTS, 3),
            dtype=np.float64,
        )

        presence = np.zeros(
            (0, FINAL_KEYPOINTS, 1),
            dtype=np.float64,
        )

        visibility = np.zeros(
            (0, FINAL_KEYPOINTS, 1),
            dtype=np.float64,
        )
    else:
        if keypoints.shape != (
            output_count,
            FINAL_KEYPOINTS,
            3,
        ):
            raise RuntimeError(
                f"{file_name}: unexpected keypoint "
                f"shape {keypoints.shape}"
            )

        if presence.reshape(
            output_count,
            -1,
        ).shape[1] != FINAL_KEYPOINTS:
            raise RuntimeError(
                f"{file_name}: unexpected presence "
                f"shape {presence.shape}"
            )

        if visibility.reshape(
            output_count,
            -1,
        ).shape[1] != FINAL_KEYPOINTS:
            raise RuntimeError(
                f"{file_name}: unexpected visibility "
                f"shape {visibility.shape}"
            )

        presence = presence.reshape(
            output_count,
            FINAL_KEYPOINTS,
            1,
        )

        visibility = visibility.reshape(
            output_count,
            FINAL_KEYPOINTS,
            1,
        )

    matched_scores = []
    used_input_indices = set()

    for output_box in output_boxes:
        matched_index = find_exact_input_index(
            output_box=output_box,
            input_boxes=input_boxes,
            used_indices=used_input_indices,
        )

        if matched_index is None:
            raise RuntimeError(
                f"{file_name}: output box has no "
                "matching YOLO input box"
            )

        used_input_indices.add(
            matched_index
        )

        matched_scores.append(
            float(input_scores[matched_index])
        )

    compatible_keypoints = np.zeros(
        (
            output_count,
            PMP_COMPAT_KEYPOINTS,
            3,
        ),
        dtype=np.float64,
    )

    compatible_presence = np.zeros(
        (
            output_count,
            PMP_COMPAT_KEYPOINTS,
            1,
        ),
        dtype=np.float64,
    )

    compatible_visibility = np.zeros(
        (
            output_count,
            PMP_COMPAT_KEYPOINTS,
            1,
        ),
        dtype=np.float64,
    )

    if output_count:
        presence_scores = presence[
            :,
            :FINAL_KEYPOINTS,
            0,
        ]

        compatible_keypoints[
            :,
            :FINAL_KEYPOINTS,
            :2,
        ] = keypoints[
            :,
            :FINAL_KEYPOINTS,
            :2,
        ]

        # 统一沿用PMPose评估口径：
        # 第三列使用presence概率。
        compatible_keypoints[
            :,
            :FINAL_KEYPOINTS,
            2,
        ] = presence_scores

        compatible_presence[
            :,
            :FINAL_KEYPOINTS,
            :,
        ] = presence[
            :,
            :FINAL_KEYPOINTS,
            :,
        ]

        compatible_visibility[
            :,
            :FINAL_KEYPOINTS,
            :,
        ] = visibility[
            :,
            :FINAL_KEYPOINTS,
            :,
        ]

    converted = dict(record)

    # 评估器要求这些字段实例数量一致。
    converted[
        "boxes_from_yolo26x"
    ] = output_boxes.tolist()

    converted[
        "bbox_scores_from_yolo26x"
    ] = matched_scores

    converted[
        "output_bboxes"
    ] = output_boxes.tolist()

    converted[
        "keypoints"
    ] = compatible_keypoints.tolist()

    converted[
        "presence"
    ] = compatible_presence.tolist()

    converted[
        "visibility"
    ] = compatible_visibility.tolist()

    converted[
        "bboxmaskpose_original_input_count"
    ] = int(len(input_boxes))

    converted[
        "bboxmaskpose_final_output_count"
    ] = int(output_count)

    return converted, output_count


def main():
    args = parse_args()

    with args.input.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        raw = json.load(file)

    raw_images = raw.get("images", [])

    if len(raw_images) != args.expected_images:
        raise RuntimeError(
            "Unexpected image count: "
            f"{len(raw_images)}"
        )

    converted_images = []
    total_instances = 0

    for record in raw_images:
        converted_record, count = (
            convert_record(record)
        )

        converted_images.append(
            converted_record
        )

        total_instances += count

    if total_instances != args.expected_instances:
        raise RuntimeError(
            "Unexpected final instance count: "
            f"{total_instances}"
        )

    converted_root = dict(raw)

    converted_root[
        "method"
    ] = (
        "YOLO26x + BBoxMaskPose "
        "(PMPose evaluator compatible)"
    )

    converted_root[
        "images"
    ] = converted_images

    converted_root[
        "evaluation_adapter"
    ] = {
        "source_method": (
            "YOLO26x + BBoxMaskPose(bmp_iters=1)"
        ),
        "final_keypoint_order": "COCO17",
        "compatible_keypoint_count": 23,
        "compatible_mapping": (
            "indices 0-16 contain COCO17; "
            "indices 17-22 are zero padding"
        ),
        "keypoint_confidence": (
            "BBoxMaskPose presence probability"
        ),
        "bbox_score_source": (
            "exactly matched original YOLO26x box"
        ),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            converted_root,
            file,
            ensure_ascii=False,
        )

    output_size_mb = (
        args.output.stat().st_size
        / 1024
        / 1024
    )

    print(
        "===== BBOXMASKPOSE EVALUATION ADAPTER ====="
    )
    print("Input images:", len(raw_images))
    print("Converted images:", len(converted_images))
    print("Final instances:", total_instances)
    print(
        "Keypoint structure:",
        "(N, 23, 3)",
    )
    print(
        "First 17 confidence values:",
        "presence",
    )
    print(
        "Trailing keypoints:",
        "6 zero-padded",
    )
    print(
        "BBox score source:",
        "exact matched YOLO26x score",
    )
    print(
        "Output size MB:",
        round(output_size_mb, 3),
    )
    print("Saved:", args.output)
    print()
    print("ADAPTER VALIDATION: PASSED")


if __name__ == "__main__":
    main()
