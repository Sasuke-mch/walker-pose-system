import argparse
import json
from pathlib import Path

import numpy as np


MAPPING = [
    0, 1, 2, 3, 4,
    5, 6, 7, 8,
    62, 41,
    9, 10, 11, 12, 13, 14,
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        source = json.load(file)

    images = source.get("images")

    if not isinstance(images, list):
        raise RuntimeError(
            "输入JSON不存在images列表"
        )

    if len(images) != 2500:
        raise RuntimeError(
            f"应有2500张图，实际为{len(images)}"
        )

    mapping = np.asarray(
        MAPPING,
        dtype=np.int64,
    )

    output_images = []

    total_input = 0
    total_output = 0
    score_over_one = 0
    all_scores = []

    for image_index, image in enumerate(images):
        instances = image.get(
            "instances",
            [],
        )

        declared_input = int(
            image.get(
                "num_input_boxes",
                len(instances),
            )
        )

        declared_output = int(
            image.get(
                "num_output_instances",
                len(instances),
            )
        )

        if declared_output != len(instances):
            raise RuntimeError(
                f"第{image_index}张图实例数量不一致"
            )

        total_input += declared_input
        total_output += declared_output

        converted = []

        for instance_index, instance in enumerate(instances):
            bbox = np.asarray(
                instance[
                    "bbox_xyxy_from_yolo26x"
                ],
                dtype=np.float64,
            ).reshape(-1)

            bbox_score = float(
                instance[
                    "bbox_score_from_yolo26x"
                ]
            )

            keypoints308 = np.asarray(
                instance["keypoints308"],
                dtype=np.float64,
            )

            scores308 = np.asarray(
                instance["keypoint_scores"],
                dtype=np.float64,
            ).reshape(-1)

            if bbox.shape != (4,):
                raise RuntimeError(
                    f"图{image_index}实例"
                    f"{instance_index}框形状错误："
                    f"{bbox.shape}"
                )

            if keypoints308.shape != (308, 2):
                raise RuntimeError(
                    f"图{image_index}实例"
                    f"{instance_index}关键点形状错误："
                    f"{keypoints308.shape}"
                )

            if scores308.shape != (308,):
                raise RuntimeError(
                    f"图{image_index}实例"
                    f"{instance_index}分数形状错误："
                    f"{scores308.shape}"
                )

            if not (
                np.all(np.isfinite(bbox))
                and np.isfinite(bbox_score)
                and np.all(np.isfinite(keypoints308))
                and np.all(np.isfinite(scores308))
            ):
                raise RuntimeError(
                    f"图{image_index}实例"
                    f"{instance_index}存在非有限数值"
                )

            xy = keypoints308[
                mapping,
                :2,
            ]

            scores = scores308[
                mapping
            ]

            all_scores.append(scores)
            score_over_one += int(
                np.sum(scores > 1.0)
            )

            keypoints17 = np.concatenate(
                [
                    xy,
                    scores[:, None],
                ],
                axis=1,
            )

            x1, y1, x2, y2 = bbox.tolist()

            width = max(
                0.0,
                x2 - x1,
            )

            height = max(
                0.0,
                y2 - y1,
            )

            converted.append(
                {
                    "bbox_xyxy_from_yolo26x": (
                        bbox.tolist()
                    ),
                    "bbox_score_from_yolo26x": (
                        bbox_score
                    ),
                    "output_bbox_xyxy": (
                        bbox.tolist()
                    ),
                    "output_bbox_score": 1.0,
                    "keypoints_coco17": (
                        keypoints17.tolist()
                    ),
                    "keypoint_scores": (
                        scores.tolist()
                    ),
                    "keypoints_probs": (
                        scores.tolist()
                    ),
                    "keypoints_visible": (
                        [1.0] * 17
                    ),
                    "official_nms_area": (
                        width * height
                    ),
                }
            )

        output_images.append(
            {
                "image_id": image.get(
                    "image_id"
                ),
                "file_name": image.get(
                    "file_name"
                ),
                "image_path": image.get(
                    "image_path"
                ),
                "num_input_boxes": (
                    declared_input
                ),
                "num_output_instances": (
                    len(converted)
                ),
                "elapsed_ms_correctness_run": (
                    image.get(
                        "elapsed_ms_correctness_run"
                    )
                ),
                "instances": converted,
            }
        )

    if total_input != 28545:
        raise RuntimeError(
            f"输入框应为28545，实际为{total_input}"
        )

    if total_output != 28545:
        raise RuntimeError(
            f"输出实例应为28545，实际为{total_output}"
        )

    all_scores = np.concatenate(
        all_scores
    )

    result = {
        "method": (
            "YOLO26x + Sapiens2-0.4B "
            "COCO17 adapted"
        ),
        "source_raw_json": str(
            input_path
        ),
        "source_method": source.get(
            "method"
        ),
        "flip_test": source.get(
            "flip_test"
        ),
        "micro_batch_size": source.get(
            "micro_batch_size"
        ),
        "source_num_keypoints": source.get(
            "num_keypoints"
        ),
        "coco17_from_sapiens308": MAPPING,
        "score_description": (
            "UDPHeatmap decoded response, "
            "not clipped to [0,1]"
        ),
        "images": output_images,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
        )

    print(
        "===== SAPIENS2 COCO17 ADAPTER ====="
    )
    print("images:", len(output_images))
    print("input_boxes:", total_input)
    print("output_instances:", total_output)
    print(
        "score_min:",
        float(all_scores.min()),
    )
    print(
        "score_mean:",
        float(all_scores.mean()),
    )
    print(
        "score_max:",
        float(all_scores.max()),
    )
    print(
        "scores_over_one:",
        score_over_one,
    )
    print("output:", output_path)
    print()
    print(
        "SAPIENS2 COCO17 ADAPTER: PASSED"
    )


if __name__ == "__main__":
    main()
