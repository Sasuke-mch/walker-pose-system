import argparse
import csv
import json
from pathlib import Path


def threshold_tag(threshold: float) -> str:
    return f"{round(threshold * 100):03d}"


def append_experiment_log(
    log_path: Path,
    input_path: Path,
    rows: list[dict],
) -> None:
    marker = "## EXP-DET-004：生成候选阈值检测缓存"

    if log_path.exists():
        content = log_path.read_text(
            encoding="utf-8-sig",
        )
    else:
        content = (
            "# 人体姿态估计实验记录\n\n"
            "本文件记录实验目的、输入、参数、结果和结论。\n"
        )

    if marker in content:
        print("Experiment already recorded:", marker)
        return

    table_lines = [
        "| conf | 检测框数 | 平均每图框数 | 无检测图片数 | 最大单图框数 |",
        "|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        table_lines.append(
            "| {threshold:.2f} | {boxes} | "
            "{mean_boxes_per_image:.4f} | "
            "{images_with_zero_boxes} | "
            "{max_boxes_per_image} |".format(**row)
        )

    section = (
        "\n\n"
        f"{marker}\n\n"
        "### 实验目的\n\n"
        "从YOLO26x的conf=0.01完整检测缓存中，离线生成多个候选"
        "置信度阈值的检测JSON，供后续PMPose关键点AP对比使用。"
        "该操作不重复运行YOLO26x。\n\n"
        "### 输入\n\n"
        f"- 低阈值检测缓存：`{input_path}`\n"
        "- 图片数量：2500\n"
        "- 候选阈值：0.50、0.56、0.60、0.64、0.70\n"
        "- 筛选条件：`detection.score >= threshold`\n"
        "- 其余检测信息保持不变\n\n"
        "### 输出统计\n\n"
        + "\n".join(table_lines)
        + "\n\n"
        "### 说明\n\n"
        "候选JSON保留原检测记录中的内部image_id、file_name、"
        "图像尺寸和检测框字段。内部image_id不能直接用于COCO评估；"
        "关键点预测转换时将通过file_name映射到OCHuman原始image_id。\n"
    )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path.write_text(
        content + section,
        encoding="utf-8",
    )

    print("Experiment recorded:", marker)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate confidence-filtered YOLO detection caches "
            "without rerunning the detector."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--experiment-log",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(
            f"Input detection JSON not found: {args.input}"
        )

    with args.input.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise TypeError(
            "Expected top-level detection JSON to be a dictionary"
        )

    images = data.get("images")

    if not isinstance(images, list):
        raise TypeError(
            "Detection JSON does not contain an images list"
        )

    if len(images) != 2500:
        raise ValueError(
            f"Expected 2500 image records, found {len(images)}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for threshold in args.thresholds:
        if threshold < 0 or threshold > 1:
            raise ValueError(
                f"Invalid threshold: {threshold}"
            )

        filtered_images = []
        per_image_counts = []

        for record in images:
            detections = record.get("detections", [])

            if not isinstance(detections, list):
                raise TypeError(
                    "Each image record must contain a detections list"
                )

            kept = [
                detection
                for detection in detections
                if float(detection.get("score", 0.0))
                >= threshold
            ]

            filtered_record = dict(record)
            filtered_record["detections"] = kept

            filtered_images.append(filtered_record)
            per_image_counts.append(len(kept))

        tag = threshold_tag(threshold)

        output_path = (
            args.output_dir
            / f"yolo26x_detections_conf{tag}.json"
        )

        output_data = {
            key: value
            for key, value in data.items()
            if key != "images"
        }

        output_data["source_detection_json"] = str(
            args.input
        )
        output_data["confidence_threshold"] = threshold
        output_data["images"] = filtered_images

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                output_data,
                file,
                ensure_ascii=False,
                separators=(",", ":"),
            )

        total_boxes = sum(per_image_counts)

        row = {
            "threshold": threshold,
            "output_file": str(output_path),
            "images": len(filtered_images),
            "boxes": total_boxes,
            "mean_boxes_per_image": (
                total_boxes / len(filtered_images)
            ),
            "images_with_zero_boxes": sum(
                count == 0
                for count in per_image_counts
            ),
            "max_boxes_per_image": max(
                per_image_counts,
                default=0,
            ),
        }

        rows.append(row)

        print(
            f"conf={threshold:.2f} | "
            f"images={len(filtered_images)} | "
            f"boxes={total_boxes} | "
            f"mean={row['mean_boxes_per_image']:.4f} | "
            f"zero_images={row['images_with_zero_boxes']} | "
            f"max={row['max_boxes_per_image']}"
        )

        print("  Saved:", output_path)

    args.summary_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.summary_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "threshold",
                "output_file",
                "images",
                "boxes",
                "mean_boxes_per_image",
                "images_with_zero_boxes",
                "max_boxes_per_image",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Summary:", args.summary_csv)

    if args.experiment_log is not None:
        append_experiment_log(
            log_path=args.experiment_log,
            input_path=args.input,
            rows=rows,
        )


if __name__ == "__main__":
    main()
