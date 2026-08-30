"""Prepare a blinded, single-view 2-D lower-limb annotation sheet for M2."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


# COCO-17 supports hip/knee/ankle only. Toe labels remain in the sheet to make
# the evaluation set reusable by a future foot-specific model, but are excluded
# from the current five-model numeric comparison.
JOINTS = (
    ("left_hip", True),
    ("right_hip", True),
    ("left_knee", True),
    ("right_knee", True),
    ("left_ankle", True),
    ("right_ankle", True),
    ("left_toe", False),
    ("right_toe", False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-selection", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selection = args.input_selection.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    manifest_path = selection / "selection_manifest.csv"
    summary_path = selection / "dataset_summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Expected M2 selection_manifest.csv and dataset_summary.json")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        pairs = list(csv.DictReader(handle))
    if len(pairs) != 60:
        raise RuntimeError("Expected the fixed 60-pair M2 selection")

    output.mkdir(parents=True)
    rows = []
    image_count = 0
    for pair in pairs:
        for side in ("left", "right"):
            raw_path = selection / "raw_fisheye" / side / pair["file_name"]
            if not raw_path.is_file():
                raise FileNotFoundError(raw_path)
            image_id = f"{side}_{pair['file_name'].removesuffix('.png')}"
            image_count += 1
            for joint, comparable_to_coco17 in JOINTS:
                rows.append(
                    {
                        "image_id": image_id,
                        "file_name": pair["file_name"],
                        "side": side,
                        "condition": pair["condition"],
                        "raw_fisheye_image": str(raw_path),
                        "joint_subject_anatomy": joint,
                        "comparable_to_current_coco17_models": str(comparable_to_coco17).lower(),
                        "x_px": "",
                        "y_px": "",
                        "visibility": "",
                        "target_person_confirmed": "",
                        "annotator": "",
                        "reviewer": "",
                        "notes": "",
                    }
                )
    header = list(rows[0])
    with (output / "single_view_foot_annotation_template.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    protocol = """# 单侧脚部 2D 盲标协议（M2）

## 范围

本表覆盖固定 60 对同步原始鱼眼帧的左右 120 张图。标注时只看**原始鱼眼图**，不看模型叠加、三维结果、另一视图或模型置信度。

## 标注对象与可见性

先确认画面中目标受试者；`target_person_confirmed` 仅填写 `true` 或 `false`。对该受试者的每个关节填写：

- `x_px`, `y_px`：原始 1920 x 1080 鱼眼像素坐标；
- `visibility=2`：点位可直接辨认；
- `visibility=1`：被遮挡或边界模糊，但可按解剖结构合理推断；
- `visibility=0`：不可判断，此时 x/y 留空；
- `left/right` 一律指受试者自身的解剖学左/右，不是画面左/右或相机左/右。

定位规则：髋为股骨近端在可见裤边下的解剖估计位置；膝为股骨—胫骨轴线转折中心；踝为胫骨轴线与足部连接中心；脚尖为最长趾前缘中央。脚尖保留给后续脚部专项模型，不与当前 COCO-17 五模型做数值误差比较。

## 评估口径

当前五个模型只比较髋、膝、踝。每张图对目标人只保留一个模型实例，规则由独立目标人 ROI/轨迹指定，不能按“最小重投影误差”挑选。

- 单关节像素误差：`||prediction - annotation||_2`；
- 尺度归一化误差：像素误差除以该受试者 2D 躯干尺度（左右髋中点到左右肩中点距离）；
- 检出率：在可标注点中，模型分数超过预先报告的阈值的比例；
- 置信度校准：按模型分数分箱，比较平均置信度与“误差低于预注册阈值”的经验比例；
- 不可见点 (`visibility=0`) 不计入定位误差，也不得被模型高置信度点静默计为成功。

至少抽取 20% 图像进行第二位标注者复核；先报告标注者间像素差，再比较模型。
"""
    (output / "ANNOTATION_PROTOCOL.md").write_text(protocol, encoding="utf-8")
    metadata = {
        "experiment_id": "E20260829-A1_single_view_foot_2d_ground_truth",
        "input_selection": str(selection),
        "raw_geometry": "original 1920x1080 fisheye pixels; no rotation or virtual-camera transform for annotation",
        "images": image_count,
        "rows": len(rows),
        "joints": [{"name": name, "comparable_to_current_coco17_models": comparable} for name, comparable in JOINTS],
        "evaluation_boundary": "Pixel error cannot be computed until independent human annotations are completed. Model confidence alone is not an accuracy estimate.",
    }
    (output / "evaluation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
