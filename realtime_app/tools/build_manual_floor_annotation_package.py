#!/usr/bin/env python3
"""Build a manual-annotation package for an independent floor-mask validation set.

Selects stereo pairs from an independent capture and exports everything a human
annotator needs: the upright left/right images, empty LabelMe label paths, the
five class names, a selection manifest with the measurable proxies, and the
labeling protocol.

Selection uses only saved artifacts (detector person boxes and the frozen
protocol quality table).  No pose model, no segmentation model and no geometry
is run here, and the selection proxies are image-space/detector quantities: they
describe proximity and occlusion, and they are not ground truth.

The exported images are unstyled originals.  This tool never writes labels: the
annotation itself is human work and must stay human work.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any


TOOLS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_ROOT.parents[1]

CLASSES = ("floor_eligible", "person", "walker", "static_other", "ignore_uncertain")

LABEL_ME_CLASS_FILE = "\n".join(CLASSES) + "\n"

PROTOCOL = """# people_2 地面掩膜标注协议（独立验证集）

## 目标

为 `floor_eligible` 双目对应可用性的独立验证提供二维身份掩膜。掩膜只回答“这个像素在二维上是否属于可确认的可见地面”，**不是三维地面真值**，也不用于任何地面精度、足地高度、接触、支撑或步态结论。

## 类别（只用这五个名字）

| 类别 | 含义 |
|---|---|
| `floor_eligible` | 可确认的可见地面。光照足、边界清楚、无遮挡、无强烈反光或运动模糊。 |
| `person` | 人体（含衣物、鞋）。 |
| `walker` | 助步器构件（把手、立柱、轮子、坐垫等）。 |
| `static_other` | 其他静态物体（箱、袋、家具、墙脚、纸箱等）。 |
| `ignore_uncertain` | 无法判断：阴影边界、反光、运动模糊、地面与物体交界、过暗或过曝区域。 |

## 规则

1. 只标注**能确认**的地面。看不清、犹豫、边界模糊的一律画进 `ignore_uncertain`，不要猜。
2. 可见性优先级（转换互斥像素标签时）：`ignore_uncertain > person > walker > static_other > floor_eligible`。因此被人体或助步器遮挡的地面不要画成 `floor_eligible`。
3. 阴影和反光**不要**画成 `floor_eligible`；如果影响判断就画 `ignore_uncertain`。
4. 地面只在图像下方区域；不要为了“补面积”把墙、桌面或近景杂物画成地面。
5. 允许地面被分成多个多边形（例如被助步器立柱隔开）；不必强行连成一块。
6. 左右图**分别**标注：同一时刻的两张图各自可见范围不同，不要照抄另一侧的形状。
7. 标注文件保存为 `labels/<side>/pair_XXXX.json`（LabelMe 默认格式）。图像尺寸必须保持 1080 x 1920，不要缩放或旋转。

## 本目录包含

- `images/left/`、`images/right/`：待标注的正立图（原始像素，未做任何处理）。
- `labels/left/`、`labels/right/`：标注输出目录（当前为空）。
- `labelme_classes.txt`：LabelMe 类别清单，可用 `labelme --labels labelme_classes.txt images/left` 启动。
- `selection_manifest.csv`：每对样本的选择代理量（人体框面积占比、框底位置、冻结协议的几何质量列）。
- `selection.json`：抽样规则与代理量的分位数，供复核抽样是否覆盖预期范围。

## 抽样说明（必须先读）

本采集是单距离近景连续采集：266 对的人体框面积占比只在 0.470--0.507 之间，框底位置只在 1045--1076（1920 高）之间。因此本集**不能**提供“远/中/近地面”的距离扫描；抽样覆盖的是同一距离下**可见地面范围与遮挡程度的差异**，以及时间轴的均匀分布。请按此理解标注，不要期望本集里有明显更远的帧。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--left-dir", type=Path, required=True)
    parser.add_argument("--right-dir", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True, help="saved detector json for the left upright images")
    parser.add_argument("--quality-csv", type=Path, help="optional frozen protocol frame quality table")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--picks-per-block", type=int, default=2)
    parser.add_argument("--min-index-separation", type=int, default=20)
    parser.add_argument("--image-width", type=int, default=1080)
    parser.add_argument("--image-height", type=int, default=1920)
    return parser.parse_args()


def person_box(entry: dict[str, Any]) -> dict[str, float] | None:
    boxes = entry.get("detections") or []
    if not boxes:
        return None
    best = max(boxes, key=lambda item: float(item["score"]))
    x1, y1, x2, y2 = (float(value) for value in best["bbox_xyxy"])
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": float(best["score"]),
            "box_count": float(len(boxes))}


def select(
    entries: list[dict[str, Any]], indices: list[int], quality: dict[int, dict[str, str]],
    blocks: int, picks: int, separation: int, area: float,
) -> list[int]:
    picks_out: list[int] = []
    size = max(1, len(indices) // blocks)
    for block in range(blocks):
        start = block * size
        stop = len(indices) if block == blocks - 1 else min(len(indices), start + size)
        window = indices[start:stop]
        scored = []
        for index in window:
            box = person_box(entries[index])
            if box is None:
                continue
            fraction = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"]) / area
            scored.append((fraction, index))
        if not scored:
            continue
        scored.sort()
        chosen: list[int] = []
        for fraction, index in reversed(scored):  # largest first
            if all(abs(index - other) >= separation for other in chosen):
                chosen.append(index)
                break
        for fraction, index in scored:  # smallest first
            if all(abs(index - other) >= separation for other in chosen):
                chosen.append(index)
                break
        for index in chosen[:picks]:
            if index not in picks_out:
                picks_out.append(index)
    return sorted(picks_out)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    entries = json.loads(args.detections.read_text(encoding="utf-8-sig"))["images"]
    indices = [int(entry["frame_index"]) for entry in entries]
    quality: dict[int, dict[str, str]] = {}
    if args.quality_csv and args.quality_csv.is_file():
        with args.quality_csv.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                quality[int(row["pair_id"])] = row
    area = float(args.image_width * args.image_height)
    picked = select(entries, indices, quality, args.blocks, args.picks_per_block, args.min_index_separation, area)
    if not picked:
        raise RuntimeError("selection produced no pairs")

    (args.output_dir / "images" / "left").mkdir(parents=True)
    (args.output_dir / "images" / "right").mkdir(parents=True)
    (args.output_dir / "labels" / "left").mkdir(parents=True)
    (args.output_dir / "labels" / "right").mkdir(parents=True)
    (args.output_dir / "labelme_classes.txt").write_text(LABEL_ME_CLASS_FILE, encoding="utf-8")
    (args.output_dir / "README_annotation.md").write_text(PROTOCOL, encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for index in picked:
        name = f"pair_{index:04d}.png"
        for side, source in (("left", args.left_dir), ("right", args.right_dir)):
            source_path = source / name
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            shutil.copy2(source_path, args.output_dir / "images" / side / name)
        box = person_box(entries[index])
        fraction = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"]) / area if box else None
        row = {
            "pair_id": index,
            "file_name": name,
            "person_box_area_fraction": None if fraction is None else round(fraction, 4),
            "person_box_bottom_y": None if box is None else round(box["y2"], 1),
            "person_box_count": None if box is None else int(box["box_count"]),
            "person_box_score": None if box is None else round(box["score"], 3),
        }
        for key, value in (quality.get(index) or {}).items():
            if key not in {"pair_id", "file_name", "pair_outcome"}:
                row[f"frozen_{key}"] = value
        rows.append(row)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (args.output_dir / "selection_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    population = sorted(
        (box["x2"] - box["x1"]) * (box["y2"] - box["y1"]) / area
        for box in (person_box(entry) for entry in entries) if box is not None
    )

    def quantile(values: list[float], q: float) -> float:
        return values[min(len(values) - 1, int(q * (len(values) - 1)))]

    selection = {
        "rule": (
            f"split the timeline into {args.blocks} equal blocks; in each block take the largest and the smallest "
            f"detector person-box area subject to at least {args.min_index_separation} pairs of separation"
        ),
        "picked_pair_ids": picked,
        "selection_proxies_are": "saved detector person boxes and the frozen protocol quality table only",
        "population_person_box_area_fraction": {
            "p05": round(quantile(population, 0.05), 4),
            "median": round(quantile(population, 0.5), 4),
            "p95": round(quantile(population, 0.95), 4),
        },
        "picked_person_box_area_fraction": {
            row["file_name"]: row["person_box_area_fraction"] for row in rows
        },
        "distance_coverage_limitation": (
            "this capture is a single-distance close capture: the person-box area fraction spans only "
            "about 0.47 to 0.51 and the box bottom only about 1045 to 1076 of 1920, so these six pairs cannot "
            "cover far/mid/near ground as a distance sweep; they cover visible-ground extent, occlusion extremes "
            "and an even spread over the timeline"
        ),
        "classes": list(CLASSES),
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_dir), "picked_pair_ids": picked,
                      "area_fraction_p05_median_p95": [selection["population_person_box_area_fraction"][key]
                                                       for key in ("p05", "median", "p95")]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
