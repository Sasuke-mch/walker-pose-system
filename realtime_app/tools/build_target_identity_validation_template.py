#!/usr/bin/env python3
"""Create a model-blind target-identity template for a frozen stereo set."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--left-raw-dir", type=Path, required=True)
    parser.add_argument("--right-raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.selection_manifest.resolve()
    left_dir = args.left_raw_dir.resolve()
    right_dir = args.right_raw_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    required = {"global_index", "file_name", "condition", "source_session_dir", "source_pair_id"}
    if not source_rows or not required.issubset(source_rows[0]):
        raise RuntimeError(f"{manifest_path} lacks required frozen-selection fields")
    rows: list[dict[str, object]] = []
    for source in source_rows:
        file_name = source["file_name"]
        for side, directory in (("left", left_dir), ("right", right_dir)):
            image = directory / file_name
            if not image.is_file():
                raise FileNotFoundError(image)
            rows.append(
                {
                    "pair_id": source["global_index"],
                    "file_name": file_name,
                    "condition": source["condition"],
                    "camera_side": side,
                    "raw_fisheye_image": str(image),
                    "source_session_dir": source["source_session_dir"],
                    "source_pair_id": source["source_pair_id"],
                    "target_subject_id": "",
                    "target_person_confirmed": "",
                    "target_bbox_x1": "",
                    "target_bbox_y1": "",
                    "target_bbox_x2": "",
                    "target_bbox_y2": "",
                    "other_person_count": "",
                    "target_overlap_with_other_person": "",
                    "target_occlusion": "",
                    "identity_annotator": "",
                    "identity_reviewer": "",
                    "notes": "",
                }
            )
    output.mkdir(parents=True)
    with (output / "target_identity_template.csv").open(
        "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    protocol = """# 目标人身份约束标注协议

在未显示任何模型框、关键点、置信度或三维结果的原始鱼眼图上标注。每一帧先填写唯一的 `target_subject_id` 与 `target_person_confirmed`；只有确认目标人后，才填写其可见人体范围的 `target_bbox_*`。

`other_person_count` 记录其他可见人员数量。`target_overlap_with_other_person` 填 true/false，`target_occlusion` 填 none / self / walker / other_person / background / indeterminate。多人或重叠帧不得按事后的三角化成功、模型分数或重投影残差换人。

这张表只提供部署前的目标身份约束和提示框匹配依据，不是关键点标注、模型准确率或三维真值。
"""
    (output / "IDENTITY_PROTOCOL.md").write_text(protocol, encoding="utf-8")
    metadata = {
        "selection_manifest": str(manifest_path),
        "raw_input": {"left": str(left_dir), "right": str(right_dir)},
        "pairs": len(source_rows),
        "images": len(rows),
        "interpretation_boundary": "The template freezes target-person identity before any model or geometry result is reviewed.",
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "pairs": len(source_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
