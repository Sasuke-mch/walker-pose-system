#!/usr/bin/env python3
"""Prepare blinded two-view reviews for lower-body geometry rejections.

The input is a saved geometry replay.  The tool never reruns a model, changes
coordinates, or accepts/rejects a point.  It renders each requested rejected
same-name joint on the original raw fisheye pair and writes a manual root-cause
template.  The annotator-facing CSV deliberately omits model names, scores and
reprojection residuals.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


LOWER_BODY = {
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
}


def parse_condition(value: str) -> dict[str, Path | str]:
    fields = value.split("|")
    if len(fields) != 6:
        raise ValueError(
            "Each --condition must be BLIND_ID|RESULTS_JSONL|LEFT_IMAGE_DIR|RIGHT_IMAGE_DIR|LEFT_ROTATION|RIGHT_ROTATION"
        )
    blind_id, results, left_dir, right_dir, left_rotation, right_rotation = fields
    if not blind_id:
        raise ValueError("BLIND_ID must not be empty")
    if left_rotation not in {"none", "cw90", "ccw90", "180"} or right_rotation not in {"none", "cw90", "ccw90", "180"}:
        raise ValueError("Image rotations must be none, cw90, ccw90 or 180")
    return {
        "blind_id": blind_id,
        "results": Path(results).resolve(),
        "left_dir": Path(left_dir).resolve(),
        "right_dir": Path(right_dir).resolve(),
        "left_rotation": left_rotation,
        "right_rotation": right_rotation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        help="BLIND_ID|RESULTS_JSONL|LEFT_IMAGE_DIR|RIGHT_IMAGE_DIR|LEFT_ROTATION|RIGHT_ROTATION; repeat per baseline.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reason",
        action="append",
        default=["high_reprojection_error"],
        choices=(
            "high_reprojection_error",
            "out_of_raw_image_bounds",
            "negative_or_zero_depth",
            "non_finite_triangulation",
        ),
        help="3-D rejection reason to audit (default: high_reprojection_error).",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def lower_point_map(record: dict) -> dict[str, dict]:
    people = record.get("persons_3d", [])
    if len(people) != 1:
        return {}
    return {
        point["name"]: point
        for point in people[0].get("keypoints_3d", [])
        if point.get("name") in LOWER_BODY
    }


def original_points(record: dict, name: str) -> tuple[list[float], list[float]]:
    people = record["persons_3d"]
    person = people[0]
    index = next(
        int(point["index"])
        for point in person["keypoints_3d"]
        if point["name"] == name
    )
    left_person_id = int(person["left_person_id"])
    right_person_id = int(person["right_person_id"])
    left = next(
        candidate for candidate in record["left"]["persons"]
        if int(candidate["person_id"]) == left_person_id
    )
    right = next(
        candidate for candidate in record["right"]["persons"]
        if int(candidate["person_id"]) == right_person_id
    )
    return list(left["keypoints"][index]), list(right["keypoints"][index])


def draw_point(image: np.ndarray, point: list[float], label: str) -> np.ndarray:
    output = image.copy()
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    color = (0, 180, 255)
    cv2.drawMarker(output, (x, y), color, cv2.MARKER_TILTED_CROSS, 26, 3, cv2.LINE_AA)
    cv2.putText(
        output,
        label,
        (max(8, x - 90), max(35, y - 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )
    return output


def restore_raw_image(image: np.ndarray, rotation: str) -> np.ndarray:
    operations = {
        "none": None,
        "cw90": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "ccw90": cv2.ROTATE_90_CLOCKWISE,
        "180": cv2.ROTATE_180,
    }
    operation = operations[rotation]
    return image if operation is None else cv2.rotate(image, operation)


def make_panel(left: np.ndarray, right: np.ndarray, point_name: str) -> np.ndarray:
    label = point_name.replace("_", " ")
    separator = np.full((left.shape[0], 12, 3), 235, dtype=np.uint8)
    panel = np.concatenate([left, separator, right], axis=1)
    header = np.full((70, panel.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(
        header,
        f"Manual 2-D review: same-name {label}",
        (24, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    return np.concatenate([header, panel], axis=0)


def protocol_text() -> str:
    return """# 双目下肢二维人工审核协议

每个条目仅显示一对原始鱼眼图及同名关节标记。请不要查看 `audit_key.csv`，也不要根据模型、置信度或重投影数值推断原因。

1. `target_person_confirmed`：目标人是否在左右图均可确认（true/false）。
2. `left_anatomical_location` 与 `right_anatomical_location`：填写 correct / misplaced / indeterminate。
3. `left_fisheye_edge_or_occlusion` 与 `right_fisheye_edge_or_occlusion`：填写 none / fisheye_edge / self_occlusion / walker_or_background_occlusion / indeterminate。
4. `primary_root_cause`：只能填写 anatomical_mislocalization、fisheye_edge_or_occlusion、local_sync_or_calibration_suspected、insufficient_evidence 或 no_visible_problem。
5. 只有在两侧点位均合理、无遮挡/边缘解释不足且残差模式在附近关节或连续邻帧重复时，才可填写 `local_sync_or_calibration_suspected`；单帧高残差不是同步或标定失效证据。

本审核用于区分根因，不构成二维准确率、三维准确率或模型优劣结论。
"""


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    conditions = [parse_condition(value) for value in args.condition]
    if len({item["blind_id"] for item in conditions}) != len(conditions):
        raise ValueError("Every BLIND_ID must be unique")
    accepted_reasons = set(args.reason)
    visual_dir = output / "blinded_panels"
    output.mkdir(parents=True)
    visual_dir.mkdir()
    manual_rows: list[dict[str, object]] = []
    key_rows: list[dict[str, object]] = []
    item_id = 0
    for condition in conditions:
        for record in read_jsonl(condition["results"]):
            points = lower_point_map(record)
            for point_name, point in sorted(points.items()):
                if point.get("reason") not in accepted_reasons:
                    continue
                file_name = str(record["file_name"])
                left_path = condition["left_dir"] / file_name
                right_path = condition["right_dir"] / file_name
                left_image = cv2.imread(str(left_path))
                right_image = cv2.imread(str(right_path))
                if left_image is None or right_image is None:
                    raise FileNotFoundError(
                        f"Missing raw audit image: left={left_path.is_file()} right={right_path.is_file()}"
                    )
                left_image = restore_raw_image(left_image, str(condition["left_rotation"]))
                right_image = restore_raw_image(right_image, str(condition["right_rotation"]))
                left_point, right_point = original_points(record, point_name)
                labeled_left = draw_point(left_image, left_point, point_name.replace("_", " "))
                labeled_right = draw_point(right_image, right_point, point_name.replace("_", " "))
                panel = make_panel(labeled_left, labeled_right, point_name)
                audit_item_id = f"A{item_id:04d}"
                panel_path = visual_dir / f"{audit_item_id}.jpg"
                if not cv2.imwrite(str(panel_path), panel):
                    raise RuntimeError(f"Cannot write {panel_path}")
                manual_rows.append(
                    {
                        "audit_item_id": audit_item_id,
                        "blind_condition": condition["blind_id"],
                        "panel_image": str(panel_path),
                        "file_name": file_name,
                        "joint_subject_anatomy": point_name,
                        "target_person_confirmed": "",
                        "left_anatomical_location": "",
                        "right_anatomical_location": "",
                        "left_fisheye_edge_or_occlusion": "",
                        "right_fisheye_edge_or_occlusion": "",
                        "primary_root_cause": "",
                        "sync_or_calibration_evidence": "",
                        "annotator": "",
                        "reviewer": "",
                        "notes": "",
                    }
                )
                key_rows.append(
                    {
                        "audit_item_id": audit_item_id,
                        "blind_condition": condition["blind_id"],
                        "source_results_jsonl": str(condition["results"]),
                        "pair_id": record["pair_id"],
                        "file_name": file_name,
                        "joint_subject_anatomy": point_name,
                        "geometry_reason": point["reason"],
                        "left_model_xy_score": json.dumps(left_point),
                        "right_model_xy_score": json.dumps(right_point),
                        "reprojection_error_mean_px": point.get("reprojection_error_mean_px"),
                    }
                )
                item_id += 1
    for path, rows in (
        (output / "manual_audit_template.csv", manual_rows),
        (output / "audit_key.csv", key_rows),
    ):
        fields = list(rows[0]) if rows else ["audit_item_id"]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (output / "AUDIT_PROTOCOL.md").write_text(protocol_text(), encoding="utf-8")
    metadata = {
        "conditions": [
            {key: str(value) for key, value in item.items()} for item in conditions
        ],
        "included_reasons": sorted(accepted_reasons),
        "items": len(manual_rows),
        "interpretation_boundary": "Manual root-cause review only; no model output, calibration, threshold or 3-D coordinate was changed.",
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "audit_items": len(manual_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
