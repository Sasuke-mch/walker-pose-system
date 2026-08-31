#!/usr/bin/env python3
"""Build an auditable one-box-per-image expansion control for upright pose runs.

The baseline preserves the highest-scoring current YOLO person detection.  The
expanded condition changes only its geometry: 15% horizontal context, 10%
above, and 25% below the original height.  All values are clipped to image
bounds.  It deliberately does not use historical boxes, pose outputs, or
reference keypoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


RULE = {
    "selection": "highest current YOLO person score only",
    "x_pad_fraction_of_box_width": 0.15,
    "top_pad_fraction_of_box_height": 0.10,
    "bottom_pad_fraction_of_box_height": 0.25,
    "uses_historical_boxes": False,
    "uses_pose_or_reference_keypoints": False,
}


def _top_person(image: dict[str, Any]) -> dict[str, Any] | None:
    persons = [x for x in image.get("detections", []) if x.get("class_name") == "person"]
    return max(persons, key=lambda x: float(x.get("score", 0.0))) if persons else None


def _expanded(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = map(float, box)
    bw, bh = x2 - x1, y2 - y1
    return [
        max(0.0, x1 - RULE["x_pad_fraction_of_box_width"] * bw),
        max(0.0, y1 - RULE["top_pad_fraction_of_box_height"] * bh),
        min(float(width - 1), x2 + RULE["x_pad_fraction_of_box_width"] * bw),
        min(float(height - 1), y2 + RULE["bottom_pad_fraction_of_box_height"] * bh),
    ]


def _draw(image_path: Path, base: list[float], expanded: list[float], out: Path, title: str) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    for box, color, label in ((base, (0, 0, 255), "YOLO baseline"), (expanded, (255, 0, 0), "expanded input box")):
        x1, y1, x2, y2 = map(lambda x: int(round(x)), box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)
        cv2.putText(image, label, (x1, max(34, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    cv2.putText(image, title, (24, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(image, title, (24, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), image):
        raise RuntimeError(f"cannot write {out}")


def _contact_sheet(items: list[Path], out: Path, cols: int = 4) -> None:
    thumbs: list[np.ndarray] = []
    for p in items:
        im = cv2.imread(str(p))
        if im is None:
            continue
        h = 420
        w = int(im.shape[1] * h / im.shape[0])
        thumbs.append(cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return
    cell_w, cell_h = max(x.shape[1] for x in thumbs), 420
    rows = math.ceil(len(thumbs) / cols)
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)
    for n, im in enumerate(thumbs):
        row, col = divmod(n, cols)
        x = col * cell_w + (cell_w - im.shape[1]) // 2
        y = row * cell_h
        sheet[y:y + im.shape[0], x:x + im.shape[1]] = im
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-json", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--baseline-json", required=True, type=Path)
    parser.add_argument("--expanded-json", required=True, type=Path)
    parser.add_argument("--manifest-csv", required=True, type=Path)
    parser.add_argument("--visualization-dir", required=True, type=Path)
    parser.add_argument("--side-label", required=True)
    args = parser.parse_args()

    source = json.loads(args.det_json.read_text(encoding="utf-8"))
    baseline_images, expanded_images, rows, rendered = [], [], [], []
    for image in source["images"]:
        det = _top_person(image)
        base_image = {k: image[k] for k in ("image_id", "frame_index", "file_name", "image_path", "width", "height")}
        exp_image = dict(base_image)
        if det is None:
            base_image["detections"] = []
            exp_image["detections"] = []
            rows.append({"file_name": image["file_name"], "has_person": 0})
        else:
            base = list(map(float, det["bbox_xyxy"]))
            expanded = _expanded(base, int(image["width"]), int(image["height"]))
            base_det = dict(det, bbox_xyxy=base)
            exp_det = dict(det, bbox_xyxy=expanded)
            base_image["detections"] = [base_det]
            exp_image["detections"] = [exp_det]
            rows.append({
                "file_name": image["file_name"], "has_person": 1, "score": det.get("score"),
                "base_x1": base[0], "base_y1": base[1], "base_x2": base[2], "base_y2": base[3],
                "expanded_x1": expanded[0], "expanded_y1": expanded[1], "expanded_x2": expanded[2], "expanded_y2": expanded[3],
                "base_width": base[2] - base[0], "base_height": base[3] - base[1],
                "expanded_width": expanded[2] - expanded[0], "expanded_height": expanded[3] - expanded[1],
            })
            vis = args.visualization_dir / image["file_name"].replace(".png", ".jpg")
            _draw(args.image_dir / image["file_name"], base, expanded, vis, f"{args.side_label} | {image['file_name']}")
            rendered.append(vis)
        baseline_images.append(base_image)
        expanded_images.append(exp_image)

    meta = {"schema_version": "a10_box_expansion_control_v1", "source_detection_json": str(args.det_json), "rule": RULE}
    for path, images, condition in ((args.baseline_json, baseline_images, "C0_current_yolo_top1"), (args.expanded_json, expanded_images, "C1_current_yolo_expanded")):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(meta, condition=condition, images=images)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with args.manifest_csv.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    for start, label in ((0, "far"), (20, "mid"), (40, "near")):
        _contact_sheet(rendered[start:start + 20], args.visualization_dir.parent / f"contact_sheet_{args.side_label}_{label}.jpg")
    print(json.dumps({"images": len(source["images"]), "rendered": len(rendered), "rule": RULE}, ensure_ascii=False))


if __name__ == "__main__":
    main()
