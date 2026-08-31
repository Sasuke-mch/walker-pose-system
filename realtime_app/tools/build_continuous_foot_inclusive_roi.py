#!/usr/bin/env python3
"""Construct a continuous, foot-inclusive pose ROI from a YOLO person box.

For r <= tau, return the original detection.  For r > tau, padding grows with
g(r)=((r-tau)/(1-tau))**power.  The lower pad is deliberately dominant so that
near walker-view frames include the shoe and a small ground-side margin.
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
    "threshold": 0.37,
    "power": 0.75,
    "horizontal_pad_max_fraction_of_box_width": 0.20,
    "top_pad_max_fraction_of_box_height": 0.08,
    "bottom_pad_max_fraction_of_box_height": 1.30,
    "intent": "aggressive pose-input ROI; preserve far boxes and prefer background context over missing feet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--det-json", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--manifest-csv", required=True, type=Path)
    parser.add_argument("--visualization-dir", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=RULE["threshold"])
    parser.add_argument("--power", type=float, default=RULE["power"])
    parser.add_argument("--side-label", required=True)
    return parser.parse_args()


def top_person(item: dict[str, Any]) -> dict[str, Any] | None:
    dets = [x for x in item.get("detections", []) if x.get("class_name") == "person"]
    return max(dets, key=lambda x: float(x.get("score", 0.0))) if dets else None


def expansion_fraction(width_fraction: float, threshold: float, power: float) -> float:
    if width_fraction <= threshold:
        return 0.0
    t = min(1.0, max(0.0, (width_fraction - threshold) / (1.0 - threshold)))
    return t ** power


def foot_inclusive_box(box: list[float], width: int, height: int, threshold: float, power: float) -> tuple[list[float], dict[str, float]]:
    x1, y1, x2, y2 = map(float, box)
    bw, bh = x2 - x1, y2 - y1
    r = bw / float(width)
    g = expansion_fraction(r, threshold, power)
    side = RULE["horizontal_pad_max_fraction_of_box_width"] * g
    top = RULE["top_pad_max_fraction_of_box_height"] * g
    bottom = RULE["bottom_pad_max_fraction_of_box_height"] * g
    output = [
        max(0.0, x1 - side * bw),
        max(0.0, y1 - top * bh),
        min(float(width - 1), x2 + side * bw),
        min(float(height - 1), y2 + bottom * bh),
    ]
    return output, {"width_fraction": r, "growth": g, "side_pad": side, "top_pad": top, "bottom_pad": bottom}


def draw(path: Path, base: list[float], roi: list[float], out: Path) -> None:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    for box, color, label in ((base, (0, 0, 255), "YOLO"), (roi, (0, 180, 0), "continuous pose ROI")):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)
        cv2.putText(image, label, (x1, max(34, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), image):
        raise RuntimeError(f"Cannot write {out}")


def contact_sheets(paths: list[Path], output_dir: Path, prefix: str, cols: int = 4) -> None:
    for chunk_id, offset in enumerate(range(0, len(paths), cols * 3)):
        chunk = paths[offset: offset + cols * 3]
        images = [cv2.imread(str(path)) for path in chunk]
        images = [image for image in images if image is not None]
        if not images:
            continue
        h = 390
        thumbs = [cv2.resize(image, (round(image.shape[1] * h / image.shape[0]), h), interpolation=cv2.INTER_AREA) for image in images]
        cell_w = max(image.shape[1] for image in thumbs)
        sheet = np.full((math.ceil(len(thumbs) / cols) * h, cols * cell_w, 3), 245, dtype=np.uint8)
        for index, image in enumerate(thumbs):
            row, col = divmod(index, cols)
            x = col * cell_w + (cell_w - image.shape[1]) // 2
            sheet[row * h: row * h + h, x: x + image.shape[1]] = image
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / f"{prefix}_{chunk_id:02d}.jpg"), sheet)


def main() -> int:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    if args.power <= 0.0:
        raise ValueError("power must be positive")
    source = json.loads(args.det_json.read_text(encoding="utf-8"))
    result_images, manifest, visuals = [], [], []
    for image in source["images"]:
        det = top_person(image)
        copied = {key: image[key] for key in ("image_id", "frame_index", "file_name", "image_path", "width", "height")}
        if det is None:
            copied["detections"] = []
            manifest.append({"file_name": image["file_name"], "has_person": 0})
        else:
            base = [float(value) for value in det["bbox_xyxy"]]
            roi, stats = foot_inclusive_box(base, int(image["width"]), int(image["height"]), args.threshold, args.power)
            copied["detections"] = [dict(det, bbox_xyxy=roi)]
            manifest.append({
                "file_name": image["file_name"], "has_person": 1, "detector_score": det.get("score"),
                "base_x1": base[0], "base_y1": base[1], "base_x2": base[2], "base_y2": base[3],
                "roi_x1": roi[0], "roi_y1": roi[1], "roi_x2": roi[2], "roi_y2": roi[3], **stats,
            })
            visual = args.visualization_dir / image["file_name"].replace(".png", ".jpg")
            draw(args.image_dir / image["file_name"], base, roi, visual)
            visuals.append(visual)
        result_images.append(copied)
    payload = {
        "schema_version": "continuous_foot_inclusive_pose_roi_v1",
        "source_detection_json": str(args.det_json),
        "side": args.side_label,
        "rule": {**RULE, "threshold": args.threshold, "power": args.power},
        "images": result_images,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in manifest for key in row})
    with args.manifest_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(manifest)
    contact_sheets(visuals, args.visualization_dir.parent / "contact_sheets", f"{args.side_label}_continuous_roi")
    print(json.dumps({"images": len(result_images), "person_detected": len(visuals), "rule": payload["rule"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
