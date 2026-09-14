#!/usr/bin/env python3
"""Render clear reference overlays from LabelMe polygons without changing labels.

The output is intended for annotator orientation: visible ``floor_eligible``
pixels are semi-transparent green with a green outline.  Pixels outside the
polygons are deliberately left uncoloured; they are not implicitly non-floor
truth, merely not part of this manual reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


FLOOR_LABEL = "floor_eligible"
FILL_BGR = (0, 205, 0)
OUTLINE_BGR = (0, 255, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-ids", type=int, nargs="+", required=True)
    return parser.parse_args()


def load_polygons(label_path: Path, image_shape: tuple[int, int]) -> list[np.ndarray]:
    value = json.loads(label_path.read_text(encoding="utf-8"))
    height, width = image_shape
    if value.get("imageHeight") != height or value.get("imageWidth") != width:
        raise ValueError(f"{label_path}: LabelMe image dimensions differ from image")
    polygons: list[np.ndarray] = []
    for shape in value.get("shapes", []):
        if shape.get("label") != FLOOR_LABEL:
            continue
        if shape.get("shape_type", "polygon") != "polygon":
            raise ValueError(f"{label_path}: {FLOOR_LABEL} must use polygon geometry")
        points = np.asarray(shape.get("points"), dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError(f"{label_path}: invalid {FLOOR_LABEL} polygon")
        polygons.append(np.rint(points).astype(np.int32))
    if not polygons:
        raise ValueError(f"{label_path}: no {FLOOR_LABEL} polygons")
    return polygons


def render_overlay(image: np.ndarray, polygons: list[np.ndarray], pair_id: int) -> np.ndarray:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, polygons, 255)
    fill = image.copy()
    fill[mask > 0] = FILL_BGR
    output = cv2.addWeighted(image, 0.52, fill, 0.48, 0.0)
    cv2.polylines(output, polygons, isClosed=True, color=OUTLINE_BGR, thickness=4, lineType=cv2.LINE_AA)
    header = "RIGHT CAMERA | MANUAL floor_eligible = GREEN"
    cv2.rectangle(output, (0, 0), (min(1180, output.shape[1]), 56), (255, 255, 255), -1)
    cv2.putText(output, header, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.84, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(output, f"pair_{pair_id:04d}", (output.shape[1] - 210, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 0), 2, cv2.LINE_AA)
    return output


def make_contact_sheet(items: list[tuple[int, np.ndarray]]) -> np.ndarray:
    target_width, target_height = 960, 540
    tiles: list[np.ndarray] = []
    for _, image in items:
        tiles.append(cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA))
    while len(tiles) % 2:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[index:index + 2]) for index in range(0, len(tiles), 2)]
    return np.vstack(rows)


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    rendered: list[tuple[int, np.ndarray]] = []
    records: list[dict[str, object]] = []
    for pair_id in args.pair_ids:
        stem = f"pair_{pair_id:04d}"
        image_path = args.image_dir / f"{stem}.png"
        label_path = args.label_dir / f"{stem}.json"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot read image: {image_path}")
        polygons = load_polygons(label_path, image.shape[:2])
        rendered_image = render_overlay(image, polygons, pair_id)
        output_path = args.output_dir / f"{stem}_right_manual_floor_reference.png"
        if not cv2.imwrite(str(output_path), rendered_image):
            raise RuntimeError(f"cannot write overlay: {output_path}")
        rendered.append((pair_id, rendered_image))
        records.append({
            "pair_id": pair_id,
            "image": str(image_path.resolve()),
            "label": str(label_path.resolve()),
            "label_name": FLOOR_LABEL,
            "polygon_count": len(polygons),
            "overlay": str(output_path.resolve()),
            "interpretation": "Green denotes manually drawn floor_eligible only; it is a reference annotation, not 3-D ground truth.",
        })
    sheet_path = args.output_dir / "right_manual_floor_reference_contact_sheet.png"
    if not cv2.imwrite(str(sheet_path), make_contact_sheet(rendered)):
        raise RuntimeError(f"cannot write contact sheet: {sheet_path}")
    (args.output_dir / "manifest.json").write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "overlays": len(records), "contact_sheet": str(sheet_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
