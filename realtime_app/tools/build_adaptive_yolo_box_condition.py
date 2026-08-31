#!/usr/bin/env python3
"""Compose a deterministic adaptive ROI condition from audited C0/C1 outputs.

For a top-down pose model, inference is independent per image once its image
and person box are fixed.  This utility therefore constructs C2 exactly from
already completed C0 (tight) and C1 (expanded) image-level outputs, avoiding a
redundant GPU rerun.  Its decision uses only current-box width / image width.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def top_box(image: dict) -> dict | None:
    dets = image.get("detections", [])
    return dets[0] if dets else None


def draw(image_path: Path, c0: list[float], c1: list[float], chosen: list[float], use_expanded: bool, out: Path) -> None:
    im = cv2.imread(str(image_path))
    if im is None:
        raise FileNotFoundError(image_path)
    for box, color, label in ((c0, (0, 0, 255), "C0 YOLO"), (c1, (255, 0, 0), "C1 expanded"), (chosen, (0, 180, 0), "C2 selected")):
        x1, y1, x2, y2 = map(lambda x: round(float(x)), box)
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)
        cv2.putText(im, label, (x1, max(34, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.80, color, 2, cv2.LINE_AA)
    title = f"C2 width-adaptive | {'expanded' if use_expanded else 'tight'}"
    cv2.putText(im, title, (22, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(im, title, (22, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (255, 255, 255), 2, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), im):
        raise RuntimeError(f"Cannot write {out}")


def contact_sheet(images: list[Path], out: Path, cols: int = 4) -> None:
    thumbs = []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            continue
        height = 420
        thumbs.append(cv2.resize(image, (round(image.shape[1] * height / image.shape[0]), height), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return
    cell_width, cell_height = max(x.shape[1] for x in thumbs), 420
    sheet = np.full((math.ceil(len(thumbs) / cols) * cell_height, cols * cell_width, 3), 245, dtype=np.uint8)
    for index, image in enumerate(thumbs):
        row, col = divmod(index, cols)
        x = col * cell_width + (cell_width - image.shape[1]) // 2
        y = row * cell_height
        sheet[y:y + image.shape[0], x:x + image.shape[1]] = image
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c0-det-json", type=Path, required=True)
    parser.add_argument("--c1-det-json", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-det-json", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--visualization-dir", type=Path, required=True)
    parser.add_argument("--width-fraction-threshold", type=float, default=0.37)
    parser.add_argument("--side", required=True)
    parser.add_argument("--c0-model-json", action="append", default=[], metavar="MODEL=PATH")
    parser.add_argument("--c1-model-json", action="append", default=[], metavar="MODEL=PATH")
    parser.add_argument("--output-model-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.width_fraction_threshold < 1:
        raise ValueError("threshold must be in (0, 1)")

    def keyed(values: list[str]) -> dict[str, Path]:
        answer = {}
        for value in values:
            key, sep, raw = value.partition("=")
            if not sep or not key or not raw:
                raise ValueError("model JSON entries must be MODEL=PATH")
            answer[key] = Path(raw)
        return answer

    c0_models, c1_models = keyed(args.c0_model_json), keyed(args.c1_model_json)
    if set(c0_models) != set(c1_models):
        raise ValueError("C0/C1 model JSON sets differ")
    c0, c1 = read_json(args.c0_det_json), read_json(args.c1_det_json)
    c0_images = {x["file_name"]: x for x in c0["images"]}
    c1_images = {x["file_name"]: x for x in c1["images"]}
    if set(c0_images) != set(c1_images):
        raise RuntimeError("C0/C1 detection image sets differ")

    chosen_names: set[str] = set()
    output_images, rows, rendered = [], [], []
    for name in sorted(c0_images):
        left, right = c0_images[name], c1_images[name]
        c0_box, c1_box = top_box(left), top_box(right)
        if (c0_box is None) != (c1_box is None):
            raise RuntimeError(f"Detection availability differs for {name}")
        use_expanded = False
        width_fraction = None
        chosen = left
        if c0_box is not None:
            box = [float(x) for x in c0_box["bbox_xyxy"]]
            width_fraction = (box[2] - box[0]) / float(left["width"])
            use_expanded = width_fraction >= args.width_fraction_threshold
            chosen = right if use_expanded else left
            if use_expanded:
                chosen_names.add(name)
            visual_path = args.visualization_dir / name.replace('.png', '.jpg')
            draw(args.image_dir / name, box, [float(x) for x in c1_box["bbox_xyxy"]], [float(x) for x in top_box(chosen)["bbox_xyxy"]], use_expanded, visual_path)
            rendered.append(visual_path)
        output_images.append(chosen)
        rows.append({"file_name": name, "side": args.side, "width_fraction": width_fraction, "threshold": args.width_fraction_threshold, "selected_source": "C1_expanded" if use_expanded else "C0_tight"})

    payload = dict(c0)
    payload["schema_version"] = "a11_width_adaptive_yolo_roi_v1"
    payload["adaptive_rule"] = {
        "input": "current YOLO top-1 person box width / image width",
        "threshold": args.width_fraction_threshold,
        "width_fraction_below_threshold": "C0 tight box",
        "width_fraction_at_or_above_threshold": "C1 fixed foot-inclusive expansion",
        "uses_distance_label": False,
        "uses_pose_feedback": False,
        "uses_historical_box": False,
    }
    payload["images"] = output_images
    args.output_det_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_det_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for start, label in ((0, "far"), (20, "mid"), (40, "near")):
        contact_sheet(rendered[start:start + 20], args.visualization_dir.parent / f"contact_sheet_{args.side}_{label}.jpg")

    for model in sorted(c0_models):
        a, b = read_json(c0_models[model]), read_json(c1_models[model])
        a_images, b_images = {x["file_name"]: x for x in a["images"]}, {x["file_name"]: x for x in b["images"]}
        if set(a_images) != set(c0_images) or set(b_images) != set(c0_images):
            raise RuntimeError(f"{model}: prediction image sets differ")
        combined = dict(a)
        combined["images"] = [b_images[name] if name in chosen_names else a_images[name] for name in sorted(c0_images)]
        combined["a11_adaptive_source"] = {"c0": str(c0_models[model]), "c1": str(c1_models[model]), "selected_c1_images": len(chosen_names)}
        out = args.output_model_dir / model / "raw_predictions.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"side": args.side, "images": len(output_images), "selected_expanded": len(chosen_names), "selected_tight": len(output_images) - len(chosen_names), "threshold": args.width_fraction_threshold}, ensure_ascii=False))


if __name__ == "__main__":
    main()
