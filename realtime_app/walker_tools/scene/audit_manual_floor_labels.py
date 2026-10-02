#!/usr/bin/env python3
"""Audit manual LabelMe floor labels against saved semantic floor candidates.

This is a single-view, image-space audit.  It accepts sparse labels and
therefore reports every labelled image individually; the labelled-image count
is reported in the summary and is never a generalisation or stereo-ground
result.

Visible human, walker, static-object, and ignore polygons take precedence over
``floor_eligible`` when polygons overlap.  This protects a hand-drawn broad
floor polygon from silently becoming a floor label on top of a shoe or walker.
"""

from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
_tool_prepare_imports()

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCHEMA_VERSION = "manual_floor_label_audit_v1"
BACKENDS = ("segformer_b0", "mask2former_swin_small", "oneformer_swin_tiny")
LABELS = ("floor_eligible", "person", "walker", "static_other", "ignore_uncertain")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-dir", type=Path, required=True, help="LabelMe JSON directory for one view")
    parser.add_argument("--image-dir", type=Path, required=True, help="matching upright pair_*.png images")
    parser.add_argument("--prediction-root", type=Path, required=True, help="one-view semantic comparison output directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="new audit output directory")
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=BACKENDS)
    return parser.parse_args()


def load_labelme(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value.get("shapes"), list):
        raise ValueError(f"{path}: LabelMe shapes list missing")
    if not isinstance(value.get("imageWidth"), int) or not isinstance(value.get("imageHeight"), int):
        raise ValueError(f"{path}: LabelMe image dimensions missing")
    return value


def rasterize_labelme(value: dict[str, Any]) -> dict[str, np.ndarray]:
    height, width = int(value["imageHeight"]), int(value["imageWidth"])
    raw = {label: np.zeros((height, width), dtype=bool) for label in LABELS}
    for shape in value["shapes"]:
        label = str(shape.get("label", ""))
        if label not in raw:
            raise ValueError(f"unsupported label {label!r}; expected one of {LABELS}")
        if shape.get("shape_type", "polygon") != "polygon":
            raise ValueError(f"{label}: only polygon shapes are supported")
        points = np.asarray(shape.get("points"), dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError(f"{label}: polygon needs >=3 finite xy points")
        polygon = np.rint(points).astype(np.int32)
        canvas = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(canvas, [polygon], 1)
        raw[label] |= canvas.astype(bool)
    # Pixels represent visible content.  A person over a walker over the floor
    # is person, not amodal walker/floor.  Ambiguous pixels are excluded first.
    exclusive: dict[str, np.ndarray] = {}
    claimed = np.zeros((height, width), dtype=bool)
    for label in ("ignore_uncertain", "person", "walker", "static_other", "floor_eligible"):
        exclusive[label] = raw[label] & ~claimed
        claimed |= exclusive[label]
    exclusive["valid_evaluation"] = ~exclusive["ignore_uncertain"]
    return exclusive


def binary_metrics(reference_floor: np.ndarray, valid: np.ndarray, candidate_floor: np.ndarray) -> dict[str, float | int]:
    if reference_floor.shape != candidate_floor.shape or valid.shape != reference_floor.shape:
        raise ValueError("reference, valid area, and candidate shapes must agree")
    truth, predicted = reference_floor & valid, candidate_floor & valid
    true_positive = int(np.logical_and(truth, predicted).sum())
    false_positive = int(np.logical_and(~truth, predicted).sum())
    false_negative = int(np.logical_and(truth, ~predicted).sum())
    union = true_positive + false_positive + false_negative
    return {
        "true_positive_px": true_positive,
        "false_positive_px": false_positive,
        "false_negative_px": false_negative,
        "reference_floor_px": int(truth.sum()),
        "candidate_floor_px_in_valid_area": int(predicted.sum()),
        "precision": float(true_positive / (true_positive + false_positive)) if true_positive + false_positive else 0.0,
        "recall": float(true_positive / (true_positive + false_negative)) if true_positive + false_negative else 0.0,
        "iou": float(true_positive / union) if union else 0.0,
    }


def manual_audit_summary_boundary(labelled_image_count: int) -> str:
    return (
        f"{labelled_image_count} manually labelled left-view images only. "
        "Pixel agreement is an initial diagnostic, not video generalisation, "
        "stereo validation, metric ground reconstruction, or model-selection proof."
    )


def grayscale_summary(image: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, dict[str, float | int | None]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result: dict[str, dict[str, float | int | None]] = {}
    for label in ("floor_eligible", "person", "walker", "static_other"):
        pixels = gray[masks[label]]
        result[label] = {
            "pixel_count": int(pixels.size),
            "median_gray_0_255": None if not pixels.size else float(np.median(pixels)),
            "p10_gray_0_255": None if not pixels.size else float(np.percentile(pixels, 10)),
            "p90_gray_0_255": None if not pixels.size else float(np.percentile(pixels, 90)),
        }
    return result


def render_error_overlay(image: np.ndarray, reference: np.ndarray, valid: np.ndarray, candidate: np.ndarray, title: str) -> np.ndarray:
    output = image.copy()
    colors = np.zeros_like(image)
    true_positive = reference & candidate & valid
    false_positive = ~reference & candidate & valid
    false_negative = reference & ~candidate & valid
    colors[true_positive] = (0, 190, 0)     # green
    colors[false_positive] = (0, 0, 230)    # red
    colors[false_negative] = (230, 0, 230)  # magenta
    output = cv2.addWeighted(output, 0.58, colors, 0.42, 0.0)
    cv2.rectangle(output, (0, 0), (min(1250, output.shape[1]), 34), (255, 255, 255), -1)
    cv2.putText(output, title + "; green=TP red=FP magenta=FN", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 0, 0), 1, cv2.LINE_AA)
    return output


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    json_paths = sorted(args.label_dir.glob("pair_*.json"))
    if not json_paths:
        raise RuntimeError(f"no pair_*.json files in {args.label_dir}")
    args.output_dir.mkdir(parents=True)
    manual_masks_dir = args.output_dir / "manual_floor_masks"
    manual_masks_dir.mkdir()
    rows: list[dict[str, Any]] = []
    for json_path in json_paths:
        labelme = load_labelme(json_path)
        masks = rasterize_labelme(labelme)
        image_path = args.image_dir / f"{json_path.stem}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot read matching image: {image_path}")
        if image.shape[:2] != masks["floor_eligible"].shape:
            raise ValueError(f"{json_path}: image shape {image.shape[:2]} differs from LabelMe shape {masks['floor_eligible'].shape}")
        pair_id = int(json_path.stem.removeprefix("pair_"))
        per_backend: dict[str, Any] = {}
        for backend in args.backends:
            prediction_path = args.prediction_root / backend / "floor_masks" / f"{json_path.stem}.png"
            prediction = cv2.imread(str(prediction_path), cv2.IMREAD_GRAYSCALE)
            if prediction is None:
                raise RuntimeError(f"cannot read candidate mask: {prediction_path}")
            candidate = prediction > 0
            metrics = binary_metrics(masks["floor_eligible"], masks["valid_evaluation"], candidate)
            walker = masks["walker"] & masks["valid_evaluation"]
            metrics["walker_pixels_predicted_floor"] = int(np.logical_and(walker, candidate).sum())
            metrics["walker_to_floor_fraction"] = None if not walker.any() else float(np.logical_and(walker, candidate).sum() / walker.sum())
            per_backend[backend] = metrics
            cv2.imwrite(str(args.output_dir / f"{json_path.stem}_{backend}_error_overlay.png"), render_error_overlay(image, masks["floor_eligible"], masks["valid_evaluation"], candidate, f"{json_path.stem} {backend}"))
        cv2.imwrite(str(args.output_dir / f"{json_path.stem}_manual_floor.png"), masks["floor_eligible"].astype(np.uint8) * 255)
        cv2.imwrite(str(manual_masks_dir / f"{json_path.stem}.png"), masks["floor_eligible"].astype(np.uint8) * 255)
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "pair_id": pair_id,
            "label_file": str(json_path.resolve()),
            "image": str(image_path.resolve()),
            "manual_pixel_counts": {label: int(masks[label].sum()) for label in (*LABELS, "valid_evaluation")},
            "grayscale_summary": grayscale_summary(image, masks),
            "backends": per_backend,
            "interpretation": "single-view initial manual-label audit; not a generalisation, stereo-ground, or physical-accuracy result",
        })
    summary: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "labelled_image_count": len(rows), "pair_ids": [row["pair_id"] for row in rows], "backend_means": {}, "interpretation_boundary": manual_audit_summary_boundary(len(rows))}
    for backend in args.backends:
        keys = ("precision", "recall", "iou", "walker_to_floor_fraction")
        summary["backend_means"][backend] = {key: float(np.mean([row["backends"][backend][key] for row in rows if row["backends"][backend][key] is not None])) if any(row["backends"][backend][key] is not None for row in rows) else None for key in keys}
    (args.output_dir / "per_image.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
