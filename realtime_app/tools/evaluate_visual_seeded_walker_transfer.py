#!/usr/bin/env python3
"""Render visually audited masks and test a human-seeded walker-template transfer.

This deliberately separates two claims.  The supplied polygons/polylines are
human visual audit annotations, not pixel ground truth.  The transfer test asks
only whether a walker template made from declared seed frames remains aligned
with independently reviewed held-out frames in the same camera-mounted video.
It is not an automatic walker detector and cannot enter geometry or ground
estimation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCHEMA_VERSION = "visual_seeded_walker_transfer_v1"
COLORS = np.asarray(((0, 0, 0), (40, 40, 235), (235, 90, 30), (35, 185, 35)), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--annotation-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-pair-ids", type=int, nargs="+", required=True)
    parser.add_argument("--heldout-pair-ids", type=int, nargs="+", required=True)
    parser.add_argument("--minimum-seed-votes", type=int, default=2)
    parser.add_argument("--dark-visible-max", type=int, default=135)
    return parser.parse_args()


def rasterize_annotation(record: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    """Return exclusive labels: unknown=0, person=1, walker=2, floor=3."""
    height, width = shape
    person = np.zeros(shape, dtype=np.uint8)
    floor = np.zeros(shape, dtype=np.uint8)
    walker = np.zeros(shape, dtype=np.uint8)
    for points in record.get("person_polygons_xy", []):
        cv2.fillPoly(person, [np.asarray(points, dtype=np.int32)], 255, cv2.LINE_AA)
    for points in record.get("floor_polygons_xy", []):
        cv2.fillPoly(floor, [np.asarray(points, dtype=np.int32)], 255, cv2.LINE_AA)
    for part in record.get("walker_polylines", []):
        points = np.asarray(part["points_xy"], dtype=np.int32)
        thickness = int(part["thickness_px"])
        if len(points) < 2 or thickness < 1:
            raise ValueError("walker polylines require at least two points and positive thickness")
        cv2.polylines(walker, [points], False, 255, thickness, cv2.LINE_AA)
    labels = np.zeros(shape, dtype=np.uint8)
    labels[floor > 0] = 3
    labels[person > 0] = 1
    labels[walker > 0] = 2  # visible structure is reviewed above human/floor polygons
    if labels.shape != (height, width):
        raise RuntimeError("unexpected annotation shape")
    return labels


def binary_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    if prediction.shape != reference.shape:
        raise ValueError("prediction/reference shapes differ")
    intersection = int(np.logical_and(prediction, reference).sum())
    union = int(np.logical_or(prediction, reference).sum())
    return {
        "prediction_px": int(prediction.sum()),
        "reference_px": int(reference.sum()),
        "intersection_px": intersection,
        "iou": float(intersection / union) if union else 0.0,
        "reference_coverage_recall": float(intersection / reference.sum()) if reference.any() else 0.0,
        "prediction_precision": float(intersection / prediction.sum()) if prediction.any() else 0.0,
    }


def build_template(seed_labels: list[np.ndarray], minimum_votes: int) -> np.ndarray:
    if not seed_labels:
        raise ValueError("at least one seed annotation is required")
    if not 1 <= minimum_votes <= len(seed_labels):
        raise ValueError("--minimum-seed-votes must be within number of seed frames")
    return np.sum(np.stack([labels == 2 for labels in seed_labels]), axis=0) >= minimum_votes


def visible_template(template: np.ndarray, image: np.ndarray, dark_visible_max: int) -> np.ndarray:
    if not 0 <= dark_visible_max <= 255:
        raise ValueError("--dark-visible-max must be in [0, 255]")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return template & (gray <= dark_visible_max)


def semantic_overlay(image: np.ndarray, labels: np.ndarray, title: str) -> np.ndarray:
    output = cv2.addWeighted(image, 0.52, COLORS[labels], 0.48, 0.0)
    cv2.rectangle(output, (0, 0), (min(1050, output.shape[1]), 48), (255, 255, 255), -1)
    cv2.putText(output, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(output, "red=visually audited person  blue=visually audited walker  green=visually audited floor", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 0, 0), 1, cv2.LINE_AA)
    return output


def transfer_overlay(image: np.ndarray, prediction: np.ndarray, reference: np.ndarray, title: str) -> np.ndarray:
    colors = np.zeros_like(image)
    colors[reference] = (0, 220, 220)  # yellow reviewed held-out walker
    colors[prediction] = (235, 90, 30)  # blue transfer prediction
    colors[prediction & reference] = (240, 240, 240)
    output = cv2.addWeighted(image, 0.48, colors, 0.52, 0.0)
    cv2.rectangle(output, (0, 0), (min(1050, output.shape[1]), 48), (255, 255, 255), -1)
    cv2.putText(output, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(output, "yellow=independent visual audit  blue=seed transfer  white=agreement", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 0, 0), 1, cv2.LINE_AA)
    return output


def load_annotations(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "walker_visual_semantic_audit_v1":
        raise ValueError("unexpected annotation schema_version")
    if not isinstance(payload.get("frames"), dict):
        raise ValueError("annotation must contain a frames mapping")
    return payload


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    if set(args.seed_pair_ids) & set(args.heldout_pair_ids):
        raise ValueError("seed and held-out pair IDs must be disjoint")
    annotations = load_annotations(args.annotation_json)
    all_ids = [*args.seed_pair_ids, *args.heldout_pair_ids]
    frame_names = {pair_id: f"pair_{pair_id:04d}.png" for pair_id in all_ids}
    if any(name not in annotations["frames"] for name in frame_names.values()):
        missing = [name for name in frame_names.values() if name not in annotations["frames"]]
        raise ValueError(f"missing annotations for {missing}")
    images = {pair_id: cv2.imread(str(args.input_dir / name), cv2.IMREAD_COLOR) for pair_id, name in frame_names.items()}
    if any(image is None for image in images.values()):
        raise RuntimeError("cannot read one or more annotated frames")
    shape = next(iter(images.values())).shape[:2]
    if any(image.shape[:2] != shape for image in images.values()):
        raise RuntimeError("annotated frames have inconsistent shapes")
    expected_size = annotations.get("image_size_wh")
    if expected_size != [shape[1], shape[0]]:
        raise ValueError(f"annotation image size {expected_size} does not match {[shape[1], shape[0]]}")
    labels = {pair_id: rasterize_annotation(annotations["frames"][name], shape) for pair_id, name in frame_names.items()}
    template = build_template([labels[pair_id] for pair_id in args.seed_pair_ids], args.minimum_seed_votes)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "manual_masks").mkdir()
    (args.output_dir / "manual_overlays").mkdir()
    (args.output_dir / "heldout_comparisons").mkdir()
    cv2.imwrite(str(args.output_dir / "seed_walker_template.png"), template.astype(np.uint8) * 255)
    rows: list[dict[str, Any]] = []
    for pair_id in all_ids:
        name = frame_names[pair_id]
        cv2.imwrite(str(args.output_dir / "manual_masks" / name), labels[pair_id])
        cv2.imwrite(str(args.output_dir / "manual_overlays" / name), semantic_overlay(images[pair_id], labels[pair_id], f"{name}; visually audited coarse semantics"))
        rows.append({"pair_id": pair_id, "frame_id": name, "split": "seed" if pair_id in args.seed_pair_ids else "heldout", "manual_class_pixel_counts": {"person": int((labels[pair_id] == 1).sum()), "walker": int((labels[pair_id] == 2).sum()), "floor": int((labels[pair_id] == 3).sum()), "unknown": int((labels[pair_id] == 0).sum())}})
    heldout_metrics: dict[str, dict[str, float | int]] = {}
    for pair_id in args.heldout_pair_ids:
        prediction = visible_template(template, images[pair_id], args.dark_visible_max)
        reference = labels[pair_id] == 2
        metrics = binary_metrics(prediction, reference)
        heldout_metrics[str(pair_id)] = metrics
        cv2.imwrite(str(args.output_dir / "heldout_comparisons" / frame_names[pair_id]), transfer_overlay(images[pair_id], prediction, reference, f"{frame_names[pair_id]}; visual-seed transfer"))
    with (args.output_dir / "frame_records.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    mean_metrics = {name: float(np.mean([metrics[name] for metrics in heldout_metrics.values()])) for name in ("iou", "reference_coverage_recall", "prediction_precision")} if heldout_metrics else {}
    summary = {"schema_version": SCHEMA_VERSION, "seed_pair_ids": args.seed_pair_ids, "heldout_pair_ids": args.heldout_pair_ids, "minimum_seed_votes": args.minimum_seed_votes, "dark_visible_max": args.dark_visible_max, "template_fraction": float(template.mean()), "heldout_metrics": heldout_metrics, "heldout_mean_metrics": mean_metrics, "interpretation_boundary": "Coarse visual-audit masks and a same-camera seed transfer test only. This is not pixel ground truth, automatic walker recognition, ground evidence, contact, or geometry."}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "run_metadata.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "input_dir": str(args.input_dir.resolve()), "annotation_json": str(args.annotation_json.resolve()), "parameters": vars(args), "interpretation_boundary": summary["interpretation_boundary"]}, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
