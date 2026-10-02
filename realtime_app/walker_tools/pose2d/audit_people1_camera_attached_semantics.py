#!/usr/bin/env python3
"""Create and audit visually defined person/walker/floor candidates over a video.

The walker candidate is not obtained from an ADE20K class (ADE20K has no
walker class).  It is a camera-attached *structural template*: dark Canny
edges that persist at the same image locations across a video-wide sample,
after repeated scene-person/floor predictions are excluded.  This implements
the testable part of the walker-mounted-camera hypothesis without claiming
that persistence alone is semantic truth.

Classes are exclusive in the rendered labels: visible dark template pixels are
``walker_candidate``; remaining SegFormer person and floor predictions are
``person_candidate`` and ``floor_candidate``.  All other pixels are unknown.
The output is strictly for visual identity auditing and must not enter pose,
stereo geometry, ground, contact, or gait code.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCHEMA_VERSION = "camera_attached_semantics_audit_v1"
COLOR_UNKNOWN = (0, 0, 0)
COLOR_PERSON = (50, 50, 235)
COLOR_WALKER = (235, 90, 30)
COLOR_FLOOR = (35, 185, 35)


@dataclass(frozen=True)
class SceneMasks:
    person: np.ndarray
    floor: np.ndarray
    metrics: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="one upright video view, pair_*.png")
    parser.add_argument("--output-dir", type=Path, required=True, help="new audit directory")
    parser.add_argument("--model-id", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--sample-stride", type=int, default=10, help="sample every N frames across whole video")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--edge-support-min", type=float, default=0.72)
    parser.add_argument("--dark-template-max", type=int, default=115)
    parser.add_argument("--dark-visible-max", type=int, default=135)
    parser.add_argument("--maximum-scene-exclusion-fraction", type=float, default=0.30)
    parser.add_argument("--template-dilate-px", type=int, default=13)
    parser.add_argument("--minimum-component-area-px", type=int, default=250)
    parser.add_argument("--visualize-every", type=int, default=50)
    parser.add_argument("--visual-reference", type=Path, help="optional visually reviewed polyline reference JSON for template audit")
    return parser.parse_args()


def sampled_paths(input_dir: Path, sample_stride: int) -> list[Path]:
    if sample_stride < 1:
        raise ValueError("--sample-stride must be positive")
    paths = sorted(input_dir.glob("pair_*.png"))
    if not paths:
        raise RuntimeError(f"no pair_*.png images in {input_dir}")
    return paths[::sample_stride]


def class_id(model: Any, label: str) -> int:
    ids = [int(index) for index, value in model.config.id2label.items() if str(value).strip().casefold() == label]
    if len(ids) != 1:
        raise ValueError(f"expected exactly one {label!r} class, found {ids}")
    return ids[0]


def load_scene_model(model_id: str, requested_device: str) -> tuple[Any, Any, Any, str, int, int]:
    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else ("cpu" if requested_device == "auto" else requested_device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    processor = SegformerImageProcessor.from_pretrained(model_id, local_files_only=True)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id, local_files_only=True).to(device).eval()
    return processor, model, torch, device, class_id(model, "person"), class_id(model, "floor")


def infer_scene_masks(images: list[np.ndarray], processor: Any, model: Any, torch: Any, device: str, person_id: int, floor_id: int, batch_size: int) -> list[SceneMasks]:
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    masks: list[SceneMasks] = []
    for start in range(0, len(images), batch_size):
        current = images[start:start + batch_size]
        rgb = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in current]
        inputs = processor(images=rgb, return_tensors="pt")
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.shape[:2] for image in rgb])
        for prediction in predictions:
            array = prediction.detach().cpu().numpy()
            person = array == person_id
            floor = array == floor_id
            masks.append(SceneMasks(person, floor, {"person_fraction": float(person.mean()), "floor_fraction": float(floor.mean())}))
    return masks


def temporal_edge_template(images: list[np.ndarray], scenes: list[SceneMasks], *, edge_support_min: float, dark_template_max: int, maximum_scene_exclusion_fraction: float, template_dilate_px: int, minimum_component_area_px: int) -> tuple[np.ndarray, dict[str, float | int]]:
    """Build an image-coordinate structural template from the full sampled span."""
    if not images or len(images) != len(scenes):
        raise ValueError("images and scene masks must be non-empty and aligned")
    if not 0.0 < edge_support_min <= 1.0 or not 0.0 <= maximum_scene_exclusion_fraction <= 1.0:
        raise ValueError("support/exclusion fractions outside valid range")
    gray = np.stack([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images])
    edge_stack = []
    for image in gray:
        edge = cv2.Canny(image, 45, 120)
        edge = cv2.dilate(edge, np.ones((3, 3), dtype=np.uint8)) > 0
        edge_stack.append(edge)
    edge_support = np.mean(np.stack(edge_stack), axis=0)
    scene_exclusion = np.mean(np.stack([scene.person | scene.floor for scene in scenes]), axis=0)
    median_gray = np.median(gray, axis=0)
    seed = (edge_support >= edge_support_min) & (median_gray <= dark_template_max) & (scene_exclusion <= maximum_scene_exclusion_fraction)
    kernel_size = max(1, int(template_dilate_px) | 1)
    expanded = cv2.dilate(seed.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(expanded, connectivity=8)
    kept = np.zeros_like(expanded)
    for label in range(1, labels_count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_component_area_px:
            kept[labels == label] = 255
    return kept > 0, {
        "sample_count": len(images),
        "persistent_edge_fraction": float((edge_support >= edge_support_min).mean()),
        "template_seed_fraction": float(seed.mean()),
        "template_fraction": float((kept > 0).mean()),
        "component_count_before_area_filter": labels_count - 1,
        "components_retained": int(len(np.unique(labels[kept > 0]))),
    }


def exclusive_labels(image: np.ndarray, template: np.ndarray, scene: SceneMasks, dark_visible_max: int) -> tuple[np.ndarray, dict[str, float]]:
    """Assign visible classes with walker only where its dark template is visible."""
    if not 0 <= dark_visible_max <= 255:
        raise ValueError("--dark-visible-max must be between 0 and 255")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    walker = template & (gray <= dark_visible_max)
    person = scene.person & ~walker
    floor = scene.floor & ~walker & ~person
    labels = np.zeros(gray.shape, dtype=np.uint8)
    labels[person] = 1
    labels[walker] = 2
    labels[floor] = 3
    return labels, {"person_candidate_fraction": float(person.mean()), "walker_candidate_fraction": float(walker.mean()), "floor_candidate_fraction": float(floor.mean()), "unknown_fraction": float((labels == 0).mean())}


def render_overlay(image: np.ndarray, labels: np.ndarray, title: str) -> np.ndarray:
    palette = np.asarray((COLOR_UNKNOWN, COLOR_PERSON, COLOR_WALKER, COLOR_FLOOR), dtype=np.uint8)
    colors = palette[labels]
    output = cv2.addWeighted(image, 0.54, colors, 0.46, 0.0)
    cv2.rectangle(output, (0, 0), (min(output.shape[1], 1020), 50), (255, 255, 255), -1)
    cv2.putText(output, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(output, "red=person candidate  blue=walker structural candidate  green=floor candidate", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)
    return output


def visual_reference_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize a reviewed structural polyline reference; it is not pixel truth."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "walker_visual_polyline_reference_v1":
        raise ValueError("unexpected visual-reference schema_version")
    height, width = shape
    if payload.get("image_size_wh") != [width, height]:
        raise ValueError(f"visual reference image size {payload.get('image_size_wh')} does not match {[width, height]}")
    output = np.zeros(shape, dtype=np.uint8)
    for part in payload.get("parts", []):
        points = np.asarray(part["points_xy"], dtype=np.int32)
        if len(points) < 2 or int(part["thickness_px"]) < 1:
            raise ValueError("every visual-reference part needs >=2 points and positive thickness")
        cv2.polylines(output, [points], False, 255, int(part["thickness_px"]), cv2.LINE_AA)
    return output > 0


def template_overlap(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    if candidate.shape != reference.shape:
        raise ValueError("candidate/reference shapes differ")
    intersection = int(np.logical_and(candidate, reference).sum())
    union = int(np.logical_or(candidate, reference).sum())
    return {"intersection_px": intersection, "candidate_px": int(candidate.sum()), "reference_px": int(reference.sum()), "iou": float(intersection / union) if union else 0.0, "reference_coverage_recall": float(intersection / reference.sum()) if reference.any() else 0.0, "candidate_precision_against_reference": float(intersection / candidate.sum()) if candidate.any() else 0.0}


def render_template_comparison(template: np.ndarray, reference: np.ndarray) -> np.ndarray:
    output = np.zeros((*template.shape, 3), dtype=np.uint8)
    output[reference] = (0, 220, 220)  # yellow reviewed reference
    output[template] = (235, 90, 30)  # blue automatic template
    output[template & reference] = (240, 240, 240)  # white agreement
    return output


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    paths = sampled_paths(args.input_dir, args.sample_stride)
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    if any(image is None for image in images):
        raise RuntimeError("cannot read one or more selected images")
    if len({image.shape[:2] for image in images}) != 1:
        raise RuntimeError("selected frames have inconsistent image shapes")
    processor, model, torch, device, person_id, floor_id = load_scene_model(args.model_id, args.device)
    scenes = infer_scene_masks(images, processor, model, torch, device, person_id, floor_id, args.batch_size)
    template, template_metrics = temporal_edge_template(images, scenes, edge_support_min=args.edge_support_min, dark_template_max=args.dark_template_max, maximum_scene_exclusion_fraction=args.maximum_scene_exclusion_fraction, template_dilate_px=args.template_dilate_px, minimum_component_area_px=args.minimum_component_area_px)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "visualizations").mkdir()
    if not cv2.imwrite(str(args.output_dir / "walker_structural_template.png"), template.astype(np.uint8) * 255):
        raise RuntimeError("cannot write walker structural template")
    reference_metrics: dict[str, float] | None = None
    if args.visual_reference is not None:
        reference = visual_reference_mask(args.visual_reference, images[0].shape[:2])
        reference_metrics = template_overlap(template, reference)
        cv2.imwrite(str(args.output_dir / "walker_visual_reference.png"), reference.astype(np.uint8) * 255)
        cv2.imwrite(str(args.output_dir / "walker_template_vs_visual_reference.png"), render_template_comparison(template, reference))
    rows: list[dict[str, Any]] = []
    for index, (path, image, scene) in enumerate(zip(paths, images, scenes)):
        labels, fractions = exclusive_labels(image, template, scene, args.dark_visible_max)
        frame_id = path.name
        pair_id = int(path.stem.removeprefix("pair_"))
        row = {"schema_version": SCHEMA_VERSION, "pair_id": pair_id, "frame_id": frame_id, "class_semantics": {"0": "unknown", "1": "person_candidate", "2": "walker_structural_candidate", "3": "floor_candidate"}, "scene_model_metrics": scene.metrics, "class_fractions": fractions, "interpretation": "visually audited candidate classes only; neither object identity nor ground/contact truth"}
        rows.append(row)
        if args.visualize_every > 0 and index % args.visualize_every == 0:
            output = render_overlay(image, labels, f"{frame_id}; sampled video-wide template")
            if not cv2.imwrite(str(args.output_dir / "visualizations" / frame_id), output):
                raise RuntimeError(f"cannot write visualization for {frame_id}")
    with (args.output_dir / "sampled_semantic_audit.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    averages = {key: float(np.mean([row["class_fractions"][key] for row in rows])) for key in rows[0]["class_fractions"]}
    summary = {"schema_version": SCHEMA_VERSION, "sampled_frame_count": len(rows), "sample_stride": args.sample_stride, "sampled_pair_ids": [row["pair_id"] for row in rows], "model": {"model_id": args.model_id, "device": device, "person_class_id": person_id, "floor_class_id": floor_id}, "template_metrics": template_metrics, "visual_reference_metrics": reference_metrics, "mean_class_fractions": averages, "interpretation_boundary": "Scene-model person/floor and camera-attached structural candidates. This is a visual identity audit, not semantic ground truth, stereo geometry, ground, contact, support, or gait evidence."}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "run_metadata.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "input_dir": str(args.input_dir.resolve()), "parameters": vars(args), "method": "full-video stratified sample; cached ADE20K person/floor candidate masks; visually defined persistent dark edge camera-attached structural template; per-frame exclusive rendering", "interpretation_boundary": summary["interpretation_boundary"]}, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
