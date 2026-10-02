#!/usr/bin/env python3
"""Propose camera-attached walker occlusion masks from short image sequences.

This is deliberately a *candidate* generator, not a walker semantic-ground
truth model.  It uses the weak observation that a walker-mounted camera sees
some walker structure with low image motion.  Static scene pixels can satisfy
the same observation, so a candidate additionally needs viewport-border
support and is rejected when it dominates the image.  The output is suitable
only for a future ground-candidate exclusion mask or for manual review.

It never changes pose estimation, association, stereo triangulation, local
ground state, contact state, or gait measurements.
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


SCHEMA_VERSION = "walker_self_occlusion_temporal_v1"


@dataclass(frozen=True)
class CandidateMask:
    status: str
    mask: np.ndarray
    reasons: list[str]
    metrics: dict[str, float | int | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-dir", type=Path, required=True, help="upright left pair_*.png images")
    parser.add_argument("--right-dir", type=Path, required=True, help="upright right pair_*.png images")
    parser.add_argument("--output-dir", type=Path, required=True, help="new result directory")
    parser.add_argument("--max-frames", type=int, default=30, help="first N paired frames; 0 means all")
    parser.add_argument("--flow-max-px", type=float, default=1.5)
    parser.add_argument("--flow-width", type=int, default=480, help="maximum width used for image-space flow; 0 keeps source width")
    parser.add_argument("--temporal-stride", type=int, default=1, help="frame separation for each stability comparison")
    parser.add_argument("--intensity-difference-max", type=float, default=18.0)
    parser.add_argument("--minimum-stability-fraction", type=float, default=0.75)
    parser.add_argument("--minimum-component-area-px", type=int, default=200)
    parser.add_argument("--maximum-mask-fraction", type=float, default=0.35)
    parser.add_argument("--border-anchor-y-fraction", type=float, default=0.35)
    parser.add_argument("--cross-view-area-ratio-max", type=float, default=4.0)
    parser.add_argument("--segformer-scene-exclusion-model-id", default=None, help="optional cached SegFormer model used only to exclude nominated scene classes")
    parser.add_argument("--segformer-exclusion-labels", nargs="+", default=("person",), help="SegFormer labels used only for exclusion, e.g. person floor")
    parser.add_argument("--segformer-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dark-structure-max", type=int, default=None, help="optional gray-value ceiling for a dark structural candidate; requires person exclusion")
    parser.add_argument("--sam-model-id", default=None, help="optional Hugging Face SAM model, e.g. facebook/sam-vit-base")
    parser.add_argument("--sam-device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def paired_paths(left_dir: Path, right_dir: Path, max_frames: int) -> list[tuple[str, Path, Path]]:
    left = {path.name: path for path in left_dir.glob("pair_*.png")}
    right = {path.name: path for path in right_dir.glob("pair_*.png")}
    names = sorted(set(left) & set(right))
    if not names:
        raise RuntimeError("left/right inputs have no shared pair_*.png images")
    missing_left, missing_right = sorted(set(right) - set(left)), sorted(set(left) - set(right))
    if missing_left or missing_right:
        raise RuntimeError(f"unpaired images: missing_left={missing_left[:3]}, missing_right={missing_right[:3]}")
    if max_frames < 0:
        raise ValueError("--max-frames must be non-negative")
    if max_frames:
        names = names[:max_frames]
    return [(name, left[name], right[name]) for name in names]


def temporal_stability(images: list[np.ndarray], flow_max_px: float, intensity_difference_max: float, flow_width: int = 0, temporal_stride: int = 1) -> tuple[list[np.ndarray], list[int]]:
    """Return per-frame fraction of neighbouring comparisons judged stable.

    Farneback flow is intentionally only an image-space cue.  It is not visual
    odometry and cannot establish a rigid walker or a ground plane.
    """
    if temporal_stride < 1:
        raise ValueError("--temporal-stride must be positive")
    if len(images) <= temporal_stride:
        shape = images[0].shape[:2] if images else (0, 0)
        return [np.zeros(shape, dtype=np.float32) for _ in images], [0 for _ in images]
    gray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
    original_shape = gray[0].shape
    if flow_width < 0:
        raise ValueError("--flow-width must be non-negative")
    if flow_width and gray[0].shape[1] > flow_width:
        flow_height = max(1, int(round(gray[0].shape[0] * flow_width / gray[0].shape[1])))
        gray = [cv2.resize(image, (flow_width, flow_height), interpolation=cv2.INTER_AREA) for image in gray]
    totals = [np.zeros(original_shape, dtype=np.float32) for _ in images]
    counts = [0 for _ in images]
    for index in range(len(gray) - temporal_stride):
        flow = cv2.calcOpticalFlowFarneback(
            gray[index], gray[index + temporal_stride], None, 0.5, 3, 21, 3, 5, 1.2, 0,
        )
        flow_magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
        intensity_difference = cv2.absdiff(gray[index], gray[index + temporal_stride]).astype(np.float32)
        stable = ((flow_magnitude <= flow_max_px) & (intensity_difference <= intensity_difference_max)).astype(np.float32)
        if stable.shape != original_shape:
            stable = cv2.resize(stable, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        totals[index] += stable
        totals[index + temporal_stride] += stable
        counts[index] += 1
        counts[index + temporal_stride] += 1
    return [total / count if count else np.zeros_like(total) for total, count in zip(totals, counts)], counts


def border_supported_components(stable_mask: np.ndarray, minimum_component_area_px: int, border_anchor_y_fraction: float) -> np.ndarray:
    """Keep stable connected components supported at lower/side image borders.

    The border rule is an explicit camera-mount prior, not a semantic label.
    """
    binary = (stable_mask.astype(np.uint8) * 255)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = binary.shape
    anchor_y = int(round(height * border_anchor_y_fraction))
    output = np.zeros_like(binary)
    for label in range(1, labels_count):
        x, y, component_width, component_height, area = stats[label]
        touches_bottom = y + component_height >= height
        touches_side_low = (x == 0 or x + component_width >= width) and y + component_height >= anchor_y
        if area >= minimum_component_area_px and (touches_bottom or touches_side_low):
            output[labels == label] = 255
    return output


def make_candidate(stability_fraction: np.ndarray, comparison_count: int, *, minimum_stability_fraction: float, minimum_component_area_px: int, maximum_mask_fraction: float, border_anchor_y_fraction: float, allowed_pixels: np.ndarray | None = None) -> CandidateMask:
    shape = stability_fraction.shape
    empty = np.zeros(shape, dtype=np.uint8)
    if comparison_count == 0:
        return CandidateMask("unavailable", empty, ["insufficient_temporal_neighbours"], {"comparison_count": 0, "mask_fraction": 0.0, "mean_stability_fraction": None})
    stable = stability_fraction >= minimum_stability_fraction
    if allowed_pixels is not None:
        if allowed_pixels.shape != stable.shape:
            raise ValueError("additional candidate mask must match image shape")
        stable &= allowed_pixels.astype(bool)
    mask = border_supported_components(stable, minimum_component_area_px, border_anchor_y_fraction)
    fraction = float((mask > 0).mean())
    mean_stability = float(stability_fraction[mask > 0].mean()) if np.any(mask) else None
    metrics: dict[str, float | int | None] = {
        "comparison_count": comparison_count,
        "mask_fraction": fraction,
        "mean_stability_fraction": mean_stability,
        "stable_pixel_fraction_before_border_prior": float(stable.mean()),
    }
    if not np.any(mask):
        return CandidateMask("unavailable", mask, ["no_border_supported_stable_component"], metrics)
    if fraction > maximum_mask_fraction:
        return CandidateMask("unavailable", empty, ["stable_region_dominates_image_ambiguous"], metrics)
    return CandidateMask("candidate", mask, [], metrics)


def cross_view_audit(left: CandidateMask, right: CandidateMask, area_ratio_max: float) -> tuple[str, list[str], dict[str, float | bool | None]]:
    left_fraction = float(left.metrics["mask_fraction"] or 0.0)
    right_fraction = float(right.metrics["mask_fraction"] or 0.0)
    if left.status != "candidate" or right.status != "candidate":
        return "unavailable", ["candidate_missing_in_one_view"], {"both_candidate": False, "area_ratio": None}
    ratio = max(left_fraction, right_fraction) / max(min(left_fraction, right_fraction), 1e-9)
    if ratio > area_ratio_max:
        return "unavailable", ["cross_view_area_disagreement"], {"both_candidate": True, "area_ratio": ratio}
    return "candidate", [], {"both_candidate": True, "area_ratio": ratio}


def overlay(image: np.ndarray, mask: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    tint = np.zeros_like(output)
    tint[..., 1] = 210
    selected = mask > 0
    output[selected] = np.rint(0.45 * output[selected].astype(np.float32) + 0.55 * tint[selected].astype(np.float32)).astype(np.uint8)
    cv2.rectangle(output, (0, 0), (min(output.shape[1], 1100), 30), (255, 255, 255), -1)
    cv2.putText(output, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return output


def optional_sam_refiner(model_id: str | None, device_request: str) -> tuple[Any | None, dict[str, Any]]:
    """Load SAM only when explicitly requested; the temporal candidate remains authoritative."""
    if model_id is None:
        return None, {"requested": False, "status": "not_requested"}
    try:
        import torch
        from transformers import SamModel, SamProcessor
    except ImportError as error:
        raise RuntimeError("SAM refinement requested but transformers/torch are unavailable") from error
    device = "cuda" if device_request == "auto" and torch.cuda.is_available() else ("cpu" if device_request == "auto" else device_request)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SAM refinement requested with --sam-device cuda but CUDA is unavailable")
    processor = SamProcessor.from_pretrained(model_id)
    model = SamModel.from_pretrained(model_id).to(device).eval()
    return (processor, model, torch, device), {"requested": True, "status": "loaded", "model_id": model_id, "device": device}


def optional_scene_excluder(model_id: str | None, device_request: str, labels: list[str] | tuple[str, ...]) -> tuple[Any | None, dict[str, Any]]:
    """Load a cached scene model only to exclude nominated scene predictions."""
    if model_id is None:
        return None, {"requested": False, "status": "not_requested"}
    try:
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    except ImportError as error:
        raise RuntimeError("SegFormer scene exclusion requested but transformers/torch are unavailable") from error
    device = "cuda" if device_request == "auto" and torch.cuda.is_available() else ("cpu" if device_request == "auto" else device_request)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SegFormer scene exclusion requested with --segformer-device cuda but CUDA is unavailable")
    processor = SegformerImageProcessor.from_pretrained(model_id, local_files_only=True)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id, local_files_only=True).to(device).eval()
    label_ids: dict[str, int] = {}
    for requested_label in labels:
        matches = [int(index) for index, label in model.config.id2label.items() if str(label).strip().casefold() == requested_label.strip().casefold()]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one {requested_label!r} label in {model_id}, found {matches}")
        label_ids[requested_label.strip().casefold()] = matches[0]
    return (processor, model, torch, device, label_ids), {"requested": True, "status": "loaded", "model_id": model_id, "device": device, "excluded_class_ids": label_ids}


def scene_exclusion_masks(images: list[np.ndarray], backend: Any | None) -> tuple[list[np.ndarray | None], list[dict[str, Any]]]:
    if backend is None:
        return [None for _ in images], [{"status": "not_run"} for _ in images]
    processor, model, torch, device, label_ids = backend
    masks: list[np.ndarray] = []
    metrics: list[dict[str, Any]] = []
    for image in images:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = processor(images=rgb, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = processor.post_process_semantic_segmentation(outputs, target_sizes=[rgb.shape[:2]])[0].detach().cpu().numpy()
        per_label_fraction = {label: float((prediction == class_id).mean()) for label, class_id in label_ids.items()}
        mask = np.isin(prediction, list(label_ids.values()))
        masks.append(mask)
        metrics.append({"status": "used_as_exclusion_only", "semantic_exclusion_fraction": float(mask.mean()), "class_fractions": per_label_fraction})
    return masks, metrics


def permitted_structural_pixels(image: np.ndarray, exclusion_mask: np.ndarray | None, dark_structure_max: int | None) -> tuple[np.ndarray | None, dict[str, float | None]]:
    """Return optional non-scene-class dark-pixel restriction, with no walker identity claim."""
    if dark_structure_max is None:
        return None, {"semantic_exclusion_fraction": None, "dark_structural_fraction": None}
    if exclusion_mask is None:
        raise ValueError("--dark-structure-max requires --segformer-scene-exclusion-model-id")
    if not 0 <= dark_structure_max <= 255:
        raise ValueError("--dark-structure-max must be between 0 and 255")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Expand the unvalidated scene exclusions slightly.  This biases toward
    # omission near hands/floor edges rather than assigning them to walker.
    dilated_exclusion = cv2.dilate(exclusion_mask.astype(np.uint8), np.ones((9, 9), dtype=np.uint8)) > 0
    allowed = (gray <= dark_structure_max) & ~dilated_exclusion
    return allowed, {"semantic_exclusion_fraction": float(dilated_exclusion.mean()), "dark_structural_fraction": float(allowed.mean())}


def sam_refine(image: np.ndarray, candidate: CandidateMask, backend: Any | None) -> tuple[CandidateMask, dict[str, Any]]:
    """Use candidate interior/background points to tighten its edge, never to invent one."""
    if backend is None or candidate.status != "candidate":
        return candidate, {"status": "not_run"}
    processor, model, torch, device = backend
    foreground_yx = np.argwhere(candidate.mask > 0)
    background_yx = np.argwhere(candidate.mask == 0)
    if len(foreground_yx) < 3 or len(background_yx) < 3:
        return candidate, {"status": "not_run", "reason": "insufficient_prompt_pixels"}
    fg = foreground_yx[np.linspace(0, len(foreground_yx) - 1, 3, dtype=np.int64)][:, ::-1]
    bg = background_yx[np.linspace(0, len(background_yx) - 1, 3, dtype=np.int64)][:, ::-1]
    points = np.vstack((fg, bg)).astype(float).tolist()
    labels = [1, 1, 1, 0, 0, 0]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, input_points=[points], input_labels=[labels], return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0][0]
    scores = outputs.iou_scores.detach().cpu().numpy()[0, 0]
    chosen = np.asarray(masks[int(np.argmax(scores))], dtype=bool).astype(np.uint8) * 255
    # SAM output is constrained by overlap with the temporal candidate, so it
    # cannot turn an unrelated prompt completion into a walker claim.
    overlap = float(np.logical_and(chosen > 0, candidate.mask > 0).sum() / max(1, (candidate.mask > 0).sum()))
    if overlap < 0.5:
        return candidate, {"status": "rejected", "reason": "sam_temporal_overlap_below_0.5", "overlap_with_temporal_candidate": overlap}
    refined = CandidateMask(candidate.status, chosen, candidate.reasons, {**candidate.metrics, "sam_mask_fraction": float((chosen > 0).mean())})
    return refined, {"status": "used", "selected_iou_score": float(np.max(scores)), "overlap_with_temporal_candidate": overlap}


def process_view(images: list[np.ndarray], args: argparse.Namespace, sam_backend: Any | None, exclusion_masks: list[np.ndarray | None]) -> tuple[list[CandidateMask], list[dict[str, Any]], list[dict[str, float | None]]]:
    stability, comparison_counts = temporal_stability(images, args.flow_max_px, args.intensity_difference_max, args.flow_width, args.temporal_stride)
    masks: list[CandidateMask] = []
    sam_metrics: list[dict[str, Any]] = []
    structure_metrics: list[dict[str, float | None]] = []
    for image, score, comparison_count, exclusion_mask in zip(images, stability, comparison_counts, exclusion_masks):
        allowed, per_image_structure_metrics = permitted_structural_pixels(image, exclusion_mask, args.dark_structure_max)
        candidate = make_candidate(score, comparison_count, minimum_stability_fraction=args.minimum_stability_fraction, minimum_component_area_px=args.minimum_component_area_px, maximum_mask_fraction=args.maximum_mask_fraction, border_anchor_y_fraction=args.border_anchor_y_fraction, allowed_pixels=allowed)
        refined, refined_metrics = sam_refine(image, candidate, sam_backend)
        masks.append(refined)
        sam_metrics.append(refined_metrics)
        structure_metrics.append(per_image_structure_metrics)
    return masks, sam_metrics, structure_metrics


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    if not 0.0 < args.minimum_stability_fraction <= 1.0 or not 0.0 < args.maximum_mask_fraction <= 1.0:
        raise ValueError("stability and maximum-mask fractions must be in (0, 1]")
    pairs = paired_paths(args.left_dir, args.right_dir, args.max_frames)
    left_images = [cv2.imread(str(left_path), cv2.IMREAD_COLOR) for _, left_path, _ in pairs]
    right_images = [cv2.imread(str(right_path), cv2.IMREAD_COLOR) for _, _, right_path in pairs]
    if any(image is None for image in left_images + right_images):
        raise RuntimeError("cannot read one or more pair images")
    if len({image.shape[:2] for image in left_images + right_images}) != 1:
        raise RuntimeError("all input images must have the same shape per short-window run")
    sam_backend, sam_load = optional_sam_refiner(args.sam_model_id, args.sam_device)
    scene_backend, scene_load = optional_scene_excluder(args.segformer_scene_exclusion_model_id, args.segformer_device, args.segformer_exclusion_labels)
    left_scene, left_scene_metrics = scene_exclusion_masks(left_images, scene_backend)
    right_scene, right_scene_metrics = scene_exclusion_masks(right_images, scene_backend)
    left_masks, left_sam, left_structure = process_view(left_images, args, sam_backend, left_scene)
    right_masks, right_sam, right_structure = process_view(right_images, args, sam_backend, right_scene)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "masks" / "left").mkdir(parents=True)
    (args.output_dir / "masks" / "right").mkdir(parents=True)
    (args.output_dir / "visualizations").mkdir(parents=True)
    if scene_backend is not None:
        (args.output_dir / "scene_exclusion_masks" / "left").mkdir(parents=True)
        (args.output_dir / "scene_exclusion_masks" / "right").mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for index, (name, _, _) in enumerate(pairs):
        cross_status, cross_reasons, cross_metrics = cross_view_audit(left_masks[index], right_masks[index], args.cross_view_area_ratio_max)
        pair_id = int(Path(name).stem.removeprefix("pair_"))
        cv2.imwrite(str(args.output_dir / "masks" / "left" / name), left_masks[index].mask)
        cv2.imwrite(str(args.output_dir / "masks" / "right" / name), right_masks[index].mask)
        if scene_backend is not None:
            cv2.imwrite(str(args.output_dir / "scene_exclusion_masks" / "left" / name), left_scene[index].astype(np.uint8) * 255)
            cv2.imwrite(str(args.output_dir / "scene_exclusion_masks" / "right" / name), right_scene[index].astype(np.uint8) * 255)
        left_panel = overlay(left_images[index], left_masks[index].mask, f"left {left_masks[index].status}; candidate only")
        right_panel = overlay(right_images[index], right_masks[index].mask, f"right {right_masks[index].status}; candidate only")
        cv2.imwrite(str(args.output_dir / "visualizations" / name), np.hstack((left_panel, right_panel)))
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "pair_id": pair_id,
            "frame_id": name,
            "pair_status": cross_status,
            "pair_reasons": cross_reasons,
            "cross_view_audit": {**cross_metrics, "geometric_identity_validation": "not_attempted"},
            "views": {
                "left": {"mask_status": left_masks[index].status, "reasons": left_masks[index].reasons, "temporal_stability_metrics": left_masks[index].metrics, "structural_candidate_metrics": left_structure[index], "scene_exclusion_metrics": left_scene_metrics[index], "sam_refinement_metrics": left_sam[index], "mask_path": str((args.output_dir / "masks" / "left" / name).resolve())},
                "right": {"mask_status": right_masks[index].status, "reasons": right_masks[index].reasons, "temporal_stability_metrics": right_masks[index].metrics, "structural_candidate_metrics": right_structure[index], "scene_exclusion_metrics": right_scene_metrics[index], "sam_refinement_metrics": right_sam[index], "mask_path": str((args.output_dir / "masks" / "right" / name).resolve())},
            },
            "interpretation": "walker_occlusion_candidate only; not walker semantic truth, ground, support, contact, or gait evidence",
        })
    with (args.output_dir / "walker_occlusion_candidates.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    reason_counts: dict[str, int] = {}
    for row in rows:
        for reason in row["pair_reasons"] + row["views"]["left"]["reasons"] + row["views"]["right"]["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    summary = {
        "schema_version": SCHEMA_VERSION,
        "frame_count": len(rows),
        "pair_candidate_count": sum(row["pair_status"] == "candidate" for row in rows),
        "pair_unavailable_count": sum(row["pair_status"] == "unavailable" for row in rows),
        "reason_counts": reason_counts,
        "sam_load": sam_load,
        "scene_segmentation_load": scene_load,
        "interpretation_boundary": "Image-space temporal candidate diagnostics only. No output establishes walker identity, ground identity, physical contact, support state, or metric geometry.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {"left_dir": str(args.left_dir.resolve()), "right_dir": str(args.right_dir.resolve()), "frames": len(pairs)},
        "parameters": vars(args),
        "method": "short-window Farneback low-motion plus intensity stability; optional SegFormer person-exclusion and dark-structure restriction; viewport-border camera-mount prior; optional prompt-constrained SAM boundary refinement; non-geometric cross-view area audit",
        "invariants": ["does not modify pose, association, triangulation, or ground state", "stable scene pixels are explicitly ambiguous", "SegFormer scene predictions are exclusion candidates, not scene or walker truth", "SAM cannot create a candidate without temporal candidate input"],
        "interpretation_boundary": summary["interpretation_boundary"],
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
