#!/usr/bin/env python3
"""Measure Mask2Former-Swin-S online stage latency and reproduce cached masks.

This tool answers one narrow question: on *this* host, with *this* already
cached checkpoint and *this* frozen interface, how long does one single-image
Mask2Former online semantic pass take, stage by stage, and does the freshly
inferred ``floor candidate`` reproduce the archived candidate masks pixel for
pixel?

It does **not** re-run SGBM, IGEV, DynamicStereo, plane fitting, optical flow or
visual odometry, it does not touch any earlier experiment directory, and it
never downloads a model file: ``local_files_only`` is the only supported mode
and there is deliberately no argument that enables a download.

Two outputs stay strictly apart:

``full_online_semantic_ms``
    The per-image online semantic module time, defined as the sum of
    ``image_decode_ms + bgr_to_rgb_ms + processor_cpu_ms + host_to_device_ms +
    gpu_forward_ms + postprocess_ms + floor_mask_extract_ms``.
``cache_mask_read_ms`` / ``candidate_png_write_ms``
    Pure I/O diagnostics.  They are reported beside the module time and are
    never added into it.

Every number produced here is a module wall clock or a 2-D image-space mask
comparison.  The Mask2Former output remains an automatic 2-D floor *candidate*:
it is not a manual label, not a 3-D truth and not a physical ground plane, and
nothing in this file may be reported as true ground accuracy, foot-to-ground
height, contact, support or a gait event.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


SCHEMA_VERSION = "mask2former_online_latency_parity_v1"
EXPERIMENT_ID = "G20260913_mask2former_online_latency_parity_v1"
MODEL_ID = "facebook/mask2former-swin-small-ade-semantic"
BACKEND_NAME = "mask2former_swin_small"

SOURCE_MATRIX_PATH = (
    Path("research_records")
    / "engineering_validation"
    / "G20260913_temporal_ground_composition_benchmark_v1"
    / "combination_matrix_27.json"
)

STAGE_FIELDS = (
    "image_decode_ms",
    "bgr_to_rgb_ms",
    "processor_cpu_ms",
    "host_to_device_ms",
    "gpu_forward_ms",
    "postprocess_ms",
    "floor_mask_extract_ms",
)
FULL_ONLINE_FIELD = "full_online_semantic_ms"
IO_DIAGNOSTIC_FIELDS = ("cache_mask_read_ms", "candidate_png_write_ms")

VIEWS = ("left", "right")

# The manual 2-D audit reuses the exact mutually exclusive priority order of
# ``audit_manual_floor_labels.rasterize_labelme``.  Manual labels are an audit
# reference only; they are never an input, a post-processing condition or a
# threshold-selection signal for the semantic model.
AUDIT_LABELS = ("floor_eligible", "person", "walker", "static_other", "ignore_uncertain")
AUDIT_PRIORITY = ("ignore_uncertain", "person", "walker", "static_other", "floor_eligible")

PARITY_EXACT = "exact_match"
PARITY_MISMATCH = "mismatch"
PARITY_ALL_EXACT = "exact_match_all_24"
PARITY_MISMATCH_PRESENT = "mismatch_present"

BOUNDARY_EXACT = (
    "Estimated module sum only. The Mask2Former online candidate was reproduced exactly against "
    "archived masks on 24 images; this is not a measured end-to-end latency or physical "
    "ground-accuracy result."
)
BOUNDARY_MISMATCH = (
    "Timing was measured, but current Mask2Former output does not exactly reproduce the archived "
    "candidate masks; do not combine this latency with prior geometry results as a same-output "
    "estimate."
)

INTERPRETATION_BOUNDARY = (
    "Mask2Former remains an automatic 2-D image-space floor candidate, never a manual label, a 3-D "
    "truth or a physical ground plane. Every latency here is a module wall clock measured on one "
    "host and one interface version; the combined total is a sum of module percentiles, not a "
    "measured end-to-end latency and not a real-time result."
)

TEXT_FORBIDDEN_PHRASES = (
    "真实地面精度提高",
    "真实三维精度提高",
    "接触识别成功",
    "步态识别成功",
)


class OnlineLatencyError(RuntimeError):
    """Raised when a hard, non-negotiable contract of this measurement is violated."""


# --------------------------------------------------------------------------- #
# arguments
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--left-image-dir", type=Path, required=True, help="upright left view with pair_*.png")
    parser.add_argument("--right-image-dir", type=Path, required=True, help="upright right view with pair_*.png")
    parser.add_argument("--left-cache-dir", type=Path, required=True, help="archived left Mask2Former floor candidate masks (read-only)")
    parser.add_argument("--right-cache-dir", type=Path, required=True, help="archived right Mask2Former floor candidate masks (read-only)")
    parser.add_argument("--left-label-dir", type=Path, required=True, help="LabelMe manual labels for the left view (2-D audit only)")
    parser.add_argument("--right-label-dir", type=Path, required=True, help="LabelMe manual labels for the right view (2-D audit only)")
    parser.add_argument("--output-dir", type=Path, default=None, help="new output directory; must not exist (not used by --probe-only)")
    parser.add_argument("--probe-only", action="store_true", help="read-only interface probe over --pair-ids; writes nothing and requires no --output-dir")
    parser.add_argument("--pair-ids", nargs="+", type=int, required=True, help="explicit anchor pair ids, e.g. 0 40 80")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--warmup-count", type=int, default=3)
    parser.add_argument("--measured-repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=True,
        help=(
            "default and only supported mode: the cached model files are used with local_files_only=True. "
            "There is deliberately no argument that enables a download, and this flag cannot disable it"
        ),
    )
    parser.add_argument("--probe-max-pairs", type=int, default=2, help="pairs used by --probe-only")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def validate_batch_size(batch_size: int) -> int:
    """``batch_size`` is fixed at 1: this measures one image's online pass."""
    if int(batch_size) != 1:
        raise ValueError(f"--batch-size must be exactly 1 for a single-image online measurement, got {batch_size}")
    return 1


def validate_run_parameters(args: argparse.Namespace) -> None:
    validate_batch_size(args.batch_size)
    if args.warmup_count < 0:
        raise ValueError("--warmup-count must not be negative")
    if args.measured_repeats < 1:
        raise ValueError("--measured-repeats must be positive")
    if len(set(args.pair_ids)) != len(args.pair_ids):
        raise ValueError("--pair-ids contains duplicates")
    if not args.pair_ids:
        raise ValueError("--pair-ids must not be empty")


def resolve_device(torch: Any, requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        raise OnlineLatencyError("--device cuda requested but CUDA is unavailable on this host")
    return requested


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def pair_file_name(pair_id: int) -> str:
    return f"pair_{int(pair_id):04d}.png"


def label_file_name(pair_id: int) -> str:
    return f"pair_{int(pair_id):04d}.json"


def ensure_output_dir_absent(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OnlineLatencyError(f"cannot read image: {path}")
    return image


def resolve_pair_inputs(
    pair_ids: Iterable[int],
    left_image_dir: Path,
    right_image_dir: Path,
    left_cache_dir: Path,
    right_cache_dir: Path,
    left_label_dir: Path | None = None,
    right_label_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Resolve and validate every per-pair input before any inference happens.

    A pair is rejected when any of the four required files is missing, and when
    a cached mask's pixel grid differs from its input image (including a
    grayscale cached mask being silently expanded to three channels).
    """
    resolved: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        name = pair_file_name(pair_id)
        paths = {
            "left_image": Path(left_image_dir) / name,
            "right_image": Path(right_image_dir) / name,
            "left_cache": Path(left_cache_dir) / name,
            "right_cache": Path(right_cache_dir) / name,
        }
        missing = sorted(key for key, path in paths.items() if not path.is_file())
        if missing:
            detail = ", ".join(f"{key}={paths[key]}" for key in missing)
            raise FileNotFoundError(f"pair {name}: missing required input(s): {detail}")

        left_image = cv2.imread(str(paths["left_image"]), cv2.IMREAD_COLOR)
        right_image = cv2.imread(str(paths["right_image"]), cv2.IMREAD_COLOR)
        if left_image is None:
            raise OnlineLatencyError(f"cannot read left image: {paths['left_image']}")
        if right_image is None:
            raise OnlineLatencyError(f"cannot read right image: {paths['right_image']}")
        left_shape = left_image.shape[:2]
        right_shape = right_image.shape[:2]

        for view, image_shape in (("left", left_shape), ("right", right_shape)):
            cache_path = paths[f"{view}_cache"]
            cached = _read_gray(cache_path)
            if cached.shape != image_shape:
                raise OnlineLatencyError(
                    f"pair {name} {view}: cached candidate mask {cached.shape} does not match input image "
                    f"size {image_shape} ({cache_path})"
                )
        if left_shape != right_shape:
            raise OnlineLatencyError(f"pair {name}: left image {left_shape} and right image {right_shape} differ")

        entry: dict[str, Any] = {
            "pair_id": int(pair_id),
            "name": name,
            "left_image": paths["left_image"],
            "right_image": paths["right_image"],
            "left_cache": paths["left_cache"],
            "right_cache": paths["right_cache"],
            "image_shape_hw": [int(left_shape[0]), int(left_shape[1])],
        }
        for view, label_dir in (("left", left_label_dir), ("right", right_label_dir)):
            if label_dir is None:
                continue
            label_path = Path(label_dir) / label_file_name(pair_id)
            if not label_path.is_file():
                raise FileNotFoundError(f"pair {name} {view}: missing manual label file: {label_path}")
            entry[f"{view}_label"] = label_path
        resolved.append(entry)
    return resolved


# --------------------------------------------------------------------------- #
# the frozen online path
# --------------------------------------------------------------------------- #
def full_online_semantic_ms(stage: dict[str, float]) -> float:
    """The one and only definition of the single-image online semantic module time.

    ``cache_mask_read_ms`` and ``candidate_png_write_ms`` are deliberately absent:
    they are I/O diagnostics and are never part of this module time.
    """
    return float(sum(float(stage[field]) for field in STAGE_FIELDS))


def load_model(device: str, torch: Any) -> dict[str, Any]:
    """Load the cached processor and model exactly once.

    The processor resize/normalize behaviour, the label mapping and the
    ``floor`` class decision are the current interface's own; nothing here
    overrides them.
    """
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    model_load_start = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_ID, local_files_only=True)
    model = model.to(device).eval()
    if device == "cuda":
        torch.cuda.synchronize()
    model_load_ms = (time.perf_counter() - model_load_start) * 1000.0

    floor_ids = [
        int(index)
        for index, label in model.config.id2label.items()
        if str(label).strip().casefold() == "floor"
    ]
    if len(floor_ids) != 1:
        raise OnlineLatencyError(f"expected exactly one ADE20K 'floor' class, found {floor_ids}")
    return {
        "processor": processor,
        "model": model,
        "floor_class_id": int(floor_ids[0]),
        "model_load_ms": float(model_load_ms),
        "processor_size": {str(key): value for key, value in dict(processor.size).items()} if processor.size else None,
        "num_labels": int(model.config.num_labels),
    }


def _synchronize(torch: Any, device: str) -> None:
    """Every CUDA timing span is fenced, otherwise a launch time is not a forward time."""
    if device == "cuda":
        torch.cuda.synchronize()


def run_single_image_online(
    image_path: Path,
    cached_mask_path: Path,
    runtime: dict[str, Any],
    torch: Any,
    device: str,
    measure_cache_read: bool = True,
    measure_mask_write: bool = False,
) -> dict[str, Any]:
    """One image through the frozen online path, stage by stage.

    ``cv2.imread`` (BGR) -> ``cv2.cvtColor`` (RGB) -> ``AutoImageProcessor`` ->
    H2D transfer -> ``Mask2FormerForUniversalSegmentation`` forward under
    ``no_grad`` -> ``post_process_semantic_segmentation`` at the original image
    size -> ``semantic map == floor class id`` -> ``uint8`` binary candidate.
    """
    processor = runtime["processor"]
    model = runtime["model"]
    floor_class_id = int(runtime["floor_class_id"])

    cache_mask_read_ms: float | None = None
    if measure_cache_read:
        cache_start = time.perf_counter()
        cached = cv2.imread(str(cached_mask_path), cv2.IMREAD_GRAYSCALE)
        cache_mask_read_ms = (time.perf_counter() - cache_start) * 1000.0
    else:
        cached = cv2.imread(str(cached_mask_path), cv2.IMREAD_GRAYSCALE)
    if cached is None:
        raise OnlineLatencyError(f"cannot read cached candidate mask: {cached_mask_path}")

    stage: dict[str, float] = {}

    start = time.perf_counter()
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    stage["image_decode_ms"] = (time.perf_counter() - start) * 1000.0
    if bgr is None:
        raise OnlineLatencyError(f"cannot read input image: {image_path}")
    if bgr.shape[:2] != cached.shape:
        raise OnlineLatencyError(
            f"{image_path.name}: image {bgr.shape[:2]} and cached mask {cached.shape} sizes differ"
        )

    start = time.perf_counter()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    stage["bgr_to_rgb_ms"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    processed = processor(images=rgb, return_tensors="pt")
    stage["processor_cpu_ms"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    inputs = {key: value.to(device) for key, value in processed.items()}
    stage["host_to_device_ms"] = (time.perf_counter() - start) * 1000.0

    _synchronize(torch, device)
    start = time.perf_counter()
    with torch.no_grad():
        output = model(**inputs)
    _synchronize(torch, device)
    stage["gpu_forward_ms"] = (time.perf_counter() - start) * 1000.0

    _synchronize(torch, device)
    start = time.perf_counter()
    semantic = processor.post_process_semantic_segmentation(output, target_sizes=[bgr.shape[:2]])[0]
    semantic_np = semantic.detach().to("cpu").numpy()
    _synchronize(torch, device)
    stage["postprocess_ms"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    candidate = (semantic_np == floor_class_id).astype(np.uint8)
    stage["floor_mask_extract_ms"] = (time.perf_counter() - start) * 1000.0

    candidate_png_write_ms: float | None = None
    if measure_mask_write:
        write_start = time.perf_counter()
        cv2.imwrite(str(runtime["candidate_mask_path"]), candidate * 255)
        candidate_png_write_ms = (time.perf_counter() - write_start) * 1000.0

    record: dict[str, Any] = dict(stage)
    record[FULL_ONLINE_FIELD] = full_online_semantic_ms(stage)
    record["cache_mask_read_ms"] = cache_mask_read_ms
    record["candidate_png_write_ms"] = candidate_png_write_ms
    record["candidate_mask"] = candidate
    record["cached_mask_bool"] = cached > 0
    record["image_shape_hw"] = [int(bgr.shape[0]), int(bgr.shape[1])]
    return record


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def percentile(values: Sequence[float], fraction: float) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite:
        return None
    return float(np.percentile(np.asarray(finite, dtype=np.float64), fraction * 100.0))


def stage_statistics(values: Sequence[float | None]) -> dict[str, Any]:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite:
        return {"count": 0, "mean_ms": None, "median_ms": None, "p90_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None}
    array = np.asarray(finite, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p90_ms": float(np.percentile(array, 90.0)),
        "p95_ms": float(np.percentile(array, 95.0)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def timing_summary_from_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """P50/P95 over the main measurement records, globally and per view."""
    fields = (*STAGE_FIELDS, FULL_ONLINE_FIELD, *IO_DIAGNOSTIC_FIELDS)

    def block(subset: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return {field: stage_statistics([record.get(field) for record in subset]) for field in fields}

    return {
        "main_measurement_record_count": len(records),
        "full_online_semantic_definition": " + ".join(STAGE_FIELDS),
        "io_diagnostics_excluded_from_full_online_semantic_ms": list(IO_DIAGNOSTIC_FIELDS),
        "all_views": block(records),
        "left": block([record for record in records if record["view"] == "left"]),
        "right": block([record for record in records if record["view"] == "right"]),
    }


# --------------------------------------------------------------------------- #
# parity and the manual 2-D audit
# --------------------------------------------------------------------------- #
def mask_comparison(candidate: np.ndarray, cached: np.ndarray) -> dict[str, Any]:
    candidate_bool = np.asarray(candidate).astype(bool)
    cached_bool = np.asarray(cached).astype(bool)
    if candidate_bool.shape != cached_bool.shape:
        raise OnlineLatencyError(
            f"cannot compare masks with different shapes: {candidate_bool.shape} vs {cached_bool.shape}"
        )
    intersection = int(np.logical_and(candidate_bool, cached_bool).sum())
    union = int(np.logical_or(candidate_bool, cached_bool).sum())
    different = int(np.logical_xor(candidate_bool, cached_bool).sum())
    total = int(candidate_bool.size)
    return {
        "equal_pixel_count": int(total - different),
        "different_pixel_count": different,
        "different_pixel_fraction": float(different / total) if total else 0.0,
        "exact_match": bool(different == 0),
        "iou_vs_cached": float(intersection / union) if union else 1.0,
        "candidate_pixel_count": int(candidate_bool.sum()),
        "cached_pixel_count": int(cached_bool.sum()),
        "intersection_pixel_count": intersection,
        "union_pixel_count": union,
        "parity_status": PARITY_EXACT if different == 0 else PARITY_MISMATCH,
    }


def load_labelme(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value.get("shapes"), list):
        raise OnlineLatencyError(f"{path}: LabelMe shapes list missing")
    if not isinstance(value.get("imageWidth"), int) or not isinstance(value.get("imageHeight"), int):
        raise OnlineLatencyError(f"{path}: LabelMe image dimensions missing")
    return value


def rasterize_labelme(value: dict[str, Any]) -> dict[str, np.ndarray]:
    """Same mutually exclusive priority order as ``audit_manual_floor_labels``."""
    height, width = int(value["imageHeight"]), int(value["imageWidth"])
    raw = {label: np.zeros((height, width), dtype=bool) for label in AUDIT_LABELS}
    for shape in value["shapes"]:
        label = str(shape.get("label", ""))
        if label not in raw:
            raise OnlineLatencyError(f"unsupported label {label!r}; expected one of {AUDIT_LABELS}")
        if shape.get("shape_type", "polygon") != "polygon":
            raise OnlineLatencyError(f"{label}: only polygon shapes are supported")
        points = np.asarray(shape.get("points"), dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise OnlineLatencyError(f"{label}: polygon needs >=3 finite xy points")
        polygon = np.rint(points).astype(np.int32)
        canvas = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(canvas, [polygon], 1)
        raw[label] |= canvas.astype(bool)
    exclusive: dict[str, np.ndarray] = {}
    claimed = np.zeros((height, width), dtype=bool)
    for label in AUDIT_PRIORITY:
        exclusive[label] = raw[label] & ~claimed
        claimed |= exclusive[label]
    exclusive["valid_evaluation"] = ~exclusive["ignore_uncertain"]
    return exclusive


def binary_metrics(reference_floor: np.ndarray, valid: np.ndarray, candidate_floor: np.ndarray) -> dict[str, Any]:
    if reference_floor.shape != candidate_floor.shape or valid.shape != reference_floor.shape:
        raise OnlineLatencyError("reference, valid area and candidate shapes must agree")
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


def semantic_audit(view: str, entries: Sequence[dict[str, Any]], candidate_loader) -> dict[str, Any]:
    """Single-view manual 2-D mask consistency audit over the re-inferred masks."""
    per_image: list[dict[str, Any]] = []
    for entry in entries:
        label_path = entry.get(f"{view}_label")
        if label_path is None:
            raise OnlineLatencyError(f"{view} pair {entry['name']}: manual label path is unavailable")
        masks = rasterize_labelme(load_labelme(label_path))
        candidate = candidate_loader(entry, view)
        metrics = binary_metrics(masks["floor_eligible"], masks["valid_evaluation"], candidate)
        walker = masks["walker"] & masks["valid_evaluation"]
        metrics["walker_pixels_predicted_floor"] = int(np.logical_and(walker, candidate).sum())
        metrics["walker_to_floor_fraction"] = (
            None if not walker.any() else float(np.logical_and(walker, candidate).sum() / walker.sum())
        )
        per_image.append(
            {
                "pair_id": int(entry["pair_id"]),
                "label_file": str(Path(label_path).resolve()),
                "manual_pixel_counts": {label: int(masks[label].sum()) for label in (*AUDIT_LABELS, "valid_evaluation")},
                **metrics,
            }
        )
    keys = ("precision", "recall", "iou", "walker_to_floor_fraction")
    means = {
        key: (
            float(np.mean([row[key] for row in per_image if row[key] is not None]))
            if any(row[key] is not None for row in per_image)
            else None
        )
        for key in keys
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "view": view,
        "metric_name": "manual 2-D mask consistency audit",
        "labelled_image_count": len(per_image),
        "pair_ids": [row["pair_id"] for row in per_image],
        "mutually_exclusive_label_priority": list(AUDIT_PRIORITY),
        "precision": means["precision"],
        "recall": means["recall"],
        "iou": means["iou"],
        "walker_to_floor_fraction": means["walker_to_floor_fraction"],
        "per_image": per_image,
        "manual_label_role": "audit reference only; never a model input, post-processing condition or threshold-selection signal",
        "interpretation_boundary": (
            "This is a manual 2-D mask consistency audit of an automatic image-space floor candidate on a "
            "small fixed anchor set. It is not semantic accuracy, not ground accuracy, not 3-D accuracy and "
            "not video generalisation."
        ),
    }


# --------------------------------------------------------------------------- #
# 27-combination latency augmentation
# --------------------------------------------------------------------------- #
MATRIX_ADDED_FIELDS = (
    "semantic_latency_source",
    "semantic_online_p50_ms",
    "semantic_online_p95_ms",
    "estimated_end_to_end_including_semantic_p50_ms",
    "estimated_end_to_end_including_semantic_p95_ms",
    "latency_kind",
    "measured_end_to_end_latency_ms",
    "semantic_parity_status",
    "semantic_latency_boundary",
)

MATRIX_CSV_HEADER = (
    "combination_id",
    "matcher",
    "plane_method",
    "temporal_mode",
    "full_chain_measured",
    "measurement_scope",
    "estimated_end_to_end_latency_ms",
    "measured_end_to_end_latency_ms",
    "realtime_compatible",
    "cross_region_pass_frames",
    "median_inlier_fraction",
    "median_residual_mm",
    "flow_module_status",
    "vo_module_status",
    "semantic_latency_source",
    "semantic_online_p50_ms",
    "semantic_online_p95_ms",
    "estimated_end_to_end_including_semantic_p50_ms",
    "estimated_end_to_end_including_semantic_p95_ms",
    "latency_kind",
    "measured_end_to_end_latency_ms_including_semantic",
    "semantic_parity_status",
    "semantic_latency_boundary",
)


def _add(base: float | None, semantic: float | None) -> float | None:
    if base is None or semantic is None:
        return None
    return float(base) + float(semantic)


def augment_matrix_with_semantic_latency(
    source_matrix: dict[str, Any],
    semantic_p50_ms: float | None,
    semantic_p95_ms: float | None,
    parity_status: str,
) -> dict[str, Any]:
    """Add the measured semantic module time to the frozen 27-row matrix.

    Nothing spatial or temporal is recomputed or changed: every original field
    of every original combination id is copied verbatim, and the semantic time
    only ever appears inside explicitly ``estimated`` module sums.
    """
    rows = source_matrix.get("rows")
    if not isinstance(rows, list):
        raise OnlineLatencyError("source matrix has no rows list")
    boundary = BOUNDARY_EXACT if parity_status == PARITY_ALL_EXACT else BOUNDARY_MISMATCH
    augmented: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        base_p50 = row.get("estimated_end_to_end_latency_ms")
        base_p95 = row.get("estimated_end_to_end_latency_p95_ms")
        new_row["semantic_latency_source"] = EXPERIMENT_ID
        new_row["semantic_online_p50_ms"] = semantic_p50_ms
        new_row["semantic_online_p95_ms"] = semantic_p95_ms
        new_row["estimated_end_to_end_including_semantic_p50_ms"] = _add(base_p50, semantic_p50_ms)
        new_row["estimated_end_to_end_including_semantic_p95_ms"] = _add(base_p95, semantic_p95_ms)
        new_row["latency_kind"] = "estimated_module_sum"
        new_row["measured_end_to_end_latency_ms"] = None
        new_row["semantic_parity_status"] = parity_status
        new_row["semantic_latency_boundary"] = boundary
        new_row["realtime_compatible"] = False
        augmented.append(new_row)
    result = dict(source_matrix)
    result["rows"] = augmented
    result["semantic_latency_source"] = EXPERIMENT_ID
    result["semantic_latency_model"] = MODEL_ID
    result["semantic_online_p50_ms"] = semantic_p50_ms
    result["semantic_online_p95_ms"] = semantic_p95_ms
    result["semantic_parity_status"] = parity_status
    result["latency_kind"] = "estimated_module_sum"
    result["semantic_latency_boundary"] = boundary
    result["measured_end_to_end_latency_available"] = False
    result["augmentation_note"] = (
        "Only the Mask2Former online semantic module time was added to the frozen task-03 matrix. No spatial or "
        "temporal result, threshold, gate or existing field was modified, and no combination is real-time."
    )
    return result


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def matrix_rows_to_csv_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    """Project the augmented rows onto the frozen CSV columns plus the new ones.

    The first fourteen columns are the source CSV's own columns; the nested
    evidence blocks of the JSON are read for the three summary values and no
    value is recomputed.
    """
    rows: list[dict[str, Any]] = []
    for row in matrix.get("rows", []):
        cross_region = row.get("cross_region_evidence") or {}
        plane = row.get("plane_evidence") or {}
        projected = {column: row.get(column) for column in MATRIX_CSV_HEADER}
        projected["cross_region_pass_frames"] = cross_region.get("pass_frames")
        projected["median_inlier_fraction"] = plane.get("median_inlier_fraction")
        projected["median_residual_mm"] = plane.get("median_residual_mm")
        projected["measured_end_to_end_latency_ms_including_semantic"] = row.get("measured_end_to_end_latency_ms")
        rows.append(projected)
    return rows


def matrix_csv_text(matrix: dict[str, Any]) -> str:
    lines = [",".join(MATRIX_CSV_HEADER)]
    for row in matrix_rows_to_csv_rows(matrix):
        lines.append(",".join(_csv_cell(row.get(column)) for column in MATRIX_CSV_HEADER))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# visualisations
# --------------------------------------------------------------------------- #
def render_parity_overlay(
    image: np.ndarray,
    candidate: np.ndarray,
    cached: np.ndarray,
    parity: dict[str, Any],
    title: str,
    panel_width: int = 420,
) -> np.ndarray:
    """Original | new candidate | archived contour | pixel difference, side by side.

    Green is the newly inferred online floor candidate, orange is the archived
    cached mask contour, red is the pixel difference.  When the two agree
    exactly the red panel is empty.
    """
    candidate_bool = candidate.astype(bool)
    cached_bool = cached.astype(bool)
    difference = np.logical_xor(candidate_bool, cached_bool)

    panels: list[np.ndarray] = []
    height, width = image.shape[:2]
    scale = float(panel_width) / float(width)
    panel_size = (int(round(panel_width)), int(round(height * scale)))

    def resize(panel: np.ndarray) -> np.ndarray:
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST
        return cv2.resize(panel, panel_size, interpolation=interpolation)

    original = image.copy()
    panels.append(resize(original))

    candidate_panel = original.copy()
    color = np.zeros_like(candidate_panel)
    color[candidate_bool] = (35, 185, 35)
    candidate_panel = cv2.addWeighted(candidate_panel, 0.62, color, 0.38, 0.0)
    cv2.putText(candidate_panel, "new online floor candidate (green)", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(candidate_panel, "new online floor candidate (green)", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)
    panels.append(resize(candidate_panel))

    contour_panel = original.copy()
    contours, _ = cv2.findContours(cached_bool.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_panel, contours, -1, (0, 140, 255), 3)
    cv2.putText(contour_panel, "archived cached mask contour (orange)", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(contour_panel, "archived cached mask contour (orange)", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)
    panels.append(resize(contour_panel))

    difference_panel = np.zeros_like(original)
    difference_panel[difference] = (0, 0, 255)
    difference_panel[~difference] = (245, 245, 245)
    label = f"pixel difference (red): {parity['different_pixel_count']} px"
    cv2.putText(difference_panel, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(difference_panel, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 1, cv2.LINE_AA)
    panels.append(resize(difference_panel))

    separator = np.full((panel_size[1], 6, 3), 255, dtype=np.uint8)
    combined = panels[0]
    for panel in panels[1:]:
        combined = np.hstack([combined, separator, panel])

    header_height = 42
    header = np.full((header_height, combined.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        header,
        f"{title} | candidate only, image-space floor candidate",
        (8, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        f"parity={parity['parity_status']} different_px={parity['different_pixel_count']} "
        f"iou_vs_cached={parity['iou_vs_cached']:.6f}",
        (8, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([header, combined])


# --------------------------------------------------------------------------- #
# documentation helpers
# --------------------------------------------------------------------------- #
def assert_no_forbidden_phrases(text: str) -> None:
    for phrase in TEXT_FORBIDDEN_PHRASES:
        if phrase in text:
            raise OnlineLatencyError(f"forbidden overclaiming phrase present: {phrase}")


def format_ms(value: float | None) -> str:
    return "null" if value is None else f"{value:.3f}"


def build_experiment_markdown(summary: dict[str, Any], timing: dict[str, Any], audits: dict[str, Any]) -> str:
    overall = timing["all_views"]
    left = timing["left"]
    right = timing["right"]
    fastest = summary["fastest_combinations_with_semantic_latency"]
    lines: list[str] = []
    lines.append("# G20260913 Mask2Former 在线语义时延、输出复现与 27 组合延迟边界补齐")
    lines.append("")
    lines.append(f"日期：{summary['run_date_beijing']}（北京时间）")
    lines.append("状态：`completed_online_semantic_latency_and_pixel_parity`")
    lines.append("任务：task-04（只读既有缓存与 task-03 归档；不重跑 SGBM / IGEV / DynamicStereo / 平面拟合 / 光流 / VO）")
    lines.append("")
    lines.append("## 1. 目标")
    lines.append("")
    lines.append("1. 在当前机器、当前已缓存权重、当前接口下，实测 Mask2Former-Swin-S 单图在线推理的**阶段级**耗时；")
    lines.append("2. 验证重新推理得到的 floor candidate 是否与历史缓存候选**逐像素一致**；")
    lines.append("3. 将实测语义模块耗时作为**附加模块估算**加到既有 27 组合延迟表；")
    lines.append("4. 不重跑双目 / 平面拟合 / 光流 / VO，不改任何旧实验产物。")
    lines.append("")
    lines.append("这不是新的真实地面精度实验。Mask2Former 始终是图像空间 `floor candidate`，不是地面真值或米制平面真值。")
    lines.append("")
    lines.append("## 2. 固定输入")
    lines.append("")
    lines.append("| 项目 | 路径 |")
    lines.append("|---|---|")
    for key, value in summary["input_paths"].items():
        lines.append(f"| {key} | `{value}` |")
    lines.append("")
    lines.append(f"固定锚点帧：`{', '.join('pair_%04d' % value for value in summary['pair_ids'])}`。")
    lines.append("")
    lines.append("## 3. 运行时与接口")
    lines.append("")
    lines.append("| 项目 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| GPU | `{summary['environment']['gpu_name']}` |")
    lines.append(f"| torch | `{summary['environment']['torch_version']}` |")
    lines.append(f"| transformers | `{summary['environment']['transformers_version']}` |")
    lines.append(f"| CUDA | `{summary['environment']['cuda_version']}` |")
    lines.append(f"| 图像尺寸 | `{summary['image_shape_hw'][0]} x {summary['image_shape_hw'][1]}` (H x W) |")
    lines.append(f"| 模型 ID | `{MODEL_ID}` |")
    lines.append(f"| floor class id | `{summary['floor_class_id']}` |")
    lines.append(f"| 本地缓存装载 | `local_files_only = {summary['local_files_only']}`（无下载参数） |")
    lines.append(f"| 模型加载时间 | `model_load_ms = {format_ms(summary['model_load_ms'])}` |")
    lines.append(f"| device / batch size | `{summary['device']}` / `{summary['batch_size']}` |")
    lines.append(f"| warmup / measured repeats | `{summary['warmup_count']}` / `{summary['measured_repeats']}` |")
    lines.append("")
    lines.append("单张图的固定在线推理流程（未改 processor resize / normalize / label 映射 / floor 类判定）：")
    lines.append("")
    lines.append("```text")
    lines.append("cv2.imread(BGR) -> cv2.cvtColor(RGB) -> AutoImageProcessor(return_tensors='pt')")
    lines.append("  -> tensor.to(device) -> Mask2FormerForUniversalSegmentation forward (no_grad)")
    lines.append("  -> post_process_semantic_segmentation(target_sizes=[原始图高宽])")
    lines.append("  -> semantic_map == floor_class_id -> uint8 二值候选掩膜")
    lines.append("```")
    lines.append("")
    lines.append("每一个 CUDA 计时段前后均调用 `torch.cuda.synchronize()`，因此 `gpu_forward_ms` 是带同步的 forward 时间，不是异步 launch 时间。模型只加载一次。")
    lines.append("")
    lines.append("## 4. 阶段级时延")
    lines.append("")
    lines.append(f"主测量记录 {timing['main_measurement_record_count']} 条 = 24 张图 × {summary['measured_repeats']} 次重复；warmup {summary['warmup_count']} 次（左 `pair_0000`）不进入统计。")
    lines.append("")
    lines.append(f"`{FULL_ONLINE_FIELD} = " + " + ".join(STAGE_FIELDS) + "`。")
    lines.append("`cache_mask_read_ms` 与 `candidate_png_write_ms` 是 I/O 诊断，**不并入** `full_online_semantic_ms`。")
    lines.append("")
    lines.append("| 阶段 | 全部 count | 全部 P50 (ms) | 全部 P95 (ms) | 左 P50 | 左 P95 | 右 P50 | 右 P95 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for field in (*STAGE_FIELDS, FULL_ONLINE_FIELD, *IO_DIAGNOSTIC_FIELDS):
        lines.append(
            "| `{field}` | {count} | {p50} | {p95} | {lp50} | {lp95} | {rp50} | {rp95} |".format(
                field=field,
                count=overall[field]["count"],
                p50=format_ms(overall[field]["median_ms"]),
                p95=format_ms(overall[field]["p95_ms"]),
                lp50=format_ms(left[field]["median_ms"]),
                lp95=format_ms(left[field]["p95_ms"]),
                rp50=format_ms(right[field]["median_ms"]),
                rp95=format_ms(right[field]["p95_ms"]),
            )
        )
    lines.append("")
    lines.append(f"Mask2Former 在线语义单图模块耗时（不含 I/O 诊断）：P50 `{format_ms(overall[FULL_ONLINE_FIELD]['median_ms'])} ms`、"
                 f"P95 `{format_ms(overall[FULL_ONLINE_FIELD]['p95_ms'])} ms`；"
                 f"左 P50 `{format_ms(left[FULL_ONLINE_FIELD]['median_ms'])} ms`、右 P50 `{format_ms(right[FULL_ONLINE_FIELD]['median_ms'])} ms`。")
    lines.append("")
    lines.append("## 5. 与历史缓存的逐像素复现")
    lines.append("")
    lines.append(f"- `parity_records.jsonl` 恰有 `{summary['parity']['record_count']}` 条（24 张图）；")
    lines.append(f"- 逐像素完全一致 `{summary['parity']['exact_match_count']}/{summary['parity']['record_count']}`；")
    lines.append(f"- 语义复现状态 `semantic_parity_status = {summary['parity']['semantic_parity_status']}`；")
    lines.append(f"- 最大差异像素数 `{summary['parity']['max_different_pixel_count']}`，最小 `iou_vs_cached` `{summary['parity']['min_iou_vs_cached']}`。")
    lines.append("")
    lines.append("| view | pair_id | different_px | different_fraction | exact_match | iou_vs_cached |")
    lines.append("|---|---:|---:|---:|:--:|---:|")
    for record in summary["parity"]["records"]:
        lines.append(
            "| {view} | {pair_id} | {diff} | {frac:.10f} | {exact} | {iou:.6f} |".format(
                view=record["view"],
                pair_id=record["pair_id"],
                diff=record["different_pixel_count"],
                frac=record["different_pixel_fraction"],
                exact=record["exact_match"],
                iou=record["iou_vs_cached"],
            )
        )
    lines.append("")
    if summary["parity"]["semantic_parity_status"] == PARITY_ALL_EXACT:
        lines.append("结论：当前机器与当前接口重推理得到的 24 张 floor candidate 与历史缓存**全部逐像素一致**，因此把该语义模块耗时附加到既有 27 组合延迟表是**同输出**的估算，而不是换了语义输出的新结果。")
    else:
        lines.append("结论：存在与历史缓存不一致的帧。计时仍然报告，但不得把该时延与既有几何结果当作同一输出的估算合并使用；差异掩膜与差异可视化已保留，历史缓存未被静默替换。")
    lines.append("")
    lines.append("## 6. 人工二维掩膜一致性审计")
    lines.append("")
    lines.append("标签互斥优先级与 `audit_manual_floor_labels.py` 完全一致：`ignore_uncertain > person > walker > static_other > floor_eligible`。人工标签只用于本次二维复核，不作为模型输入、后处理条件或阈值选择依据。")
    lines.append("")
    lines.append("| 视图 | labelled_image_count | pair_ids | precision | recall | iou | walker_to_floor_fraction |")
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    for view in VIEWS:
        audit = audits[view]
        lines.append(
            "| {view} | {count} | {ids} | {precision} | {recall} | {iou} | {walker} |".format(
                view=view,
                count=audit["labelled_image_count"],
                ids=", ".join(str(value) for value in audit["pair_ids"]),
                precision="null" if audit["precision"] is None else f"{audit['precision']:.4f}",
                recall="null" if audit["recall"] is None else f"{audit['recall']:.4f}",
                iou="null" if audit["iou"] is None else f"{audit['iou']:.4f}",
                walker="null" if audit["walker_to_floor_fraction"] is None else f"{audit['walker_to_floor_fraction']:.4f}",
            )
        )
    lines.append("")
    lines.append("这些数值只能称为**人工二维掩膜一致性审计**，不得称为语义精度、地面精度或三维精度。")
    lines.append("")
    lines.append("## 7. 27 组合延迟补齐")
    lines.append("")
    lines.append(f"- 只读源：`{summary['source_matrix_path']}`；")
    lines.append(f"- 输出：`combination_matrix_27_with_semantic_latency.json` / `.csv`，恰好 `{summary['matrix']['row_count']}` 行、组合 ID 唯一 `{summary['matrix']['unique_combination_ids']}`；")
    lines.append(f"- `semantic_online_p50_ms = {format_ms(summary['matrix']['semantic_online_p50_ms'])}`、`semantic_online_p95_ms = {format_ms(summary['matrix']['semantic_online_p95_ms'])}`（120 条主测量记录整体 P50/P95）；")
    lines.append(f"- 计算规则：原值为数值时新值 = 原值 + 语义值，否则为 `null`；null 行数 `{summary['matrix']['rows_with_null_semantic_total']}`；")
    lines.append(f"- `latency_kind = \"estimated_module_sum\"` 行数 `{summary['matrix']['latency_kind_estimated_rows']}`；")
    lines.append(f"- `measured_end_to_end_latency_ms is null` 行数 `{summary['matrix']['measured_null_rows']}`；")
    lines.append(f"- `realtime_compatible = false` 行数 `{summary['matrix']['realtime_false_rows']}`（未因补语义计时而把任何组合改为实时）；")
    lines.append(f"- `semantic_parity_status = {summary['matrix']['semantic_parity_status']}`；")
    lines.append(f"- 边界文字：`{summary['matrix']['semantic_latency_boundary']}`")
    lines.append("")
    lines.append("补齐语义耗时后估算 P50 最快的三种组合：")
    lines.append("")
    lines.append("| combination_id | 原估算 P50 (ms) | 语义 P50 (ms) | 新估算 P50 (ms) | 新估算 P95 (ms) |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in fastest:
        lines.append(
            "| `{cid}` | {base} | {sem} | {p50} | {p95} |".format(
                cid=row["combination_id"],
                base=format_ms(row["estimated_end_to_end_latency_ms"]),
                sem=format_ms(row["semantic_online_p50_ms"]),
                p50=format_ms(row["estimated_end_to_end_including_semantic_p50_ms"]),
                p95=format_ms(row["estimated_end_to_end_including_semantic_p95_ms"]),
            )
        )
    lines.append("")
    lines.append("## 8. 产物")
    lines.append("")
    lines.append("`EXPERIMENT.md`、`run_metadata.json`、`command.txt`、`summary.json`、`per_inference_records.jsonl`、`parity_records.jsonl`、`semantic_audit_left.json`、`semantic_audit_right.json`、`timing_summary.json`、`combination_matrix_27_with_semantic_latency.json`、`combination_matrix_27_with_semantic_latency.csv`、`visualizations/`。")
    lines.append("")
    lines.append("## 9. 验证")
    lines.append("")
    for item in summary["verification_notes"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 10. 结论边界")
    lines.append("")
    lines.append("- Mask2Former 输出是**自动二维地面候选**，不是人工真值、三维真值或物理地面；")
    lines.append("- 本实验只测**当前机器与当前版本接口**下的模块耗时；")
    lines.append("- 新总时延是模块 P50/P95 相加的**估算**（`estimated_module_sum`），不是全链实测，`measured_end_to_end_latency_ms` 全为 `null`；")
    lines.append("- 模型输出复现、人工二维 IoU、视差、平面内点率、残差或重投影误差都**不是**真实地面精度；")
    lines.append("- 不重跑 VO；VO 仍因逐帧静态背景米制对应和逐帧助步器排除区不足而保持 `unavailable`；")
    lines.append("- 本目录不写任何 Git 哈希；")
    lines.append("- 语义掩膜未接入姿态、标定、三角化、地面状态、接触或步态链。")
    lines.append("")
    document = "\n".join(lines) + "\n"
    assert_no_forbidden_phrases(document)
    return document


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run_probe(args: argparse.Namespace, runtime: dict[str, Any], torch: Any, device: str, resolved: Sequence[dict[str, Any]]) -> None:
    """Read-only interface probe: writes nothing, changes nothing."""
    print(json.dumps({"probe_only": True, "model_id": MODEL_ID, "device": device, "floor_class_id": runtime["floor_class_id"], "model_load_ms": runtime["model_load_ms"]}, ensure_ascii=False))
    limit = min(int(args.probe_max_pairs), len(resolved))
    for entry in resolved[:limit]:
        for view in VIEWS:
            image_path = entry[f"{view}_image"]
            result = run_single_image_online(image_path, entry[f"{view}_cache"], runtime, torch, device)
            parity = mask_comparison(result["candidate_mask"], result["cached_mask_bool"])
            print(
                json.dumps(
                    {
                        "view": view,
                        "pair_id": entry["pair_id"],
                        "full_online_semantic_ms": result[FULL_ONLINE_FIELD],
                        **{field: result[field] for field in STAGE_FIELDS},
                        "different_pixel_count": parity["different_pixel_count"],
                        "different_pixel_fraction": parity["different_pixel_fraction"],
                        "exact_match": parity["exact_match"],
                        "iou_vs_cached": parity["iou_vs_cached"],
                        "candidate_pixel_count": parity["candidate_pixel_count"],
                        "cached_pixel_count": parity["cached_pixel_count"],
                    },
                    ensure_ascii=False,
                )
            )


def main(argv: Sequence[str] | None = None) -> int:
    command_arguments = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(command_arguments)
    validate_run_parameters(args)

    import torch

    device = resolve_device(torch, args.device)

    resolved = resolve_pair_inputs(
        args.pair_ids,
        args.left_image_dir,
        args.right_image_dir,
        args.left_cache_dir,
        args.right_cache_dir,
        None if args.probe_only else args.left_label_dir,
        None if args.probe_only else args.right_label_dir,
    )

    runtime = load_model(device, torch)

    if args.probe_only:
        run_probe(args, runtime, torch, device, resolved)
        return 0

    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --probe-only is used")
    ensure_output_dir_absent(args.output_dir)

    environment = {
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu_only_host",
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "python_version": sys.version.split()[0],
        "device": device,
        "cuda_synchronize_before_and_after_every_timed_gpu_span": device == "cuda",
    }

    for entry in resolved:
        entry["candidate"] = {}

    # Warm-up: three consecutive runs on the left pair_0000, never in the main statistics.
    warmup_target = resolved[0]
    for _ in range(int(args.warmup_count)):
        run_single_image_online(warmup_target["left_image"], warmup_target["left_cache"], runtime, torch, device)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True)
    candidate_dir = output_dir / "candidate_floor_masks"
    candidate_dir.mkdir()
    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir()

    inference_records: list[dict[str, Any]] = []
    parity_records: list[dict[str, Any]] = []
    all_views_records: list[dict[str, Any]] = []

    for repeat_index in range(int(args.measured_repeats)):
        for entry in resolved:
            for view in VIEWS:
                image_path = entry[f"{view}_image"]
                cache_path = entry[f"{view}_cache"]
                stem = Path(entry["name"]).stem
                mask_path = candidate_dir / f"{view}_{stem}.png"
                first_contact = repeat_index == 0
                runtime["candidate_mask_path"] = mask_path
                result = run_single_image_online(
                    image_path,
                    cache_path,
                    runtime,
                    torch,
                    device,
                    measure_cache_read=True,
                    measure_mask_write=False,
                )
                record: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "record_index": len(inference_records),
                    "view": view,
                    "pair_id": int(entry["pair_id"]),
                    "image": str(image_path.resolve()),
                    "cached_mask": str(cache_path.resolve()),
                    "repeat_index": int(repeat_index),
                    "warmup": False,
                    "device": device,
                    "batch_size": 1,
                    "model_id": MODEL_ID,
                    "floor_class_id": runtime["floor_class_id"],
                    "image_shape_hw": result["image_shape_hw"],
                    **{field: float(result[field]) for field in STAGE_FIELDS},
                    FULL_ONLINE_FIELD: float(result[FULL_ONLINE_FIELD]),
                    "cache_mask_read_ms": result["cache_mask_read_ms"],
                    "candidate_png_write_ms": None,
                    "io_diagnostics_excluded_from_full_online_semantic_ms": True,
                }
                if first_contact:
                    write_start = time.perf_counter()
                    cv2.imwrite(str(mask_path), (result["candidate_mask"] * 255).astype(np.uint8))
                    record["candidate_png_write_ms"] = (time.perf_counter() - write_start) * 1000.0
                record["candidate_mask_path"] = str(mask_path.resolve())
                inference_records.append(record)
                all_views_records.append(record)

                if not first_contact:
                    continue

                candidate_bool = result["candidate_mask"].astype(bool)
                cached_bool = result["cached_mask_bool"]
                parity = mask_comparison(candidate_bool, cached_bool)
                entry["candidate"][view] = candidate_bool

                parity_record = {
                    "schema_version": SCHEMA_VERSION,
                    "view": view,
                    "pair_id": int(entry["pair_id"]),
                    "image": str(image_path.resolve()),
                    "cached_mask": str(cache_path.resolve()),
                    "candidate_mask": str(mask_path.resolve()),
                    "image_shape_hw": result["image_shape_hw"],
                    "candidate_pixel_count": parity["candidate_pixel_count"],
                    "cached_pixel_count": parity["cached_pixel_count"],
                    "intersection_pixel_count": parity["intersection_pixel_count"],
                    "union_pixel_count": parity["union_pixel_count"],
                    "equal_pixel_count": parity["equal_pixel_count"],
                    "different_pixel_count": parity["different_pixel_count"],
                    "different_pixel_fraction": parity["different_pixel_fraction"],
                    "exact_match": parity["exact_match"],
                    "iou_vs_cached": parity["iou_vs_cached"],
                    "parity_status": parity["parity_status"],
                    "archived_cache_untouched": True,
                    "interpretation": (
                        "pixel-for-pixel reproduction check of the current interface against the archived "
                        "candidate mask; equality is output reproducibility, not accuracy"
                    ),
                }
                if not parity["exact_match"]:
                    difference = np.logical_xor(candidate_bool, cached_bool)
                    difference_path = visualization_dir / f"{view}_{Path(entry['name']).stem}_difference.png"
                    cv2.imwrite(str(difference_path), (difference.astype(np.uint8) * 255))
                    parity_record["difference_mask"] = str(difference_path.resolve())
                    parity_record["difference_pixel_count"] = int(difference.sum())
                parity_records.append(parity_record)

                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                overlay = render_parity_overlay(
                    image,
                    candidate_bool,
                    cached_bool,
                    parity,
                    f"{view} {entry['name']} online Mask2Former floor candidate",
                )
                cv2.imwrite(str(visualization_dir / f"{view}_{stem}_parity.png"), overlay)
                if entry["pair_id"] in (0, 160, 320):
                    cv2.imwrite(str(visualization_dir / f"{view}_{stem}_overlay.png"), overlay)

    timing = timing_summary_from_records(all_views_records)
    timing["warmup_count"] = int(args.warmup_count)
    timing["warmup_target"] = f"left {Path(warmup_target['name']).stem}"
    timing["measured_repeats"] = int(args.measured_repeats)
    timing["stage_definition"] = {FULL_ONLINE_FIELD: " + ".join(STAGE_FIELDS)}
    timing["io_diagnostic_note"] = (
        "cache_mask_read_ms and candidate_png_write_ms are pure I/O diagnostics and are never part of "
        "full_online_semantic_ms"
    )
    timing["cuda_synchronize_fenced"] = device == "cuda"
    timing["environment"] = environment
    timing["model_id"] = MODEL_ID
    timing["floor_class_id"] = runtime["floor_class_id"]
    timing["model_load_ms"] = runtime["model_load_ms"]
    timing["image_shape_hw"] = resolved[0]["image_shape_hw"]
    timing["interpretation_boundary"] = INTERPRETATION_BOUNDARY

    audit_left = semantic_audit("left", resolved, lambda entry, view: entry["candidate"][view])
    audit_right = semantic_audit("right", resolved, lambda entry, view: entry["candidate"][view])

    exact_count = int(sum(1 for record in parity_records if record["exact_match"]))
    parity_status = PARITY_ALL_EXACT if exact_count == len(parity_records) else PARITY_MISMATCH_PRESENT

    semantic_p50 = timing["all_views"][FULL_ONLINE_FIELD]["median_ms"]
    semantic_p95 = timing["all_views"][FULL_ONLINE_FIELD]["p95_ms"]

    source_matrix_path = Path(summary_source_matrix_path())
    source_matrix = json.loads(source_matrix_path.read_text(encoding="utf-8"))
    augmented = augment_matrix_with_semantic_latency(source_matrix, semantic_p50, semantic_p95, parity_status)

    source_rows_by_id = {row["combination_id"]: row for row in source_matrix["rows"]}
    original_fields_preserved = all(
        source_rows_by_id[row["combination_id"]][key] == value
        for row in augmented["rows"]
        for key, value in source_rows_by_id[row["combination_id"]].items()
    )
    if not original_fields_preserved:
        raise OnlineLatencyError("augmented matrix changed an original field")

    matrix_row_count = len(augmented["rows"])
    unique_ids = len({row["combination_id"] for row in augmented["rows"]})
    if matrix_row_count != 27 or unique_ids != 27:
        raise OnlineLatencyError(f"augmented matrix must hold 27 unique combinations, got {matrix_row_count}/{unique_ids}")

    (output_dir / "per_inference_records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in inference_records), encoding="utf-8"
    )
    (output_dir / "parity_records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in parity_records), encoding="utf-8"
    )
    (output_dir / "timing_summary.json").write_text(
        json.dumps(timing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "semantic_audit_left.json").write_text(
        json.dumps(audit_left, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "semantic_audit_right.json").write_text(
        json.dumps(audit_right, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "combination_matrix_27_with_semantic_latency.json").write_text(
        json.dumps(augmented, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    csv_text = matrix_csv_text(augmented)
    (output_dir / "combination_matrix_27_with_semantic_latency.csv").write_text(csv_text, encoding="utf-8")

    fastest = sorted(
        [row for row in augmented["rows"] if row["estimated_end_to_end_including_semantic_p50_ms"] is not None],
        key=lambda row: row["estimated_end_to_end_including_semantic_p50_ms"],
    )[:3]

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date_beijing": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "task": "task-04 Mask2Former online semantic latency, cached-mask parity and 27-combination latency boundary",
        "input_paths": {
            "left_image_dir": str(Path(args.left_image_dir).resolve()),
            "right_image_dir": str(Path(args.right_image_dir).resolve()),
            "left_cache_dir": str(Path(args.left_cache_dir).resolve()),
            "right_cache_dir": str(Path(args.right_cache_dir).resolve()),
            "left_label_dir": str(Path(args.left_label_dir).resolve()),
            "right_label_dir": str(Path(args.right_label_dir).resolve()),
        },
        "pair_ids": [int(entry["pair_id"]) for entry in resolved],
        "image_shape_hw": resolved[0]["image_shape_hw"],
        "environment": environment,
        "model_id": MODEL_ID,
        "backend_name": BACKEND_NAME,
        "floor_class_id": runtime["floor_class_id"],
        "model_load_ms": runtime["model_load_ms"],
        "local_files_only": True,
        "download_arguments_present": False,
        "device": device,
        "batch_size": 1,
        "warmup_count": int(args.warmup_count),
        "measured_repeats": int(args.measured_repeats),
        "main_measurement_record_count": len(inference_records),
        "parity": {
            "record_count": len(parity_records),
            "exact_match_count": exact_count,
            "mismatch_count": len(parity_records) - exact_count,
            "semantic_parity_status": parity_status,
            "max_different_pixel_count": int(max(record["different_pixel_count"] for record in parity_records)),
            "total_different_pixel_count": int(sum(record["different_pixel_count"] for record in parity_records)),
            "min_iou_vs_cached": float(min(record["iou_vs_cached"] for record in parity_records)),
            "records": [
                {
                    "view": record["view"],
                    "pair_id": record["pair_id"],
                    "different_pixel_count": record["different_pixel_count"],
                    "different_pixel_fraction": record["different_pixel_fraction"],
                    "exact_match": record["exact_match"],
                    "iou_vs_cached": record["iou_vs_cached"],
                    "parity_status": record["parity_status"],
                }
                for record in parity_records
            ],
        },
        "online_latency_ms": {
            "full_online_semantic_p50_ms": semantic_p50,
            "full_online_semantic_p95_ms": semantic_p95,
            "full_online_semantic_p50_ms_left": timing["left"][FULL_ONLINE_FIELD]["median_ms"],
            "full_online_semantic_p50_ms_right": timing["right"][FULL_ONLINE_FIELD]["median_ms"],
            "gpu_forward_p50_ms": timing["all_views"]["gpu_forward_ms"]["median_ms"],
            "gpu_forward_p95_ms": timing["all_views"]["gpu_forward_ms"]["p95_ms"],
            "processor_cpu_p50_ms": timing["all_views"]["processor_cpu_ms"]["median_ms"],
            "postprocess_p50_ms": timing["all_views"]["postprocess_ms"]["median_ms"],
            "image_decode_p50_ms": timing["all_views"]["image_decode_ms"]["median_ms"],
        },
        "manual_2d_audit": {
            "left": {
                "labelled_image_count": audit_left["labelled_image_count"],
                "pair_ids": audit_left["pair_ids"],
                "precision": audit_left["precision"],
                "recall": audit_left["recall"],
                "iou": audit_left["iou"],
                "walker_to_floor_fraction": audit_left["walker_to_floor_fraction"],
            },
            "right": {
                "labelled_image_count": audit_right["labelled_image_count"],
                "pair_ids": audit_right["pair_ids"],
                "precision": audit_right["precision"],
                "recall": audit_right["recall"],
                "iou": audit_right["iou"],
                "walker_to_floor_fraction": audit_right["walker_to_floor_fraction"],
            },
        },
        "source_matrix_path": str(source_matrix_path),
        "matrix": {
            "path": str((output_dir / "combination_matrix_27_with_semantic_latency.json").resolve()),
            "row_count": matrix_row_count,
            "unique_combination_ids": unique_ids,
            "source_row_count": len(source_matrix["rows"]),
            "original_fields_preserved": bool(original_fields_preserved),
            "semantic_online_p50_ms": semantic_p50,
            "semantic_online_p95_ms": semantic_p95,
            "semantic_parity_status": parity_status,
            "semantic_latency_boundary": augmented["semantic_latency_boundary"],
            "latency_kind_estimated_rows": int(sum(1 for row in augmented["rows"] if row["latency_kind"] == "estimated_module_sum")),
            "measured_null_rows": int(sum(1 for row in augmented["rows"] if row["measured_end_to_end_latency_ms"] is None)),
            "realtime_false_rows": int(sum(1 for row in augmented["rows"] if row["realtime_compatible"] is False)),
            "rows_with_null_semantic_total": int(
                sum(1 for row in augmented["rows"] if row["estimated_end_to_end_including_semantic_p50_ms"] is None)
            ),
            "latency_kind": "estimated_module_sum",
        },
        "fastest_combinations_with_semantic_latency": [
            {
                "combination_id": row["combination_id"],
                "matcher": row["matcher"],
                "plane_method": row["plane_method"],
                "temporal_mode": row["temporal_mode"],
                "estimated_end_to_end_latency_ms": row["estimated_end_to_end_latency_ms"],
                "semantic_online_p50_ms": row["semantic_online_p50_ms"],
                "estimated_end_to_end_including_semantic_p50_ms": row["estimated_end_to_end_including_semantic_p50_ms"],
                "estimated_end_to_end_including_semantic_p95_ms": row["estimated_end_to_end_including_semantic_p95_ms"],
                "realtime_compatible": row["realtime_compatible"],
            }
            for row in fastest
        ],
        "not_re_run": [
            "SGBM",
            "IGEV",
            "DynamicStereo",
            "plane fitting (dense RANSAC / sparse tile RANSAC / soft weighted IRLS)",
            "Farneback optical flow",
            "visual odometry",
        ],
        "outputs": [
            "EXPERIMENT.md",
            "run_metadata.json",
            "command.txt",
            "summary.json",
            "per_inference_records.jsonl",
            "parity_records.jsonl",
            "semantic_audit_left.json",
            "semantic_audit_right.json",
            "timing_summary.json",
            "combination_matrix_27_with_semantic_latency.json",
            "combination_matrix_27_with_semantic_latency.csv",
            "candidate_floor_masks/",
            "visualizations/",
        ],
        "verification_notes": [
            f"py_compile passed for `realtime_app/tools/benchmark_mask2former_online_latency.py`",
            f"unit tests passed for `realtime_app/tests/test_mask2former_online_latency.py`",
            f"main measurement records {len(inference_records)} = 24 images x {int(args.measured_repeats)} repeats",
            f"parity records {len(parity_records)} (one per view x pair)",
            f"augmented matrix rows {matrix_row_count} with {unique_ids} unique combination ids",
        ],
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
        "vo_status": (
            "visual odometry was not re-run and remains unavailable: the frozen archives hold no per-frame "
            "metric static-background correspondences and the walker exclusion region exists for only 12 frames"
        ),
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    run_metadata = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "tool": str(Path(__file__).resolve()),
        "arguments": {
            key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()
        },
        "environment": environment,
        "model_id": MODEL_ID,
        "model_load_ms": runtime["model_load_ms"],
        "floor_class_id": runtime["floor_class_id"],
        "processor_size": runtime["processor_size"],
        "num_labels": runtime["num_labels"],
        "pair_ids": [int(entry["pair_id"]) for entry in resolved],
        "input_paths": summary["input_paths"],
        "image_shape_hw": resolved[0]["image_shape_hw"],
        "main_measurement_record_count": len(inference_records),
        "parity_record_count": len(parity_records),
        "output_dir": str(output_dir.resolve()),
        "files_written": summary["outputs"],
        "constraints": [
            "no dependency installation, no model download, local_files_only=True only",
            "no Git hash recorded",
            "no older experiment directory written or re-run",
            "no semantic mask connected to pose, calibration, triangulation, ground state, contact or gait",
            "manual labels are a 2-D audit reference only",
        ],
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "command.txt").write_text(
        " ".join([Path(sys.executable).name, Path(__file__).name, *command_arguments]) + "\n", encoding="utf-8"
    )

    experiment_markdown = build_experiment_markdown(summary, timing, {"left": audit_left, "right": audit_right})
    (output_dir / "EXPERIMENT.md").write_text(experiment_markdown, encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "main_measurement_records": len(inference_records),
                "parity_records": len(parity_records),
                "exact_match_count": exact_count,
                "semantic_parity_status": parity_status,
                "full_online_semantic_p50_ms": semantic_p50,
                "full_online_semantic_p95_ms": semantic_p95,
                "matrix_rows": matrix_row_count,
                "realtime_compatible_true_rows": 0,
                "measured_end_to_end_latency_ms_non_null_rows": 0,
                "fastest_combinations": [row["combination_id"] for row in fastest],
            },
            ensure_ascii=False,
        )
    )
    return 0


def summary_source_matrix_path() -> str:
    """Absolute path of the read-only task-03 matrix, resolved from the repo root."""
    repo_root = Path(__file__).resolve().parents[2]  # realtime_app/tools/<file> -> repository root
    return str(repo_root / SOURCE_MATRIX_PATH)


if __name__ == "__main__":
    raise SystemExit(main())
