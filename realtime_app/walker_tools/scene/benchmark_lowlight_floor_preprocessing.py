#!/usr/bin/env python3
"""Controlled comparison of fixed classical low-light preprocessing for the floor candidate.

Question
--------
Does a *fixed* classical low-light front-end change the existing Mask2Former-Swin-S
``floor`` candidate on the existing manual holdout labels, and what happens to the
candidate geometry when that mask is fed into the unchanged strict stereo chain?

Absolute boundaries enforced by this tool
-----------------------------------------
* Every stereo match, disparity field, mask-directed rectification, depth limit,
  triangulation and RANSAC parameter is computed from the **raw upright** left and
  right images.  No enhanced image and no enhanced pixel ever enters geometry.
* An enhanced image is only ever the semantic network's input.
* A mask produced from an enhanced image is written in exactly the original
  upright pixel grid.  ``load_binary_mask_exact`` refuses to resample a mask, and
  the tool asserts the mask shape equals the original upright shape before any
  geometry runs.
* PMPose, person association, the existing strict triangulation, the calibration,
  the frozen disparity parameters, the corrected reverse-disparity direction and
  every threshold are read-only inputs.  No command line switch can change them.
* Masks carry ``automatic_candidate`` status and ``provided_unvalidated`` identity
  evidence, so the strict chain can only ever return ``unavailable`` and the plane
  is reported as ``unaccepted_candidate_plane``.  Any other state aborts the run.
* Exactly four comparison groups.  No fifth group, no per-label parameter search,
  no training, no fine-tuning and no model download.

Reported numbers are internal candidate evidence on a small fixed frame set.  They
are not ground truth and not a physical accuracy measurement.
"""

from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
from walker_tools._compat import LEGACY_TOOLS as _tool_legacy_tools
_tool_prepare_imports()

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


TOOLS_ROOT = _tool_legacy_tools
PROJECT_ROOT = TOOLS_ROOT.parents[1]
REALTIME_ROOT = PROJECT_ROOT / "realtime_app"

from pose_app.calibration import StereoCalibration  # noqa: E402
from pose_app.lowlight_floor_preprocessing import (  # noqa: E402
    ALL_PAIR_IDS,
    CHAIN_MAXIMUM_DEPTH_MM,
    CHAIN_MAXIMUM_PHOTOMETRIC_DIFFERENCE,
    CHAIN_MINIMUM_DEPTH_MM,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    CONSERVATIVE_RAW_MARGIN,
    DEV_PAIR_IDS,
    EXPLANATION_TEXTS,
    GAMMA_0P6_THEN_CLAHE_LAB,
    GAMMA_EXPONENT,
    HOLDOUT_PAIR_IDS,
    MARGIN_COMPARISON_EPSILON,
    MASK_IDENTITY_MANUALLY_AUDITED,
    MASK_IDENTITY_PROVIDED_UNVALIDATED,
    MASK_STATUS_AUTOMATIC_CANDIDATE,
    METHODS,
    RAW,
    SAFE_PLANE_FIELD,
    apply_preprocessing,
    assert_explanation_text,
    assert_transform_contract,
    audit_payload_texts,
    gray_region_statistics,
    load_binary_mask_exact,
    method_parameters,
    pair_name,
    split_of,
)
from pose_app.scene_geometry_variants import (  # noqa: E402
    angular_error_degrees,
    estimate_region_consensus,
)
from walker_tools.scene.audit_manual_floor_labels import binary_metrics, load_labelme, rasterize_labelme  # noqa: E402
from walker_tools.scene.benchmark_floor_semantic_backends import render_overlay  # noqa: E402
from walker_tools.scene.benchmark_ground_walker_reconstruction import REGION_MAXIMUM_ANGLE_DEG, REGION_MINIMUM_POINTS
import walker_tools.scene.diagnose_floor_mask_stereo_correspondence as diagnostic  # noqa: E402
import walker_tools.scene.observe_local_ground_semantic_stereo as strict  # noqa: E402
import walker_tools.scene.observe_local_ground_semantic_stereo_lr_direction_control as corrected  # noqa: E402


SCHEMA_VERSION = "lowlight_floor_preprocessing_benchmark_v1"
INFERENCE_BACKEND = "mask2former_swin_small"
INFERENCE_MODEL_ID = "facebook/mask2former-swin-small-ade-semantic"
VIEWS = ("left", "right")

# The existing Mask2Former interface writes a fixed
# ``<output>/<backend>/floor_masks/pair_XXXX.png`` tree with names this tool
# cannot shorten, so its scratch directory needs a short name.  The mapping is
# recorded in ``run_metadata.json`` and in EXPERIMENT.md.
ARM_CODES = {
    ("raw", "left"): "raw_l",
    ("raw", "right"): "raw_r",
    ("gamma_0p6", "left"): "gamma_l",
    ("gamma_0p6", "right"): "gamma_r",
    ("clahe_lab", "left"): "clahe_l",
    ("clahe_lab", "right"): "clahe_r",
    ("gamma_0p6_then_clahe_lab", "left"): "combo_l",
    ("gamma_0p6_then_clahe_lab", "right"): "combo_r",
}

# Windows resolves a relative or short path against the process working
# directory before the MAX_PATH check, so a deep experiment directory can fail
# with ERROR_FILENAME_EXCED_RANGE mid-run.  Every planned path is checked before
# anything is created.
MAXIMUM_SAFE_PATH_LENGTH = 235

FROZEN_PARAMETER_KEYS = (
    "runtime_width",
    "runtime_height",
    "virtual_focal_px",
    "num_disparities",
    "lr_consistency_px",
    "ransac_distance_mm",
    "ransac_iterations",
    "minimum_candidates",
    "minimum_inliers",
    "minimum_inlier_fraction",
    "minimum_coverage_fraction",
    "maximum_median_residual_mm",
    "maximum_median_reprojection_px",
    "frame_step",
    "max_frames",
)

INTERPRETATION_BOUNDARY = (
    "Internal image-space candidate evidence and internal strict-geometry gate results on a fixed small frame set. "
    "These records are not ground truth. Brightness, mask area, candidate points, inlier fraction and plane "
    "residual are diagnostics. They are not correctness evidence and they are not a physical accuracy "
    "measurement. The manual floor_eligible masks are the two-dimensional condition of this comparison, not a "
    "physical ground reference. No accepted ground state, no foot height, no contact judgement, no support "
    "judgement and no gait measurement is produced."
)

FORBIDDEN_ITEMS = (
    "no new data capture",
    "no model training or fine-tuning",
    "no model download or replacement",
    "no DA3, GroundNet, Retinex, Zero-DCE, LLIE or any other extra model",
    "no PMPose rerun and no person-association change",
    "no calibration, triangulation, frozen disparity parameter or threshold change",
    "no reversal of the corrected reverse-disparity search direction",
    "no enhanced image in any geometric computation",
    "no mask resampled from a different pixel grid",
    "no fifth preprocessing group and no per-label parameter search",
    "no automatic mask promoted to manually_audited or to an accepted ground state",
    "no overwrite of an existing experiment directory",
    "no git commit and no hash written into any record",
)


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


# ---------------------------------------------------------------------------
# Command line and frozen inputs
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--left-dir", type=Path, required=True, help="upright left_ccw90 pair_*.png frames")
    parser.add_argument("--right-dir", type=Path, required=True, help="upright right_cw90 pair_*.png frames")
    parser.add_argument("--left-label-dir", type=Path, required=True, help="left LabelMe json directory")
    parser.add_argument("--right-label-dir", type=Path, required=True, help="right LabelMe json directory")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument(
        "--frozen-baseline-dir", type=Path, required=True,
        help="frozen stereo parameter source holding run_metadata.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--inference-script", type=Path, default=TOOLS_ROOT / "benchmark_floor_semantic_backends.py",
        help="existing Mask2Former interface, invoked as a separate process so torch never enters this process",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--pair-ids", nargs="*", type=int, default=None,
        help="development only: restrict to a subset for a smoke test; the official run uses the fixed 12 frames",
    )
    parser.add_argument("--no-visualization", action="store_true")
    return parser.parse_args(argv)


def read_frozen_parameters(frozen_baseline_dir: Path) -> dict[str, Any]:
    metadata_path = frozen_baseline_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"frozen baseline metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    parameters = metadata.get("parameters")
    if not isinstance(parameters, Mapping):
        raise RuntimeError(f"{metadata_path}: parameters block missing")
    missing = [key for key in FROZEN_PARAMETER_KEYS if key not in parameters]
    if missing:
        raise RuntimeError(f"{metadata_path}: frozen parameters missing keys {missing}")
    return {key: parameters[key] for key in FROZEN_PARAMETER_KEYS}


def frozen_frame_names(left_dir: Path, right_dir: Path, parameters: Mapping[str, Any]) -> list[str]:
    pairs = strict.read_pairs(
        left_dir, right_dir, int(parameters["frame_step"]), int(parameters["max_frames"])
    )
    return [Path(name).stem for name, _, _ in pairs]


def selected_pairs(left_dir: Path, right_dir: Path, pair_ids: Sequence[int]) -> list[tuple[int, Path, Path]]:
    selected: list[tuple[int, Path, Path]] = []
    for pair_id in pair_ids:
        name = pair_name(pair_id)
        left, right = left_dir / name, right_dir / name
        if not left.is_file() or not right.is_file():
            raise FileNotFoundError(f"fixed frame {name} is missing from {left_dir} or {right_dir}")
        selected.append((int(pair_id), left, right))
    return selected


def arm_code(method: str, view: str) -> str:
    try:
        return ARM_CODES[(method, view)]
    except KeyError as error:  # pragma: no cover - guarded by METHODS and VIEWS
        raise ValueError(f"no arm code for {method!r}/{view!r}") from error


def preflight_path_lengths(output_dir: Path) -> dict[str, Any]:
    """Refuse to start when a planned output path would exceed the safe length."""
    base = len(str(Path(output_dir).resolve()))
    probes = [
        (f"enhanced_images/{method}/{view}/pair_0440.png",
         base + 1 + len(f"enhanced_images/{method}/{view}/pair_0440.png"))
        for method in METHODS for view in VIEWS
    ]
    probes += [
        (f"floor_masks/{method}/{view}/pair_0440.png",
         base + 1 + len(f"floor_masks/{method}/{view}/pair_0440.png"))
        for method in METHODS for view in VIEWS
    ]
    probes += [
        (f"overlays/{method}/{view}/pair_0440.png",
         base + 1 + len(f"overlays/{method}/{view}/pair_0440.png"))
        for method in METHODS for view in VIEWS
    ]
    probes += [
        (f"geometry_visualizations/{method}/p0440.png",
         base + 1 + len(f"geometry_visualizations/{method}/p0440.png"))
        for method in METHODS
    ]
    probes.append(("geometry_visualizations/manual_reference/p0440.png",
                   base + 1 + len("geometry_visualizations/manual_reference/p0440.png")))
    probes += [
        (f"inf/{arm_code(method, view)}/{INFERENCE_BACKEND}/pair_0440_floor_candidate.png",
         base + 1 + len(f"inf/{arm_code(method, view)}/{INFERENCE_BACKEND}/pair_0440_floor_candidate.png"))
        for method in METHODS for view in VIEWS
    ]
    probes += [
        (f"representative_visualizations/g2_improve_p0440_{arm_code(method, view)}.png",
         base + 1 + len(f"representative_visualizations/g2_improve_p0440_{arm_code(method, view)}.png"))
        for method in METHODS for view in VIEWS
    ]
    worst = max(probes, key=lambda item: item[1])
    if worst[1] > MAXIMUM_SAFE_PATH_LENGTH:
        raise RuntimeError(
            f"refusing to run: the planned path {worst[0]!r} would be {worst[1]} characters, above the safe "
            f"limit of {MAXIMUM_SAFE_PATH_LENGTH}; choose a shorter output directory"
        )
    return {
        "output_base_length": base,
        "worst_planned_path": worst[0],
        "worst_planned_length": worst[1],
        "maximum_safe_path_length": MAXIMUM_SAFE_PATH_LENGTH,
        "checked_path_count": len(probes),
    }


def mask_source(method: str) -> dict[str, Any]:
    return {
        "backend": INFERENCE_BACKEND,
        "model_id": INFERENCE_MODEL_ID,
        "model_role": "existing frozen floor-candidate network, reused unchanged",
        "preprocessing_group": method,
        "preprocessing_parameters": method_parameters(method),
        "mask_status": MASK_STATUS_AUTOMATIC_CANDIDATE,
        "semantic_identity_evidence": MASK_IDENTITY_PROVIDED_UNVALIDATED,
        "mask_pixel_grid": "exactly the original upright image grid, shape asserted before any geometry runs",
        "training_or_fine_tuning": "none",
    }


# ---------------------------------------------------------------------------
# Manual labels
# ---------------------------------------------------------------------------


def load_manual_view(label_dir: Path, pair_id: int, image_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    label_path = label_dir / f"pair_{pair_id:04d}.json"
    if not label_path.is_file():
        raise FileNotFoundError(f"manual label is missing: {label_path}")
    masks = rasterize_labelme(load_labelme(label_path))
    if masks["floor_eligible"].shape != tuple(image_shape):
        raise ValueError(
            f"{label_path}: label shape {masks['floor_eligible'].shape} differs from the upright image shape "
            f"{tuple(image_shape)}"
        )
    return masks


# ---------------------------------------------------------------------------
# Two-dimensional semantic evaluation
# ---------------------------------------------------------------------------


def evaluate_semantic_candidate(
    masks: Mapping[str, np.ndarray], candidate: np.ndarray | None
) -> dict[str, Any]:
    """Evaluate one candidate mask against one view's exclusive manual labels."""
    valid = masks["valid_evaluation"]
    reference = masks["floor_eligible"]
    record: dict[str, Any] = {
        "status": "unavailable",
        "reasons": [],
        "metrics": None,
        "person_labelled_pixels_in_valid_area": int(np.logical_and(masks["person"], valid).sum()),
        "walker_labelled_pixels_in_valid_area": int(np.logical_and(masks["walker"], valid).sum()),
        "static_other_labelled_pixels_in_valid_area": int(np.logical_and(masks["static_other"], valid).sum()),
        "person_pixels_predicted_floor": None,
        "person_to_floor_fraction": None,
        "walker_pixels_predicted_floor": None,
        "walker_to_floor_fraction": None,
        "static_other_pixels_predicted_floor": None,
        "static_other_to_floor_fraction": None,
    }
    if candidate is None:
        record["reasons"] = ["no_semantic_prediction"]
        return record
    candidate = np.asarray(candidate, dtype=bool)
    if candidate.shape != reference.shape:
        raise ValueError(
            f"candidate mask shape {candidate.shape} differs from the manual label shape {reference.shape}; "
            "refusing to evaluate across different pixel grids"
        )
    if not np.any(np.logical_and(reference, valid)):
        record["reasons"] = ["no_reference_floor_pixels_in_valid_area"]
        return record
    record["status"] = "candidate"
    record["metrics"] = binary_metrics(reference, valid, candidate)
    if record["metrics"]["candidate_floor_px_in_valid_area"] == 0:
        record["reasons"] = ["empty_candidate_mask"]
    for label in ("person", "walker", "static_other"):
        region = np.logical_and(masks[label], valid)
        count = int(region.sum())
        predicted = int(np.logical_and(region, candidate).sum())
        record[f"{label}_labelled_pixels_in_valid_area"] = count
        record[f"{label}_pixels_predicted_floor"] = predicted
        record[f"{label}_to_floor_fraction"] = None if count == 0 else float(predicted / count)
    return record


def summarize_values(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0, "mean": None, "median": None, "std": None, "variance": None,
            "min": None, "max": None, "values": [],
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "variance": float(np.var(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "values": [float(value) for value in array],
    }


def aggregate_view(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_frame: list[dict[str, Any]] = []
    for row in records:
        evaluation = row["evaluation"]
        metrics = evaluation["metrics"]
        per_frame.append({
            "pair_id": int(row["pair_id"]),
            "status": evaluation["status"],
            "reasons": list(evaluation["reasons"]),
            "precision": None if metrics is None else float(metrics["precision"]),
            "recall": None if metrics is None else float(metrics["recall"]),
            "iou": None if metrics is None else float(metrics["iou"]),
            "floor_candidate_px": None if metrics is None else int(metrics["candidate_floor_px_in_valid_area"]),
        })
    evaluated = [row for row in per_frame if row["iou"] is not None]
    reason_counts: dict[str, int] = {}
    for row in per_frame:
        for reason in row["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "frame_count": len(per_frame),
        "evaluated_frame_count": len(evaluated),
        "unavailable_frame_count": len(per_frame) - len(evaluated),
        "macro_precision": float(np.mean([row["precision"] for row in evaluated])) if evaluated else None,
        "macro_recall": float(np.mean([row["recall"] for row in evaluated])) if evaluated else None,
        "macro_iou": float(np.mean([row["iou"] for row in evaluated])) if evaluated else None,
        "unavailable_reason_counts": reason_counts,
        "per_frame": per_frame,
    }


def _labelled_class_summary(records: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    frames_with_label = [
        row for row in records if int(row["evaluation"][f"{label}_labelled_pixels_in_valid_area"]) > 0
    ]
    pooled_pixels = sum(int(row["evaluation"][f"{label}_labelled_pixels_in_valid_area"]) for row in frames_with_label)
    pooled_predicted = sum(int(row["evaluation"][f"{label}_pixels_predicted_floor"]) for row in frames_with_label)
    return {
        "labelled_view_frame_count": len(frames_with_label),
        "labelled_pixels_total": pooled_pixels,
        "pixels_predicted_floor_total": pooled_predicted,
        "pooled_to_floor_fraction": None if pooled_pixels == 0 else float(pooled_predicted / pooled_pixels),
        "per_frame": [
            {
                "pair_id": int(row["pair_id"]),
                "view": row["view"],
                "labelled_pixels": int(row["evaluation"][f"{label}_labelled_pixels_in_valid_area"]),
                "pixels_predicted_floor": int(row["evaluation"][f"{label}_pixels_predicted_floor"]),
                "to_floor_fraction": row["evaluation"][f"{label}_to_floor_fraction"],
            }
            for row in frames_with_label
        ],
        "population_metric": False,
        "note": (
            f"{label} polygons only exist on {len(frames_with_label)} of the evaluated view-frames, so this is a "
            "per-sample diagnostic with an explicit sample count and never a population indicator"
        ),
    }


def brightness_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    floor_median = [row["brightness"]["floor_region"]["median_gray_0_255"] for row in records]
    non_floor_median = [row["brightness"]["non_floor_region"]["median_gray_0_255"] for row in records]
    delta = [row["brightness"]["floor_region_median_gray_delta_vs_raw"] for row in records]
    return {
        "floor_region_median_gray": summarize_values(floor_median),
        "non_floor_region_median_gray": summarize_values(non_floor_median),
        "floor_region_median_gray_delta_vs_raw": summarize_values(delta),
        "role": (
            "explanatory diagnostic only. Brightness is not correctness evidence and a brighter region is not a "
            "better floor candidate"
        ),
    }


def aggregate_method(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    views: dict[str, Any] = {
        view: aggregate_view([row for row in records if row["view"] == view]) for view in VIEWS
    }
    pair_values = [
        [value for value in (views[view]["macro_precision"], views[view]["macro_recall"], views[view]["macro_iou"])
         if value is not None]
        for view in VIEWS
    ]
    precision_pair = [views[view]["macro_precision"] for view in VIEWS if views[view]["macro_precision"] is not None]
    recall_pair = [views[view]["macro_recall"] for view in VIEWS if views[view]["macro_recall"] is not None]
    iou_pair = [views[view]["macro_iou"] for view in VIEWS if views[view]["macro_iou"] is not None]
    all_iou = [row["iou"] for view in VIEWS for row in views[view]["per_frame"] if row["iou"] is not None]
    unavailable: dict[str, int] = {}
    for view in VIEWS:
        for reason, count in views[view]["unavailable_reason_counts"].items():
            unavailable[reason] = unavailable.get(reason, 0) + count
    return {
        "views": views,
        "left_right_macro": {
            "precision": None if not precision_pair else float(np.mean(precision_pair)),
            "recall": None if not recall_pair else float(np.mean(recall_pair)),
            "iou": None if not iou_pair else float(np.mean(iou_pair)),
            "views_contributing": len(iou_pair),
            "view_macro_triples": pair_values,
        },
        "iou_distribution": summarize_values(all_iou),
        "unavailable_reason_counts": unavailable,
        "person_to_floor": _labelled_class_summary(records, "person"),
        "walker_to_floor": _labelled_class_summary(records, "walker"),
        "static_other_to_floor": _labelled_class_summary(records, "static_other"),
        "brightness_diagnostics": brightness_summary(records),
    }


def build_semantic_split_payload(
    *, split_name: str, pair_ids: Sequence[int], records: Sequence[Mapping[str, Any]],
    raw_records: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    raw_macro = aggregate_method([row for row in records if row["method"] == RAW])["left_right_macro"]["iou"]
    methods: dict[str, Any] = {}
    for method in METHODS:
        entry = aggregate_method([row for row in records if row["method"] == method])
        per_view_frame: list[dict[str, Any]] = []
        for view in VIEWS:
            for frame in entry["views"][view]["per_frame"]:
                reference = raw_records.get((view, frame["pair_id"]))
                raw_iou = None
                if reference is not None:
                    raw_metrics = reference["evaluation"]["metrics"]
                    raw_iou = None if raw_metrics is None else float(raw_metrics["iou"])
                difference = None
                if frame["iou"] is not None and raw_iou is not None:
                    difference = float(frame["iou"] - raw_iou)
                per_view_frame.append({
                    "pair_id": frame["pair_id"],
                    "view": view,
                    "iou": frame["iou"],
                    "raw_iou": raw_iou,
                    "iou_difference_vs_raw": difference,
                })
        differences = [row["iou_difference_vs_raw"] for row in per_view_frame
                       if row["iou_difference_vs_raw"] is not None]
        macro_iou = entry["left_right_macro"]["iou"]
        entry["iou_difference_vs_raw"] = {
            "macro_difference": (
                None if macro_iou is None or raw_macro is None else float(macro_iou - raw_macro)
            ),
            "per_view_frame": per_view_frame,
            "distribution": summarize_values(differences),
        }
        entry["mask_source"] = mask_source(method)
        methods[method] = entry
    return {
        "schema_version": SCHEMA_VERSION,
        "split": split_name,
        "pair_ids": [int(value) for value in pair_ids],
        "frame_count": len(pair_ids),
        "views": list(VIEWS),
        "methods": methods,
        "manual_label_condition": {
            "source": "existing LabelMe floor_eligible polygons, exclusive priority "
                      "ignore_uncertain > person > walker > static_other > floor_eligible",
            "role": "two-dimensional image-space condition of this comparison only",
        },
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }


# ---------------------------------------------------------------------------
# Development-split-only method selection
# ---------------------------------------------------------------------------


def _dev_macro(development_metrics: Mapping[str, Any], method: str, field: str) -> float | None:
    value = development_metrics["methods"][method]["left_right_macro"][field]
    return None if value is None else float(value)


def _dev_variance(development_metrics: Mapping[str, Any], method: str) -> float | None:
    value = development_metrics["methods"][method]["iou_distribution"]["variance"]
    return None if value is None else float(value)


def select_primary_frontend(development_metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Choose one preprocessing group using the development split results only.

    The argument carries development-split metrics exclusively.  The holdout
    split is never read here, so it cannot influence, re-select or re-tune the
    group that is later labelled the candidate primary front-end.
    """
    raw_iou = _dev_macro(development_metrics, RAW, "iou")
    if raw_iou is None:
        raise RuntimeError("development split has no evaluable raw floor IoU")

    def ranking_key(method: str) -> tuple[float, float, float, int]:
        iou = _dev_macro(development_metrics, method, "iou")
        precision = _dev_macro(development_metrics, method, "precision")
        variance = _dev_variance(development_metrics, method)
        return (
            -1.0 if iou is None else -iou,
            -1.0 if precision is None else -precision,
            float("inf") if variance is None else variance,
            METHODS.index(method),
        )

    ordered = sorted(METHODS, key=ranking_key)
    ranking = [
        {
            "method": method,
            "development_left_right_macro_iou": _dev_macro(development_metrics, method, "iou"),
            "development_left_right_macro_precision": _dev_macro(development_metrics, method, "precision"),
            "development_left_right_macro_recall": _dev_macro(development_metrics, method, "recall"),
            "development_iou_variance": _dev_variance(development_metrics, method),
        }
        for method in ordered
    ]
    best = ordered[0]
    best_iou = float(_dev_macro(development_metrics, best, "iou"))
    margin = float(best_iou - raw_iou)
    conservative = margin <= CONSERVATIVE_RAW_MARGIN + MARGIN_COMPARISON_EPSILON
    selected = RAW if conservative else best
    tied_with_best = [
        method for method in METHODS
        if _dev_macro(development_metrics, method, "iou") is not None
        and abs(float(_dev_macro(development_metrics, method, "iou")) - best_iou) <= 1e-12
    ]
    tie_break = "not_required"
    if not conservative and len(tied_with_best) > 1:
        tie_break = "floor_precision_then_iou_variance"
    return {
        "criterion": "development split left-right macro average floor IoU",
        "development_pair_ids": list(DEV_PAIR_IDS),
        "holdout_pair_ids_used_for_selection": [],
        "ranking": ranking,
        "best_development_method": best,
        "best_development_macro_iou": best_iou,
        "raw_development_macro_iou": float(raw_iou),
        "best_margin_over_raw": margin,
        "conservative_raw_margin": CONSERVATIVE_RAW_MARGIN,
        "conservative_raw_applied": bool(conservative),
        "tie_break_applied": tie_break,
        "selected_method": selected,
        "locked_parameters": method_parameters(selected),
        "selection_note": (
            "A development margin of at most the conservative margin keeps raw, so a group is only promoted when "
            "the development split separates it from raw by more than the fixed margin"
        ),
    }


# ---------------------------------------------------------------------------
# Strict stereo candidate geometry
# ---------------------------------------------------------------------------


def stereo_candidate_funnel(
    *, left_local: np.ndarray, right_local: np.ndarray, left_mask_local: np.ndarray,
    right_mask_local: np.ndarray, rectification: Any, forward: np.ndarray, reverse: np.ndarray,
    consistency_px: float, candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Per-stage counts of the frozen candidate chain on the same fields.

    The frozen chain is reused unchanged through
    ``observe_local_ground_semantic_stereo_lr_direction_control.reconstruct``.  The
    stage counts re-evaluate that chain's own conditions on the same forward and
    reverse disparity fields purely so the funnel can be reported, and the result
    is accepted only when the surviving pixel set is exactly equal to the frozen
    chain's candidate pixels.  A mismatch aborts the run instead of reporting a
    second, silently different candidate set.
    """
    height, width = left_local.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    translation_x = float(rectification.translation_right[0])
    if translation_x >= 0.0:
        raise RuntimeError("this pipeline assumes the left camera as reference")
    partner_x = np.rint(xx.astype(np.float32) - forward).astype(np.int32)
    clipped = np.clip(partner_x, 0, width - 1)
    at_partner = reverse[yy, clipped]
    depth_scale = abs(float(rectification.virtual_K[0, 0] * translation_x))
    depth = depth_scale / np.where(forward > 0, forward, np.nan)
    left_gray = cv2.cvtColor(left_local, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_local, cv2.COLOR_BGR2GRAY)
    in_bounds = (partner_x >= 0) & (partner_x < width)
    conditions = (
        ("left_mask", left_mask_local > 0),
        ("forward_disparity_above_one", forward > 1.0),
        ("partner_inside_image", in_bounds),
        ("right_mask_at_partner", (right_mask_local[yy, clipped] > 0) & in_bounds),
        ("reverse_disparity_above_one", at_partner > 1.0),
        ("left_right_consistency", np.abs(forward - at_partner) <= float(consistency_px)),
        (
            "photometric_le_45",
            np.abs(left_gray.astype(np.int16) - right_gray[yy, clipped].astype(np.int16))
            <= CHAIN_MAXIMUM_PHOTOMETRIC_DIFFERENCE,
        ),
        (
            "depth_in_range",
            np.isfinite(depth) & (depth > CHAIN_MINIMUM_DEPTH_MM) & (depth < CHAIN_MAXIMUM_DEPTH_MM),
        ),
    )
    cumulative = np.ones((height, width), dtype=bool)
    stages: list[dict[str, Any]] = []
    previous_count: int | None = None
    for stage_name, condition in conditions:
        cumulative = cumulative & condition
        count = int(cumulative.sum())
        stages.append({
            "stage": stage_name,
            "cumulative_pixels": count,
            "lost_at_this_stage": None if previous_count is None else int(previous_count - count),
        })
        previous_count = count
    ys, xs = np.nonzero(cumulative)
    funnel_pixels = {(int(x), int(y)) for x, y in zip(xs, ys)}
    candidate_pixels = {
        (int(x), int(y)) for x, y in zip(candidate["left_pixels"][:, 0], candidate["left_pixels"][:, 1])
    }
    if funnel_pixels != candidate_pixels:
        raise RuntimeError(
            "the reported stage funnel does not reproduce the frozen candidate pixel set; refusing to report a "
            f"second candidate set (funnel {len(funnel_pixels)} pixels, frozen chain {len(candidate_pixels)} pixels)"
        )
    by_stage = {entry["stage"]: entry["cumulative_pixels"] for entry in stages}
    return {
        "stages": stages,
        "mask_pixels_in_local_view": int(by_stage["left_mask"]),
        "strict_stereo_candidate_pixels": int(by_stage["depth_in_range"]),
        "pixels_reaching_left_right_consistency_test": int(by_stage["reverse_disparity_above_one"]),
        "pixels_passing_left_right_consistency": int(by_stage["left_right_consistency"]),
        "pixels_passing_photometric_gate": int(by_stage["photometric_le_45"]),
        "pixels_passing_depth_range": int(by_stage["depth_in_range"]),
        "reproduced_frozen_candidate_pixels": True,
    }


def region_consensus_passed(region: Any) -> bool:
    """Report whether the frozen cross-region gate produced a consensus plane.

    ``pose_app.scene_geometry_variants`` reports a *successful* fit with status
    ``candidate`` and only a failure with ``unavailable``, so the status string
    must not be compared against ``available``; the presence of a consensus
    normal is the reliable predicate.
    """
    return bool(region.normal_left_camera is not None)


def run_stereo_arm(
    *, calibration: StereoCalibration, parameters: Mapping[str, Any], decision_args: Namespace,
    left_upright: np.ndarray, right_upright: np.ndarray, left_mask_upright: np.ndarray,
    right_mask_upright: np.ndarray, frame_index: int, identity_evidence: str,
    runtime_size: tuple[int, int],
) -> dict[str, Any]:
    """Run the frozen strict candidate chain on raw images and one mask pair."""
    shape_upright = left_upright.shape[:2]
    record: dict[str, Any] = {
        "semantic_identity_evidence": identity_evidence,
        "bilateral_masks_present": bool(np.any(left_mask_upright > 0) and np.any(right_mask_upright > 0)),
        "left_mask_fraction_upright": float((left_mask_upright > 0).mean()),
        "right_mask_fraction_upright": float((right_mask_upright > 0).mean()),
        "state": "unavailable",
        "reasons": [],
        "funnel": None,
        "quality": None,
        "region_consensus": None,
        "unaccepted_candidate_plane": None,
        "accepted_plane": None,
        "distinct_inlier_rows": 0,
        "visual_context": None,
    }
    if not record["bilateral_masks_present"]:
        record["reasons"] = ["empty_semantic_candidate_mask"]
        return record

    left_seed = strict.mask_seed_upright(left_mask_upright)
    right_seed = strict.mask_seed_upright(right_mask_upright)
    if left_seed is None or right_seed is None:
        record["reasons"] = ["empty_semantic_candidate_mask"]
        return record

    left_raw, right_raw = strict.inverse_upright(left_upright, right_upright)
    raw_size = (left_raw.shape[1], left_raw.shape[0])
    left_seed_raw = strict.upright_point_to_raw(
        left_seed, "left", (shape_upright[1], shape_upright[0])
    ) * np.asarray((runtime_size[0] / raw_size[0], runtime_size[1] / raw_size[1]))
    right_seed_raw = strict.upright_point_to_raw(
        right_seed, "right", (shape_upright[1], shape_upright[0])
    ) * np.asarray((runtime_size[0] / raw_size[0], runtime_size[1] / raw_size[1]))
    rectification = strict.make_mask_directed_rectification(
        calibration, left_seed_raw, right_seed_raw, runtime_size, float(parameters["virtual_focal_px"])
    )
    left_raw_runtime = cv2.resize(left_raw, runtime_size, interpolation=cv2.INTER_AREA)
    right_raw_runtime = cv2.resize(right_raw, runtime_size, interpolation=cv2.INTER_AREA)
    left_mask_raw = cv2.resize(
        strict.rotate_mask_to_raw(left_mask_upright, "left"), runtime_size, interpolation=cv2.INTER_NEAREST
    )
    right_mask_raw = cv2.resize(
        strict.rotate_mask_to_raw(right_mask_upright, "right"), runtime_size, interpolation=cv2.INTER_NEAREST
    )
    left_local = cv2.remap(left_raw_runtime, rectification.left_map_x, rectification.left_map_y, cv2.INTER_LINEAR)
    right_local = cv2.remap(right_raw_runtime, rectification.right_map_x, rectification.right_map_y, cv2.INTER_LINEAR)
    left_mask_local = cv2.remap(left_mask_raw, rectification.left_map_x, rectification.left_map_y, cv2.INTER_NEAREST)
    right_mask_local = cv2.remap(right_mask_raw, rectification.right_map_x, rectification.right_map_y, cv2.INTER_NEAREST)
    shape = left_local.shape[:2]

    num_disparities = int(parameters["num_disparities"])
    consistency = float(parameters["lr_consistency_px"])
    forward = strict.dense_disparity(left_local, right_local, num_disparities)
    reverse = diagnostic.dense_disparity_right_direction(right_local, left_local, num_disparities)
    candidate = corrected.reconstruct(
        left_local, right_local, left_mask_local, right_mask_local, rectification,
        forward, reverse, num_disparities, consistency,
    )
    score = corrected.fit_and_score(
        candidate=candidate, rectification=rectification, calibration=calibration,
        parameters=parameters, decision_args=decision_args, index=frame_index, shape=shape,
    )
    state, reasons = strict.decide_observation(
        semantic_source="external_binary_mask",
        semantic_identity_evidence=identity_evidence,
        candidate_count=len(candidate["points"]),
        inlier_count=len(score["inliers"]),
        inlier_fraction=score["inlier_fraction"],
        coverage_fraction=score["coverage"],
        residual_mm=score["residual"],
        reprojection_left_px=score["left_error"],
        reprojection_right_px=score["right_error"],
        args=decision_args,
    )
    if identity_evidence == MASK_IDENTITY_PROVIDED_UNVALIDATED:
        if state != "unavailable" or "semantic_identity_unvalidated" not in reasons:
            raise RuntimeError(
                "an automatic candidate mask reached a state other than unavailable; refusing to continue"
            )
    funnel = stereo_candidate_funnel(
        left_local=left_local, right_local=right_local, left_mask_local=left_mask_local,
        right_mask_local=right_mask_local, rectification=rectification, forward=forward,
        reverse=reverse, consistency_px=consistency, candidate=candidate,
    )
    region = estimate_region_consensus(
        candidate["points"], candidate["left_pixels"], image_shape=shape,
        distance_threshold_mm=float(parameters["ransac_distance_mm"]),
        minimum_region_points=REGION_MINIMUM_POINTS,
        maximum_region_angle_deg=REGION_MAXIMUM_ANGLE_DEG,
    )
    inlier_mask = score["inlier_mask"]
    record.update({
        "state": state,
        "reasons": list(reasons),
        "funnel": funnel,
        "quality": {
            "candidate_points": int(len(candidate["points"])),
            "ransac_inliers": int(len(score["inliers"])),
            "ransac_inlier_fraction": score["inlier_fraction"],
            "inlier_coverage_fraction": float(score["coverage"]),
            "median_plane_residual_mm": score["residual"],
            "median_fisheye_reprojection_left_px": score["left_error"],
            "median_fisheye_reprojection_right_px": score["right_error"],
        },
        "region_consensus": {
            "passed": region_consensus_passed(region),
            "status": region.status,
            "reasons": list(region.reasons),
            "inlier_count": int(region.inlier_mask.sum()),
            "metrics": region.metrics,
        },
        "unaccepted_candidate_plane": score["plane"],
        "accepted_plane": None,
        "distinct_inlier_rows": (
            int(len(np.unique(candidate["left_pixels"][inlier_mask][:, 1])))
            if inlier_mask is not None and int(inlier_mask.sum()) else 0
        ),
        "visual_context": (
            left_raw_runtime, left_local, right_local, candidate["left_pixels"], inlier_mask, rectification,
        ),
    })
    return record


def normalize_geometry_row(
    *, pair_id: int, method: str, arm: Mapping[str, Any], reference_comparison: Mapping[str, Any] | None,
    mask_status: str, identity_evidence: str,
) -> dict[str, Any]:
    """Give every arm a complete row shape so no frame is silently dropped."""
    region = arm["region_consensus"]
    if region is None:
        region = {
            "passed": False,
            "status": "not_reached",
            "reasons": ["region_consensus_not_reached_because_the_mask_or_seed_was_unavailable"],
            "inlier_count": 0,
            "metrics": {},
        }
    comparison = reference_comparison
    if comparison is None:
        comparison = {
            "status": "unavailable",
            "reasons": ["reference_comparison_not_applicable_to_the_internal_reference_arm"],
            "normal_angle_deg": None,
            "offset_difference_mm": None,
            "offset_absolute_difference_mm": None,
            "note": "the internal reference is compared against the automatic groups, not against itself",
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "geometry_frame",
        "pair_id": int(pair_id),
        "split": "holdout",
        "method": method,
        "mask_status": mask_status,
        "semantic_identity_evidence": identity_evidence,
        "raw_left_right_images_used_for_geometry": True,
        "enhanced_image_used_for_geometry": False,
        "bilateral_masks_present": bool(arm["bilateral_masks_present"]),
        "left_mask_fraction_upright": arm["left_mask_fraction_upright"],
        "right_mask_fraction_upright": arm["right_mask_fraction_upright"],
        "state": arm["state"],
        "reasons": list(arm["reasons"]),
        "funnel": arm["funnel"],
        "quality": arm["quality"],
        "region_consensus": region,
        "distinct_inlier_rows": arm.get("distinct_inlier_rows", 0),
        "unaccepted_candidate_plane": arm["unaccepted_candidate_plane"],
        "accepted_plane": None,
        "candidate_plane_field": SAFE_PLANE_FIELD,
        "reference_comparison": comparison,
    }


def candidate_mask_or_zero(path: Path, shape: tuple[int, int], reasons: list[str]) -> np.ndarray:
    """Read a saved candidate mask, or substitute an empty mask with a reason.

    A missing saved mask must never abort the frame: the arm still runs and
    reports ``empty_semantic_candidate_mask`` so the failure stays visible.
    """
    if path.is_file():
        return load_binary_mask_exact(path, shape).astype(np.uint8) * 255
    reasons.append("saved_candidate_mask_missing_at_geometry_time")
    return np.zeros(shape, dtype=np.uint8)


def compare_candidate_to_reference(
    candidate_plane: Mapping[str, Any] | None, reference_plane: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Compare one candidate plane with the manual-mask dense-RANSAC reference."""
    reasons: list[str] = []
    if candidate_plane is None:
        reasons.append("candidate_plane_unavailable")
    if reference_plane is None:
        reasons.append("reference_plane_unavailable")
    if reasons:
        return {
            "status": "unavailable",
            "reasons": reasons,
            "normal_angle_deg": None,
            "offset_difference_mm": None,
            "offset_absolute_difference_mm": None,
            "note": (
                "the manual-mask dense-RANSAC plane is a conditional internal reference from the same frozen "
                "chain, not a physical ground reference"
            ),
        }
    angle = angular_error_degrees(
        np.asarray(candidate_plane["normal_left_camera"], dtype=np.float64),
        np.asarray(reference_plane["normal_left_camera"], dtype=np.float64),
        signless=False,
    )
    signed = float(candidate_plane["offset_mm"] - reference_plane["offset_mm"])
    return {
        "status": "available",
        "reasons": [],
        "normal_angle_deg": float(angle),
        "offset_difference_mm": signed,
        "offset_absolute_difference_mm": abs(signed),
        "note": (
            "agreement between two pipelines on shared images and one shared calibration. It is not a physical "
            "ground reference and not a physical accuracy measurement"
        ),
    }


def geometry_aggregate(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    for row in frames:
        for reason in row["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    available = [row for row in frames if row["quality"] is not None]
    with_funnel = [row for row in frames if row["funnel"] is not None]
    comparisons = [row["reference_comparison"] for row in frames
                   if row["reference_comparison"]["status"] == "available"]
    return {
        "frame_count": len(frames),
        "bilateral_mask_frames": sum(1 for row in frames if row["bilateral_masks_present"]),
        "geometry_available_frames": len(available),
        "unavailable_frames": len(frames) - len(available),
        "candidate_points": summarize_values([row["quality"]["candidate_points"] for row in available]),
        "ransac_inliers": summarize_values([row["quality"]["ransac_inliers"] for row in available]),
        "ransac_inlier_fraction": summarize_values(
            [row["quality"]["ransac_inlier_fraction"] for row in available
             if row["quality"]["ransac_inlier_fraction"] is not None]
        ),
        "inlier_coverage_fraction": summarize_values(
            [row["quality"]["inlier_coverage_fraction"] for row in available]
        ),
        "median_plane_residual_mm": summarize_values(
            [row["quality"]["median_plane_residual_mm"] for row in available
             if row["quality"]["median_plane_residual_mm"] is not None]
        ),
        "funnel_stage_medians": {
            key: summarize_values([row["funnel"][key] for row in with_funnel])["median"]
            for key in (
                "mask_pixels_in_local_view",
                "pixels_reaching_left_right_consistency_test",
                "pixels_passing_left_right_consistency",
                "pixels_passing_photometric_gate",
                "pixels_passing_depth_range",
                "strict_stereo_candidate_pixels",
            )
        },
        "funnel_stage_totals": {
            key: int(sum(row["funnel"][key] for row in with_funnel))
            for key in (
                "mask_pixels_in_local_view",
                "pixels_reaching_left_right_consistency_test",
                "pixels_passing_left_right_consistency",
                "pixels_passing_photometric_gate",
                "pixels_passing_depth_range",
                "strict_stereo_candidate_pixels",
            )
        },
        "region_consensus_passed_frames": sum(1 for row in frames if row["region_consensus"]["passed"]),
        "region_consensus_reason_counts": _count_region_reasons(frames),
        "state_counts": _count_states(frames),
        "rejection_reason_counts": reason_counts,
        "reference_comparison": {
            "comparable_frames": len(comparisons),
            "normal_angle_deg": summarize_values([row["normal_angle_deg"] for row in comparisons]),
            "offset_absolute_difference_mm": summarize_values(
                [row["offset_absolute_difference_mm"] for row in comparisons]
            ),
        },
    }


def _count_states(frames: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in frames:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    return counts


def _count_region_reasons(frames: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in frames:
        for reason in row["region_consensus"]["reasons"]:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def geometry_regression_flags(
    raw_frame: Mapping[str, Any] | None, method_frame: Mapping[str, Any]
) -> dict[str, Any]:
    """Flag frames where a better two-dimensional IoU comes with weaker geometry.

    Every condition is reported separately so the reader can see exactly which
    internal quantity moved.
    """
    flags: dict[str, bool | None] = {
        "fewer_candidate_points": None,
        "fewer_ransac_inliers": None,
        "lower_inlier_fraction": None,
        "lower_inlier_coverage": None,
        "higher_median_plane_residual": None,
        "region_consensus_lost": None,
        "geometry_unavailable_where_raw_available": None,
    }
    if raw_frame is None:
        return {"status": "unavailable", "reasons": ["raw_arm_missing"], "flags": flags, "any_geometry_weaker": None}
    if method_frame["quality"] is None or raw_frame["quality"] is None:
        flags["geometry_unavailable_where_raw_available"] = bool(
            method_frame["quality"] is None and raw_frame["quality"] is not None
        )
        return {
            "status": "partial",
            "reasons": ["one_arm_has_no_plane_candidate"],
            "flags": flags,
            "any_geometry_weaker": bool(flags["geometry_unavailable_where_raw_available"]),
        }
    method_quality, raw_quality = method_frame["quality"], raw_frame["quality"]
    flags["fewer_candidate_points"] = bool(method_quality["candidate_points"] < raw_quality["candidate_points"])
    flags["fewer_ransac_inliers"] = bool(method_quality["ransac_inliers"] < raw_quality["ransac_inliers"])
    if method_quality["ransac_inlier_fraction"] is not None and raw_quality["ransac_inlier_fraction"] is not None:
        flags["lower_inlier_fraction"] = bool(
            method_quality["ransac_inlier_fraction"] < raw_quality["ransac_inlier_fraction"]
        )
    flags["lower_inlier_coverage"] = bool(
        method_quality["inlier_coverage_fraction"] < raw_quality["inlier_coverage_fraction"]
    )
    if method_quality["median_plane_residual_mm"] is not None and raw_quality["median_plane_residual_mm"] is not None:
        flags["higher_median_plane_residual"] = bool(
            method_quality["median_plane_residual_mm"] > raw_quality["median_plane_residual_mm"]
        )
    flags["region_consensus_lost"] = bool(
        raw_frame["region_consensus"]["passed"] and not method_frame["region_consensus"]["passed"]
    )
    flags["geometry_unavailable_where_raw_available"] = False
    return {
        "status": "available",
        "reasons": [],
        "flags": flags,
        "any_geometry_weaker": any(value for value in flags.values() if value is not None),
    }


# ---------------------------------------------------------------------------
# Rendering and writing helpers
# ---------------------------------------------------------------------------


def panel(image: np.ndarray, label: str, size: tuple[int, int] = (960, 540)) -> np.ndarray:
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(resized, (0, 0), (size[0], 30), (255, 255, 255), -1)
    cv2.putText(resized, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return resized


def write_sheet(path: Path, panels: Sequence[tuple[np.ndarray, str]], title: str) -> None:
    rendered = [panel(image, label) for image, label in panels]
    while len(rendered) % 2:
        rendered.append(np.zeros_like(rendered[0]))
    sheet = np.vstack([np.hstack(rendered[index:index + 2]) for index in range(0, len(rendered), 2)])
    cv2.rectangle(sheet, (0, 0), (sheet.shape[1], 36), (255, 255, 255), -1)
    cv2.putText(sheet, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"cannot write {path}")


def write_json(path: Path, payload: Any) -> None:
    audit_payload_texts(payload, label=path.name)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    assert_explanation_text(text, path.name)
    path.write_text(text, encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# Inference through the existing Mask2Former interface, in a separate process
# ---------------------------------------------------------------------------


def probe_inference_environment(log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    code = (
        "import json, platform\n"
        "import torch, transformers\n"
        "payload = {'torch_version': torch.__version__, 'cuda_available': bool(torch.cuda.is_available()), "
        "'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "
        "'transformers_version': transformers.__version__, 'python_version': platform.python_version(), "
        "'platform': platform.platform()}\n"
        "print(json.dumps(payload))\n"
    )
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run([sys.executable, "-c", code], stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        return {"status": "unavailable", "reasons": ["inference_environment_probe_failed"], "log": str(log_path)}
    for line in reversed(log_path.read_text(encoding="utf-8").strip().splitlines()):
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict) and "torch_version" in payload:
            return {"status": "ok", **payload}
    return {"status": "unavailable", "reasons": ["inference_environment_probe_unparsable"], "log": str(log_path)}


def run_mask2former(
    *, input_dir: Path, output_dir: Path, pair_ids: Sequence[int], device: str,
    inference_script: Path, log_path: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite inference output: {output_dir}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(inference_script),
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--backends", INFERENCE_BACKEND,
        "--pair-ids", *[str(value) for value in pair_ids],
        "--batch-size", "1",
        "--device", device,
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(command) + "\n\n")
        handle.flush()
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"Mask2Former interface failed ({completed.returncode}); see {log_path}")
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Mask2Former interface wrote no summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    meta = summary.get("backend_metadata", {}).get(INFERENCE_BACKEND, {})
    if meta.get("status") == "unavailable":
        raise RuntimeError(f"Mask2Former backend unavailable: {json.dumps(meta, ensure_ascii=False)}")
    return {"command": command, "backend_metadata": meta, "summary": str(summary_path), "log": str(log_path)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")
    path_preflight = preflight_path_lengths(args.output_dir)

    requested_pair_ids = list(ALL_PAIR_IDS) if not args.pair_ids else [int(value) for value in args.pair_ids]
    parameters = read_frozen_parameters(args.frozen_baseline_dir)
    frozen_names = frozen_frame_names(args.left_dir, args.right_dir, parameters)
    expected_names = [Path(pair_name(value)).stem for value in ALL_PAIR_IDS]
    if frozen_names != expected_names:
        raise RuntimeError(
            f"the frozen frame selection {frozen_names} does not match the fixed 12-frame set {expected_names}"
        )
    for pair_id in requested_pair_ids:
        split_of(pair_id)
    official_run = sorted(requested_pair_ids) == sorted(ALL_PAIR_IDS)
    pairs = selected_pairs(args.left_dir, args.right_dir, requested_pair_ids)
    dev_ids = [value for value in requested_pair_ids if value in DEV_PAIR_IDS]
    holdout_ids = [value for value in requested_pair_ids if value in HOLDOUT_PAIR_IDS]
    runtime_size = (int(parameters["runtime_width"]), int(parameters["runtime_height"]))
    decision_args = Namespace(**parameters)
    calibration = StereoCalibration.load(args.calibration).for_runtime_sizes(runtime_size, runtime_size)

    args.output_dir.mkdir(parents=True)
    write_text(args.output_dir / "command.txt", " ".join([sys.executable, *sys.argv]) + "\n")

    # ---- read the raw upright frames and the manual labels ----------------
    originals: dict[tuple[str, int], np.ndarray] = {}
    manual_masks: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for pair_id, left_path, right_path in pairs:
        for view, path in (("left", left_path), ("right", right_path)):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"cannot read {path}")
            originals[(view, pair_id)] = image
        if originals[("left", pair_id)].shape[:2] != originals[("right", pair_id)].shape[:2]:
            raise RuntimeError(f"upright left/right shape mismatch for pair_{pair_id:04d}")
        shape = originals[("left", pair_id)].shape[:2]
        for view, label_dir in (("left", args.left_label_dir), ("right", args.right_label_dir)):
            manual_masks[(view, pair_id)] = load_manual_view(label_dir, pair_id, shape)

    # ---- phase 1: enhanced images, with the pixel contract enforced -------
    enhanced_root = args.output_dir / "enhanced_images"
    transform_checks: dict[tuple[str, str, int], dict[str, Any]] = {}
    for method in METHODS:
        for view in VIEWS:
            target_dir = enhanced_root / method / view
            target_dir.mkdir(parents=True, exist_ok=True)
            for pair_id, left_path, right_path in pairs:
                source_path = left_path if view == "left" else right_path
                original = originals[(view, pair_id)]
                destination = target_dir / pair_name(pair_id)
                if method == RAW:
                    shutil.copyfile(source_path, destination)
                    reread = cv2.imread(str(destination), cv2.IMREAD_COLOR)
                    if reread is None or not np.array_equal(reread, original):
                        raise RuntimeError("the raw arm copy is not a pixel-exact identity of the original frame")
                    enhanced = original.copy()
                else:
                    enhanced = apply_preprocessing(method, original)
                    if not cv2.imwrite(str(destination), enhanced):
                        raise RuntimeError(f"cannot write {destination}")
                    reread = cv2.imread(str(destination), cv2.IMREAD_COLOR)
                    if reread is None or not np.array_equal(reread, enhanced):
                        raise RuntimeError(f"{destination}: the written enhanced frame is not pixel-identical")
                checks = assert_transform_contract(original, enhanced, method)
                checks["written_file_pixel_identical_to_transform_output"] = True
                transform_checks[(method, view, pair_id)] = checks

    # ---- phase 2: Mask2Former, one separate process per group and view ----
    inference_root = args.output_dir / "inf"
    inference_metadata: dict[str, Any] = {}
    environment_probe = probe_inference_environment(inference_root / "inference_environment.txt")
    for method in METHODS:
        for view in VIEWS:
            code = arm_code(method, view)
            inference_metadata[f"{method}_{view}"] = run_mask2former(
                input_dir=enhanced_root / method / view,
                output_dir=inference_root / code,
                pair_ids=[pair_id for pair_id, _, _ in pairs],
                device=args.device,
                inference_script=args.inference_script,
                log_path=inference_root / f"{code}_stdout.txt",
            )
            inference_metadata[f"{method}_{view}"]["arm_code"] = code

    # ---- phase 3: masks, overlays, semantic records ----------------------
    masks_root = args.output_dir / "floor_masks"
    overlays_root = args.output_dir / "overlays"
    semantic_records: list[dict[str, Any]] = []
    raw_reference: dict[tuple[str, int], dict[str, Any]] = {}
    for method in METHODS:
        for view in VIEWS:
            (masks_root / method / view).mkdir(parents=True, exist_ok=True)
            (overlays_root / method / view).mkdir(parents=True, exist_ok=True)
    for pair_id, _, _ in pairs:
        split = split_of(pair_id)
        for view in VIEWS:
            shape = originals[(view, pair_id)].shape[:2]
            original = originals[(view, pair_id)]
            masks = manual_masks[(view, pair_id)]
            for method in METHODS:
                inference_mask = (
                    inference_root / arm_code(method, view) / INFERENCE_BACKEND / "floor_masks" / pair_name(pair_id)
                )
                mask_reasons: list[str] = []
                if inference_mask.is_file():
                    candidate = load_binary_mask_exact(inference_mask, shape)
                    mask_status = MASK_STATUS_AUTOMATIC_CANDIDATE
                else:
                    candidate = None
                    mask_status = "unavailable"
                    mask_reasons = ["semantic_interface_wrote_no_mask"]
                if candidate is not None:
                    written_mask = masks_root / method / view / pair_name(pair_id)
                    if not cv2.imwrite(str(written_mask), candidate.astype(np.uint8) * 255):
                        raise RuntimeError(f"cannot write {written_mask}")
                    reloaded = load_binary_mask_exact(written_mask, shape)
                    if not np.array_equal(reloaded, candidate):
                        raise RuntimeError("the written candidate mask changed value")
                enhanced = original if method == RAW else apply_preprocessing(method, original)
                evaluation = evaluate_semantic_candidate(masks, candidate)
                if mask_reasons:
                    evaluation["reasons"] = list(mask_reasons) + list(evaluation["reasons"])
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "semantic_frame",
                    "pair_id": pair_id,
                    "split": split,
                    "view": view,
                    "method": method,
                    "mask_status": mask_status,
                    "semantic_identity_evidence": MASK_IDENTITY_PROVIDED_UNVALIDATED,
                    "mask_shape_matches_original_upright_grid": bool(
                        candidate is not None and candidate.shape == shape
                    ),
                    "transform_checks": transform_checks[(method, view, pair_id)],
                    "evaluation": evaluation,
                    "brightness": {
                        "floor_region": gray_region_statistics(enhanced, masks["floor_eligible"]),
                        "non_floor_region": gray_region_statistics(
                            enhanced, np.logical_and(masks["valid_evaluation"], ~masks["floor_eligible"])
                        ),
                        "floor_region_median_gray_delta_vs_raw": None,
                    },
                    "brightness_role": "explanatory diagnostic only, not correctness evidence",
                }
                semantic_records.append(record)
                if candidate is not None:
                    overlay = render_overlay(
                        enhanced, candidate,
                        f"{method} | {pair_name(pair_id)} | green=floor candidate (automatic, unvalidated)",
                    )
                    if not cv2.imwrite(str(overlays_root / method / view / pair_name(pair_id)), overlay):
                        raise RuntimeError("cannot write overlay")
                if method == RAW:
                    raw_reference[(view, pair_id)] = record
    for record in semantic_records:
        reference = raw_reference.get((record["view"], record["pair_id"]))
        raw_median = None if reference is None else reference["brightness"]["floor_region"]["median_gray_0_255"]
        current = record["brightness"]["floor_region"]["median_gray_0_255"]
        if raw_median is not None and current is not None:
            record["brightness"]["floor_region_median_gray_delta_vs_raw"] = float(current - raw_median)

    dev_payload = build_semantic_split_payload(
        split_name="dev", pair_ids=dev_ids,
        records=[row for row in semantic_records if row["split"] == "dev"], raw_records=raw_reference,
    )
    holdout_payload = build_semantic_split_payload(
        split_name="holdout", pair_ids=holdout_ids,
        records=[row for row in semantic_records if row["split"] == "holdout"], raw_records=raw_reference,
    )
    selection = select_primary_frontend(dev_payload)
    selected_method = selection["selected_method"]
    holdout_payload["development_selected_candidate_primary_frontend"] = selected_method
    holdout_payload["holdout_used_for_reselection"] = False

    # ---- phase 4: strict stereo candidate geometry on the holdout split ---
    geometry_root = args.output_dir / "geometry_visualizations"
    geometry_rows: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    reference_rows: list[dict[str, Any]] = []
    arms_by_method: dict[str, dict[int, dict[str, Any]]] = {method: {} for method in METHODS}
    for frame_index, pair_id in enumerate(holdout_ids):
        left_upright = originals[("left", pair_id)]
        right_upright = originals[("right", pair_id)]
        shape = left_upright.shape[:2]
        manual_left = manual_masks[("left", pair_id)]["floor_eligible"].astype(np.uint8) * 255
        manual_right = manual_masks[("right", pair_id)]["floor_eligible"].astype(np.uint8) * 255
        reference_arm = run_stereo_arm(
            calibration=calibration, parameters=parameters, decision_args=decision_args,
            left_upright=left_upright, right_upright=right_upright,
            left_mask_upright=manual_left, right_mask_upright=manual_right,
            frame_index=frame_index, identity_evidence=MASK_IDENTITY_MANUALLY_AUDITED,
            runtime_size=runtime_size,
        )
        reference_plane = reference_arm["unaccepted_candidate_plane"]
        if not args.no_visualization and reference_arm["visual_context"] is not None:
            strict.write_visualization(
                geometry_root / "manual_reference" / f"p{pair_id:04d}.png",
                *reference_arm["visual_context"],
                f"pair_{pair_id:04d} manual floor_eligible internal reference: {reference_arm['state']}",
            )
        reference_rows.append(normalize_geometry_row(
            pair_id=pair_id, method="manual_reference_internal", arm=reference_arm,
            reference_comparison=None, mask_status="manually_audited_for_this_comparison_only",
            identity_evidence=MASK_IDENTITY_MANUALLY_AUDITED,
        ))
        for method in METHODS:
            mask_reasons: list[str] = []
            left_mask = candidate_mask_or_zero(
                masks_root / method / "left" / pair_name(pair_id), shape, mask_reasons
            )
            right_mask = candidate_mask_or_zero(
                masks_root / method / "right" / pair_name(pair_id), shape, mask_reasons
            )
            arm = run_stereo_arm(
                calibration=calibration, parameters=parameters, decision_args=decision_args,
                left_upright=left_upright, right_upright=right_upright,
                left_mask_upright=left_mask, right_mask_upright=right_mask,
                frame_index=frame_index, identity_evidence=MASK_IDENTITY_PROVIDED_UNVALIDATED,
                runtime_size=runtime_size,
            )
            if mask_reasons:
                arm["reasons"] = list(mask_reasons) + list(arm["reasons"])
            comparison = compare_candidate_to_reference(arm["unaccepted_candidate_plane"], reference_plane)
            visual_context = arm.pop("visual_context")
            arms_by_method[method][pair_id] = arm
            geometry_rows[method].append(normalize_geometry_row(
                pair_id=pair_id, method=method, arm=arm, reference_comparison=comparison,
                mask_status=MASK_STATUS_AUTOMATIC_CANDIDATE,
                identity_evidence=MASK_IDENTITY_PROVIDED_UNVALIDATED,
            ))
            if not args.no_visualization and visual_context is not None:
                strict.write_visualization(
                    geometry_root / method / f"p{pair_id:04d}.png",
                    *visual_context,
                    (
                        f"pair_{pair_id:04d} {method}: {arm['state']}; "
                        f"{','.join(arm['reasons']) if arm['reasons'] else 'all internal gates passed'}"
                    ),
                )

    holdout_iou: dict[tuple[str, int, str], float | None] = {}
    for row in semantic_records:
        if row["split"] != "holdout":
            continue
        metrics = row["evaluation"]["metrics"]
        holdout_iou[(row["method"], row["pair_id"], row["view"])] = (
            None if metrics is None else float(metrics["iou"])
        )

    def method_frame_iou(method: str, pair_id: int) -> float | None:
        values = [holdout_iou.get((method, pair_id, view)) for view in VIEWS]
        values = [value for value in values if value is not None]
        return None if not values else float(np.mean(values))

    two_dimensional_vs_geometry: list[dict[str, Any]] = []
    for pair_id in holdout_ids:
        raw_frame = next((row for row in geometry_rows[RAW] if row["pair_id"] == pair_id), None)
        for method in METHODS:
            if method == RAW:
                continue
            method_frame = next(row for row in geometry_rows[method] if row["pair_id"] == pair_id)
            method_iou = method_frame_iou(method, pair_id)
            raw_iou = method_frame_iou(RAW, pair_id)
            delta = None if method_iou is None or raw_iou is None else float(method_iou - raw_iou)
            comparison = geometry_regression_flags(raw_frame, method_frame)
            two_dimensional_vs_geometry.append({
                "pair_id": pair_id,
                "method": method,
                "left_right_mean_iou": method_iou,
                "raw_left_right_mean_iou": raw_iou,
                "iou_difference_vs_raw": delta,
                "two_dimensional_better": None if delta is None else bool(delta > 0.0),
                "geometry": comparison,
                "two_dimensional_better_but_geometry_weaker": (
                    None if delta is None or comparison["any_geometry_weaker"] is None
                    else bool(delta > 0.0 and comparison["any_geometry_weaker"])
                ),
            })

    geometry_methods = {
        method: {
            "frames": geometry_rows[method],
            "aggregate": geometry_aggregate(geometry_rows[method]),
            "is_development_selected_candidate_primary_frontend": bool(method == selected_method),
        }
        for method in METHODS
    }
    geometry_payload = {
        "schema_version": SCHEMA_VERSION,
        "split": "holdout",
        "pair_ids": list(holdout_ids),
        "coordinate_frame": "left_camera",
        "length_unit": "millimeter",
        "frozen_parameters": dict(parameters),
        "chain_constants": {
            "minimum_depth_mm": CHAIN_MINIMUM_DEPTH_MM,
            "maximum_depth_mm": CHAIN_MAXIMUM_DEPTH_MM,
            "maximum_photometric_difference": CHAIN_MAXIMUM_PHOTOMETRIC_DIFFERENCE,
            "reverse_disparity_direction": (
                "corrected rightward search, unchanged from the frozen single-variable control"
            ),
        },
        "geometry_invariants": {
            "raw_left_right_images_used_for_geometry": True,
            "enhanced_image_used_for_geometry": False,
            "calibration": str(args.calibration),
            "frozen_parameter_source": str(args.frozen_baseline_dir / "run_metadata.json"),
            "thresholds_changed": False,
            "mask_pixel_grid": "original upright grid, shape asserted before geometry and never resampled",
            "accepted_ground_state_possible_for_automatic_masks": False,
            "candidate_plane_field": SAFE_PLANE_FIELD,
            "ransac_seed_policy": (
                "the same seed for every arm of one frame, so a comparison between groups is not confounded by a "
                "different RANSAC sample sequence"
            ),
        },
        "methods": geometry_methods,
        "manual_reference_internal": {
            "source": "manual floor_eligible masks through the same frozen chain and dense RANSAC",
            "role": (
                "conditional internal reference under this comparison only, not a physical ground reference and "
                "not a physical accuracy measurement"
            ),
            "mask_identity_evidence": MASK_IDENTITY_MANUALLY_AUDITED,
            "frames": reference_rows,
            "aggregate": geometry_aggregate(reference_rows),
        },
        "two_dimensional_better_but_geometry_weaker": two_dimensional_vs_geometry,
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }

    # ---- phase 5: representative visualizations -------------------------
    representative_manifest = build_representative_visualizations(
        representative_root=args.output_dir / "representative_visualizations",
        pairs=pairs, originals=originals, manual_masks=manual_masks, masks_root=masks_root,
        semantic_records=semantic_records, holdout_ids=holdout_ids, selected_method=selected_method,
        method_frame_iou=method_frame_iou, write=not args.no_visualization,
    )

    # ---- write machine artifacts ----------------------------------------
    with (args.output_dir / "frame_records.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in semantic_records:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
        for row in reference_rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
        for method in METHODS:
            for row in geometry_rows[method]:
                handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")

    write_json(args.output_dir / "semantic_dev_metrics.json", dev_payload)
    write_json(args.output_dir / "semantic_holdout_metrics.json", holdout_payload)
    write_json(args.output_dir / "geometry_holdout_metrics.json", geometry_payload)

    holdout_deltas = {
        method: holdout_payload["methods"][method]["iou_difference_vs_raw"]["macro_difference"]
        for method in METHODS
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "official_run": bool(official_run),
        "pair_ids": [int(value) for value in requested_pair_ids],
        "development_pair_ids": list(dev_ids),
        "holdout_pair_ids": list(holdout_ids),
        "comparison_groups": list(METHODS),
        "single_changed_variable": (
            "the semantic network's input image only; every geometric input, parameter, threshold and gate is the "
            "frozen raw-image chain"
        ),
        "development_selection": selection,
        "development_semantic_metrics": {method: _compact_semantic(dev_payload["methods"][method]) for method in METHODS},
        "holdout_semantic_metrics": {
            method: _compact_semantic(holdout_payload["methods"][method]) for method in METHODS
        },
        "holdout_semantic_difference_vs_raw": holdout_deltas,
        "holdout_geometry_metrics": {method: geometry_methods[method]["aggregate"] for method in METHODS},
        "holdout_geometry_manual_reference": geometry_payload["manual_reference_internal"]["aggregate"],
        "two_dimensional_better_but_geometry_weaker": summarize_two_dimensional_vs_geometry(two_dimensional_vs_geometry),
        "failure_reasons": collect_failure_reasons(semantic_records, geometry_rows, reference_rows),
        "representative_visualizations": representative_manifest,
        "conclusion": build_conclusion(holdout_deltas, selected_method, selection),
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
        "explanation_guard": {
            "validated_vocabulary_strings": len(EXPLANATION_TEXTS) + 1,
            "rule": (
                "any explanation sentence that mentions the evidence boundary subjects must also carry an explicit "
                "negation, otherwise the tool refuses to write it"
            ),
        },
    }
    write_json(args.output_dir / "summary.json", summary)

    run_metadata = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "G20260913_lowlight_floor_preprocessing_v1",
        "official_run": bool(official_run),
        "objective": (
            "test whether a fixed classical low-light front-end changes the existing Mask2Former-Swin-S floor "
            "candidate on the existing manual labels, and what it does to the candidate geometry of the unchanged "
            "frozen strict stereo chain"
        ),
        "frame_partition": {
            "all_pair_ids": [int(value) for value in requested_pair_ids],
            "development_pair_ids": list(dev_ids),
            "dev_use": "method selection only",
            "holdout_pair_ids": list(holdout_ids),
            "holdout_use": "held-out reporting for all four groups; never used to re-select or re-tune",
        },
        "comparison_groups": {method: method_parameters(method) for method in METHODS},
        "fixed_parameters": {
            "preprocessing": {
                "gamma_exponent": GAMMA_EXPONENT,
                "gamma_applied_per_bgr_channel": True,
                "clahe_clip_limit": CLAHE_CLIP_LIMIT,
                "clahe_tile_grid_size": list(CLAHE_TILE_GRID_SIZE),
                "clahe_channel": "LAB lightness channel only",
                "output_dtype": "uint8",
                "output_size": "identical to the input upright image, no crop, rotation or scale",
                "fifth_group_allowed": False,
                "parameter_search_allowed": False,
            },
            "frozen_stereo_parameters": dict(parameters),
            "frozen_parameter_source": str(args.frozen_baseline_dir / "run_metadata.json"),
            "chain_constants": {
                "minimum_depth_mm": CHAIN_MINIMUM_DEPTH_MM,
                "maximum_depth_mm": CHAIN_MAXIMUM_DEPTH_MM,
                "maximum_photometric_difference": CHAIN_MAXIMUM_PHOTOMETRIC_DIFFERENCE,
            },
        },
        "method_selection_rule": {
            "criterion": "development split left-right macro average floor IoU",
            "conservative_raw_margin": CONSERVATIVE_RAW_MARGIN,
            "tie_break": "higher floor precision, then lower per-frame IoU variance",
            "holdout_influence": "none",
        },
        "model_source": {
            "backend": INFERENCE_BACKEND,
            "model_id": INFERENCE_MODEL_ID,
            "interface": str(args.inference_script),
            "interface_process": "separate process, so torch never enters the geometry process",
            "downloads_allowed": False,
            "training_or_fine_tuning": "none",
        },
        "inference": inference_metadata,
        "inference_arm_codes": {f"{method}_{view}": arm_code(method, view) for method in METHODS for view in VIEWS},
        "inference_environment": environment_probe,
        "path_preflight": path_preflight,
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_processors": os.cpu_count(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "opencv_version": cv2.__version__,
        },
        "inputs": {
            "left_dir": str(args.left_dir),
            "right_dir": str(args.right_dir),
            "left_label_dir": str(args.left_label_dir),
            "right_label_dir": str(args.right_label_dir),
            "calibration": str(args.calibration),
            "frozen_baseline_dir": str(args.frozen_baseline_dir),
        },
        "geometry_input_declaration": (
            "every stereo match, disparity field, mask-directed rectification, depth limit, triangulation, RANSAC "
            "fit and region-consensus gate reads the raw upright left and right images, the formal calibration and "
            "the frozen parameters. No enhanced image and no enhanced pixel is used for geometry"
        ),
        "mask_declaration": (
            "every mask from an enhanced image is written in exactly the original upright grid and carries "
            "automatic_candidate status with provided_unvalidated identity evidence, so the strict chain can only "
            "return unavailable and the plane is reported as unaccepted_candidate_plane"
        ),
        "forbidden_items": list(FORBIDDEN_ITEMS),
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }
    write_json(args.output_dir / "run_metadata.json", run_metadata)

    write_text(
        args.output_dir / "EXPERIMENT.md",
        build_experiment_markdown(
            args=args, summary=summary, dev_payload=dev_payload, holdout_payload=holdout_payload,
            geometry_payload=geometry_payload, parameters=parameters, dev_ids=dev_ids,
            holdout_ids=holdout_ids, official_run=official_run,
        ),
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "selected_method": selected_method,
        "conservative_raw_applied": selection["conservative_raw_applied"],
        "holdout_semantic_difference_vs_raw": holdout_deltas,
        "holdout_geometry_available_frames": {
            method: geometry_methods[method]["aggregate"]["geometry_available_frames"] for method in METHODS
        },
    }, ensure_ascii=False))
    return 0


def _compact_semantic(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "left_right_macro": {
            key: value for key, value in entry["left_right_macro"].items() if key != "view_macro_triples"
        },
        "per_view_macro": {
            view: {
                "precision": entry["views"][view]["macro_precision"],
                "recall": entry["views"][view]["macro_recall"],
                "iou": entry["views"][view]["macro_iou"],
                "evaluated_frame_count": entry["views"][view]["evaluated_frame_count"],
            }
            for view in VIEWS
        },
        "iou_distribution": {key: value for key, value in entry["iou_distribution"].items() if key != "values"},
        "iou_difference_vs_raw": {
            "macro_difference": entry["iou_difference_vs_raw"]["macro_difference"],
            "distribution": {
                key: value for key, value in entry["iou_difference_vs_raw"]["distribution"].items()
                if key != "values"
            },
        },
        "person_to_floor": {
            "labelled_view_frame_count": entry["person_to_floor"]["labelled_view_frame_count"],
            "pooled_to_floor_fraction": entry["person_to_floor"]["pooled_to_floor_fraction"],
            "population_metric": False,
        },
        "walker_to_floor": {
            "labelled_view_frame_count": entry["walker_to_floor"]["labelled_view_frame_count"],
            "pooled_to_floor_fraction": entry["walker_to_floor"]["pooled_to_floor_fraction"],
            "population_metric": False,
        },
        "brightness_floor_region_median_gray_delta_vs_raw": entry["brightness_diagnostics"][
            "floor_region_median_gray_delta_vs_raw"
        ],
        "unavailable_reason_counts": entry["unavailable_reason_counts"],
    }


def summarize_two_dimensional_vs_geometry(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if row["two_dimensional_better_but_geometry_weaker"] is not None]
    flagged = [row for row in available if row["two_dimensional_better_but_geometry_weaker"]]
    condition_counts: dict[str, int] = {}
    for row in available:
        for key, value in row["geometry"]["flags"].items():
            if value:
                condition_counts[key] = condition_counts.get(key, 0) + 1
    return {
        "comparable_frame_method_pairs": len(available),
        "flagged_frames": [
            {
                "pair_id": row["pair_id"],
                "method": row["method"],
                "iou_difference_vs_raw": row["iou_difference_vs_raw"],
                "geometry_flags": {key: value for key, value in row["geometry"]["flags"].items() if value},
            }
            for row in flagged
        ],
        "flagged_count": len(flagged),
        "geometry_condition_counts": condition_counts,
        "definition": (
            "a frame and method pair is flagged when its left-right mean floor IoU is higher than the raw arm on "
            "the same frame and at least one internal geometric quantity is weaker than the raw arm: fewer "
            "candidate points, fewer RANSAC inliers, lower inlier fraction, lower inlier coverage, higher median "
            "plane residual, or a lost region-consensus pass. This is a description of internal quantities, not a "
            "verdict about the floor"
        ),
        "per_frame_method": list(rows),
    }


def collect_failure_reasons(
    semantic_records: Sequence[Mapping[str, Any]],
    geometry_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    reference_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    semantic: dict[str, int] = {}
    for row in semantic_records:
        for reason in row["evaluation"]["reasons"]:
            semantic[reason] = semantic.get(reason, 0) + 1
    geometry: dict[str, int] = {}
    region: dict[str, int] = {}
    for rows in geometry_rows.values():
        for row in rows:
            for reason in row["reasons"]:
                geometry[reason] = geometry.get(reason, 0) + 1
            for reason in row["region_consensus"]["reasons"]:
                region[reason] = region.get(reason, 0) + 1
    reference: dict[str, int] = {}
    for row in reference_rows:
        for reason in row["reasons"]:
            reference[reason] = reference.get(reason, 0) + 1
    return {
        "semantic_status_reason_counts": semantic,
        "geometry_status_reason_counts": geometry,
        "geometry_region_consensus_reason_counts": region,
        "manual_reference_internal_reason_counts": reference,
        "note": "every rejection reason is retained; no failing frame is dropped or replaced by a more successful one",
    }


def build_conclusion(
    holdout_deltas: Mapping[str, float | None], selected_method: str, selection: Mapping[str, Any]
) -> dict[str, Any]:
    deltas = {method: value for method, value in holdout_deltas.items() if value is not None}
    selected_delta = deltas.get(selected_method)
    above_margin = [method for method, value in deltas.items()
                    if method != RAW and value > CONSERVATIVE_RAW_MARGIN + MARGIN_COMPARISON_EPSILON]
    if selected_method == RAW:
        headline = "classical_enhancement_frontend_not_shown_effective_on_this_data"
        statement = (
            "No classical group separated from raw by more than the conservative margin on the development split, "
            "so raw was kept as the candidate primary front-end. On the holdout split the honest conclusion is that "
            "the classical enhancement front-end was not shown to be effective on this data, and the next step is "
            "to improve capture exposure or scene illumination rather than to scan more gamma or CLAHE values."
        )
    elif selected_delta is not None and selected_delta > CONSERVATIVE_RAW_MARGIN + MARGIN_COMPARISON_EPSILON:
        headline = "selected_frontend_kept_a_holdout_candidate_iou_advantage"
        statement = (
            f"The development-selected group {selected_method} also shows a higher holdout left-right macro floor "
            "IoU than raw. With six holdout frames and unaudited automatic masks this is candidate image-space "
            "evidence only, and it is not ground truth and not a physical accuracy measurement."
        )
    else:
        headline = "selected_frontend_did_not_hold_on_the_holdout_split"
        statement = (
            f"The development-selected group {selected_method} did not keep its advantage on the holdout split, so "
            "the classical enhancement front-end was not shown to be effective on this data. The next step is to "
            "improve capture exposure or scene illumination rather than to scan more gamma or CLAHE values."
        )
    return {
        "headline": headline,
        "selected_method": selected_method,
        "conservative_raw_applied": bool(selection["conservative_raw_applied"]),
        "holdout_macro_iou_difference_vs_raw": dict(holdout_deltas),
        "groups_above_the_conservative_margin_on_holdout": above_margin,
        "statement": statement,
        "next_step": (
            "Only if the selected front-end improves the holdout split consistently is a single learned low-light "
            "front-end worth testing as one further single-variable experiment; otherwise prioritise capture "
            "exposure or added illumination"
        ),
    }


def build_representative_visualizations(
    *, representative_root: Path, pairs: Sequence[tuple[int, Path, Path]],
    originals: Mapping[tuple[str, int], np.ndarray], manual_masks: Mapping[tuple[str, int], Mapping[str, np.ndarray]],
    masks_root: Path, semantic_records: Sequence[Mapping[str, Any]], holdout_ids: Sequence[int],
    selected_method: str, method_frame_iou: Any, write: bool,
) -> dict[str, Any]:
    """Write the three required representative sheets and describe their selection."""
    all_ids = [pair_id for pair_id, _, _ in pairs]
    dark_arm = selected_method
    dark_arm_role = "development_selected_candidate_primary_frontend"
    if selected_method == RAW:
        dark_arm = GAMMA_0P6_THEN_CLAHE_LAB
        dark_arm_role = "strongest_enhancement_arm_shown_because_the_selected_frontend_is_raw"

    def floor_median(pair_id: int, method: str, view: str) -> float | None:
        for row in semantic_records:
            if row["pair_id"] == pair_id and row["method"] == method and row["view"] == view:
                return row["brightness"]["floor_region"]["median_gray_0_255"]
        return None

    darkness = [floor_median(pair_id, RAW, "left") for pair_id in all_ids]
    darkness_values = [value for value in darkness if value is not None]
    darkness_median = float(np.median(darkness_values)) if darkness_values else None
    dark_candidates = []
    for pair_id in all_ids:
        raw_median = floor_median(pair_id, RAW, "left")
        arm_median = floor_median(pair_id, dark_arm, "left")
        if raw_median is None or arm_median is None:
            continue
        dark_candidates.append({
            "pair_id": pair_id,
            "raw_floor_region_median_gray": raw_median,
            "arm_floor_region_median_gray": arm_median,
            "delta": arm_median - raw_median,
            "is_dark_frame": None if darkness_median is None else bool(raw_median <= darkness_median),
        })
    shortlist = [row for row in dark_candidates if row["is_dark_frame"]] or dark_candidates
    dark_choice = max(shortlist, key=lambda row: abs(row["delta"])) if shortlist else None

    def arm_ranking(arm: str) -> list[dict[str, Any]]:
        rows = [
            {"pair_id": pair_id,
             "delta": float(method_frame_iou(arm, pair_id) - method_frame_iou(RAW, pair_id))}
            for pair_id in holdout_ids
            if method_frame_iou(arm, pair_id) is not None and method_frame_iou(RAW, pair_id) is not None
        ]
        return sorted(rows, key=lambda row: -row["delta"])

    def arm_mean_delta(arm: str) -> float | None:
        values = [row["delta"] for row in arm_ranking(arm)]
        return None if not values else float(np.mean(values))

    other_arms = [method for method in METHODS if method != RAW]
    if selected_method == RAW:
        ranked_arms = [method for method in other_arms if arm_mean_delta(method) is not None]
        ranked_arms.sort(key=lambda method: -float(arm_mean_delta(method)))
        improvement_arm = ranked_arms[0] if ranked_arms else None
        degradation_arm = ranked_arms[-1] if ranked_arms else None
        arm_role = "diagnostic_arm_shown_because_the_selected_frontend_is_raw"
    else:
        improvement_arm = selected_method
        degradation_arm = selected_method
        arm_role = "development_selected_candidate_primary_frontend"

    improvement = arm_ranking(improvement_arm) if improvement_arm else []
    degradation = arm_ranking(degradation_arm) if degradation_arm else []
    best = improvement[0] if improvement and improvement[0]["delta"] != 0.0 else None
    worst = degradation[-1] if degradation and degradation[-1]["delta"] != 0.0 else None

    manifest: dict[str, Any] = {
        "selected_method": selected_method,
        "group1_dark_frame": {
            "role": dark_arm_role,
            "arm_used": dark_arm,
            "selection": "darkest frame by raw floor_region median gray, then the largest absolute brightness change",
            "chosen": dark_choice,
            "candidates": dark_candidates,
        },
        "group2_holdout_improvement": {
            "method": improvement_arm,
            "role": arm_role,
            "selection": "largest left-right mean floor IoU increase versus raw on the holdout split",
            "chosen": best,
            "ranking": improvement,
        },
        "group3_holdout_degradation": {
            "method": degradation_arm,
            "role": arm_role,
            "selection": "largest left-right mean floor IoU decrease versus raw on the holdout split",
            "chosen": worst,
            "ranking": degradation,
            "note": (
                "if no holdout frame separates from raw the chosen entry is null, and an empty or smaller mean "
                "difference is reported instead of an invented example"
            ),
        },
        "files": [],
    }
    if not write:
        manifest["note"] = "visualization disabled by --no-visualization; the selection is still recorded"
        return manifest

    def sheet_for(pair_id: int, arm: str, title: str, path: Path) -> None:
        panels: list[tuple[np.ndarray, str]] = []
        for view in VIEWS:
            original = originals[(view, pair_id)]
            panels.append((original, f"raw {view} {pair_name(pair_id)}"))
            mask_path = masks_root / arm / view / pair_name(pair_id)
            candidate = load_binary_mask_exact(mask_path, original.shape[:2]) if mask_path.is_file() else None
            if candidate is None:
                panels.append((original, f"{arm} {view}: no candidate mask"))
            else:
                shown = original if arm == RAW else apply_preprocessing(arm, original)
                panels.append((
                    render_overlay(shown, candidate, f"{arm} {view} green=floor candidate"),
                    f"{arm} {view}",
                ))
            floor = manual_masks[(view, pair_id)]["floor_eligible"].astype(bool)
            panels.append((
                render_overlay(original, floor, f"manual floor_eligible {view} (2-D condition input)"),
                f"manual label {view}",
            ))
        write_sheet(path, panels, title)
        manifest["files"].append(str(path))

    if dark_choice is not None:
        pair_id = dark_choice["pair_id"]
        sheet_for(
            pair_id, dark_arm,
            (
                f"group 1 dark frame pair_{pair_id:04d}: raw floor median "
                f"{dark_choice['raw_floor_region_median_gray']:.1f} -> {dark_arm} "
                f"{dark_choice['arm_floor_region_median_gray']:.1f} (delta {dark_choice['delta']:+.1f})"
            ),
            representative_root / f"g1_dark_p{pair_id:04d}_{arm_code(dark_arm, 'left')}.png",
        )
    if best is not None and improvement_arm is not None:
        sheet_for(
            best["pair_id"], improvement_arm,
            f"group 2 holdout improvement pair_{best['pair_id']:04d}: {improvement_arm} IoU delta {best['delta']:+.4f}",
            representative_root / f"g2_improve_p{best['pair_id']:04d}_{arm_code(improvement_arm, 'left')}.png",
        )
    if worst is not None and degradation_arm is not None:
        sheet_for(
            worst["pair_id"], degradation_arm,
            f"group 3 holdout degradation pair_{worst['pair_id']:04d}: {degradation_arm} IoU delta {worst['delta']:+.4f}",
            representative_root / f"g3_degrade_p{worst['pair_id']:04d}_{arm_code(degradation_arm, 'left')}.png",
        )
    if not manifest["files"]:
        manifest["note"] = "no representative sheet was written; see the selection blocks for the reason"
    return manifest


def _format_metric(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _reproduction_command(args: argparse.Namespace) -> str:
    return (
        "python realtime_app/tools/benchmark_lowlight_floor_preprocessing.py "
        f"--left-dir {args.left_dir} --right-dir {args.right_dir} "
        f"--left-label-dir {args.left_label_dir} --right-label-dir {args.right_label_dir} "
        f"--calibration {args.calibration} --frozen-baseline-dir {args.frozen_baseline_dir} "
        f"--output-dir {args.output_dir} --device {args.device}"
    )


def build_experiment_markdown(
    *, args: argparse.Namespace, summary: Mapping[str, Any], dev_payload: Mapping[str, Any],
    holdout_payload: Mapping[str, Any], geometry_payload: Mapping[str, Any],
    parameters: Mapping[str, Any], dev_ids: Sequence[int], holdout_ids: Sequence[int], official_run: bool,
) -> str:
    selection = summary["development_selection"]
    selected = selection["selected_method"]
    lines: list[str] = []
    lines.append("# people_1 固定经典暗光预处理对 Mask2Former 地面候选的受控比较")
    lines.append("")
    lines.append(f"状态：`{'official_run' if official_run else 'development_smoke_run'}`")
    lines.append("")
    lines.append("## 实验目标")
    lines.append("")
    lines.append(
        "在既有 people_1 图像、既有左右 12 对人工地面标注、当前 Mask2Former 环境和当前正式标定下，检验四组固定"
        "经典暗光预处理是否改变 floor 候选，并检验该候选掩膜送入原始图像的冻结严格双目链后的内部候选几何。"
    )
    lines.append("")
    lines.append("## 输入与固定条件")
    lines.append("")
    lines.append(f"- 左正立图：`{args.left_dir}`")
    lines.append(f"- 右正立图：`{args.right_dir}`")
    lines.append(f"- 左人工标注：`{args.left_label_dir}`")
    lines.append(f"- 右人工标注：`{args.right_label_dir}`")
    lines.append(f"- 正式标定：`{args.calibration}`")
    lines.append(f"- 冻结双目参数来源：`{args.frozen_baseline_dir / 'run_metadata.json'}`")
    lines.append(f"- 语义后端：`{INFERENCE_MODEL_ID}`，既有接口，独立进程，未下载、未训练、未微调")
    lines.append("- 人工标签互斥优先级：`ignore_uncertain > person > walker > static_other > floor_eligible`")
    lines.append("- 固定双目参数只读，命令行不可改写：")
    lines.append("")
    lines.append("| 参数 | 数值 |")
    lines.append("|---|---:|")
    for key in FROZEN_PARAMETER_KEYS:
        lines.append(f"| `{key}` | {parameters[key]} |")
    lines.append("")
    lines.append("## 唯一变量与四组对照")
    lines.append("")
    lines.append(
        "唯一变量是语义分割网络的输入图像。四组固定为 raw、gamma_0p6（每通道 `255*(x/255)^0.6`）、clahe_lab"
        f"（仅 LAB 的 L 通道，`clipLimit={CLAHE_CLIP_LIMIT}`，`tileGridSize={list(CLAHE_TILE_GRID_SIZE)}`）和 "
        "gamma_0p6_then_clahe_lab。未增加第五组，未按标签调参。"
    )
    lines.append("")
    lines.append("## 开发集与留出集划分")
    lines.append("")
    lines.append(f"- 开发集，只用于选方法：{', '.join(f'pair_{value:04d}' for value in dev_ids)}")
    lines.append(
        f"- 完全留出验收集，报告四组但不用于重新选择：{', '.join(f'pair_{value:04d}' for value in holdout_ids)}"
    )
    lines.append("")
    lines.append("## 二维语义结果")
    lines.append("")
    for split_name, payload in (("开发集", dev_payload), ("留出集", holdout_payload)):
        lines.append(f"### {split_name}：左右宏平均 floor precision / recall / IoU")
        lines.append("")
        lines.append("| 组 | 左 P | 左 R | 左 IoU | 右 P | 右 R | 右 IoU | 左右宏平均 IoU | 相对 raw 的宏平均 IoU 差 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for method in METHODS:
            entry = payload["methods"][method]
            left, right = entry["views"]["left"], entry["views"]["right"]
            lines.append(
                f"| `{method}` | {_format_metric(left['macro_precision'])} | {_format_metric(left['macro_recall'])} | "
                f"{_format_metric(left['macro_iou'])} | {_format_metric(right['macro_precision'])} | "
                f"{_format_metric(right['macro_recall'])} | {_format_metric(right['macro_iou'])} | "
                f"{_format_metric(entry['left_right_macro']['iou'])} | "
                f"{_format_metric(entry['iou_difference_vs_raw']['macro_difference'])} |"
            )
        lines.append("")
        lines.append("逐帧 IoU 分布，左右视图合并，样本数见表：")
        lines.append("")
        lines.append("| 组 | 样本数 | 均值 | 中位 | 标准差 | 最小 | 最大 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for method in METHODS:
            distribution = payload["methods"][method]["iou_distribution"]
            lines.append(
                f"| `{method}` | {distribution['count']} | {_format_metric(distribution['mean'])} | "
                f"{_format_metric(distribution['median'])} | {_format_metric(distribution['std'])} | "
                f"{_format_metric(distribution['min'])} | {_format_metric(distribution['max'])} |"
            )
        lines.append("")
    lines.append("### person 与 walker 标记像素被预测为 floor 的比例")
    lines.append("")
    lines.append("| 划分 | 组 | person 有标注的视图帧数 | person 误报比例 | walker 有标注的视图帧数 | walker 误报比例 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for split_name, payload in (("开发集", dev_payload), ("留出集", holdout_payload)):
        for method in METHODS:
            entry = payload["methods"][method]
            person, walker = entry["person_to_floor"], entry["walker_to_floor"]
            lines.append(
                f"| {split_name} | `{method}` | {person['labelled_view_frame_count']} | "
                f"{_format_metric(person['pooled_to_floor_fraction'])} | {walker['labelled_view_frame_count']} | "
                f"{_format_metric(walker['pooled_to_floor_fraction'])} |"
            )
    lines.append("")
    lines.append(
        "person 与 walker 多边形只存在于少数视图帧，因此上表同时给出样本帧数，这些数字只是逐样本诊断，不是总体指标。"
        "本批人工标注中 person 与 walker 只出现在左 pair_0000，因此留出集没有这两类样本。"
    )
    lines.append("")
    lines.append("### 亮度诊断，仅作解释性诊断")
    lines.append("")
    lines.append("| 组 | 留出集地面区域中位灰度 | 相对 raw 的地面区域中位灰度差 | 非地面区域中位灰度 |")
    lines.append("|---|---:|---:|---:|")
    for method in METHODS:
        brightness = holdout_payload["methods"][method]["brightness_diagnostics"]
        lines.append(
            f"| `{method}` | {_format_metric(brightness['floor_region_median_gray']['median'], 2)} | "
            f"{_format_metric(brightness['floor_region_median_gray_delta_vs_raw']['median'], 2)} | "
            f"{_format_metric(brightness['non_floor_region_median_gray']['median'], 2)} |"
        )
    lines.append("")
    lines.append("## 开发集方法选择")
    lines.append("")
    lines.append(f"- 判据：{selection['criterion']}")
    lines.append(
        f"- 开发集最佳组：`{selection['best_development_method']}`，宏平均 IoU "
        f"{_format_metric(selection['best_development_macro_iou'])}"
    )
    lines.append(f"- raw 开发集宏平均 IoU：{_format_metric(selection['raw_development_macro_iou'])}")
    lines.append(
        f"- 最佳组相对 raw 的差值：{_format_metric(selection['best_margin_over_raw'])}；保守阈值 "
        f"{selection['conservative_raw_margin']}；是否保守回退 raw：{selection['conservative_raw_applied']}"
    )
    lines.append(f"- 并列处理：{selection['tie_break_applied']}")
    lines.append(
        f"- 选定并锁定的候选主前端：`{selected}`，固定参数 "
        f"{json.dumps(selection['locked_parameters'], ensure_ascii=False)}"
    )
    lines.append("- 选择只读取开发集结果，留出集不参与选择或调参。")
    lines.append("")
    lines.append("## 留出集相对 raw 的变化")
    lines.append("")
    lines.append("| 组 | 留出集左右宏平均 IoU | 相对 raw 的宏平均差 | 逐视图帧差值中位 |")
    lines.append("|---|---:|---:|---:|")
    for method in METHODS:
        entry = holdout_payload["methods"][method]
        lines.append(
            f"| `{method}` | {_format_metric(entry['left_right_macro']['iou'])} | "
            f"{_format_metric(entry['iou_difference_vs_raw']['macro_difference'])} | "
            f"{_format_metric(entry['iou_difference_vs_raw']['distribution']['median'])} |"
        )
    lines.append("")
    lines.append("## 留出集几何候选统计")
    lines.append("")
    lines.append("所有几何数字都来自原始左右图、正式标定、冻结视差参数和修正后的反向视差方向；增强图只作为语义分割输入。")
    lines.append("")
    lines.append(
        "| 组 | 双侧掩膜帧 | 几何可用帧 | 局部视图掩膜像素中位 | 到达一致性检验中位 | 一致性通过中位 | "
        "光度通过中位 | 深度有效中位 | 严格双目候选点中位 | 内点中位 | 内点率中位 | 覆盖率中位 | 残差中位 mm | 分区共识通过 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for method in METHODS:
        aggregate = summary["holdout_geometry_metrics"][method]
        funnel = aggregate["funnel_stage_medians"]
        lines.append(
            f"| `{method}` | {aggregate['bilateral_mask_frames']} | {aggregate['geometry_available_frames']} | "
            f"{_format_metric(funnel['mask_pixels_in_local_view'], 1)} | "
            f"{_format_metric(funnel['pixels_reaching_left_right_consistency_test'], 1)} | "
            f"{_format_metric(funnel['pixels_passing_left_right_consistency'], 1)} | "
            f"{_format_metric(funnel['pixels_passing_photometric_gate'], 1)} | "
            f"{_format_metric(funnel['pixels_passing_depth_range'], 1)} | "
            f"{_format_metric(funnel['strict_stereo_candidate_pixels'], 1)} | "
            f"{_format_metric(aggregate['ransac_inliers']['median'], 1)} | "
            f"{_format_metric(aggregate['ransac_inlier_fraction']['median'])} | "
            f"{_format_metric(aggregate['inlier_coverage_fraction']['median'])} | "
            f"{_format_metric(aggregate['median_plane_residual_mm']['median'], 3)} | "
            f"{aggregate['region_consensus_passed_frames']}/{aggregate['frame_count']} |"
        )
    lines.append("")
    lines.append(
        "漏斗各列是累计值，从左到右依次是同一批像素通过更多条件的剩余数量，因此逐列不增；"
        "深度有效一列就是冻结链的严格双目候选点数。"
    )
    lines.append("")
    lines.append("拒绝原因计数，可以重叠：")
    lines.append("")
    lines.append("| 组 | 状态计数 | 拒绝原因计数 | 分区共识未通过原因计数 |")
    lines.append("|---|---|---|---|")
    for method in METHODS:
        aggregate = summary["holdout_geometry_metrics"][method]
        lines.append(
            f"| `{method}` | {json.dumps(aggregate['state_counts'], ensure_ascii=False)} | "
            f"{json.dumps(aggregate['rejection_reason_counts'], ensure_ascii=False)} | "
            f"{json.dumps(aggregate['region_consensus_reason_counts'], ensure_ascii=False)} |"
        )
    lines.append("")
    lines.append(
        "条件内部参考使用人工 floor_eligible 掩膜和同一冻结链的稠密 RANSAC，只用于比较候选平面的法向夹角与 "
        "offset 差，它不是物理地面参考，也不是真实地面。"
    )
    lines.append("")
    lines.append("| 组 | 可比帧 | 相对参考的法向夹角中位 deg | 相对参考的 offset 绝对差中位 mm |")
    lines.append("|---|---:|---:|---:|")
    for method in METHODS:
        comparison = summary["holdout_geometry_metrics"][method]["reference_comparison"]
        lines.append(
            f"| `{method}` | {comparison['comparable_frames']} | "
            f"{_format_metric(comparison['normal_angle_deg']['median'], 3)} | "
            f"{_format_metric(comparison['offset_absolute_difference_mm']['median'], 3)} |"
        )
    reference = geometry_payload["manual_reference_internal"]["aggregate"]
    lines.append("")
    lines.append(
        f"同一冻结链下的人工掩膜内部参考：几何可用 {reference['geometry_available_frames']}/"
        f"{reference['frame_count']} 帧，分区共识通过 {reference['region_consensus_passed_frames']}/"
        f"{reference['frame_count']} 帧，内点率中位 {_format_metric(reference['ransac_inlier_fraction']['median'])}，"
        f"残差中位 {_format_metric(reference['median_plane_residual_mm']['median'], 3)} mm。"
    )
    lines.append("")
    lines.append("## 二维更好但几何更差")
    lines.append("")
    marker = summary["two_dimensional_better_but_geometry_weaker"]
    lines.append(f"- 可比较的帧与组配对：{marker['comparable_frame_method_pairs']}")
    lines.append(f"- 被标记的配对数量：{marker['flagged_count']}")
    lines.append(f"- 定义：{marker['definition']}")
    if marker["flagged_frames"]:
        lines.append("")
        lines.append("| pair | 组 | IoU 差 | 转弱的内部几何量 |")
        lines.append("|---|---|---:|---|")
        for row in marker["flagged_frames"]:
            lines.append(
                f"| pair_{row['pair_id']:04d} | `{row['method']}` | "
                f"{_format_metric(row['iou_difference_vs_raw'])} | "
                f"{json.dumps(row['geometry_flags'], ensure_ascii=False)} |"
            )
    else:
        lines.append("- 没有出现二维 IoU 提高同时内部几何量转弱的配对。")
    lines.append("")
    lines.append("## 失败原因")
    lines.append("")
    failure = summary["failure_reasons"]
    if failure["semantic_status_reason_counts"]:
        lines.append(f"- 二维语义状态与原因计数：{json.dumps(failure['semantic_status_reason_counts'], ensure_ascii=False)}")
    else:
        lines.append("- 二维语义没有失败原因：全部视图帧都给出了候选掩膜与可评价指标。")
    lines.append(f"- 几何状态与拒绝原因计数：{json.dumps(failure['geometry_status_reason_counts'], ensure_ascii=False)}")
    lines.append(
        f"- 分区共识未通过原因计数：{json.dumps(failure['geometry_region_consensus_reason_counts'], ensure_ascii=False)}"
    )
    lines.append(
        f"- 条件内部参考的几何拒绝原因计数：{json.dumps(failure['manual_reference_internal_reason_counts'], ensure_ascii=False)}"
    )
    lines.append(f"- {failure['note']}")
    lines.append("")
    lines.append("## 代表性可视化")
    lines.append("")
    representative = summary["representative_visualizations"]
    for key in ("group1_dark_frame", "group2_holdout_improvement", "group3_holdout_degradation"):
        block = representative[key]
        arm = block.get("arm_used", block.get("method"))
        lines.append(
            f"- {key}：{block['selection']}；使用的组 `{arm}`（角色：{block.get('role', 'n/a')}）；"
            f"选定 {json.dumps(block['chosen'], ensure_ascii=False)}"
        )
    lines.append(f"- 目录：`{args.output_dir / 'representative_visualizations'}`")
    lines.append("")
    lines.append(
        "当开发集保守回退 raw 时，第二和第三组会改用留出集差值最大和最小的非 raw 组，并在角色字段中明确标注它"
        "不是所选前端；这三组图只用于目视核对，不作为正确性证据。"
    )
    lines.append("")
    lines.append("## 结论与边界")
    lines.append("")
    lines.append(f"- {summary['conclusion']['statement']}")
    lines.append(f"- 下一步：{summary['conclusion']['next_step']}")
    lines.append(
        "- 本实验只报告二维候选身份与内部门控证据；更亮的画面、更多的候选点或更高的内点比例都不代表真实地面精度更高。"
    )
    lines.append(
        "- 自动掩膜始终是 automatic_candidate 且身份证据为 provided_unvalidated，因此只输出 unaccepted_candidate_plane "
        "候选字段，不输出已接受地面、足地高度、落地判定、接触判定、支撑判定或步态量。"
    )
    lines.append("- 人工 floor_eligible 掩膜是本比较的二维条件输入，不是物理地面参考，也不是真实地面。")
    lines.append("- 语义指标与几何候选都固定在这 12 帧上，样本很小，不能外推为泛化结论。")
    lines.append("- 所有几何结果都在左相机毫米系内表达；相机随助步器运动且没有已验收的相对位姿，因此不做跨帧地面稳定结论。")
    lines.append("- 本轮没有采集新数据、没有训练或微调任何模型、没有下载或替换语义模型、没有使用 DA3，也没有放宽任何门限。")
    lines.append("")
    lines.append("## 产物")
    lines.append("")
    for name in (
        "command.txt", "run_metadata.json", "semantic_dev_metrics.json", "semantic_holdout_metrics.json",
        "geometry_holdout_metrics.json", "frame_records.jsonl", "summary.json", "EXPERIMENT.md",
    ):
        lines.append(f"- `{name}`")
    for folder in ("enhanced_images", "floor_masks", "overlays", "geometry_visualizations",
                   "representative_visualizations", "inf"):
        lines.append(f"- `{folder}/`")
    lines.append("")
    lines.append(
        "`inf/<arm_code>/` 保存既有 Mask2Former 接口的原始输出与运行日志；arm_code 映射为 "
        + ", ".join(
            f"`{arm_code(method, view)}` = {method}/{view}" for method in METHODS for view in VIEWS
        )
        + "。"
    )
    lines.append("")
    lines.append("## 复现命令")
    lines.append("")
    lines.append("```powershell")
    lines.append(_reproduction_command(args))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
