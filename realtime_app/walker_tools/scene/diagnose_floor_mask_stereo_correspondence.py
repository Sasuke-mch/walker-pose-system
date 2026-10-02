#!/usr/bin/env python3
"""Read-only attribution diagnostic for manual-floor-mask stereo failures.

This tool answers exactly one question: with the *same* left/right manually
audited ``floor_eligible`` masks, the *same* calibration and the *same* frozen
parameters that made ``observe_local_ground_semantic_stereo.py`` report
``direct=2/12``, where do the left/right floor matches disappear?

Boundaries of this tool:

* It never emits a ground state, never writes ``local_ground_state.jsonl`` and
  never changes a rejection reason.  Every accept/reject decision stays in the
  frozen baseline output; this tool only measures and attributes.
* It reads the frozen parameter block from the baseline ``run_metadata.json``
  instead of accepting new values for runtime size, virtual focal length,
  disparity range, left/right consistency, RANSAC distance or any gate
  threshold.  No CLI switch can change any of them.
* It reproduces the baseline candidate set, plane fit, inlier set, coverage and
  reprojection medians per frame and *verifies* them against the frozen baseline
  JSONL; a mismatch aborts the run instead of reporting new numbers.
* The additional measurements (relaxed-matcher probe, approximate block-SAD
  cost-curve probe, texture-free searchability bound) are labelled
  ``attribution_probe``.  They diagnose the frozen pipeline; they are not
  alternative results and cannot be promoted to a candidate set.

No PMPose, no DA3, no temporal propagation, no GroundNet, no walker geometry.

Masks are only ever used as an input image-space restriction.  The tool cannot
and does not claim that a manual mask is correct 2-D geometry, and it separates
"the 2-D manual mask is audited" from "the left and right pixels really
correspond".
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
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


TOOLS_ROOT = _tool_legacy_tools
PROJECT_ROOT = TOOLS_ROOT.parents[1]
REALTIME_ROOT = PROJECT_ROOT / "realtime_app"

from pose_app.calibration import StereoCalibration  # noqa: E402
import walker_tools.scene.observe_local_ground_semantic_stereo as strict  # noqa: E402


SCHEMA_VERSION = "floor_mask_stereo_correspondence_diagnostic_v1"
ASCII_CELL = 12

# Descriptive-only thresholds.  They label measurements; they never take part in
# an accept/reject decision and no gate below depends on them.
TEXTURE_GRADIENT_VERY_LOW = 2.0
TEXTURE_GRADIENT_LOW = 5.0
TEXTURE_WINDOW_STD_FLAT = 1.5
COST_CONTRAST_STRONG_GRAY = 5.0
COST_CONTRAST_WEAK_GRAY = 2.0
PROBE_WINDOW = 7
PROBE_MIN_DISPARITY = 8
PROBE_EXTENDED_DISPARITIES = 320

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

NUMERIC_FIELDS_VERIFIED = (
    "candidate_points",
    "ransac_inliers",
    "ransac_inlier_fraction",
    "inlier_coverage_fraction",
    "median_plane_residual_mm",
    "median_fisheye_reprojection_left_px",
    "median_fisheye_reprojection_right_px",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--left-dir", type=Path, required=True, help="upright left_ccw90 pair_*.png frames")
    parser.add_argument("--right-dir", type=Path, required=True, help="upright right_cw90 pair_*.png frames")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--semantic-left-dir", type=Path, required=True)
    parser.add_argument("--semantic-right-dir", type=Path, required=True)
    parser.add_argument(
        "--frozen-baseline-dir",
        type=Path,
        required=True,
        help="directory holding the frozen run_metadata.json and local_ground_state.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe-samples", type=int, default=4000, help="mask pixels sampled for the cost-curve probe")
    parser.add_argument("--split-half-repeats", type=int, default=3)
    parser.add_argument("--no-ascii", action="store_true", help="skip the text raster inspection files")
    parser.add_argument("--no-visualization", action="store_true")
    return parser.parse_args()


def frozen_parameters(baseline_dir: Path) -> dict[str, Any]:
    metadata = json.loads((baseline_dir / "run_metadata.json").read_text(encoding="utf-8"))
    parameters = metadata["parameters"]
    missing = [key for key in FROZEN_PARAMETER_KEYS if key not in parameters]
    if missing:
        raise RuntimeError(f"frozen baseline metadata is missing parameters: {missing}")
    return {key: parameters[key] for key in FROZEN_PARAMETER_KEYS}


def frozen_rows(baseline_dir: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with (baseline_dir / "local_ground_state.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows[int(row["pair_id"])] = row
    return rows


# ---------------------------------------------------------------------------
# measurement helpers
# ---------------------------------------------------------------------------


def numeric_stats(values: np.ndarray, prefix: str) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {f"{prefix}_count": 0}
    return {
        f"{prefix}_count": int(values.size),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_p05": float(np.percentile(values, 5)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_max": float(np.max(values)),
    }


def texture_statistics(gray: np.ndarray, region: np.ndarray, prefix: str) -> dict[str, Any]:
    if not np.any(region):
        return {f"{prefix}_pixels": 0}
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    values = magnitude[region]
    float_gray = gray.astype(np.float32)
    local_mean = cv2.blur(float_gray, (7, 7))
    local_square = cv2.blur(float_gray ** 2, (7, 7))
    window_std = np.sqrt(np.maximum(local_square - local_mean ** 2, 0.0))
    std_values = window_std[region]
    histogram = cv2.calcHist([gray], [0], (region.astype(np.uint8) * 255), [32], [0, 256]).ravel()
    probabilities = histogram / max(1.0, float(histogram.sum()))
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum())
    return {
        f"{prefix}_pixels": int(values.size),
        f"{prefix}_gradient_median": float(np.median(values)),
        f"{prefix}_gradient_mean": float(np.mean(values)),
        f"{prefix}_gradient_p90": float(np.percentile(values, 90)),
        f"{prefix}_gradient_fraction_below_2": float(np.mean(values < TEXTURE_GRADIENT_VERY_LOW)),
        f"{prefix}_gradient_fraction_below_5": float(np.mean(values < TEXTURE_GRADIENT_LOW)),
        f"{prefix}_window_std_median": float(np.median(std_values)),
        f"{prefix}_window_std_fraction_flat": float(np.mean(std_values < TEXTURE_WINDOW_STD_FLAT)),
        f"{prefix}_gray_median": float(np.median(gray[region])),
        f"{prefix}_gray_entropy_bits": entropy,
    }


def dense_disparity_relaxed(reference: np.ndarray, partner: np.ndarray, num_disparities: int) -> np.ndarray:
    """Frozen matcher with its internal left-right, uniqueness and speckle filters disabled.

    Attribution probe: measures how many pixels have any local block match, so
    the loss caused by the matcher's own filtering can be separated from the
    loss caused by the tool gates.  It never produces a candidate set.
    """
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    partner_gray = cv2.cvtColor(partner, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=7,
        P1=8 * 3 * 7 * 7,
        P2=32 * 3 * 7 * 7,
        disp12MaxDiff=-1,
        uniquenessRatio=0,
        speckleWindowSize=0,
        speckleRange=0,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return matcher.compute(clahe.apply(reference_gray), clahe.apply(partner_gray)).astype(np.float32) / 16.0


def dense_disparity_right_direction(
    reference: np.ndarray, partner: np.ndarray, num_disparities: int
) -> np.ndarray:
    """Matcher settings of the frozen tool with the search direction that matches this geometry.

    The frozen tool computes its reverse field with
    ``dense_disparity(right_local, left_local, ...)``, and OpenCV's SGBM always
    searches the partner to the *left* of the reference pixel.  In this
    calibration the right camera centre lies at +x in the rectified left frame,
    so a right pixel's partner lies to its *right*; the frozen reverse field can
    therefore not represent the correspondence at all.  This helper searches the
    other direction by using a negative ``minDisparity`` and returns the result
    already re-expressed as a positive left-referenced disparity, with 0 for
    pixels the matcher did not match.

    Attribution probe only: it is used to test whether the frozen left/right
    consistency gate carries correspondence information.  It emits no ground
    state.
    """
    if num_disparities < 16 or num_disparities % 16:
        raise ValueError("--num-disparities must be a multiple of 16")
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    partner_gray = cv2.cvtColor(partner, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    minimum = -(num_disparities - 1)
    matcher = cv2.StereoSGBM_create(
        minDisparity=minimum, numDisparities=num_disparities, blockSize=7,
        P1=8 * 3 * 7 * 7, P2=32 * 3 * 7 * 7, disp12MaxDiff=1,
        uniquenessRatio=12, speckleWindowSize=60, speckleRange=2,
        preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    raw = matcher.compute(clahe.apply(reference_gray), clahe.apply(partner_gray)).astype(np.float32) / 16.0
    valid = (raw > float(minimum)) & (raw <= 0.0)
    return np.where(valid, -raw, 0.0).astype(np.float32)


def block_cost_curves(
    left_clahe: np.ndarray,
    right_clahe: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    num_disparities: int,
    label: str,
    minimum_disparity: int = 1,
) -> dict[str, Any]:
    """Approximate per-pixel block cost curves over a declared disparity interval.

    Two costs are evaluated on the same CLAHE-preprocessed gray images:

    * plain 7x7 SAD, which is biased towards small disparities whenever an
      additive brightness gradient exists (fisheye vignetting makes nearby
      patches look more similar for reasons unrelated to correspondence);
    * 7x7 zero-mean SAD (ZSAD), which removes exactly that additive bias.

    ``minimum_disparity`` defaults to 8 in the callers so that the two windows
    can never overlap; overlapping windows have a trivial near-zero cost and
    would otherwise produce a fake minimum at the smallest disparity.

    Questions answered: is there any distinct minimum inside the interval, and
    does the best match sit on the interval boundary.  These curves are
    approximate relative to SGBM's Birchfield-Tomasi cost and they are
    diagnostics of the frozen pipeline, never a candidate set.
    """
    base = {
        "probe": "approximate_block_cost_curves",
        "probe_kind": "attribution_probe_not_a_candidate_set",
        "probe_label": label,
        "probe_searched_disparity_interval": [int(minimum_disparity), int(num_disparities - 1)],
        "probe_note": (
            "7x7 SAD and 7x7 zero-mean SAD on CLAHE gray over the declared interval, starting above the window "
            "width so the patches never overlap; evaluated only to locate each pixel's own cost minimum, approximate "
            "relative to SGBM Birchfield-Tomasi cost"
        ),
        "probe_sampled_pixels": int(len(ys)),
    }
    if len(ys) == 0:
        return base
    half = PROBE_WINDOW // 2
    offsets = [(dy, dx) for dy in range(-half, half + 1) for dx in range(-half, half + 1)]
    area = float(len(offsets))
    left_patches = np.empty((len(ys), len(offsets)), dtype=np.float32)
    for index, (dy, dx) in enumerate(offsets):
        left_patches[:, index] = left_clahe[ys + dy, xs + dx].astype(np.float32)
    left_zero_mean = left_patches - left_patches.mean(axis=1, keepdims=True)
    sad = np.full((num_disparities, len(ys)), np.inf, dtype=np.float32)
    zsad = np.full((num_disparities, len(ys)), np.inf, dtype=np.float32)
    for disparity in range(max(1, minimum_disparity), num_disparities):
        partner_x = xs - disparity
        usable = partner_x >= half
        if not np.any(usable):
            continue
        right_patches = np.empty((len(ys), len(offsets)), dtype=np.float32)
        for index, (dy, dx) in enumerate(offsets):
            right_patches[:, index] = right_clahe[ys + dy, partner_x + dx].astype(np.float32)
        sad[disparity, usable] = np.abs(left_patches - right_patches).sum(axis=1)[usable] / area
        right_zero_mean = right_patches - right_patches.mean(axis=1, keepdims=True)
        zsad[disparity, usable] = np.abs(left_zero_mean - right_zero_mean).sum(axis=1)[usable] / area
    result = dict(base)
    for name, costs in (("sad", sad), ("zsad", zsad)):
        finite = np.isfinite(costs)
        any_finite = finite.any(axis=0)
        rows = np.arange(len(ys))
        best = np.where(any_finite, np.argmin(np.where(finite, costs, np.inf), axis=0), 0)
        best_cost = costs[best, rows]
        suppressed = costs.copy()
        for shift in range(-4, 5):
            suppressed[np.clip(best + shift, 0, num_disparities - 1), rows] = np.inf
        second_cost = np.min(suppressed, axis=0)
        valid = any_finite & np.isfinite(best_cost) & np.isfinite(second_cost)
        if not valid.any():
            result[f"probe_{name}_pixels_with_finite_cost_curve"] = 0
            continue
        best_disparities = best[valid].astype(np.float64)
        contrasts = (second_cost[valid] - best_cost[valid]).astype(np.float64)
        at_top = (best == num_disparities - 1) & valid
        result.update({
            f"probe_{name}_pixels_with_finite_cost_curve": int(valid.sum()),
            **numeric_stats(best_disparities, f"probe_{name}_best_disparity"),
            **numeric_stats(contrasts, f"probe_{name}_best_second_contrast_gray"),
            f"probe_{name}_fraction_best_at_interval_top": float(at_top.sum() / max(1, int(valid.sum()))),
            f"probe_{name}_fraction_contrast_strong": float(np.mean(contrasts >= COST_CONTRAST_STRONG_GRAY)),
            f"probe_{name}_fraction_contrast_weak": float(np.mean(contrasts < COST_CONTRAST_WEAK_GRAY)),
            f"probe_{name}_median_best_disparity_over_interval": float(
                np.median(best_disparities) / float(num_disparities)
            ),
        })
    if "probe_sad_best_disparity_median" in result and "probe_zsad_best_disparity_median" in result:
        result["probe_sad_minus_zsad_best_disparity_median"] = float(
            result["probe_sad_best_disparity_median"] - result["probe_zsad_best_disparity_median"]
        )
    return result


def sampled_cost_curve(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    pixels: np.ndarray,
    num_disparities: int,
    maximum: int,
    label: str,
    measured: np.ndarray | None = None,
    minimum_disparity: int = PROBE_MIN_DISPARITY,
) -> dict[str, Any]:
    """Cost curves evaluated at a specific pixel set, for example the frozen candidates."""
    if pixels is None or len(pixels) == 0:
        return {
            "probe": "approximate_block_cost_curves",
            "probe_kind": "attribution_probe_not_a_candidate_set",
            "probe_label": label,
            "probe_sampled_pixels": 0,
        }
    half = PROBE_WINDOW // 2
    height, width = left_gray.shape[:2]
    stride = max(1, len(pixels) // max(1, maximum))
    selected = np.arange(0, len(pixels), stride)
    ys = pixels[selected, 1].astype(np.int64)
    xs = pixels[selected, 0].astype(np.int64)
    usable = (
        (ys >= half) & (ys < height - half)
        & (xs >= num_disparities + half) & (xs < width - half)
    )
    left_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(left_gray)
    right_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(right_gray)
    result = block_cost_curves(
        left_clahe, right_clahe, ys[usable], xs[usable], num_disparities, label,
        minimum_disparity=minimum_disparity,
    )
    result["probe_available_pixels"] = int(len(pixels))
    result["probe_selected_pixels"] = int(len(selected))
    result["probe_pixels_dropped_at_window_or_range_border"] = int(len(selected) - int(usable.sum()))
    if measured is not None and int(usable.sum()):
        measured_selected = measured[selected][usable].astype(np.float64)
        result.update(numeric_stats(measured_selected, f"probe_{label}_measured_sgbm_disparity"))
        if "probe_zsad_best_disparity_median" in result:
            result[f"probe_{label}_measured_minus_zsad_best_median"] = float(
                result[f"probe_{label}_measured_sgbm_disparity_median"] - result["probe_zsad_best_disparity_median"]
            )
    return result


def consistency_gate_controls(
    disparity: np.ndarray,
    reverse_frozen: np.ndarray,
    reverse_corrected: np.ndarray,
    reaches_gate: np.ndarray,
    consistency_px: float,
    num_disparities: int,
) -> dict[str, Any]:
    """Does the frozen left/right consistency gate carry correspondence information?

    The gate compares the forward disparity with an independently computed
    reverse field at the partner position.  Two null controls sample the same
    reverse field at a deliberately wrong position (shifted partner, other row):
    if the observed pass rate is no better than the nulls, the gate is not
    measuring left/right agreement.  ``reverse_corrected`` is the same matcher
    run with the search direction that matches this calibration, so it also lets
    the same gate be evaluated against a field that can represent the
    correspondence.
    """
    height, width = disparity.shape
    yy, xx = np.mgrid[0:height, 0:width]
    partner = np.rint(xx.astype(np.float32) - disparity).astype(np.int32)
    clip_x = np.clip(partner, 0, width - 1)
    clip_x_shift = np.clip(partner + 17, 0, width - 1)
    clip_y = np.clip(yy + 5, 0, height - 1)
    total = int(reaches_gate.sum())
    chance = float(2.0 * consistency_px + 1.0) / float(num_disparities)

    def rate(field: np.ndarray, sample_x: np.ndarray, sample_y: np.ndarray) -> float | None:
        if total == 0:
            return None
        difference = np.abs(disparity - field[sample_y, sample_x])
        valid = (field[sample_y, sample_x] > 1.0) & reaches_gate
        if not np.any(valid):
            return None
        return float(np.mean(difference[valid] <= consistency_px))

    observed_frozen = rate(reverse_frozen, clip_x, yy)
    observed_corrected = rate(reverse_corrected, clip_x, yy)
    return {
        "probe_kind": "attribution_probe_not_a_candidate_set",
        "gate_limit_px": consistency_px,
        "disparity_range_px": int(num_disparities),
        "chance_rate_for_independent_quantised_fields": chance,
        "pixels_reaching_gate": total,
        "observed_pass_rate_frozen_reverse_field_at_partner": observed_frozen,
        "null_pass_rate_frozen_reverse_field_partner_shifted_17px": rate(reverse_frozen, clip_x_shift, yy),
        "null_pass_rate_frozen_reverse_field_other_row_5px": rate(reverse_frozen, clip_x, clip_y),
        "observed_pass_rate_direction_corrected_field_at_partner": observed_corrected,
        "null_pass_rate_direction_corrected_field_partner_shifted_17px": rate(reverse_corrected, clip_x_shift, yy),
        "null_pass_rate_direction_corrected_field_other_row_5px": rate(reverse_corrected, clip_x, clip_y),
        "note": (
            "the frozen reverse field is produced by OpenCV SGBM with the partner searched to the left of the "
            "reference pixel; in this calibration the right pixel's partner lies to its right, so that field cannot "
            "represent the correspondence.  All numbers here are read-only probes"
        ),
    }


def range_top_population(disparity: np.ndarray, mask: np.ndarray, num_disparities: int) -> dict[str, Any]:
    """How much of the disparity field sits exactly on the top of the frozen range.

    A field pinned at ``numDisparities - 1`` is the classic range saturation
    signature: the true match lies outside the searched interval, so both the
    forward and the reverse field stop at the same boundary value and the
    left/right consistency test then passes for a reason unrelated to matching
    evidence.  This measurement needs no probe and no extra parameter.
    """
    top = disparity >= float(num_disparities - 1)
    inside = top & mask
    components, _, stats, _ = cv2.connectedComponentsWithStats(inside.astype(np.uint8), connectivity=8)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if components > 1 else 0
    return {
        "mask_pixels_at_range_top": int(inside.sum()),
        "mask_fraction_at_range_top": float(inside.sum() / max(1, int(mask.sum()))),
        "mask_range_top_largest_blob_px": largest,
        "whole_view_pixels_at_range_top": int(top.sum()),
        "whole_view_fraction_at_range_top": float(top.mean()),
        "note": "disparity >= numDisparities - 1 in the frozen forward field, restricted to / contrasted with the left mask",
    }


def mask_searchability(left_mask: np.ndarray, right_mask: np.ndarray, num_disparities: int) -> dict[str, Any]:
    """Texture-free upper bound imposed by the frozen disparity search range alone.

    A left-mask pixel can only match a right-mask pixel in the same row inside
    ``x - num_disparities < partner <= x - 1``.  No texture, matcher or
    consistency gate is involved.
    """
    left_pixels = left_mask > 0
    right_pixels = right_mask > 0
    height, width = left_pixels.shape
    prefix = np.zeros((height, width + 1), dtype=np.int32)
    np.cumsum(right_pixels, axis=1, out=prefix[:, 1:])
    column = np.arange(width)
    low = np.clip(column - num_disparities + 1, 0, width)
    high = np.clip(column, 0, width)
    has_partner = (prefix[:, high] - prefix[:, low]) > 0
    searchable = left_pixels & has_partner
    return {
        "left_mask_pixels_local": int(left_pixels.sum()),
        "right_mask_pixels_local": int(right_pixels.sum()),
        "left_mask_pixels_with_any_right_mask_partner_in_range": int(searchable.sum()),
        "left_mask_fraction_searchable_in_range": float(searchable.sum() / max(1, int(left_pixels.sum()))),
        "left_mask_pixels_left_of_range_border": int((left_pixels & (column[None, :] < num_disparities)).sum()),
        "probe_note": "texture-free upper bound from the frozen disparity search range",
    }


def mask_component_breakdown(
    left_mask_local: np.ndarray, valid_disparity: np.ndarray,
    candidate_pixels: np.ndarray, inlier_pixels: np.ndarray,
) -> dict[str, Any]:
    """Per-component breakdown of the manual floor mask in the local view.

    A manual floor mask is a union of several polygons, so a global coverage
    ratio can be satisfied by a couple of pixels landing in a distant patch.
    This breakdown shows which mask patch actually carries matches.
    """
    components, labels, stats, _ = cv2.connectedComponentsWithStats(
        (left_mask_local > 0).astype(np.uint8), connectivity=8
    )
    if components <= 1:
        return {"left_mask_component_count": 0}
    entries: list[dict[str, Any]] = []
    largest_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest_index = min(largest_index, components - 1)
    for label in range(1, components):
        area = int(stats[label, cv2.CC_STAT_AREA])
        component = labels == label
        entry = {
            "component": label,
            "area_px": area,
            "bbox_xywh": [int(value) for value in stats[label, :4]],
            "valid_disparity_fraction": float((valid_disparity & component).sum() / max(1, area)),
            "candidate_pixels": int(component[candidate_pixels[:, 1], candidate_pixels[:, 0]].sum())
            if len(candidate_pixels) else 0,
            "inlier_pixels": int(component[inlier_pixels[:, 1], inlier_pixels[:, 0]].sum())
            if len(inlier_pixels) else 0,
        }
        entries.append(entry)
    inliers_outside_largest = sum(
        entry["inlier_pixels"] for entry in entries if entry["component"] != largest_index
    )
    return {
        "left_mask_component_count": int(components - 1),
        "left_mask_largest_component_area_px": int(stats[largest_index, cv2.CC_STAT_AREA]),
        "left_mask_largest_component_share": float(
            stats[largest_index, cv2.CC_STAT_AREA] / max(1, int((left_mask_local > 0).sum()))
        ),
        "inliers_outside_largest_mask_component": int(inliers_outside_largest),
        "components": entries,
    }


def tls_plane(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    if len(points) < 3:
        return None
    centre = np.mean(points, axis=0)
    _, _, vectors = np.linalg.svd(points - centre, full_matrices=False)
    normal = vectors[-1]
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        return None
    normal = normal / norm
    return normal, -float(normal @ centre)


def normal_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(
        np.clip(abs(first @ second) / (np.linalg.norm(first) * np.linalg.norm(second)), 0.0, 1.0)
    )
    return float(np.degrees(np.arccos(cosine)))


def split_half_plane_stability(points: np.ndarray, repeats: int, seed: int) -> dict[str, Any]:
    """Within-frame plane stability; needs no common coordinate frame."""
    base = {
        "split_half_repeats": 0,
        "split_half_normal_angle_deg_median": None,
        "split_half_normal_angle_deg_max": None,
        "split_half_centroid_distance_diff_mm_median": None,
    }
    if len(points) < 6:
        return base
    centre = np.mean(points, axis=0)
    angles: list[float] = []
    spreads: list[float] = []
    rng = np.random.default_rng(seed)
    for _ in range(max(1, repeats)):
        order = rng.permutation(len(points))
        halves = (order[: len(order) // 2], order[len(order) // 2:])
        fits = [tls_plane(points[half]) for half in halves]
        if any(fit is None for fit in fits):
            continue
        (normal_a, offset_a) = fits[0]  # type: ignore[misc]
        (normal_b, offset_b) = fits[1]  # type: ignore[misc]
        angles.append(normal_angle_deg(normal_a, normal_b))
        distance_a = abs(float(normal_a @ centre) + float(offset_a))
        distance_b = abs(float(normal_b @ centre) + float(offset_b))
        spreads.append(abs(distance_a - distance_b))
    if not angles:
        return base
    return {
        "split_half_repeats": len(angles),
        "split_half_normal_angle_deg_median": float(np.median(angles)),
        "split_half_normal_angle_deg_max": float(np.max(angles)),
        "split_half_centroid_distance_diff_mm_median": float(np.median(spreads)),
    }


def inlier_geometry(
    points: np.ndarray, left_pixels: np.ndarray, left_mask_local: np.ndarray, shape: tuple[int, int]
) -> dict[str, Any]:
    if len(points) == 0 or len(left_pixels) == 0:
        return {"inlier_point_count": 0}
    centre = np.mean(points, axis=0)
    _, singular, _ = np.linalg.svd(points - centre, full_matrices=False)
    distances = np.linalg.norm(points, axis=1)
    result: dict[str, Any] = {
        "inlier_point_count": int(len(points)),
        **numeric_stats(distances, "inlier_camera_distance_mm"),
        "inlier_pca_sigma1_mm": float(singular[0] / np.sqrt(len(points))),
        "inlier_pca_sigma2_mm": float(singular[1] / np.sqrt(len(points))),
        "inlier_pca_sigma3_mm": float(singular[2] / np.sqrt(len(points))),
        "inlier_pca_sigma2_over_sigma1": float(singular[1] / max(1e-9, singular[0])),
    }
    pixel_centre = np.mean(left_pixels, axis=0)
    _, pixel_singular, _ = np.linalg.svd((left_pixels - pixel_centre).astype(np.float64), full_matrices=False)
    result.update({
        "inlier_pixel_pca_sigma1_px": float(pixel_singular[0] / np.sqrt(len(left_pixels))),
        "inlier_pixel_pca_sigma2_px": float(pixel_singular[1] / np.sqrt(len(left_pixels))),
        "inlier_pixel_linearity": float(pixel_singular[1] / max(1e-9, pixel_singular[0])),
        "inlier_distinct_rows": int(np.unique(left_pixels[:, 1]).size),
        "inlier_distinct_columns": int(np.unique(left_pixels[:, 0]).size),
        "inlier_column_span_px": float(np.ptp(left_pixels[:, 0])),
        "inlier_row_span_px": float(np.ptp(left_pixels[:, 1])),
    })
    raster = np.zeros(shape, dtype=np.uint8)
    raster[left_pixels[:, 1], left_pixels[:, 0]] = 1
    components, _, stats, _ = cv2.connectedComponentsWithStats(raster, connectivity=8)
    if components > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        result.update({
            "inlier_connected_components": int(components - 1),
            "inlier_largest_component_share": float(areas.max() / max(1, areas.sum())),
        })
    else:
        result.update({"inlier_connected_components": 0, "inlier_largest_component_share": None})
    hull_area = float(cv2.contourArea(cv2.convexHull(left_pixels.astype(np.float32))))
    mask_pixels = int((left_mask_local > 0).sum())
    mask_hull_area = 0.0
    if mask_pixels >= 3:
        mask_rows, mask_columns = np.nonzero(left_mask_local > 0)
        mask_hull = cv2.convexHull(np.column_stack((mask_columns, mask_rows)).astype(np.float32))
        mask_hull_area = float(cv2.contourArea(mask_hull))
    result.update({
        "inlier_hull_area_px2": hull_area,
        "inlier_points_per_1000px2_in_hull": float(1000.0 * len(left_pixels) / max(1.0, hull_area)),
        "inlier_hull_over_mask_hull": float(hull_area / mask_hull_area) if mask_hull_area > 0 else None,
        "inlier_hull_over_mask_pixels": float(hull_area / max(1, mask_pixels)),
    })
    boundary_distance = cv2.distanceTransform((left_mask_local > 0).astype(np.uint8), cv2.DIST_L2, 3)
    distances_to_boundary = boundary_distance[left_pixels[:, 1], left_pixels[:, 0]]
    result.update({
        "inlier_mask_boundary_distance_px_median": float(np.median(distances_to_boundary)),
        "inlier_mask_boundary_distance_px_p10": float(np.percentile(distances_to_boundary, 10)),
        "inlier_fraction_within_3px_of_mask_boundary": float(np.mean(distances_to_boundary <= 3.0)),
    })
    return result


def disparity_residual_of_plane(
    pixels: np.ndarray, measured: np.ndarray, normal: np.ndarray, offset: float,
    rotation_left: np.ndarray, virtual_k: np.ndarray, translation_x: float,
) -> dict[str, Any]:
    """Disparity-space residual of the fitted plane: an honest, non-trivial check.

    The tool's fisheye reprojection medians are algebraic identities by
    construction (points are built from the left pixel ray and the integer
    partner pixel), so they can never reject anything.  This residual instead
    asks how well the measured disparities agree with the disparity field that
    the fitted plane predicts.
    """
    if len(pixels) == 0:
        return {"disparity_plane_residual_count": 0}
    focal = float(virtual_k[0, 0])
    cx, cy = float(virtual_k[0, 2]), float(virtual_k[1, 2])
    directions = np.column_stack((
        (pixels[:, 0].astype(np.float64) - cx) / focal,
        (pixels[:, 1].astype(np.float64) - cy) / focal,
        np.ones(len(pixels)),
    ))
    rays = (rotation_left.T @ directions.T).T
    denominator = rays @ normal
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = -offset / denominator
    predicted = np.abs(focal * translation_x) / depth
    residual = np.abs(measured - predicted)
    finite = np.isfinite(residual)
    return {
        "disparity_plane_residual_kind": "per-inlier |measured_disparity - plane_predicted_disparity| in local rectified px",
        **{key: value for key, value in numeric_stats(residual[finite], "disparity_plane_residual_px").items()},
    }


# ---------------------------------------------------------------------------
# text raster inspection
# ---------------------------------------------------------------------------


def ascii_raster(
    inlier_pixels: np.ndarray,
    candidate_pixels: np.ndarray,
    mask_local: np.ndarray,
    valid_disparity: np.ndarray,
    cell: int = ASCII_CELL,
) -> str:
    height, width = mask_local.shape[:2]
    rows, columns = height // cell, width // cell
    grid = np.full((rows, columns), ".", dtype="<U1")

    def paint(character: str, condition: np.ndarray) -> None:
        if condition.size == 0 or not np.any(condition):
            return
        reduced_rows = np.nonzero(condition)[0] // cell
        reduced_columns = np.nonzero(condition)[1] // cell
        keep = (reduced_rows < rows) & (reduced_columns < columns)
        grid[reduced_rows[keep], reduced_columns[keep]] = character

    paint("v", valid_disparity & (mask_local == 0))
    paint("m", (mask_local > 0) & ~valid_disparity)
    paint("W", (mask_local > 0) & valid_disparity)
    if candidate_pixels is not None and len(candidate_pixels):
        raster = np.zeros_like(valid_disparity, dtype=bool)
        raster[candidate_pixels[:, 1], candidate_pixels[:, 0]] = True
        paint("C", raster)
    if inlier_pixels is not None and len(inlier_pixels):
        raster = np.zeros_like(valid_disparity, dtype=bool)
        raster[inlier_pixels[:, 1], inlier_pixels[:, 0]] = True
        paint("I", raster)
    header = (
        "legend: I=plane inlier  C=stereo candidate  W=mask with valid disparity  "
        "m=mask without valid disparity  v=valid disparity outside mask  .=nothing"
    )
    body = ["".join(grid[row]) for row in range(rows)]
    return "\n".join([header, *body])


def ascii_texture(gray: np.ndarray, cell: int = ASCII_CELL) -> str:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    height, width = gray.shape[:2]
    rows, columns = height // cell, width // cell
    blocked = magnitude[: rows * cell, : columns * cell].reshape(rows, cell, columns, cell).mean(axis=(1, 3))
    levels = " .:-=+*#%@"
    scale = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 32.0]
    lines = [
        "legend: mean |grad| per cell of the mask-directed local left view; "
        "' '=flat; level thresholds " + ",".join(str(value) for value in scale)
    ]
    for row in range(rows):
        characters = []
        for column in range(columns):
            value = float(blocked[row, column])
            index = 0
            for position, threshold in enumerate(scale):
                if value >= threshold:
                    index = position + 1
            characters.append(levels[min(index, len(levels) - 1)])
        lines.append("".join(characters))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# visualization
# ---------------------------------------------------------------------------


def draw_mask_contour(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()
    cv2.drawContours(output, contours, -1, color, 2)
    return output


def scatter(image: np.ndarray, pixels: np.ndarray, color: tuple[int, int, int], radius: int = 1) -> np.ndarray:
    output = image
    if pixels is None or len(pixels) == 0:
        return output
    for x, y in pixels:
        cv2.circle(output, (int(x), int(y)), radius, color, -1)
    return output


def panel(image: np.ndarray, label: str, size: tuple[int, int] = (960, 540)) -> np.ndarray:
    current = cv2.resize(image, size, interpolation=cv2.INTER_AREA) if image.shape[1::-1] != size else image.copy()
    cv2.rectangle(current, (0, 0), (min(900, current.shape[1]), 26), (255, 255, 255), -1)
    cv2.putText(current, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return current


def write_visualization(
    path: Path,
    *,
    left_local: np.ndarray,
    right_local: np.ndarray,
    left_mask_local: np.ndarray,
    right_mask_local: np.ndarray,
    disparity: np.ndarray,
    num_disparities: int,
    candidate_pixels: np.ndarray,
    inlier_pixels: np.ndarray,
    header: str,
) -> None:
    left_panel = draw_mask_contour(left_local, left_mask_local, (0, 255, 0))
    left_panel = scatter(left_panel, candidate_pixels, (0, 170, 255), 1)
    left_panel = scatter(left_panel, inlier_pixels, (0, 0, 255), 1)
    right_panel = draw_mask_contour(right_local, right_mask_local, (0, 255, 0))
    scaled = np.clip(disparity / float(num_disparities), 0.0, 1.0)
    disparity_panel = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    disparity_panel[disparity <= 1.0] = 0
    disparity_panel = draw_mask_contour(disparity_panel, left_mask_local, (255, 255, 255))
    left_gray = cv2.cvtColor(left_local, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(left_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(left_gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    texture_panel = cv2.applyColorMap(np.clip(magnitude * 4.0, 0, 255).astype(np.uint8), cv2.COLORMAP_BONE)
    texture_panel = draw_mask_contour(texture_panel, left_mask_local, (0, 255, 0))
    sheet = np.vstack((
        np.hstack((
            panel(left_panel, "local left: green=mask, orange=candidate, red=inlier"),
            panel(right_panel, "local right: green=mask"),
        )),
        np.hstack((
            panel(disparity_panel, "frozen SGBM disparity (turbo), white=left mask, black=invalid"),
            panel(texture_panel, "local left |grad| x4, green=mask"),
        )),
    ))
    cv2.rectangle(sheet, (0, 0), (sheet.shape[1], 30), (255, 255, 255), -1)
    cv2.putText(sheet, header[:150], (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"cannot write {path}")


def write_mask_overlay(
    path: Path,
    left_upright: np.ndarray,
    right_upright: np.ndarray,
    left_mask_upright: np.ndarray,
    right_mask_upright: np.ndarray,
    left_seed: np.ndarray,
    right_seed: np.ndarray,
) -> None:
    panels = []
    for image, mask, seed, label in (
        (left_upright, left_mask_upright, left_seed, "upright left: manual floor mask + seed"),
        (right_upright, right_mask_upright, right_seed, "upright right: manual floor mask + seed"),
    ):
        drawn = draw_mask_contour(image, mask, (0, 255, 0))
        cv2.drawMarker(drawn, (int(seed[0]), int(seed[1])), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
        panels.append(panel(drawn, label, (540, 960)))
    sheet = np.hstack(panels)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"cannot write {path}")


# ---------------------------------------------------------------------------
# per-frame processing
# ---------------------------------------------------------------------------


def process_frame(
    *,
    name: str,
    index: int,
    left_dir: Path,
    right_dir: Path,
    semantic_left_dir: Path,
    semantic_right_dir: Path,
    calibration: StereoCalibration,
    runtime_size: tuple[int, int],
    parameters: dict[str, Any],
    frozen_row: dict[str, Any],
    probe_samples: int,
    split_half_repeats: int,
    want_visualization: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    num_disparities = int(parameters["num_disparities"])
    consistency = float(parameters["lr_consistency_px"])
    ransac_distance = float(parameters["ransac_distance_mm"])
    ransac_iterations = int(parameters["ransac_iterations"])
    focal = float(parameters["virtual_focal_px"])

    left_upright = cv2.imread(str(left_dir / name), cv2.IMREAD_COLOR)
    right_upright = cv2.imread(str(right_dir / name), cv2.IMREAD_COLOR)
    if left_upright is None or right_upright is None:
        raise RuntimeError(f"cannot read pair {name}")
    shape_upright = left_upright.shape[:2]
    if right_upright.shape[:2] != shape_upright:
        raise RuntimeError(f"upright left/right shape mismatch for {name}")

    left_mask_upright = strict.load_mask(semantic_left_dir / name, shape_upright)
    right_mask_upright = strict.load_mask(semantic_right_dir / name, shape_upright)
    left_seed = strict.mask_seed_upright(left_mask_upright)
    right_seed = strict.mask_seed_upright(right_mask_upright)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pair_id": int(Path(name).stem.removeprefix("pair_")),
        "frame_id": name,
        "coordinate_frame": "left_camera",
        "length_unit": "millimeter",
        "role": "read-only attribution diagnostic; no ground state is emitted",
        "frozen_baseline_observation_state": frozen_row["observation_state"],
        "frozen_baseline_reason": frozen_row["reason"],
        "frozen_baseline_quality": frozen_row["quality"],
    }

    # --- 2-D mask geometry, before any geometry or matching -------------------
    upright_area = float(shape_upright[0] * shape_upright[1])
    left_upright_pixels = int((left_mask_upright > 0).sum())
    right_upright_pixels = int((right_mask_upright > 0).sum())
    record["mask_geometry_upright"] = {
        "upright_size_width_height": [int(shape_upright[1]), int(shape_upright[0])],
        "left_mask_pixels": left_upright_pixels,
        "right_mask_pixels": right_upright_pixels,
        "left_mask_fraction": float(left_upright_pixels / upright_area),
        "right_mask_fraction": float(right_upright_pixels / upright_area),
        "left_seed_upright_xy": [float(left_seed[0]), float(left_seed[1])] if left_seed is not None else None,
        "right_seed_upright_xy": [float(right_seed[0]), float(right_seed[1])] if right_seed is not None else None,
        "note": "mask area and identity are image-space facts only; they are not stereo correspondence evidence",
    }

    if left_seed is None or right_seed is None:
        record["status"] = "unavailable_empty_mask"
        return record, {}

    left_raw, right_raw = strict.inverse_upright(left_upright, right_upright)
    raw_size = (left_raw.shape[1], left_raw.shape[0])
    left_seed_raw = strict.upright_point_to_raw(left_seed, "left", (shape_upright[1], shape_upright[0])) * np.asarray(
        (runtime_size[0] / raw_size[0], runtime_size[1] / raw_size[1])
    )
    right_seed_raw = strict.upright_point_to_raw(right_seed, "right", (shape_upright[1], shape_upright[0])) * np.asarray(
        (runtime_size[0] / raw_size[0], runtime_size[1] / raw_size[1])
    )
    rectification = strict.make_mask_directed_rectification(calibration, left_seed_raw, right_seed_raw, runtime_size, focal)
    left_raw = cv2.resize(left_raw, runtime_size, interpolation=cv2.INTER_AREA)
    right_raw = cv2.resize(right_raw, runtime_size, interpolation=cv2.INTER_AREA)
    left_mask_raw = cv2.resize(strict.rotate_mask_to_raw(left_mask_upright, "left"), runtime_size, interpolation=cv2.INTER_NEAREST)
    right_mask_raw = cv2.resize(strict.rotate_mask_to_raw(right_mask_upright, "right"), runtime_size, interpolation=cv2.INTER_NEAREST)
    left_local = cv2.remap(left_raw, rectification.left_map_x, rectification.left_map_y, cv2.INTER_LINEAR)
    right_local = cv2.remap(right_raw, rectification.right_map_x, rectification.right_map_y, cv2.INTER_LINEAR)
    left_mask_local = cv2.remap(left_mask_raw, rectification.left_map_x, rectification.left_map_y, cv2.INTER_NEAREST)
    right_mask_local = cv2.remap(right_mask_raw, rectification.right_map_x, rectification.right_map_y, cv2.INTER_NEAREST)
    shape = left_local.shape[:2]

    right_seed_on_left = right_seed_raw.copy()

    def usable_map(map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
        return (
            np.isfinite(map_x) & np.isfinite(map_y)
            & (map_x >= 0.0) & (map_x <= runtime_size[0] - 1.0)
            & (map_y >= 0.0) & (map_y <= runtime_size[1] - 1.0)
        )

    left_usable = usable_map(rectification.left_map_x, rectification.left_map_y)
    right_usable = usable_map(rectification.right_map_x, rectification.right_map_y)
    left_mask_pixels_local = int((left_mask_local > 0).sum())
    right_mask_pixels_local = int((right_mask_local > 0).sum())
    record["mask_geometry_local"] = {
        "runtime_size_width_height": [int(runtime_size[0]), int(runtime_size[1])],
        "left_mask_pixels_raw_runtime": int((left_mask_raw > 0).sum()),
        "right_mask_pixels_raw_runtime": int((right_mask_raw > 0).sum()),
        "left_mask_pixels_local": left_mask_pixels_local,
        "right_mask_pixels_local": right_mask_pixels_local,
        "left_mask_survival_raw_to_local": float(left_mask_pixels_local / max(1, int((left_mask_raw > 0).sum()))),
        "right_mask_survival_raw_to_local": float(right_mask_pixels_local / max(1, int((right_mask_raw > 0).sum()))),
        "left_local_view_pixels_mapping_inside_source": int(left_usable.sum()),
        "left_local_view_usable_fraction": float(left_usable.mean()),
        "right_local_view_usable_fraction": float(right_usable.mean()),
        "left_mask_local_over_usable_pixels": float(left_mask_pixels_local / max(1, int(left_usable.sum()))),
        "left_mask_local_pixels_on_unusable_map": int(((left_mask_local > 0) & ~left_usable).sum()),
        "left_seed_raw_runtime_xy": [float(left_seed_raw[0]), float(left_seed_raw[1])],
        "right_seed_raw_runtime_xy": [float(right_seed_on_left[0]), float(right_seed_on_left[1])],
        "note": (
            "survival = fraction of the runtime-frame mask kept inside the mask-directed local virtual view; "
            "usable = local pixels whose rectified ray maps back inside the runtime source image"
        ),
    }

    # --- frozen matcher, reproduced bit-for-bit -------------------------------
    left_gray = cv2.cvtColor(left_local, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_local, cv2.COLOR_BGR2GRAY)
    disparity = strict.dense_disparity(left_local, right_local, num_disparities)
    reverse = strict.dense_disparity(right_local, left_local, num_disparities)

    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    partner_x = np.rint(xx.astype(np.float32) - disparity).astype(np.int32)
    clipped = np.clip(partner_x, 0, width - 1)
    at_partner = reverse[yy, clipped]
    translation_x = float(rectification.translation_right[0])
    focal_used = float(rectification.virtual_K[0, 0])
    # Mirrors the frozen tool exactly: the numerator stays a Python float, so
    # NumPy 2 promotes the division with the float32 disparity map to float32.
    # The frozen reconstruction therefore evaluates its depth scale in float32,
    # and this diagnostic keeps that behaviour so the numbers are comparable
    # bit-for-bit instead of "close enough".
    depth_scale = abs(float(rectification.virtual_K[0, 0] * translation_x))
    depth = depth_scale / np.where(disparity > 0, disparity, np.nan)
    in_bounds = (partner_x >= 0) & (partner_x < width)

    conditions = {
        "left_mask": left_mask_local > 0,
        "disparity_above_one": disparity > 1.0,
        "partner_inside_image": in_bounds,
        "right_mask_at_partner": (right_mask_local[yy, clipped] > 0) & in_bounds,
        "reverse_disparity_above_one": at_partner > 1.0,
        "left_right_consistency": np.abs(disparity - at_partner) <= consistency,
        "photometric_le_45": np.abs(left_gray.astype(np.int16) - right_gray[yy, clipped].astype(np.int16)) <= 45,
        "depth_in_range": np.isfinite(depth) & (depth > 250.0) & (depth < 8000.0),
    }
    cumulative = np.ones(shape, dtype=bool)
    funnel_entries: list[dict[str, Any]] = []
    previous_count: int | None = None
    for key, condition in conditions.items():
        cumulative = cumulative & condition
        count = int(cumulative.sum())
        funnel_entries.append({
            "stage": key,
            "cumulative_pixels": count,
            "lost_at_this_stage": None if previous_count is None else int(previous_count - count),
        })
        previous_count = count
    candidate_mask = cumulative
    drop_one = {}
    for key in conditions:
        if key == "disparity_above_one":
            continue
        reduced = np.ones(shape, dtype=bool)
        for other_key, other in conditions.items():
            if other_key == key:
                continue
            reduced &= other
        drop_one[key] = int(reduced.sum())
    record["funnel"] = {
        "stages": funnel_entries,
        "final_candidate_pixels": int(candidate_mask.sum()),
        "drop_one_condition_candidate_counts": drop_one,
        "note": "stage counts are cumulative pixels surviving all conditions up to and including that stage",
    }

    # How far apart the forward and the independently computed reverse disparity
    # field actually are, over the pixels that reach the consistency test.  This
    # quantifies what the frozen <= lr_consistency_px gate is throwing away.
    reaches_consistency = (
        conditions["left_mask"]
        & conditions["disparity_above_one"]
        & conditions["partner_inside_image"]
        & conditions["right_mask_at_partner"]
        & conditions["reverse_disparity_above_one"]
    )
    mismatch = np.abs(disparity - at_partner)[reaches_consistency]
    record["consistency_mismatch"] = {
        "kind": "absolute difference between the forward and reverse disparity fields in the local rectified view",
        "pixels_reaching_consistency_test": int(reaches_consistency.sum()),
        **numeric_stats(mismatch, "consistency_mismatch_px"),
        "fraction_within_0_5px": float(np.mean(mismatch <= 0.5)) if mismatch.size else None,
        "fraction_within_1px": float(np.mean(mismatch <= 1.0)) if mismatch.size else None,
        "fraction_within_1_5px": float(np.mean(mismatch <= consistency)) if mismatch.size else None,
        "fraction_within_2px": float(np.mean(mismatch <= 2.0)) if mismatch.size else None,
        "fraction_within_3px": float(np.mean(mismatch <= 3.0)) if mismatch.size else None,
        "fraction_within_5px": float(np.mean(mismatch <= 5.0)) if mismatch.size else None,
        "frozen_limit_px": consistency,
    }

    # Does the frozen consistency gate carry correspondence information at all?
    # The frozen reverse field is produced with OpenCV's leftward search, which
    # cannot represent this calibration's correspondence; the corrected field
    # searches the geometrically valid direction with every other setting frozen.
    reverse_corrected = dense_disparity_right_direction(right_local, left_local, num_disparities)
    record["consistency_gate_controls"] = consistency_gate_controls(
        disparity, reverse, reverse_corrected, reaches_consistency, consistency, num_disparities
    )
    partner_for_corrected = np.clip(partner_x, 0, width - 1)
    at_partner_corrected = reverse_corrected[yy, partner_for_corrected]
    corrected_conditions = dict(conditions)
    corrected_conditions["reverse_disparity_above_one"] = at_partner_corrected > 1.0
    corrected_conditions["left_right_consistency"] = np.abs(disparity - at_partner_corrected) <= consistency
    corrected_all = np.ones(shape, dtype=bool)
    for condition in corrected_conditions.values():
        corrected_all &= condition
    record["consistency_gate_controls"].update({
        "direction_corrected_candidate_pixels_probe": int(corrected_all.sum()),
        "frozen_candidate_pixels": int(candidate_mask.sum()),
        "direction_corrected_field_valid_fraction": float((reverse_corrected > 1.0).mean()),
        "frozen_reverse_field_valid_fraction": float((reverse > 1.0).mean()),
        "direction_corrected_field_median_disparity": float(np.median(reverse_corrected[reverse_corrected > 1.0]))
        if np.any(reverse_corrected > 1.0) else None,
        "frozen_reverse_field_median_disparity": float(np.median(reverse[reverse > 1.0]))
        if np.any(reverse > 1.0) else None,
    })

    # --- candidate points, identical expressions and identical ordering -------
    ys, left_x = np.nonzero(candidate_mask)
    right_x = partner_x[ys, left_x]
    measured = disparity[ys, left_x]
    candidate_depth = depth_scale / measured
    rectified_x = (left_x.astype(np.float64) - rectification.virtual_K[0, 2]) * candidate_depth / focal_used
    rectified_y = (ys.astype(np.float64) - rectification.virtual_K[1, 2]) * candidate_depth / focal_used
    points = (rectification.rotation_left.T @ np.column_stack((rectified_x, rectified_y, candidate_depth)).T).T
    left_pixels = np.column_stack((left_x, ys)).astype(np.int32)
    right_pixels = np.column_stack((right_x, ys)).astype(np.int32)

    fit_points = points if len(points) <= 12000 else points[np.linspace(0, len(points) - 1, 12000, dtype=np.int64)]
    fit = strict.fit_plane_ransac(fit_points, ransac_distance, ransac_iterations, 20260909 + index)
    inlier_mask = None if fit is None else np.abs(points @ fit.normal + fit.offset) <= ransac_distance
    inliers = points[inlier_mask] if inlier_mask is not None else np.empty((0, 3))
    inlier_left_pixels = left_pixels[inlier_mask] if inlier_mask is not None else np.empty((0, 2), dtype=np.int32)
    inlier_right_pixels = right_pixels[inlier_mask] if inlier_mask is not None else np.empty((0, 2), dtype=np.int32)
    coverage = strict.local_coverage_fraction(inlier_left_pixels, shape) if inlier_mask is not None else 0.0
    left_error, right_error = strict.reprojection_medians(
        inliers, inlier_left_pixels, inlier_right_pixels, rectification, calibration
    )
    fraction = None if len(points) == 0 else float(len(inliers) / len(points))
    residual = None if fit is None else fit.median_distance_mm
    reproduced = {
        "candidate_points": int(len(points)),
        "ransac_inliers": int(len(inliers)),
        "ransac_inlier_fraction": fraction,
        "inlier_coverage_fraction": float(coverage),
        "median_plane_residual_mm": residual,
        "median_fisheye_reprojection_left_px": left_error,
        "median_fisheye_reprojection_right_px": right_error,
    }
    verification: dict[str, Any] = {"matched": True, "exact": True, "fields": {}, "tolerance_relative": 1e-6}
    for key in NUMERIC_FIELDS_VERIFIED:
        expected = frozen_row["quality"][key]
        actual = reproduced[key]
        if expected is None or actual is None:
            exact = expected == actual
            within = exact
            relative = 0.0 if exact else None
        else:
            exact = float(expected) == float(actual)
            relative = abs(float(expected) - float(actual)) / max(1e-12, abs(float(expected)))
            within = relative <= 1e-6
        verification["fields"][key] = {
            "frozen": expected,
            "reproduced": actual,
            "exact": bool(exact),
            "relative_difference": relative,
            "within_tolerance": bool(within),
        }
        if not exact:
            verification["exact"] = False
        if not within:
            verification["matched"] = False
    record["frozen_baseline_reproduction"] = {
        "candidate_points": reproduced["candidate_points"],
        "ransac_inliers": reproduced["ransac_inliers"],
        "ransac_inlier_fraction": fraction,
        "inlier_coverage_fraction": float(coverage),
        "median_plane_residual_mm": residual,
        "median_fisheye_reprojection_left_px": left_error,
        "median_fisheye_reprojection_right_px": right_error,
        "depth_scale_evaluated_in_float32": True,
        "depth_scale_note": (
            "the frozen tool divides a Python-float depth scale by the float32 disparity map, so the frozen "
            "reconstruction itself evaluates depth in float32; this diagnostic mirrors that to stay comparable"
        ),
        "verification": verification,
    }

    # --- gate attribution with the frozen thresholds --------------------------
    gates = {
        "insufficient_stereo_candidates": int(len(points)) < int(parameters["minimum_candidates"]),
        "insufficient_plane_inliers": int(len(inliers)) < int(parameters["minimum_inliers"]),
        "low_ransac_inlier_fraction": fraction is None or fraction < float(parameters["minimum_inlier_fraction"]),
        "insufficient_inlier_coverage": float(coverage) < float(parameters["minimum_coverage_fraction"]),
        "high_plane_residual": residual is None or residual > float(parameters["maximum_median_residual_mm"]),
        "high_or_missing_fisheye_reprojection": (
            left_error is None
            or right_error is None
            or max(left_error, right_error) > float(parameters["maximum_median_reprojection_px"])
        ),
    }
    record["frozen_gate_attribution"] = {
        "thresholds": {
            "minimum_candidates": int(parameters["minimum_candidates"]),
            "minimum_inliers": int(parameters["minimum_inliers"]),
            "minimum_inlier_fraction": float(parameters["minimum_inlier_fraction"]),
            "minimum_coverage_fraction": float(parameters["minimum_coverage_fraction"]),
            "maximum_median_residual_mm": float(parameters["maximum_median_residual_mm"]),
            "maximum_median_reprojection_px": float(parameters["maximum_median_reprojection_px"]),
        },
        "gate_failed": {key: bool(value) for key, value in gates.items()},
        "gate_margin": {
            "candidates_shortfall": int(parameters["minimum_candidates"]) - int(len(points)),
            "inliers_shortfall": int(parameters["minimum_inliers"]) - int(len(inliers)),
            "coverage_ratio_to_threshold": float(coverage) / max(1e-12, float(parameters["minimum_coverage_fraction"])),
        },
        "reason_matches_frozen_baseline": sorted(
            [key for key, value in gates.items() if value]
        ) == sorted(frozen_row["reason"]),
    }
    record["reprojection_gate_note"] = {
        "left_reprojection_is_algebraic_identity": True,
        "right_reprojection_is_algebraic_identity": True,
        "explanation": (
            "candidate points are built from the left rectified pixel ray and the integer partner pixel, so both "
            "fisheye reprojection medians are forced to ~0 px for every frame; this gate has no discriminating power "
            "and must not be read as independent geometric agreement"
        ),
    }

    # --- disparity / depth distributions -------------------------------------
    mask_valid = (left_mask_local > 0) & (disparity > 1.0)
    mask_invalid = (left_mask_local > 0) & (disparity <= 1.0)
    record["disparity_distribution"] = {
        "left_mask_pixels_with_valid_disparity": int(mask_valid.sum()),
        "left_mask_pixels_without_valid_disparity": int(mask_invalid.sum()),
        "left_mask_valid_fraction": float(mask_valid.sum() / max(1, int((left_mask_local > 0).sum()))),
        "whole_local_image_valid_disparity_fraction": float((disparity > 1.0).mean()),
        "whole_local_image_valid_disparity_fraction_outside_mask": float(
            ((disparity > 1.0) & (left_mask_local == 0)).sum() / max(1, int((left_mask_local == 0).sum()))
        ),
        **numeric_stats(disparity[mask_valid], "left_mask_valid_disparity"),
        **numeric_stats(disparity[ys, left_x], "candidate_disparity"),
        **numeric_stats(measured[inlier_mask] if inlier_mask is not None else np.empty(0), "inlier_disparity"),
        **numeric_stats(candidate_depth, "candidate_depth_mm"),
        **numeric_stats(
            depth_scale / (disparity[mask_valid] if mask_valid.any() else np.empty(0)),
            "left_mask_valid_depth_mm",
        ),
    }
    if inlier_mask is not None and len(inlier_mask) and inlier_mask.any():
        inlier_depth = depth_scale / measured[inlier_mask]
        record["disparity_distribution"].update(numeric_stats(inlier_depth, "inlier_depth_mm"))

    # --- texture --------------------------------------------------------------
    record["texture"] = {
        "left_local_mask_region": texture_statistics(left_gray, left_mask_local > 0, "left_local_mask"),
        "left_local_whole_view": texture_statistics(left_gray, np.ones(shape, dtype=bool), "left_local_all"),
        "left_local_mask_region_outside_mask": texture_statistics(
            left_gray, left_mask_local == 0, "left_local_outside_mask"
        ),
        "right_local_mask_region": texture_statistics(right_gray, right_mask_local > 0, "right_local_mask"),
        "left_local_valid_disparity_region": texture_statistics(
            left_gray, disparity > 1.0, "left_local_valid_disparity"
        ),
        "left_local_mask_region_matched_by_frozen_matcher": texture_statistics(
            left_gray, mask_valid, "left_local_mask_valid_disparity"
        ),
        "note": "gradients use the unnormalised gray image; the matcher itself sees a CLAHE-preprocessed copy",
    }

    # --- attribution probes ---------------------------------------------------
    relaxed = dense_disparity_relaxed(left_local, right_local, num_disparities)
    relaxed_valid = (left_mask_local > 0) & (relaxed > 1.0) & (relaxed < num_disparities)
    record["attribution_probes"] = {
        "probe_kind": "attribution_probe_not_a_candidate_set",
        "relaxed_matcher": {
            "description": (
                "identical matcher with disp12MaxDiff=-1, uniquenessRatio=0, speckleWindowSize=0: isolates how many "
                "mask pixels have any local block match before the matcher's own filters"
            ),
            "left_mask_pixels_with_any_match": int(relaxed_valid.sum()),
            "left_mask_fraction_with_any_match": float(relaxed_valid.sum() / max(1, int((left_mask_local > 0).sum()))),
            "whole_local_image_fraction_with_any_match": float(((relaxed > 1.0) & (relaxed < num_disparities)).mean()),
            **numeric_stats(relaxed[relaxed_valid], "relaxed_disparity_in_mask"),
        },
        "mask_searchability": mask_searchability(left_mask_local, right_mask_local, num_disparities),
    }

    sample_pixels = np.nonzero(left_mask_local > 0)
    sample_count = len(sample_pixels[0])
    if sample_count:
        stride = max(1, sample_count // max(1, probe_samples))
        sample_ys = sample_pixels[0][::stride]
        sample_xs = sample_pixels[1][::stride]
        half = PROBE_WINDOW // 2
        inside = (
            (sample_ys >= half) & (sample_ys < height - half)
            & (sample_xs >= num_disparities + half) & (sample_xs < width - half)
        )
        left_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(left_gray)
        right_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(right_gray)
        record["attribution_probes"]["cost_curve"] = block_cost_curves(
            left_clahe, right_clahe, sample_ys[inside], sample_xs[inside], num_disparities, "uniform_mask_sample",
            minimum_disparity=PROBE_MIN_DISPARITY,
        )
        record["attribution_probes"]["cost_curve"]["probe_pixels_dropped_at_window_or_range_border"] = int(
            sample_count - int(inside.sum())
        )
        # Extended-interval probe: does the *frozen* range truncate the floor?
        # Evaluated over a wider interval than the frozen one, purely to measure
        # how many mask pixels prefer a disparity the frozen search cannot reach.
        extended = PROBE_EXTENDED_DISPARITIES
        extended_stride = max(1, sample_count // max(1, probe_samples // 2))
        extended_ys = sample_pixels[0][::extended_stride]
        extended_xs = sample_pixels[1][::extended_stride]
        extended_inside = (
            (extended_ys >= half) & (extended_ys < height - half)
            & (extended_xs >= extended + half) & (extended_xs < width - half)
        )
        extended_curves = block_cost_curves(
            left_clahe, right_clahe, extended_ys[extended_inside], extended_xs[extended_inside],
            extended, "extended_interval_probe", minimum_disparity=PROBE_MIN_DISPARITY,
        )
        if "probe_zsad_best_disparity_median" in extended_curves:
            extended_curves["probe_zsad_fraction_best_beyond_frozen_range"] = float(
                extended_curves.get("probe_zsad_best_disparity_p95", 0.0) >= num_disparities
            )
        record["attribution_probes"]["cost_curve_extended_interval"] = extended_curves
    else:
        record["attribution_probes"]["cost_curve"] = {"probe_sampled_pixels": 0}
        record["attribution_probes"]["cost_curve_extended_interval"] = {"probe_sampled_pixels": 0}

    # Cost curves evaluated at the pixels that actually survived the frozen chain.
    # These answer whether the survivors are evidence-supported matches or
    # boundary artefacts of the frozen search range.
    record["attribution_probes"]["cost_curve_at_candidates"] = sampled_cost_curve(
        left_gray, right_gray, left_pixels, num_disparities, probe_samples, "frozen_candidates", measured
    )
    record["attribution_probes"]["cost_curve_at_inliers"] = sampled_cost_curve(
        left_gray, right_gray, inlier_left_pixels, num_disparities, probe_samples, "frozen_inliers",
        measured[inlier_mask] if inlier_mask is not None else None,
    )
    record["attribution_probes"]["range_top_population"] = range_top_population(
        disparity, left_mask_local > 0, num_disparities
    )
    if len(measured):
        record["attribution_probes"]["candidate_disparity_boundary_population"] = {
            "candidate_pixels": int(len(measured)),
            "candidate_fraction_at_range_top": float(np.mean(measured >= num_disparities - 1)),
            "candidate_fraction_at_or_above_150": float(np.mean(measured >= 150.0)),
            "candidate_fraction_below_100": float(np.mean(measured < 100.0)),
        }

    # --- inlier geometry, conditioning, within-frame stability ----------------
    record["mask_component_breakdown"] = mask_component_breakdown(
        left_mask_local, mask_valid, left_pixels, inlier_left_pixels
    )
    record["inlier_geometry"] = inlier_geometry(inliers, inlier_left_pixels, left_mask_local, shape)
    record["inlier_geometry"].update(split_half_plane_stability(inliers, split_half_repeats, 20260909 + index))
    if fit is not None:
        record["inlier_geometry"]["tls_plane_on_all_inliers"] = {
            "normal_left_camera": [float(value) for value in fit.normal],
            "offset_mm": float(fit.offset),
            **disparity_residual_of_plane(
                inlier_left_pixels, measured[inlier_mask], fit.normal, fit.offset,
                rectification.rotation_left, rectification.virtual_K, translation_x,
            ),
        }
    else:
        record["inlier_geometry"]["tls_plane_on_all_inliers"] = None

    record["status"] = "measured"
    visual_context = {
        "left_local": left_local,
        "right_local": right_local,
        "left_mask_local": left_mask_local,
        "right_mask_local": right_mask_local,
        "disparity": disparity,
        "candidate_pixels": left_pixels,
        "inlier_pixels": inlier_left_pixels,
        "mask_valid": mask_valid,
        "left_gray": left_gray,
        "left_upright": left_upright,
        "right_upright": right_upright,
        "left_mask_upright": left_mask_upright,
        "right_mask_upright": right_mask_upright,
        "left_seed": left_seed,
        "right_seed": right_seed,
        "header": (
            f"{name}: {frozen_row['observation_state']}; candidates={len(points)} inliers={len(inliers)} "
            f"coverage={coverage:.5f} reasons={','.join(frozen_row['reason']) or 'none'}"
        ),
        "visualization_requested": want_visualization,
    }
    return record, visual_context


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    parameters = frozen_parameters(args.frozen_baseline_dir)
    rows = frozen_rows(args.frozen_baseline_dir)
    runtime_size = (int(parameters["runtime_width"]), int(parameters["runtime_height"]))

    pairs = strict.read_pairs(args.left_dir, args.right_dir, int(parameters["frame_step"]), int(parameters["max_frames"]))
    if not pairs:
        raise RuntimeError("no pairs selected")
    if len(pairs) != len(rows):
        raise RuntimeError(f"frozen baseline holds {len(rows)} rows but {len(pairs)} pairs were selected")

    source_calibration = StereoCalibration.load(args.calibration)
    calibration = source_calibration.for_runtime_sizes(runtime_size, runtime_size)
    args.output_dir.mkdir(parents=True)

    records: list[dict[str, Any]] = []
    for index, (name, _, _) in enumerate(pairs):
        pair_id = int(Path(name).stem.removeprefix("pair_"))
        if pair_id not in rows:
            raise RuntimeError(f"pair {name} is not present in the frozen baseline")
        record, visual = process_frame(
            name=name,
            index=index,
            left_dir=args.left_dir,
            right_dir=args.right_dir,
            semantic_left_dir=args.semantic_left_dir,
            semantic_right_dir=args.semantic_right_dir,
            calibration=calibration,
            runtime_size=runtime_size,
            parameters=parameters,
            frozen_row=rows[pair_id],
            probe_samples=args.probe_samples,
            split_half_repeats=args.split_half_repeats,
            want_visualization=not args.no_visualization,
        )
        records.append(record)
        verification = record.get("frozen_baseline_reproduction", {}).get("verification", {})
        if verification and not verification.get("matched", False):
            raise RuntimeError(
                f"frozen baseline reproduction failed for {name}: "
                f"{json.dumps(verification, ensure_ascii=False)}"
            )
        if visual:
            if not args.no_visualization:
                write_visualization(
                    args.output_dir / "visualizations" / f"{Path(name).stem}_correspondence.png",
                    left_local=visual["left_local"],
                    right_local=visual["right_local"],
                    left_mask_local=visual["left_mask_local"],
                    right_mask_local=visual["right_mask_local"],
                    disparity=visual["disparity"],
                    num_disparities=int(parameters["num_disparities"]),
                    candidate_pixels=visual["candidate_pixels"],
                    inlier_pixels=visual["inlier_pixels"],
                    header=visual["header"],
                )
                write_mask_overlay(
                    args.output_dir / "mask_overlays" / f"{Path(name).stem}_mask_overlay.png",
                    visual["left_upright"], visual["right_upright"],
                    visual["left_mask_upright"], visual["right_mask_upright"],
                    visual["left_seed"], visual["right_seed"],
                )
            if not args.no_ascii:
                raster = ascii_raster(
                    visual["inlier_pixels"], visual["candidate_pixels"],
                    visual["left_mask_local"], visual["disparity"] > 1.0,
                )
                texture = ascii_texture(visual["left_gray"])
                ascii_dir = args.output_dir / "ascii"
                ascii_dir.mkdir(parents=True, exist_ok=True)
                (ascii_dir / f"{Path(name).stem}_coverage.txt").write_text(
                    visual["header"] + "\n" + raster + "\n", encoding="utf-8"
                )
                (ascii_dir / f"{Path(name).stem}_texture.txt").write_text(
                    visual["header"] + "\n" + texture + "\n", encoding="utf-8"
                )
        print(json.dumps({
            "pair": pair_id,
            "state": record.get("frozen_baseline_observation_state"),
            "candidates": record.get("frozen_baseline_reproduction", {}).get("candidate_points"),
            "inliers": record.get("frozen_baseline_reproduction", {}).get("ransac_inliers"),
            "reproduced": verification.get("matched"),
        }, ensure_ascii=False))

    with (args.output_dir / "correspondence_diagnostic.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = build_summary(records, parameters)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "role": "read-only attribution diagnostic; emits no ground state and changes no rejection reason",
        "inputs": {
            "left_dir": str(args.left_dir),
            "right_dir": str(args.right_dir),
            "calibration": str(args.calibration),
            "semantic_left_dir": str(args.semantic_left_dir),
            "semantic_right_dir": str(args.semantic_right_dir),
            "frozen_baseline_dir": str(args.frozen_baseline_dir),
            "semantic_identity_evidence": "manually_audited (inherited from the frozen baseline run)",
        },
        "frozen_parameters_read_from_baseline": parameters,
        "cli_parameters": {
            "probe_samples": args.probe_samples,
            "split_half_repeats": args.split_half_repeats,
            "ascii": not args.no_ascii,
            "visualization": not args.no_visualization,
        },
        "interpretation_boundary": summary["interpretation_boundary"],
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_dir), **{k: v for k, v in summary.items() if k != "per_frame"}}, ensure_ascii=False))


def build_summary(records: list[dict[str, Any]], parameters: dict[str, Any]) -> dict[str, Any]:
    measured = [record for record in records if record.get("status") == "measured"]
    reason_counts: dict[str, int] = {}
    for record in records:
        for reason in record.get("frozen_baseline_reason", []):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    stage_totals: dict[str, int] = {}
    stage_losses: dict[str, int] = {}
    for record in measured:
        for entry in record["funnel"]["stages"]:
            stage_totals[entry["stage"]] = stage_totals.get(entry["stage"], 0) + entry["cumulative_pixels"]
            if entry["lost_at_this_stage"]:
                stage_losses[entry["stage"]] = stage_losses.get(entry["stage"], 0) + entry["lost_at_this_stage"]
    drop_one_totals: dict[str, int] = {}
    for record in measured:
        for key, value in record["funnel"]["drop_one_condition_candidate_counts"].items():
            drop_one_totals[key] = drop_one_totals.get(key, 0) + value

    def aggregate(getter, records_source=None) -> dict[str, Any]:
        values = []
        for record in (records_source if records_source is not None else measured):
            value = getter(record)
            if value is not None:
                values.append(float(value))
        if not values:
            return {"count": 0}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.size),
            "min": float(array.min()),
            "median": float(np.median(array)),
            "max": float(array.max()),
        }

    per_frame = []
    for record in records:
        if record.get("status") != "measured":
            per_frame.append({
                "pair_id": record["pair_id"],
                "state": record.get("frozen_baseline_observation_state"),
                "status": record.get("status"),
            })
            continue
        probes = record["attribution_probes"]
        per_frame.append({
            "pair_id": record["pair_id"],
            "state": record["frozen_baseline_observation_state"],
            "reasons": record["frozen_baseline_reason"],
            "candidate_points": record["frozen_baseline_reproduction"]["candidate_points"],
            "ransac_inliers": record["frozen_baseline_reproduction"]["ransac_inliers"],
            "coverage": record["frozen_baseline_reproduction"]["inlier_coverage_fraction"],
            "median_plane_residual_mm": record["frozen_baseline_reproduction"]["median_plane_residual_mm"],
            "left_mask_pixels_local": record["mask_geometry_local"]["left_mask_pixels_local"],
            "left_mask_survival_raw_to_local": record["mask_geometry_local"]["left_mask_survival_raw_to_local"],
            "left_local_view_usable_fraction": record["mask_geometry_local"]["left_local_view_usable_fraction"],
            "left_mask_component_count": record["mask_component_breakdown"].get("left_mask_component_count"),
            "left_mask_largest_component_share": record["mask_component_breakdown"].get("left_mask_largest_component_share"),
            "inliers_outside_largest_mask_component": record["mask_component_breakdown"].get("inliers_outside_largest_mask_component"),
            "left_mask_valid_disparity_fraction": record["disparity_distribution"]["left_mask_valid_fraction"],
            "whole_image_valid_disparity_fraction": record["disparity_distribution"]["whole_local_image_valid_disparity_fraction"],
            "left_mask_fraction_searchable_in_range": probes["mask_searchability"]["left_mask_fraction_searchable_in_range"],
            "relaxed_left_mask_fraction_with_any_match": probes["relaxed_matcher"]["left_mask_fraction_with_any_match"],
            "candidate_disparity_median": record["disparity_distribution"].get("candidate_disparity_median"),
            "inlier_disparity_median": record["disparity_distribution"].get("inlier_disparity_median"),
            "left_local_mask_gradient_median": record["texture"]["left_local_mask_region"].get("left_local_mask_gradient_median"),
            "left_local_all_gradient_median": record["texture"]["left_local_whole_view"].get("left_local_all_gradient_median"),
            "inlier_distinct_rows": record["inlier_geometry"].get("inlier_distinct_rows"),
            "inlier_pixel_linearity": record["inlier_geometry"].get("inlier_pixel_linearity"),
            "inlier_pca_sigma2_over_sigma1": record["inlier_geometry"].get("inlier_pca_sigma2_over_sigma1"),
            "inlier_camera_distance_mm_median": record["inlier_geometry"].get("inlier_camera_distance_mm_median"),
            "split_half_normal_angle_deg_median": record["inlier_geometry"].get("split_half_normal_angle_deg_median"),
            "inlier_fraction_within_3px_of_mask_boundary": record["inlier_geometry"].get("inlier_fraction_within_3px_of_mask_boundary"),
            "probe_best_disparity_median": probes["cost_curve"].get("probe_zsad_best_disparity_median"),
            "probe_sad_best_disparity_median": probes["cost_curve"].get("probe_sad_best_disparity_median"),
            "probe_fraction_contrast_strong": probes["cost_curve"].get("probe_zsad_fraction_contrast_strong"),
            "probe_fraction_contrast_weak": probes["cost_curve"].get("probe_zsad_fraction_contrast_weak"),
            "probe_fraction_best_at_interval_top": probes["cost_curve"].get("probe_zsad_fraction_best_at_interval_top"),
            "extended_probe_zsad_best_disparity_median": probes["cost_curve_extended_interval"].get("probe_zsad_best_disparity_median"),
            "extended_probe_zsad_best_disparity_p95": probes["cost_curve_extended_interval"].get("probe_zsad_best_disparity_p95"),
            "extended_probe_fraction_contrast_strong": probes["cost_curve_extended_interval"].get("probe_zsad_fraction_contrast_strong"),
            "gate_pass_rate_frozen_field": record["consistency_gate_controls"].get("observed_pass_rate_frozen_reverse_field_at_partner"),
            "gate_null_partner_shift_rate_frozen_field": record["consistency_gate_controls"].get("null_pass_rate_frozen_reverse_field_partner_shifted_17px"),
            "gate_null_other_row_rate_frozen_field": record["consistency_gate_controls"].get("null_pass_rate_frozen_reverse_field_other_row_5px"),
            "gate_chance_rate": record["consistency_gate_controls"].get("chance_rate_for_independent_quantised_fields"),
            "gate_pass_rate_direction_corrected_field": record["consistency_gate_controls"].get("observed_pass_rate_direction_corrected_field_at_partner"),
            "gate_null_partner_shift_rate_direction_corrected_field": record["consistency_gate_controls"].get("null_pass_rate_direction_corrected_field_partner_shifted_17px"),
            "direction_corrected_candidate_pixels_probe": record["consistency_gate_controls"].get("direction_corrected_candidate_pixels_probe"),
            "mask_fraction_at_range_top": probes["range_top_population"].get("mask_fraction_at_range_top"),
            "mask_range_top_largest_blob_px": probes["range_top_population"].get("mask_range_top_largest_blob_px"),
            "candidate_fraction_at_range_top": probes.get("candidate_disparity_boundary_population", {}).get("candidate_fraction_at_range_top"),
            "candidate_fraction_at_or_above_150": probes.get("candidate_disparity_boundary_population", {}).get("candidate_fraction_at_or_above_150"),
            "candidate_zsad_best_disparity_median": probes["cost_curve_at_candidates"].get("probe_zsad_best_disparity_median"),
            "candidate_measured_minus_zsad_best_median": probes["cost_curve_at_candidates"].get("probe_frozen_candidates_measured_minus_zsad_best_median"),
            "candidate_probe_fraction_contrast_weak": probes["cost_curve_at_candidates"].get("probe_zsad_fraction_contrast_weak"),
            "consistency_mismatch_px_median": record["consistency_mismatch"].get("consistency_mismatch_px_median"),
            "consistency_mismatch_px_p95": record["consistency_mismatch"].get("consistency_mismatch_px_p95"),
            "consistency_pixels_reaching_test": record["consistency_mismatch"]["pixels_reaching_consistency_test"],
            "consistency_fraction_within_1_5px": record["consistency_mismatch"].get("fraction_within_1_5px"),
            "reproduced_frozen_baseline": record["frozen_baseline_reproduction"]["verification"]["matched"],
            "reproduced_frozen_baseline_exact": record["frozen_baseline_reproduction"]["verification"]["exact"],
            "reason_matches_frozen_baseline": record["frozen_gate_attribution"]["reason_matches_frozen_baseline"],
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "frame_count": len(records),
        "measured_frame_count": len(measured),
        "frozen_baseline_direct_count": sum(
            record.get("frozen_baseline_observation_state") == "direct" for record in records
        ),
        "frozen_baseline_reason_counts": reason_counts,
        "funnel_stage_totals_across_frames": stage_totals,
        "funnel_stage_losses_across_frames": stage_losses,
        "drop_one_condition_candidate_totals": drop_one_totals,
        "aggregates": {
            "left_mask_pixels_local": aggregate(lambda r: r["mask_geometry_local"]["left_mask_pixels_local"]),
            "left_mask_survival_raw_to_local": aggregate(lambda r: r["mask_geometry_local"]["left_mask_survival_raw_to_local"]),
            "left_mask_valid_disparity_fraction": aggregate(lambda r: r["disparity_distribution"]["left_mask_valid_fraction"]),
            "left_local_view_usable_fraction": aggregate(lambda r: r["mask_geometry_local"]["left_local_view_usable_fraction"]),
            "left_mask_component_count": aggregate(lambda r: r["mask_component_breakdown"].get("left_mask_component_count")),
            "left_mask_largest_component_share": aggregate(lambda r: r["mask_component_breakdown"].get("left_mask_largest_component_share")),
            "whole_image_valid_disparity_fraction": aggregate(lambda r: r["disparity_distribution"]["whole_local_image_valid_disparity_fraction"]),
            "left_mask_fraction_searchable_in_range": aggregate(lambda r: r["attribution_probes"]["mask_searchability"]["left_mask_fraction_searchable_in_range"]),
            "relaxed_left_mask_fraction_with_any_match": aggregate(lambda r: r["attribution_probes"]["relaxed_matcher"]["left_mask_fraction_with_any_match"]),
            "probe_fraction_contrast_strong": aggregate(lambda r: r["attribution_probes"]["cost_curve"].get("probe_zsad_fraction_contrast_strong")),
            "probe_fraction_contrast_weak": aggregate(lambda r: r["attribution_probes"]["cost_curve"].get("probe_zsad_fraction_contrast_weak")),
            "probe_fraction_best_at_interval_top": aggregate(lambda r: r["attribution_probes"]["cost_curve"].get("probe_zsad_fraction_best_at_interval_top")),
            "probe_best_disparity_median": aggregate(lambda r: r["attribution_probes"]["cost_curve"].get("probe_zsad_best_disparity_median")),
            "probe_sad_best_disparity_median": aggregate(lambda r: r["attribution_probes"]["cost_curve"].get("probe_sad_best_disparity_median")),
            "probe_sad_minus_zsad_best_disparity_median": aggregate(lambda r: r["attribution_probes"]["cost_curve"].get("probe_sad_minus_zsad_best_disparity_median")),
            "extended_probe_zsad_best_disparity_median": aggregate(lambda r: r["attribution_probes"]["cost_curve_extended_interval"].get("probe_zsad_best_disparity_median")),
            "extended_probe_zsad_best_disparity_p95": aggregate(lambda r: r["attribution_probes"]["cost_curve_extended_interval"].get("probe_zsad_best_disparity_p95")),
            "extended_probe_fraction_contrast_strong": aggregate(lambda r: r["attribution_probes"]["cost_curve_extended_interval"].get("probe_zsad_fraction_contrast_strong")),
            "gate_pass_rate_frozen_field": aggregate(lambda r: r["consistency_gate_controls"].get("observed_pass_rate_frozen_reverse_field_at_partner")),
            "gate_null_partner_shift_rate_frozen_field": aggregate(lambda r: r["consistency_gate_controls"].get("null_pass_rate_frozen_reverse_field_partner_shifted_17px")),
            "gate_null_other_row_rate_frozen_field": aggregate(lambda r: r["consistency_gate_controls"].get("null_pass_rate_frozen_reverse_field_other_row_5px")),
            "gate_chance_rate": aggregate(lambda r: r["consistency_gate_controls"].get("chance_rate_for_independent_quantised_fields")),
            "gate_pass_rate_direction_corrected_field": aggregate(lambda r: r["consistency_gate_controls"].get("observed_pass_rate_direction_corrected_field_at_partner")),
            "gate_null_partner_shift_rate_direction_corrected_field": aggregate(lambda r: r["consistency_gate_controls"].get("null_pass_rate_direction_corrected_field_partner_shifted_17px")),
            "direction_corrected_candidate_pixels_probe": aggregate(lambda r: r["consistency_gate_controls"].get("direction_corrected_candidate_pixels_probe")),
            "mask_fraction_at_range_top": aggregate(lambda r: r["attribution_probes"]["range_top_population"].get("mask_fraction_at_range_top")),
            "candidate_fraction_at_range_top": aggregate(lambda r: r["attribution_probes"].get("candidate_disparity_boundary_population", {}).get("candidate_fraction_at_range_top")),
            "candidate_fraction_at_or_above_150": aggregate(lambda r: r["attribution_probes"].get("candidate_disparity_boundary_population", {}).get("candidate_fraction_at_or_above_150")),
            "candidate_zsad_best_disparity_median": aggregate(lambda r: r["attribution_probes"]["cost_curve_at_candidates"].get("probe_zsad_best_disparity_median")),
            "candidate_measured_minus_zsad_best_median": aggregate(lambda r: r["attribution_probes"]["cost_curve_at_candidates"].get("probe_frozen_candidates_measured_minus_zsad_best_median")),
            "candidate_probe_fraction_contrast_weak": aggregate(lambda r: r["attribution_probes"]["cost_curve_at_candidates"].get("probe_zsad_fraction_contrast_weak")),
            "left_local_mask_gradient_median": aggregate(lambda r: r["texture"]["left_local_mask_region"].get("left_local_mask_gradient_median")),
            "left_local_whole_view_gradient_median": aggregate(lambda r: r["texture"]["left_local_whole_view"].get("left_local_all_gradient_median")),
            "candidate_disparity_median": aggregate(lambda r: r["disparity_distribution"].get("candidate_disparity_median")),
            "inlier_pixel_linearity": aggregate(lambda r: r["inlier_geometry"].get("inlier_pixel_linearity")),
            "inlier_pca_sigma2_over_sigma1": aggregate(lambda r: r["inlier_geometry"].get("inlier_pca_sigma2_over_sigma1")),
            "split_half_normal_angle_deg_median": aggregate(lambda r: r["inlier_geometry"].get("split_half_normal_angle_deg_median")),
            "inlier_fraction_within_3px_of_mask_boundary": aggregate(lambda r: r["inlier_geometry"].get("inlier_fraction_within_3px_of_mask_boundary")),
            "consistency_mismatch_px_median": aggregate(lambda r: r["consistency_mismatch"].get("consistency_mismatch_px_median")),
            "consistency_mismatch_px_p95": aggregate(lambda r: r["consistency_mismatch"].get("consistency_mismatch_px_p95")),
            "consistency_fraction_within_1_5px": aggregate(lambda r: r["consistency_mismatch"].get("fraction_within_1_5px")),
            "consistency_fraction_within_3px": aggregate(lambda r: r["consistency_mismatch"].get("fraction_within_3px")),
        },
        "per_frame": per_frame,
        "interpretation_boundary": (
            "Read-only attribution diagnostics of one frozen, internally inconsistent local-ground pipeline on 12 "
            "manually audited floor masks. These numbers are not real ground truth, not ground-plane accuracy, not "
            "foot height, contact, support or gait, and the cross-frame plane parameters are NOT comparable because no "
            "accepted relative pose exists to bring frames into one reference frame."
        ),
    }


if __name__ == "__main__":
    main()
