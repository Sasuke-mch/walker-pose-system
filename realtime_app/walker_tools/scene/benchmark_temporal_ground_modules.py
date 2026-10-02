#!/usr/bin/env python3
"""Temporal ground modules (optical flow, static-background VO) plus the fixed
27-combination composition matrix and the final recommendation.

The task-01 (``G20260912_modular_ground_benchmark_v1``) and task-02
(``G20260912_learned_stereo_replacement_benchmark_v1``) directories are the only
sources of spatial reconstruction: this tool **reads them and never writes into
them**, never re-runs SGBM / IGEV / DynamicStereo, and never trains or
fine-tunes a model.  It adds exactly two new measurements and one aggregation:

1. an **optical-flow** module that only ever produces *additional* temporal
   semantic evidence.  The current frame's Mask2Former candidate mask stays the
   authoritative semantic input; a propagated mask never replaces it and never
   bypasses the current frame's stereo, triangulation, RANSAC or region gate.
   The resulting temporal mode is only ``flow_assisted_current_frame``;
2. a **static-background visual-odometry** module that may propagate only an
   already accepted direct plane through relative poses estimated from static
   background features excluding the Mask2Former floor candidate, the PMPose
   person region and the unverified walker candidate region.  It fails closed:
   with no certified static-background chain the mode stays
   ``vo_propagated_plane = unavailable`` and no ground / person / walker point is
   ever used to make up the difference;
3. the fixed **3 matchers x 3 plane methods x 3 temporal modes = 27**
   combination rows with layered evidence, module timing, a clearly labelled
   *estimated* (never measured) end-to-end latency, and the final A-G
   recommendation.

Strict fact boundaries carried unchanged from the upstream records:

* Mask2Former is an **online ground semantic candidate**, not ground truth;
* the manual floor masks provide **2-D identity audit only**, not 3-D physical
  ground truth;
* low residual / high inlier fraction / reprojection consistency of SGBM, IGEV
  and DynamicStereo are **internal geometric evidence only**;
* DynamicStereo passes the frozen shared region gate on **0/12** frames, is a
  five-frame forward offline window that needs four future frames, must never be
  a VO ``direct`` anchor and must never be reported as real time;
* no independent physical ground truth exists in this data, so this tool reports
  no true ground accuracy, no camera-height accuracy, no foot-to-ground height,
  no gait event, no contact and no clinical quantity.
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
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np


TOOLS_ROOT = _tool_legacy_tools
PROJECT_ROOT = TOOLS_ROOT.parents[1]
REALTIME_ROOT = PROJECT_ROOT / "realtime_app"

from walker_tools.scene.audit_manual_floor_labels import binary_metrics, load_labelme, rasterize_labelme, render_error_overlay
from pose_app.benchmark_timing import TimingCollector, percentile_ms  # noqa: E402
from pose_app.local_plane_propagation import (  # noqa: E402
    LocalPlane,
    RelativePose,
    STATIC_BACKGROUND_DOMAIN,
    compose_relative_poses,
    plane_agreement,
    propagate_plane,
)
from pose_app.temporal_floor_mask_propagation import (  # noqa: E402
    PropagationThresholds,
    dense_flow,
    flow_consistency,
    grayscale_u8,
    propagate_adjacent,
    warp_from_previous,
)


SCHEMA_VERSION = "temporal_ground_composition_benchmark_v1"

MATCHERS = ("sgbm", "igev", "dynamicstereo")
PLANE_METHODS = ("dense_ransac", "sparse_tile_ransac", "soft_weighted_irls")
TEMPORAL_MODES = ("direct_current_frame", "flow_assisted_current_frame", "vo_propagated_plane")

UPSTREAM_SEMANTIC_SOURCE = "mask2former_floor"
ONLINE_SEMANTIC_MODEL = "mask2former_swin_small"
VIEWS = ("left", "right")

ANCHOR_PAIR_IDS = (0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440)

REQUIRED_TIMING_FIELDS = (
    "count", "median_ms", "p90_ms", "p95_ms", "min_ms", "max_ms", "warmup_excluded_count",
)

#: task-01 / task-02 excluded the first two pairs from every percentile, so a
#: module compared across the two tasks must declare the same exclusion.
UPSTREAM_WARMUP_EXCLUDED = 2

#: The source capture's effective paired frame rate.  Every real-time statement
#: in this run is measured against this declared budget on purpose, so that a
#: fast GPU forward cannot be turned into a real-time claim.
SOURCE_EFFECTIVE_PAIR_FPS = 27.818846453775496
DECLARED_FRAME_BUDGET_MS = 1000.0 / SOURCE_EFFECTIVE_PAIR_FPS

FLOW_ASSISTED_SEMANTICS = (
    "flow_assisted_current_frame only adds evidence about whether the previous frame's Mask2Former candidate "
    "agrees with the current frame's under optical flow. The current frame keeps its own Mask2Former mask and its "
    "own stereo geometry; the propagated mask never replaces the current mask and never bypasses the current "
    "frame's stereo, triangulation, RANSAC or region gate. An inconsistent flow evidence degrades the state "
    "instead of creating ground."
)

VO_CONFIDENCE_BOUNDARY = (
    "Plane propagation agreement against a direct pipeline observation; not physical ground accuracy."
)

VO_STATIC_DOMAIN_NOTE = (
    "Static-background VO features may only come from regions outside the Mask2Former floor candidate, the PMPose "
    "person region and the unverified walker candidate region. No ground, person or walker point may be used to "
    "make up a missing pose, and a rejected or missing adjacent pose fails the chain closed."
)

DIRECT_DEFINITION = (
    "In this task a VO-eligible 'direct' anchor means exactly: the task-01/task-02 Mask2Former-candidate plane of "
    "that frame passed the shared cross-region gate (cross_region_evidence.status == 'pass'). It is NOT the "
    "stricter project 'direct' state that additionally requires manually audited mask identity, and it is not "
    "physical ground truth."
)

MATRIX_BOUNDARY = (
    "Every number here is a module time or an internal evidence statistic under shared images, shared calibration "
    "and a shared Mask2Former candidate semantic input. No independent physical ground truth exists in this data, "
    "so nothing here is true ground accuracy, camera-height accuracy, foot-to-ground height, a gait event, "
    "contact or a clinical quantity."
)

LATENCY_SUM_CAVEAT = (
    "estimated_end_to_end_latency_ms is a sum of independently measured per-module statistics and is labelled "
    "estimated everywhere. The P95 variant sums per-module P95 values, which is not the P95 of the sum. No "
    "end-to-end wall clock was measured."
)


# --------------------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------------------


def pair_id_from_name(name: str) -> int:
    stem = Path(name).stem
    suffix = stem.removeprefix("pair_")
    if suffix == stem or not suffix.isdigit():
        raise ValueError(f"frame name {name} is not of the form pair_<digits>.png")
    return int(suffix)


def frame_name(pair_id: int) -> str:
    return f"pair_{pair_id:04d}.png"


def frame_stem(pair_id: int) -> str:
    return f"pair_{pair_id:04d}"


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def timing_statistics(samples: Iterable[float | None], *, warmup_excluded_count: int = 0) -> dict[str, Any]:
    """The required timing shape, always with every key present."""
    values = [float(value) for value in samples if value is not None and math.isfinite(float(value))]
    result: dict[str, Any] = {field: None for field in REQUIRED_TIMING_FIELDS}
    result["count"] = int(len(values))
    result["warmup_excluded_count"] = int(warmup_excluded_count)
    if not values:
        return result
    result["mean_ms"] = float(sum(values) / len(values))
    result["median_ms"] = float(percentile_ms(values, 50.0))
    result["p90_ms"] = float(percentile_ms(values, 90.0))
    result["p95_ms"] = float(percentile_ms(values, 95.0))
    result["min_ms"] = float(min(values))
    result["max_ms"] = float(max(values))
    return result


def median_or_none(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not present else float(np.median(present))


def distribution(values: Iterable[float | None]) -> dict[str, Any]:
    present = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not present:
        return {"count": 0, "median": None, "p90": None, "p95": None, "min": None, "max": None}
    array = np.asarray(present, dtype=np.float64)
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def linear_trend(values: Sequence[float | None]) -> float | None:
    """Least-squares slope of ``values`` against their step index (drift direction)."""
    present = [(index, float(value)) for index, value in enumerate(values) if value is not None]
    if len(present) < 2:
        return None
    x = np.asarray([item[0] for item in present], dtype=np.float64)
    y = np.asarray([item[1] for item in present], dtype=np.float64)
    if float(np.ptp(x)) == 0.0:
        return None
    return float(np.polyfit(x, y, 1)[0])


def mean_or_none(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return None if not present else float(np.mean(present))


def snapshot_tree(root: Path) -> dict[str, list[int]]:
    """Read-only (size, mtime_ns) snapshot used to prove an upstream dir is untouched."""
    snapshot: dict[str, list[int]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            snapshot[str(path.relative_to(root))] = [int(stat.st_size), int(stat.st_mtime_ns)]
    return snapshot


def snapshot_difference(before: dict[str, list[int]], after: dict[str, list[int]]) -> dict[str, Any]:
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(
        name for name in set(before) & set(after)
        if before[name][0] != after[name][0] or before[name][1] != after[name][1]
    )
    return {
        "files_before": len(before),
        "files_after": len(after),
        "added": added,
        "removed": removed,
        "changed": changed,
        "unchanged": not added and not removed and not changed,
    }


class RunLog:
    """Collects stdout so the exact same text can be stored as run_stdout.txt."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, message: str = "") -> None:
        self.lines.append(str(message))
        print(message, flush=True)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


# --------------------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--left-dir", type=Path, default=PROJECT_ROOT / "research_records/engineering_validation/V20260908_people0_1_2_pmpose_c3_chain/people_1/input_448pairs/left_ccw90")
    parser.add_argument("--right-dir", type=Path, default=PROJECT_ROOT / "research_records/engineering_validation/V20260908_people0_1_2_pmpose_c3_chain/people_1/input_448pairs/right_cw90")
    parser.add_argument("--left-floor-mask-dir", type=Path, default=PROJECT_ROOT / "research_records/engineering_validation/G20260911_floor_semantic_backend_comparison_v1/left_people1_stride40_12frames_stereo_interface/mask2former_swin_small/floor_masks")
    parser.add_argument("--right-floor-mask-dir", type=Path, default=PROJECT_ROOT / "research_records/engineering_validation/G20260911_floor_semantic_backend_comparison_v1/right_people1_stride40_12frames_stereo_interface/mask2former_swin_small/floor_masks")
    parser.add_argument("--manual-left-label-dir", type=Path, default=PROJECT_ROOT / "research_records/engineering_validation/G20260911_floor_semantic_backend_comparison_v1/manual_labels_holdout_v1/labels/left")
    parser.add_argument("--manual-right-label-dir", type=Path, default=PROJECT_ROOT / "research_records/engineering_validation/G20260911_floor_semantic_backend_comparison_v1/manual_labels_holdout_v1/labels/right")
    parser.add_argument("--task01-dir", type=Path, default=PROJECT_ROOT / "research_records/engineering_validation/G20260912_modular_ground_benchmark_v1")
    parser.add_argument("--task02-dir", type=Path, default=PROJECT_ROOT / "research_records/engineering_validation/G20260912_learned_stereo_replacement_benchmark_v1")
    parser.add_argument("--person-region-jsonl", type=Path, default=PROJECT_ROOT / "research_records/engineering_validation/V20260908_people0_1_2_pmpose_c3_chain/people_1/pmpose_strict_stereo/offline_stereo_results.jsonl", help="saved PMPose 2-D results, used only to certify the VO exclusion domain")
    parser.add_argument("--walker-candidate-mask-dir", type=Path, default=PROJECT_ROOT / "research_records/engineering_validation/G20260910_walker_self_occlusion_semantic_exclusion_v2/people_1_first12_stride5_semantic_person_dark70/masks", help="unverified walker candidate masks; present for a few frames only")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "research_records/engineering_validation/G20260913_temporal_ground_composition_benchmark_v1")
    parser.add_argument("--maximum-flow-intervals", type=int, default=0, help="0 = all adjacent anchor intervals")
    parser.add_argument("--maximum-anchor-intervals", type=int, default=0, help="0 = all manual anchor intervals")
    parser.add_argument("--maximum-flow-steps", type=int, default=0, help="0 = full anchor gap; debugging aid only")
    parser.add_argument("--maximum-forward-backward-error-px", type=float, default=1.5)
    parser.add_argument("--minimum-candidate-flow-consistency", type=float, default=0.70)
    parser.add_argument("--maximum-candidate-photometric-median", type=float, default=35.0)
    parser.add_argument("--minimum-candidate-pixels", type=int, default=200)
    parser.add_argument("--repository-test-result", default="not_recorded_in_this_process")
    parser.add_argument("--py-compile-result", default="not_recorded_in_this_process")
    parser.add_argument("--no-visualization", action="store_true")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------------------
# upstream (task-01 / task-02) readers -- read only, never written
# --------------------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


class Upstream:
    """Read-only view over the two archived spatial reconstruction benchmarks."""

    def __init__(self, task01_dir: Path, task02_dir: Path) -> None:
        self.task01_dir = task01_dir
        self.task02_dir = task02_dir
        self.task01_summary = json.loads((task01_dir / "summary.json").read_text(encoding="utf-8"))
        self.task02_summary = json.loads((task02_dir / "summary.json").read_text(encoding="utf-8"))
        self.task01_records = read_jsonl(task01_dir / "frame_records.jsonl")
        self.task02_records = read_jsonl(task02_dir / "frame_records.jsonl")
        self.combinations: dict[tuple[str, str, str], dict[str, Any]] = {}
        for combination in self.task02_summary["combinations"]:
            key = (str(combination["semantic_source"]), str(combination["matcher"]), str(combination["plane_method"]))
            self.combinations[key] = combination
        # (frame_id, plane_method) -> per-frame cross-region status, online source only
        self.task01_cross_region: dict[tuple[str, str], str] = {}
        for record in self.task01_records:
            sources = {item["source_name"]: item for item in record["floor_sources"]}
            source = sources.get(UPSTREAM_SEMANTIC_SOURCE)
            if source is None:
                continue
            for method, entry in source["methods"].items():
                evidence = entry.get("evidence", {}).get("cross_region_evidence", {})
                if evidence:
                    self.task01_cross_region[(str(record["frame_id"]), str(method))] = str(evidence.get("status"))
        # (frame_id, matcher) -> per-frame cross-region status, learned matchers
        self.task02_cross_region: dict[tuple[str, str], str] = {}
        for record in self.task02_records:
            if str(record.get("semantic_source")) != UPSTREAM_SEMANTIC_SOURCE:
                continue
            evidence = record.get("cross_region_evidence", {})
            if evidence:
                self.task02_cross_region[(str(record["frame_id"]), str(record["matcher"]))] = str(evidence.get("status"))

    def combination(self, matcher: str, plane_method: str) -> dict[str, Any]:
        key = (UPSTREAM_SEMANTIC_SOURCE, matcher, plane_method)
        if key not in self.combinations:
            raise RuntimeError(f"upstream task-02 has no combination {key}")
        return self.combinations[key]

    def cross_region_status(self, matcher: str, pair_id: int, plane_method: str) -> str | None:
        name = frame_name(pair_id)
        if matcher == "sgbm":
            return self.task01_cross_region.get((name, plane_method))
        return self.task02_cross_region.get((name, matcher))

    def comparable_timing(self, matcher: str, plane_method: str) -> dict[str, Any]:
        """Module timings on the frame set both tasks timed (count = 10)."""
        combination = self.combination(matcher, plane_method)
        timing = combination.get("timing_on_task01_comparable_frames")
        if timing is None:
            raise RuntimeError(
                f"upstream combination {matcher}/{plane_method} has no task-01 comparable timing block; "
                "refusing to rank percentiles measured on different frame counts"
            )
        return timing

    def direct_plane(self, matcher: str, plane_method: str, pair_id: int) -> LocalPlane | None:
        """One frame's Mask2Former-candidate plane from the archived task outputs."""
        name = frame_name(pair_id)
        if matcher == "sgbm":
            for record in self.task01_records:
                if str(record["frame_id"]) != name:
                    continue
                sources = {item["source_name"]: item for item in record["floor_sources"]}
                source = sources.get(UPSTREAM_SEMANTIC_SOURCE)
                if source is None:
                    return None
                entry = source["methods"].get(plane_method)
                if entry is None or entry.get("plane") is None:
                    return None
                plane = entry["plane"]
                return LocalPlane(plane["normal_left_camera"], float(plane["offset_mm"]))
            return None
        for record in self.task02_records:
            if str(record.get("matcher")) != matcher or str(record.get("semantic_source")) != UPSTREAM_SEMANTIC_SOURCE:
                continue
            if str(record.get("frame_id")) != name or str(record.get("plane_method")) != plane_method:
                continue
            evidence = record.get("plane_evidence") or {}
            if evidence.get("normal_left_camera") is None or evidence.get("offset_mm") is None:
                return None
            return LocalPlane(evidence["normal_left_camera"], float(evidence["offset_mm"]))
        return None


# --------------------------------------------------------------------------------------
# optical flow module
# --------------------------------------------------------------------------------------


def flow_fields(previous_bgr: np.ndarray, current_bgr: np.ndarray, timings: dict[str, float]) -> dict[str, Any]:
    """The adjacent bidirectional flow field, timed on its own (one per frame pair)."""
    previous_gray, current_gray = grayscale_u8(previous_bgr), grayscale_u8(current_bgr)
    started = time.perf_counter()
    forward, backward = dense_flow(previous_gray, current_gray)
    residual, consistency_domain = flow_consistency(forward, backward)
    timings["flow_field_ms"] = (time.perf_counter() - started) * 1000.0
    return {
        "previous_gray": previous_gray,
        "current_gray": current_gray,
        "forward": forward,
        "backward": backward,
        "residual": residual,
        "consistency_domain": consistency_domain,
    }


def apply_flow_to_mask(
    fields: dict[str, Any], source_mask: np.ndarray, thresholds: PropagationThresholds, timings: dict[str, float]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Transport one mask through an already computed flow field.

    Identical decisions, metrics and reasons to
    :func:`pose_app.temporal_floor_mask_propagation.propagate_adjacent`; only the
    flow-field computation is factored out so several seeds can share one field
    and so the flow cost and the judgment cost can be reported separately.
    Guarded by a unit test that compares the two implementations directly.
    """
    thresholds.validate()
    source_mask = np.asarray(source_mask, dtype=bool)
    if source_mask.shape != fields["previous_gray"].shape:
        raise ValueError("source_mask must match the previous frame shape")
    started = time.perf_counter()
    backward = fields["backward"]
    current_gray = fields["current_gray"]
    warped_mask_u8, in_bounds = warp_from_previous(source_mask.astype(np.uint8), backward, cv2.INTER_NEAREST)
    propagated_mask = warped_mask_u8.astype(bool) & in_bounds
    warped_previous_gray, _ = warp_from_previous(fields["previous_gray"], backward, cv2.INTER_LINEAR)
    candidate_domain = propagated_mask & fields["consistency_domain"]
    candidate_pixels = int(propagated_mask.sum())
    consistency_fraction = (
        float((fields["residual"][candidate_domain] <= thresholds.maximum_forward_backward_error_px).mean())
        if candidate_domain.any() else 0.0
    )
    photometric_median = (
        float(np.median(np.abs(
            current_gray[candidate_domain].astype(np.int16) - warped_previous_gray[candidate_domain].astype(np.int16)
        ))) if candidate_domain.any() else None
    )
    residual_median = float(np.median(fields["residual"][propagated_mask])) if propagated_mask.any() else None
    timings["consistency_ms"] = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    reasons: list[str] = []
    if candidate_pixels < thresholds.minimum_candidate_pixels:
        reasons.append("too_few_propagated_floor_pixels")
    if consistency_fraction < thresholds.minimum_candidate_flow_consistency:
        reasons.append("forward_backward_flow_inconsistent")
    if photometric_median is None or photometric_median > thresholds.maximum_candidate_photometric_median:
        reasons.append("candidate_photometric_change_too_large")
    status = "candidate" if not reasons else "unavailable"
    timings["decision_ms"] = (time.perf_counter() - started) * 1000.0
    timings["step_total_ms"] = (
        timings.get("flow_field_ms", 0.0) + timings["consistency_ms"] + timings["decision_ms"]
    )
    return propagated_mask, {
        "status": status,
        "reasons": reasons,
        "propagated_floor_pixel_count": candidate_pixels,
        "candidate_flow_consistency_fraction": consistency_fraction,
        "candidate_photometric_median_abs_gray": photometric_median,
        "forward_backward_residual_median_px": residual_median,
        "flow_method": "Farneback adjacent bidirectional optical flow",
        "maximum_forward_backward_error_px": thresholds.maximum_forward_backward_error_px,
        "minimum_candidate_flow_consistency": thresholds.minimum_candidate_flow_consistency,
        "maximum_candidate_photometric_median": thresholds.maximum_candidate_photometric_median,
        "minimum_candidate_pixels": thresholds.minimum_candidate_pixels,
        "interpretation": (
            "image-space propagated candidate only; not audited ground, stereo evidence, geometry, contact, or gait"
        ),
    }


def propagate_adjacent_timed(
    previous_bgr: np.ndarray,
    current_bgr: np.ndarray,
    source_mask: np.ndarray,
    thresholds: PropagationThresholds,
    timings: dict[str, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """One adjacent step: the shared flow field, then the frozen acceptance rule."""
    fields = flow_fields(previous_bgr, current_bgr, timings)
    return apply_flow_to_mask(fields, source_mask, thresholds, timings)


def propagate_chains_locked(
    *,
    seed_masks: dict[str, np.ndarray],
    load_image: Callable[[int], np.ndarray],
    first_pair_id: int,
    step_count: int,
    thresholds: PropagationThresholds,
    phase_log: list[str] | None = None,
    view: str = "left",
) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, Any]]], list[dict[str, float]]]:
    """Transport every seed through the *same* flow fields, one field per step.

    Sharing the field keeps the optical-flow cost proportional to the number of
    frame pairs instead of the number of seeds, and guarantees that all seeds
    see literally the same motion estimate.  Only the previous and the current
    image are held in memory, so a 40-step chain never needs 41 frames resident.
    """
    if step_count < 1:
        raise ValueError("a propagation chain needs at least one step")
    masks = {name: np.asarray(mask, dtype=bool) for name, mask in seed_masks.items()}
    steps: dict[str, list[dict[str, Any]]] = {name: [] for name in masks}
    flow_steps: list[dict[str, float]] = []
    previous = load_image(first_pair_id)
    for index in range(step_count):
        current = load_image(first_pair_id + index + 1)
        timings: dict[str, float] = {}
        fields = flow_fields(previous, current, timings)
        for name in masks:
            seed_timings = dict(timings)
            seed_timings.pop("flow_field_ms", None)
            masks[name], record = apply_flow_to_mask(fields, masks[name], thresholds, seed_timings)
            steps[name].append({
                "step_index": index + 1,
                "source_pair_id": first_pair_id + index,
                "target_pair_id": first_pair_id + index + 1,
                **record,
                "flow_field_ms": float(timings["flow_field_ms"]),
                "consistency_ms": float(seed_timings["consistency_ms"]),
                "decision_ms": float(seed_timings["decision_ms"]),
                "step_total_ms": float(timings["flow_field_ms"] + seed_timings["consistency_ms"] + seed_timings["decision_ms"]),
            })
        flow_steps.append({"flow_field_ms": float(timings["flow_field_ms"]), "step_index": index + 1})
        previous = current
    if phase_log is not None:
        phase_log.append(f"propagate_chain:{view}:{step_count}_steps")
    return masks, steps, flow_steps


def mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def interval_flow_record(
    *,
    view: str,
    previous_pair_id: int,
    current_pair_id: int,
    previous_image: np.ndarray,
    current_image: np.ndarray,
    previous_mask: np.ndarray,
    current_mask: np.ndarray,
    thresholds: PropagationThresholds,
) -> tuple[dict[str, Any], np.ndarray]:
    """One ``previous -> current`` flow comparison for one view of one frame pair."""
    shape = current_image.shape[:2]
    empty = np.zeros(shape, dtype=bool)
    base: dict[str, Any] = {
        "pair_id": current_pair_id,
        "previous_pair_id": previous_pair_id,
        "current_pair_id": current_pair_id,
        "frame_id": frame_name(current_pair_id),
        "previous_frame_id": frame_name(previous_pair_id),
        "view": view,
        "flow_status": "unavailable",
        "forward_backward_consistency_fraction": None,
        "propagated_mask_area_fraction": None,
        "current_mask_area_fraction": None,
        "propagated_vs_current_iou": None,
        "new_area_fraction": None,
        "lost_area_fraction": None,
        "flow_runtime_ms": None,
        "reasons": [],
        "current_mask_is_authoritative": True,
        "propagated_mask_used_as_geometry_input": False,
        "temporal_mode_meaning": FLOW_ASSISTED_SEMANTICS,
        "definition": {
            "propagated_vs_current_iou": "IoU of the flow-propagated previous-frame mask and the current Mask2Former mask",
            "new_area_fraction": "share of the current Mask2Former mask that the propagated mask does not support",
            "lost_area_fraction": "share of the propagated mask that the current Mask2Former mask does not confirm",
            "forward_backward_consistency_fraction": "share of propagated pixels whose forward/backward residual is inside the frozen limit",
        },
    }
    if mask_area(previous_mask) == 0:
        base["reasons"] = ["empty_previous_mask2former_mask"]
        return base, empty
    if mask_area(current_mask) == 0:
        base["reasons"] = ["empty_current_mask2former_mask"]
        return base, empty
    timings: dict[str, float] = {}
    propagated, step = propagate_adjacent_timed(
        previous_image, current_image, previous_mask, thresholds, timings
    )
    propagated_px, current_px = mask_area(propagated), mask_area(current_mask)
    intersection = int(np.count_nonzero(propagated & current_mask))
    union = int(np.count_nonzero(propagated | current_mask))
    base.update({
        "flow_status": "available" if step["status"] == "candidate" else "unavailable",
        "forward_backward_consistency_fraction": step["candidate_flow_consistency_fraction"],
        "propagated_mask_area_fraction": safe_ratio(propagated_px, shape[0] * shape[1]),
        "current_mask_area_fraction": safe_ratio(current_px, shape[0] * shape[1]),
        "propagated_vs_current_iou": safe_ratio(intersection, union),
        "new_area_fraction": safe_ratio(current_px - intersection, current_px),
        "lost_area_fraction": safe_ratio(propagated_px - intersection, propagated_px),
        "flow_runtime_ms": float(timings["step_total_ms"]),
        "flow_field_ms": float(timings["flow_field_ms"]),
        "flow_consistency_ms": float(timings["consistency_ms"]),
        "flow_decision_ms": float(timings["decision_ms"]),
        "reasons": list(step["reasons"]),
        "propagated_pixel_count": propagated_px,
        "current_mask_pixel_count": current_px,
        "intersection_pixel_count": intersection,
        "new_area_pixel_count": current_px - intersection,
        "lost_area_pixel_count": propagated_px - intersection,
        "candidate_photometric_median_abs_gray": step["candidate_photometric_median_abs_gray"],
        "forward_backward_residual_median_px": step["forward_backward_residual_median_px"],
    })
    return base, propagated


# --------------------------------------------------------------------------------------
# manual anchor holdout evaluation
# --------------------------------------------------------------------------------------


def manual_floor_mask(label_path: Path, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    labels = rasterize_labelme(load_labelme(label_path))
    mask = labels["floor_eligible"]
    if mask.shape != shape:
        raise ValueError(f"{label_path}: label shape {mask.shape} differs from image shape {shape}")
    return mask, labels["valid_evaluation"]


def anchor_metrics(propagated: np.ndarray, reference_floor: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    metrics = binary_metrics(reference_floor, valid, propagated)
    truth = reference_floor & valid
    non_floor = (~reference_floor) & valid
    reference_px = int(truth.sum())
    return {
        "one_anchor_to_next_anchor_iou": float(metrics["iou"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "true_positive_px": int(metrics["true_positive_px"]),
        "false_positive_px": int(metrics["false_positive_px"]),
        "false_negative_px": int(metrics["false_negative_px"]),
        "false_positive_on_non_floor": safe_ratio(int(metrics["false_positive_px"]), int(non_floor.sum())),
        "false_negative_on_floor": safe_ratio(int(metrics["false_negative_px"]), reference_px),
        "mask_area_change": safe_ratio(mask_area(propagated) - reference_px, reference_px),
        "propagated_pixel_count": mask_area(propagated),
        "reference_floor_pixel_count": reference_px,
        "evaluation_valid_area_pixel_count": int(valid.sum()),
        "definition": {
            "reference": "the target anchor's manual floor_eligible polygon rasterised to the upright frame",
            "false_positive_on_non_floor": "false-positive pixels divided by the audited non-floor area",
            "false_negative_on_floor": "false-negative pixels divided by the audited floor area",
            "mask_area_change": "(propagated pixels - audited floor pixels) / audited floor pixels",
        },
    }


def step_diagnostics(steps: Sequence[dict[str, Any]], frame_shape: tuple[int, int]) -> dict[str, Any]:
    pixels = frame_shape[0] * frame_shape[1]
    areas = [safe_ratio(step["propagated_floor_pixel_count"], pixels) for step in steps]
    consistencies = [step["candidate_flow_consistency_fraction"] for step in steps]
    quarter = max(1, len(steps) // 4)
    return {
        "step_count": len(steps),
        "propagated_area_fraction_first_quarter_mean": mean_or_none(areas[:quarter]),
        "propagated_area_fraction_last_quarter_mean": mean_or_none(areas[-quarter:]),
        "propagated_area_fraction_trend_per_step": linear_trend(areas),
        "flow_consistency_first_quarter_mean": mean_or_none(consistencies[:quarter]),
        "flow_consistency_last_quarter_mean": mean_or_none(consistencies[-quarter:]),
        "flow_consistency_trend_per_step": linear_trend(consistencies),
        "step_area_fraction": areas,
        "step_flow_consistency": consistencies,
        "step_status": [step["status"] for step in steps],
        "step_reason_counts": dict(Counter(reason for step in steps for reason in step["reasons"])),
        "step_flow_field_ms": [step["flow_field_ms"] for step in steps],
        "step_consistency_ms": [step["consistency_ms"] for step in steps],
    }


def evaluate_anchor_interval(
    *,
    view: str,
    source_pair_id: int,
    target_pair_id: int,
    seed_masks: dict[str, np.ndarray],
    load_image: Callable[[int], np.ndarray],
    step_count: int,
    thresholds: PropagationThresholds,
    target_loader: Callable[[], tuple[np.ndarray, np.ndarray]],
    phase_log: list[str],
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, float]]]:
    """Propagate every seed chain first, and only then read the target label.

    ``target_loader`` is called once and strictly after every chain has been
    produced, which is the code-level guarantee that the target label cannot
    influence the intermediate propagation (enforced by an ordering test).
    """
    propagated, steps_by_seed, flow_steps = propagate_chains_locked(
        seed_masks=seed_masks,
        load_image=load_image,
        first_pair_id=source_pair_id,
        step_count=step_count,
        thresholds=thresholds,
        phase_log=phase_log,
        view=view,
    )
    shape = load_image(source_pair_id).shape[:2]
    phase_log.append(f"load_target_label:{view}")
    target_floor, target_valid = target_loader()
    seeds: dict[str, Any] = {}
    for seed_name in sorted(propagated):
        seeds[seed_name] = {
            "against_manual_target": anchor_metrics(propagated[seed_name], target_floor, target_valid),
            "step_diagnostics": step_diagnostics(steps_by_seed[seed_name], shape),
        }
    record = {
        "view": view,
        "source_anchor_pair_id": source_pair_id,
        "target_anchor_pair_id": target_pair_id,
        "gap_frames": target_pair_id - source_pair_id,
        "propagation_step_count": step_count,
        "seeds": seeds,
        "target_label_read_phase": "after_propagation",
    }
    return record, propagated, flow_steps


# --------------------------------------------------------------------------------------
# static-background VO module (fail closed)
# --------------------------------------------------------------------------------------


def static_background_domain_certificate(
    *, floor_masks_available: bool, person_regions_available: bool, walker_regions_available: bool
) -> dict[str, Any]:
    """Certify that every excluded region is actually available for a frame pair."""
    missing: list[str] = []
    if not floor_masks_available:
        missing.append("mask2former_floor_candidate_region_unavailable")
    if not person_regions_available:
        missing.append("pmpsoe_person_region_unavailable")
    if not walker_regions_available:
        missing.append("unverified_walker_candidate_region_unavailable")
    return {
        "feature_domain": STATIC_BACKGROUND_DOMAIN,
        "excluded_regions": [
            "mask2former_floor_candidate",
            "pmpsoe_person_region",
            "unverified_walker_candidate_region",
        ],
        "ground_region_used_for_motion": False,
        "certified": not missing,
        "missing_exclusions": missing,
        "note": VO_STATIC_DOMAIN_NOTE,
    }


def evaluate_vo_interval(
    *,
    matcher: str,
    plane_method: str,
    source_pair_id: int,
    target_pair_id: int,
    source_plane: LocalPlane | None,
    source_anchor_cross_region_status: str | None,
    target_anchor_cross_region_status: str | None,
    target_plane: LocalPlane | None,
    relative_pose_provider: Callable[[int, int], tuple[RelativePose | None, str]],
    domain_certificate: Callable[[int, int], dict[str, Any]],
    runtime_recorder: Callable[[int, int, float], None] | None = None,
) -> dict[str, Any]:
    """Propagate one direct source plane to a target anchor, or fail closed.

    The target plane is only ever touched in the comparison at the very end; it
    can never enter the propagation.
    """
    step_count = int(target_pair_id - source_pair_id)
    record: dict[str, Any] = {
        "schema_version": "vo_plane_propagation_v1",
        "matcher": matcher,
        "plane_method": plane_method,
        "source_anchor_pair_id": int(source_pair_id),
        "target_anchor_pair_id": int(target_pair_id),
        "source_plane_status": "unavailable",
        "vo_status": "unavailable",
        "static_feature_count": None,
        "pose_inlier_count": None,
        "relative_pose_reprojection_px": None,
        "propagation_step_count": step_count,
        "propagated_vs_target_normal_deg": None,
        "propagated_vs_target_offset_mm": None,
        "vo_runtime_ms_per_step": None,
        "vo_runtime_ms_total": None,
        "reasons": [],
        "confidence_boundary": VO_CONFIDENCE_BOUNDARY,
        "source_anchor_cross_region_status": source_anchor_cross_region_status,
        "target_anchor_cross_region_status": target_anchor_cross_region_status,
        "vo_source_anchor_rule": "cross_region_evidence.status == 'pass'",
        "direct_definition": DIRECT_DEFINITION,
        "target_plane_used_for_comparison_only": True,
        "target_plane_fed_back_into_propagation": False,
        "reference_pose_status_counts": {},
        "static_background_domain": domain_certificate(source_pair_id, source_pair_id + 1),
        "interpretation": VO_CONFIDENCE_BOUNDARY,
    }
    if source_plane is None or source_anchor_cross_region_status != "pass":
        record["reasons"] = ["no_direct_anchor_after_cross_region_gate"]
        return record
    record["source_plane_status"] = "direct"
    if target_plane is None:
        record["reasons"] = ["target_anchor_has_no_direct_plane_for_comparison"]
        return record

    status_counts: Counter = Counter()
    poses: list[RelativePose] = []
    step_times: list[float] = []
    certificate_times: list[float] = []
    feature_counts: list[int] = []
    inlier_counts: list[int] = []
    for offset in range(step_count):
        from_id, to_id = source_pair_id + offset, source_pair_id + offset + 1
        started = time.perf_counter()
        pose, reason = relative_pose_provider(from_id, to_id)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        certificate_times.append(elapsed_ms)
        if pose is None:
            # A step that produced no pose is not a propagation step: its cost is
            # reported separately and never enters the propagation-latency estimate.
            status_counts[reason] += 1
            break
        if runtime_recorder is not None:
            runtime_recorder(from_id, to_id, elapsed_ms)
        step_times.append(elapsed_ms)
        poses.append(pose)
        status_counts["accepted"] += 1
        feature_counts.append(int(pose.static_3d_correspondence_count))
        inlier_counts.append(int(pose.ransac_inlier_count))
    record["reference_pose_status_counts"] = dict(status_counts)
    record["certificate_check_ms"] = median_or_none(certificate_times)
    record["vo_runtime_ms_per_step"] = median_or_none(step_times)
    record["vo_runtime_ms_total"] = float(sum(step_times)) if step_times else None
    if feature_counts:
        record["static_feature_count"] = int(median_or_none(feature_counts) or 0)
    if inlier_counts:
        record["pose_inlier_count"] = int(median_or_none(inlier_counts) or 0)
    if len(poses) != step_count:
        reasons = sorted({reason for reason in status_counts if reason != "accepted"})
        if not record["static_background_domain"]["certified"]:
            reasons.append("static_background_domain_not_certifiable")
        record["reasons"] = reasons or ["static_background_relative_pose_chain_incomplete"]
        return record

    folded = source_plane
    for pose in poses:
        folded = propagate_plane(folded, pose)
    rotation, translation = compose_relative_poses(poses)
    composed = LocalPlane(
        rotation @ source_plane.normal,
        source_plane.offset - float((rotation @ source_plane.normal) @ translation),
    )
    if abs(folded.offset - composed.offset) > 1e-6 or float(np.linalg.norm(folded.normal - composed.normal)) > 1e-9:
        raise RuntimeError(
            "sequential plane propagation and composed-pose propagation disagree; refusing to report the result"
        )
    agreement = plane_agreement(folded, target_plane)
    record.update({
        "vo_status": "propagated",
        "propagated_vs_target_normal_deg": float(agreement["normal_angle_deg"]),
        "propagated_vs_target_offset_mm": float(agreement["camera_plane_distance_delta"]),
        "reasons": [],
    })
    return record


# --------------------------------------------------------------------------------------
# the fixed 27-combination matrix
# --------------------------------------------------------------------------------------


def combination_id(matcher: str, plane_method: str, temporal_mode: str) -> str:
    return f"{ONLINE_SEMANTIC_MODEL}__{matcher}__{plane_method}__{temporal_mode}"


def build_combination_matrix(
    upstream: Upstream,
    *,
    flow_summary: dict[str, Any],
    vo_by_key: dict[tuple[str, str], list[dict[str, Any]]],
    vo_runtime_stats: dict[str, Any],
    declared_frame_budget_ms: float,
) -> list[dict[str, Any]]:
    """Exactly 3 matchers x 3 plane methods x 3 temporal modes = 27 rows."""
    rows: list[dict[str, Any]] = []
    for matcher in MATCHERS:
        offline_window = upstream.task02_summary["dynamicstereo_offline_window"] if matcher == "dynamicstereo" else None
        for plane_method in PLANE_METHODS:
            combination = upstream.combination(matcher, plane_method)
            comparable = upstream.comparable_timing(matcher, plane_method)
            direct_median = float(comparable["estimated_direct_pipeline_ms"]["median_ms"])
            direct_p95 = float(comparable["estimated_direct_pipeline_ms"]["p95_ms"])
            base_modules: dict[str, Any] = {}
            for name in ("mask_load_ms", "rectification_ms", "candidate_filter_ms", "triangulation_ms", "plane_fit_ms"):
                if isinstance(comparable.get(name), dict):
                    base_modules[name] = {**comparable[name], "warmup_excluded_count": UPSTREAM_WARMUP_EXCLUDED}
            for temporal_mode in TEMPORAL_MODES:
                rows.append(build_combination_row(
                    matcher=matcher,
                    plane_method=plane_method,
                    temporal_mode=temporal_mode,
                    combination=combination,
                    direct_modules=base_modules,
                    direct_median_ms=direct_median,
                    direct_p95_ms=direct_p95,
                    flow_summary=flow_summary,
                    vo_records=vo_by_key.get((matcher, plane_method), []),
                    vo_runtime_stats=vo_runtime_stats,
                    offline_window=offline_window,
                    declared_frame_budget_ms=declared_frame_budget_ms,
                ))
    identifiers = [row["combination_id"] for row in rows]
    if len(rows) != 27:
        raise RuntimeError(f"the fixed matrix must have 27 rows, produced {len(rows)}")
    if len(set(identifiers)) != 27:
        raise RuntimeError("combination ids must be unique")
    return rows


def build_combination_row(
    *,
    matcher: str,
    plane_method: str,
    temporal_mode: str,
    combination: dict[str, Any],
    direct_modules: dict[str, Any],
    direct_median_ms: float,
    direct_p95_ms: float,
    flow_summary: dict[str, Any],
    vo_records: list[dict[str, Any]],
    vo_runtime_stats: dict[str, Any],
    offline_window: dict[str, Any] | None,
    declared_frame_budget_ms: float,
) -> dict[str, Any]:
    timing: dict[str, Any] = {f"direct_{name}": value for name, value in direct_modules.items()}
    requires_future_frames = bool(offline_window and offline_window.get("future_lookahead_frames"))
    full_chain_measured = temporal_mode == "direct_current_frame"
    estimated: float | None = None
    estimated_p95: float | None = None
    estimated_reason: str | None = None
    temporal_evidence: dict[str, Any]

    if temporal_mode == "direct_current_frame":
        temporal_evidence = {
            "temporal_mode": temporal_mode,
            "temporal_module_used": False,
            "note": "the plane, its inliers and its gate state are the current frame's own; no temporal module contributes",
        }
        estimated, estimated_p95 = direct_median_ms, direct_p95_ms
    elif temporal_mode == "flow_assisted_current_frame":
        temporal_evidence = {**flow_summary, "temporal_mode": temporal_mode, "temporal_module_used": True}
        flow_median = flow_summary["two_view_module_ms"]["median_ms"]
        flow_p95 = flow_summary["two_view_module_ms"]["p95_ms"]
        if flow_median is None or flow_p95 is None:
            estimated_reason = "flow_module_not_measured"
        else:
            estimated, estimated_p95 = float(direct_median_ms + flow_median), float(direct_p95_ms + flow_p95)
        for name, value in flow_summary["timing_modules"].items():
            timing[f"flow_{name}"] = value
    else:
        temporal_evidence = {
            **summarize_vo_records(vo_records), "temporal_mode": temporal_mode, "temporal_module_used": True,
        }
        per_step_stats = vo_runtime_stats
        per_step = per_step_stats["median_ms"]
        steps = temporal_evidence["propagation_step_count_max"]
        if per_step is None or not steps:
            estimated_reason = "vo_module_unavailable_no_certified_static_background_pose_chain"
        else:
            estimated = float(per_step) * float(steps)
            estimated_p95 = float(per_step_stats["p95_ms"] or per_step) * float(steps)
        timing["vo_runtime_ms_per_step"] = per_step_stats
        timing["vo_propagation_step_count"] = steps
        temporal_evidence["runtime_ms_per_step"] = per_step_stats

    row: dict[str, Any] = {
        "combination_id": combination_id(matcher, plane_method, temporal_mode),
        "semantic_model": ONLINE_SEMANTIC_MODEL,
        "matcher": matcher,
        "plane_method": plane_method,
        "temporal_mode": temporal_mode,
        "direct_spatial_evidence_status": "measured",
        "flow_module_status": "not_applicable" if temporal_mode != "flow_assisted_current_frame" else "measured",
        "vo_module_status": (
            "measured" if temporal_mode == "vo_propagated_plane" and temporal_evidence.get("propagated_intervals")
            else ("unavailable" if temporal_mode == "vo_propagated_plane" else "unavailable")
        ),
        "vo_module_applicability": (
            "used_by_this_temporal_mode" if temporal_mode == "vo_propagated_plane"
            else "not_applicable_for_this_temporal_mode_so_no_propagated_plane_exists"
        ),
        "full_chain_measured": bool(full_chain_measured),
        "measurement_scope": "direct_chain" if full_chain_measured else "temporal_module_composition",
        "semantic_evidence": combination["semantic_evidence"],
        "stereo_evidence": combination["stereo_evidence"],
        "plane_evidence": combination["plane_evidence"],
        "cross_region_evidence": combination["cross_region_evidence"],
        "temporal_evidence": temporal_evidence,
        "timing": timing,
        "estimated_end_to_end_latency_ms": estimated,
        "estimated_end_to_end_latency_p95_ms": estimated_p95,
        "estimated_end_to_end_latency_kind": "estimated_module_time_sum",
        "estimated_end_to_end_latency_caveat": LATENCY_SUM_CAVEAT,
        "measured_end_to_end_latency_ms": None,
        "measured_end_to_end_latency_reason": (
            "module times are summed; no end-to-end wall clock was measured for this composition"
        ),
        "estimated_end_to_end_latency_unavailable_reason": estimated_reason,
        "declared_frame_budget_ms": float(declared_frame_budget_ms),
        "realtime_compatible": False,
        "realtime_incompatibility_reasons": [],
        "confidence_boundary": MATRIX_BOUNDARY,
    }
    realtime_reasons: list[str] = []
    if estimated is None:
        realtime_reasons.append("latency_not_estimable_for_this_composition")
    elif estimated > declared_frame_budget_ms:
        realtime_reasons.append("estimated_module_time_exceeds_declared_frame_budget")
    if requires_future_frames:
        realtime_reasons.append("offline_five_frame_window_requires_four_future_frames")
    if temporal_mode == "vo_propagated_plane":
        realtime_reasons.append("plane_available_only_after_a_multi_frame_propagation_chain")
    row["realtime_incompatibility_reasons"] = realtime_reasons
    row["realtime_compatible"] = not realtime_reasons
    if offline_window is not None:
        row.update({
            "future_lookahead_frames": int(offline_window["future_lookahead_frames"]),
            "offline_window_inference": True,
            "realtime_compatible": False,
            "offline_window": dict(offline_window),
        })
    return row


def summarize_vo_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    propagated = [item for item in records if item["vo_status"] == "propagated"]
    reason_counts: Counter = Counter(reason for item in records for reason in item["reasons"])
    return {
        "vo_source_anchor_rule": "cross_region_evidence.status == 'pass'",
        "anchor_intervals_evaluated": len(records),
        "source_anchors_eligible": sum(1 for item in records if item["source_plane_status"] == "direct"),
        "propagated_intervals": len(propagated),
        "unavailable_intervals": sum(1 for item in records if item["vo_status"] == "unavailable"),
        "reason_counts": dict(sorted(reason_counts.items())),
        "propagated_vs_target_normal_deg": distribution([item["propagated_vs_target_normal_deg"] for item in propagated]),
        "propagated_vs_target_offset_mm": distribution([item["propagated_vs_target_offset_mm"] for item in propagated]),
        "propagation_step_count_max": max((item["propagation_step_count"] for item in records), default=0),
        "static_feature_count": distribution([item["static_feature_count"] for item in records]),
        "pose_inlier_count": distribution([item["pose_inlier_count"] for item in records]),
        "confidence_boundary": VO_CONFIDENCE_BOUNDARY,
        "fail_closed": True,
    }


# --------------------------------------------------------------------------------------
# visualizations
# --------------------------------------------------------------------------------------


def _tint(image: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int], alpha: float) -> np.ndarray:
    selected = np.asarray(mask) > 0
    if not np.any(selected):
        return image
    output = image.copy()
    output[selected] = np.clip(
        (1.0 - float(alpha)) * image[selected].astype(np.float64) + float(alpha) * np.asarray(colour, dtype=np.float64),
        0.0, 255.0,
    ).astype(np.uint8)
    return output


def _banner(image: np.ndarray, text: str, height: int = 22) -> np.ndarray:
    cv2.rectangle(image, (0, 0), (image.shape[1], height), (255, 255, 255), -1)
    cv2.putText(image, text[:190], (4, int(height * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def render_flow_panel(
    *, previous_image: np.ndarray, current_image: np.ndarray, previous_mask: np.ndarray,
    current_mask: np.ndarray, propagated: np.ndarray, title: str, target_height: int = 620,
) -> np.ndarray:
    first = _banner(_tint(previous_image, previous_mask, (0, 200, 0), 0.35), "previous frame + Mask2Former candidate (green)")
    second = _banner(
        _tint(_tint(current_image, current_mask, (0, 200, 0), 0.35), propagated, (0, 140, 255), 0.35),
        "current frame + current Mask2Former candidate (green) + flow-propagated mask (orange)",
    )
    valid = np.ones(current_image.shape[:2], dtype=bool)
    error = render_error_overlay(current_image, current_mask, valid, propagated, title)
    panels = []
    for panel in (first, second, error):
        scale = target_height / panel.shape[0]
        panels.append(cv2.resize(panel, (max(1, int(round(panel.shape[1] * scale))), target_height), interpolation=cv2.INTER_AREA))
    separator = np.zeros((target_height, 6, 3), dtype=np.uint8)
    return np.hstack((panels[0], separator, panels[1], separator, panels[2]))


def render_drift_curve(interval: dict[str, Any], title: str, *, width: int = 960, height: int = 480) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    _banner(canvas, title, height=26)
    left = interval["left"]["seeds"]["manual_seed"]["step_diagnostics"]
    right = interval["right"]["seeds"]["manual_seed"]["step_diagnostics"]
    area_left, area_right = left["step_area_fraction"], right["step_area_fraction"]
    steps = len(area_left)
    margin = 60
    plot_w, plot_h = width - 2 * margin, height - 2 * margin
    ceiling = max(max(area_left), max(area_right)) or 1.0

    def to_px(index: int, value: float, top: float) -> tuple[int, int]:
        x = margin + int(round(plot_w * index / max(1, steps - 1)))
        y = height - margin - int(round(plot_h * min(max(value, 0.0) / top, 1.0)))
        return x, y

    cv2.line(canvas, (margin, height - margin), (width - margin, height - margin), (0, 0, 0), 1)
    cv2.line(canvas, (margin, margin), (margin, height - margin), (0, 0, 0), 1)
    for values, colour in ((area_left, (0, 160, 0)), (area_right, (0, 0, 220))):
        points = np.asarray([to_px(i, float(v or 0.0), ceiling) for i, v in enumerate(values)], dtype=np.int32)
        cv2.polylines(canvas, [points], False, colour, 2, cv2.LINE_AA)
    for diagnostics, colour in ((left, (0, 160, 0)), (right, (0, 0, 220))):
        points = np.asarray(
            [to_px(i, float(v or 0.0), 1.0) for i, v in enumerate(diagnostics["step_flow_consistency"])], dtype=np.int32
        )
        cv2.polylines(canvas, [points], False, colour, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"green=left view blue=right view; thick=propagated area fraction (max {ceiling:.3f}) thin=flow consistency",
                (margin, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "steps 1..%d; manual-seed final IoU left=%.3f right=%.3f" % (
        steps,
        interval["left"]["seeds"]["manual_seed"]["against_manual_target"]["one_anchor_to_next_anchor_iou"],
        interval["right"]["seeds"]["manual_seed"]["against_manual_target"]["one_anchor_to_next_anchor_iou"],
    ), (margin, margin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def render_matrix_status(rows: Sequence[dict[str, Any]], *, width: int = 1560) -> np.ndarray:
    line_height = 26
    header_height = 34
    canvas = np.full((header_height + line_height * (len(rows) + 1), width, 3), 255, dtype=np.uint8)
    columns = (30, 120, 170, 260, 200, 210, 150, 110, 130)
    labels = ("#", "matcher", "plane_method", "temporal_mode", "full_chain_measured", "measurement_scope",
              "est latency ms", "realtime", "cross gate pass")
    x = 8
    for label, offset in zip(labels, columns):
        cv2.putText(canvas, label, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        x += offset
    cv2.line(canvas, (4, header_height - 8), (width - 4, header_height - 8), (0, 0, 0), 1)
    for index, row in enumerate(rows):
        y = header_height + line_height * (index + 1)
        gate = row["cross_region_evidence"]
        latency = row["estimated_end_to_end_latency_ms"]
        values = (
            str(index + 1), row["matcher"], row["plane_method"], row["temporal_mode"],
            str(row["full_chain_measured"]), row["measurement_scope"],
            "n/a" if latency is None else f"{latency:.1f}", str(row["realtime_compatible"]),
            f"{gate['pass_frames']}/{gate['pass_frames'] + gate['fail_frames'] + gate['unavailable_frames']}",
        )
        cv2.rectangle(canvas, (4, y - 18), (width - 4, y + 6), (246, 246, 246) if index % 2 else (255, 255, 255), -1)
        x = 8
        for value, offset in zip(values, columns):
            cv2.putText(canvas, str(value)[:36], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)
            x += offset
    return canvas


# --------------------------------------------------------------------------------------
# report writers
# --------------------------------------------------------------------------------------


def render_recommendation(
    *, matrix: Sequence[dict[str, Any]], flow_evaluation: dict[str, Any], vo_summary: dict[str, Any],
    upstream: Upstream, declared_frame_budget_ms: float,
) -> str:
    direct = [row for row in matrix if row["temporal_mode"] == "direct_current_frame"]
    flow = [row for row in matrix if row["temporal_mode"] == "flow_assisted_current_frame"]

    def ranked(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(rows, key=lambda row: (row["estimated_end_to_end_latency_ms"] is None,
                                             row["estimated_end_to_end_latency_ms"] or 1e18))

    fastest = ranked(direct)[0]

    def strength(row: dict[str, Any]) -> tuple:
        plane, gate = row["plane_evidence"], row["cross_region_evidence"]
        total = gate["pass_frames"] + gate["fail_frames"] + gate["unavailable_frames"]
        # The inlier fraction is rounded before it is compared so that a 0.05
        # percentage-point difference cannot outrank a large residual
        # difference; ties then fall through to the next criterion.
        return (
            round(float(plane["median_inlier_fraction"] or 0.0), 2),
            -float(plane["median_residual_mm"] or 1e9),
            float(gate["pass_frames"]) / float(total or 1),
            float(plane["median_coverage_fraction"] or 0.0),
        )

    strongest = sorted(direct, key=strength, reverse=True)[0]
    fastest_flow = ranked(flow)[0]
    fastest_dynamic = ranked([row for row in matrix
                              if row["matcher"] == "dynamicstereo" and row["temporal_mode"] == "direct_current_frame"])[0]
    offline = upstream.task02_summary["dynamicstereo_offline_latency"]
    manual = flow_evaluation["macro_metrics"]["manual_seed_vs_manual_target"]
    production = flow_evaluation["macro_metrics"]["production_compatible_mask2former_seed_vs_mask2former_target"]
    return "\n".join([
        "# 最终推荐：27 组合时空地面重构（G20260913）",
        "",
        f"日期：2026-09-13。在线语义输入固定为 `{ONLINE_SEMANTIC_MODEL}` 地面**候选**；人工地面掩膜只用于二维身份审计与内部对照，不是在线的时序输入，也不是三维真值。",
        "",
        "## A. 当前最快的直接空间组合",
        "",
        f"- `{fastest['combination_id']}`",
        f"- 估算模块时间和（P50）`{fastest['estimated_end_to_end_latency_ms']:.1f} ms`，P95 估算 `{fastest['estimated_end_to_end_latency_p95_ms']:.1f} ms`。",
        "- 判据：三种平面拟合器中 `soft_weighted_irls` 的拟合耗时最低（P50 约 30 ms 量级），而三种匹配器中 SGBM 的空间链最便宜。",
        "- 这是**模块时间相加的估算**，不是实测端到端时间；同时注意 `soft_weighted_irls` 在 Mask2Former 候选下的中位内点率最低、中位残差最高（见 B 节），“最快”与“最强”不同源。",
        "",
        "## B. 当前内部几何证据最强的直接空间组合",
        "",
        f"- `{strongest['combination_id']}`",
        f"- 中位内点率 `{strongest['plane_evidence']['median_inlier_fraction']:.4f}`、中位残差 `{strongest['plane_evidence']['median_residual_mm']:.3f} mm`、"
        f"分区门通过 `{strongest['cross_region_evidence']['pass_frames']}/12`、中位覆盖率 `{strongest['plane_evidence']['median_coverage_fraction']:.4f}`、"
        f"中位重投影 `{strongest['stereo_evidence']['median_reprojection_px']:.4f} px`。",
        f"- 估算模块时间和（P50）`{strongest['estimated_end_to_end_latency_ms']:.1f} ms`。",
        "- 排序键（依次）：中位内点率（先四舍五入到两位，避免把 0.05 个百分点的差异排在 43% 的残差差异之上）→ 中位残差 → 分区门通过率 → 中位覆盖率。"
        "这是**多指标**规则，因此“最强”不等于耗时最短，也不等于视差最可靠。",
        "- 该“最强”只是共享图像、共享标定、共享 Mask2Former 候选条件下的**内部一致性**。学习型匹配器的中位视差有一半以上落在冻结 SGBM 搜索范围之外，"
        "其高内点率与低残差可以来自平滑稠密的视差场，**不能**读成真实地面精度更高。",
        "",
        "## C. 适合实时研究原型的组合",
        "",
        f"- 架构上最接近实时的是 `{fastest['combination_id']}`：无未来帧等待、模块边界清楚、平面拟合只跑一次。",
        f"- 但它仍**不满足**本次声明的帧预算 `{declared_frame_budget_ms:.3f} ms`（源采集有效配对帧率 {SOURCE_EFFECTIVE_PAIR_FPS:.4f} pairs/s）："
        f"估算 P50 `{fastest['estimated_end_to_end_latency_ms']:.1f} ms`，因此 `realtime_compatible = false`。",
        f"- 加入光流后最快的 `{fastest_flow['combination_id']}` 估算 P50 `{fastest_flow['estimated_end_to_end_latency_ms']:.1f} ms`，"
        f"每帧双视图光流代价中位 `{fastest_flow['temporal_evidence']['two_view_module_ms']['median_ms']:.1f} ms`；光流只增加证据，不降低任何门限。",
        "- 结论：当前没有任何组合达到本机实时预算。可以称为“研究原型可承载”的只有其**架构**（无未来帧、模块可替换、失败可关闭），不是其测得速度。",
        "",
        "## D. 仅适合离线研究的组合",
        "",
        f"- 全部 `dynamicstereo` 组合（9 行）：五帧前向窗口，目标帧取窗口第一帧，需要 `{offline['future_lookahead_frames']}` 帧未来信息，`realtime_compatible = false`。",
        f"- 仅未来等待就是 `{offline['future_information_wait_ms_at_effective_pair_rate']:.2f} ms`（按有效配对帧率）；窗口推理中位 `{offline['window_inference_ms_median']:.2f} ms`，"
        f"离线端到端延迟中位 `{offline['offline_latency_ms_median']:.2f} ms`，窗口吞吐 `{offline['window_throughput_frames_per_second']:.3f}` 窗口/秒。",
        f"- 其最快的直接行 `{fastest_dynamic['combination_id']}` 估算模块时间和仍为 `{fastest_dynamic['estimated_end_to_end_latency_ms']:.1f} ms`；"
        "即便只看 GPU forward 很快，也不得改成实时。",
        "- DynamicStereo 使用官方代码与官方权重，但运行于兼容 CUDA 12.8 / torch 2.11 环境，不是官方 README 钉死的 torch 1.12 环境复现；这一条必须随任何 DynamicStereo 结论一起引用。",
        "",
        "## E. 必须拒绝输出或标为 unavailable 的组合",
        "",
        f"- 全部 `vo_propagated_plane` 组合（9 行）：`vo_status = unavailable`。DynamicStereo 的 3 行原因为 `no_direct_anchor_after_cross_region_gate`"
        "（冻结分区门 0/12，不得作为 VO direct 锚帧，也不得用候选平面替代 direct 锚帧，更不得为了凑出传播区间而放宽分区门）。",
        "- SGBM / IGEV 中源锚帧通过分区门的区间同样为 unavailable，原因是冻结产物里没有逐帧静态背景米制对应；补出它必须重新运行双目匹配器，本任务明令禁止。",
        f"- 统计：可评估锚点区间 `{vo_summary['anchor_intervals_evaluated']}` 个，源锚帧合格 `{vo_summary['source_anchors_eligible']}` 个，"
        f"成功传播 `{vo_summary['propagated_intervals']}` 个；原因计数 `{json.dumps(vo_summary['reason_counts'], ensure_ascii=False)}`。",
        "- 未验证助步器候选排除区只有 12 帧（`pair_0000`–`pair_0011`），而相机装在助步器上，助步器像素在每一帧都存在且近似静止；"
        "缺少逐帧排除区时它会被当成“静态背景”特征并把位姿拉向零运动。因此 VO 失败关闭，绝不用地面、人或助步器点补足。",
        "",
        "## F. 只属于内部一致性的结论",
        "",
        "- IGEV 在共享分区门上 `12/12` 通过、中位内点率约 `0.97`、中位残差 `1.7–2.5 mm`：这是**内部自洽**，不是真实精度；"
        "它的中位重投影（约 `0.289 px`）高于 SGBM（约 `0.226 px`），且中位视差有一半以上超出冻结 SGBM 搜索范围。",
        "- SGBM 与两个学习型模型的一切内点率、残差、覆盖率、重投影一致性、左右一致率都只在同一批图像与同一标定下成立。",
        "- 光流的一切 IoU / precision / recall 都是二维掩膜一致性：",
        f"  人工种子 40 步传播到下一人工锚点的宏 IoU 左 `{manual['left']['one_anchor_to_next_anchor_iou']:.3f}`、右 `{manual['right']['one_anchor_to_next_anchor_iou']:.3f}`；",
        f"  生产兼容（Mask2Former 当前帧 vs 由上一帧 Mask2Former 连续传播）宏 IoU 左 `{production['left']['one_anchor_to_next_anchor_iou']:.3f}`、"
        f"右 `{production['right']['one_anchor_to_next_anchor_iou']:.3f}`。",
        "- 这三者不是精度关系：Mask2Former 掩膜本身是候选，人工掩膜只是二维身份审计；光流一致性也只是图像空间证据。",
        "",
        "## G. 要比较真实精度仍缺少什么独立真值",
        "",
        "1. **独立测量的地面平面**：平放标定板或静态参考建立 `measured_locked` 地面系，且不得用同一套 RANSAC 结果自证。",
        "2. **未参与拟合的独立样本**：跨序列、跨位置、跨人的留出帧，而不是同一 12 帧内的重排。",
        "3. **逐帧助步器候选排除区**：需要覆盖全部 448 对，才能把“静态背景”域真正认证下来；当前只有 12 帧。",
        "4. **逐帧米制静态背景对应**：VO 的相邻位姿需要它；冻结产物中不存在，补出它等于重新运行双目匹配器，必须另立单变量任务。",
        "5. **硬件级同步与曝光时间戳**：现有配对是主机单调时钟、25 ms 门限，不是硬件触发。",
        "6. **动态基线的官方环境复现**：DynamicStereo 需在官方钉死的 torch 1.12 环境复核，才能与论文口径对齐。",
        "",
        "在上述独立真值到位之前，本目录的任何数字都不得表述为真实地面精度、相机高度精度、足地高度、步态事件、接触或临床指标。",
        "",
    ])


def render_experiment_markdown(
    *, args: argparse.Namespace, summary: dict[str, Any], flow_evaluation: dict[str, Any],
    matrix: Sequence[dict[str, Any]], log: RunLog,
) -> str:
    flow_stats = summary["flow_summary"]
    vo_stats = summary["vo_summary"]
    manual = flow_evaluation["macro_metrics"]["manual_seed_vs_manual_target"]
    production = flow_evaluation["macro_metrics"]["production_compatible_mask2former_seed_vs_mask2former_target"]
    drift = flow_evaluation["drift_trend"]
    return "\n".join([
        "# G20260913 光流 / 视觉里程计验证与 27 组合时空地面重构汇总",
        "",
        "日期：2026-09-13（北京时间）",
        f"状态：`{summary['status']}`",
        "任务：task-03（只读 task-01 / task-02 归档；不重跑 SGBM / IGEV / DynamicStereo）",
        "",
        "## 1. 目标与新增内容",
        "",
        "1. 光流的地面掩膜**时序一致性**与**传播漂移**验证；",
        "2. 视觉里程计（VO）的**局部平面传播误差**与**漂移**验证；",
        "3. 汇总固定的 **3 匹配器 × 3 平面拟合器 × 3 时序状态 = 27** 组合，含统一可信度证据、模块时间、预计端到端时间与最终推荐。",
        "",
        "新增的只有光流与 VO 两个时序模块和一次汇总；空间重构部分**只读** task-01 的 `G20260912_modular_ground_benchmark_v1` 与 task-02 的 "
        "`G20260912_learned_stereo_replacement_benchmark_v1`，既未写入也未重跑。",
        "",
        "## 2. 输入与固定的 12 帧",
        "",
        "| 项目 | 路径 |",
        "|---|---|",
        f"| 左图（正立） | `{args.left_dir}` |",
        f"| 右图（正立） | `{args.right_dir}` |",
        f"| Mask2Former 地面候选（左） | `{args.left_floor_mask_dir}` |",
        f"| Mask2Former 地面候选（右） | `{args.right_floor_mask_dir}` |",
        f"| 人工地面标注（左/右） | `{args.manual_left_label_dir}` / `{args.manual_right_label_dir}` |",
        f"| task-01（SGBM 归档，只读） | `{args.task01_dir}` |",
        f"| task-02（IGEV / DynamicStereo 归档，只读） | `{args.task02_dir}` |",
        "",
        "固定人工锚点帧（间隔 40）：`pair_0000, pair_0040, …, pair_0440`。左右视图**各自独立**运行光流，左图掩膜从不复制到右图。",
        "",
        "## 3. 严格事实边界（原样保留）",
        "",
        "1. Mask2Former 是在线地面语义**候选**，不是地面真值；",
        "2. 人工地面掩膜只提供二维身份审计，不是三维物理地面真值；",
        "3. SGBM / IGEV / DynamicStereo 的低残差、内点率或重投影一致性都只是内部几何证据；",
        "4. IGEV 在现有 12 帧内部分区门表现较好，但不得称真实精度提升；",
        "5. DynamicStereo 在冻结分区门下为 **0/12** 通过；",
        "6. DynamicStereo **不得**作为 VO 平面传播的 direct 锚帧；",
        "7. DynamicStereo 使用官方代码与官方权重，但运行于兼容 CUDA 12.8 / torch 2.11 环境，不是官方 README 钉死的 torch 1.12 环境复现；",
        "8. 现有数据没有独立已知物理地面真值。",
        "",
        "## 4. 光流模块（只做时序语义证据）",
        "",
        f"- 左右视图独立执行；每视图每连续帧对一条记录，本次 `{flow_stats['record_count']}` 条 = `{flow_stats['interval_count']}` 个相邻锚点区间 × 2 视图。",
        "- 冻结阈值（沿用既有 `PropagationThresholds`，未改）：前后向残差上限 "
        f"`{args.maximum_forward_backward_error_px} px`、最少流一致性 `{args.minimum_candidate_flow_consistency}`、光度中位上限 "
        f"`{args.maximum_candidate_photometric_median}`、最少传播像素 `{args.minimum_candidate_pixels}`。",
        "- **光流不覆盖当前帧的 Mask2Former 掩膜**：每条记录显式写入 `current_mask_is_authoritative = true` 与 "
        "`propagated_mask_used_as_geometry_input = false`；时序状态只能是 `flow_assisted_current_frame`。",
        "- 一致性不足时只降级状态或输出 `unavailable`，绝不因传播掩膜存在而绕过当前帧的双目、三角化、RANSAC 或分区门。",
        "",
        "### 4.1 逐帧一致性（Mask2Former 前帧 → 传播 → Mask2Former 当前帧）",
        "",
        f"- 流可用 `{flow_stats['available_records']}/{flow_stats['record_count']}`，不可用原因计数 `{json.dumps(flow_stats['reason_counts'], ensure_ascii=False)}`；",
        f"- 传播掩膜 vs 当前掩膜 IoU 中位：左 `{flow_stats['left']['propagated_vs_current_iou']['median']}`、右 `{flow_stats['right']['propagated_vs_current_iou']['median']}`；",
        f"- 前后向一致性分数中位：左 `{flow_stats['left']['forward_backward_consistency_fraction']['median']}`、右 `{flow_stats['right']['forward_backward_consistency_fraction']['median']}`；",
        f"- `new_area_fraction` 中位：左 `{flow_stats['left']['new_area_fraction']['median']}`、右 `{flow_stats['right']['new_area_fraction']['median']}`；",
        f"- `lost_area_fraction` 中位：左 `{flow_stats['left']['lost_area_fraction']['median']}`、右 `{flow_stats['right']['lost_area_fraction']['median']}`。",
        "",
        "### 4.2 人工锚点留出漂移评估（每区间连续传播 40 步）",
        "",
        "从每个源锚点开始连续传播 40 次到下一个锚点，**只在终点**与该终点人工地面掩膜比较。终点标签在传播完成后才读取"
        "（`target_label_read_phase = after_propagation`，并由单测的调用顺序检查强制），绝不反馈给中间传播，也不用于训练。",
        "",
        "| 宏平均指标 | 左视图 | 右视图 |",
        "|---|---:|---:|",
        f"| one_anchor_to_next_anchor IoU（人工种子） | {manual['left']['one_anchor_to_next_anchor_iou']:.4f} | {manual['right']['one_anchor_to_next_anchor_iou']:.4f} |",
        f"| precision | {manual['left']['precision']:.4f} | {manual['right']['precision']:.4f} |",
        f"| recall | {manual['left']['recall']:.4f} | {manual['right']['recall']:.4f} |",
        f"| false_positive_on_non_floor | {manual['left']['false_positive_on_non_floor']:.4f} | {manual['right']['false_positive_on_non_floor']:.4f} |",
        f"| false_negative_on_floor | {manual['left']['false_negative_on_floor']:.4f} | {manual['right']['false_negative_on_floor']:.4f} |",
        f"| mask_area_change | {manual['left']['mask_area_change']:.4f} | {manual['right']['mask_area_change']:.4f} |",
        "",
        f"同一条链改用 Mask2Former 种子后对同一人工终点的宏 IoU：左 `{flow_evaluation['macro_metrics']['mask2former_seed_vs_manual_target']['left']['one_anchor_to_next_anchor_iou']:.4f}`、"
        f"右 `{flow_evaluation['macro_metrics']['mask2former_seed_vs_manual_target']['right']['one_anchor_to_next_anchor_iou']:.4f}`。",
        "",
        f"生产兼容评估（Mask2Former 当前帧 vs 由上一帧 Mask2Former 连续传播 40 步）：宏 IoU 左 "
        f"`{production['left']['one_anchor_to_next_anchor_iou']:.4f}`、右 `{production['right']['one_anchor_to_next_anchor_iou']:.4f}`；仅报告一致性，不作为真值。",
        "",
        "漂移趋势（传播面积分数对步序的斜率，正值为增长/泄漏）：",
        "",
        f"- 左视图 `{drift['left']['propagated_area_fraction_trend_per_step']}`，第一/最后四分之一面积分数 "
        f"`{drift['left']['propagated_area_fraction_first_quarter_mean']}` → `{drift['left']['propagated_area_fraction_last_quarter_mean']}`；",
        f"- 右视图 `{drift['right']['propagated_area_fraction_trend_per_step']}`，第一/最后四分之一面积分数 "
        f"`{drift['right']['propagated_area_fraction_first_quarter_mean']}` → `{drift['right']['propagated_area_fraction_last_quarter_mean']}`；",
        f"- 流一致性斜率：左 `{drift['left']['flow_consistency_trend_per_step']}`、右 `{drift['right']['flow_consistency_trend_per_step']}`。",
        "",
        "## 5. VO 模块（只传播已验收的局部平面，失败关闭）",
        "",
        "- 只允许静态背景特征；排除区固定为 Mask2Former floor 候选区、PMPose 人体区、未验证助步器候选区；",
        "- 源锚帧规则 `cross_region_evidence.status == \"pass\"`（本任务定义的 `direct`，**不是**项目更严格的人工掩膜 `manually_audited` direct，也不是物理地面真值）；",
        "- 源平面经逐帧相对位姿传播到目标锚点，再与该目标锚点的 direct 平面比较；目标平面只用于比较，绝不参与传播，也不反馈给前面的传播；",
        "- 没有足够静态背景特征或缺少任一排除区时输出 `unavailable`，绝不用地面、人或助步器点补足。",
        "",
        f"结果：可评估锚点区间 `{vo_stats['anchor_intervals_evaluated']}`，源锚帧合格 `{vo_stats['source_anchors_eligible']}`，成功传播 "
        f"`{vo_stats['propagated_intervals']}`，不可用 `{vo_stats['unavailable_intervals']}`；原因计数 `{json.dumps(vo_stats['reason_counts'], ensure_ascii=False)}`。",
        "",
        "失败关闭的两个可核查原因：",
        "",
        "1. **DynamicStereo 分区门 0/12** → 其全部组合写 `no_direct_anchor_after_cross_region_gate`，不得用候选平面替代 direct 锚帧，也不得放宽分区门；",
        "2. **冻结产物缺少逐帧静态背景米制对应** → 40 步链需要 440 个中间帧的米制双目信息，归档中不存在；补出它必须重新运行双目匹配器，本任务禁止。"
        "同时未验证助步器候选排除区只有 12 帧，而相机装在助步器上、助步器像素在每帧都存在且近似静止，缺少逐帧排除区就无法认证“静态背景”域。",
        "",
        "## 6. 固定的 27 组合矩阵",
        "",
        f"- 行数 `{len(matrix)}`（3 匹配器 × 3 平面拟合器 × 3 时序状态），组合 ID 唯一，`semantic_model = {ONLINE_SEMANTIC_MODEL}`；",
        "- `direct_current_frame`：`full_chain_measured = true`，`measurement_scope = direct_chain`；",
        "- `flow_assisted_current_frame`：`full_chain_measured = false`，`measurement_scope = temporal_module_composition`，空间链来自已测 direct，光流是独立测得的附加语义门；",
        "- `vo_propagated_plane`：`full_chain_measured = false`，平面传播模块独立实测，不重新运行完整空间链；",
        f"- `realtime_compatible = true` 的行数 `{summary['matrix_summary']['realtime_compatible_true']}`（声明帧预算 `{DECLARED_FRAME_BUDGET_MS:.3f} ms`）；",
        f"- `measured_end_to_end_latency_ms` 非空的行数 `{summary['matrix_summary']['measured_end_to_end_latency_rows']}`；"
        "`estimated_end_to_end_latency_ms` 一律按模块时间相加并单列为估算。",
        "",
        "## 7. 速度",
        "",
        "空间直接链的比较只使用 task-02 的 `timing_on_task01_comparable_frames`（共同 `count = 10`），"
        "不把 SGBM 的 `count = 10` 与学习型的 `count = 11` 百分位直接排名。所有时间统计都带 "
        "`count / median_ms / p90_ms / p95_ms / min_ms / max_ms / warmup_excluded_count`。",
        "",
        "**延迟口径必须连带引用的三条限制**：",
        "",
        f"1. {summary['latency_composition_caveats'][0]}",
        f"2. {summary['latency_composition_caveats'][1]}",
        f"3. {summary['latency_composition_caveats'][3]}",
        "",
        "```text",
        log.text().strip(),
        "```",
        "",
        "## 8. 测试与保护",
        "",
        f"- 全仓 `python run_tests.py`：`{summary['tests']['repository']}`；",
        f"- `python -m py_compile realtime_app/tools/benchmark_temporal_ground_modules.py`：`{summary['tests']['py_compile']}`；",
        f"- task-01 目录快照：前后文件数 `{summary['upstream_snapshots']['task01']['difference']['files_before']}` / "
        f"`{summary['upstream_snapshots']['task01']['difference']['files_after']}`，未变化 `{summary['upstream_snapshots']['task01']['difference']['unchanged']}`；",
        f"- task-02 目录快照：前后文件数 `{summary['upstream_snapshots']['task02']['difference']['files_before']}` / "
        f"`{summary['upstream_snapshots']['task02']['difference']['files_after']}`，未变化 `{summary['upstream_snapshots']['task02']['difference']['unchanged']}`。",
        "",
        "## 9. 产物",
        "",
        "`flow_records.jsonl`、`flow_manual_anchor_evaluation.json`、`vo_records.jsonl`、`combination_matrix_27.json`、"
        "`combination_matrix_27.csv`、`summary.json`、`timing_summary.json`、`command.txt`、`run_stdout.txt`、`EXPERIMENT.md`、"
        "`FINAL_RECOMMENDATION.md`、`visualizations/`。",
        "",
        "## 10. 结论边界",
        "",
        MATRIX_BOUNDARY,
        "",
        "光流与 VO 的一切数字都是同一批图像、同一标定、同一 Mask2Former 候选条件下的内部一致性；"
        "不得报告真实地面精度、相机高度精度、足地高度、步态事件、接触或临床指标。",
        "",
    ])


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    thresholds = PropagationThresholds(
        maximum_forward_backward_error_px=args.maximum_forward_backward_error_px,
        minimum_candidate_flow_consistency=args.minimum_candidate_flow_consistency,
        maximum_candidate_photometric_median=args.maximum_candidate_photometric_median,
        minimum_candidate_pixels=args.minimum_candidate_pixels,
    )
    thresholds.validate()
    if args.maximum_flow_intervals < 0 or args.maximum_anchor_intervals < 0 or args.maximum_flow_steps < 0:
        raise ValueError("interval and step limits must be non-negative")

    log = RunLog()
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "visualizations").mkdir()
    log("task-03 temporal ground composition benchmark")
    log(f"output_dir: {args.output_dir}")

    upstream = Upstream(args.task01_dir, args.task02_dir)
    log(f"task-01 frame records: {len(upstream.task01_records)} | task-02 frame records: {len(upstream.task02_records)} "
        f"| task-02 combinations: {len(upstream.combinations)}")

    snapshot_before = {"task01": snapshot_tree(args.task01_dir), "task02": snapshot_tree(args.task02_dir)}
    log(f"upstream snapshot before: task-01 {len(snapshot_before['task01'])} files, task-02 {len(snapshot_before['task02'])} files")

    anchor_ids = list(ANCHOR_PAIR_IDS)
    intervals = list(zip(anchor_ids, anchor_ids[1:]))
    if args.maximum_anchor_intervals > 0:
        intervals = intervals[: args.maximum_anchor_intervals]
    flow_intervals = intervals if args.maximum_flow_intervals <= 0 else intervals[: args.maximum_flow_intervals]
    log(f"anchor intervals: {len(intervals)} (flow intervals: {len(flow_intervals)})")

    floor_mask_dirs = {"left": args.left_floor_mask_dir, "right": args.right_floor_mask_dir}
    floor_masks: dict[str, dict[int, np.ndarray]] = {"left": {}, "right": {}}
    person_pairs = {int(record["pair_id"]) for record in read_jsonl(args.person_region_jsonl)}
    walker_pairs: dict[str, set[int]] = {}
    for view in VIEWS:
        directory = args.walker_candidate_mask_dir / view
        walker_pairs[view] = {pair_id_from_name(path.name) for path in directory.glob("pair_*.png")} if directory.is_dir() else set()
    for view in VIEWS:
        for pair_id in anchor_ids:
            path = floor_mask_dirs[view] / frame_name(pair_id)
            if not path.is_file():
                raise RuntimeError(f"Mask2Former cached floor mask missing: {path}")
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"cannot read Mask2Former cached floor mask: {path}")
            floor_masks[view][pair_id] = mask > 0
    log(f"Mask2Former masks loaded for both views at {len(anchor_ids)} anchors; "
        f"PMPose person region records {len(person_pairs)}; unverified walker masks left/right "
        f"{len(walker_pairs['left'])}/{len(walker_pairs['right'])} frames")

    image_dirs = {"left": args.left_dir, "right": args.right_dir}

    def load_image(view: str, pair_id: int) -> np.ndarray:
        image = cv2.imread(str(image_dirs[view] / frame_name(pair_id)), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot read {view} image: {image_dirs[view] / frame_name(pair_id)}")
        return image

    # ---- 1) single-step flow records -----------------------------------------------
    flow_collector = TimingCollector(gpu_synchronize=False)
    flow_records: list[dict[str, Any]] = []
    flow_propagated: dict[tuple[int, int, str], np.ndarray] = {}
    for previous_id, current_id in flow_intervals:
        for view in VIEWS:
            record, propagated = interval_flow_record(
                view=view, previous_pair_id=previous_id, current_pair_id=current_id,
                previous_image=load_image(view, previous_id), current_image=load_image(view, current_id),
                previous_mask=floor_masks[view][previous_id], current_mask=floor_masks[view][current_id],
                thresholds=thresholds,
            )
            flow_records.append(record)
            flow_propagated[(previous_id, current_id, view)] = propagated
            for name in ("flow_field_ms", "flow_consistency_ms", "flow_decision_ms", "flow_runtime_ms"):
                if record.get(name) is not None:
                    flow_collector.record_ms(name, float(record[name]))
            log(f"  flow {frame_stem(previous_id)}->{frame_stem(current_id)} {view}: {record['flow_status']} "
                f"iou={None if record['propagated_vs_current_iou'] is None else round(record['propagated_vs_current_iou'], 4)} "
                f"ms={None if record['flow_runtime_ms'] is None else round(record['flow_runtime_ms'], 1)}")
    log(f"flow records: {len(flow_records)}")

    # ---- 2) 40-step manual anchor drift evaluation ---------------------------------
    phase_log: list[str] = []
    interval_records: list[dict[str, Any]] = []
    for previous_id, current_id in intervals:
        step_count = current_id - previous_id
        if args.maximum_flow_steps > 0:
            step_count = min(step_count, args.maximum_flow_steps)
        interval: dict[str, Any] = {
            "source_anchor_pair_id": previous_id, "target_anchor_pair_id": current_id,
            "propagation_step_count": step_count,
        }
        for view in VIEWS:
            manual_label_dir = args.manual_left_label_dir if view == "left" else args.manual_right_label_dir
            shape = floor_masks[view][previous_id].shape
            manual_seed, _ = manual_floor_mask(manual_label_dir / f"{frame_stem(previous_id)}.json", shape)
            record, propagated, flow_steps = evaluate_anchor_interval(
                view=view, source_pair_id=previous_id, target_pair_id=current_id,
                seed_masks={"manual_seed": manual_seed, "mask2former_seed": floor_masks[view][previous_id]},
                load_image=lambda pair_id, v=view: load_image(v, pair_id),
                step_count=step_count,
                thresholds=thresholds,
                target_loader=lambda v=view, p=current_id: manual_floor_mask(
                    (args.manual_left_label_dir if v == "left" else args.manual_right_label_dir) / f"{frame_stem(p)}.json",
                    floor_masks[v][p].shape,
                ),
                phase_log=phase_log,
            )
            for step in flow_steps:
                flow_collector.record_ms("flow_field_ms", float(step["flow_field_ms"]))
            current_mask = floor_masks[view][current_id]
            valid = np.ones(shape, dtype=bool)
            metrics = binary_metrics(current_mask, valid, propagated["mask2former_seed"])
            record["production_compatible_mask2former_target"] = {
                "one_anchor_to_next_anchor_iou": float(metrics["iou"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "reference": "current anchor Mask2Former candidate mask (a candidate, not truth)",
            }
            interval[view] = record
            log(f"  anchor {frame_stem(previous_id)}->{frame_stem(current_id)} {view}: steps={step_count} "
                f"manual-seed IoU={record['seeds']['manual_seed']['against_manual_target']['one_anchor_to_next_anchor_iou']:.4f} "
                f"production IoU={metrics['iou']:.4f}")
        interval_records.append(interval)

    def macro_manual(seed_key: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for view in VIEWS:
            rows = [interval[view]["seeds"][seed_key]["against_manual_target"] for interval in interval_records]
            result[view] = {
                key: float(np.mean([row[key] for row in rows]))
                for key in ("one_anchor_to_next_anchor_iou", "precision", "recall",
                            "false_positive_on_non_floor", "false_negative_on_floor", "mask_area_change")
            }
            result[view]["interval_count"] = len(rows)
        return result

    def macro_production() -> dict[str, Any]:
        result: dict[str, Any] = {}
        for view in VIEWS:
            rows = [interval[view]["production_compatible_mask2former_target"] for interval in interval_records]
            result[view] = {
                key: float(np.mean([row[key] for row in rows]))
                for key in ("one_anchor_to_next_anchor_iou", "precision", "recall")
            }
            result[view]["interval_count"] = len(rows)
        return result

    flow_evaluation = {
        "schema_version": SCHEMA_VERSION + "_flow_anchor_evaluation_v1",
        "anchor_pair_ids": anchor_ids,
        "intervals": [[source, target] for source, target in intervals],
        "propagation_steps_per_interval": "target_pair_id - source_pair_id (40 for every manual anchor interval)",
        "views_are_independent": True,
        "left_mask_never_copied_to_right": True,
        "thresholds": {
            "maximum_forward_backward_error_px": thresholds.maximum_forward_backward_error_px,
            "minimum_candidate_flow_consistency": thresholds.minimum_candidate_flow_consistency,
            "maximum_candidate_photometric_median": thresholds.maximum_candidate_photometric_median,
            "minimum_candidate_pixels": thresholds.minimum_candidate_pixels,
        },
        "target_label_guard": {
            "target_label_read_phase": "after_propagation",
            "target_mask_used_in_propagation": False,
            "target_mask_fed_back_to_intermediate_frames": False,
            "phase_order_observed": phase_log[:12],
            "note": (
                "propagate_chains_locked() cannot receive a target mask: the target label is loaded once, after "
                "every chain of the interval has been produced, and is used only for the final comparison. No "
                "label was used for training."
            ),
        },
        "none_of_this_is_training_data": True,
        "per_interval": interval_records,
        "macro_metrics": {
            "manual_seed_vs_manual_target": macro_manual("manual_seed"),
            "mask2former_seed_vs_manual_target": macro_manual("mask2former_seed"),
            "production_compatible_mask2former_seed_vs_mask2former_target": macro_production(),
        },
        "drift_trend": {
            view: {
                "propagated_area_fraction_trend_per_step": mean_or_none([
                    interval[view]["seeds"]["manual_seed"]["step_diagnostics"]["propagated_area_fraction_trend_per_step"]
                    for interval in interval_records
                ]),
                "flow_consistency_trend_per_step": mean_or_none([
                    interval[view]["seeds"]["manual_seed"]["step_diagnostics"]["flow_consistency_trend_per_step"]
                    for interval in interval_records
                ]),
                "propagated_area_fraction_first_quarter_mean": mean_or_none([
                    interval[view]["seeds"]["manual_seed"]["step_diagnostics"]["propagated_area_fraction_first_quarter_mean"]
                    for interval in interval_records
                ]),
                "propagated_area_fraction_last_quarter_mean": mean_or_none([
                    interval[view]["seeds"]["manual_seed"]["step_diagnostics"]["propagated_area_fraction_last_quarter_mean"]
                    for interval in interval_records
                ]),
            }
            for view in VIEWS
        },
        "interpretation_boundary": (
            "Image-space optical-flow propagation drift against a 2-D manual identity audit at the interval end. "
            "It is not stereo correspondence and not a metric ground plane, and the endpoint label was never fed "
            "back into the propagation."
        ),
    }
    (args.output_dir / "flow_manual_anchor_evaluation.json").write_text(
        json.dumps(flow_evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    paired: dict[tuple[int, int], dict[str, float]] = {}
    for record in flow_records:
        if record["flow_runtime_ms"] is not None:
            paired.setdefault((record["previous_pair_id"], record["current_pair_id"]), {})[record["view"]] = float(record["flow_runtime_ms"])
    two_view = [entry["left"] + entry["right"] for entry in paired.values() if len(entry) == 2]
    flow_summary = {
        "record_count": len(flow_records),
        "interval_count": len(flow_intervals),
        "available_records": sum(1 for record in flow_records if record["flow_status"] == "available"),
        "unavailable_records": sum(1 for record in flow_records if record["flow_status"] == "unavailable"),
        "reason_counts": dict(Counter(reason for record in flow_records for reason in record["reasons"])),
        "warmup_excluded_count": UPSTREAM_WARMUP_EXCLUDED,
        "left": {
            name: distribution([record[name] for record in flow_records if record["view"] == "left"])
            for name in ("propagated_vs_current_iou", "forward_backward_consistency_fraction", "new_area_fraction", "lost_area_fraction")
        },
        "right": {
            name: distribution([record[name] for record in flow_records if record["view"] == "right"])
            for name in ("propagated_vs_current_iou", "forward_backward_consistency_fraction", "new_area_fraction", "lost_area_fraction")
        },
        "two_view_module_ms": timing_statistics(two_view, warmup_excluded_count=UPSTREAM_WARMUP_EXCLUDED),
        "timing_modules": {
            name: timing_statistics(flow_collector.samples_ms(name), warmup_excluded_count=UPSTREAM_WARMUP_EXCLUDED)
            for name in ("flow_field_ms", "flow_consistency_ms", "flow_decision_ms", "flow_runtime_ms")
        },
        "temporal_mode": "flow_assisted_current_frame",
        "semantics": FLOW_ASSISTED_SEMANTICS,
    }
    flow_summary["timing_modules"]["flow_two_view_combined_ms"] = flow_summary["two_view_module_ms"]
    with (args.output_dir / "flow_records.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in flow_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---- 3) VO module (fail closed) ------------------------------------------------
    vo_per_step: list[float] = []

    def domain_certificate(first: int, second: int) -> dict[str, Any]:
        return static_background_domain_certificate(
            floor_masks_available=first in floor_masks["left"] and second in floor_masks["left"],
            person_regions_available=first in person_pairs and second in person_pairs,
            walker_regions_available=first in walker_pairs["left"] and second in walker_pairs["left"],
        )

    def pose_provider(from_id: int, to_id: int) -> tuple[None, str]:
        # The frozen archives carry no per-frame metric static-background stereo for the
        # 440 intermediate frames, and manufacturing it would require re-running a stereo
        # matcher, which this task forbids.  Fail closed.
        return None, "no_per_frame_static_background_metric_correspondences_in_frozen_archives"

    vo_records: list[dict[str, Any]] = []
    for matcher in MATCHERS:
        for plane_method in PLANE_METHODS:
            for source_id, target_id in intervals:
                vo_records.append(evaluate_vo_interval(
                    matcher=matcher, plane_method=plane_method,
                    source_pair_id=source_id, target_pair_id=target_id,
                    source_plane=upstream.direct_plane(matcher, plane_method, source_id),
                    source_anchor_cross_region_status=upstream.cross_region_status(matcher, source_id, plane_method),
                    target_anchor_cross_region_status=upstream.cross_region_status(matcher, target_id, plane_method),
                    target_plane=upstream.direct_plane(matcher, plane_method, target_id),
                    relative_pose_provider=pose_provider,
                    domain_certificate=domain_certificate,
                    runtime_recorder=lambda a, b, ms: vo_per_step.append(float(ms)),
                ))
    log(f"vo records: {len(vo_records)}; unavailable={sum(1 for r in vo_records if r['vo_status'] == 'unavailable')}")
    with (args.output_dir / "vo_records.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in vo_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    vo_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in vo_records:
        vo_by_key.setdefault((record["matcher"], record["plane_method"]), []).append(record)
    vo_summary = summarize_vo_records(vo_records)
    vo_summary["runtime_ms_per_step"] = timing_statistics(vo_per_step)
    full_chains = [
        sum(vo_per_step[index:index + 40]) for index in range(0, max(0, len(vo_per_step) - 39), 40)
    ] if vo_per_step else []
    vo_summary["runtime_ms_40_step_total"] = timing_statistics(full_chains)
    if not full_chains:
        vo_summary["runtime_ms_40_step_total"]["unavailable_reason"] = (
            "no certified static-background pose chain exists, so no propagation step was executed"
        )

    # ---- 4) the fixed 27-combination matrix ----------------------------------------
    matrix = build_combination_matrix(
        upstream, flow_summary=flow_summary, vo_by_key=vo_by_key,
        vo_runtime_stats=vo_summary["runtime_ms_per_step"],
        declared_frame_budget_ms=DECLARED_FRAME_BUDGET_MS,
    )
    (args.output_dir / "combination_matrix_27.json").write_text(
        json.dumps({
            "schema_version": SCHEMA_VERSION + "_combination_matrix_v1",
            "row_count": len(matrix),
            "matchers": list(MATCHERS),
            "plane_methods": list(PLANE_METHODS),
            "temporal_modes": list(TEMPORAL_MODES),
            "semantic_model": ONLINE_SEMANTIC_MODEL,
            "declared_frame_budget_ms": DECLARED_FRAME_BUDGET_MS,
            "latency_caveat": LATENCY_SUM_CAVEAT,
            "rows": matrix,
            "interpretation_boundary": MATRIX_BOUNDARY,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "combination_matrix_27.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "combination_id", "matcher", "plane_method", "temporal_mode", "full_chain_measured", "measurement_scope",
            "estimated_end_to_end_latency_ms", "measured_end_to_end_latency_ms", "realtime_compatible",
            "cross_region_pass_frames", "median_inlier_fraction", "median_residual_mm", "flow_module_status", "vo_module_status",
        ])
        for row in matrix:
            writer.writerow([
                row["combination_id"], row["matcher"], row["plane_method"], row["temporal_mode"],
                row["full_chain_measured"], row["measurement_scope"],
                "" if row["estimated_end_to_end_latency_ms"] is None else f"{row['estimated_end_to_end_latency_ms']:.3f}",
                "", row["realtime_compatible"], row["cross_region_evidence"]["pass_frames"],
                row["plane_evidence"]["median_inlier_fraction"], row["plane_evidence"]["median_residual_mm"],
                row["flow_module_status"], row["vo_module_status"],
            ])

    # ---- 5) timing summary ---------------------------------------------------------
    def upstream_module_row(matcher: str, plane_method: str) -> dict[str, Any]:
        timing = upstream.comparable_timing(matcher, plane_method)
        return {
            "matcher": matcher,
            "plane_method": plane_method,
            "frame_set": "timing_on_task01_comparable_frames (common count = 10)",
            "modules": {
                name: {**value, "warmup_excluded_count": UPSTREAM_WARMUP_EXCLUDED}
                for name, value in timing.items() if isinstance(value, dict) and "median_ms" in value
            },
        }

    upstream_rows = [upstream_module_row(matcher, plane_method) for matcher in MATCHERS for plane_method in PLANE_METHODS]
    timing_summary = {
        "schema_version": SCHEMA_VERSION + "_timing_v1",
        "measurement_unit": "ms",
        "clock": "time.perf_counter",
        "gpu_synchronized": False,
        "gpu_synchronization_note": (
            "This process performs no GPU work and deliberately does not import torch: on this host the Anaconda "
            "MKL NumPy and torch cannot share one process (OMP Error #15), and the unsafe KMP_DUPLICATE_LIB_OK "
            "workaround is not used. GPU forward and worker-wall numbers are quoted from the archived task-02 records."
        ),
        "required_statistics_fields": list(REQUIRED_TIMING_FIELDS),
        "upstream_warmup_excluded_count": UPSTREAM_WARMUP_EXCLUDED,
        "comparable_frame_rule": (
            "matcher speed comparisons use only task-02 timing_on_task01_comparable_frames; SGBM count=10 and "
            "learned count=11 percentiles are never ranked against each other"
        ),
        "sgbm_cpu_spatial_chain": [row for row in upstream_rows if row["matcher"] == "sgbm"],
        "learned_gpu_forward_and_worker_wall": [row for row in upstream_rows if row["matcher"] != "sgbm"],
        "learned_model_overheads": {
            "igev_model_load_ms_median": upstream.combination("igev", "dense_ransac")["timing"].get("model_load_ms", {}).get("median_ms"),
            "dynamicstereo_model_load_ms_median": upstream.combination("dynamicstereo", "dense_ransac")["timing"].get("model_load_ms", {}).get("median_ms"),
            "dynamicstereo_offline_window": upstream.task02_summary["dynamicstereo_offline_window"],
            "dynamicstereo_offline_latency": upstream.task02_summary["dynamicstereo_offline_latency"],
            "dynamicstereo_offline_latency_note": (
                "four future frames are waited for; the five-frame window throughput is not a single-frame "
                "real-time latency and this configuration must not be reported as real time"
            ),
        },
        "optical_flow": flow_summary["timing_modules"],
        "optical_flow_note": (
            "flow_field_ms is the two Farneback passes of one view of one frame pair; flow_consistency_ms is the "
            "warp/consistency/photometric judgment; flow_decision_ms is the frozen accept/reject rule; "
            "flow_two_view_combined_ms is left + right for the same pair"
        ),
        "vo": {
            "per_adjacent_pose_step_ms": vo_summary["runtime_ms_per_step"],
            "planar_propagation_step_ms": timing_statistics([], warmup_excluded_count=0),
            "forty_step_total_ms": vo_summary["runtime_ms_40_step_total"],
            "unavailable_reason": vo_summary["runtime_ms_40_step_total"].get("unavailable_reason"),
            "note": (
                "the per-step field times the fail-closed provider call, i.e. the certificate check that refuses "
                "to produce a pose; it is not a completed VO step cost"
            ),
        },
        "declared_frame_budget_ms": DECLARED_FRAME_BUDGET_MS,
        "latency_composition_caveats": [
            "mask_load_ms reads the cached Mask2Former candidate mask PNG; the semantic model forward pass was "
            "computed once before task-01 and its cost is inside no number in this directory, so every direct-chain "
            "latency estimate excludes the Mask2Former inference.",
            "the learned matchers need two passes (forward plus the mirrored reverse field) and both are included in "
            "the upstream estimated_direct_pipeline_ms; SGBM already contains both disparity passes in rectification_ms.",
            "process-isolated image exchange, model construction and checkpoint loading are excluded from the "
            "upstream estimate, matching the task-02 timing formula.",
        ],
        "estimated_vs_measured": {
            "estimated_end_to_end_latency_ms": "module-time sum, labelled estimated in every row",
            "measured_end_to_end_latency_ms": "null for every row; no end-to-end wall clock was taken",
            "caveat": LATENCY_SUM_CAVEAT,
        },
        "interpretation_boundary": "Wall-clock module measurements on this machine. They are not a real-time, field or clinical claim.",
    }
    (args.output_dir / "timing_summary.json").write_text(
        json.dumps(timing_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # ---- 6) visualizations ---------------------------------------------------------
    if not args.no_visualization:
        for record in flow_records:
            propagated = flow_propagated.get(
                (record["previous_pair_id"], record["current_pair_id"], record["view"]),
                np.zeros(load_image(record["view"], record["current_pair_id"]).shape[:2], dtype=bool),
            )
            sheet = render_flow_panel(
                previous_image=load_image(record["view"], record["previous_pair_id"]),
                current_image=load_image(record["view"], record["current_pair_id"]),
                previous_mask=floor_masks[record["view"]][record["previous_pair_id"]],
                current_mask=floor_masks[record["view"]][record["current_pair_id"]],
                propagated=propagated,
                title=(f"{frame_stem(record['previous_pair_id'])}->{frame_stem(record['current_pair_id'])} {record['view']} "
                       f"IoU={record['propagated_vs_current_iou']:.3f} consistency={record['forward_backward_consistency_fraction']:.3f} "
                       f"({record['flow_status']})"),
            )
            cv2.imwrite(str(args.output_dir / "visualizations" /
                            f"flow_{frame_stem(record['previous_pair_id'])}_{frame_stem(record['current_pair_id'])}_{record['view']}.png"), sheet)
        for interval in interval_records:
            if interval["propagation_step_count"] < 2:
                continue
            cv2.imwrite(
                str(args.output_dir / "visualizations" /
                    f"drift_{frame_stem(interval['source_anchor_pair_id'])}_{frame_stem(interval['target_anchor_pair_id'])}.png"),
                render_drift_curve(
                    interval,
                    f"{frame_stem(interval['source_anchor_pair_id'])} -> {frame_stem(interval['target_anchor_pair_id'])}: "
                    f"{interval['propagation_step_count']}-step flow drift",
                ),
            )
        cv2.imwrite(str(args.output_dir / "visualizations" / "combination_matrix_27_status.png"), render_matrix_status(matrix))
        log("visualizations written")

    # ---- 7) snapshots, summary, reports -------------------------------------------
    snapshot_after = {"task01": snapshot_tree(args.task01_dir), "task02": snapshot_tree(args.task02_dir)}
    upstream_snapshots = {
        key: {
            "root": str((args.task01_dir if key == "task01" else args.task02_dir).resolve()),
            "difference": snapshot_difference(snapshot_before[key], snapshot_after[key]),
        }
        for key in ("task01", "task02")
    }
    for key in ("task01", "task02"):
        if not upstream_snapshots[key]["difference"]["unchanged"]:
            raise RuntimeError(f"upstream {key} directory changed during the run: {upstream_snapshots[key]['difference']}")
    log(f"upstream snapshot after: task-01 unchanged={upstream_snapshots['task01']['difference']['unchanged']}, "
        f"task-02 unchanged={upstream_snapshots['task02']['difference']['unchanged']}")

    dynamic = [c for c in upstream.task02_summary["combinations"]
               if c["matcher"] == "dynamicstereo" and c["semantic_source"] == UPSTREAM_SEMANTIC_SOURCE]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_temporal_modules_and_composition_matrix",
        "task": "task-03",
        "date": "2026-09-13",
        "semantic_model_online": ONLINE_SEMANTIC_MODEL,
        "semantic_source_rule": (
            "only the Mask2Former online ground candidate is used for the final matrix; manual floor masks are 2-D "
            "identity audit only and are never an online temporal input"
        ),
        "anchor_pair_ids": anchor_ids,
        "anchor_intervals": [[source, target] for source, target in intervals],
        "flow_summary": flow_summary,
        "vo_summary": vo_summary,
        "matrix_summary": {
            "row_count": len(matrix),
            "unique_combination_ids": len({row["combination_id"] for row in matrix}),
            "full_chain_measured_true": sum(1 for row in matrix if row["full_chain_measured"]),
            "full_chain_measured_false": sum(1 for row in matrix if not row["full_chain_measured"]),
            "realtime_compatible_true": sum(1 for row in matrix if row["realtime_compatible"]),
            "realtime_compatible_false": sum(1 for row in matrix if not row["realtime_compatible"]),
            "temporal_mode_counts": dict(Counter(row["temporal_mode"] for row in matrix)),
            "matcher_counts": dict(Counter(row["matcher"] for row in matrix)),
            "plane_method_counts": dict(Counter(row["plane_method"] for row in matrix)),
            "flow_module_status_counts": dict(Counter(row["flow_module_status"] for row in matrix)),
            "vo_module_status_counts": dict(Counter(row["vo_module_status"] for row in matrix)),
            "measurement_scope_counts": dict(Counter(row["measurement_scope"] for row in matrix)),
            "measured_end_to_end_latency_rows": sum(1 for row in matrix if row["measured_end_to_end_latency_ms"] is not None),
        },
        "upstream_snapshots": upstream_snapshots,
        "upstream_facts": {
            "dynamicstereo_cross_region_pass_frames": dynamic[0]["cross_region_evidence"]["pass_frames"] if dynamic else None,
            "dynamicstereo_cross_region_fail_frames": dynamic[0]["cross_region_evidence"]["fail_frames"] if dynamic else None,
            "dynamicstereo_offline_window": upstream.task02_summary["dynamicstereo_offline_window"],
            "dynamicstereo_offline_latency": upstream.task02_summary["dynamicstereo_offline_latency"],
            "upstream_unavailable_combinations": upstream.task02_summary["unavailable_combinations"],
            "sgbm_reference_rows_read_from": str(args.task01_dir.resolve()),
            "learned_rows_read_from": str(args.task02_dir.resolve()),
        },
        "declared_frame_budget_ms": DECLARED_FRAME_BUDGET_MS,
        "declared_frame_budget_derivation": (
            f"the source capture's effective paired rate is {SOURCE_EFFECTIVE_PAIR_FPS} pairs/s, so one pair is "
            f"{DECLARED_FRAME_BUDGET_MS} ms"
        ),
        "estimated_vs_measured": {
            "estimated_end_to_end_latency_ms": "module-time sum, always labelled estimated",
            "measured_end_to_end_latency_ms": "null for every row; no end-to-end wall clock was taken",
            "caveat": LATENCY_SUM_CAVEAT,
        },
        "latency_composition_caveats": [
            "mask_load_ms reads the cached Mask2Former candidate mask PNG; the semantic model forward pass was "
            "computed once before task-01 and its cost is inside no number in this directory, so every direct-chain "
            "latency estimate excludes the Mask2Former inference.",
            "the learned matchers need two passes (forward plus the mirrored reverse field) and both are included in "
            "the upstream estimated_direct_pipeline_ms; SGBM already contains both disparity passes in rectification_ms.",
            "process-isolated image exchange, model construction and checkpoint loading are excluded from the "
            "upstream estimate, matching the task-02 timing formula.",
            "the optical-flow module was measured on this host at full upright resolution with the frozen Farneback "
            "settings; its cost is data dependent and is reported as measured samples, not as a model of cost.",
        ],
        "tests": {"repository": args.repository_test_result, "py_compile": args.py_compile_result},
        "input_inventory": {
            "left_image_dir": str(args.left_dir.resolve()),
            "right_image_dir": str(args.right_dir.resolve()),
            "left_floor_mask_dir": str(args.left_floor_mask_dir.resolve()),
            "right_floor_mask_dir": str(args.right_floor_mask_dir.resolve()),
            "manual_left_label_dir": str(args.manual_left_label_dir.resolve()),
            "manual_right_label_dir": str(args.manual_right_label_dir.resolve()),
            "task01_dir": str(args.task01_dir.resolve()),
            "task02_dir": str(args.task02_dir.resolve()),
            "person_region_records": len(person_pairs),
            "walker_candidate_mask_pairs": {view: len(walker_pairs[view]) for view in VIEWS},
        },
        "interpretation_boundary": MATRIX_BOUNDARY,
    }

    (args.output_dir / "FINAL_RECOMMENDATION.md").write_text(
        render_recommendation(
            matrix=matrix, flow_evaluation=flow_evaluation, vo_summary=vo_summary,
            upstream=upstream, declared_frame_budget_ms=DECLARED_FRAME_BUDGET_MS,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "command.txt").write_text(
        "python .\\realtime_app\\tools\\benchmark_temporal_ground_modules.py\n"
        f"# fixed defaults: --output-dir \"{args.output_dir}\"\n"
        f"# recorded test results: repository=\"{args.repository_test_result}\" py_compile=\"{args.py_compile_result}\"\n",
        encoding="utf-8",
    )
    (args.output_dir / "EXPERIMENT.md").write_text(
        render_experiment_markdown(args=args, summary=summary, flow_evaluation=flow_evaluation, matrix=matrix, log=log),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "run_stdout.txt").write_text(log.text(), encoding="utf-8")
    log(f"summary written: {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
