#!/usr/bin/env python3
"""Compare ground and walker reconstruction strategies on the same stereo pairs.

The tool uses externally supplied LabelMe semantics as a controlled oracle for
semantic identity, then varies only the *geometric representation*:

Ground: dense RANSAC, sparse spatial tiles, soft weighted IRLS, and independent
image-region consensus.

Walker: dense visible point cloud, connected-component line/tube primitives,
and camera-attached multi-frame voxel consensus.  A measured sparse-template
fit is implemented in :mod:`pose_app.scene_geometry_variants`, but is reported
as unavailable here because the current labels do not contain named 3-D walker
landmarks or a measured configuration.

Optional cached automatic masks may be supplied as additional candidate-only
sources.  They are never upgraded to manually audited semantic evidence.

Besides the geometry, every frame/source/method output now carries two additions
that change no estimator, threshold or existing field:

* a module-level CPU wall-clock timing record measured with
  :class:`pose_app.benchmark_timing.TimingCollector` (see ``TIMING_MODULES``);
* a four-layer confidence *evidence* record (semantic / stereo / plane /
  cross-region) that keeps the layers separate and never collapses them into a
  pseudo-probability.  Mask2Former output stays a candidate even when a plane is
  fitted on it, and a low RANSAC residual is not labelled real ground.
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
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import cv2
import numpy as np


TOOLS_ROOT = (Path(__file__).resolve().parents[2] / "tools")
PROJECT_ROOT = TOOLS_ROOT.parents[1]
REALTIME_ROOT = PROJECT_ROOT / "realtime_app"
for candidate in (str(REALTIME_ROOT), str(TOOLS_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from pose_app.benchmark_timing import TimingCollector  # noqa: E402
from pose_app.calibration import StereoCalibration  # noqa: E402
from pose_app.scene_geometry_variants import (  # noqa: E402
    angular_error_degrees,
    camera_attached_temporal_consensus,
    estimate_dense_ransac,
    estimate_region_consensus,
    estimate_sparse_tiles,
    estimate_weighted_irls,
    reconstruct_walker_components,
)
from walker_tools.scene.audit_manual_floor_labels import load_labelme, rasterize_labelme  # noqa: E402
import walker_tools.scene.diagnose_floor_mask_stereo_correspondence as diagnostic  # noqa: E402
import walker_tools.scene.observe_local_ground_semantic_stereo as strict  # noqa: E402
import walker_tools.scene.observe_local_ground_semantic_stereo_lr_direction_control as corrected  # noqa: E402


SCHEMA_VERSION = "ground_walker_reconstruction_benchmark_v1"
METHODS = ("dense_ransac", "sparse_tile_ransac", "soft_weighted_irls", "region_consensus")

# The three plane fitters that are ranked by time.  ``region_consensus`` is not
# a fourth main fitter: it is preserved as the shared quality-rejection gate and
# is therefore excluded from ``timing_by_source_and_method``.
MAIN_METHODS = ("dense_ransac", "sparse_tile_ransac", "soft_weighted_irls")
REGION_METHOD = "region_consensus"

TIMING_MODULES = (
    "mask_load_ms",
    "rectification_and_disparity_ms",
    "stereo_candidate_filter_ms",
    "triangulation_ms",
    "plane_fit_ms",
    "region_consensus_ms",
    "evidence_ms",
    "visualization_ms",
    "total_cpu_wall_ms",
)

# Additional module that exists only because the unified evidence record needs
# per-method coverage and reprojection percentiles.  It is reported separately
# instead of being hidden inside the frozen estimator time.
EVIDENCE_MODULE = "evidence_ms"

CONFIDENCE_BOUNDARY = (
    "Internal semantic/stereo/plane evidence only; not physical ground accuracy."
)

WARMUP_FRAMES_EXCLUDED = 2

# Copied verbatim from ``pose_app.scene_geometry_variants.estimate_region_consensus``
# so the cross-region evidence cannot be produced with a different threshold.
REGION_MINIMUM_POINTS = 30
REGION_MAXIMUM_ANGLE_DEG = 5.0
REGION_ITERATIONS = 300

REGION_GATE_NOTE = (
    "Shared quality-rejection gate only. It is reported for every method of the "
    "same frame from the same stereo candidate set and is excluded from the "
    "per-method speed ranking."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-dir", type=Path, required=True)
    parser.add_argument("--right-dir", type=Path, required=True)
    parser.add_argument("--left-label-dir", type=Path, required=True)
    parser.add_argument("--right-label-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--frozen-baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--auto-floor-left-dir", type=Path)
    parser.add_argument("--auto-floor-right-dir", type=Path)
    parser.add_argument("--auto-floor-name", default="automatic_floor_candidate")
    parser.add_argument("--maximum-pairs", type=int, default=0)
    parser.add_argument("--no-visualization", action="store_true")
    return parser.parse_args()


def pair_id(path: Path) -> int:
    return int(path.stem.removeprefix("pair_"))


def selected_names(args: argparse.Namespace) -> list[str]:
    left = {path.name for path in args.left_label_dir.glob("pair_*.json")}
    right = {path.name for path in args.right_label_dir.glob("pair_*.json")}
    names = sorted(left & right, key=lambda name: pair_id(Path(name)))
    if args.maximum_pairs > 0:
        names = names[: args.maximum_pairs]
    return [Path(name).with_suffix(".png").name for name in names]


def load_manual_masks(label_path: Path, image_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    masks = rasterize_labelme(load_labelme(label_path))
    if masks["floor_eligible"].shape != image_shape:
        raise ValueError(f"{label_path}: label/image dimensions differ")
    return {key: value.astype(np.uint8) * 255 for key, value in masks.items()}


def load_binary(path: Path, shape: tuple[int, int]) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise RuntimeError(f"cannot read mask {path}")
    if value.shape != shape:
        value = cv2.resize(value, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (value > 0).astype(np.uint8) * 255


def describe_hardware() -> tuple[dict[str, Any], str]:
    """Machine description for the timing protocol, without changing any result.

    ``torch`` is deliberately not imported here.  This host ships an Anaconda
    MKL NumPy that loads its own ``libiomp5md.dll`` while the locally installed
    ``torch`` loads a second copy; importing torch makes the next NumPy
    linear-algebra call (used by the plane estimators) abort the process with
    ``OMP: Error #15``.  This pipeline performs no GPU work, so no CUDA
    synchronisation is needed and every measurement stays CPU wall-clock.
    """
    cpu = platform.processor() or platform.machine() or "unknown_cpu"
    cores = os.cpu_count()
    note = (
        f"CPU wall-clock timing on {cpu} ({cores} logical processors), "
        f"{platform.system()} {platform.release()}, Python {platform.python_version()}. "
        "gpu_synchronized=false: every measured module is pure CPU OpenCV/NumPy work and no GPU kernel is "
        "launched, so a CUDA synchronisation would not change these numbers. torch is deliberately not "
        "imported into this benchmark process because on this host the Anaconda MKL NumPy and torch cannot "
        "share one process: they load two copies of libiomp5md.dll and the second OpenMP initialisation "
        "aborts the process (OMP Error #15) as soon as the plane estimators call NumPy linear algebra. The "
        "unsafe KMP_DUPLICATE_LIB_OK=TRUE workaround is not used. The real torch.cuda.is_available() probe "
        "lives in pose_app.benchmark_timing.cuda_is_available() and is exercised by "
        "tests/test_benchmark_timing.py in a separate interpreter."
    )
    return (
        {
            "processor": cpu,
            "logical_processors": cores,
            "platform": f"{platform.system()} {platform.release()}",
            "python_version": platform.python_version(),
            "cuda_available": "not_probed_torch_not_imported",
            "cuda_synchronization": "disabled_by_policy_cpu_only_pipeline",
        },
        note,
    )


def reprojection_percentiles(
    points_left: np.ndarray,
    left_pixels: np.ndarray,
    right_pixels: np.ndarray,
    rectification: Any,
    calibration: StereoCalibration,
) -> dict[str, float | None]:
    """Per-point worst-side fisheye reprojection error, median and P95.

    Mirrors ``observe_local_ground_semantic_stereo.reprojection_medians`` (same
    expected local map lookups, same calibrated fisheye projections) and adds a
    P95 so the stereo evidence is not limited to a single percentile.
    """
    empty = {
        "median_reprojection_px": None,
        "p95_reprojection_px": None,
        "median_reprojection_left_px": None,
        "median_reprojection_right_px": None,
    }
    if len(points_left) == 0:
        return empty
    left_expected = np.column_stack((
        rectification.left_map_x[left_pixels[:, 1], left_pixels[:, 0]],
        rectification.left_map_y[left_pixels[:, 1], left_pixels[:, 0]],
    ))
    right_expected = np.column_stack((
        rectification.right_map_x[right_pixels[:, 1], right_pixels[:, 0]],
        rectification.right_map_y[right_pixels[:, 1], right_pixels[:, 0]],
    ))
    left_error = np.linalg.norm(calibration.project_left(points_left) - left_expected, axis=1)
    right_error = np.linalg.norm(calibration.project_right(points_left) - right_expected, axis=1)
    worst = np.maximum(left_error, right_error)
    return {
        "median_reprojection_px": float(np.median(worst)),
        "p95_reprojection_px": float(np.percentile(worst, 95)),
        "median_reprojection_left_px": float(np.median(left_error)),
        "median_reprojection_right_px": float(np.median(right_error)),
    }


def region_consensus_evidence(
    points: np.ndarray,
    pixels: np.ndarray,
    image_shape: tuple[int, int],
    distance_threshold_mm: float,
) -> dict[str, Any]:
    """Re-derive the per-region spread of the shared region-consensus gate.

    The adaptive split rule, the per-region estimator, its seed and its
    iteration count are copied verbatim from
    ``pose_app.scene_geometry_variants.estimate_region_consensus`` so this
    evidence cannot be produced with a different threshold.  The result is
    descriptive only: it selects nothing, rejects nothing by itself and is never
    used as a fourth main fitter.
    """
    points = np.asarray(points, dtype=np.float64)
    pixels = np.asarray(pixels, dtype=np.float64)
    if len(points) == 0:
        return {
            "status": "unavailable",
            "region_count": 0,
            "normal_spread_deg": None,
            "offset_spread_mm": None,
            "reasons": ["no_stereo_candidates"],
        }
    split_x = float(np.median(pixels[:, 0]))
    split_y = float(np.median(pixels[:, 1]))
    region_ids = (pixels[:, 0] >= split_x).astype(np.int32) + 2 * (pixels[:, 1] >= split_y).astype(np.int32)
    regional: list[tuple[np.ndarray, float]] = []
    for region_id in range(4):
        indices = np.flatnonzero(region_ids == region_id)
        if len(indices) < REGION_MINIMUM_POINTS:
            continue
        estimate = estimate_dense_ransac(
            points[indices], distance_threshold_mm=distance_threshold_mm,
            iterations=REGION_ITERATIONS, seed=731 + region_id, minimum_points=REGION_MINIMUM_POINTS,
        )
        if estimate.normal_left_camera is None or estimate.offset_mm is None:
            continue
        regional.append((estimate.normal_left_camera, float(estimate.offset_mm)))
    if len(regional) < 2:
        return {
            "status": "unavailable",
            "region_count": int(len(regional)),
            "normal_spread_deg": None,
            "offset_spread_mm": None,
            "reasons": ["fewer_than_two_independent_regions"],
        }
    angles = [
        angular_error_degrees(first[0], second[0])
        for index, first in enumerate(regional) for second in regional[index + 1:]
    ]
    median_angle = float(np.median(angles))
    reference = regional[0][0]
    aligned_normals = [normal if float(normal @ reference) >= 0 else -normal for normal, _ in regional]
    aligned_offsets = [
        offset if float(normal @ aligned) >= 0 else -offset
        for (normal, offset), aligned in zip(regional, aligned_normals)
    ]
    consensus_normal = np.mean(np.stack(aligned_normals), axis=0)
    consensus_normal /= np.linalg.norm(consensus_normal)
    consensus_offset = float(np.median(aligned_offsets))
    if consensus_offset < 0.0:
        consensus_normal, consensus_offset = -consensus_normal, -consensus_offset
    coarse = np.abs(points @ consensus_normal + consensus_offset) <= 2.0 * distance_threshold_mm
    support = int(coarse.sum())
    spread = {
        "region_count": int(len(regional)),
        "normal_spread_deg": median_angle,
        "offset_spread_mm": float(max(aligned_offsets) - min(aligned_offsets)),
        "consensus_coarse_support": support,
    }
    if median_angle > REGION_MAXIMUM_ANGLE_DEG:
        return {**spread, "status": "fail", "reasons": ["regional_planes_disagree"]}
    if support < REGION_MINIMUM_POINTS:
        return {**spread, "status": "fail", "reasons": ["insufficient_consensus_support"]}
    return {**spread, "status": "pass", "reasons": []}


def stereo_evidence(candidate: dict[str, Any]) -> dict[str, Any]:
    """Layer-2 evidence for the shared stereo candidate set of one frame/source."""
    funnel = candidate.get("matching_funnel", {}) or {}
    mask_pixels = int(funnel.get("left_mask_pixels", 0) or 0)
    points = candidate.get("points", np.empty((0, 3)))
    rectified = candidate.get("rectified_xyz")
    positive_depth_fraction = None
    if rectified is not None and len(rectified):
        positive_depth_fraction = float(np.mean(np.asarray(rectified)[:, 2] > 0.0))
    def fraction(key: str) -> float | None:
        if mask_pixels <= 0:
            return None
        return float(int(funnel.get(key, 0) or 0) / mask_pixels)
    return {
        "candidate_pixels": int(len(points)),
        "forward_disparity_valid_fraction": fraction("forward_valid"),
        "right_mask_correspondence_fraction": fraction("right_mask_valid"),
        "lr_consistency_fraction": fraction("lr_consistent"),
        "positive_depth_fraction": positive_depth_fraction,
        "median_reprojection_px": None,
        "p95_reprojection_px": None,
        "fraction_denominator": "left semantic mask pixels in the rectified local view",
        "positive_depth_definition": "share of accepted candidates whose rectified depth is positive",
    }


def semantic_evidence_block(
    source_name: str,
    identity_evidence: str,
    mask_area_fraction: float | None,
) -> dict[str, Any]:
    """Layer-1 evidence: what the mask is, never upgraded to ground truth."""
    manually_audited = identity_evidence == "manually_audited"
    return {
        "source": source_name,
        "semantic_identity_evidence": identity_evidence,
        "manual_audit_available": bool(manually_audited),
        "mask_area_fraction": None if mask_area_fraction is None else float(mask_area_fraction),
        "mask_status": "manually_audited" if manually_audited else "candidate",
        "mask_area_fraction_definition": "share of the upright left input image covered by this source's left floor mask",
        "boundary": (
            "A manually audited and a candidate mask control 2-D region identity only; neither is a 3-D plane "
            "measurement and neither may be reported as real ground."
        ),
    }


def stereo_candidates(
    left_upright: np.ndarray,
    right_upright: np.ndarray,
    left_mask_upright: np.ndarray,
    right_mask_upright: np.ndarray,
    calibration: StereoCalibration,
    parameters: dict[str, Any],
    *,
    mask_dilation_px: int = 0,
    module_times: dict[str, float | None] | None = None,
    module_reasons: dict[str, str] | None = None,
) -> dict[str, Any]:
    runtime_size = (int(parameters["runtime_width"]), int(parameters["runtime_height"]))
    num_disparities = int(parameters["num_disparities"])
    left_seed = strict.mask_seed_upright(left_mask_upright)
    right_seed = strict.mask_seed_upright(right_mask_upright)
    if left_seed is None or right_seed is None:
        if module_times is not None:
            for name in ("rectification_and_disparity_ms", "stereo_candidate_filter_ms", "triangulation_ms"):
                module_times[name] = None
                if module_reasons is not None:
                    module_reasons[name] = "empty_semantic_mask"
        return {"status": "unavailable", "reasons": ["empty_semantic_mask"]}
    rectification_start = time.perf_counter()
    left_raw, right_raw = strict.inverse_upright(left_upright, right_upright)
    raw_size = (left_raw.shape[1], left_raw.shape[0])
    scale = np.asarray((runtime_size[0] / raw_size[0], runtime_size[1] / raw_size[1]))
    upright_size = (left_upright.shape[1], left_upright.shape[0])
    left_seed_raw = strict.upright_point_to_raw(left_seed, "left", upright_size) * scale
    right_seed_raw = strict.upright_point_to_raw(right_seed, "right", upright_size) * scale
    rectification = strict.make_mask_directed_rectification(
        calibration, left_seed_raw, right_seed_raw, runtime_size, float(parameters["virtual_focal_px"])
    )
    left_raw = cv2.resize(left_raw, runtime_size, interpolation=cv2.INTER_AREA)
    right_raw = cv2.resize(right_raw, runtime_size, interpolation=cv2.INTER_AREA)
    left_mask_raw = cv2.resize(
        strict.rotate_mask_to_raw(left_mask_upright, "left"), runtime_size, interpolation=cv2.INTER_NEAREST
    )
    right_mask_raw = cv2.resize(
        strict.rotate_mask_to_raw(right_mask_upright, "right"), runtime_size, interpolation=cv2.INTER_NEAREST
    )
    left_local = cv2.remap(left_raw, rectification.left_map_x, rectification.left_map_y, cv2.INTER_LINEAR)
    right_local = cv2.remap(right_raw, rectification.right_map_x, rectification.right_map_y, cv2.INTER_LINEAR)
    left_mask_local = cv2.remap(
        left_mask_raw, rectification.left_map_x, rectification.left_map_y, cv2.INTER_NEAREST
    )
    right_mask_local = cv2.remap(
        right_mask_raw, rectification.right_map_x, rectification.right_map_y, cv2.INTER_NEAREST
    )
    left_mask_for_matching = left_mask_local
    right_mask_for_matching = right_mask_local
    if mask_dilation_px > 0:
        radius = int(mask_dilation_px)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        left_mask_for_matching = cv2.dilate(left_mask_local, kernel)
        right_mask_for_matching = cv2.dilate(right_mask_local, kernel)
    forward = strict.dense_disparity(left_local, right_local, num_disparities)
    reverse = diagnostic.dense_disparity_right_direction(right_local, left_local, num_disparities)
    if module_times is not None:
        # Carries the necessary mask-directed rectification and both SGBM
        # disparity fields (forward and the corrected reverse field).
        module_times["rectification_and_disparity_ms"] = (time.perf_counter() - rectification_start) * 1000.0
    filter_start = time.perf_counter()
    height, width = forward.shape
    yy, xx = np.mgrid[0:height, 0:width]
    partner_x_grid = np.rint(xx.astype(np.float32) - forward).astype(np.int32)
    partner_clipped = np.clip(partner_x_grid, 0, width - 1)
    in_bounds_grid = (partner_x_grid >= 0) & (partner_x_grid < width)
    left_semantic = left_mask_for_matching > 0
    forward_valid = left_semantic & (forward > 1.0) & in_bounds_grid
    right_semantic = right_mask_for_matching[yy, partner_clipped] > 0
    right_mask_valid = forward_valid & right_semantic
    reverse_at_partner = reverse[yy, partner_clipped]
    reverse_valid = right_mask_valid & (reverse_at_partner > 1.0)
    lr_valid = reverse_valid & (np.abs(forward - reverse_at_partner) <= float(parameters["lr_consistency_px"]))
    filter_elapsed_ms = (time.perf_counter() - filter_start) * 1000.0
    triangulation_start = time.perf_counter()
    candidate = corrected.reconstruct(
        left_local, right_local, left_mask_for_matching, right_mask_for_matching, rectification,
        forward, reverse, num_disparities, float(parameters["lr_consistency_px"]),
    )
    if module_times is not None:
        # The frozen candidate chain re-derives its own validity mask and then
        # back-projects every accepted pixel into the left-camera frame.
        module_times["triangulation_ms"] = (time.perf_counter() - triangulation_start) * 1000.0
    left_pixels = candidate["left_pixels"]
    right_pixels = candidate["right_pixels"]
    filter_start = time.perf_counter()
    left_gray = cv2.cvtColor(left_local, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_local, cv2.COLOR_BGR2GRAY)
    gradient = cv2.magnitude(
        cv2.Sobel(left_gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(left_gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    if len(left_pixels):
        texture = gradient[left_pixels[:, 1], left_pixels[:, 0]].astype(np.float64)
        texture /= max(float(np.percentile(texture, 90)), 1.0)
        texture = np.clip(texture, 0.0, 1.0)
        photo = np.abs(
            left_gray[left_pixels[:, 1], left_pixels[:, 0]].astype(np.float64)
            - right_gray[right_pixels[:, 1], right_pixels[:, 0]].astype(np.float64)
        )
        # No learned confidence is claimed.  This deterministic score only
        # ranks already accepted stereo candidates by texture and photometric
        # agreement for the soft/sparse geometric estimators.
        confidence = (0.1 + 0.9 * texture) * np.exp(-photo / 20.0)
    else:
        photo = np.empty(0, dtype=np.float64)
        confidence = np.empty(0, dtype=np.float64)
    if module_times is not None:
        module_times["stereo_candidate_filter_ms"] = (
            filter_elapsed_ms + (time.perf_counter() - filter_start) * 1000.0
        )
    return {
        "status": "candidate",
        "reasons": [],
        **candidate,
        "confidence": confidence,
        "photometric_difference": photo,
        "left_local": left_local,
        "right_local": right_local,
        "left_mask_local": left_mask_local,
        "right_mask_local": right_mask_local,
        "left_mask_for_matching": left_mask_for_matching,
        "right_mask_for_matching": right_mask_for_matching,
        "mask_dilation_px": int(mask_dilation_px),
        "rectification": rectification,
        "matching_funnel": {
            "left_mask_pixels": int(left_semantic.sum()),
            "forward_valid": int(forward_valid.sum()),
            "right_mask_valid": int(right_mask_valid.sum()),
            "reverse_valid": int(reverse_valid.sum()),
            "lr_consistent": int(lr_valid.sum()),
            "final_candidates": int(len(candidate["points"])),
        },
    }


def plane_evidence_block(
    estimate: Any,
    candidate: dict[str, Any],
    rectification: Any,
    calibration: StereoCalibration,
    shape: tuple[int, int],
) -> tuple[dict[str, Any], dict[str, float | None]]:
    """Layer-3 evidence of one fitted representation plus its reprojection.

    Returns the evidence block and the reprojection percentiles, which the
    caller merges into the stereo layer of the same method.  Nothing here
    re-fits, re-selects or relaxes anything; it only describes the estimator's
    own output.
    """
    points = candidate["points"]
    mapping = estimate.as_mapping()
    metrics = mapping.get("metrics") or {}
    plane = mapping.get("plane")
    inlier_mask = np.asarray(estimate.inlier_mask, dtype=bool)
    coverage: float | None = None
    reprojection: dict[str, float | None] = {
        "median_reprojection_px": None,
        "p95_reprojection_px": None,
        "median_reprojection_left_px": None,
        "median_reprojection_right_px": None,
    }
    if plane is not None and inlier_mask.shape == (len(points),) and int(inlier_mask.sum()) > 0:
        inlier_left = candidate["left_pixels"][inlier_mask]
        inlier_right = candidate["right_pixels"][inlier_mask]
        coverage = float(strict.local_coverage_fraction(inlier_left, shape))
        reprojection = reprojection_percentiles(
            points[inlier_mask], inlier_left, inlier_right, rectification, calibration
        )
    block = {
        "status": "candidate" if plane is not None else "unavailable",
        "inlier_count": int(mapping.get("inlier_count", 0) or 0),
        "inlier_fraction": metrics.get("inlier_fraction"),
        "median_residual_mm": metrics.get("median_residual_mm"),
        "coverage_fraction": coverage,
        "normal_left_camera": None if plane is None else plane.get("normal_left_camera"),
        "offset_mm": None if plane is None else plane.get("offset_mm"),
        "reasons": list(mapping.get("reasons", []) or []),
        "median_residual_definition": "median absolute plane distance of this method's inliers, in millimetres",
        "coverage_fraction_definition": (
            "convex-hull area of this method's inliers over the rectified local image area"
        ),
        "boundary": (
            "A fitted plane whose residual is inside the frozen RANSAC distance is a candidate representation "
            "only. A low residual does not make these pixels the physical floor."
        ),
    }
    return block, reprojection


def floor_methods(
    candidate: dict[str, Any],
    parameters: dict[str, Any],
    seed: int,
    *,
    calibration: StereoCalibration,
    source_name: str,
    identity_evidence: str,
    mask_area_fraction: float | None,
    module_times: dict[str, float | None],
    module_reasons: dict[str, str],
) -> dict[str, Any]:
    """Fit every ground representation, with per-method timing and evidence.

    The estimators, their arguments, their order and their seeds are unchanged
    from the frozen benchmark.  Only measurement and evidence assembly are
    added, so the geometric fields remain directly comparable with the
    previously archived run.
    """
    semantic_block = semantic_evidence_block(source_name, identity_evidence, mask_area_fraction)
    if candidate["status"] != "candidate":
        reasons = list(candidate.get("reasons", []))
        unavailable = {
            "status": "unavailable",
            "reason": "stereo_candidate_stage_unavailable",
            "stage_reasons": reasons,
        }
        methods = {
            method: {"method": method, "status": "unavailable", "reasons": reasons} for method in METHODS
        }
        evidence = {
            method: {
                "semantic_evidence": semantic_block,
                "stereo_evidence": {**stereo_evidence(candidate), **unavailable},
                "plane_evidence": {
                    "status": "unavailable",
                    "inlier_count": None,
                    "inlier_fraction": None,
                    "median_residual_mm": None,
                    "coverage_fraction": None,
                    "normal_left_camera": None,
                    "offset_mm": None,
                    "reasons": reasons,
                },
                "cross_region_evidence": {
                    "status": "unavailable",
                    "region_count": None,
                    "normal_spread_deg": None,
                    "offset_spread_mm": None,
                    "reasons": reasons,
                    "applies_to": "shared_stereo_candidate_set_gate",
                    "note": REGION_GATE_NOTE,
                },
                "confidence_boundary": CONFIDENCE_BOUNDARY,
            }
            for method in METHODS
        }
        reason = "estimator_not_executed_stereo_candidates_unavailable"
        return {
            "methods": methods,
            "evidence": evidence,
            "timing": {method: {"plane_fit_ms": None, EVIDENCE_MODULE: None, "region_consensus_ms": None}
                       for method in MAIN_METHODS},
            "timing_reasons": {method: {"plane_fit_ms": reason, EVIDENCE_MODULE: reason, "region_consensus_ms": reason}
                               for method in MAIN_METHODS},
            "region_consensus_ms": None,
            "region_evidence_ms": None,
            "completion_perf": {},
        }
    points = candidate["points"]
    pixels = candidate["left_pixels"]
    confidence = candidate["confidence"]
    shape = candidate["left_local"].shape[:2]
    rectification = candidate["rectification"]
    threshold = float(parameters["ransac_distance_mm"])
    iterations = int(parameters["ransac_iterations"])
    common_stereo = stereo_evidence(candidate)
    estimates: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    fit_times: dict[str, float] = {}
    evidence_elapsed: dict[str, float] = {}
    completion_perf: dict[str, float] = {}

    def finish_method(name: str, estimate: Any, fit_ms: float) -> None:
        """Record one method's fit, its evidence and the instant it was ready."""
        evidence_started = time.perf_counter()
        plane_block, reprojection = plane_evidence_block(
            estimate, candidate, rectification, calibration, shape
        )
        evidence_elapsed[name] = (time.perf_counter() - evidence_started) * 1000.0
        fit_times[name] = float(fit_ms)
        estimates[name] = estimate
        evidence[name] = {
            "semantic_evidence": semantic_block,
            "stereo_evidence": {
                **common_stereo,
                **reprojection,
                "reprojection_scope": "this method's inliers",
                "median_reprojection_definition": (
                    "median over this method's inliers of the per-point maximum of the left and right "
                    "calibrated fisheye reprojection error, in pixels"
                ),
            },
            "plane_evidence": plane_block,
            "cross_region_evidence": {},  # filled in once the shared gate has run
            "confidence_boundary": CONFIDENCE_BOUNDARY,
        }
        completion_perf[name] = time.perf_counter()

    # The estimator calls, their arguments, their order and their seeds are the
    # frozen ones.  Python evaluates call arguments left to right, so the
    # elapsed time is read immediately after the estimator returns and before
    # the evidence work of ``finish_method`` starts.
    started = time.perf_counter()
    finish_method(
        "dense_ransac",
        estimate_dense_ransac(
            points, distance_threshold_mm=threshold, iterations=iterations, seed=seed, minimum_points=150,
        ),
        (time.perf_counter() - started) * 1000.0,
    )
    started = time.perf_counter()
    finish_method(
        "sparse_tile_ransac",
        estimate_sparse_tiles(
            points, pixels, confidence, image_shape=shape, tile_size_px=32, maximum_per_tile=4,
            distance_threshold_mm=threshold, iterations=iterations, seed=seed, minimum_points=30,
        ),
        (time.perf_counter() - started) * 1000.0,
    )
    started = time.perf_counter()
    finish_method(
        "soft_weighted_irls",
        estimate_weighted_irls(
            points, confidence, huber_scale_mm=20.0, iterations=8,
            distance_threshold_mm=threshold, minimum_points=30,
        ),
        (time.perf_counter() - started) * 1000.0,
    )
    # Shared quality-rejection gate.  It is timed once per frame/source and is
    # reported for every method; it is not ranked as a fourth fitter.
    region_started = time.perf_counter()
    regional = estimate_region_consensus(
        points, pixels, image_shape=shape, distance_threshold_mm=threshold,
        # ``REGION_MINIMUM_POINTS``/``REGION_MAXIMUM_ANGLE_DEG`` are the frozen
        # literal gate values 30 and 5.0, named so the mirror below cannot drift.
        minimum_region_points=REGION_MINIMUM_POINTS, maximum_region_angle_deg=REGION_MAXIMUM_ANGLE_DEG,
    )
    region_block = region_consensus_evidence(points, pixels, shape, threshold)
    if (region_block["status"] == "pass") != (regional.normal_left_camera is not None):
        raise RuntimeError(
            "cross-region evidence disagrees with the frozen region-consensus gate; "
            "refusing to report an inconsistent gate state"
        )
    region_evidence_started = time.perf_counter()
    region_plane_block, region_reprojection = plane_evidence_block(
        regional, candidate, rectification, calibration, shape
    )
    region_evidence_elapsed_ms = (time.perf_counter() - region_evidence_started) * 1000.0
    region_elapsed_ms = (time.perf_counter() - region_started) * 1000.0
    estimates[REGION_METHOD] = regional
    evidence[REGION_METHOD] = {
        "semantic_evidence": semantic_block,
        "stereo_evidence": {
            **common_stereo,
            **region_reprojection,
            "reprojection_scope": "this gate's inliers",
        },
        "plane_evidence": region_plane_block,
        "cross_region_evidence": {
            **region_block,
            "applies_to": "shared_stereo_candidate_set_gate",
            "note": REGION_GATE_NOTE,
        },
        "confidence_boundary": CONFIDENCE_BOUNDARY,
    }
    shared_region_block = {
        **region_block,
        "applies_to": "shared_stereo_candidate_set_gate",
        "note": REGION_GATE_NOTE,
    }
    for name in MAIN_METHODS:
        evidence[name]["cross_region_evidence"] = shared_region_block
    methods = {name: estimate.as_mapping() for name, estimate in estimates.items()}
    timing: dict[str, dict[str, float | None]] = {}
    timing_reasons: dict[str, dict[str, str]] = {}
    shared_region_ms = float(region_elapsed_ms)
    for name in MAIN_METHODS:
        timing[name] = {
            "plane_fit_ms": float(fit_times[name]),
            EVIDENCE_MODULE: float(evidence_elapsed[name]),
            "region_consensus_ms": shared_region_ms,
        }
        timing_reasons[name] = {}
    return {
        "methods": methods,
        "evidence": evidence,
        "timing": timing,
        "timing_reasons": timing_reasons,
        "region_consensus_ms": shared_region_ms,
        "region_evidence_ms": region_evidence_elapsed_ms,
        "completion_perf": completion_perf,
    }


def method_comparisons(methods: dict[str, Any]) -> list[dict[str, Any]]:
    names = [name for name in METHODS if methods.get(name, {}).get("plane")]
    rows: list[dict[str, Any]] = []
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            a, b = methods[first]["plane"], methods[second]["plane"]
            rows.append({
                "first": first,
                "second": second,
                "normal_angle_deg": angular_error_degrees(a["normal_left_camera"], b["normal_left_camera"]),
                "absolute_offset_difference_mm": abs(float(a["offset_mm"]) - float(b["offset_mm"])),
            })
    return rows


def render_panel(
    image: np.ndarray, floor_mask: np.ndarray, walker_mask: np.ndarray,
    floor_candidate: dict[str, Any], walker_candidate: dict[str, Any], title: str,
) -> np.ndarray:
    overlay = image.copy()
    color = np.zeros_like(image)
    color[floor_mask > 0] = (0, 180, 0)
    color[walker_mask > 0] = (0, 180, 255)
    overlay = cv2.addWeighted(overlay, 0.62, color, 0.38, 0.0)
    cv2.rectangle(overlay, (0, 0), (min(overlay.shape[1], 1400), 48), (255, 255, 255), -1)
    text = (
        f"{title}; floor stereo={len(floor_candidate.get('points', []))}; "
        f"walker stereo={len(walker_candidate.get('points', []))}; green=floor yellow=walker"
    )
    cv2.putText(overlay, text, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1, cv2.LINE_AA)
    return overlay


def source_record(
    source_name: str,
    identity_evidence: str,
    floor_candidate: dict[str, Any],
    parameters: dict[str, Any],
    seed: int,
    *,
    calibration: StereoCalibration,
    mask_area_fraction: float | None,
    source_module_times: dict[str, float | None],
    source_module_reasons: dict[str, str],
    frame_start_perf: float,
) -> dict[str, Any]:
    """One ground source of one frame: geometry, evidence and timing.

    ``total_cpu_wall_ms`` is captured for each method at the moment that
    method's output exists, measured from the start of the frame's image read
    and excluding only the PNG/JSONL/report disk writes.  Because the three
    methods are evaluated serially in the fixed order ``dense_ransac``,
    ``sparse_tile_ransac``, ``soft_weighted_irls``, the later totals contain the
    earlier methods' work as well.
    """
    result = floor_methods(
        floor_candidate,
        parameters,
        seed,
        calibration=calibration,
        source_name=source_name,
        identity_evidence=identity_evidence,
        mask_area_fraction=mask_area_fraction,
        module_times=source_module_times,
        module_reasons=source_module_reasons,
    )
    methods = result["methods"]
    completion_perf = result.get("completion_perf", {})
    for method in MAIN_METHODS:
        merged: dict[str, float | None] = {}
        reasons: dict[str, str] = {}
        for module in TIMING_MODULES:
            if module in source_module_times:
                value = source_module_times.get(module)
                merged[module] = None if value is None else float(value)
                if value is None:
                    reasons[module] = source_module_reasons.get(module, "module_not_executed")
                continue
            if module == "total_cpu_wall_ms":
                # From reading this frame to this method's own plane output and
                # per-method evidence being ready.  The shared cross-region gate
                # runs afterwards and is reported separately as
                # region_consensus_ms.
                finished = completion_perf.get(method)
                merged[module] = None if finished is None else (finished - frame_start_perf) * 1000.0
                if finished is None:
                    reasons[module] = "method_output_not_reached"
                continue
            if module == "visualization_ms":
                # Filled in by the caller once the frame panel is rendered; the
                # PNG write itself is deliberately outside this measurement.
                merged[module] = None
                reasons[module] = "visualization_measured_after_method_outputs"
                continue
            value = result["timing"].get(method, {}).get(module)
            merged[module] = None if value is None else float(value)
            if value is None:
                reasons[module] = result["timing_reasons"].get(method, {}).get(module, "module_not_executed")
        methods[method]["timing_ms"] = merged
        methods[method]["timing_null_reasons"] = reasons
        methods[method]["evidence"] = result["evidence"][method]
    methods[REGION_METHOD]["evidence"] = result["evidence"][REGION_METHOD]
    methods[REGION_METHOD]["timing_note"] = REGION_GATE_NOTE
    return {
        "source_name": source_name,
        "semantic_identity_evidence": identity_evidence,
        "stereo_candidate_count": int(len(floor_candidate.get("points", []))),
        "methods": methods,
        "within_frame_method_comparisons": method_comparisons(methods),
        "region_consensus_ms": result.get("region_consensus_ms"),
        "acceptance_boundary": (
            "manually audited labels condition a geometry comparison but are not 3-D ground truth"
            if identity_evidence == "manually_audited"
            else "automatic mask remains candidate-only even when a geometric estimator returns a plane"
        ),
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    source_names = sorted({source["source_name"] for record in records for source in record["floor_sources"]})
    sources: dict[str, Any] = {}
    for source_name in source_names:
        selected = [source for record in records for source in record["floor_sources"] if source["source_name"] == source_name]
        method_summary: dict[str, Any] = {}
        for method in METHODS:
            outputs = [item["methods"][method] for item in selected]
            available = [item for item in outputs if item.get("plane") is not None]
            method_summary[method] = {
                "plane_candidate_frames": len(available),
                "total_frames": len(outputs),
                "median_inlier_fraction": (
                    float(np.median([item["metrics"]["inlier_fraction"] for item in available])) if available else None
                ),
                "median_residual_mm": (
                    float(np.median([item["metrics"]["median_residual_mm"] for item in available])) if available else None
                ),
            }
        sources[source_name] = {"methods": method_summary}
    conditional_reference_comparisons: dict[str, Any] = {}
    for source_name in source_names:
        if source_name == "manual_floor":
            continue
        per_method: dict[str, Any] = {}
        for method in METHODS:
            angles: list[float] = []
            offsets: list[float] = []
            for record in records:
                source_map = {item["source_name"]: item for item in record["floor_sources"]}
                reference = source_map.get("manual_floor", {}).get("methods", {}).get("dense_ransac", {}).get("plane")
                estimate = source_map.get(source_name, {}).get("methods", {}).get(method, {}).get("plane")
                if reference is None or estimate is None:
                    continue
                angles.append(angular_error_degrees(reference["normal_left_camera"], estimate["normal_left_camera"]))
                offsets.append(abs(float(reference["offset_mm"]) - float(estimate["offset_mm"])))
            per_method[method] = {
                "compared_frames": len(angles),
                "median_normal_angle_to_manual_dense_deg": float(np.median(angles)) if angles else None,
                "p90_normal_angle_to_manual_dense_deg": float(np.percentile(angles, 90)) if angles else None,
                "median_absolute_offset_to_manual_dense_mm": float(np.median(offsets)) if offsets else None,
            }
        conditional_reference_comparisons[source_name] = per_method
    walker_available = [record["walker"]["component_model"] for record in records if record["walker"]["component_model"]["status"] == "candidate"]
    return {
        "schema_version": SCHEMA_VERSION,
        "pair_count": len(records),
        "floor_sources": sources,
        "automatic_sources_vs_manual_mask_conditioned_dense_reference": conditional_reference_comparisons,
        "walker_component_candidate_frames": len(walker_available),
        "walker_total_frames": len(records),
        "interpretation_boundary": (
            "This benchmark compares representations under controlled masks and calibrated stereo. "
            "Cross-source differences use the manual-mask dense plane only as a conditional reference, not physical truth. "
            "It does not measure physical ground/walker accuracy, gait events, support, or hand contact."
        ),
    }


def median_or_none(values: list[Any]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return float(np.median(present)) if present else None


def evidence_layer_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per source/method counts of the independent evidence layers.

    The layers are counted, never combined: a candidate plane on an automatic
    mask is counted as a candidate, and it is never promoted to audited or real
    ground.
    """
    source_names = sorted({source["source_name"] for record in records for source in record["floor_sources"]})
    summary: dict[str, Any] = {}
    for source_name in source_names:
        selected = [
            source for record in records for source in record["floor_sources"] if source["source_name"] == source_name
        ]
        per_method: dict[str, Any] = {}
        for method in METHODS:
            blocks = [item["methods"][method].get("evidence", {}) for item in selected]
            plane_states = [block.get("plane_evidence", {}).get("status") for block in blocks]
            region_states = [block.get("cross_region_evidence", {}).get("status") for block in blocks]
            semantic = [block.get("semantic_evidence", {}) for block in blocks]
            per_method[method] = {
                "frames": len(blocks),
                "plane_evidence_candidate_frames": int(sum(state == "candidate" for state in plane_states)),
                "plane_evidence_unavailable_frames": int(sum(state == "unavailable" for state in plane_states)),
                "cross_region_pass_frames": int(sum(state == "pass" for state in region_states)),
                "cross_region_fail_frames": int(sum(state == "fail" for state in region_states)),
                "cross_region_unavailable_frames": int(sum(state == "unavailable" for state in region_states)),
                "manual_audit_available_frames": int(
                    sum(bool(item.get("manual_audit_available")) for item in semantic)
                ),
                "median_mask_area_fraction": median_or_none(
                    [item.get("mask_area_fraction") for item in semantic]
                ),
                "confidence_boundary": CONFIDENCE_BOUNDARY,
            }
        summary[source_name] = {"methods": per_method}
    return summary


def timing_collectors_table(
    collectors: dict[tuple[str, str], TimingCollector]
) -> dict[str, dict[str, dict[str, dict[str, float | int]]]]:
    """``{source: {method: {module: summary}}}`` built from the live collectors."""
    table: dict[str, Any] = {}
    for (source_name, method), collector in sorted(collectors.items()):
        table.setdefault(source_name, {})[method] = collector.summary()
    return table


def without_warmup(collector: TimingCollector, skip: int) -> dict[str, dict[str, float | int]]:
    """Frame-level summary with the warm-up frames dropped, as for the methods."""
    reduced = TimingCollector(gpu_synchronize=False)
    for name in collector.module_names():
        for value in collector.samples_ms(name)[skip:]:
            reduced.record_ms(name, value)
    return reduced.summary()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    names = selected_names(args)
    if not names:
        raise RuntimeError("no paired LabelMe annotations found")
    parameters, _ = corrected.load_frozen(args.frozen_baseline_dir)
    runtime_size = (int(parameters["runtime_width"]), int(parameters["runtime_height"]))
    calibration = StereoCalibration.load(args.calibration).for_runtime_sizes(runtime_size, runtime_size)
    # Every collector opts out of CUDA synchronisation on purpose: this
    # pipeline is pure CPU OpenCV/NumPy work and importing torch into this
    # process would abort it on the next NumPy linear-algebra call (see
    # describe_hardware).
    gpu_synchronized = False
    hardware, hardware_note = describe_hardware()
    args.output_dir.mkdir(parents=True)
    if not args.no_visualization:
        (args.output_dir / "visualizations").mkdir()
    records: list[dict[str, Any]] = []
    walker_point_sets: list[np.ndarray] = []
    frame_level = TimingCollector(gpu_synchronize=False)
    collectors: dict[tuple[str, str], TimingCollector] = {}
    per_frame_timing: list[dict[str, Any]] = []
    input_resolution: list[int] | None = None
    for index, name in enumerate(names):
        frame_start_perf = time.perf_counter()
        with frame_level.measure("image_read_ms"):
            left = cv2.imread(str(args.left_dir / name), cv2.IMREAD_COLOR)
            right = cv2.imread(str(args.right_dir / name), cv2.IMREAD_COLOR)
            if left is None or right is None or left.shape[:2] != right.shape[:2]:
                raise RuntimeError(f"cannot read equal-size stereo pair {name}")
        input_resolution = [int(left.shape[1]), int(left.shape[0])]
        manual_module_times: dict[str, float | None] = {}
        manual_module_reasons: dict[str, str] = {}
        with frame_level.measure("manual_mask_load_ms"):
            left_masks = load_manual_masks(args.left_label_dir / Path(name).with_suffix(".json"), left.shape[:2])
            right_masks = load_manual_masks(args.right_label_dir / Path(name).with_suffix(".json"), right.shape[:2])
            manual_mask_area_fraction = float((left_masks["floor_eligible"] > 0).mean())
        manual_module_times["mask_load_ms"] = frame_level.samples_ms("manual_mask_load_ms")[-1]
        floor_candidate = stereo_candidates(
            left, right, left_masks["floor_eligible"], right_masks["floor_eligible"], calibration, parameters,
            module_times=manual_module_times, module_reasons=manual_module_reasons,
        )
        walker_candidate = stereo_candidates(
            left, right, left_masks["walker"], right_masks["walker"], calibration, parameters,
            mask_dilation_px=6,
        )
        floor_sources = [
            source_record(
                "manual_floor", "manually_audited", floor_candidate, parameters, 20260912 + index,
                calibration=calibration, mask_area_fraction=manual_mask_area_fraction,
                source_module_times=manual_module_times, source_module_reasons=manual_module_reasons,
                frame_start_perf=frame_start_perf,
            )
        ]
        if args.auto_floor_left_dir is not None or args.auto_floor_right_dir is not None:
            if args.auto_floor_left_dir is None or args.auto_floor_right_dir is None:
                raise ValueError("automatic floor mask directories must be supplied as a left/right pair")
            auto_module_times: dict[str, float | None] = {}
            auto_module_reasons: dict[str, str] = {}
            with frame_level.measure("auto_mask_load_ms"):
                auto_floor_left = load_binary(args.auto_floor_left_dir / name, left.shape[:2])
                auto_floor_right = load_binary(args.auto_floor_right_dir / name, right.shape[:2])
                auto_mask_area_fraction = float((auto_floor_left > 0).mean())
            auto_module_times["mask_load_ms"] = frame_level.samples_ms("auto_mask_load_ms")[-1]
            auto_candidate = stereo_candidates(
                left, right, auto_floor_left, auto_floor_right, calibration, parameters,
                module_times=auto_module_times, module_reasons=auto_module_reasons,
            )
            floor_sources.append(
                source_record(
                    args.auto_floor_name, "provided_unvalidated", auto_candidate, parameters, 20261912 + index,
                    calibration=calibration, mask_area_fraction=auto_mask_area_fraction,
                    source_module_times=auto_module_times, source_module_reasons=auto_module_reasons,
                    frame_start_perf=frame_start_perf,
                )
            )
        if walker_candidate["status"] == "candidate":
            component_model = reconstruct_walker_components(
                walker_candidate["points"], walker_candidate["left_pixels"], walker_candidate["left_mask_local"],
                minimum_component_pixels=30, minimum_component_points=10,
            )
            walker_point_sets.append(walker_candidate["points"])
        else:
            component_model = {
                "status": "unavailable", "reasons": walker_candidate["reasons"],
                "interpretation": "No stereo walker candidates; no geometry was manufactured.",
            }
            walker_point_sets.append(np.empty((0, 3), dtype=np.float64))
        record = {
            "schema_version": SCHEMA_VERSION,
            "pair_id": pair_id(Path(name)),
            "frame_id": name,
            "warmup_frame": bool(index < WARMUP_FRAMES_EXCLUDED),
            "floor_sources": floor_sources,
            "walker": {
                "semantic_identity_evidence": "manually_audited_visible_pixels",
                "dense_visible_point_count": int(len(walker_candidate.get("points", []))),
                "matching_mask_dilation_px": int(walker_candidate.get("mask_dilation_px", 0)),
                "matching_funnel": walker_candidate.get("matching_funnel", {}),
                "component_model": component_model,
                "sparse_measured_template_fit": {
                    "status": "unavailable",
                    "reasons": ["named_3d_walker_landmarks_and_measured_configuration_missing"],
                    "implementation": "pose_app.scene_geometry_variants.fit_rigid_template",
                },
            },
        }
        records.append(record)
        panel = None
        if not args.no_visualization:
            with frame_level.measure("visualization_ms"):
                panel = render_panel(
                    left, left_masks["floor_eligible"], left_masks["walker"], floor_candidate, walker_candidate, name
                )
            visualization_ms = float(frame_level.samples_ms("visualization_ms")[-1])
            for source in floor_sources:
                for method in MAIN_METHODS:
                    entry = source["methods"][method]
                    entry["timing_ms"]["visualization_ms"] = visualization_ms
                    entry["timing_null_reasons"].pop("visualization_ms", None)
        else:
            for source in floor_sources:
                for method in MAIN_METHODS:
                    source["methods"][method]["timing_null_reasons"]["visualization_ms"] = "visualization_disabled"
        for source in floor_sources:
            for method in MAIN_METHODS:
                if index < WARMUP_FRAMES_EXCLUDED:
                    # Warm-up frames are measured and kept in the per-frame
                    # record but never enter the P50/P90/P95 statistics.
                    continue
                collector = collectors.setdefault(
                    (source["source_name"], method), TimingCollector(gpu_synchronize=False)
                )
                for module, value in source["methods"][method]["timing_ms"].items():
                    if value is None:
                        continue
                    collector.record_ms(module, value)
        per_frame_timing.append({
            "pair_id": record["pair_id"],
            "frame_id": name,
            "warmup_frame": bool(index < WARMUP_FRAMES_EXCLUDED),
            "image_read_ms": float(frame_level.samples_ms("image_read_ms")[-1]),
            "sources": {
                source["source_name"]: {
                    method: source["methods"][method]["timing_ms"] for method in MAIN_METHODS
                }
                for source in floor_sources
            },
            "region_consensus_ms": {
                source["source_name"]: source.get("region_consensus_ms") for source in floor_sources
            },
        })
        if panel is not None:
            # Disk writes are deliberately outside every measured total.
            cv2.imwrite(str(args.output_dir / "visualizations" / name), panel)
    temporal = camera_attached_temporal_consensus(
        walker_point_sets, voxel_size_mm=20.0, minimum_frame_support=max(2, min(4, len(records) // 3))
    )
    with (args.output_dir / "frame_records.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    timing_protocol = {
        "warmup_frames_excluded": WARMUP_FRAMES_EXCLUDED,
        "measurement_unit": "ms",
        "input_resolution": input_resolution if input_resolution is not None else None,
        "input_resolution_order": "width_height",
        "gpu_synchronized": bool(gpu_synchronized),
        "hardware_note": hardware_note,
        "hardware": hardware,
        "clock": "time.perf_counter",
        "timing_modules": list(TIMING_MODULES),
        "module_definitions": {
            "mask_load_ms": (
                "read and rasterise this ground source's semantic mask for the frame. For the manual source the "
                "LabelMe polygons of both images are rasterised in one shared call that also serves the walker "
                "branch; for the automatic source the two cached floor-mask PNGs are read"
            ),
            "rectification_and_disparity_ms": (
                "mask-directed local fisheye rectification plus both SGBM disparity fields "
                "(forward and the corrected reverse field)"
            ),
            "stereo_candidate_filter_ms": (
                "shared candidate-filter grids (mask, forward validity, right-mask correspondence, reverse validity, "
                "left/right consistency) and the deterministic texture/photometric candidate scoring"
            ),
            "triangulation_ms": (
                "the frozen candidate chain: re-derivation of the accepted pixel set and its back-projection into "
                "the calibrated left-camera frame"
            ),
            "plane_fit_ms": "the single plane estimator named by the method key",
            "region_consensus_ms": (
                "the shared cross-region quality gate (frozen gate plus its per-region spread extraction and the "
                "gate's own evidence); identical for every method of the frame because the gate is shared, and not "
                "ranked as a fourth fitter"
            ),
            "evidence_ms": (
                "assembly of this method's coverage and fisheye reprojection evidence; reported separately instead "
                "of being hidden inside the estimator time"
            ),
            "visualization_ms": "frame panel rendering; the PNG encode/write is excluded",
            "total_cpu_wall_ms": (
                "from reading this frame's images to this method's own plane output and per-method evidence being "
                "ready, excluding PNG/JSONL/report disk writes and excluding the shared cross-region gate. It "
                "therefore also contains the walker candidate chain and, for the second ground source, the whole "
                "first source. The three methods are evaluated serially in the fixed order dense_ransac, "
                "sparse_tile_ransac, soft_weighted_irls, so a later total contains the earlier methods' work as well"
            ),
        },
        "excluded_from_total_cpu_wall_ms": [
            "visualization PNG writes", "frame_records.jsonl", "summary.json", "run_metadata.json", "timing_summary.json",
        ],
        "null_policy": (
            "A module that did not execute is null with a reason under timing_null_reasons, never 0."
        ),
        "warmup_note": (
            "The first two pairs of the fixed 12-pair protocol are warm-up only: their geometry and reasons are "
            "kept, their timings appear in the per-frame record, and they are excluded from every P50/P90/P95."
        ),
    }
    summary = aggregate(records)
    summary["walker_camera_attached_temporal_consensus"] = temporal
    summary["timing_protocol"] = timing_protocol
    summary["timing_by_source_and_method"] = timing_collectors_table(collectors)
    summary["evidence_layer_summary"] = evidence_layer_summary(records)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    timing_summary = {
        "schema_version": SCHEMA_VERSION + "_timing_v1",
        "timing_protocol": timing_protocol,
        "timing_by_source_and_method": summary["timing_by_source_and_method"],
        "frame_level_modules": without_warmup(frame_level, WARMUP_FRAMES_EXCLUDED),
        "accelerator_probe": {
            "status": "not_performed_in_this_process",
            "reason": (
                "This pipeline launches no GPU kernel and the host's Anaconda MKL NumPy cannot coexist with "
                "torch in one process (OMP Error #15 on the next NumPy linear-algebra call), so torch is not "
                "imported here. pose_app.benchmark_timing.cuda_is_available()/synchronize_if_cuda() implement "
                "the real probe and are exercised out of process by tests/test_benchmark_timing.py."
            ),
        },
        "per_frame_timing": per_frame_timing,
        "method_order": list(MAIN_METHODS),
        "shared_gate_only": REGION_METHOD,
        "interpretation_boundary": (
            "Wall-clock measurements of this tool's own pipeline modules on this machine. They are not a real-time "
            "or field-performance claim, and region_consensus is a shared gate, not a fourth ranked fitter."
        ),
    }
    (args.output_dir / "timing_summary.json").write_text(
        json.dumps(timing_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "left_dir": str(args.left_dir.resolve()), "right_dir": str(args.right_dir.resolve()),
            "left_label_dir": str(args.left_label_dir.resolve()), "right_label_dir": str(args.right_label_dir.resolve()),
            "calibration": str(args.calibration.resolve()), "frozen_baseline_dir": str(args.frozen_baseline_dir.resolve()),
            "auto_floor_left_dir": None if args.auto_floor_left_dir is None else str(args.auto_floor_left_dir.resolve()),
            "auto_floor_right_dir": None if args.auto_floor_right_dir is None else str(args.auto_floor_right_dir.resolve()),
        },
        "frozen_parameters": parameters,
        "pair_names": names,
        "methods": list(METHODS),
        "ranked_methods": list(MAIN_METHODS),
        "shared_gate_method": REGION_METHOD,
        "timing_protocol": timing_protocol,
        "evidence_schema": {
            "layers": ["semantic_evidence", "stereo_evidence", "plane_evidence", "cross_region_evidence"],
            "confidence_boundary": CONFIDENCE_BOUNDARY,
            "note": (
                "The four layers are stored separately on every frame/source/method. They are never collapsed into a "
                "single pseudo-probability. An automatic mask stays candidate-only, and a small plane residual is not "
                "evidence of real ground."
            ),
        },
        "notes": [
            "corrected reverse disparity search direction is inherited from the controlled experiment",
            "soft weights are deterministic texture/photometric scores over already accepted stereo candidates",
            "no DA3, pose model, GroundNet, IMU, persistent world frame, or contact label is used",
            "no SGBM setting, left/right consistency limit, triangulation rule, RANSAC threshold or decision gate was changed",
        ],
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
