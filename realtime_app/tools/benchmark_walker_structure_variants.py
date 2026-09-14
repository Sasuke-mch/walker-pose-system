#!/usr/bin/env python3
"""Controlled ``dense`` / ``skeleton-guided`` / ``temporal-skeleton`` walker chain.

The tool answers one question under strictly fixed inputs: given the *same*
stereo images, the *same* calibration, externally supplied left/right walker
masks and the *same* frozen stereo parameters, does constraining the mask to the
neighbourhood of its morphological skeleton reduce obviously anomalous thick
tubes / background bleed-through and give more interpretable 3-D rod
candidates?  There is no 3-D walker ground truth anywhere in this run, so no
field of this output may be called an accuracy improvement.

Three methods are evaluated per frame, on the identical images/calibration/
masks/frozen parameters:

``dense_component_pca``
    every strict stereo point inside each 2-D connected component of the
    supplied walker mask, fitted with one 3-D line primitive.

``skeleton_guided_sparse_stereo``
    the morphological skeleton of each supplied mask is dilated with an
    elliptical kernel of exactly ``--skeleton-band-radius-px`` and the band -
    and nothing else - replaces the mask handed to the strictly identical
    frozen stereo chain; the strict points are then grouped by the *original*
    skeleton's connected components.

``skeleton_guided_temporal_consensus``
    method B's strict points from a centred local window are reduced by the
    frozen camera-attached voxel consensus, and the consensus points are fitted
    with the same line logic.

There is deliberately no silent-fallback switch: when the skeleton branch
produces no candidate, that frame is reported as ``unavailable`` with its
reason and the dense result is never substituted for it.

Only ``numpy``, ``opencv-python`` and the repository's own modules are used.
No semantic model is trained, downloaded or replaced, and no pose, calibration
or frozen stereo parameter is touched.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import cv2
import numpy as np


TOOLS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_ROOT.parents[1]
REALTIME_ROOT = PROJECT_ROOT / "realtime_app"
for candidate in (str(REALTIME_ROOT), str(TOOLS_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from pose_app import walker_structure_variants as variants  # noqa: E402
from pose_app.calibration import StereoCalibration  # noqa: E402
from pose_app.scene_geometry_variants import (  # noqa: E402
    fit_line_primitive,
    morphological_skeleton,
    reconstruct_walker_components,
)
from benchmark_ground_walker_reconstruction import load_binary, stereo_candidates  # noqa: E402
import observe_local_ground_semantic_stereo_lr_direction_control as corrected  # noqa: E402


SCHEMA_VERSION = "walker_structure_variants_benchmark_v1"

METHOD_DENSE = "dense_component_pca"
METHOD_SKELETON = "skeleton_guided_sparse_stereo"
METHOD_TEMPORAL = "skeleton_guided_temporal_consensus"
METHODS = (METHOD_DENSE, METHOD_SKELETON, METHOD_TEMPORAL)

METHOD_INTERPRETATIONS = {
    METHOD_DENSE: variants.INTERPRETATION_DENSE_COMPONENT_PCA,
    METHOD_SKELETON: variants.INTERPRETATION_SKELETON_GUIDED,
    METHOD_TEMPORAL: variants.INTERPRETATION_TEMPORAL_CONSENSUS,
}

# Same connected-component minima the archived walker candidate benchmark used, so
# the dense and the skeleton branch cannot differ by an accidental threshold.
DENSE_MINIMUM_COMPONENT_PIXELS = 30
DENSE_MINIMUM_COMPONENT_POINTS = variants.SKELETON_COMPONENT_MINIMUM_POINTS

# Both branches hand their mask to the frozen chain without any *additional*
# matching dilation. The single mask difference is the band itself.
MATCHING_MASK_DILATION_PX = 0

TEMPORAL_VOXEL_SIZE_MM = 20.0

RECORD_BOUNDARY = (
    "Semantic evidence is an input variable of this run and is never upgraded: an automatic candidate mask stays "
    "an automatic candidate even when stereo or temporal geometry looks consistent. A fitted line candidate is a "
    "visible-region representation of whatever the supplied mask covered; it is not walker identity, not a "
    "complete CAD model, not a support point, not contact evidence, and not a 3-D accuracy measurement."
)

COLOURS_BGR = {
    "mask": (150, 70, 0),
    "band": (200, 190, 0),
    "skeleton": (0, 255, 255),
    "dense_pixels": (0, 140, 255),
    "skeleton_pixels": (255, 0, 255),
    METHOD_DENSE: (0, 220, 0),
    METHOD_SKELETON: (0, 0, 255),
    METHOD_TEMPORAL: (255, 0, 200),
}

PANEL_HEIGHT = 720
PANEL_GAP = 6


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--left-dir", type=Path, required=True, help="upright left pair_*.png images")
    parser.add_argument("--right-dir", type=Path, required=True, help="upright right pair_*.png images")
    parser.add_argument("--left-mask-dir", type=Path, required=True, help="upright left walker mask directory")
    parser.add_argument("--right-mask-dir", type=Path, required=True, help="upright right walker mask directory")
    parser.add_argument(
        "--semantic-evidence", choices=("manually_audited", "automatic_candidate"), required=True,
        help="provenance of the supplied masks; recorded as an independent variable and never upgraded",
    )
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--frozen-baseline-dir", type=Path, required=True, help="frozen stereo parameter source")
    parser.add_argument("--output-dir", type=Path, required=True, help="new, previously non-existent result dir")
    parser.add_argument("--maximum-pairs", type=int, default=12, help="positive integer; lowest pair numbers first")
    parser.add_argument("--temporal-window", type=int, default=5, help="positive odd centred window in frames")
    parser.add_argument(
        "--skeleton-band-radius-px", type=int, default=3,
        help="positive integer radius of the elliptical skeleton band; the only new structural parameter",
    )
    return parser.parse_args(argv)


def pair_id(name: str) -> int:
    stem = Path(name).stem
    suffix = stem.removeprefix("pair_")
    if suffix == stem or not suffix.isdigit():
        raise RuntimeError(f"frame name {name} is not of the form pair_<digits>.png")
    return int(suffix)


def png_names(directory: Path) -> set[str]:
    return {path.name for path in directory.glob("pair_*.png")}


def inventory(args: argparse.Namespace) -> dict[str, Any]:
    """Read-only count of images and masks, before anything is executed."""
    for label, directory in (
        ("--left-dir", args.left_dir), ("--right-dir", args.right_dir),
        ("--left-mask-dir", args.left_mask_dir), ("--right-mask-dir", args.right_mask_dir),
    ):
        if not directory.is_dir():
            raise RuntimeError(f"{label} is not an existing directory: {directory}")
    left_images = png_names(args.left_dir)
    right_images = png_names(args.right_dir)
    left_masks = png_names(args.left_mask_dir)
    right_masks = png_names(args.right_mask_dir)
    paired_images = left_images & right_images
    paired_masks = left_masks & right_masks
    usable = paired_images & paired_masks
    if not usable:
        raise RuntimeError(
            "no frame has a same-named left and right image *and* a same-named left and right walker mask: "
            f"left_images={len(left_images)}, right_images={len(right_images)}, "
            f"left_masks={len(left_masks)}, right_masks={len(right_masks)}, "
            f"paired_images={len(paired_images)}, paired_masks={len(paired_masks)}. "
            "No mask is copied between sides and no mask is generated by this tool."
        )
    return {
        "left_image_count": len(left_images),
        "right_image_count": len(right_images),
        "left_mask_count": len(left_masks),
        "right_mask_count": len(right_masks),
        "paired_image_count": len(paired_images),
        "paired_mask_count": len(paired_masks),
        "usable_frame_count": len(usable),
        "usable_names": usable,
        "left_only_mask_names": sorted(left_masks - right_masks),
        "right_only_mask_names": sorted(right_masks - left_masks),
    }


def select_frame_names(found: dict[str, Any], maximum_pairs: int) -> list[str]:
    if int(maximum_pairs) < 1:
        raise ValueError("--maximum-pairs must be a positive integer")
    names = sorted(found["usable_names"], key=pair_id)
    return names[: int(maximum_pairs)]


def frame_mask_status(left_mask: np.ndarray, right_mask: np.ndarray, name: str) -> dict[str, Any]:
    left_pixels = int(np.count_nonzero(left_mask))
    right_pixels = int(np.count_nonzero(right_mask))
    reasons: list[str] = []
    if left_pixels == 0:
        reasons.append("empty_left_walker_mask")
    if right_pixels == 0:
        reasons.append("empty_right_walker_mask")
    return {
        "frame_id": name,
        "left_mask_file_present": True,
        "right_mask_file_present": True,
        "left_mask_pixels": left_pixels,
        "right_mask_pixels": right_pixels,
        "left_mask_area_fraction": float(left_pixels / left_mask.size),
        "right_mask_area_fraction": float(right_pixels / right_mask.size),
        "both_nonempty": not reasons,
        "reasons": reasons,
        "identity_boundary": (
            "A mask only conditions 2-D region identity. Neither submitted side is 3-D walker truth, and a "
            "floor_eligible mask is never accepted here as a walker mask."
        ),
    }


def funnel_ratios(funnel: dict[str, Any]) -> dict[str, float | None]:
    """Stage-to-stage survival of the frozen matching funnel, per method.

    Reported so a low candidate count can be attributed to a specific frozen
    gate instead of being blamed on the line fit.  These are fractions of the
    method's own mask pixels and are not a quality or accuracy measure.
    """
    mask_pixels = int(funnel.get("left_mask_pixels", 0) or 0)
    forward = int(funnel.get("forward_valid", 0) or 0)
    right_valid = int(funnel.get("right_mask_valid", 0) or 0)
    reverse_valid = int(funnel.get("reverse_valid", 0) or 0)
    consistent = int(funnel.get("lr_consistent", 0) or 0)

    def ratio(numerator: int, denominator: int) -> float | None:
        return None if denominator <= 0 else float(numerator / denominator)

    return {
        "forward_valid_of_mask_pixels": ratio(forward, mask_pixels),
        "right_mask_correspondence_of_forward_valid": ratio(right_valid, forward),
        "reverse_valid_of_right_mask_valid": ratio(reverse_valid, right_valid),
        "lr_consistent_of_reverse_valid": ratio(consistent, reverse_valid),
        "final_candidates_of_mask_pixels": ratio(
            int(funnel.get("final_candidates", 0) or 0), mask_pixels
        ),
        "definition": (
            "stage survival inside the frozen candidate chain: mask pixels -> forward disparity -> partner inside "
            "the other side's mask -> reverse disparity valid -> left/right consistency. It is a diagnostic of the "
            "supplied mask geometry, not a quality or accuracy score."
        ),
    }


def component_pixel_indices(
    pixels: np.ndarray, labels: np.ndarray, component_id: int, shape: tuple[int, int]
) -> np.ndarray:
    """Mirror of ``reconstruct_walker_components``' own pixel-to-component lookup."""
    array = np.asarray(pixels, dtype=np.int64).reshape(-1, 2)
    height, width = int(shape[0]), int(shape[1])
    if not len(array):
        return np.empty(0, dtype=np.int64)
    inside = (
        (array[:, 0] >= 0) & (array[:, 0] < width) & (array[:, 1] >= 0) & (array[:, 1] < height)
    )
    clipped = np.clip(array, (0, 0), (width - 1, height - 1))
    return np.flatnonzero(inside & (labels[clipped[:, 1], clipped[:, 0]] == int(component_id)))


def dense_method_record(candidate: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    """Method A on one frame, built on the existing component reconstruction."""
    base: dict[str, Any] = {
        "method": METHOD_DENSE,
        "interpretation": METHOD_INTERPRETATIONS[METHOD_DENSE],
        "matching_mask_dilation_px": MATCHING_MASK_DILATION_PX,
        "minimum_component_pixels": DENSE_MINIMUM_COMPONENT_PIXELS,
        "minimum_component_points": DENSE_MINIMUM_COMPONENT_POINTS,
    }
    if candidate.get("status") != "candidate":
        reasons = list(candidate.get("reasons", ["stereo_candidate_stage_unavailable"]))
        return {
            **base,
            "status": "unavailable",
            "reasons": reasons,
            "components": [],
            "line_primitives": [],
            "failure_reasons": dict(Counter(reasons)),
            "input_points": {
                "count": 0,
                "kind": "strict_stereo_points",
                "definition": "strict stereo candidate pixels of this method's own mask under the frozen chain",
            },
            "matching_funnel": dict(candidate.get("matching_funnel", {})),
            "matching_funnel_stage_ratios": funnel_ratios(candidate.get("matching_funnel", {})),
            "mask_seed_upright_px": None,
        }
    points = np.asarray(candidate["points"], dtype=np.float64)
    pixels = np.asarray(candidate["left_pixels"], dtype=np.int64)
    mask = candidate["left_mask_for_matching"]
    model = reconstruct_walker_components(
        points, pixels, mask,
        minimum_component_pixels=DENSE_MINIMUM_COMPONENT_PIXELS,
        minimum_component_points=DENSE_MINIMUM_COMPONENT_POINTS,
    )
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    _, labels, _, _ = cv2.connectedComponentsWithStats(binary, 8)
    components: list[dict[str, Any]] = []
    for record in model["components"]:
        component_id = int(record["component_id"])
        indices = component_pixel_indices(pixels, labels, component_id, binary.shape)
        if int(record["stereo_point_count"]) != int(len(indices)):
            raise RuntimeError(
                "dense component mirror disagrees with reconstruct_walker_components; "
                "refusing to report residual fields from an inconsistent mirror"
            )
        entry: dict[str, Any] = {
            "component_id": component_id,
            "mask_area_px": int(record["mask_area_px"]),
            "stereo_point_count": int(record["stereo_point_count"]),
            "status": str(record["status"]),
            "reasons": list(record.get("reasons", [])),
        }
        if record["status"] == "candidate":
            primitive = fit_line_primitive(points[indices], component_id)
            reported = record["primitive"]
            if not np.allclose(primitive.center_left_camera_mm, reported["center_left_camera_mm"]):
                raise RuntimeError("dense line refit disagrees with the frozen component reconstruction")
            if int(primitive.point_count) != int(reported["point_count"]):
                raise RuntimeError("dense line refit point count disagrees with the frozen reconstruction")
            entry.update(variants.line_record(primitive, points[indices]))
        components.append(entry)
    primitives = [entry for entry in components if entry["status"] == "candidate"]
    failures = [entry for entry in components if entry["status"] != "candidate"]
    reasons = sorted({reason for entry in failures for reason in entry.get("reasons", [])})
    if not primitives and not reasons:
        reasons = ["no_reconstructable_walker_component"]
    return {
        **base,
        "status": "candidate" if primitives else "unavailable",
        "reasons": [] if primitives else reasons,
        "components": components,
        "line_primitives": primitives,
        "failure_reasons": dict(
            Counter(reason for entry in failures for reason in entry.get("reasons", []))
        ),
        "input_points": {
            "count": int(len(points)),
            "kind": "strict_stereo_points",
            "definition": "strict stereo candidate pixels of this method's own mask under the frozen chain",
        },
        "matching_funnel": dict(candidate.get("matching_funnel", {})),
        "matching_funnel_stage_ratios": funnel_ratios(candidate.get("matching_funnel", {})),
        "mask_seed_upright_px": None,
        "component_count": len(components),
    }


def skeleton_method(
    *,
    left_upright: np.ndarray,
    right_upright: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    calibration: StereoCalibration,
    parameters: dict[str, Any],
    band_radius_px: int,
    stereo_fn: Callable[..., dict[str, Any]] = stereo_candidates,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Method B on one frame; returns the record and the state method C needs."""
    skeleton_left = morphological_skeleton(left_mask)
    skeleton_right = morphological_skeleton(right_mask)
    band_left = variants.skeleton_band(skeleton_left, band_radius_px)
    band_right = variants.skeleton_band(skeleton_right, band_radius_px)
    labels_left, stats_left = variants.skeleton_component_labels(skeleton_left)
    lengths_left = variants.skeleton_component_lengths_px(stats_left)
    counts = {
        "left_skeleton_pixels": int(cv2.countNonZero(skeleton_left)),
        "right_skeleton_pixels": int(cv2.countNonZero(skeleton_right)),
        "left_skeleton_band_pixels": int(cv2.countNonZero(band_left)),
        "right_skeleton_band_pixels": int(cv2.countNonZero(band_right)),
        "left_skeleton_component_count": int(len(stats_left) - 1),
        "skeleton_length_definition": (
            "per-component 2-D skeleton length in upright pixels, counted as the pixel count of that 8-connected "
            "component of the morphological skeleton"
        ),
    }
    record: dict[str, Any] = {
        "method": METHOD_SKELETON,
        "interpretation": METHOD_INTERPRETATIONS[METHOD_SKELETON],
        "skeleton_band_radius_px": int(band_radius_px),
        "skeleton_band_kernel": "cv2.MORPH_ELLIPSE",
        "band_domain": (
            "the band is built on the externally supplied upright mask; the frozen chain's own rotate/resize/"
            "remap then applies to the band exactly as it does to the dense mask"
        ),
        "matching_mask_dilation_px": MATCHING_MASK_DILATION_PX,
        "minimum_component_points": variants.SKELETON_COMPONENT_MINIMUM_POINTS,
        **counts,
    }
    state: dict[str, Any] = {
        "points": np.empty((0, 3), dtype=np.float64),
        "upright": np.empty((0, 2), dtype=np.float64),
        "component_id": np.empty(0, dtype=np.int64),
        "labels_left": labels_left,
        "skeleton_lengths_left": lengths_left,
        "skeleton_left": skeleton_left,
        "skeleton_right": skeleton_right,
        "band_left": band_left,
        "band_right": band_right,
        "candidate": None,
    }
    candidate = stereo_fn(
        left_upright, right_upright, band_left, band_right, calibration, parameters,
        mask_dilation_px=MATCHING_MASK_DILATION_PX,
    )
    state["skeleton_candidate"] = candidate
    if candidate.get("status") != "candidate":
        reasons = list(candidate.get("reasons", ["stereo_candidate_stage_unavailable"]))
        record.update({
            "status": "unavailable",
            "reasons": reasons,
            "line_primitives": [],
            "components": [],
            "component_count": 0,
            "failure_reasons": dict(Counter(reasons)),
            "input_points": {
                "count": 0,
                "kind": "strict_stereo_points",
                "definition": "strict stereo candidate pixels of this method's own skeleton band mask",
            },
            "matching_funnel": dict(candidate.get("matching_funnel", {})),
            "matching_funnel_stage_ratios": funnel_ratios(candidate.get("matching_funnel", {})),
            "mask_seed_upright_px": None,
        })
        state["candidate"] = candidate
        return record, state
    state["candidate"] = candidate
    points = np.asarray(candidate["points"], dtype=np.float64)
    rectification = candidate["rectification"]
    runtime_size = (int(parameters["runtime_width"]), int(parameters["runtime_height"]))
    upright_shape = (int(left_upright.shape[0]), int(left_upright.shape[1]))
    upright = variants.local_pixels_to_upright(
        candidate["left_pixels"], rectification.left_map_x, rectification.left_map_y,
        runtime_size=runtime_size, upright_shape=upright_shape, side="left",
    )
    assignment = variants.nearest_skeleton_components(
        upright, labels_left, band_radius_px=int(band_radius_px)
    )
    component_id = np.asarray(assignment["component_id"], dtype=np.int64)
    state.update({"points": points, "upright": upright, "component_id": component_id})
    metadata: dict[int, dict[str, Any]] = {}
    for identifier in sorted({int(value) for value in component_id if int(value) > 0}):
        ys, xs = np.nonzero(labels_left == identifier)
        metadata[identifier] = {
            "skeleton_length_px": float(lengths_left.get(identifier, 0.0)),
            "skeleton_bbox_upright_xywh": [
                int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1),
            ] if len(xs) else None,
        }
    rejections = {
        identifier: "degenerate_component_geometry"
        for identifier, item in metadata.items()
        if float(item["skeleton_length_px"]) <= 0.0
    }
    grouped = variants.grouped_line_candidates(
        points, component_id,
        minimum_points=variants.SKELETON_COMPONENT_MINIMUM_POINTS,
        group_metadata=metadata, group_reasons=rejections,
    )
    primitives = grouped["candidates"]
    failures = grouped["failures"]
    failure_reasons = Counter(reason for entry in failures for reason in entry.get("reasons", []))
    reasons: list[str] = []
    if not primitives:
        if not metadata:
            reasons = ["no_skeleton_component"]
        elif int(len(points)) == 0:
            reasons = ["no_strict_stereo_point_from_skeleton_band"]
        elif not any(int(value) > 0 for value in component_id):
            reasons = ["no_skeleton_component_matched_to_a_strict_stereo_point"]
        else:
            reasons = sorted(failure_reasons) or ["no_reconstructable_skeleton_component"]
    unfilled = int(len(points)) - sum(
        int(entry["stereo_point_count"]) for entry in primitives + failures
    )
    record.update({
        "status": "candidate" if primitives else "unavailable",
        "reasons": [] if primitives else reasons,
        "line_primitives": primitives,
        "components": primitives + failures,
        "component_count": len(primitives) + len(failures),
        "failure_reasons": dict(failure_reasons),
        "unassigned_strict_point_pixels": int(assignment["unassigned"]),
        "strict_points_without_skeleton_component": int(unfilled),
        "input_points": {
            "count": int(len(points)),
            "kind": "strict_stereo_points",
            "definition": "strict stereo candidate pixels of this method's own skeleton band mask",
        },
        "matching_funnel": dict(candidate.get("matching_funnel", {})),
        "matching_funnel_stage_ratios": funnel_ratios(candidate.get("matching_funnel", {})),
        "mask_seed_upright_px": (
            None if variants.mask_seed_upright(band_left) is None
            else variants.mask_seed_upright(band_left).tolist()
        ),
        "component_assignment_rule": (
            "each strict stereo point is placed in the upright mask domain through the analytic inverse of the "
            "frozen rotate/resize/remap chain and assigned to the component owning the closest skeleton pixel "
            "(Euclidean, upright pixels); exact ties go to the smaller component id, and a point with no skeleton "
            "pixel inside the band radius plus a small search margin stays unassigned and is counted"
        ),
    })
    return record, state


def temporal_method(
    *,
    frame_index: int,
    window_indices: list[int],
    requested_window: int,
    states: list[dict[str, Any]],
    band_radius_px: int,
    line_count_skeleton: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """Method C on one centre frame, using method B's points only."""
    center = states[frame_index]
    point_sets = [states[index]["points"] for index in window_indices]
    minimum_support = max(2, math.ceil(len(window_indices) / 2))
    consensus = variants.consensus_with_provenance(
        point_sets, voxel_size_mm=TEMPORAL_VOXEL_SIZE_MM, minimum_frame_support=minimum_support
    )
    points = np.asarray(consensus["consensus_points_left_camera_mm"], dtype=np.float64).reshape(-1, 3)
    record: dict[str, Any] = {
        "method": METHOD_TEMPORAL,
        "interpretation": METHOD_INTERPRETATIONS[METHOD_TEMPORAL],
        "voxel_size_mm": TEMPORAL_VOXEL_SIZE_MM,
        "minimum_frame_support": int(minimum_support),
        "minimum_frame_support_definition": "max(2, ceil(actual window frame count / 2))",
        "window_frame_indices": [int(value) for value in window_indices],
        "window_frame_count": int(len(window_indices)),
        "window_frame_ids": [states[index]["frame_id"] for index in window_indices],
        "window_is_clipped_at_sequence_edge": bool(len(window_indices) < int(requested_window)),
        "input_point_count": int(consensus["input_point_count"]),
        "consensus_voxel_count": int(consensus["consensus_voxel_count"]),
        "consensus_status": str(consensus["status"]),
        "consensus_reasons": list(consensus.get("reasons", [])),
        "input_points": {
            "count": int(consensus["consensus_voxel_count"]),
            "kind": "temporal_consensus_points",
            "definition": (
                "accepted 20 mm voxel medians in the left-camera frame over this centre frame's local window; the "
                "window input total is reported separately as input_point_count"
            ),
        },
        "matching_mask_dilation_px": MATCHING_MASK_DILATION_PX,
        "minimum_component_points": variants.SKELETON_COMPONENT_MINIMUM_POINTS,
    }
    if len(points) == 0:
        return {
            **record,
            "status": "unavailable",
            "reasons": list(consensus.get("reasons", ["no_temporally_repeated_geometry"])),
            "line_primitives": [],
            "components": [],
            "component_count": 0,
            "failure_reasons": {},
            "consensus_points_without_skeleton_component": 0,
            "line_count_delta_vs_skeleton": int(-line_count_skeleton),
        }, points
    locations = np.asarray(
        [
            np.median(
                np.asarray(
                    [states[window_indices[frame]]["upright"][point] for frame, point in members],
                    dtype=np.float64,
                ),
                axis=0,
            )
            for members in consensus["voxel_members"]
        ],
        dtype=np.float64,
    ).reshape(-1, 2)
    assignment = variants.nearest_skeleton_components(
        locations, center["labels_left"], band_radius_px=int(band_radius_px)
    )
    component_id = np.asarray(assignment["component_id"], dtype=np.int64)
    metadata = {
        identifier: {
            "skeleton_length_px": float(center["skeleton_lengths_left"].get(identifier, 0.0)),
        }
        for identifier in sorted({int(value) for value in component_id if int(value) > 0})
    }
    grouped = variants.grouped_line_candidates(
        points, component_id,
        minimum_points=variants.SKELETON_COMPONENT_MINIMUM_POINTS,
        group_metadata=metadata,
    )
    primitives = grouped["candidates"]
    failures = grouped["failures"]
    failure_reasons = Counter(reason for entry in failures for reason in entry.get("reasons", []))
    without_component = int((component_id <= 0).sum())
    reasons: list[str] = []
    if not primitives:
        if not metadata:
            reasons = ["no_skeleton_component"]
        else:
            reasons = sorted(failure_reasons) or ["no_reconstructable_skeleton_component"]
    return {
        **record,
        "status": "candidate" if primitives else "unavailable",
        "reasons": [] if primitives else reasons,
        "line_primitives": primitives,
        "components": primitives + failures,
        "component_count": len(primitives) + len(failures),
        "failure_reasons": dict(failure_reasons),
        "consensus_points_without_skeleton_component": without_component,
        "line_count_delta_vs_skeleton": int(len(primitives) - int(line_count_skeleton)),
        "component_assignment_rule": (
            "each accepted voxel is located in the centre frame's upright mask domain as the median of its "
            "member pixels' upright coordinates and then inherits the nearest centre-frame skeleton component "
            "under the same nearest/tie rule as the skeleton method; voxels with no skeleton component in reach "
            "are counted and excluded"
        ),
    }, points


def evaluate_frame(
    *,
    name: str,
    left_upright: np.ndarray,
    right_upright: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    calibration: StereoCalibration | None,
    parameters: dict[str, Any],
    semantic_evidence: str,
    band_radius_px: int,
    stereo_fn: Callable[..., dict[str, Any]] = stereo_candidates,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run methods A and B on one frame and return the record plus shared state."""
    status = frame_mask_status(left_mask, right_mask, name)
    dense_candidate = stereo_fn(
        left_upright, right_upright, left_mask, right_mask, calibration, parameters,
        mask_dilation_px=MATCHING_MASK_DILATION_PX,
    )
    dense = dense_method_record(dense_candidate, parameters)
    if dense_candidate.get("status") == "candidate":
        dense["mask_seed_upright_px"] = (
            None if variants.mask_seed_upright(left_mask) is None
            else variants.mask_seed_upright(left_mask).tolist()
        )
    skeleton, state = skeleton_method(
        left_upright=left_upright, right_upright=right_upright,
        left_mask=left_mask, right_mask=right_mask,
        calibration=calibration, parameters=parameters,
        band_radius_px=band_radius_px, stereo_fn=stereo_fn,
    )
    record = {
        "frame_id": name,
        "pair_id": pair_id(name),
        "semantic_evidence": semantic_evidence,
        "semantic_evidence_upgraded": False,
        "mask_status": status,
        "methods": {METHOD_DENSE: dense, METHOD_SKELETON: skeleton},
        "interpretation_boundary": RECORD_BOUNDARY,
    }
    state.update({
        "frame_id": name,
        "pair_id": pair_id(name),
        "left_upright": left_upright,
        "right_upright": right_upright,
        "left_mask": left_mask,
        "right_mask": right_mask,
        "dense_candidate": dense_candidate,
        "dense_record": dense,
        "skeleton_record": skeleton,
    })
    return record, state


def write_ply(path: Path, points: np.ndarray) -> None:
    """ASCII PLY with the left-camera millimetre coordinates."""
    array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(array)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "end_header\n"
        )
        for x, y, z in array:
            handle.write(f"{x:.4f} {y:.4f} {z:.4f}\n")


def project_primitives_to_upright(
    primitives: Sequence[dict[str, Any]], calibration: StereoCalibration, upright_shape: tuple[int, int]
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Project 3-D segment endpoints back to upright left pixels for display only."""
    if not primitives:
        return []
    endpoints = np.asarray(
        [
            [entry["endpoint_min_left_camera_mm"], entry["endpoint_max_left_camera_mm"]]
            for entry in primitives
        ],
        dtype=np.float64,
    ).reshape(-1, 3)
    raw = calibration.project_left(endpoints)
    upright = variants.raw_points_to_upright(raw, upright_shape=upright_shape, side="left")
    segments = []
    for index in range(len(primitives)):
        first = tuple(int(round(value)) for value in upright[2 * index])
        second = tuple(int(round(value)) for value in upright[2 * index + 1])
        segments.append((first, second))
    return segments


def _blend(image: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int], alpha: float) -> np.ndarray:
    """Tint only the selected pixels; the rest of the frame is left untouched."""
    selected = np.asarray(mask) > 0
    if not np.any(selected):
        return image
    output = image.copy()
    tinted = (
        (1.0 - float(alpha)) * image[selected].astype(np.float64)
        + float(alpha) * np.asarray(colour, dtype=np.float64)
    )
    output[selected] = np.clip(tinted, 0.0, 255.0).astype(np.uint8)
    return output


def _label(panel: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 20), (255, 255, 255), -1)
    cv2.putText(panel, text[:150], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)
    return panel


def render_panel(
    *,
    name: str,
    state: dict[str, Any],
    temporal: dict[str, Any],
    calibration: StereoCalibration,
    runtime_size: tuple[int, int],
) -> np.ndarray:
    """Three-column sheet: left evidence, right evidence, three methods on the left."""
    upright_shape = (int(state["left_upright"].shape[0]), int(state["left_upright"].shape[1]))
    dense_candidate = state["dense_candidate"] if state["dense_candidate"].get("status") == "candidate" else None
    skeleton_candidate = (
        state["skeleton_candidate"] if state["skeleton_candidate"] is not None
        and state["skeleton_candidate"].get("status") == "candidate" else None
    )

    def transported(candidate: dict[str, Any] | None, side: str) -> np.ndarray:
        if candidate is None:
            return np.empty((0, 2), dtype=np.float64)
        rectification = candidate["rectification"]
        map_x = rectification.left_map_x if side == "left" else rectification.right_map_x
        map_y = rectification.left_map_y if side == "left" else rectification.right_map_y
        key = "left_pixels" if side == "left" else "right_pixels"
        return variants.local_pixels_to_upright(
            candidate[key], map_x, map_y,
            runtime_size=runtime_size, upright_shape=upright_shape, side=side,
        )

    panels: list[np.ndarray] = []
    for side in ("left", "right"):
        image = state[f"{side}_upright"].copy()
        mask = state[f"{side}_mask"]
        skeleton = state["skeleton_left"] if side == "left" else state["skeleton_right"]
        band = state["band_left"] if side == "left" else state["band_right"]
        panel = _blend(image, mask, COLOURS_BGR["mask"], 0.30)
        panel = _blend(panel, band, COLOURS_BGR["band"], 0.45)
        panel = _blend(panel, variants.skeleton_band(skeleton, 1), COLOURS_BGR["skeleton"], 0.95)
        panel = _blend(panel, variants.scatter_upright_mask(transported(dense_candidate, side), upright_shape),
                       COLOURS_BGR["dense_pixels"], 0.95)
        panel = _blend(panel, variants.scatter_upright_mask(transported(skeleton_candidate, side), upright_shape),
                       COLOURS_BGR["skeleton_pixels"], 0.95)
        panels.append(_label(
            panel,
            f"{name} {side}: blue=walker mask cyan=skeleton band (r=3) yellow=skeleton (3 px display stroke)",
        ))
    # third panel: left image plus the three methods' projected 2-D segments
    panel = _blend(state["left_upright"].copy(), state["left_mask"], COLOURS_BGR["mask"], 0.22)
    for method in METHODS:
        primitives = state["methods"][method]["line_primitives"] if method in state["methods"] else temporal["line_primitives"]
        segments = project_primitives_to_upright(primitives, calibration, upright_shape)
        colour = COLOURS_BGR[method]
        for first, second in segments:
            cv2.line(panel, first, second, colour, 4, cv2.LINE_AA)
            cv2.circle(panel, first, 5, colour, -1)
            cv2.circle(panel, second, 5, colour, -1)
    rows = []
    for method in METHODS:
        entry = state["methods"].get(method, temporal)
        label = "candidate" if entry["line_primitives"] else "unavailable"
        detail = ""
        if label == "unavailable" and entry.get("reasons"):
            detail = " " + ",".join(entry["reasons"])[:60]
        rows.append(f"{method}: {len(entry['line_primitives'])} line(s) {label}{detail}")
    panel = _label(panel, f"{name} left: projected 3-D line candidates (A green / B red / C magenta)")
    for index, text in enumerate(rows):
        cv2.rectangle(panel, (0, 22 + 18 * index), (panel.shape[1], 38 + 18 * index), (255, 255, 255), -1)
        cv2.putText(panel, text[:150], (4, 35 + 18 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    COLOURS_BGR[METHODS[index]], 1, cv2.LINE_AA)
    panels.append(panel)
    scaled = []
    for item in panels:
        scale = PANEL_HEIGHT / item.shape[0]
        scaled.append(cv2.resize(item, (max(1, int(round(item.shape[1] * scale))), PANEL_HEIGHT),
                                 interpolation=cv2.INTER_AREA))
    separator = np.zeros((PANEL_HEIGHT, PANEL_GAP, 3), dtype=np.uint8)
    sheet = scaled[0]
    for item in scaled[1:]:
        sheet = np.hstack((sheet, separator, item))
    return sheet


def method_statistics(records: Sequence[dict[str, Any]], method: str) -> dict[str, Any]:
    outputs = [record["methods"][method] for record in records]
    line_counts = [len(entry["line_primitives"]) for entry in outputs]
    primitives = [item for entry in outputs for item in entry["line_primitives"]]
    failure_reasons: Counter = Counter()
    for entry in outputs:
        failure_reasons.update(entry.get("failure_reasons", {}))
        if entry["status"] == "unavailable" and not entry.get("failure_reasons"):
            failure_reasons.update(entry.get("reasons", []))
    return {
        "method": method,
        "input_frames": len(records),
        "frames_with_both_nonempty_masks": int(sum(record["mask_status"]["both_nonempty"] for record in records)),
        "frames_with_any_input_points": int(sum(entry["input_points"]["count"] > 0 for entry in outputs)),
        "input_point_count_median_per_frame": float(
            np.median([entry["input_points"]["count"] for entry in outputs])
        ) if outputs else None,
        "window_input_point_total_median_per_frame": variants.distribution(
            [entry.get("input_point_count") for entry in outputs]
        )["median"],
        "frames_with_at_least_one_line_candidate": int(sum(count > 0 for count in line_counts)),
        "median_line_candidates_per_frame": float(np.median(line_counts)) if line_counts else None,
        "total_line_candidates": int(len(primitives)),
        "line_length_mm": variants.distribution([item["length_mm"] for item in primitives]),
        "line_fit_residual_median_mm": variants.distribution(
            [item["fit_residual_median_mm"] for item in primitives]
        ),
        "line_fit_residual_p90_mm": variants.distribution(
            [item["fit_residual_p90_mm"] for item in primitives]
        ),
        "line_radius_median_mm": variants.distribution([item["radius_median_mm"] for item in primitives]),
        "line_radius_p90_mm": variants.distribution([item["radius_p90_mm"] for item in primitives]),
        "failure_reason_counts": dict(sorted(failure_reasons.items())),
        "line_candidate_count_per_frame": line_counts,
        "matching_funnel_right_mask_correspondence_of_forward_valid": variants.distribution(
            [
                entry.get("matching_funnel_stage_ratios", {}).get(
                    "right_mask_correspondence_of_forward_valid"
                )
                for entry in outputs
            ]
        ),
        "statistics_boundary": variants.STATISTICS_BOUNDARY,
        "interpretation": METHOD_INTERPRETATIONS[method],
    }


def availability_comparison(records: Sequence[dict[str, Any]], first: str, second: str) -> dict[str, Any]:
    both, only_first, only_second, neither = [], [], [], []
    differences = []
    for record in records:
        name = record["frame_id"]
        first_lines = len(record["methods"][first]["line_primitives"])
        second_lines = len(record["methods"][second]["line_primitives"])
        differences.append(second_lines - first_lines)
        if first_lines and second_lines:
            both.append(name)
        elif first_lines:
            only_first.append(name)
        elif second_lines:
            only_second.append(name)
        else:
            neither.append(name)
    return {
        "first_method": first,
        "second_method": second,
        "frames": len(records),
        "both_available_frames": len(both),
        "only_first_available_frames": len(only_first),
        "only_first_frames": only_first,
        "only_second_available_frames": len(only_second),
        "only_second_frames": only_second,
        "neither_available_frames": len(neither),
        "neither_frames": neither,
        "line_count_difference_second_minus_first": variants.distribution(differences),
        "boundary": (
            "Same-frame availability of this tool's own line candidates only. A difference is not an accuracy, "
            "recall or recognition difference."
        ),
    }


def build_summary(
    records: Sequence[dict[str, Any]], run_parameters: dict[str, Any]
) -> dict[str, Any]:
    evidence = sorted({str(record["semantic_evidence"]) for record in records})
    if len(evidence) > 1:
        raise RuntimeError(
            "frames carry more than one mask provenance in a single run; refusing to aggregate them together"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "frame_count": len(records),
        "semantic_evidence": evidence[0] if evidence else run_parameters.get("semantic_evidence"),
        "semantic_evidence_note": (
            "The mask provenance is an input variable of this comparison and is recorded verbatim. "
            "automatic_candidate is never promoted to manually_audited, and manually_audited only means an "
            "audited 2-D visible region, never a 3-D walker measurement."
        ),
        "run_parameters": run_parameters,
        "method_statistics": {
            method: method_statistics(records, method) for method in METHODS
        },
        "availability_differences": {
            "dense_vs_skeleton": availability_comparison(records, METHOD_DENSE, METHOD_SKELETON),
            "skeleton_vs_temporal": availability_comparison(records, METHOD_SKELETON, METHOD_TEMPORAL),
        },
        "frame_ids": [record["frame_id"] for record in records],
        "interpretation_boundary": RECORD_BOUNDARY,
        "statistics_boundary": variants.STATISTICS_BOUNDARY,
    }


def _format_number(value: Any, digits: int = 2) -> Any:
    """Fixed-point rendering that keeps the empty case visible as ``None``."""
    return None if value is None else "{:.{digits}f}".format(float(value), digits=digits)


def render_result_diagnosis(summary: dict[str, Any]) -> list[str]:
    """Data-driven reading of a completed run, with every claim kept internal."""
    stats = summary["method_statistics"]
    dense, skeleton, temporal = (
        stats[METHOD_DENSE], stats[METHOD_SKELETON], stats[METHOD_TEMPORAL]
    )
    dense_points = dense["input_point_count_median_per_frame"]
    skeleton_points = skeleton["input_point_count_median_per_frame"]
    retained = None if not dense_points else float(skeleton_points) / float(dense_points)
    dense_funnel = dense["matching_funnel_right_mask_correspondence_of_forward_valid"]["median"]
    skeleton_funnel = skeleton["matching_funnel_right_mask_correspondence_of_forward_valid"]["median"]
    difference = summary["availability_differences"]
    return [
        "## 结果诊断（由 `summary.json` 计算，随运行写入）",
        "",
        "- 每帧严格三维输入点中位：方法 A `{}`，方法 B `{}`{}。".format(
            _format_number(dense_points, 1), _format_number(skeleton_points, 1),
            "" if retained is None else "（B 保留 A 的 `{:.3%}`）".format(retained),
        ),
        "- 冻结候选链“对侧掩膜对应”存活率中位（`right_mask_valid / forward_valid`）：A `{}`，B `{}`。".format(
            _format_number(dense_funnel, 4), _format_number(skeleton_funnel, 4),
        ),
        "  该比值只描述所提交掩膜在冻结门内的几何存活，不是质量或准确率。它把方法 B 点稀少的原因定位到共享的双侧掩膜",
        "  互相对应门，而不是线段拟合、SGBM 参数或三角化规则。",
        "- 线段长度中位：A `{}` mm，B `{}` mm；半径与拟合残差中位：A `{}` mm，B `{}` mm。".format(
            _format_number(dense["line_length_mm"]["median"], 1),
            _format_number(skeleton["line_length_mm"]["median"], 1),
            _format_number(dense["line_radius_median_mm"]["median"], 1),
            _format_number(skeleton["line_radius_median_mm"]["median"], 1),
        ),
        "  在没有三维助步器真值的前提下，本次数据只能给出内部几何事实：A 的单一候选管在物理尺度上不可能是单根助步器杆件，",
        "  而 B 不再产生该管，改为若干短线段。B 的短线段同样没有经过助步器身份验证，因此两者都不能被称为更准确。",
        "- 方法 C：窗口内输入点总数中位 `{}`，共识体素中位 `{}`，有线段帧数 `{}`；体素共识只表示相机系中的重复观测。".format(
            _format_number(temporal["window_input_point_total_median_per_frame"], 1),
            _format_number(temporal["input_point_count_median_per_frame"], 1),
            temporal["frames_with_at_least_one_line_candidate"],
        ),
        "- 同帧可用性差异（本工具自身候选的可用性，不是识别成功率）：`dense vs skeleton` 双方可用 `{}` 帧、仅 A 可用 `{}` 帧、".format(
            difference["dense_vs_skeleton"]["both_available_frames"],
            difference["dense_vs_skeleton"]["only_first_available_frames"],
        ),
        "  仅 B 可用 `{}` 帧；`skeleton vs temporal` 双方可用 `{}` 帧、仅 B 可用 `{}` 帧、仅 C 可用 `{}` 帧。".format(
            difference["dense_vs_skeleton"]["only_second_available_frames"],
            difference["skeleton_vs_temporal"]["both_available_frames"],
            difference["skeleton_vs_temporal"]["only_first_available_frames"],
            difference["skeleton_vs_temporal"]["only_second_available_frames"],
        ),
        "",
    ]


def render_experiment_markdown(
    *, run_parameters: dict[str, Any], found: dict[str, Any], names: Sequence[str], summary: dict[str, Any]
) -> str:
    lines = [
        "# 助步器“稠密点云 vs 骨架引导 vs 时序骨架”受控对比",
        "",
        "## 目标与对照",
        "",
        "在完全相同的图像、标定、左右助步器掩膜和冻结双目参数下，比较三种由掩膜到三维助步器杆件候选的方法：",
        "",
        "- `dense_component_pca`：掩膜内每个二维连通域的全部严格双目点拟合一条三维线段。",
        "- `skeleton_guided_sparse_stereo`：掩膜形态学骨架膨胀成的骨架带替代掩膜送入同一冻结链，再按原始骨架连通组件分组。",
        "- `skeleton_guided_temporal_consensus`：只以方法 B 的三维点做短窗体素共识，再对共识点用同一套线段拟合。",
        "",
        "## 唯一变量与不变量",
        "",
        "- 唯一新增结构参数：`--skeleton-band-radius-px = {}`（圆形/椭圆形核，方法 B 与方法 C 共用）。".format(
            run_parameters["skeleton_band_radius_px"]
        ),
        "- 方法 B 相对方法 A 只改变**送入冻结链的二值掩膜**。SGBM 参数、左右一致性的反向视差方向、视差范围、",
        "  三角化规则、光度门和重投影规则全部继承冻结来源，未改动。两侧都不使用额外的匹配掩膜膨胀。",
        "- 掩膜来源通过命令行显式输入，是本次实验的独立变量；`automatic_candidate` 在任何情况下都不升级为",
        "  `manually_audited`，`manually_audited` 也只代表人工审计的二维可见区域。",
        "- 工具不提供静默回退开关：骨架分支失败时输出 `unavailable` 与原因，绝不用稠密分支的结果顶替。",
        "",
        "## 输入",
        "",
        f"- 左正立图像目录：`{run_parameters['left_dir']}`（{found['left_image_count']} 张 `pair_*.png`）",
        f"- 右正立图像目录：`{run_parameters['right_dir']}`（{found['right_image_count']} 张）",
        f"- 左助步器掩膜目录：`{run_parameters['left_mask_dir']}`（{found['left_mask_count']} 张）",
        f"- 右助步器掩膜目录：`{run_parameters['right_mask_dir']}`（{found['right_mask_count']} 张）",
        f"- 左右同名图像：{found['paired_image_count']}；左右同名掩膜：{found['paired_mask_count']}；",
        f"  四者同名可用帧：{found['usable_frame_count']}；本次实际运行 {len(names)} 帧。",
        f"- 掩膜语义证据：`{run_parameters['semantic_evidence']}`（记录为输入变量，不升级）。",
        f"- 标定：`{run_parameters['calibration']}`；冻结双目参数来源：`{run_parameters['frozen_baseline_dir']}`。",
        "",
        "## 参数",
        "",
        "```json",
        json.dumps(run_parameters, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 结果",
        "",
        "| 方法 | 有输入点帧数 | 有三维线段候选帧数 | 每帧线段数中位 | 线段总数 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for method in METHODS:
        stats = summary["method_statistics"][method]
        lines.append(
            "| `{}` | {} | {} | {} | {} |".format(
                method, stats["frames_with_any_input_points"],
                stats["frames_with_at_least_one_line_candidate"],
                stats["median_line_candidates_per_frame"], stats["total_line_candidates"],
            )
        )
    lines += [
        "",
        "同名可用性差异：",
        "",
        "```json",
        json.dumps(summary["availability_differences"], ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    lines += render_result_diagnosis(summary)
    lines += [
        "## 失败原因",
        "",
        "```json",
        json.dumps(
            {method: summary["method_statistics"][method]["failure_reason_counts"] for method in METHODS},
            ensure_ascii=False, indent=2,
        ),
        "```",
        "",
        "## 结论边界",
        "",
        "- 本文所有数字都是本工具自身候选输出的计数与分布；**不是**准确率、召回率、真实长度误差或助步器识别成功率。",
        "- 本次没有三维助步器真值，因此不得声明任何方法“精度提高”。骨架约束是否减少明显异常的粗管/背景混入，",
        "  只能由同帧可用性差异、失败原因和可视化逐帧核对来判断，不能由点更多或残差更小来升级。",
        "- 候选线段只是所提交掩膜覆盖区域的可见三维杆件代表，不是完整助步器 CAD、不是支撑点、不是接触证据。",
        "- 相机附着一致性只表示重复观测，不证明助步器刚体不动，也不证明助步器与地面接触。",
        "- 掩膜语义证据由输入决定：自动候选掩膜的双目/时序几何再自洽也不能升级为助步器身份真值。",
        "",
        "详细逐帧状态见 `frame_records.jsonl`，汇总见 `summary.json`，三栏核对图见 `visualizations/`，",
        "逐方法点云见 `pointclouds/`。",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    if args.temporal_window < 1 or args.temporal_window % 2 == 0:
        raise ValueError("--temporal-window must be a positive odd integer")
    if args.maximum_pairs < 1:
        raise ValueError("--maximum-pairs must be a positive integer")
    if args.skeleton_band_radius_px < 1:
        raise ValueError("--skeleton-band-radius-px must be a positive integer")
    found = inventory(args)
    names = select_frame_names(found, args.maximum_pairs)
    parameters, _ = corrected.load_frozen(args.frozen_baseline_dir)
    runtime_size = (int(parameters["runtime_width"]), int(parameters["runtime_height"]))
    calibration = StereoCalibration.load(args.calibration).for_runtime_sizes(runtime_size, runtime_size)

    args.output_dir.mkdir(parents=True)
    (args.output_dir / "visualizations").mkdir()
    (args.output_dir / "pointclouds").mkdir()

    half_window = (args.temporal_window - 1) // 2
    records: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    unreadable: list[dict[str, str]] = []
    for name in names:
        left = cv2.imread(str(args.left_dir / name), cv2.IMREAD_COLOR)
        right = cv2.imread(str(args.right_dir / name), cv2.IMREAD_COLOR)
        if left is None or right is None or left.shape[:2] != right.shape[:2]:
            unreadable.append({"frame_id": name, "reason": "unreadable_or_mismatched_stereo_pair"})
            continue
        left_mask = load_binary(args.left_mask_dir / name, left.shape[:2])
        right_mask = load_binary(args.right_mask_dir / name, right.shape[:2])
        record, state = evaluate_frame(
            name=name, left_upright=left, right_upright=right,
            left_mask=left_mask, right_mask=right_mask,
            calibration=calibration, parameters=parameters,
            semantic_evidence=args.semantic_evidence, band_radius_px=args.skeleton_band_radius_px,
        )
        records.append(record)
        states.append(state)
    if not records:
        raise RuntimeError(
            "no frame of the selected set produced a readable equal-size stereo pair: "
            f"{json.dumps(unreadable, ensure_ascii=False)}"
        )

    pointcloud_reasons: dict[str, list[str]] = {METHOD_DENSE: [], METHOD_SKELETON: [], METHOD_TEMPORAL: []}
    for index, (record, state) in enumerate(zip(records, states)):
        window_indices = [
            value for value in range(index - half_window, index + half_window + 1)
            if 0 <= value < len(states)
        ]
        temporal, temporal_points = temporal_method(
            frame_index=index, window_indices=window_indices, requested_window=int(args.temporal_window),
            states=states, band_radius_px=args.skeleton_band_radius_px,
            line_count_skeleton=len(state["skeleton_record"]["line_primitives"]),
        )
        record["methods"][METHOD_TEMPORAL] = temporal
        record["temporal_window"] = {
            "requested_window": int(args.temporal_window),
            "actual_frame_count": len(window_indices),
            "frame_ids": temporal["window_frame_ids"],
            "clipped_at_sequence_edge": bool(len(window_indices) < int(args.temporal_window)),
            "minimum_frame_support": temporal["minimum_frame_support"],
        }
        state["methods"] = record["methods"]
        state["temporal"] = temporal

        stem = Path(record["frame_id"]).stem
        clouds = {
            METHOD_DENSE: (
                np.asarray(state["dense_candidate"].get("points", np.empty((0, 3))), dtype=np.float64),
                f"{stem}_dense_points.ply",
            ),
            METHOD_SKELETON: (
                np.asarray(state["skeleton_candidate"].get("points", np.empty((0, 3)))
                           if state["skeleton_candidate"] is not None else np.empty((0, 3)), dtype=np.float64),
                f"{stem}_skeleton_points.ply",
            ),
            METHOD_TEMPORAL: (temporal_points, f"{stem}_temporal_consensus_points.ply"),
        }
        for method, (cloud, filename) in clouds.items():
            if len(cloud):
                write_ply(args.output_dir / "pointclouds" / filename, cloud)
                record["methods"][method]["pointcloud"] = {
                    "path": f"pointclouds/{filename}", "point_count": int(len(cloud)),
                }
            else:
                reason = (
                    "no_strict_stereo_points_entered_this_method"
                    if method != METHOD_TEMPORAL
                    else "no_temporally_repeated_geometry_in_this_window"
                )
                pointcloud_reasons[method].append(record["frame_id"])
                record["methods"][method]["pointcloud"] = {
                    "path": None, "point_count": 0, "reason": reason,
                    "note": "no empty PLY is written",
                }
        sheet = render_panel(
            name=record["frame_id"], state=state, temporal=temporal,
            calibration=calibration, runtime_size=runtime_size,
        )
        if not cv2.imwrite(str(args.output_dir / "visualizations" / record["frame_id"]), sheet):
            raise RuntimeError(f"cannot write visualization for {record['frame_id']}")

    run_parameters = {
        "semantic_evidence": args.semantic_evidence,
        "left_dir": str(args.left_dir.resolve()),
        "right_dir": str(args.right_dir.resolve()),
        "left_mask_dir": str(args.left_mask_dir.resolve()),
        "right_mask_dir": str(args.right_mask_dir.resolve()),
        "calibration": str(args.calibration.resolve()),
        "frozen_baseline_dir": str(args.frozen_baseline_dir.resolve()),
        "maximum_pairs": int(args.maximum_pairs),
        "temporal_window": int(args.temporal_window),
        "temporal_window_half": int(half_window),
        "temporal_window_rule": "centred; an edge frame uses the frames that actually exist and the count is recorded",
        "skeleton_band_radius_px": int(args.skeleton_band_radius_px),
        "skeleton_band_radius_px_definition": (
            "radius of the elliptical dilation applied to the morphological skeleton of the externally supplied "
            "upright mask; it is the only structural parameter this comparison adds"
        ),
        "temporal_voxel_size_mm": TEMPORAL_VOXEL_SIZE_MM,
        "temporal_minimum_frame_support_rule": "max(2, ceil(actual window frame count / 2))",
        "matching_mask_dilation_px": MATCHING_MASK_DILATION_PX,
        "dense_minimum_component_pixels": DENSE_MINIMUM_COMPONENT_PIXELS,
        "minimum_component_points": DENSE_MINIMUM_COMPONENT_POINTS,
        "skeleton_search_margin_px": variants.SKELETON_SEARCH_MARGIN_PX,
        "silent_fallback": "not_available",
        "frozen_stereo_parameters": parameters,
    }
    summary = build_summary(records, run_parameters)
    summary["input_inventory"] = {
        key: value for key, value in found.items() if key != "usable_names"
    }
    summary["unreadable_frames"] = unreadable
    summary["pointcloud_absent_frames"] = {
        method: sorted(values) for method, values in pointcloud_reasons.items()
    }

    with (args.output_dir / "frame_records.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "methods": list(METHODS),
        "method_definitions": {
            METHOD_DENSE: (
                "strict stereo candidates of the full supplied left/right walker masks, grouped by 2-D connected "
                "component of the frozen chain's own matching mask, one 3-D line primitive per component"
            ),
            METHOD_SKELETON: (
                "morphological skeleton of each supplied mask dilated by an elliptical kernel of exactly "
                "skeleton_band_radius_px; only that band is handed to the strictly identical frozen stereo chain; "
                "strict points are grouped by the original skeleton's connected components"
            ),
            METHOD_TEMPORAL: (
                "method B strict points only, reduced by the frozen camera-attached voxel consensus over the "
                "centred local window, then fitted with the same line logic"
            ),
        },
        "method_interpretations": METHOD_INTERPRETATIONS,
        "run_parameters": run_parameters,
        "input_inventory": summary["input_inventory"],
        "frame_names": [record["frame_id"] for record in records],
        "unreadable_frames": unreadable,
        "interpretation_boundary": RECORD_BOUNDARY,
        "statistics_boundary": variants.STATISTICS_BOUNDARY,
        "notes": [
            "the only difference between method A and method B is the binary mask handed to the frozen chain",
            "no SGBM setting, disparity range, left/right consistency direction or limit, triangulation rule, "
            "photometric limit, depth limit or reprojection rule was changed",
            "method C consumes only method B's own strict points and adds no semantic or geometric evidence",
            "no semantic segmentation model was trained, downloaded or replaced; no DA3 is used",
            "no pose model, association, calibration or frozen stereo parameter was modified",
            "no silent fallback exists: a failed skeleton branch stays unavailable with its reason",
        ],
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    command = (
        "python .\\realtime_app\\tools\\benchmark_walker_structure_variants.py "
        f"--left-dir \"{args.left_dir}\" --right-dir \"{args.right_dir}\" "
        f"--left-mask-dir \"{args.left_mask_dir}\" --right-mask-dir \"{args.right_mask_dir}\" "
        f"--semantic-evidence {args.semantic_evidence} --calibration \"{args.calibration}\" "
        f"--frozen-baseline-dir \"{args.frozen_baseline_dir}\" --output-dir \"{args.output_dir}\" "
        f"--maximum-pairs {args.maximum_pairs} --temporal-window {args.temporal_window} "
        f"--skeleton-band-radius-px {args.skeleton_band_radius_px}"
    )
    (args.output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (args.output_dir / "EXPERIMENT.md").write_text(
        render_experiment_markdown(
            run_parameters=run_parameters, found=found, names=[record["frame_id"] for record in records],
            summary=summary,
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "frames": len(records),
        "unreadable_frames": unreadable,
        "method_summary": {
            method: {
                "frames_with_any_input_points": summary["method_statistics"][method]["frames_with_any_input_points"],
                "frames_with_at_least_one_line_candidate":
                    summary["method_statistics"][method]["frames_with_at_least_one_line_candidate"],
                "total_line_candidates": summary["method_statistics"][method]["total_line_candidates"],
                "failure_reason_counts": summary["method_statistics"][method]["failure_reason_counts"],
            }
            for method in METHODS
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
