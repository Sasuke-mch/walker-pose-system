"""Skeleton-guided walker structure candidate primitives.

This module adds exactly one controlled structural variable on top of the
already frozen ``mask -> strict stereo`` chain: *which pixels of an externally
supplied walker mask are allowed to enter the identical stereo pipeline*.  It
never changes the matcher, the triangulation rule or the left/right consistency
limit, and it never turns a mask into semantic walker truth.

Everything here starts at the binary mask handed to
``benchmark_ground_walker_reconstruction.stereo_candidates``.  The functions
reused from :mod:`pose_app.scene_geometry_variants` are the frozen
morphological skeleton, the connected-component walker reconstruction, the line
primitive fit and the camera-attached temporal consensus.

Two coordinate domains are used and are never mixed silently:

* the *upright* mask domain, i.e. the externally supplied mask/image pixel grid
  (``mask.shape``); this is the domain of every connected component, skeleton
  length and 2-D pixel count;
* the *local* mask-directed rectified domain produced inside the frozen stereo
  chain (``runtime_width`` x ``runtime_height``); this is the domain of the
  strict stereo candidate pixels.

``local_pixels_to_upright`` inverts the frozen mask transform
(``rotate_mask_to_raw`` then ``cv2.resize`` then ``cv2.remap``) analytically from
the rectification maps, so a local candidate pixel can be located in the upright
mask domain without re-running or altering any part of the chain.

All metric quantities are millimetres in the left-camera frame; there is no
persistent world frame anywhere in this module.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from pose_app.scene_geometry_variants import (
    camera_attached_temporal_consensus,
    fit_line_primitive,
    morphological_skeleton,
)


__all__ = [
    "SCHEMA_VERSION",
    "SKELETON_COMPONENT_MINIMUM_POINTS",
    "SKELETON_SEARCH_MARGIN_PX",
    "INTERPRETATION_DENSE_COMPONENT_PCA",
    "INTERPRETATION_SKELETON_GUIDED",
    "INTERPRETATION_TEMPORAL_CONSENSUS",
    "STATISTICS_BOUNDARY",
    "distribution",
    "mask_seed_upright",
    "skeleton_band",
    "skeleton_component_labels",
    "skeleton_component_lengths_px",
    "local_pixels_to_upright",
    "upright_points_to_raw",
    "raw_points_to_upright",
    "scatter_upright_mask",
    "nearest_skeleton_components",
    "line_record",
    "grouped_line_candidates",
    "consensus_with_provenance",
    "morphological_skeleton",
    "camera_attached_temporal_consensus",
    "fit_line_primitive",
]


SCHEMA_VERSION = "walker_structure_variants_v1"

#: A skeleton component needs at least this many own strict stereo points before
#: a 3-D line primitive is fitted.  It is deliberately the same minimum the
#: archived walker candidate benchmark used for connected-component tubes, so
#: the sparse and the dense branch cannot differ by an accidental threshold.
SKELETON_COMPONENT_MINIMUM_POINTS = 10

#: Extra search margin (upright pixels) added to the skeleton band radius when
#: the nearest skeleton component of an analytically transported candidate pixel
#: is looked up.  The transport is exact for the frozen resize/remap chain up to
#: nearest-neighbour rounding, so a small margin keeps genuinely nearby
#: components reachable without turning the lookup into a global nearest search.
SKELETON_SEARCH_MARGIN_PX = 4

INTERPRETATION_DENSE_COMPONENT_PCA = (
    "Visible-region 3-D rod/tube candidate: every strict stereo point inside one 2-D connected component of the "
    "supplied walker mask is fitted with one 3-D line primitive. It is not a complete walker CAD model, not a "
    "support point, not contact evidence, and it is not a metric accuracy claim."
)

INTERPRETATION_SKELETON_GUIDED = (
    "Visible-region 3-D rod candidate constrained by mask morphology: the morphological skeleton and a fixed "
    "elliptical band around it decide which pixels may enter the strictly identical frozen stereo chain. No SGBM "
    "setting, triangulation rule, left/right consistency limit or photometric limit was changed, so the only "
    "structural variable is which mask pixels are offered to the matcher. It is not a complete walker CAD model, "
    "not a support point, not contact evidence, and it is not a metric accuracy claim."
)

INTERPRETATION_TEMPORAL_CONSENSUS = (
    "Camera-attached repeated-observation candidate: a voxel is kept only when geometry was observed in at least "
    "the minimum number of frames of the local window, in the left-camera frame. That only states repeated "
    "observation of the same camera-frame geometry; it does not prove that the walker is rigid, that it is "
    "stationary in the world, that it rests on the floor, or that a hand or the body touches it. It is not a "
    "metric accuracy claim."
)

STATISTICS_BOUNDARY = (
    "Counts and distributions of this tool's own candidate output under fixed inputs. They are not accuracy, "
    "recall, true length error, walker recognition success rate, or any physical property of the walker."
)


def distribution(values: Iterable[Any]) -> dict[str, Any]:
    """Median/P90/min/max of the finite values, with an explicit empty case."""
    present = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not present:
        return {"count": 0, "median": None, "p90": None, "minimum": None, "maximum": None}
    array = np.asarray(present, dtype=np.float64)
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _binary(mask: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"{label} must be a 2-D mask")
    return (array > 0).astype(np.uint8)


def skeleton_band(skeleton: np.ndarray, radius_px: int) -> np.ndarray:
    """Dilate a skeleton with an elliptical kernel of exactly ``radius_px``.

    The band is the set of mask pixels that the skeleton-guided branch is
    allowed to offer to the frozen stereo chain.  ``radius_px`` is validated
    here so a silently different kernel size cannot enter the pipeline.
    """
    radius = int(radius_px)
    if radius < 1:
        raise ValueError("skeleton band radius must be a positive integer")
    binary = _binary(skeleton, "skeleton")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(binary * 255, kernel)


def skeleton_component_labels(skeleton: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """8-connected component labels and stats of a binary skeleton.

    The component count of this labelling *is* the 2-D skeleton length in
    pixels per component; see :func:`skeleton_component_lengths_px`.
    """
    binary = _binary(skeleton, "skeleton")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    return labels.astype(np.int32), stats


def skeleton_component_lengths_px(stats: np.ndarray) -> dict[int, float]:
    """Per-component 2-D skeleton length, counted as its pixel count.

    A morphological skeleton is one pixel wide, so its pixel count is the
    standard discrete length proxy of that component.  The definition is
    repeated verbatim in every record that carries the number.
    """
    return {
        int(component_id): float(stats[component_id, cv2.CC_STAT_AREA])
        for component_id in range(1, len(stats))
    }


def _check_side(side: str) -> str:
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    return side


def local_pixels_to_upright(
    pixels_local: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    *,
    runtime_size: tuple[int, int],
    upright_shape: tuple[int, int],
    side: str,
) -> np.ndarray:
    """Locate local rectified pixels in the upright mask domain.

    The frozen chain builds its local view as ``remap`` of a ``resize`` of a
    ``rotate`` of the upright image, and the rectification maps already hold the
    source coordinate of every local pixel.  Inverting those two fixed linear
    steps recovers the raw fisheye pixel, and the fixed 90-degree rotation
    recovers the upright pixel.  Nothing about the chain is re-estimated.
    """
    _check_side(side)
    pixels = np.asarray(pixels_local, dtype=np.int64).reshape(-1, 2)
    source_map_x = np.asarray(map_x, dtype=np.float64)
    source_map_y = np.asarray(map_y, dtype=np.float64)
    if source_map_x.shape != source_map_y.shape or source_map_x.ndim != 2:
        raise ValueError("rectification maps must be two equally shaped 2-D arrays")
    height, width = source_map_x.shape
    if (width, height) != (int(runtime_size[0]), int(runtime_size[1])):
        raise ValueError("rectification maps do not match the declared runtime size")
    upright_height, upright_width = int(upright_shape[0]), int(upright_shape[1])
    raw_height, raw_width = upright_width, upright_height
    if not len(pixels):
        return np.empty((0, 2), dtype=np.float64)
    xs = np.clip(pixels[:, 0], 0, width - 1)
    ys = np.clip(pixels[:, 1], 0, height - 1)
    source_x = source_map_x[ys, xs] / (width / raw_width)
    source_y = source_map_y[ys, xs] / (height / raw_height)
    return _raw_to_upright(source_x, source_y, raw_height, raw_width, side)


def _raw_to_upright(
    source_x: np.ndarray, source_y: np.ndarray, raw_height: int, raw_width: int, side: str
) -> np.ndarray:
    if side == "left":
        upright_x = source_y
        upright_y = (raw_width - 1.0) - source_x
    else:
        upright_x = (raw_height - 1.0) - source_y
        upright_y = source_x
    return np.column_stack((upright_x, upright_y))


def raw_points_to_upright(
    points_raw: np.ndarray, *, upright_shape: tuple[int, int], side: str
) -> np.ndarray:
    """Map raw fisheye pixels to upright pixels (inverse of the frozen rotate)."""
    _check_side(side)
    points = np.asarray(points_raw, dtype=np.float64).reshape(-1, 2)
    upright_height, upright_width = int(upright_shape[0]), int(upright_shape[1])
    raw_height, raw_width = upright_width, upright_height
    if not len(points):
        return np.empty((0, 2), dtype=np.float64)
    return _raw_to_upright(points[:, 0], points[:, 1], raw_height, raw_width, side)


def upright_points_to_raw(
    points_upright: np.ndarray, *, upright_shape: tuple[int, int], side: str
) -> np.ndarray:
    """Map upright pixels to raw fisheye pixels (the frozen rotate)."""
    _check_side(side)
    points = np.asarray(points_upright, dtype=np.float64).reshape(-1, 2)
    upright_height, upright_width = int(upright_shape[0]), int(upright_shape[1])
    if not len(points):
        return np.empty((0, 2), dtype=np.float64)
    if side == "left":
        raw_x = (upright_height - 1.0) - points[:, 1]
        raw_y = points[:, 0]
    else:
        raw_x = points[:, 1]
        raw_y = (upright_width - 1.0) - points[:, 0]
    return np.column_stack((raw_x, raw_y))


def scatter_upright_mask(
    points_upright: np.ndarray, shape: tuple[int, int], *, radius_px: int = 0
) -> np.ndarray:
    """Rasterise transported upright coordinates into a mask for display.

    This is a rendering helper only: it rounds coordinates to the nearest
    upright pixel and never feeds a geometric estimator.
    """
    points = np.asarray(points_upright, dtype=np.float64).reshape(-1, 2)
    height, width = int(shape[0]), int(shape[1])
    output = np.zeros((height, width), dtype=np.uint8)
    if not len(points):
        return output
    xs = np.rint(points[:, 0]).astype(np.int64)
    ys = np.rint(points[:, 1]).astype(np.int64)
    inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    if int(radius_px) > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * int(radius_px) + 1, 2 * int(radius_px) + 1)
        )
        seeded = np.zeros_like(output)
        seeded[ys[inside], xs[inside]] = 255
        return cv2.dilate(seeded, kernel)
    output[ys[inside], xs[inside]] = 255
    return output


def nearest_skeleton_components(
    points_upright: np.ndarray,
    labels: np.ndarray,
    *,
    band_radius_px: int,
    margin_px: int = SKELETON_SEARCH_MARGIN_PX,
    chunk_size: int = 4096,
) -> dict[str, Any]:
    """Assign upright query points to the nearest skeleton component.

    A query point is assigned to the component owning the closest skeleton
    pixel, measured as the Euclidean distance in upright pixels from the query
    coordinate to the integer skeleton pixel centre.  Distances that are exactly
    equal are broken by the smaller component id, which makes the assignment
    independent of iteration order.

    Only skeleton pixels inside a Chebyshev window of
    ``band_radius_px + margin_px`` around the rounded query point are searched.
    A query point with no skeleton pixel in that window stays unassigned
    (component id 0, infinite distance) and is counted instead of being forced
    onto a distant component.
    """
    points = np.asarray(points_upright, dtype=np.float64).reshape(-1, 2)
    label_map = np.asarray(labels)
    if label_map.ndim != 2:
        raise ValueError("skeleton labels must be a 2-D label image")
    radius = int(band_radius_px) + int(margin_px)
    if int(band_radius_px) < 1 or radius < 1:
        raise ValueError("band_radius_px must be a positive integer")
    identifiers = np.zeros(len(points), dtype=np.int64)
    distances = np.full(len(points), np.inf, dtype=np.float64)
    if not len(points):
        return {"component_id": identifiers, "distance_px": distances, "unassigned": 0}
    height, width = label_map.shape
    offsets = np.asarray(
        sorted(
            (
                (dy, dx)
                for dy in range(-radius, radius + 1)
                for dx in range(-radius, radius + 1)
                if dy * dy + dx * dx <= radius * radius
            ),
            key=lambda item: (item[0] * item[0] + item[1] * item[1], item[0], item[1]),
        ),
        dtype=np.int64,
    )
    integer_guess = np.rint(points).astype(np.int64)
    unassigned = 0
    for start in range(0, len(points), int(chunk_size)):
        block = points[start : start + int(chunk_size)]
        guess = integer_guess[start : start + int(chunk_size)]
        candidate_y = guess[:, 1][:, None] + offsets[None, :, 0]
        candidate_x = guess[:, 0][:, None] + offsets[None, :, 1]
        inside = (
            (candidate_y >= 0) & (candidate_y < height)
            & (candidate_x >= 0) & (candidate_x < width)
        )
        gathered = np.zeros(candidate_y.shape, dtype=np.int64)
        gathered[inside] = label_map[candidate_y[inside], candidate_x[inside]]
        delta_y = block[:, 1][:, None] - candidate_y
        delta_x = block[:, 0][:, None] - candidate_x
        distance = np.sqrt(delta_y * delta_y + delta_x * delta_x)
        valid = inside & (gathered > 0)
        minimum = np.where(valid, distance, np.inf).min(axis=1)
        found = np.isfinite(minimum)
        tied = valid & (distance == minimum[:, None])
        best = np.where(tied, gathered, np.iinfo(np.int64).max).min(axis=1)
        identifiers[start : start + len(block)] = np.where(found, best, 0)
        distances[start : start + len(block)] = np.where(found, minimum, np.inf)
        unassigned += int((~found).sum())
    return {"component_id": identifiers, "distance_px": distances, "unassigned": unassigned}


def line_record(primitive: Any, points: np.ndarray) -> dict[str, Any]:
    """Common line/tube fields shared by every method of this comparison.

    For a line primitive the perpendicular distance of a point to the fitted
    line is simultaneously the tube radius and the line-fit residual, so both
    are reported from the same explicit definition instead of pretending they
    are two independent measurements.
    """
    array = np.asarray(points, dtype=np.float64)
    center = np.asarray(primitive.center_left_camera_mm, dtype=np.float64)
    direction = np.asarray(primitive.direction_unit, dtype=np.float64)
    endpoint_min = np.asarray(primitive.endpoint_min_left_camera_mm, dtype=np.float64)
    endpoint_max = np.asarray(primitive.endpoint_max_left_camera_mm, dtype=np.float64)
    delta = array - center
    projection = delta @ direction
    closest = center + projection[:, None] * direction
    perpendicular = np.linalg.norm(array - closest, axis=1)
    return {
        "component_id": int(primitive.component_id),
        "status": "candidate",
        "coordinate_frame": "left_camera",
        "length_unit": "millimeter",
        "center_left_camera_mm": center.tolist(),
        "direction_unit": direction.tolist(),
        "endpoint_min_left_camera_mm": endpoint_min.tolist(),
        "endpoint_max_left_camera_mm": endpoint_max.tolist(),
        "length_mm": float(np.linalg.norm(endpoint_max - endpoint_min)),
        "radius_median_mm": float(primitive.radius_median_mm),
        "radius_p90_mm": float(primitive.radius_p90_mm),
        "fit_residual_median_mm": float(np.median(perpendicular)),
        "fit_residual_p90_mm": float(np.percentile(perpendicular, 90)),
        "point_count": int(len(array)),
        "length_definition": (
            "distance between the 5th and 95th percentile projections of this component's strict stereo points on "
            "the fitted 3-D line"
        ),
        "radius_definition": (
            "perpendicular distance of this component's strict stereo points to the fitted 3-D line "
            "(median and P90); it is the tube-radius proxy, not a measured physical tube diameter"
        ),
        "fit_residual_definition": (
            "the same perpendicular distance to the fitted 3-D line as the radius, reported separately so the fit "
            "quality of the line is explicit; it is not a ground-truth error"
        ),
    }


def grouped_line_candidates(
    points: np.ndarray,
    group_ids: np.ndarray,
    *,
    minimum_points: int = SKELETON_COMPONENT_MINIMUM_POINTS,
    group_metadata: Mapping[int, Mapping[str, Any]] | None = None,
    group_reasons: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Fit one 3-D line primitive per non-empty group id.

    Groups are visited in ascending id order, so the output order never depends
    on the pixel or voxel enumeration order.  A group that cannot produce a
    primitive keeps a record with its own reason instead of disappearing.
    """
    array = np.asarray(points, dtype=np.float64)
    identifiers = np.asarray(group_ids, dtype=np.int64).reshape(-1)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] != identifiers.shape[0]:
        raise ValueError("points and group_ids must be an aligned Nx3 / N array")
    metadata = dict(group_metadata or {})
    reasons = dict(group_reasons or {})
    candidates: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for group_id in sorted({int(value) for value in identifiers if int(value) > 0}):
        indices = np.flatnonzero(identifiers == group_id)
        record: dict[str, Any] = {
            "component_id": group_id,
            "stereo_point_count": int(len(indices)),
            **dict(metadata.get(group_id, {})),
        }
        rejection = reasons.get(group_id)
        if rejection is not None:
            record.update({"status": "unavailable", "reasons": [str(rejection)]})
            failures.append(record)
            continue
        if len(indices) < int(minimum_points):
            record.update({"status": "unavailable", "reasons": ["insufficient_component_stereo_points"]})
            failures.append(record)
            continue
        try:
            primitive = fit_line_primitive(array[indices], group_id)
        except ValueError:
            record.update({"status": "unavailable", "reasons": ["degenerate_component_geometry"]})
            failures.append(record)
            continue
        record.update(line_record(primitive, array[indices]))
        candidates.append(record)
    return {"candidates": candidates, "failures": failures}


def consensus_with_provenance(
    point_sets: Sequence[np.ndarray],
    *,
    voxel_size_mm: float = 20.0,
    minimum_frame_support: int = 2,
) -> dict[str, Any]:
    """Run the frozen camera-attached consensus and keep per-voxel provenance.

    The authoritative status, voxel count, consensus points, voxel size,
    minimum support and reasons come from
    :func:`pose_app.scene_geometry_variants.camera_attached_temporal_consensus`.
    The voxel key derivation and the per-voxel median are mirrored only to learn
    which input frame and point index produced each returned consensus point;
    the mirror is compared against the frozen function and the call is rejected
    if the two ever disagree.
    """
    frames = [np.asarray(points, dtype=np.float64).reshape(-1, 3) for points in point_sets]
    official = camera_attached_temporal_consensus(
        frames, voxel_size_mm=voxel_size_mm, minimum_frame_support=minimum_frame_support
    )
    support: dict[tuple[int, int, int], set[int]] = {}
    samples: dict[tuple[int, int, int], list[np.ndarray]] = {}
    provenance: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    for frame_index, points in enumerate(frames):
        keys = np.floor(points / float(voxel_size_mm)).astype(np.int64)
        for point_index, (key_array, point) in enumerate(zip(keys, points)):
            key = tuple(int(value) for value in key_array)
            support.setdefault(key, set()).add(frame_index)
            samples.setdefault(key, []).append(point)
            provenance.setdefault(key, []).append((frame_index, int(point_index)))
    accepted = sorted(key for key, seen in support.items() if len(seen) >= int(minimum_frame_support))
    mirrored = np.asarray(
        [np.median(np.stack(samples[key]), axis=0) for key in accepted], dtype=np.float64
    ).reshape(-1, 3)
    reported = np.asarray(official["consensus_points_left_camera_mm"], dtype=np.float64).reshape(-1, 3)
    if mirrored.shape != reported.shape or not np.array_equal(mirrored, reported):
        raise RuntimeError(
            "camera-attached consensus provenance mirror disagrees with the frozen consensus function; "
            "refusing to report labels derived from an inconsistent mirror"
        )
    return {
        **official,
        "voxel_keys": [list(key) for key in accepted],
        "voxel_members": [provenance[key] for key in accepted],
        "provenance_note": (
            "voxel_members lists (frame_index_in_window, point_index) for every input point that fell in the "
            "accepted voxel; it is derived by mirroring the frozen voxel key rule and is verified equal to the "
            "frozen function's own consensus points before it is used."
        ),
    }


def mask_seed_upright(mask: np.ndarray) -> np.ndarray | None:
    """Median upright mask pixel, the seed the frozen chain derives per mask.

    Provided so a run can record the seed that each method's own mask produced
    through the identical frozen rule, without touching that rule.
    """
    binary = _binary(mask, "mask")
    yy, xx = np.nonzero(binary)
    if len(xx) == 0:
        return None
    return np.asarray((float(np.median(xx)), float(np.median(yy))), dtype=np.float64)
