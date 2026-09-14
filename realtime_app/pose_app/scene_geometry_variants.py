"""Comparable local-ground and walker reconstruction primitives.

The functions in this module deliberately start *after* calibrated stereo
correspondences have been established.  They do not infer semantic identity
and they never turn an unreviewed mask into ground/walker truth.  This keeps
the experiment variable explicit: the same 3-D candidates can be passed to
several estimators without silently changing calibration or matching.

All metric quantities use millimetres and the current left-camera frame.
There is no persistent world frame in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable

import cv2
import numpy as np


EPS = 1e-9


def _points3(value: np.ndarray, label: str = "points") -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError(f"{label} must be a finite Nx3 array")
    return points


def _pixels2(value: np.ndarray, count: int) -> np.ndarray:
    pixels = np.asarray(value, dtype=np.float64)
    if pixels.shape != (count, 2) or not np.all(np.isfinite(pixels)):
        raise ValueError("pixels must be a finite Nx2 array aligned with points")
    return pixels


def _canonical_plane(normal: np.ndarray, offset: float) -> tuple[np.ndarray, float]:
    normal = np.asarray(normal, dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if normal.shape != (3,) or not math.isfinite(norm) or norm < EPS or not math.isfinite(float(offset)):
        raise ValueError("plane must contain a finite non-zero normal and finite offset")
    normal, offset = normal / norm, float(offset) / norm
    # Normal points from the plane towards the camera origin.  Since the
    # origin evaluates to d, this convention requires d > 0.
    if offset < 0:
        normal, offset = -normal, -offset
    return normal, offset


def angular_error_degrees(first: np.ndarray, second: np.ndarray, *, signless: bool = True) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    a /= max(float(np.linalg.norm(a)), EPS)
    b /= max(float(np.linalg.norm(b)), EPS)
    dot = float(a @ b)
    if signless:
        dot = abs(dot)
    return float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))


@dataclass(frozen=True)
class PlaneEstimate:
    method: str
    status: str
    normal_left_camera: np.ndarray | None
    offset_mm: float | None
    inlier_mask: np.ndarray
    reasons: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_mapping(self) -> dict[str, Any]:
        plane = None
        if self.normal_left_camera is not None and self.offset_mm is not None:
            plane = {
                "normal_left_camera": self.normal_left_camera.tolist(),
                "offset_mm": float(self.offset_mm),
                "normal_orientation": "toward_camera",
            }
        return {
            "method": self.method,
            "status": self.status,
            "coordinate_frame": "left_camera",
            "length_unit": "millimeter",
            "plane": plane,
            "inlier_count": int(self.inlier_mask.sum()),
            "reasons": list(self.reasons),
            "metrics": self.metrics,
            "interpretation": (
                "Estimator output conditioned on externally supplied semantic and stereo candidates; "
                "not physical ground truth or a persistent world frame."
            ),
        }


def weighted_plane_svd(points: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    points = _points3(points)
    if len(points) < 3:
        raise ValueError("at least three points are required")
    if weights is None:
        weights = np.ones(len(points), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(points),) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite, non-negative, and aligned with points")
    positive = weights > 0
    if int(positive.sum()) < 3 or float(weights.sum()) < EPS:
        raise ValueError("at least three points need positive weight")
    weights = weights / float(weights.sum())
    centroid = np.sum(points * weights[:, None], axis=0)
    centered = points - centroid
    covariance = (centered * weights[:, None]).T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if float(eigenvalues[1]) < EPS:
        raise ValueError("points are collinear or degenerate")
    normal = eigenvectors[:, 0]
    offset = -float(normal @ centroid)
    return _canonical_plane(normal, offset)


def _plane_result(
    method: str,
    points: np.ndarray,
    normal: np.ndarray,
    offset: float,
    distance_threshold_mm: float,
    *,
    extra_metrics: dict[str, Any] | None = None,
) -> PlaneEstimate:
    distances = np.abs(points @ normal + offset)
    inliers = distances <= float(distance_threshold_mm)
    metrics: dict[str, Any] = {
        "candidate_count": int(len(points)),
        "inlier_fraction": float(inliers.mean()) if len(inliers) else 0.0,
        "median_residual_mm": float(np.median(distances[inliers])) if np.any(inliers) else None,
        "p95_residual_mm": float(np.percentile(distances[inliers], 95)) if np.any(inliers) else None,
    }
    metrics.update(extra_metrics or {})
    return PlaneEstimate(method, "candidate", normal, float(offset), inliers, (), metrics)


def estimate_dense_ransac(
    points: np.ndarray,
    *,
    distance_threshold_mm: float = 25.0,
    iterations: int = 900,
    seed: int = 0,
    minimum_points: int = 150,
) -> PlaneEstimate:
    """Conventional dense candidate RANSAC baseline."""
    points = _points3(points)
    empty = np.zeros(len(points), dtype=bool)
    if len(points) < max(3, minimum_points):
        return PlaneEstimate("dense_ransac", "unavailable", None, None, empty, ("insufficient_candidates",))
    rng = np.random.default_rng(seed)
    best: tuple[int, float, np.ndarray, float] | None = None
    for _ in range(int(iterations)):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < EPS:
            continue
        normal /= norm
        offset = -float(normal @ sample[0])
        distances = np.abs(points @ normal + offset)
        inliers = distances <= distance_threshold_mm
        count = int(inliers.sum())
        median = float(np.median(distances[inliers])) if count else math.inf
        score = (count, -median)
        if best is None or score > (best[0], best[1]):
            best = (count, -median, normal, offset)
    if best is None:
        return PlaneEstimate("dense_ransac", "unavailable", None, None, empty, ("degenerate_candidates",))
    initial_normal, initial_offset = _canonical_plane(best[2], best[3])
    initial_inliers = np.abs(points @ initial_normal + initial_offset) <= distance_threshold_mm
    if int(initial_inliers.sum()) < 3:
        return PlaneEstimate("dense_ransac", "unavailable", None, None, empty, ("insufficient_inliers",))
    normal, offset = weighted_plane_svd(points[initial_inliers])
    return _plane_result(
        "dense_ransac", points, normal, offset, distance_threshold_mm,
        extra_metrics={"ransac_iterations": int(iterations), "fit_candidate_count": int(len(points))},
    )


def tile_balanced_indices(
    pixels: np.ndarray,
    scores: np.ndarray,
    *,
    image_shape: tuple[int, int],
    tile_size_px: int = 32,
    maximum_per_tile: int = 4,
) -> np.ndarray:
    """Select spatially distributed high-score points instead of dense clusters."""
    pixels = np.asarray(pixels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or scores.shape != (len(pixels),):
        raise ValueError("pixels/scores must be aligned")
    if tile_size_px < 1 or maximum_per_tile < 1:
        raise ValueError("tile size and maximum per tile must be positive")
    height, width = image_shape
    valid = (
        np.isfinite(scores) & np.all(np.isfinite(pixels), axis=1)
        & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )
    groups: dict[tuple[int, int], list[int]] = {}
    for index in np.flatnonzero(valid):
        key = (int(pixels[index, 1]) // tile_size_px, int(pixels[index, 0]) // tile_size_px)
        groups.setdefault(key, []).append(int(index))
    selected: list[int] = []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda idx: (-float(scores[idx]), idx))
        selected.extend(ordered[:maximum_per_tile])
    return np.asarray(selected, dtype=np.int64)


def estimate_sparse_tiles(
    points: np.ndarray,
    pixels: np.ndarray,
    scores: np.ndarray | None,
    *,
    image_shape: tuple[int, int],
    tile_size_px: int = 32,
    maximum_per_tile: int = 4,
    distance_threshold_mm: float = 25.0,
    iterations: int = 900,
    seed: int = 0,
    minimum_points: int = 30,
) -> PlaneEstimate:
    """Fit from a few reliable points per image tile.

    This is the implemented alternative to a full mask: it still needs a
    semantic candidate region, but the geometric estimator consumes only a
    distributed set of points rather than every mask pixel.
    """
    points = _points3(points)
    pixels = _pixels2(pixels, len(points))
    if scores is None:
        scores = np.ones(len(points), dtype=np.float64)
    indices = tile_balanced_indices(
        pixels, scores, image_shape=image_shape, tile_size_px=tile_size_px,
        maximum_per_tile=maximum_per_tile,
    )
    if len(indices) < minimum_points:
        return PlaneEstimate(
            "sparse_tile_ransac", "unavailable", None, None, np.zeros(len(points), dtype=bool),
            ("insufficient_spatially_distributed_points",),
            {"selected_count": int(len(indices)), "occupied_tiles": int(len(np.unique(pixels[indices] // tile_size_px, axis=0))) if len(indices) else 0},
        )
    sparse = estimate_dense_ransac(
        points[indices], distance_threshold_mm=distance_threshold_mm,
        iterations=iterations, seed=seed, minimum_points=minimum_points,
    )
    if sparse.normal_left_camera is None or sparse.offset_mm is None:
        return PlaneEstimate(
            "sparse_tile_ransac", "unavailable", None, None, np.zeros(len(points), dtype=bool),
            sparse.reasons, {**sparse.metrics, "selected_count": int(len(indices))},
        )
    result = _plane_result(
        "sparse_tile_ransac", points, sparse.normal_left_camera, sparse.offset_mm,
        distance_threshold_mm,
        extra_metrics={
            "selected_count": int(len(indices)),
            "selection_fraction": float(len(indices) / len(points)),
            "tile_size_px": int(tile_size_px),
            "maximum_per_tile": int(maximum_per_tile),
        },
    )
    return result


def estimate_weighted_irls(
    points: np.ndarray,
    prior_weights: np.ndarray | None = None,
    *,
    huber_scale_mm: float = 20.0,
    iterations: int = 8,
    distance_threshold_mm: float = 25.0,
    minimum_points: int = 30,
) -> PlaneEstimate:
    """Soft-confidence plane using robust iteratively reweighted least squares."""
    points = _points3(points)
    empty = np.zeros(len(points), dtype=bool)
    if len(points) < minimum_points:
        return PlaneEstimate("soft_weighted_irls", "unavailable", None, None, empty, ("insufficient_candidates",))
    if prior_weights is None:
        prior_weights = np.ones(len(points), dtype=np.float64)
    prior = np.asarray(prior_weights, dtype=np.float64)
    if prior.shape != (len(points),) or np.any(prior < 0) or not np.all(np.isfinite(prior)):
        raise ValueError("prior_weights must be finite, non-negative and aligned")
    if int((prior > 0).sum()) < 3:
        return PlaneEstimate("soft_weighted_irls", "unavailable", None, None, empty, ("insufficient_positive_weight",))
    weights = prior.copy()
    normal: np.ndarray | None = None
    offset: float | None = None
    for _ in range(int(iterations)):
        try:
            normal, offset = weighted_plane_svd(points, weights)
        except ValueError:
            return PlaneEstimate("soft_weighted_irls", "unavailable", None, None, empty, ("degenerate_candidates",))
        residual = np.abs(points @ normal + offset)
        robust = np.ones_like(residual)
        large = residual > huber_scale_mm
        robust[large] = huber_scale_mm / np.maximum(residual[large], EPS)
        weights = prior * robust
    assert normal is not None and offset is not None
    effective = float((weights.sum() ** 2) / max(float(np.square(weights).sum()), EPS))
    return _plane_result(
        "soft_weighted_irls", points, normal, offset, distance_threshold_mm,
        extra_metrics={
            "huber_scale_mm": float(huber_scale_mm),
            "irls_iterations": int(iterations),
            "effective_weighted_points": effective,
            "prior_weight_mean": float(prior.mean()),
        },
    )


def estimate_region_consensus(
    points: np.ndarray,
    pixels: np.ndarray,
    *,
    image_shape: tuple[int, int],
    distance_threshold_mm: float = 25.0,
    minimum_region_points: int = 30,
    maximum_region_angle_deg: float = 5.0,
) -> PlaneEstimate:
    """Require independently fitted image regions to agree before refitting."""
    points = _points3(points)
    pixels = _pixels2(pixels, len(points))
    empty = np.zeros(len(points), dtype=bool)
    # Split the *observed support* at its medians rather than the whole image.
    # A floor mask can occupy only one image quadrant while still spanning a
    # broad local region; full-image quadrants would incorrectly report that
    # as a single-region observation.
    split_x = float(np.median(pixels[:, 0])) if len(pixels) else image_shape[1] / 2
    split_y = float(np.median(pixels[:, 1])) if len(pixels) else image_shape[0] / 2
    region_ids = (pixels[:, 0] >= split_x).astype(np.int32) + 2 * (pixels[:, 1] >= split_y).astype(np.int32)
    regional: list[tuple[int, np.ndarray, float, int]] = []
    for region_id in range(4):
        indices = np.flatnonzero(region_ids == region_id)
        if len(indices) < minimum_region_points:
            continue
        estimate = estimate_dense_ransac(
            points[indices], distance_threshold_mm=distance_threshold_mm,
            iterations=300, seed=731 + region_id, minimum_points=minimum_region_points,
        )
        if estimate.normal_left_camera is None or estimate.offset_mm is None:
            continue
        normal, offset = estimate.normal_left_camera, estimate.offset_mm
        regional.append((region_id, normal, offset, int(len(indices))))
    if len(regional) < 2:
        return PlaneEstimate(
            "region_consensus", "unavailable", None, None, empty,
            ("fewer_than_two_independent_regions",), {"accepted_regions": len(regional)},
        )
    angles = [angular_error_degrees(a[1], b[1]) for i, a in enumerate(regional) for b in regional[i + 1 :]]
    median_angle = float(np.median(angles))
    if median_angle > maximum_region_angle_deg:
        return PlaneEstimate(
            "region_consensus", "unavailable", None, None, empty,
            ("regional_planes_disagree",),
            {"accepted_regions": len(regional), "median_pairwise_normal_angle_deg": median_angle},
        )
    reference = regional[0][1]
    aligned = [normal if float(normal @ reference) >= 0 else -normal for _, normal, _, _ in regional]
    consensus_normal = np.mean(np.stack(aligned), axis=0)
    consensus_normal /= np.linalg.norm(consensus_normal)
    offsets = []
    for (_, normal, offset, _), aligned_normal in zip(regional, aligned):
        offsets.append(offset if float(normal @ aligned_normal) >= 0 else -offset)
    consensus_offset = float(np.median(offsets))
    consensus_normal, consensus_offset = _canonical_plane(consensus_normal, consensus_offset)
    coarse = np.abs(points @ consensus_normal + consensus_offset) <= 2.0 * distance_threshold_mm
    if int(coarse.sum()) < minimum_region_points:
        return PlaneEstimate(
            "region_consensus", "unavailable", None, None, empty,
            ("insufficient_consensus_support",),
            {"accepted_regions": len(regional), "median_pairwise_normal_angle_deg": median_angle},
        )
    normal, offset = weighted_plane_svd(points[coarse])
    return _plane_result(
        "region_consensus", points, normal, offset, distance_threshold_mm,
        extra_metrics={
            "accepted_regions": len(regional),
            "region_candidate_counts": {str(rid): count for rid, _, _, count in regional},
            "adaptive_split_xy": [split_x, split_y],
            "median_pairwise_normal_angle_deg": median_angle,
            "maximum_region_angle_deg": float(maximum_region_angle_deg),
        },
    )


def morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    """Return a one-pixel-ish skeleton using only core OpenCV operations."""
    image = (np.asarray(mask) > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(image):
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(image, opened))
        image = cv2.erode(image, element)
    return skeleton


@dataclass(frozen=True)
class LinePrimitive:
    component_id: int
    center_left_camera_mm: np.ndarray
    direction_unit: np.ndarray
    endpoint_min_left_camera_mm: np.ndarray
    endpoint_max_left_camera_mm: np.ndarray
    radius_median_mm: float
    radius_p90_mm: float
    point_count: int

    def as_mapping(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "center_left_camera_mm": self.center_left_camera_mm.tolist(),
            "direction_unit": self.direction_unit.tolist(),
            "endpoint_min_left_camera_mm": self.endpoint_min_left_camera_mm.tolist(),
            "endpoint_max_left_camera_mm": self.endpoint_max_left_camera_mm.tolist(),
            "radius_median_mm": self.radius_median_mm,
            "radius_p90_mm": self.radius_p90_mm,
            "point_count": self.point_count,
        }


def fit_line_primitive(points: np.ndarray, component_id: int = 0) -> LinePrimitive:
    points = _points3(points)
    if len(points) < 3:
        raise ValueError("line primitive needs at least three points")
    center = np.median(points, axis=0)
    _, singular, vt = np.linalg.svd(points - center, full_matrices=False)
    if len(singular) < 2 or float(singular[0]) < EPS:
        raise ValueError("degenerate line points")
    direction = vt[0]
    projection = (points - center) @ direction
    closest = center + projection[:, None] * direction
    radius = np.linalg.norm(points - closest, axis=1)
    lo, hi = np.percentile(projection, [5, 95])
    return LinePrimitive(
        int(component_id), center, direction,
        center + float(lo) * direction, center + float(hi) * direction,
        float(np.median(radius)), float(np.percentile(radius, 90)), int(len(points)),
    )


def reconstruct_walker_components(
    points: np.ndarray,
    pixels: np.ndarray,
    walker_mask: np.ndarray,
    *,
    minimum_component_pixels: int = 100,
    minimum_component_points: int = 20,
) -> dict[str, Any]:
    """Fit 3-D line/tube primitives to visible walker mask components."""
    points = _points3(points)
    pixels = _pixels2(pixels, len(points)).astype(np.int32)
    mask = (np.asarray(walker_mask) > 0).astype(np.uint8)
    if mask.ndim != 2:
        raise ValueError("walker_mask must be 2-D")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    primitives: list[LinePrimitive] = []
    component_records: list[dict[str, Any]] = []
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < minimum_component_pixels:
            continue
        inside = (
            (pixels[:, 0] >= 0) & (pixels[:, 0] < mask.shape[1])
            & (pixels[:, 1] >= 0) & (pixels[:, 1] < mask.shape[0])
        )
        indices = np.flatnonzero(inside & (labels[np.clip(pixels[:, 1], 0, mask.shape[0] - 1), np.clip(pixels[:, 0], 0, mask.shape[1] - 1)] == component_id))
        record = {"component_id": component_id, "mask_area_px": area, "stereo_point_count": int(len(indices))}
        if len(indices) < minimum_component_points:
            record["status"] = "unavailable"
            record["reasons"] = ["insufficient_component_stereo_points"]
        else:
            try:
                primitive = fit_line_primitive(points[indices], component_id)
            except ValueError:
                record["status"] = "unavailable"
                record["reasons"] = ["degenerate_component_geometry"]
            else:
                primitives.append(primitive)
                record["status"] = "candidate"
                record["primitive"] = primitive.as_mapping()
        component_records.append(record)
    status = "candidate" if primitives else "unavailable"
    return {
        "schema_version": "walker_component_reconstruction_v1",
        "status": status,
        "coordinate_frame": "left_camera",
        "length_unit": "millimeter",
        "dense_point_count": int(len(points)),
        "components": component_records,
        "line_primitives": [item.as_mapping() for item in primitives],
        "reasons": [] if primitives else ["no_reconstructable_walker_component"],
        "interpretation": (
            "Visible tube/edge geometry candidate from an externally supplied walker mask. "
            "It is not an amodal CAD model, support/contact truth, or proof that the walker is on the floor."
        ),
    }


def fit_rigid_template(
    template_points_walker_mm: np.ndarray,
    observed_points_left_camera_mm: np.ndarray,
    *,
    maximum_rms_mm: float,
) -> dict[str, Any]:
    """Fit a measured sparse walker template from named point correspondences."""
    source = _points3(template_points_walker_mm, "template_points_walker_mm")
    target = _points3(observed_points_left_camera_mm, "observed_points_left_camera_mm")
    if source.shape != target.shape or len(source) < 3:
        raise ValueError("template and observed correspondences need matching Nx3 arrays with N>=3")
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    predicted = (rotation @ source.T).T + translation
    residual = np.linalg.norm(predicted - target, axis=1)
    rms = float(np.sqrt(np.mean(np.square(residual))))
    return {
        "schema_version": "walker_sparse_template_fit_v1",
        "status": "candidate" if rms <= maximum_rms_mm else "unavailable",
        "rotation_left_camera_from_walker": rotation.tolist(),
        "translation_left_camera_from_walker_mm": translation.tolist(),
        "rms_mm": rms,
        "maximum_rms_mm": float(maximum_rms_mm),
        "point_count": int(len(source)),
        "reasons": [] if rms <= maximum_rms_mm else ["template_fit_residual_too_large"],
        "interpretation": (
            "Requires independently identified, named walker correspondences and a measured configuration; "
            "the template never manufactures missing observations."
        ),
    }


def fit_articulated_template(
    segments: dict[str, tuple[np.ndarray, np.ndarray]],
    joints: Iterable[tuple[str, np.ndarray, str, np.ndarray]],
    *,
    maximum_segment_rms_mm: float,
    maximum_joint_gap_mm: float,
) -> dict[str, Any]:
    """Fit each measured walker segment and audit shared-joint consistency.

    ``segments`` maps a segment name to ``(template_points, observed_points)``.
    Each joint specifies ``(segment_a, point_a_local, segment_b,
    point_b_local)``.  The function allows the segments to rotate separately,
    which models small walker articulation, but rejects solutions whose shared
    physical joints separate beyond the declared tolerance.
    """
    if not segments:
        raise ValueError("at least one articulated segment is required")
    fitted: dict[str, dict[str, Any]] = {}
    transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in sorted(segments):
        template, observed = segments[name]
        result = fit_rigid_template(
            template, observed, maximum_rms_mm=maximum_segment_rms_mm
        )
        fitted[name] = result
        transforms[name] = (
            np.asarray(result["rotation_left_camera_from_walker"], dtype=np.float64),
            np.asarray(result["translation_left_camera_from_walker_mm"], dtype=np.float64),
        )
    joint_records: list[dict[str, Any]] = []
    for segment_a, point_a, segment_b, point_b in joints:
        if segment_a not in transforms or segment_b not in transforms:
            raise ValueError("joint references an unknown articulated segment")
        ra, ta = transforms[segment_a]
        rb, tb = transforms[segment_b]
        pa = ra @ np.asarray(point_a, dtype=np.float64) + ta
        pb = rb @ np.asarray(point_b, dtype=np.float64) + tb
        gap = float(np.linalg.norm(pa - pb))
        joint_records.append({
            "segment_a": segment_a,
            "segment_b": segment_b,
            "point_a_left_camera_mm": pa.tolist(),
            "point_b_left_camera_mm": pb.tolist(),
            "gap_mm": gap,
            "accepted": gap <= maximum_joint_gap_mm,
        })
    segment_ok = all(result["status"] == "candidate" for result in fitted.values())
    joint_ok = all(record["accepted"] for record in joint_records)
    reasons: list[str] = []
    if not segment_ok:
        reasons.append("one_or_more_segment_fits_failed")
    if not joint_ok:
        reasons.append("articulated_joint_gap_too_large")
    return {
        "schema_version": "walker_articulated_template_fit_v1",
        "status": "candidate" if segment_ok and joint_ok else "unavailable",
        "segments": fitted,
        "joints": joint_records,
        "maximum_segment_rms_mm": float(maximum_segment_rms_mm),
        "maximum_joint_gap_mm": float(maximum_joint_gap_mm),
        "reasons": reasons,
        "interpretation": (
            "Measured, named segment correspondences are required. Separate segment rotations model small "
            "articulation; joint constraints audit rather than invent the missing parts."
        ),
    }


def camera_attached_temporal_consensus(
    point_sets: Iterable[np.ndarray],
    *,
    voxel_size_mm: float = 20.0,
    minimum_frame_support: int = 2,
) -> dict[str, Any]:
    """Find geometry repeatedly observed in the camera frame without freezing it.

    A point is retained only through voxel occupancy across frames.  The output
    is a stable self-body *candidate*; articulated or lifted parts may disappear
    and are therefore reported rather than forced into a static template.
    """
    frames = [_points3(points) for points in point_sets]
    if voxel_size_mm <= 0 or minimum_frame_support < 1:
        raise ValueError("voxel_size_mm and minimum_frame_support must be positive")
    support: dict[tuple[int, int, int], set[int]] = {}
    samples: dict[tuple[int, int, int], list[np.ndarray]] = {}
    for frame_index, points in enumerate(frames):
        keys = np.floor(points / voxel_size_mm).astype(np.int64)
        for key_array, point in zip(keys, points):
            key = tuple(int(v) for v in key_array)
            support.setdefault(key, set()).add(frame_index)
            samples.setdefault(key, []).append(point)
    accepted_keys = sorted(key for key, seen in support.items() if len(seen) >= minimum_frame_support)
    consensus = np.asarray([np.median(np.stack(samples[key]), axis=0) for key in accepted_keys], dtype=np.float64)
    if not accepted_keys:
        consensus = np.empty((0, 3), dtype=np.float64)
    return {
        "schema_version": "walker_camera_attached_temporal_consensus_v1",
        "status": "candidate" if len(consensus) else "unavailable",
        "coordinate_frame": "left_camera",
        "length_unit": "millimeter",
        "frame_count": len(frames),
        "input_point_count": int(sum(len(points) for points in frames)),
        "consensus_voxel_count": int(len(consensus)),
        "consensus_points_left_camera_mm": consensus.tolist(),
        "voxel_size_mm": float(voxel_size_mm),
        "minimum_frame_support": int(minimum_frame_support),
        "reasons": [] if len(consensus) else ["no_temporally_repeated_geometry"],
        "interpretation": (
            "Camera-attached repeated geometry candidate only. It does not imply rigidity, ground support, "
            "or semantic walker identity without an external mask/keypoint source."
        ),
    }
