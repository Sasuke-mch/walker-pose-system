"""RANSAC backend for static-background stereo visual odometry.

The feature front end is intentionally outside this module: it must create
stereo-triangulated 3-D correspondences only after masking the person, walker,
and the region used by a direct ground fit.  Keeping that condition explicit
in the input record makes the motion estimate auditable and avoids ground
self-validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .local_plane_propagation import STATIC_BACKGROUND_DOMAIN


@dataclass(frozen=True)
class StaticCorrespondences:
    from_frame_index: int
    to_frame_index: int
    previous_xyz_left_camera_mm: np.ndarray
    current_xyz_left_camera_mm: np.ndarray

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "StaticCorrespondences":
        if value.get("feature_domain") != STATIC_BACKGROUND_DOMAIN:
            raise ValueError("correspondences are not restricted to static background")
        if value.get("ground_region_used_for_motion") is not False:
            raise ValueError("ground-fit region must be excluded from static VO")
        previous = np.asarray(value["previous_xyz_left_camera_mm"], dtype=np.float64)
        current = np.asarray(value["current_xyz_left_camera_mm"], dtype=np.float64)
        if previous.ndim != 2 or previous.shape[1:] != (3,) or current.shape != previous.shape:
            raise ValueError("previous/current static correspondence arrays must have matching Nx3 shape")
        if len(previous) < 3 or not np.all(np.isfinite(previous)) or not np.all(np.isfinite(current)):
            raise ValueError("at least three finite static 3-D correspondences are required")
        start, end = int(value["from_frame_index"]), int(value["to_frame_index"])
        if end != start + 1:
            raise ValueError("static VO input must connect adjacent frames")
        return cls(start, end, previous, current)


def fit_rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares proper rotation and translation, mapping source to target."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("rigid fit requires matching Nx3 arrays with N >= 3")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def _sample_is_degenerate(points: np.ndarray) -> bool:
    return np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-7) < 2


def estimate_rigid_transform_ransac(
    correspondences: StaticCorrespondences,
    *,
    distance_threshold_mm: float,
    iterations: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return R, q, inlier mask and residuals for ``X_to = R X_from + q``."""
    if distance_threshold_mm <= 0 or iterations < 1:
        raise ValueError("RANSAC threshold and iteration count must be positive")
    source, target = correspondences.previous_xyz_left_camera_mm, correspondences.current_xyz_left_camera_mm
    random = np.random.default_rng(random_seed)
    best: np.ndarray | None = None
    best_key: tuple[int, float] | None = None
    for _ in range(iterations):
        sample = random.choice(len(source), size=3, replace=False)
        if _sample_is_degenerate(source[sample]) or _sample_is_degenerate(target[sample]):
            continue
        rotation, translation = fit_rigid_transform(source[sample], target[sample])
        residuals = np.linalg.norm((rotation @ source.T).T + translation - target, axis=1)
        inliers = residuals <= distance_threshold_mm
        if int(inliers.sum()) < 3:
            continue
        key = (int(inliers.sum()), -float(np.median(residuals[inliers])))
        if best_key is None or key > best_key:
            best, best_key = inliers, key
    if best is None:
        raise ValueError("RANSAC found no non-degenerate static-background rigid transform")
    rotation, translation = fit_rigid_transform(source[best], target[best])
    residuals = np.linalg.norm((rotation @ source.T).T + translation - target, axis=1)
    inliers = residuals <= distance_threshold_mm
    if int(inliers.sum()) < 3:
        raise ValueError("refined RANSAC transform has fewer than three inliers")
    # Refit after the final inlier test to avoid reporting a sample-only pose.
    rotation, translation = fit_rigid_transform(source[inliers], target[inliers])
    residuals = np.linalg.norm((rotation @ source.T).T + translation - target, axis=1)
    return rotation, translation, residuals <= distance_threshold_mm, residuals


def relative_pose_record(
    correspondences: StaticCorrespondences,
    *,
    distance_threshold_mm: float,
    iterations: int,
    random_seed: int,
    minimum_inliers: int,
) -> dict[str, Any]:
    """Create a propagation-compatible record, retaining every acceptance fact."""
    try:
        rotation, translation, inliers, residuals = estimate_rigid_transform_ransac(
            correspondences, distance_threshold_mm=distance_threshold_mm, iterations=iterations, random_seed=random_seed
        )
    except ValueError as exc:
        return {
            "from_frame_index": correspondences.from_frame_index, "to_frame_index": correspondences.to_frame_index,
            "status": "rejected", "reason": str(exc), "feature_domain": STATIC_BACKGROUND_DOMAIN,
            "ground_region_used_for_motion": False, "static_3d_correspondence_count": int(len(correspondences.previous_xyz_left_camera_mm)),
        }
    inlier_count = int(inliers.sum())
    accepted = inlier_count >= minimum_inliers
    return {
        "from_frame_index": correspondences.from_frame_index, "to_frame_index": correspondences.to_frame_index,
        "status": "accepted" if accepted else "rejected",
        "reason": None if accepted else "too_few_static_background_ransac_inliers",
        "R_to_from": rotation.tolist(), "q_to_from": translation.tolist(),
        "feature_domain": STATIC_BACKGROUND_DOMAIN, "ground_region_used_for_motion": False,
        "static_3d_correspondence_count": int(len(correspondences.previous_xyz_left_camera_mm)),
        "ransac_inlier_count": inlier_count,
        "ransac_distance_threshold_mm": float(distance_threshold_mm),
        "ransac_inlier_fraction": float(inlier_count / len(correspondences.previous_xyz_left_camera_mm)),
        "ransac_inlier_residual_median_mm": float(np.median(residuals[inliers])) if inlier_count else None,
        "ransac_inlier_residual_p95_mm": float(np.percentile(residuals[inliers], 95)) if inlier_count else None,
        "static_feature_frontend": "external_input; must document person/walker/ground-region exclusion",
    }
