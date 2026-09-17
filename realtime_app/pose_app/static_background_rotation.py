"""Fast monocular fisheye rotation candidate from moving static-scene tracks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .calibration import StereoCalibration
from .static_background_stereo_frontend import person_exclusion_feature_mask


@dataclass(frozen=True)
class RotationTrackCriteria:
    maximum_corners: int = 1400
    quality_level: float = 0.01
    minimum_distance_px: float = 7.0
    maximum_forward_backward_error_px: float = 1.25
    minimum_flow_px: float = 0.40
    maximum_flow_px: float = 70.0
    essential_threshold_normalized: float = 0.004
    minimum_tracks: int = 16
    minimum_inliers: int = 10
    rotation_only_ransac_iterations: int = 80
    rotation_only_threshold_deg: float = 0.75
    maximum_tracks_per_grid_cell: int = 30
    grid_columns: int = 8
    grid_rows: int = 5


def estimate_rotation_from_normalized_tracks(
    previous_normalized: np.ndarray,
    current_normalized: np.ndarray,
    criteria: RotationTrackCriteria,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    previous = np.asarray(previous_normalized, dtype=np.float64).reshape(-1, 2)
    current = np.asarray(current_normalized, dtype=np.float64).reshape(-1, 2)
    if len(previous) < criteria.minimum_tracks or len(current) != len(previous):
        return None, {"status": "unavailable", "reason": "too_few_normalized_tracks", "track_count": len(previous)}
    essential, mask = cv2.findEssentialMat(
        previous, current, np.eye(3), method=cv2.RANSAC, prob=0.999,
        threshold=criteria.essential_threshold_normalized,
    )
    essential_rotation = None
    essential_count = 0
    essential_ransac_count = int(np.count_nonzero(mask)) if mask is not None else 0
    if essential is not None:
        if essential.shape[0] > 3:
            essential = essential[:3]
        inlier_count, candidate, _, _ = cv2.recoverPose(
            essential, previous, current, np.eye(3), mask=mask,
        )
        essential_count = int(inlier_count)
        if essential_count >= criteria.minimum_inliers:
            essential_rotation = candidate

    bearing_rotation, bearing_inliers = _estimate_rotation_only_ransac(previous, current, criteria)
    bearing_count = int(np.count_nonzero(bearing_inliers))
    candidates: list[tuple[str, np.ndarray, int]] = []
    if essential_rotation is not None:
        candidates.append(("essential_recover_pose", essential_rotation, essential_count))
    if bearing_rotation is not None and bearing_count >= criteria.minimum_inliers:
        candidates.append(("bearing_rotation_ransac", bearing_rotation, bearing_count))
    if not candidates:
        return None, {
            "status": "rejected", "reason": "too_few_rotation_inliers", "track_count": len(previous),
            "essential_ransac_inlier_count": essential_ransac_count,
            "essential_rotation_inlier_count": essential_count,
            "bearing_rotation_inlier_count": bearing_count,
        }
    # Translation makes a rotation-only fit biased even when it has many angular
    # inliers. Prefer the geometrically complete essential solution whenever it
    # passes; use bearing rotation only for the pure/small-motion degeneracy.
    source, rotation, selected_count = candidates[0]
    return rotation, {
        "status": "accepted", "reason": None, "track_count": len(previous),
        "rotation_source": source,
        "essential_ransac_inlier_count": essential_ransac_count,
        "essential_rotation_inlier_count": essential_count,
        "bearing_rotation_inlier_count": bearing_count,
        "rotation_inlier_count": selected_count,
        "rotation_inlier_fraction": selected_count / len(previous),
    }


def _fit_bearing_rotation(previous_bearings: np.ndarray, current_bearings: np.ndarray) -> np.ndarray | None:
    cross_covariance = previous_bearings.T @ current_bearings
    if np.linalg.matrix_rank(cross_covariance) < 2:
        return None
    u, _, vt = np.linalg.svd(cross_covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    return rotation


def _estimate_rotation_only_ransac(
    previous_normalized: np.ndarray,
    current_normalized: np.ndarray,
    criteria: RotationTrackCriteria,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Fit current_bearing ~= R * previous_bearing with translation-parallax rejection."""

    previous = np.column_stack((previous_normalized, np.ones(len(previous_normalized))))
    current = np.column_stack((current_normalized, np.ones(len(current_normalized))))
    previous /= np.linalg.norm(previous, axis=1, keepdims=True)
    current /= np.linalg.norm(current, axis=1, keepdims=True)
    threshold = np.deg2rad(criteria.rotation_only_threshold_deg)
    rng = np.random.default_rng(0)
    best = np.zeros(len(previous), dtype=bool)
    best_median = np.inf
    if len(previous) < 3:
        return None, best
    for _ in range(criteria.rotation_only_ransac_iterations):
        sample = rng.choice(len(previous), 3, replace=False)
        rotation = _fit_bearing_rotation(previous[sample], current[sample])
        if rotation is None:
            continue
        predicted = (rotation @ previous.T).T
        residual = np.arccos(np.clip(np.sum(predicted * current, axis=1), -1.0, 1.0))
        inliers = residual <= threshold
        count = int(np.count_nonzero(inliers))
        median = float(np.median(residual[inliers])) if count else np.inf
        if count > int(np.count_nonzero(best)) or (count == int(np.count_nonzero(best)) and median < best_median):
            best, best_median = inliers, median
    if np.count_nonzero(best) < 3:
        return None, best
    rotation = _fit_bearing_rotation(previous[best], current[best])
    if rotation is None:
        return None, best
    predicted = (rotation @ previous.T).T
    residual = np.arccos(np.clip(np.sum(predicted * current, axis=1), -1.0, 1.0))
    refined = residual <= threshold
    if np.count_nonzero(refined) >= 3:
        best = refined
        rotation = _fit_bearing_rotation(previous[best], current[best])
    return rotation, best


def _grid_select(
    points: np.ndarray, scores: np.ndarray, shape: tuple[int, int], criteria: RotationTrackCriteria,
) -> np.ndarray:
    height, width = shape
    cells: dict[tuple[int, int], list[int]] = {}
    for index, (x, y) in enumerate(points):
        cell = (
            min(criteria.grid_columns - 1, max(0, int(x * criteria.grid_columns / width))),
            min(criteria.grid_rows - 1, max(0, int(y * criteria.grid_rows / height))),
        )
        cells.setdefault(cell, []).append(index)
    selected = []
    for indices in cells.values():
        indices.sort(key=lambda item: float(scores[item]))
        selected.extend(indices[:criteria.maximum_tracks_per_grid_cell])
    return np.asarray(selected, dtype=np.int32)


def track_static_scene_rotation(
    previous_bgr: np.ndarray,
    current_bgr: np.ndarray,
    previous_allowed_mask: np.ndarray,
    current_allowed_mask: np.ndarray,
    calibration: StereoCalibration,
    criteria: RotationTrackCriteria = RotationTrackCriteria(),
    *,
    side: str = "left",
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Track non-person pixels and reject near-zero camera-attached motion."""

    previous_gray = cv2.cvtColor(previous_bgr, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(
        previous_gray, mask=previous_allowed_mask, maxCorners=criteria.maximum_corners,
        qualityLevel=criteria.quality_level, minDistance=criteria.minimum_distance_px,
        blockSize=7, useHarrisDetector=False,
    )
    if corners is None or len(corners) < criteria.minimum_tracks:
        return None, {"status": "unavailable", "reason": "too_few_detected_corners", "detected_corner_count": 0 if corners is None else len(corners)}
    current, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, corners, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray, previous_gray, current, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    previous_points = corners.reshape(-1, 2)
    current_points = current.reshape(-1, 2)
    backward_points = backward.reshape(-1, 2)
    height, width = previous_gray.shape
    rounded = np.rint(current_points).astype(int)
    in_bounds = (
        (rounded[:, 0] >= 0) & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
    )
    current_mask_valid = np.zeros(len(previous_points), dtype=bool)
    valid_indices = np.flatnonzero(in_bounds)
    current_mask_valid[valid_indices] = current_allowed_mask[
        rounded[valid_indices, 1], rounded[valid_indices, 0]
    ] > 0
    forward_backward = np.linalg.norm(backward_points - previous_points, axis=1)
    flow = np.linalg.norm(current_points - previous_points, axis=1)
    valid = (
        (forward_status.reshape(-1) > 0) & (backward_status.reshape(-1) > 0)
        & in_bounds & current_mask_valid
        & (forward_backward <= criteria.maximum_forward_backward_error_px)
        & (flow >= criteria.minimum_flow_px) & (flow <= criteria.maximum_flow_px)
    )
    tracked_previous = previous_points[valid]
    tracked_current = current_points[valid]
    tracked_fb = forward_backward[valid]
    if len(tracked_previous) >= criteria.minimum_tracks:
        selected = _grid_select(tracked_previous, tracked_fb, previous_gray.shape, criteria)
        tracked_previous = tracked_previous[selected]
        tracked_current = tracked_current[selected]
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    previous_normalized = calibration.undistort_normalized(tracked_previous, side) if len(tracked_previous) else np.empty((0, 2))
    current_normalized = calibration.undistort_normalized(tracked_current, side) if len(tracked_current) else np.empty((0, 2))
    rotation, audit = estimate_rotation_from_normalized_tracks(previous_normalized, current_normalized, criteria)
    audit.update({
        "detected_corner_count": len(previous_points),
        "forward_backward_valid_count": int(np.count_nonzero(
            (forward_status.reshape(-1) > 0) & (backward_status.reshape(-1) > 0)
            & in_bounds & current_mask_valid & (forward_backward <= criteria.maximum_forward_backward_error_px)
        )),
        "moving_static_scene_track_count": len(tracked_previous),
        "raw_flow_median_px": float(np.median(flow)) if len(flow) else None,
        "selected_flow_median_px": float(np.median(flow[valid])) if np.any(valid) else None,
        "feature_domain": "person_excluded_nonzero_flow_scene_candidate",
        "walker_exclusion_mechanism": "camera_attached_near_zero_flow_rejection",
        "camera_side": side,
    })
    return rotation, audit


class RealtimeStereoRotationTracker:
    """Causal two-view rotation consensus for the live stereo loop."""

    def __init__(
        self,
        calibration: StereoCalibration,
        *,
        processing_width: int = 480,
        maximum_stereo_disagreement_deg: float = 1.5,
        criteria: RotationTrackCriteria = RotationTrackCriteria(),
    ) -> None:
        if processing_width < 240:
            raise ValueError("processing_width must be at least 240")
        self.calibration_full = calibration
        self.processing_width = processing_width
        self.maximum_stereo_disagreement_deg = maximum_stereo_disagreement_deg
        self.criteria = criteria
        self.previous_images: dict[str, np.ndarray] | None = None
        self.previous_masks: dict[str, np.ndarray] | None = None

    @staticmethod
    def _person_mapping(result: Any) -> list[dict[str, Any]]:
        persons = list(getattr(result, "persons", []) or [])
        if not persons:
            return []
        primary = max(persons, key=lambda person: (float(person.pose_score), float(person.bbox_score)))
        return [{"keypoints": [list(item) for item in primary.keypoints]}]

    @staticmethod
    def _rotation_angle_deg(rotation: np.ndarray) -> float:
        return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))))

    def update(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        left_result: Any,
        right_result: Any,
        *,
        estimate: bool,
    ) -> dict[str, Any]:
        started = cv2.getTickCount()
        images, masks = {}, {}
        for side, source, result in (
            ("left", left_image, left_result), ("right", right_image, right_result)
        ):
            scale = min(1.0, self.processing_width / source.shape[1])
            image = source if scale == 1.0 else cv2.resize(
                source, (round(source.shape[1] * scale), round(source.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
            images[side] = image
            masks[side] = person_exclusion_feature_mask(
                image.shape[:2], self._person_mapping(result), scale=scale,
                keypoint_score_threshold=0.25, dilation_fraction=0.055,
            )
        calibration = self.calibration_full.for_runtime_sizes(
            (images["left"].shape[1], images["left"].shape[0]),
            (images["right"].shape[1], images["right"].shape[0]),
        )
        if self.previous_images is None or not estimate:
            record = {
                "status": "unavailable", "reason": "first_frame" if self.previous_images is None else "stage_does_not_request_rotation",
                "R_to_from": None,
            }
        else:
            rotations, side_audits = {}, {}
            for side in ("left", "right"):
                rotations[side], side_audits[side] = track_static_scene_rotation(
                    self.previous_images[side], images[side], self.previous_masks[side], masks[side],
                    calibration, self.criteria, side=side,
                )
            if rotations["left"] is None or rotations["right"] is None:
                record = {
                    "status": "unavailable", "reason": "both_camera_rotations_required",
                    "R_to_from": None, "side_rotation_status": side_audits,
                }
            else:
                right_in_left = calibration.R.T @ rotations["right"] @ calibration.R
                disagreement = self._rotation_angle_deg(rotations["left"] @ right_in_left.T)
                if disagreement > self.maximum_stereo_disagreement_deg:
                    record = {
                        "status": "rejected", "reason": "left_right_rotation_disagreement",
                        "left_right_rotation_disagreement_deg": disagreement,
                        "R_to_from": None, "side_rotation_status": side_audits,
                    }
                else:
                    u, _, vt = np.linalg.svd(rotations["left"] + right_in_left)
                    rotation = u @ vt
                    if np.linalg.det(rotation) < 0:
                        u[:, -1] *= -1.0
                        rotation = u @ vt
                    record = {
                        "status": "accepted", "reason": None,
                        "source": "independent_left_right_fisheye_rotation_consensus",
                        "left_right_rotation_disagreement_deg": disagreement,
                        "R_to_from": rotation.tolist(), "side_rotation_status": side_audits,
                    }
        self.previous_images, self.previous_masks = images, masks
        record["processing_ms"] = (cv2.getTickCount() - started) * 1000.0 / cv2.getTickFrequency()
        record["causal"] = True
        return record
