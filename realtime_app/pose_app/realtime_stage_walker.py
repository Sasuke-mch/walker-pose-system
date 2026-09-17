from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from .calibration import StereoCalibration
from .schema import InferenceResult, PersonPose
from .stereo_sources import StereoFramePair


STAGE_WARMING_UP = "warming_up"
STAGE_ONE = "stage1_walker_static_human_moving"
STAGE_TWO = "stage2_feet_static_walker_moving"
STAGE_TRANSITION = "transition"
STAGE_UNKNOWN = "unknown"


@dataclass(frozen=True)
class StageCriteria:
    static_background_px: float = 0.8
    moving_background_px: float = 1.5
    moving_human_px: float = 1.2
    moving_ankle_px: float = 1.2
    ankle_direction_cosine: float = 0.35
    ankle_separation_change_px: float = 1.8
    max_foot_background_residual_px: float = 2.5
    max_compensated_ankle_separation_error_px: float = 2.0
    min_background_tracks: int = 18
    confirmation_frames: int = 2
    stage1_confirmation_frames: int = 3
    stage2_confirmation_frames: int = 2
    motion_window_frames: int = 5
    moving_window_displacement_px: float = 1.5
    moving_window_path_px: float = 1.8
    moving_window_coherence: float = 0.55
    static_window_displacement_px: float = 1.0


class TemporalMotionWindow:
    def __init__(self, criteria: StageCriteria) -> None:
        self.criteria = criteria
        self.vectors: deque[np.ndarray] = deque(maxlen=criteria.motion_window_frames)

    def update(self, background: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(background)
        if not background.get("available"):
            self.vectors.clear()
            enriched.update(
                {
                    "window_available": False,
                    "motion_active": False,
                    "static_confident": False,
                }
            )
            return enriched
        self.vectors.append(np.asarray(background["median_vector_px"], dtype=np.float64))
        vectors = np.asarray(self.vectors, dtype=np.float64)
        cumulative_vector = np.sum(vectors, axis=0)
        cumulative = float(np.linalg.norm(cumulative_vector))
        path = float(np.sum(np.linalg.norm(vectors, axis=1)))
        coherence = cumulative / max(path, 1e-9)
        window_ready = len(self.vectors) == self.criteria.motion_window_frames
        instantaneous = float(background["median_motion_px"])
        motion_active = instantaneous >= self.criteria.moving_background_px or (
            window_ready
            and cumulative >= self.criteria.moving_window_displacement_px
            and path >= self.criteria.moving_window_path_px
            and coherence >= self.criteria.moving_window_coherence
        )
        static_confident = instantaneous <= self.criteria.static_background_px and (
            not window_ready or cumulative <= self.criteria.static_window_displacement_px
        )
        enriched.update(
            {
                "window_available": window_ready,
                "window_frame_count": len(self.vectors),
                "window_cumulative_vector_px": cumulative_vector.astype(float).tolist(),
                "window_displacement_px": cumulative,
                "window_path_px": path,
                "window_direction_coherence": coherence,
                "motion_active": bool(motion_active),
                "static_confident": bool(static_confident),
            }
        )
        return enriched


def _primary_person(result: InferenceResult | None) -> PersonPose | None:
    if result is None or not result.persons:
        return None
    return max(result.persons, key=lambda person: (person.pose_score, person.bbox_score))


def _resize_gray(image: np.ndarray, processing_width: int) -> tuple[np.ndarray, float]:
    scale = min(1.0, float(processing_width) / float(image.shape[1]))
    if scale < 1.0:
        resized = cv2.resize(
            image,
            (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = image
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), scale


def _valid_points(person: PersonPose | None, threshold: float, scale: float) -> dict[int, np.ndarray]:
    if person is None:
        return {}
    result: dict[int, np.ndarray] = {}
    for index, item in enumerate(person.keypoints):
        if len(item) >= 3 and float(item[2]) >= threshold:
            point = np.asarray(item[:2], dtype=np.float32) * float(scale)
            if np.isfinite(point).all():
                result[index] = point
    return result


def _person_exclusion_mask(
    shape: tuple[int, int], person: PersonPose | None, threshold: float, scale: float
) -> np.ndarray:
    mask = np.full(shape, 255, dtype=np.uint8)
    points = list(_valid_points(person, threshold, scale).values())
    if len(points) >= 3:
        hull = cv2.convexHull(np.asarray(points, dtype=np.int32))
        cv2.fillConvexPoly(mask, hull, 0)
        radius = max(12, round(0.035 * max(shape)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        excluded = cv2.bitwise_not(mask)
        excluded = cv2.dilate(excluded, kernel)
        mask = cv2.bitwise_not(excluded)
    elif points:
        radius = max(15, round(0.05 * max(shape)))
        for point in points:
            cv2.circle(mask, tuple(np.rint(point).astype(int)), radius, 0, -1)
    return mask


def _distributed_background_features(
    gray: np.ndarray,
    mask: np.ndarray,
    rows: int = 3,
    columns: int = 4,
    max_per_cell: int = 28,
) -> np.ndarray | None:
    """Prevent one high-contrast handle or image corner from dominating KLT."""

    height, width = gray.shape
    margin = max(3, round(0.025 * min(height, width)))
    working_mask = mask.copy()
    working_mask[:margin, :] = 0
    working_mask[-margin:, :] = 0
    working_mask[:, :margin] = 0
    working_mask[:, -margin:] = 0
    collected: list[np.ndarray] = []
    for row in range(rows):
        y1, y2 = round(row * height / rows), round((row + 1) * height / rows)
        for column in range(columns):
            x1, x2 = round(column * width / columns), round((column + 1) * width / columns)
            roi = gray[y1:y2, x1:x2]
            roi_mask = working_mask[y1:y2, x1:x2]
            features = cv2.goodFeaturesToTrack(
                roi,
                maxCorners=max_per_cell,
                qualityLevel=0.01,
                minDistance=8,
                mask=roi_mask,
                blockSize=7,
            )
            if features is None:
                continue
            shifted = features.reshape(-1, 2)
            shifted[:, 0] += x1
            shifted[:, 1] += y1
            collected.append(shifted)
    if not collected:
        return None
    return np.concatenate(collected, axis=0).reshape(-1, 1, 2).astype(np.float32)


def estimate_background_motion(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_person: PersonPose | None,
    current_person: PersonPose | None,
    keypoint_threshold: float,
    scale: float,
    min_tracks: int,
    extra_exclusion_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    mask = _person_exclusion_mask(previous_gray.shape, previous_person, keypoint_threshold, scale)
    if extra_exclusion_mask is not None:
        if extra_exclusion_mask.shape != mask.shape:
            raise ValueError("extra exclusion mask shape does not match the flow image")
        mask[extra_exclusion_mask > 0] = 0
    features = _distributed_background_features(previous_gray, mask)
    if features is None or len(features) < min_tracks:
        return {"available": False, "reason": "insufficient_background_features", "track_count": 0}
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        features,
        None,
        winSize=(21, 21),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if tracked is None or status is None:
        return {"available": False, "reason": "optical_flow_failed", "track_count": 0}
    previous = features.reshape(-1, 2)[status.reshape(-1) == 1]
    current = tracked.reshape(-1, 2)[status.reshape(-1) == 1]
    current_mask = _person_exclusion_mask(current_gray.shape, current_person, keypoint_threshold, scale)
    inside = (
        (current[:, 0] >= 0)
        & (current[:, 0] < current_gray.shape[1])
        & (current[:, 1] >= 0)
        & (current[:, 1] < current_gray.shape[0])
    )
    previous, current = previous[inside], current[inside]
    if current.size:
        xi = np.clip(np.rint(current[:, 0]).astype(int), 0, current_gray.shape[1] - 1)
        yi = np.clip(np.rint(current[:, 1]).astype(int), 0, current_gray.shape[0] - 1)
        keep = current_mask[yi, xi] > 0
        previous, current = previous[keep], current[keep]
    if len(previous) < min_tracks:
        return {
            "available": False,
            "reason": "insufficient_tracks_after_person_exclusion",
            "track_count": int(len(previous)),
        }
    affine, inliers = cv2.estimateAffinePartial2D(
        previous,
        current,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=2000,
        confidence=0.995,
    )
    if affine is None or inliers is None:
        return {"available": False, "reason": "background_affine_failed", "track_count": int(len(previous))}
    selected = inliers.reshape(-1).astype(bool)
    if int(selected.sum()) < min_tracks:
        return {
            "available": False,
            "reason": "insufficient_background_inliers",
            "track_count": int(len(previous)),
            "inlier_count": int(selected.sum()),
        }
    flow = current[selected] - previous[selected]
    median_vector = np.median(flow, axis=0)
    motion = np.linalg.norm(flow, axis=1)
    rotation_deg = math.degrees(math.atan2(float(affine[1, 0]), float(affine[0, 0])))
    return {
        "available": True,
        "track_count": int(len(previous)),
        "inlier_count": int(selected.sum()),
        "inlier_ratio": float(selected.mean()),
        "median_motion_px": float(np.median(motion)),
        "median_vector_px": [float(median_vector[0]), float(median_vector[1])],
        "affine_translation_px": [float(affine[0, 2]), float(affine[1, 2])],
        "affine_rotation_deg": float(rotation_deg),
        "affine_2x3": affine.astype(float).tolist(),
    }


def fuse_background_motion(backgrounds: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], str | None]:
    """Fuse views without systematically hiding motion seen by one valid camera."""

    reliable = [
        side
        for side in ("left", "right")
        if backgrounds.get(side, {}).get("available")
        and int(backgrounds[side].get("inlier_count", 0)) >= 18
        and float(backgrounds[side].get("inlier_ratio", 0.0)) >= 0.35
    ]
    if not reliable:
        return {"available": False, "reason": "no_reliable_view"}, None
    moving_reliable = [side for side in reliable if backgrounds[side].get("motion_active")]
    selection_pool = moving_reliable or reliable
    selected = max(
        selection_pool,
        key=lambda side: (
            float(backgrounds[side].get("window_displacement_px", 0.0)),
            float(backgrounds[side]["median_motion_px"]),
            float(backgrounds[side].get("inlier_ratio", 0.0)),
        ),
    )
    motions = [float(backgrounds[side]["median_motion_px"]) for side in reliable]
    fused = dict(backgrounds[selected])
    fused.update(
        {
            "fusion": "motion_preserving_reliable_view",
            "selected_view": selected,
            "reliable_views": reliable,
            "view_motion_px": {
                side: float(backgrounds[side]["median_motion_px"]) for side in reliable
            },
            "cross_view_motion_difference_px": (
                abs(motions[0] - motions[1]) if len(motions) == 2 else None
            ),
            "motion_active": any(backgrounds[side].get("motion_active", False) for side in reliable),
            "static_confident": all(backgrounds[side].get("static_confident", False) for side in reliable),
        }
    )
    return fused, selected


def keypoint_motion_evidence(
    previous: PersonPose | None,
    current: PersonPose | None,
    threshold: float,
    scale: float,
    background_vector: np.ndarray | None,
    background_affine: np.ndarray | None = None,
) -> dict[str, Any]:
    old = _valid_points(previous, threshold, scale)
    new = _valid_points(current, threshold, scale)
    common = sorted(set(old) & set(new))
    upper_body = [index for index in common if 5 <= index <= 12]
    vectors = np.asarray([new[index] - old[index] for index in upper_body], dtype=np.float32)
    human_motion = float(np.median(np.linalg.norm(vectors, axis=1))) if len(vectors) else None
    compensated_vectors = vectors
    if len(upper_body) and background_affine is not None:
        affine = np.asarray(background_affine, dtype=np.float64)
        if affine.shape != (2, 3) or not np.isfinite(affine).all():
            raise ValueError("background affine must be a finite 2x3 matrix")
        old_points = np.asarray([old[index] for index in upper_body], dtype=np.float64)
        predicted = (affine[:, :2] @ old_points.T).T + affine[:, 2]
        new_points = np.asarray([new[index] for index in upper_body], dtype=np.float64)
        compensated_vectors = (new_points - predicted).astype(np.float32)
    compensated_human_motion = (
        float(np.median(np.linalg.norm(compensated_vectors, axis=1)))
        if len(compensated_vectors) else None
    )
    ankle_indices = [index for index in (15, 16) if index in old and index in new]
    ankle_vectors = np.asarray([new[index] - old[index] for index in ankle_indices], dtype=np.float32)
    evidence: dict[str, Any] = {
        "common_keypoints": len(common),
        "upper_body_keypoints": len(upper_body),
        "upper_body_median_motion_px": human_motion,
        "upper_body_compensated_motion_px": compensated_human_motion,
        "ankle_count": len(ankle_indices),
        "ankle_median_motion_px": (
            float(np.median(np.linalg.norm(ankle_vectors, axis=1))) if len(ankle_vectors) else None
        ),
        "ankle_direction_cosine": None,
        "ankle_separation_change_px": None,
        "ankle_background_direction_cosine": None,
        "ankle_background_residual_median_px": None,
        "ankle_background_residual_max_px": None,
        "ankle_compensated_separation_error_px": None,
    }
    if len(ankle_vectors) == 2:
        norms = np.linalg.norm(ankle_vectors, axis=1)
        if np.all(norms > 1e-6):
            evidence["ankle_direction_cosine"] = float(
                np.dot(ankle_vectors[0], ankle_vectors[1]) / (norms[0] * norms[1])
            )
        old_separation = float(np.linalg.norm(old[15] - old[16]))
        new_separation = float(np.linalg.norm(new[15] - new[16]))
        evidence["ankle_separation_change_px"] = abs(new_separation - old_separation)
        if background_vector is not None:
            median_ankle = np.median(ankle_vectors, axis=0)
            denom = float(np.linalg.norm(median_ankle) * np.linalg.norm(background_vector))
            if denom > 1e-6:
                evidence["ankle_background_direction_cosine"] = float(
                    np.dot(median_ankle, background_vector) / denom
                )
        if background_affine is not None:
            affine = np.asarray(background_affine, dtype=np.float64)
            old_ankles = np.asarray([old[index] for index in (15, 16)], dtype=np.float64)
            predicted_ankles = (affine[:, :2] @ old_ankles.T).T + affine[:, 2]
            current_ankles = np.asarray([new[index] for index in (15, 16)], dtype=np.float64)
            residuals = np.linalg.norm(current_ankles - predicted_ankles, axis=1)
            evidence["ankle_background_residual_median_px"] = float(np.median(residuals))
            evidence["ankle_background_residual_max_px"] = float(np.max(residuals))
            evidence["ankle_compensated_separation_error_px"] = abs(
                float(np.linalg.norm(current_ankles[0] - current_ankles[1]))
                - float(np.linalg.norm(predicted_ankles[0] - predicted_ankles[1]))
            )
    return evidence


class TwoStageRecognizer:
    def __init__(self, criteria: StageCriteria | None = None) -> None:
        self.criteria = criteria or StageCriteria()
        self.pending = STAGE_UNKNOWN
        self.pending_count = 0
        self.confirmed = STAGE_WARMING_UP
        self.last_stable: str | None = None
        self.stage1_history: deque[bool] = deque(maxlen=5)
        self.stage2_history: deque[bool] = deque(maxlen=3)
        self.stage2_missing_frames = 0

    def update(self, background: dict[str, Any], pose: dict[str, Any]) -> tuple[str, str, int]:
        candidate = STAGE_UNKNOWN
        camera_moving = False
        camera_static = False
        if background.get("available"):
            bg = float(background["median_motion_px"])
            human = pose.get(
                "upper_body_compensated_motion_px",
                pose.get("upper_body_median_motion_px"),
            )
            ankle_motion = pose.get("ankle_median_motion_px")
            ankle_cos = pose.get("ankle_direction_cosine")
            separation = pose.get("ankle_separation_change_px")
            background_cos = pose.get("ankle_background_direction_cosine")
            foot_residual = pose.get("ankle_background_residual_max_px")
            compensated_separation = pose.get("ankle_compensated_separation_error_px")
            camera_moving = bool(background.get("motion_active", bg >= self.criteria.moving_background_px))
            camera_static = bool(background.get("static_confident", bg <= self.criteria.static_background_px))
            if (
                camera_moving
                and pose.get("ankle_count") == 2
                and foot_residual is not None
                and foot_residual <= self.criteria.max_foot_background_residual_px
                and compensated_separation is not None
                and compensated_separation
                <= self.criteria.max_compensated_ankle_separation_error_px
            ):
                candidate = STAGE_TWO
            elif camera_static and human is not None and human >= self.criteria.moving_human_px:
                candidate = STAGE_ONE
            elif self.last_stable in (STAGE_ONE, STAGE_TWO):
                candidate = STAGE_TRANSITION

        if candidate == self.pending:
            self.pending_count += 1
        else:
            self.pending = candidate
            self.pending_count = 1
        self.stage1_history.append(candidate == STAGE_ONE)
        self.stage2_history.append(candidate == STAGE_TWO)
        stage1_ready = (
            camera_static
            and sum(self.stage1_history) >= self.criteria.stage1_confirmation_frames
        )
        stage2_direct = sum(self.stage2_history) >= self.criteria.stage2_confirmation_frames
        evidence_ready = bool(background.get("window_available", False))
        neutral_state = (
            STAGE_TRANSITION
            if self.last_stable is not None or evidence_ready or camera_moving
            else STAGE_WARMING_UP
        )
        if candidate == STAGE_TWO:
            self.stage2_missing_frames = 0
        elif camera_moving:
            self.stage2_missing_frames += 1
        else:
            self.stage2_missing_frames = 0

        if camera_moving:
            # A moving camera/walker is a hard exclusion for stage 1.  Once stage 2
            # is established, tolerate short pose dropouts without changing it to
            # stage 1; longer losses remain explicit transition frames.
            hold_stage2 = self.last_stable == STAGE_TWO and self.stage2_missing_frames <= 3
            if stage2_direct or hold_stage2:
                self.confirmed = STAGE_TWO
                self.last_stable = STAGE_TWO
            else:
                self.confirmed = neutral_state
        elif camera_static:
            # Human motion is allowed to pause briefly inside a walker-static phase.
            if stage1_ready or (
                self.last_stable == STAGE_ONE and any(self.stage1_history)
            ):
                self.confirmed = STAGE_ONE
                self.last_stable = STAGE_ONE
            else:
                self.confirmed = neutral_state
        else:
            self.confirmed = neutral_state
        return self.confirmed, candidate, self.pending_count


class StableStructureView:
    def __init__(self, shape: tuple[int, int]) -> None:
        self.shape = shape
        self.score = np.zeros(shape, dtype=np.float32)
        self.moving_updates = 0

    def update(self, gray: np.ndarray, person_mask: np.ndarray, allow_update: bool) -> np.ndarray:
        edges = cv2.Canny(gray, 55, 140)
        edges = cv2.bitwise_and(edges, person_mask)
        h, w = gray.shape
        prior = np.zeros_like(edges)
        border_x = max(1, round(w * 0.24))
        border_y = max(1, round(h * 0.25))
        prior[:, :border_x] = 255
        prior[:, w - border_x :] = 255
        prior[:border_y, :] = 255
        edges = cv2.bitwise_and(edges, prior)
        edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8))
        if allow_update:
            present = edges > 0
            self.score *= 0.94
            self.score[present] += 0.12
            np.clip(self.score, 0.0, 1.0, out=self.score)
            self.moving_updates += 1
        candidate = np.where(self.score >= 0.28, 255, 0).astype(np.uint8)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return candidate

    def exclusion_mask(self) -> np.ndarray:
        mask = np.where(self.score >= 0.18, 255, 0).astype(np.uint8)
        return cv2.dilate(mask, np.ones((15, 15), dtype=np.uint8))


def _line_segments(mask: np.ndarray, limit: int = 10) -> list[list[int]]:
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180.0, threshold=24, minLineLength=28, maxLineGap=10)
    if lines is None:
        return []
    values = [line.reshape(4).astype(int).tolist() for line in lines]
    values.sort(key=lambda line: -math.hypot(line[2] - line[0], line[3] - line[1]))
    selected: list[list[int]] = []
    descriptors: list[tuple[float, float, float]] = []
    for line in values:
        x1, y1, x2, y2 = line
        angle = math.atan2(y2 - y1, x2 - x1) % math.pi
        midpoint_x = 0.5 * (x1 + x2)
        midpoint_y = 0.5 * (y1 + y2)
        duplicate = False
        for old_angle, old_x, old_y in descriptors:
            angle_delta = abs(angle - old_angle)
            angle_delta = min(angle_delta, math.pi - angle_delta)
            if angle_delta < math.radians(7.0) and math.hypot(midpoint_x - old_x, midpoint_y - old_y) < 24.0:
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(line)
        descriptors.append((angle, midpoint_x, midpoint_y))
        if len(selected) >= limit:
            break
    return selected


def _sampson_error(left: np.ndarray, right: np.ndarray, essential: np.ndarray) -> np.ndarray:
    ones = np.ones((len(left), 1), dtype=np.float64)
    x1 = np.concatenate([left, ones], axis=1)
    x2 = np.concatenate([right, ones], axis=1)
    ex1 = (essential @ x1.T).T
    etx2 = (essential.T @ x2.T).T
    numerator = np.sum(x2 * ex1, axis=1) ** 2
    denominator = ex1[:, 0] ** 2 + ex1[:, 1] ** 2 + etx2[:, 0] ** 2 + etx2[:, 1] ** 2
    return numerator / np.maximum(denominator, 1e-12)


def sparse_stereo_structure(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    calibration: StereoCalibration,
    left_scale: float,
    right_scale: float,
) -> dict[str, Any]:
    orb = cv2.ORB_create(nfeatures=600, fastThreshold=10)
    feature_kernel = np.ones((11, 11), dtype=np.uint8)
    left_feature_mask = cv2.dilate(left_mask, feature_kernel)
    right_feature_mask = cv2.dilate(right_mask, feature_kernel)
    left_kp, left_desc = orb.detectAndCompute(left_gray, left_feature_mask)
    right_kp, right_desc = orb.detectAndCompute(right_gray, right_feature_mask)
    if left_desc is None or right_desc is None or len(left_kp) < 4 or len(right_kp) < 4:
        return {"status": "unavailable", "reason": "insufficient_masked_features", "points_3d": []}
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(left_desc, right_desc, k=2)
    good = [
        items[0]
        for items in pairs
        if len(items) == 2 and items[0].distance < 0.72 * items[1].distance
    ]
    if len(good) < 4:
        return {"status": "unavailable", "reason": "insufficient_descriptor_matches", "points_3d": []}
    left_raw = np.asarray([left_kp[item.queryIdx].pt for item in good], dtype=np.float64) / left_scale
    right_raw = np.asarray([right_kp[item.trainIdx].pt for item in good], dtype=np.float64) / right_scale
    left_norm = calibration.undistort_normalized(left_raw, "left")
    right_norm = calibration.undistort_normalized(right_raw, "right")
    errors = _sampson_error(left_norm, right_norm, calibration.essential_matrix)
    keep = errors <= 2.5e-4
    left_norm, right_norm = left_norm[keep], right_norm[keep]
    if len(left_norm) < 4:
        return {"status": "unavailable", "reason": "epipolar_filter_rejected_matches", "points_3d": []}
    projection_left = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)
    projection_right = np.concatenate([calibration.R, calibration.T.reshape(3, 1)], axis=1)
    homogeneous = cv2.triangulatePoints(projection_left, projection_right, left_norm.T, right_norm.T)
    xyz = (homogeneous[:3] / homogeneous[3]).T
    right_xyz = (calibration.R @ xyz.T + calibration.T.reshape(3, 1)).T
    finite = np.isfinite(xyz).all(axis=1)
    positive = (xyz[:, 2] > 50.0) & (right_xyz[:, 2] > 50.0)
    range_ok = np.linalg.norm(xyz, axis=1) < 8000.0
    xyz = xyz[finite & positive & range_ok]
    if len(xyz) < 3:
        return {"status": "unavailable", "reason": "triangulation_rejected_matches", "points_3d": []}
    center = np.median(xyz, axis=0)
    centered = xyz - center
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    projections = centered @ direction
    start = center + np.percentile(projections, 10) * direction
    end = center + np.percentile(projections, 90) * direction
    return {
        "status": "basic_candidate",
        "reason": None,
        "matched_points": int(len(good)),
        "epipolar_inliers": int(keep.sum()),
        "points_3d": xyz.astype(float).tolist(),
        "principal_line_3d": {"start": start.astype(float).tolist(), "end": end.astype(float).tolist()},
        "pca_linearity": float(singular[0] / max(float(np.sum(singular)), 1e-9)),
        "coordinate_frame": "left_camera",
        "length_unit": calibration.length_unit,
    }


class RealtimeStageWalkerWriter:
    def __init__(
        self,
        output_path: Path,
        calibration: StereoCalibration,
        processing_width: int = 480,
        reconstruction_interval: int = 10,
        keypoint_threshold: float = 0.25,
        criteria: StageCriteria | None = None,
    ) -> None:
        if processing_width < 160:
            raise ValueError("processing_width must be at least 160")
        if reconstruction_interval < 1:
            raise ValueError("reconstruction_interval must be positive")
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.output_path.open("w", encoding="utf-8", buffering=1)
        self.calibration = calibration
        self.processing_width = processing_width
        self.reconstruction_interval = reconstruction_interval
        self.keypoint_threshold = keypoint_threshold
        self.recognizer = TwoStageRecognizer(criteria)
        self.motion_windows = {
            side: TemporalMotionWindow(self.recognizer.criteria) for side in ("left", "right")
        }
        self.previous_gray: dict[str, np.ndarray] = {}
        self.previous_person: dict[str, PersonPose | None] = {}
        self.structure: dict[str, StableStructureView] = {}
        self.last_reconstruction: dict[str, Any] = {
            "status": "warming_up", "reason": "no_moving_background_updates", "points_3d": []
        }
        self.reconstruction_attempts = 0
        self.successful_reconstructions = 0
        self.failed_reconstructions = 0
        self.last_lines: dict[str, list[list[int]]] = {"left": [], "right": []}
        self.count = 0
        self.stage_counts: Counter[str] = Counter()
        self.latencies_ms: list[float] = []

    def consume(
        self,
        pair: StereoFramePair,
        left_result: InferenceResult,
        right_result: InferenceResult,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        images = {"left": pair.left.image, "right": pair.right.image}
        results = {"left": left_result, "right": right_result}
        grays: dict[str, np.ndarray] = {}
        scales: dict[str, float] = {}
        backgrounds: dict[str, dict[str, Any]] = {}
        poses: dict[str, dict[str, Any]] = {}
        masks: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            gray, scale = _resize_gray(images[side], self.processing_width)
            grays[side], scales[side] = gray, scale
            person = _primary_person(results[side])
            masks[side] = _person_exclusion_mask(gray.shape, person, self.keypoint_threshold, scale)
            if side in self.previous_gray:
                attached_exclusion = (
                    self.structure[side].exclusion_mask() if side in self.structure else None
                )
                background = estimate_background_motion(
                    self.previous_gray[side], gray, self.previous_person.get(side), person,
                    self.keypoint_threshold, scale, self.recognizer.criteria.min_background_tracks,
                    attached_exclusion,
                )
                background = self.motion_windows[side].update(background)
                vector = (
                    np.asarray(background["median_vector_px"], dtype=np.float32)
                    if background.get("available") else None
                )
                pose = keypoint_motion_evidence(
                    self.previous_person.get(side), person, self.keypoint_threshold, scale, vector,
                    np.asarray(background["affine_2x3"], dtype=np.float64)
                    if background.get("affine_2x3") is not None else None,
                )
            else:
                background = {"available": False, "reason": "first_frame", "track_count": 0}
                pose = {"common_keypoints": 0, "upper_body_keypoints": 0, "ankle_count": 0}
            backgrounds[side], poses[side] = background, pose
            if side not in self.structure:
                self.structure[side] = StableStructureView(gray.shape)

        background_for_stage, chosen = fuse_background_motion(backgrounds)
        backgrounds["fused"] = background_for_stage
        pose_for_stage = poses[chosen] if chosen is not None else {}
        confirmed, candidate, pending_count = self.recognizer.update(background_for_stage, pose_for_stage)
        stage_vote_confidence = (
            sum(self.recognizer.stage1_history) / max(1, len(self.recognizer.stage1_history))
            if confirmed == STAGE_ONE
            else sum(self.recognizer.stage2_history) / max(1, len(self.recognizer.stage2_history))
            if confirmed == STAGE_TWO
            else 0.0
        )
        moving_background = bool(background_for_stage.get("motion_active", False))
        stable_masks = {
            side: self.structure[side].update(grays[side], masks[side], moving_background)
            for side in ("left", "right")
        }
        moving_update_count = min(
            self.structure["left"].moving_updates, self.structure["right"].moving_updates
        )
        latest_attempt: dict[str, Any] | None = None
        if moving_background and moving_update_count % self.reconstruction_interval == 0:
            latest_attempt = sparse_stereo_structure(
                grays["left"], grays["right"], stable_masks["left"], stable_masks["right"],
                self.calibration, scales["left"], scales["right"],
            )
            self.reconstruction_attempts += 1
            if latest_attempt.get("status") == "basic_candidate":
                self.successful_reconstructions += 1
                self.last_reconstruction = {**latest_attempt, "source_pair_id": pair.pair_id}
            else:
                self.failed_reconstructions += 1
        if moving_background:
            self.last_lines = {
                side: _line_segments(stable_masks[side]) for side in ("left", "right")
            }
        lines = self.last_lines
        latency_ms = (time.perf_counter() - started) * 1000.0
        record = {
            "pair_id": pair.pair_id,
            "pair_timestamp_sec": pair.timestamp_sec,
            "stage": {
                "confirmed": confirmed,
                "operational": confirmed,
                "candidate": candidate,
                "candidate_streak": pending_count,
                "last_stable": self.recognizer.last_stable,
                "vote_confidence": float(stage_vote_confidence),
                "evidence_view": chosen,
                "background": backgrounds,
                "pose_motion": poses,
                "criteria": self.recognizer.criteria.__dict__,
                "truth_label_used": False,
            },
            "walker_reconstruction": {
                **self.last_reconstruction,
                "latest_attempt_status": (
                    latest_attempt.get("status") if latest_attempt is not None else "not_due"
                ),
                "latest_attempt_reason": (
                    latest_attempt.get("reason") if latest_attempt is not None else None
                ),
                "candidate_age_frames": (
                    pair.pair_id - int(self.last_reconstruction["source_pair_id"])
                    if "source_pair_id" in self.last_reconstruction else None
                ),
                "left_stable_line_segments": lines["left"],
                "right_stable_line_segments": lines["right"],
                "moving_background_updates": {
                    side: self.structure[side].moving_updates for side in ("left", "right")
                },
                "semantic_identity": "camera_attached_structure_candidate_not_class_verified",
                "processing_width": self.processing_width,
            },
            "processing_ms": latency_ms,
        }
        self.handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.count += 1
        self.stage_counts[confirmed] += 1
        self.latencies_ms.append(latency_ms)
        self.previous_gray = grays
        self.previous_person = {side: _primary_person(results[side]) for side in ("left", "right")}
        return record

    def close(self, completed: bool = True) -> dict[str, Any]:
        if not self.handle.closed:
            self.handle.close()
        values = np.asarray(self.latencies_ms, dtype=np.float64)
        return {
            "completed": bool(completed),
            "record_count": self.count,
            "output_jsonl": str(self.output_path.resolve()),
            "stage_counts": dict(self.stage_counts),
            "processing_width": self.processing_width,
            "reconstruction_interval": self.reconstruction_interval,
            "mean_processing_ms": float(np.mean(values)) if values.size else 0.0,
            "p95_processing_ms": float(np.percentile(values, 95)) if values.size else 0.0,
            "last_reconstruction_status": self.last_reconstruction.get("status", "unknown"),
            "reconstruction_attempts": self.reconstruction_attempts,
            "successful_reconstructions": self.successful_reconstructions,
            "failed_reconstructions": self.failed_reconstructions,
            "last_reconstruction_point_count": len(self.last_reconstruction.get("points_3d", [])),
            "truth_label_used": False,
        }
