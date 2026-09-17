"""Sparse four-view front end for candidate static-background stereo VO.

The front end links the same feature across left/right stereo views and two
adjacent times.  It intentionally reports the mask-policy evidence separately:
an automatic person mask is not sufficient to certify that walker pixels have
been removed.  Callers must retain that boundary when interpreting poses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np

from .calibration import StereoCalibration


@dataclass(frozen=True)
class StereoFrontendCriteria:
    orb_features: int = 2600
    ratio_test: float = 0.74
    stereo_sampson_threshold: float = 2.5e-4
    maximum_reprojection_error_px: float = 3.0
    minimum_depth_mm: float = 80.0
    maximum_range_mm: float = 8000.0
    keypoint_score_threshold: float = 0.25
    person_mask_dilation_fraction: float = 0.055

    def __post_init__(self) -> None:
        if self.orb_features < 100:
            raise ValueError("orb_features must be at least 100")
        if not 0.0 < self.ratio_test < 1.0:
            raise ValueError("ratio_test must be in (0, 1)")
        if min(
            self.stereo_sampson_threshold,
            self.maximum_reprojection_error_px,
            self.minimum_depth_mm,
            self.maximum_range_mm,
        ) <= 0.0:
            raise ValueError("geometry thresholds must be positive")
        if self.maximum_range_mm <= self.minimum_depth_mm:
            raise ValueError("maximum range must exceed minimum depth")


@dataclass(frozen=True)
class FrameStereoFeatures:
    keypoints_left: tuple[cv2.KeyPoint, ...]
    descriptors_left: np.ndarray | None
    xyz_by_left_index_mm: Mapping[int, np.ndarray]
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class LocalStereoRectification:
    map_left_x: np.ndarray
    map_left_y: np.ndarray
    map_right_x: np.ndarray
    map_right_y: np.ndarray
    rotation_left: np.ndarray
    virtual_K: np.ndarray
    translation_right: np.ndarray


@dataclass(frozen=True)
class RectifiedFrameFeatures:
    keypoints_left: tuple[cv2.KeyPoint, ...]
    descriptors_left: np.ndarray | None
    xyz_by_left_index_mm: Mapping[int, np.ndarray]
    rectification: LocalStereoRectification
    diagnostics: Mapping[str, Any]


def upright_persons_to_raw(
    persons: list[dict[str, Any]],
    side: str,
    upright_size: tuple[int, int],
) -> list[dict[str, Any]]:
    """Rotate saved upright-view person keypoints back to raw fisheye pixels."""

    width, height = upright_size
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    converted = []
    for person in persons:
        item = dict(person)
        keypoints = []
        for point in person.get("keypoints", []):
            if len(point) < 2:
                keypoints.append(list(point))
                continue
            x, y = float(point[0]), float(point[1])
            if side == "left":
                raw_x, raw_y = height - 1.0 - y, x
            else:
                raw_x, raw_y = y, width - 1.0 - x
            keypoints.append([raw_x, raw_y, *point[2:]])
        item["keypoints"] = keypoints
        converted.append(item)
    return converted


def person_exclusion_feature_mask(
    image_shape: tuple[int, int],
    persons: list[dict[str, Any]],
    *,
    scale: float,
    keypoint_score_threshold: float,
    dilation_fraction: float,
) -> np.ndarray:
    """Return an ORB feature mask with detected people set to zero."""

    height, width = image_shape
    allowed = np.full((height, width), 255, dtype=np.uint8)
    for person in persons:
        points = []
        for item in person.get("keypoints", []):
            if len(item) >= 3 and float(item[2]) >= keypoint_score_threshold:
                point = np.asarray(item[:2], dtype=np.float64) * float(scale)
                if np.isfinite(point).all():
                    points.append(point)
        if len(points) >= 3:
            hull = cv2.convexHull(np.rint(np.asarray(points)).astype(np.int32))
            excluded = np.zeros_like(allowed)
            cv2.fillConvexPoly(excluded, hull, 255)
            radius = max(10, round(dilation_fraction * max(height, width)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            excluded = cv2.dilate(excluded, kernel)
            allowed[excluded > 0] = 0
        elif points:
            radius = max(12, round(dilation_fraction * max(height, width)))
            for point in points:
                cv2.circle(allowed, tuple(np.rint(point).astype(int)), radius, 0, -1)
    border = max(4, round(0.0125 * min(height, width)))
    allowed[:border] = 0
    allowed[-border:] = 0
    allowed[:, :border] = 0
    allowed[:, -border:] = 0
    return allowed


def mutual_ratio_matches(
    descriptors_a: np.ndarray | None,
    descriptors_b: np.ndarray | None,
    *,
    ratio: float,
) -> list[cv2.DMatch]:
    """Lowe-ratio matches that also agree in the reverse direction."""

    if descriptors_a is None or descriptors_b is None or len(descriptors_a) < 2 or len(descriptors_b) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    forward_knn = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    reverse_knn = matcher.knnMatch(descriptors_b, descriptors_a, k=2)
    forward = {
        items[0].queryIdx: items[0]
        for items in forward_knn
        if len(items) == 2 and items[0].distance < ratio * items[1].distance
    }
    reverse = {
        items[0].queryIdx: items[0].trainIdx
        for items in reverse_knn
        if len(items) == 2 and items[0].distance < ratio * items[1].distance
    }
    return [
        item for query, item in forward.items()
        if reverse.get(item.trainIdx) == query
    ]


def sampson_error(left: np.ndarray, right: np.ndarray, essential: np.ndarray) -> np.ndarray:
    ones = np.ones((len(left), 1), dtype=np.float64)
    x1 = np.concatenate([left, ones], axis=1)
    x2 = np.concatenate([right, ones], axis=1)
    ex1 = (essential @ x1.T).T
    etx2 = (essential.T @ x2.T).T
    numerator = np.sum(x2 * ex1, axis=1) ** 2
    denominator = ex1[:, 0] ** 2 + ex1[:, 1] ** 2 + etx2[:, 0] ** 2 + etx2[:, 1] ** 2
    return numerator / np.maximum(denominator, 1e-12)


def triangulate_indexed_matches(
    calibration: StereoCalibration,
    keypoints_left: tuple[cv2.KeyPoint, ...],
    keypoints_right: tuple[cv2.KeyPoint, ...],
    matches: list[cv2.DMatch],
    criteria: StereoFrontendCriteria,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    if not matches:
        return {}, {"stereo_match_count": 0, "epipolar_inliers": 0, "geometry_inliers": 0}
    raw_left = np.asarray([keypoints_left[item.queryIdx].pt for item in matches], dtype=np.float64)
    raw_right = np.asarray([keypoints_right[item.trainIdx].pt for item in matches], dtype=np.float64)
    normalized_left = calibration.undistort_normalized(raw_left, "left")
    normalized_right = calibration.undistort_normalized(raw_right, "right")
    epipolar = sampson_error(normalized_left, normalized_right, calibration.essential_matrix)
    keep_epi = epipolar <= criteria.stereo_sampson_threshold
    if int(keep_epi.sum()) == 0:
        return {}, {
            "stereo_match_count": len(matches), "epipolar_inliers": 0, "geometry_inliers": 0,
            "sampson_median": float(np.median(epipolar)),
        }
    left_epi = normalized_left[keep_epi]
    right_epi = normalized_right[keep_epi]
    selected_matches = [item for item, keep in zip(matches, keep_epi) if keep]
    projection_left = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)
    projection_right = np.concatenate([calibration.R, calibration.T.reshape(3, 1)], axis=1)
    homogeneous = cv2.triangulatePoints(projection_left, projection_right, left_epi.T, right_epi.T)
    xyz = (homogeneous[:3] / homogeneous[3]).T
    right_xyz = (calibration.R @ xyz.T + calibration.T.reshape(3, 1)).T
    projected_left = calibration.project_left(xyz)
    projected_right = calibration.project_right(xyz)
    raw_left_epi = raw_left[keep_epi]
    raw_right_epi = raw_right[keep_epi]
    reprojection = np.maximum(
        np.linalg.norm(projected_left - raw_left_epi, axis=1),
        np.linalg.norm(projected_right - raw_right_epi, axis=1),
    )
    valid = (
        np.isfinite(xyz).all(axis=1)
        & (xyz[:, 2] >= criteria.minimum_depth_mm)
        & (right_xyz[:, 2] >= criteria.minimum_depth_mm)
        & (np.linalg.norm(xyz, axis=1) <= criteria.maximum_range_mm)
        & (reprojection <= criteria.maximum_reprojection_error_px)
    )
    result = {
        int(item.queryIdx): point
        for item, point, accepted in zip(selected_matches, xyz, valid)
        if accepted
    }
    accepted_reprojection = reprojection[valid]
    return result, {
        "stereo_match_count": len(matches),
        "epipolar_inliers": int(keep_epi.sum()),
        "geometry_inliers": len(result),
        "sampson_median": float(np.median(epipolar)),
        "reprojection_median_px": (
            float(np.median(accepted_reprojection)) if len(accepted_reprojection) else None
        ),
    }


def extract_frame_stereo_features(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    calibration: StereoCalibration,
    criteria: StereoFrontendCriteria,
) -> FrameStereoFeatures:
    gray_left = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=criteria.orb_features, fastThreshold=10, edgeThreshold=19)
    keypoints_left, descriptors_left = orb.detectAndCompute(gray_left, left_mask)
    keypoints_right, descriptors_right = orb.detectAndCompute(gray_right, right_mask)
    keypoints_left = tuple(keypoints_left or [])
    keypoints_right = tuple(keypoints_right or [])
    stereo_matches = mutual_ratio_matches(descriptors_left, descriptors_right, ratio=criteria.ratio_test)
    xyz, geometry = triangulate_indexed_matches(
        calibration, keypoints_left, keypoints_right, stereo_matches, criteria
    )
    diagnostics = {
        "left_feature_count": len(keypoints_left),
        "right_feature_count": len(keypoints_right),
        **geometry,
    }
    return FrameStereoFeatures(keypoints_left, descriptors_left, xyz, diagnostics)


def adjacent_3d_correspondences(
    previous: FrameStereoFeatures,
    current: FrameStereoFeatures,
    criteria: StereoFrontendCriteria,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    temporal = mutual_ratio_matches(
        previous.descriptors_left, current.descriptors_left, ratio=criteria.ratio_test
    )
    previous_points = []
    current_points = []
    pixel_displacements = []
    for item in temporal:
        if item.queryIdx not in previous.xyz_by_left_index_mm or item.trainIdx not in current.xyz_by_left_index_mm:
            continue
        previous_points.append(previous.xyz_by_left_index_mm[item.queryIdx])
        current_points.append(current.xyz_by_left_index_mm[item.trainIdx])
        old_px = np.asarray(previous.keypoints_left[item.queryIdx].pt, dtype=np.float64)
        new_px = np.asarray(current.keypoints_left[item.trainIdx].pt, dtype=np.float64)
        pixel_displacements.append(float(np.linalg.norm(new_px - old_px)))
    old = np.asarray(previous_points, dtype=np.float64).reshape(-1, 3)
    new = np.asarray(current_points, dtype=np.float64).reshape(-1, 3)
    return old, new, {
        "temporal_match_count": len(temporal),
        "four_view_correspondence_count": len(old),
        "left_temporal_displacement_median_px": (
            float(np.median(pixel_displacements)) if pixel_displacements else None
        ),
    }


def make_local_rectification(
    calibration: StereoCalibration,
    runtime_size: tuple[int, int],
    *,
    focal_px: float,
    seed_fraction_xy: tuple[float, float] = (0.5, 0.25),
) -> LocalStereoRectification:
    """Create one fixed virtual pinhole view aimed into the stereo overlap.

    The seed only selects the virtual viewing direction.  It is fixed for the
    whole sequence and therefore cannot inject per-frame person or foot motion
    into the camera-motion estimate.
    """

    if calibration.camera_model != "fisheye":
        raise ValueError("local rectification currently requires fisheye calibration")
    width, height = runtime_size
    if focal_px <= 0.0 or not (0.0 <= seed_fraction_xy[0] <= 1.0 and 0.0 <= seed_fraction_xy[1] <= 1.0):
        raise ValueError("invalid local-rectification focal length or seed")
    seed = np.asarray((seed_fraction_xy[0] * width, seed_fraction_xy[1] * height), dtype=np.float64)

    def ray(side: str) -> np.ndarray:
        normalized = calibration.undistort_normalized(seed.reshape(1, 2), side)[0]
        value = np.asarray((normalized[0], normalized[1], 1.0), dtype=np.float64)
        return value / np.linalg.norm(value)

    left_ray = ray("left")
    right_ray_left = calibration.R.T @ ray("right")
    baseline_left = -calibration.R.T @ calibration.T
    baseline_left /= np.linalg.norm(baseline_left)
    forward = left_ray + right_ray_left
    forward -= baseline_left * float(forward @ baseline_left)
    if np.linalg.norm(forward) < 1e-9:
        raise ValueError("local rectification seed is degenerate")
    forward /= np.linalg.norm(forward)
    vertical = np.cross(forward, baseline_left)
    vertical /= np.linalg.norm(vertical)
    rotation_left = np.stack((baseline_left, vertical, forward))
    rotation_right = rotation_left @ calibration.R.T
    translation_right = rotation_right @ calibration.T
    if abs(float(translation_right[1])) > 1e-5 or abs(float(translation_right[2])) > 1e-5:
        raise ValueError("local rectification failed to align the stereo baseline")
    virtual_K = np.asarray(
        ((focal_px, 0.0, width / 2.0), (0.0, focal_px, height / 2.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    left_map_x, left_map_y = cv2.fisheye.initUndistortRectifyMap(
        calibration.left_K, calibration.left_D.reshape(-1, 1), rotation_left,
        virtual_K, runtime_size, cv2.CV_32FC1,
    )
    right_map_x, right_map_y = cv2.fisheye.initUndistortRectifyMap(
        calibration.right_K, calibration.right_D.reshape(-1, 1), rotation_right,
        virtual_K, runtime_size, cv2.CV_32FC1,
    )
    return LocalStereoRectification(
        left_map_x, left_map_y, right_map_x, right_map_y,
        rotation_left, virtual_K, translation_right,
    )


def _rectified_triangulation(
    keypoints_left: tuple[cv2.KeyPoint, ...],
    keypoints_right: tuple[cv2.KeyPoint, ...],
    matches: list[cv2.DMatch],
    rectification: LocalStereoRectification,
    criteria: StereoFrontendCriteria,
    *,
    maximum_vertical_error_px: float,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    selected = [
        item for item in matches
        if abs(keypoints_left[item.queryIdx].pt[1] - keypoints_right[item.trainIdx].pt[1])
        <= maximum_vertical_error_px
    ]
    if not selected:
        return {}, {"stereo_match_count": len(matches), "rectified_epipolar_inliers": 0, "geometry_inliers": 0}
    left = np.asarray([keypoints_left[item.queryIdx].pt for item in selected], dtype=np.float64)
    right = np.asarray([keypoints_right[item.trainIdx].pt for item in selected], dtype=np.float64)
    projection_left = rectification.virtual_K @ np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
    projection_right = rectification.virtual_K @ np.concatenate(
        (np.eye(3), rectification.translation_right.reshape(3, 1)), axis=1
    )
    homogeneous = cv2.triangulatePoints(projection_left, projection_right, left.T, right.T)
    xyz_rectified = (homogeneous[:3] / homogeneous[3]).T
    xyz_right = xyz_rectified + rectification.translation_right
    projected_left = (projection_left @ np.column_stack((xyz_rectified, np.ones(len(xyz_rectified)))).T).T
    projected_right = (projection_right @ np.column_stack((xyz_rectified, np.ones(len(xyz_rectified)))).T).T
    projected_left = projected_left[:, :2] / projected_left[:, 2:3]
    projected_right = projected_right[:, :2] / projected_right[:, 2:3]
    reprojection = np.maximum(
        np.linalg.norm(projected_left - left, axis=1),
        np.linalg.norm(projected_right - right, axis=1),
    )
    xyz_left = (rectification.rotation_left.T @ xyz_rectified.T).T
    valid = (
        np.isfinite(xyz_left).all(axis=1)
        & (xyz_rectified[:, 2] >= criteria.minimum_depth_mm)
        & (xyz_right[:, 2] >= criteria.minimum_depth_mm)
        & (np.linalg.norm(xyz_left, axis=1) <= criteria.maximum_range_mm)
        & (reprojection <= criteria.maximum_reprojection_error_px)
    )
    result = {
        int(item.queryIdx): point
        for item, point, accepted in zip(selected, xyz_left, valid)
        if accepted
    }
    return result, {
        "stereo_match_count": len(matches),
        "rectified_epipolar_inliers": len(selected),
        "geometry_inliers": len(result),
        "vertical_error_median_px": float(np.median(np.abs(left[:, 1] - right[:, 1]))),
        "reprojection_median_px": float(np.median(reprojection[valid])) if np.any(valid) else None,
    }


def extract_rectified_frame_features(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    rectification: LocalStereoRectification,
    criteria: StereoFrontendCriteria,
    *,
    maximum_vertical_error_px: float = 2.5,
    use_dense_feature_depth: bool = False,
    num_disparities: int = 192,
    lr_consistency_px: float = 1.5,
    dense_scale: float = 1.0,
) -> RectifiedFrameFeatures:
    left = cv2.remap(left_bgr, rectification.map_left_x, rectification.map_left_y, cv2.INTER_LINEAR)
    right = cv2.remap(right_bgr, rectification.map_right_x, rectification.map_right_y, cv2.INTER_LINEAR)
    left_allowed = cv2.remap(left_mask, rectification.map_left_x, rectification.map_left_y, cv2.INTER_NEAREST)
    right_allowed = cv2.remap(right_mask, rectification.map_right_x, rectification.map_right_y, cv2.INTER_NEAREST)
    orb = cv2.ORB_create(nfeatures=criteria.orb_features, fastThreshold=7, edgeThreshold=19)
    keypoints_left, descriptors_left = orb.detectAndCompute(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), left_allowed)
    keypoints_right, descriptors_right = orb.detectAndCompute(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), right_allowed)
    keypoints_left = tuple(keypoints_left or [])
    keypoints_right = tuple(keypoints_right or [])
    matches = mutual_ratio_matches(descriptors_left, descriptors_right, ratio=criteria.ratio_test)
    sparse_xyz, geometry = _rectified_triangulation(
        keypoints_left, keypoints_right, matches, rectification, criteria,
        maximum_vertical_error_px=maximum_vertical_error_px,
    )
    if use_dense_feature_depth:
        xyz, dense_geometry = _dense_depth_at_keypoints(
            left, right, left_allowed, right_allowed, keypoints_left, rectification, criteria,
            num_disparities=num_disparities, lr_consistency_px=lr_consistency_px,
            dense_scale=dense_scale,
        )
    else:
        xyz, dense_geometry = sparse_xyz, {"dense_feature_depth_enabled": False}
    return RectifiedFrameFeatures(
        keypoints_left, descriptors_left, xyz, rectification,
        {
            "left_feature_count": len(keypoints_left), "right_feature_count": len(keypoints_right),
            **geometry, **dense_geometry,
        },
    )


def _sgbm_disparity(
    reference: np.ndarray, partner: np.ndarray, num_disparities: int, *, reverse: bool,
) -> np.ndarray:
    if num_disparities < 16 or num_disparities % 16:
        raise ValueError("num_disparities must be a positive multiple of 16")
    minimum = -(num_disparities - 1) if reverse else 0
    matcher = cv2.StereoSGBM_create(
        minDisparity=minimum, numDisparities=num_disparities, blockSize=7,
        P1=8 * 7 * 7, P2=32 * 7 * 7, disp12MaxDiff=1,
        uniquenessRatio=12, speckleWindowSize=60, speckleRange=2,
        preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    first = clahe.apply(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY))
    second = clahe.apply(cv2.cvtColor(partner, cv2.COLOR_BGR2GRAY))
    raw = matcher.compute(first, second).astype(np.float32) / 16.0
    if not reverse:
        return raw
    valid = (raw > float(minimum)) & (raw <= 0.0)
    return np.where(valid, -raw, 0.0).astype(np.float32)


def _dense_depth_at_keypoints(
    left: np.ndarray,
    right: np.ndarray,
    left_allowed: np.ndarray,
    right_allowed: np.ndarray,
    keypoints_left: tuple[cv2.KeyPoint, ...],
    rectification: LocalStereoRectification,
    criteria: StereoFrontendCriteria,
    *,
    num_disparities: int,
    lr_consistency_px: float,
    dense_scale: float,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Sample left/right-consistent dense stereo only at temporal ORB sites."""

    if float(rectification.translation_right[0]) >= 0.0:
        raise ValueError("dense feature depth expects negative rectified right translation")
    if not 0.2 <= dense_scale <= 1.0:
        raise ValueError("dense_scale must be in [0.2, 1.0]")
    if dense_scale < 1.0:
        size = (max(1, round(left.shape[1] * dense_scale)), max(1, round(left.shape[0] * dense_scale)))
        left = cv2.resize(left, size, interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, size, interpolation=cv2.INTER_AREA)
        left_allowed = cv2.resize(left_allowed, size, interpolation=cv2.INTER_NEAREST)
        right_allowed = cv2.resize(right_allowed, size, interpolation=cv2.INTER_NEAREST)
    forward = _sgbm_disparity(left, right, num_disparities, reverse=False)
    reverse = _sgbm_disparity(right, left, num_disparities, reverse=True)
    height, width = forward.shape
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    focal = float(rectification.virtual_K[0, 0] * dense_scale)
    principal_x = float(rectification.virtual_K[0, 2] * dense_scale)
    principal_y = float(rectification.virtual_K[1, 2] * dense_scale)
    depth_scale = abs(float(focal * rectification.translation_right[0]))
    result: dict[int, np.ndarray] = {}
    rejection = {"invalid_disparity": 0, "out_of_bounds_or_mask": 0, "lr_or_photo": 0, "range": 0}
    for index, keypoint in enumerate(keypoints_left):
        x, y = np.rint(np.asarray(keypoint.pt) * dense_scale).astype(int)
        if not (0 <= x < width and 0 <= y < height):
            rejection["out_of_bounds_or_mask"] += 1
            continue
        disparity = float(forward[y, x])
        if disparity <= 1.0:
            rejection["invalid_disparity"] += 1
            continue
        partner_x = int(round(x - disparity))
        if (
            partner_x < 0 or partner_x >= width or left_allowed[y, x] == 0
            or right_allowed[y, partner_x] == 0
        ):
            rejection["out_of_bounds_or_mask"] += 1
            continue
        reverse_value = float(reverse[y, partner_x])
        photo = abs(int(left_gray[y, x]) - int(right_gray[y, partner_x]))
        if reverse_value <= 1.0 or abs(disparity - reverse_value) > lr_consistency_px or photo > 45:
            rejection["lr_or_photo"] += 1
            continue
        depth = depth_scale / disparity
        point_rectified = np.asarray(
            (
                (x - principal_x) * depth / focal,
                (y - principal_y) * depth / focal,
                depth,
            ), dtype=np.float64,
        )
        point_left = rectification.rotation_left.T @ point_rectified
        if (
            depth < criteria.minimum_depth_mm
            or not np.isfinite(point_left).all()
            or np.linalg.norm(point_left) > criteria.maximum_range_mm
        ):
            rejection["range"] += 1
            continue
        result[index] = point_left
    return result, {
        "dense_feature_depth_enabled": True,
        "dense_feature_depth_count": len(result),
        "dense_feature_depth_rejections": rejection,
        "dense_num_disparities": num_disparities,
        "dense_lr_consistency_px": lr_consistency_px,
        "dense_scale": dense_scale,
    }


def adjacent_rectified_correspondences(
    previous: RectifiedFrameFeatures,
    current: RectifiedFrameFeatures,
    criteria: StereoFrontendCriteria,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return previous 3-D, current 3-D, and current 2-D temporal links."""

    temporal = mutual_ratio_matches(
        previous.descriptors_left, current.descriptors_left, ratio=criteria.ratio_test
    )
    previous_3d_pnp: list[np.ndarray] = []
    current_2d_pnp: list[np.ndarray] = []
    previous_3d_both: list[np.ndarray] = []
    current_3d_both: list[np.ndarray] = []
    for item in temporal:
        if item.queryIdx in previous.xyz_by_left_index_mm:
            previous_3d_pnp.append(previous.xyz_by_left_index_mm[item.queryIdx])
            current_2d_pnp.append(np.asarray(current.keypoints_left[item.trainIdx].pt, dtype=np.float64))
            if item.trainIdx in current.xyz_by_left_index_mm:
                previous_3d_both.append(previous.xyz_by_left_index_mm[item.queryIdx])
                current_3d_both.append(current.xyz_by_left_index_mm[item.trainIdx])
    return (
        np.asarray(previous_3d_both, dtype=np.float64).reshape(-1, 3),
        np.asarray(current_3d_both, dtype=np.float64).reshape(-1, 3),
        np.column_stack((
            np.asarray(previous_3d_pnp, dtype=np.float64).reshape(-1, 3),
            np.asarray(current_2d_pnp, dtype=np.float64).reshape(-1, 2),
        )) if previous_3d_pnp else np.empty((0, 5), dtype=np.float64),
        {
            "temporal_match_count": len(temporal),
            "three_d_three_d_count": len(previous_3d_both),
            "three_d_two_d_count": len(previous_3d_pnp),
        },
    )


def estimate_rectified_pnp_candidate(
    linked_3d_2d: np.ndarray,
    rectification: LocalStereoRectification,
    *,
    reprojection_threshold_px: float,
    iterations: int,
    minimum_inliers: int,
) -> dict[str, Any]:
    """Estimate ``X_current_left = R X_previous_left + q`` with PnP RANSAC."""

    if len(linked_3d_2d) < 4:
        return {"status": "unavailable", "reason": "fewer_than_four_3d_2d_links", "correspondence_count": len(linked_3d_2d)}
    object_points = linked_3d_2d[:, :3].astype(np.float64)
    image_points = linked_3d_2d[:, 3:].astype(np.float64)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points, image_points, rectification.virtual_K, None,
        iterationsCount=iterations, reprojectionError=reprojection_threshold_px,
        confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
    )
    count = 0 if inliers is None else int(len(inliers))
    if not success or count < minimum_inliers:
        return {
            "status": "rejected", "reason": "pnp_failed_or_too_few_inliers",
            "correspondence_count": len(linked_3d_2d), "ransac_inlier_count": count,
        }
    selected = inliers.reshape(-1)
    success, rvec, tvec = cv2.solvePnP(
        object_points[selected], image_points[selected], rectification.virtual_K, None,
        rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return {"status": "rejected", "reason": "pnp_refinement_failed", "ransac_inlier_count": count}
    rotation_rectified, _ = cv2.Rodrigues(rvec)
    rotation_left = rectification.rotation_left.T @ rotation_rectified
    translation_left = rectification.rotation_left.T @ tvec.reshape(3)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, rectification.virtual_K, None)
    residual = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    return {
        "status": "accepted", "reason": None,
        "R_to_from": rotation_left.tolist(), "q_to_from": translation_left.tolist(),
        "correspondence_count": len(linked_3d_2d), "ransac_inlier_count": count,
        "ransac_inlier_fraction": count / len(linked_3d_2d),
        "reprojection_inlier_median_px": float(np.median(residual[selected])),
        "reprojection_inlier_p95_px": float(np.percentile(residual[selected], 95)),
    }
