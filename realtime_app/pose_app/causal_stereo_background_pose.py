"""Experimental metric stereo background edges with withheld reprojection checks.

Automatic person exclusion does not certify walker exclusion or static objects.
This module is diagnostic until that separate evidence gate passes.
"""

import time

import cv2
import numpy as np

from .static_background_rotation import RealtimeStereoRotationTracker
from .static_background_stereo_frontend import (
    StereoFrontendCriteria,
    extract_frame_stereo_features,
    extract_rectified_frame_features,
    make_local_rectification,
    mutual_ratio_matches,
    person_exclusion_feature_mask,
)


def _spread_indexes(points, indexes, limit):
    if len(indexes) <= limit:
        return indexes
    pool = points[indexes]
    picked = [int(np.argmax(np.linalg.norm(pool - pool.mean(axis=0), axis=1)))]
    while len(picked) < limit:
        distance = np.min(
            np.linalg.norm(pool[:, None] - pool[picked][None], axis=2), axis=1
        )
        distance[picked] = -1
        picked.append(int(np.argmax(distance)))
    return indexes[picked]


def estimate_background_edge(xyz, normalized):
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    normalized = np.asarray(normalized, float).reshape(-1, 2)
    if (
        len(xyz) != len(normalized)
        or not np.isfinite(xyz).all()
        or not np.isfinite(normalized).all()
    ):
        raise ValueError("invalid background correspondences")
    available_count = len(xyz)
    if len(xyz) < 6:
        return {
            "status": "unavailable",
            "reason": "fewer_than_six_links",
            "correspondence_count": len(xyz),
        }
    original_indexes = np.arange(len(xyz))
    held_indexes = _spread_indexes(
        normalized, original_indexes[original_indexes % 5 == 0], 4
    )
    train_indexes = _spread_indexes(
        normalized, original_indexes[original_indexes % 5 != 0], 16
    )
    selected = np.sort(np.concatenate((held_indexes, train_indexes)))
    xyz, normalized = xyz[selected], normalized[selected]
    hold = selected % 5 == 0
    train = ~hold
    if (
        np.linalg.svd(normalized - normalized.mean(axis=0), compute_uv=False)[1] < 0.01
        or np.linalg.svd(xyz - xyz.mean(axis=0), compute_uv=False)[1] < 20
    ):
        return {
            "status": "rejected",
            "reason": "spatially_degenerate_links",
            "correspondence_count": len(xyz),
            "available_count": available_count,
        }
    # Normalized camera coordinates: thresholds are dimensionless, not raw pixels.
    ok, rv, tv, inliers = cv2.solvePnPRansac(
        xyz[train],
        normalized[train],
        np.eye(3),
        None,
        iterationsCount=150,
        reprojectionError=0.004,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    count = 0 if inliers is None else len(inliers)
    base = {
        "correspondence_count": len(xyz),
        "training_inliers": count,
        "withheld_count": int(hold.sum()),
        "available_count": available_count,
        "selected_original_indexes": selected.tolist(),
    }
    if not ok or count < max(4, int(np.ceil(0.6 * train.sum()))):
        return dict(base, status="rejected", reason="pnp_inliers_insufficient")
    idx = np.flatnonzero(train)[inliers.ravel()]
    ok, rv, tv = cv2.solvePnP(
        xyz[idx],
        normalized[idx],
        np.eye(3),
        None,
        rv,
        tv,
        True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return dict(base, status="rejected", reason="pnp_refinement_failed")
    rot = cv2.Rodrigues(rv)[0]
    moved = xyz[hold] @ rot.T + tv.reshape(3)
    error = np.linalg.norm(moved[:, :2] / moved[:, 2:3] - normalized[hold], axis=1)
    base["withheld_p95_normalized"] = float(np.percentile(error, 95))
    angle = float(np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1))))
    base["rotation_step_deg"] = angle
    base["translation_step_mm"] = float(np.linalg.norm(tv))
    if np.any(moved[:, 2] <= 0) or base["withheld_p95_normalized"] > 0.008:
        return dict(base, status="rejected", reason="withheld_reprojection_failed")
    # After the independent check passes, use every accepted point, capped at 20.
    all_valid = np.sort(np.concatenate((idx, np.flatnonzero(hold))))
    ok, rv, tv = cv2.solvePnP(
        xyz[all_valid],
        normalized[all_valid],
        np.eye(3),
        None,
        rv,
        tv,
        True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return dict(base, status="rejected", reason="all_point_refinement_failed")
    rot = cv2.Rodrigues(rv)[0]
    base["final_solver_point_count"] = len(all_valid)
    base["withheld_check_before_all_point_refinement"] = True
    angle = float(np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1))))
    base["rotation_step_deg"] = angle
    base["translation_step_mm"] = float(np.linalg.norm(tv))
    final_moved = xyz[all_valid] @ rot.T + tv.reshape(3)
    if not np.isfinite(final_moved).all() or np.any(final_moved[:, 2] <= 0):
        return dict(base, status="rejected", reason="all_point_nonpositive_depth")
    final_error = np.linalg.norm(
        final_moved[:, :2] / final_moved[:, 2:3] - normalized[all_valid], axis=1
    )
    base["final_maximum_reprojection_normalized"] = float(final_error.max())
    if final_error.max() > 0.008:
        return dict(base, status="rejected", reason="all_point_reprojection_failed")
    if angle > 5 or np.linalg.norm(tv) > 120:
        return dict(base, status="rejected", reason="relative_motion_limit")
    return dict(
        base,
        status="accepted",
        reason=None,
        R_to_from=rot.tolist(),
        t_to_from_mm=tv.reshape(3).tolist(),
    )


class CausalStereoBackgroundPose:
    def __init__(
        self, calibration, processing_width=480, *, rectified=False, focal_fraction=0.65
    ):
        if calibration.length_unit not in ("mm", "millimeter"):
            raise ValueError("background VO requires millimeter calibration")
        if processing_width <= 0:
            raise ValueError("processing width must be positive")
        self.calibration = calibration
        self.width = processing_width
        self.previous = None
        self.rectified = rectified
        if not 0 < focal_fraction <= 1:
            raise ValueError("invalid virtual focal fraction")
        self.focal_fraction = focal_fraction
        self.rectification = None
        self.criteria = StereoFrontendCriteria()

    def update(self, left, right, left_result, right_result):
        started = time.perf_counter()
        images, masks = {}, {}
        for side, source, result in [
            ("left", left, left_result),
            ("right", right, right_result),
        ]:
            scale = min(1.0, self.width / source.shape[1])
            image = cv2.resize(
                source, (round(source.shape[1] * scale), round(source.shape[0] * scale))
            )
            images[side] = image
            masks[side] = person_exclusion_feature_mask(
                image.shape[:2],
                RealtimeStereoRotationTracker._person_mapping(result),
                scale=scale,
                keypoint_score_threshold=0.25,
                dilation_fraction=0.055,
            )
        cal = self.calibration.for_runtime_sizes(
            (images["left"].shape[1], images["left"].shape[0]),
            (images["right"].shape[1], images["right"].shape[0]),
        )
        if self.rectified:
            if self.rectification is None:
                self.rectification = make_local_rectification(
                    cal, cal.left_image_size, focal_px=self.width * self.focal_fraction
                )
            current = extract_rectified_frame_features(
                images["left"],
                images["right"],
                masks["left"],
                masks["right"],
                self.rectification,
                self.criteria,
            )
        else:
            current = extract_frame_stereo_features(
                images["left"],
                images["right"],
                masks["left"],
                masks["right"],
                cal,
                self.criteria,
            )
        result = {"status": "unavailable", "reason": "first_frame"}
        if self.previous is not None:
            matches = mutual_ratio_matches(
                self.previous.descriptors_left,
                current.descriptors_left,
                ratio=self.criteria.ratio_test,
            )
            linked = [
                m for m in matches if m.queryIdx in self.previous.xyz_by_left_index_mm
            ]
            old = np.asarray(
                [self.previous.xyz_by_left_index_mm[m.queryIdx] for m in linked]
            ).reshape(-1, 3)
            pixels = np.asarray(
                [current.keypoints_left[m.trainIdx].pt for m in linked]
            ).reshape(-1, 2)
            if self.rectified:
                homogeneous = np.column_stack((pixels, np.ones(len(pixels))))
                rays = homogeneous @ np.linalg.inv(self.rectification.virtual_K).T
                rays = rays @ self.rectification.rotation_left
                valid = np.isfinite(rays).all(axis=1) & (rays[:, 2] > 0.05)
                old = old[valid]
                normalized = rays[valid, :2] / rays[valid, 2:3]
            else:
                normalized = (
                    cal.undistort_normalized(pixels, "left") if len(pixels) else pixels
                )
            result = estimate_background_edge(old, normalized)
        self.previous = current
        return dict(
            result,
            frontend=dict(current.diagnostics),
            processing_ms=(time.perf_counter() - started) * 1000,
            mask_certified=False,
            rectified_frontend=self.rectified,
            focal_fraction=self.focal_fraction if self.rectified else None,
            metric_pose_usable=False,
            evidence="person mask only; walker/background static membership unverified",
        )
