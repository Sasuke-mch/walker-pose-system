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


def estimate_background_edge(xyz, normalized):
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    normalized = np.asarray(normalized, float).reshape(-1, 2)
    if (
        len(xyz) != len(normalized)
        or not np.isfinite(xyz).all()
        or not np.isfinite(normalized).all()
    ):
        raise ValueError("invalid background correspondences")
    if len(xyz) < 20:
        return {
            "status": "unavailable",
            "reason": "fewer_than_twenty_links",
            "correspondence_count": len(xyz),
        }
    hold = np.arange(len(xyz)) % 5 == 0
    train = ~hold
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
    }
    if not ok or count < 12:
        return dict(base, status="rejected", reason="pnp_inliers_below_twelve")
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
    def __init__(self, calibration, processing_width=480, *, rectified=False):
        if calibration.length_unit not in ("mm", "millimeter"):
            raise ValueError("background VO requires millimeter calibration")
        self.calibration = calibration
        self.width = processing_width
        self.previous = None
        self.rectified = rectified
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
                    cal, cal.left_image_size, focal_px=self.width * 0.65
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
            metric_pose_usable=False,
            evidence="person mask only; walker/background static membership unverified",
        )
