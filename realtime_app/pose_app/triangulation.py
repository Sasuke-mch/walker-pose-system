from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import cv2
import numpy as np

from .calibration import StereoCalibration
from .schema import PersonPose


COCO17_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(frozen=True)
class PersonMatch:
    left_index: int
    right_index: int
    association_cost: float
    common_keypoints: int


@dataclass
class TriangulatedPerson:
    stereo_person_id: int
    left_person_id: int
    right_person_id: int
    association_cost: float
    common_keypoints: int
    valid_keypoints: int
    mean_reprojection_error_px: float | None
    keypoints_3d: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stereo_person_id": self.stereo_person_id,
            "left_person_id": self.left_person_id,
            "right_person_id": self.right_person_id,
            "association_cost": (self.association_cost if math.isfinite(self.association_cost) else None),
            "common_keypoints": self.common_keypoints,
            "valid_keypoints": self.valid_keypoints,
            "mean_reprojection_error_px": self.mean_reprojection_error_px,
            "keypoints_3d": self.keypoints_3d,
        }


def _visible_pairs(
    left: PersonPose,
    right: PersonPose,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    left_points: list[list[float]] = []
    right_points: list[list[float]] = []
    indices: list[int] = []
    for index, (left_point, right_point) in enumerate(
        zip(left.keypoints, right.keypoints)
    ):
        if (
            left_point[2] >= threshold
            and right_point[2] >= threshold
            and math.isfinite(left_point[0])
            and math.isfinite(left_point[1])
            and math.isfinite(right_point[0])
            and math.isfinite(right_point[1])
        ):
            left_points.append([left_point[0], left_point[1]])
            right_points.append([right_point[0], right_point[1]])
            indices.append(index)
    return (
        np.asarray(left_points, dtype=np.float64).reshape(-1, 2),
        np.asarray(right_points, dtype=np.float64).reshape(-1, 2),
        indices,
    )


def _bbox_center(person: PersonPose) -> np.ndarray:
    x1, y1, x2, y2 = person.bbox
    return np.asarray([[(x1 + x2) * 0.5, (y1 + y2) * 0.5]], dtype=np.float64)


def _epipolar_cost(
    left_points: np.ndarray,
    right_points: np.ndarray,
    calibration: StereoCalibration,
) -> float:
    if len(left_points) == 0:
        return float("inf")
    left_normalized = calibration.undistort_normalized(left_points, "left")
    right_normalized = calibration.undistort_normalized(right_points, "right")
    ones = np.ones((len(left_normalized), 1), dtype=np.float64)
    x1 = np.concatenate([left_normalized, ones], axis=1)
    x2 = np.concatenate([right_normalized, ones], axis=1)
    E = calibration.essential_matrix
    Ex1 = (E @ x1.T).T
    Etx2 = (E.T @ x2.T).T
    numerator = np.sum(x2 * Ex1, axis=1) ** 2
    denominator = (
        Ex1[:, 0] ** 2
        + Ex1[:, 1] ** 2
        + Etx2[:, 0] ** 2
        + Etx2[:, 1] ** 2
    )
    distances = np.sqrt(numerator / np.maximum(denominator, 1e-12))
    finite = distances[np.isfinite(distances)]
    return float(np.median(finite)) if finite.size else float("inf")


def association_cost(
    left: PersonPose,
    right: PersonPose,
    calibration: StereoCalibration,
    keypoint_threshold: float,
    minimum_common_keypoints: int = 4,
) -> tuple[float, int]:
    left_points, right_points, indices = _visible_pairs(
        left, right, keypoint_threshold
    )
    if len(indices) >= minimum_common_keypoints:
        return _epipolar_cost(left_points, right_points, calibration), len(indices)
    # A bbox-center fallback keeps the baseline usable when lower-body joints are
    # temporarily missing.  It is only a weak association cue and is therefore
    # accompanied by common_keypoints=0..3 in the output.
    return (
        _epipolar_cost(
            _bbox_center(left), _bbox_center(right), calibration
        ),
        len(indices),
    )


def associate_persons(
    left_persons: list[PersonPose],
    right_persons: list[PersonPose],
    calibration: StereoCalibration,
    keypoint_threshold: float,
    max_association_cost: float,
) -> list[PersonMatch]:
    if not left_persons or not right_persons:
        return []
    candidates: list[PersonMatch] = []
    for left_index, left in enumerate(left_persons):
        for right_index, right in enumerate(right_persons):
            cost, common = association_cost(
                left, right, calibration, keypoint_threshold
            )
            candidates.append(
                PersonMatch(left_index, right_index, cost, common)
            )

    # In the intended walker experiment there is normally one target person.
    # Do not discard that only pair merely because motion between unsynchronised
    # frames raises the epipolar residual; retain the measured cost for analysis.
    if len(left_persons) == 1 and len(right_persons) == 1:
        return candidates

    candidates.sort(key=lambda item: item.association_cost)
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[PersonMatch] = []
    for candidate in candidates:
        if candidate.association_cost > max_association_cost:
            continue
        if candidate.left_index in used_left or candidate.right_index in used_right:
            continue
        used_left.add(candidate.left_index)
        used_right.add(candidate.right_index)
        matches.append(candidate)
    return matches


def _invalid_keypoint(
    index: int,
    left_score: float,
    right_score: float,
    reason: str,
) -> dict[str, Any]:
    return {
        "index": index,
        "name": COCO17_NAMES[index],
        "valid": False,
        "xyz": None,
        "score": math.sqrt(max(0.0, left_score) * max(0.0, right_score)),
        "left_score": left_score,
        "right_score": right_score,
        "depth_left": None,
        "depth_right": None,
        "reprojection_error_left_px": None,
        "reprojection_error_right_px": None,
        "reprojection_error_mean_px": None,
        "reason": reason,
    }


def triangulate_person(
    left: PersonPose,
    right: PersonPose,
    calibration: StereoCalibration,
    keypoint_threshold: float,
    max_reprojection_error_px: float,
    stereo_person_id: int,
    association_cost_value: float,
    common_keypoints: int,
) -> TriangulatedPerson:
    output: list[dict[str, Any] | None] = [None] * 17
    valid_indices: list[int] = []
    left_pixels: list[list[float]] = []
    right_pixels: list[list[float]] = []

    for index, (left_point, right_point) in enumerate(
        zip(left.keypoints, right.keypoints)
    ):
        left_score = float(left_point[2])
        right_score = float(right_point[2])
        if left_score < keypoint_threshold or right_score < keypoint_threshold:
            output[index] = _invalid_keypoint(
                index, left_score, right_score, "low_2d_score"
            )
            continue
        values = [left_point[0], left_point[1], right_point[0], right_point[1]]
        if not all(math.isfinite(float(value)) for value in values):
            output[index] = _invalid_keypoint(
                index, left_score, right_score, "non_finite_2d_point"
            )
            continue
        valid_indices.append(index)
        left_pixels.append([float(left_point[0]), float(left_point[1])])
        right_pixels.append([float(right_point[0]), float(right_point[1])])

    if valid_indices:
        left_array = np.asarray(left_pixels, dtype=np.float64)
        right_array = np.asarray(right_pixels, dtype=np.float64)
        left_normalized = calibration.undistort_normalized(left_array, "left")
        right_normalized = calibration.undistort_normalized(right_array, "right")
        projection_left = np.concatenate(
            [np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)],
            axis=1,
        )
        projection_right = np.concatenate(
            [calibration.R, calibration.T.reshape(3, 1)], axis=1
        )
        homogeneous = cv2.triangulatePoints(
            projection_left,
            projection_right,
            left_normalized.T,
            right_normalized.T,
        )
        denominator = homogeneous[3]
        xyz = np.full((len(valid_indices), 3), np.nan, dtype=np.float64)
        good_w = np.abs(denominator) > 1e-12
        xyz[good_w] = (homogeneous[:3, good_w] / denominator[good_w]).T

        projected_left = calibration.project_left(xyz)
        projected_right = calibration.project_right(xyz)
        left_errors = np.linalg.norm(projected_left - left_array, axis=1)
        right_errors = np.linalg.norm(projected_right - right_array, axis=1)
        right_xyz = (calibration.R @ xyz.T + calibration.T.reshape(3, 1)).T

        for local_index, keypoint_index in enumerate(valid_indices):
            left_score = float(left.keypoints[keypoint_index][2])
            right_score = float(right.keypoints[keypoint_index][2])
            point = xyz[local_index]
            depth_left = float(point[2])
            depth_right = float(right_xyz[local_index, 2])
            left_error = float(left_errors[local_index])
            right_error = float(right_errors[local_index])
            mean_error = 0.5 * (left_error + right_error)
            finite_geometry = bool(
                np.all(np.isfinite(point))
                and math.isfinite(depth_left)
                and math.isfinite(depth_right)
                and math.isfinite(mean_error)
            )
            positive_depth = finite_geometry and depth_left > 0.0 and depth_right > 0.0
            reprojection_ok = positive_depth and mean_error <= max_reprojection_error_px
            if not finite_geometry:
                reason = "non_finite_triangulation"
            elif not positive_depth:
                reason = "negative_or_zero_depth"
            elif not reprojection_ok:
                reason = "high_reprojection_error"
            else:
                reason = None
            output[keypoint_index] = {
                "index": keypoint_index,
                "name": COCO17_NAMES[keypoint_index],
                "valid": bool(reprojection_ok),
                "xyz": [float(value) for value in point] if finite_geometry else None,
                "score": math.sqrt(max(0.0, left_score) * max(0.0, right_score)),
                "left_score": left_score,
                "right_score": right_score,
                "depth_left": depth_left if finite_geometry else None,
                "depth_right": depth_right if finite_geometry else None,
                "reprojection_error_left_px": left_error if finite_geometry else None,
                "reprojection_error_right_px": right_error if finite_geometry else None,
                "reprojection_error_mean_px": mean_error if finite_geometry else None,
                "reason": reason,
            }

    complete_output = [
        item
        if item is not None
        else _invalid_keypoint(index, 0.0, 0.0, "missing_keypoint")
        for index, item in enumerate(output)
    ]
    valid_errors = [
        float(item["reprojection_error_mean_px"])
        for item in complete_output
        if item["valid"] and item["reprojection_error_mean_px"] is not None
    ]
    return TriangulatedPerson(
        stereo_person_id=stereo_person_id,
        left_person_id=left.person_id,
        right_person_id=right.person_id,
        association_cost=float(association_cost_value),
        common_keypoints=int(common_keypoints),
        valid_keypoints=sum(bool(item["valid"]) for item in complete_output),
        mean_reprojection_error_px=(
            float(np.mean(valid_errors)) if valid_errors else None
        ),
        keypoints_3d=complete_output,
    )


def triangulate_matches(
    left_persons: list[PersonPose],
    right_persons: list[PersonPose],
    calibration: StereoCalibration,
    keypoint_threshold: float = 0.25,
    max_association_cost: float = 0.05,
    max_reprojection_error_px: float = 10.0,
) -> list[TriangulatedPerson]:
    matches = associate_persons(
        left_persons,
        right_persons,
        calibration,
        keypoint_threshold,
        max_association_cost,
    )
    return [
        triangulate_person(
            left_persons[match.left_index],
            right_persons[match.right_index],
            calibration,
            keypoint_threshold,
            max_reprojection_error_px,
            stereo_person_id=index,
            association_cost_value=match.association_cost,
            common_keypoints=match.common_keypoints,
        )
        for index, match in enumerate(matches)
    ]
