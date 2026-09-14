"""Static ChArUco floor reference for a walker-mounted stereo rig.

The reference is valid only while the walker and both cameras keep the same
rigid pose as during the board capture.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np


SCHEMA = "static_charuco_ground_reference_v1"


def make_charuco_board(board_spec: Mapping[str, Any]) -> cv2.aruco.CharucoBoard:
    dictionary_name = str(board_spec.get("dictionary", "DICT_4X4_50"))
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"unsupported ArUco dictionary: {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    return cv2.aruco.CharucoBoard(
        (int(board_spec["squares_x"]), int(board_spec["squares_y"])),
        float(board_spec["square_mm"]),
        float(board_spec["marker_mm"]),
        dictionary,
    )


def detect_charuco(
    image: np.ndarray,
    board: cv2.aruco.CharucoBoard,
) -> tuple[np.ndarray, np.ndarray]:
    detector = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = detector.detectBoard(image)
    if corners is None or ids is None:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.int32)
    return (
        np.asarray(corners, dtype=np.float64).reshape(-1, 2),
        np.asarray(ids, dtype=np.int32).reshape(-1),
    )


def _project_camera(
    xyz_camera_mm: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    camera_model: str,
) -> np.ndarray:
    points = np.asarray(xyz_camera_mm, dtype=np.float64).reshape(-1, 1, 3)
    zero = np.zeros((3, 1), dtype=np.float64)
    if camera_model == "fisheye":
        projected, _ = cv2.fisheye.projectPoints(points, zero, zero, K, D.reshape(-1, 1))
    else:
        projected, _ = cv2.projectPoints(points, zero, zero, K, D)
    return projected.reshape(-1, 2)


def estimate_board_pose(
    corners_px: np.ndarray,
    ids: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    K: np.ndarray,
    D: np.ndarray,
    camera_model: str,
) -> dict[str, Any]:
    corners = np.asarray(corners_px, dtype=np.float64).reshape(-1, 2)
    ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    if len(corners) != len(ids):
        raise ValueError("ChArUco corner and ID counts differ")
    if len(ids) < 4:
        raise ValueError("at least four ChArUco corners are required")
    board_points = np.asarray(board.getChessboardCorners(), dtype=np.float64).reshape(-1, 3)
    if int(ids.min()) < 0 or int(ids.max()) >= len(board_points):
        raise ValueError("ChArUco ID is outside the board definition")
    object_points = board_points[ids]
    image_points = corners.reshape(-1, 1, 2)
    if camera_model == "fisheye":
        normalized = cv2.fisheye.undistortPoints(image_points, K, D.reshape(-1, 1))
    else:
        normalized = cv2.undistortPoints(image_points, K, D)
    ok, rvec, tvec = cv2.solvePnP(
        object_points.reshape(-1, 1, 3),
        normalized,
        np.eye(3, dtype=np.float64),
        np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise ValueError("solvePnP failed for ChArUco board")
    rotation, _ = cv2.Rodrigues(rvec)
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    xyz_camera = (rotation @ object_points.T).T + translation
    projected = _project_camera(xyz_camera, K, D, camera_model)
    errors = np.linalg.norm(projected - corners, axis=1)
    return {
        "corner_count": int(len(ids)),
        "ids": ids.astype(int).tolist(),
        "rotation_camera_from_board": rotation.tolist(),
        "translation_camera_from_board_mm": translation.tolist(),
        "reprojection_rmse_px": float(np.sqrt(np.mean(errors**2))),
        "reprojection_median_px": float(np.median(errors)),
        "reprojection_max_px": float(np.max(errors)),
    }


def right_pose_in_left_frame(
    rotation_right_from_board: np.ndarray,
    translation_right_from_board_mm: np.ndarray,
    rotation_right_from_left: np.ndarray,
    translation_right_from_left_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation_lr = np.asarray(rotation_right_from_left, dtype=np.float64).reshape(3, 3)
    translation_lr = np.asarray(translation_right_from_left_mm, dtype=np.float64).reshape(3)
    rotation_rb = np.asarray(rotation_right_from_board, dtype=np.float64).reshape(3, 3)
    translation_rb = np.asarray(translation_right_from_board_mm, dtype=np.float64).reshape(3)
    rotation_lb = rotation_lr.T @ rotation_rb
    translation_lb = rotation_lr.T @ (translation_rb - translation_lr)
    return rotation_lb, translation_lb


def rotation_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left).reshape(3, 3).T @ np.asarray(right).reshape(3, 3)
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def average_rotations(rotations: Iterable[np.ndarray]) -> np.ndarray:
    values = [np.asarray(rotation, dtype=np.float64).reshape(3, 3) for rotation in rotations]
    if not values:
        raise ValueError("no rotations to average")
    u, _, vt = np.linalg.svd(np.sum(values, axis=0))
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


@dataclass(frozen=True)
class StaticGroundReference:
    rotation_ground_from_left: np.ndarray
    translation_ground_from_left_mm: np.ndarray
    plane_normal_left: np.ndarray
    plane_offset_left_mm: float
    source_capture_session: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StaticGroundReference":
        if value.get("schema_version") != SCHEMA:
            raise ValueError(f"ground reference schema must be {SCHEMA}")
        if value.get("status") != "measured_static_reference":
            raise ValueError("ground reference must have status='measured_static_reference'")
        rotation = np.asarray(value["rotation_ground_from_left"], dtype=np.float64)
        translation = np.asarray(value["translation_ground_from_left_mm"], dtype=np.float64)
        normal = np.asarray(value["plane_left_camera"]["normal"], dtype=np.float64)
        offset = float(value["plane_left_camera"]["offset_mm"])
        if rotation.shape != (3, 3) or translation.shape != (3,) or normal.shape != (3,):
            raise ValueError("invalid static ground reference shapes")
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            raise ValueError("static ground reference contains non-finite values")
        if abs(float(np.linalg.det(rotation)) - 1.0) > 1e-5:
            raise ValueError("rotation_ground_from_left is not a proper rotation")
        normal = normal / np.linalg.norm(normal)
        return cls(rotation, translation, normal, offset, str(value["source_capture_session"]))

    @classmethod
    def load(cls, path: str | Path) -> "StaticGroundReference":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def transform(self, xyz_left_mm: Any) -> np.ndarray:
        point = np.asarray(xyz_left_mm, dtype=np.float64).reshape(3)
        return self.rotation_ground_from_left @ point + self.translation_ground_from_left_mm

    def signed_height_mm(self, xyz_left_mm: Any) -> float:
        point = np.asarray(xyz_left_mm, dtype=np.float64).reshape(3)
        return float(self.plane_normal_left @ point + self.plane_offset_left_mm)


def build_reference(
    accepted_left_poses: list[tuple[np.ndarray, np.ndarray]],
    *,
    source_capture_session: str,
    per_pair_results: list[dict[str, Any]],
    board_spec: Mapping[str, Any],
    board_thickness_mm: float,
) -> dict[str, Any]:
    if not accepted_left_poses:
        raise ValueError("no accepted board poses")
    if not math.isfinite(board_thickness_mm) or board_thickness_mm < 0:
        raise ValueError("board_thickness_mm must be finite and non-negative")
    rotations = [pose[0] for pose in accepted_left_poses]
    translations = np.asarray([pose[1] for pose in accepted_left_poses], dtype=np.float64)
    rotation_left_from_ground = average_rotations(rotations)
    origin_left = np.median(translations, axis=0)
    normal_left = rotation_left_from_ground[:, 2]
    if float(-normal_left @ origin_left) < 0:
        # Printed board face must point toward the camera. Flip both in-plane Y and Z
        # axes to keep a proper rotation while making ground height positive.
        rotation_left_from_ground[:, 1] *= -1.0
        rotation_left_from_ground[:, 2] *= -1.0
        normal_left = rotation_left_from_ground[:, 2]
    board_top_offset = float(-normal_left @ origin_left)
    # The detected ChArUco corners lie on the printed top surface.  With the
    # printed face upward, the physical floor is board_thickness_mm farther
    # from the camera along -normal_left.
    floor_origin_left = origin_left - float(board_thickness_mm) * normal_left
    offset = float(-normal_left @ floor_origin_left)
    rotation_ground_from_left = rotation_left_from_ground.T
    translation_ground_from_left = -rotation_ground_from_left @ floor_origin_left
    origin_spread = np.linalg.norm(translations - origin_left, axis=1)
    # A plane normal has a sign ambiguity: n and -n describe the same plane.
    # The final ground frame may also flip the board Y/Z axes to make +Z point
    # upward.  Comparing that flipped full rotation with the original PnP
    # rotations therefore produces an artificial value near 180 degrees even
    # for a perfectly stationary board.  Repeatability here concerns the floor
    # plane, so compare the per-frame Z axes sign-invariantly.
    normal_spread = []
    for rotation in rotations:
        frame_normal = np.asarray(rotation, dtype=np.float64).reshape(3, 3)[:, 2]
        cosine = float(np.clip(abs(normal_left @ frame_normal), 0.0, 1.0))
        normal_spread.append(math.degrees(math.acos(cosine)))
    return {
        "schema_version": SCHEMA,
        "status": "measured_static_reference",
        "source_capture_session": source_capture_session,
        "valid_only_while": "walker and both cameras remain rigidly stationary after board capture",
        "board": dict(board_spec),
        "board_thickness_mm": float(board_thickness_mm),
        "accepted_pair_count": len(accepted_left_poses),
        "rotation_ground_from_left": rotation_ground_from_left.tolist(),
        "translation_ground_from_left_mm": translation_ground_from_left.tolist(),
        "plane_left_camera": {
            "normal": normal_left.tolist(),
            "offset_mm": offset,
            "camera_height_mm": offset,
            "equation": "normal dot X_left_mm + offset_mm = 0",
        },
        "detected_board_top_plane_left_camera": {
            "normal": normal_left.tolist(),
            "offset_mm": board_top_offset,
            "floor_shift_from_board_top_mm": float(board_thickness_mm),
        },
        "repeatability": {
            "origin_spread_median_mm": float(np.median(origin_spread)),
            "origin_spread_p95_mm": float(np.percentile(origin_spread, 95)),
            "normal_angle_median_deg": float(np.median(normal_spread)),
            "normal_angle_p95_deg": float(np.percentile(normal_spread, 95)),
        },
        "per_pair": per_pair_results,
        "interpretation": (
            "The board plane defines z=0 for one stationary walker/camera setup. "
            "It must not be reused after the walker or either camera moves."
        ),
    }
