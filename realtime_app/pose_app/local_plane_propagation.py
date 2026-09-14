"""Local plane propagation contracts and geometry.

This module deliberately has no global/world frame.  A plane is expressed in
the *current left-camera frame* as ``n.T @ X + d = 0``.  It is only allowed to
be propagated through a relative pose obtained from static background 3-D
features that excluded the person, walker and the previously fitted ground
region.  The separation prevents the invalid loop of using a fitted floor to
both estimate and validate the same floor.

The public JSONL contracts are intentionally small so that the ground
observation route and a future IMU route can produce their own inputs:

direct-plane input record (``local_ground_state_v1``)::

  {"frame_index": 12, "frame_id": "pair_0012.png",
   "observation_state": "direct",
   "plane_in_left_camera": {"normal_toward_camera_unit": [nx, ny, nz],
                              "offset_mm": d}}

relative-pose input record::

  {"from_frame_index": 12, "to_frame_index": 13, "status": "accepted",
   "R_to_from": [[...], [...], [...]], "q_to_from": [qx, qy, qz],
   "feature_domain": "static_background_excluding_person_walker_ground",
   "static_3d_correspondence_count": 42, "ransac_inlier_count": 35,
   "ground_region_used_for_motion": false}

The pose convention is ``X_to = R_to_from @ X_from + q_to_from``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np


DIRECT = "direct"
PROPAGATED = "propagated"
UNAVAILABLE = "unavailable"
STATIC_BACKGROUND_DOMAIN = "static_background_excluding_person_walker_ground"


@dataclass(frozen=True)
class LocalPlane:
    normal: np.ndarray
    offset: float

    def __post_init__(self) -> None:
        normal = np.asarray(self.normal, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(normal)) or not math.isfinite(float(self.offset)):
            raise ValueError("plane must contain finite normal and offset")
        magnitude = float(np.linalg.norm(normal))
        if magnitude <= 1e-9:
            raise ValueError("plane normal must be non-zero")
        offset = float(self.offset) / magnitude
        normal = normal / magnitude
        # Canonicalise so the left-camera origin is on the positive normal
        # side.  This makes normal-angle comparisons sign-safe.
        if offset < 0.0:
            normal = -normal
            offset = -offset
        if offset <= 1e-9:
            raise ValueError("plane through camera origin is not a usable local ground plane")
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "offset", offset)

    def to_json(self) -> dict[str, Any]:
        return {
            "plane_in_left_camera": {
                "normal_toward_camera_unit": [float(value) for value in self.normal],
                "offset_mm": float(self.offset),
            },
        }


@dataclass(frozen=True)
class RelativePose:
    from_frame_index: int
    to_frame_index: int
    rotation: np.ndarray
    translation: np.ndarray
    static_3d_correspondence_count: int
    ransac_inlier_count: int

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)
        translation = np.asarray(self.translation, dtype=np.float64).reshape(3)
        if self.to_frame_index != self.from_frame_index + 1:
            raise ValueError("relative poses must connect adjacent ordered frames")
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            raise ValueError("relative pose must be finite")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
            raise ValueError("relative pose rotation is not orthonormal")
        if not np.isclose(float(np.linalg.det(rotation)), 1.0, atol=1e-4):
            raise ValueError("relative pose rotation determinant is not +1")
        if self.static_3d_correspondence_count < 3 or self.ransac_inlier_count < 3:
            raise ValueError("relative pose needs at least three static 3-D correspondences/inliers")
        if self.ransac_inlier_count > self.static_3d_correspondence_count:
            raise ValueError("RANSAC inlier count cannot exceed correspondence count")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)


def parse_direct_plane(record: dict[str, Any]) -> LocalPlane | None:
    """Parse a direct plane; non-direct records explicitly provide no anchor."""
    state = record.get("observation_state", record.get("status"))
    if state != DIRECT:
        return None
    nested = record.get("plane_in_left_camera")
    if isinstance(nested, dict):
        return LocalPlane(nested["normal_toward_camera_unit"], float(nested["offset_mm"]))
    # Temporary compatibility with an early private fixture; new producers
    # must use the local_ground_state_v1 nested form above.
    return LocalPlane(record["normal_left_camera"], float(record["offset_left_camera"]))


def parse_relative_pose(record: dict[str, Any]) -> RelativePose | None:
    """Fail closed unless the relative pose documents independent static support."""
    if record.get("status") != "accepted":
        return None
    if record.get("feature_domain") != STATIC_BACKGROUND_DOMAIN:
        return None
    if record.get("ground_region_used_for_motion") is not False:
        return None
    return RelativePose(
        from_frame_index=int(record["from_frame_index"]),
        to_frame_index=int(record["to_frame_index"]),
        rotation=record["R_to_from"],
        translation=record["q_to_from"],
        static_3d_correspondence_count=int(record["static_3d_correspondence_count"]),
        ransac_inlier_count=int(record["ransac_inlier_count"]),
    )


def propagate_plane(plane: LocalPlane, pose: RelativePose) -> LocalPlane:
    """Return the same physical plane in the next left-camera frame."""
    normal = pose.rotation @ plane.normal
    offset = plane.offset - float(normal @ pose.translation)
    return LocalPlane(normal, offset)


def compose_relative_poses(poses: Iterable[RelativePose]) -> tuple[np.ndarray, np.ndarray]:
    """Compose adjacent poses into ``X_end = R @ X_start + q``."""
    rotation = np.eye(3, dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    for pose in poses:
        rotation = pose.rotation @ rotation
        translation = pose.rotation @ translation + pose.translation
    return rotation, translation


def plane_agreement(predicted: LocalPlane, reference: LocalPlane) -> dict[str, float]:
    cosine = float(np.clip(predicted.normal @ reference.normal, -1.0, 1.0))
    return {
        "normal_angle_deg": float(np.degrees(np.arccos(cosine))),
        "camera_plane_distance_delta": float(abs(predicted.offset - reference.offset)),
    }


def sequential_propagation(
    direct_records: list[dict[str, Any]],
    relative_by_from_index: dict[int, RelativePose],
    max_propagation_frames: int,
) -> list[dict[str, Any]]:
    """State machine for direct, propagated and unavailable local planes."""
    if max_propagation_frames < 1:
        raise ValueError("max_propagation_frames must be positive")
    ordered = sorted(direct_records, key=lambda item: int(item["frame_index"]))
    if len({int(item["frame_index"]) for item in ordered}) != len(ordered):
        raise ValueError("direct plane input has duplicate frame_index")

    rows: list[dict[str, Any]] = []
    prior_plane: LocalPlane | None = None
    prior_index: int | None = None
    anchor_index: int | None = None
    propagation_age: int | None = None
    for record in ordered:
        index = int(record["frame_index"])
        frame_id = str(record.get("frame_id", index))
        plane = parse_direct_plane(record)
        if plane is not None:
            row = {"local_ground_state_version": "local_ground_state_v1", "frame_index": index, "frame_id": frame_id,
                   "observation_state": DIRECT, "source": "independent_direct_plane_input", "reason": None,
                   "quality": {"anchor_frame_index": index, "propagation_age_frames": 0}, **plane.to_json()}
            prior_plane, prior_index, anchor_index, propagation_age = plane, index, index, 0
        elif prior_plane is None or prior_index is None or anchor_index is None or propagation_age is None:
            row = {"local_ground_state_version": "local_ground_state_v1", "frame_index": index, "frame_id": frame_id,
                   "observation_state": UNAVAILABLE, "source": "none", "reason": "no_prior_direct_anchor",
                   "quality": {"anchor_frame_index": None, "propagation_age_frames": None}, "plane_in_left_camera": None}
            prior_plane, prior_index, propagation_age = None, index, None
        elif index != prior_index + 1:
            row = {"local_ground_state_version": "local_ground_state_v1", "frame_index": index, "frame_id": frame_id,
                   "observation_state": UNAVAILABLE, "source": "none", "reason": "non_adjacent_frame_input",
                   "quality": {"anchor_frame_index": anchor_index, "propagation_age_frames": None}, "plane_in_left_camera": None}
            prior_plane, prior_index, propagation_age = None, index, None
        elif propagation_age >= max_propagation_frames:
            row = {"local_ground_state_version": "local_ground_state_v1", "frame_index": index, "frame_id": frame_id,
                   "observation_state": UNAVAILABLE, "source": "none", "reason": "maximum_propagation_age_exceeded",
                   "quality": {"anchor_frame_index": anchor_index, "propagation_age_frames": propagation_age}, "plane_in_left_camera": None}
            prior_plane, prior_index, propagation_age = None, index, None
        else:
            pose = relative_by_from_index.get(prior_index)
            if pose is None:
                row = {"local_ground_state_version": "local_ground_state_v1", "frame_index": index, "frame_id": frame_id,
                       "observation_state": UNAVAILABLE, "source": "none", "reason": "missing_or_rejected_static_background_relative_pose",
                       "quality": {"anchor_frame_index": anchor_index, "propagation_age_frames": propagation_age}, "plane_in_left_camera": None}
                prior_plane, prior_index, propagation_age = None, index, None
            else:
                propagated = propagate_plane(prior_plane, pose)
                propagation_age += 1
                row = {"local_ground_state_version": "local_ground_state_v1", "frame_index": index, "frame_id": frame_id,
                       "observation_state": PROPAGATED, "source": "static_background_vo_propagation", "reason": None,
                       "quality": {"anchor_frame_index": anchor_index, "propagation_age_frames": propagation_age,
                                   "static_3d_correspondence_count": pose.static_3d_correspondence_count,
                                   "ransac_inlier_count": pose.ransac_inlier_count}, **propagated.to_json()}
                prior_plane, prior_index = propagated, index
        rows.append(row)
    return rows
