"""Causal experimental stage handoff in a single fixed world frame.

No physical contact labels are inferred. Anchors are operational COCO ankles.
"""

import json
import time
from collections import deque
from dataclasses import replace

import numpy as np

from .dynamic_ground_live import STAGE2, RealtimeDynamicGroundWriter
from .dynamic_ground_pose import (
    RotationFootPoseCriteria,
    solve_pose_from_relative_rotation_and_feet,
)
from .static_ground_reference import StaticGroundReference
from .two_stage_motion import WorldPose


class ContinuousStageGroundWriter:
    def __init__(self, output, reference):
        ref = StaticGroundReference.load(reference)
        self.pose = WorldPose(
            ref.rotation_ground_from_left, ref.translation_ground_from_left_mm
        )
        self.file = open(output, "x", encoding="utf-8")  # noqa: SIM115 -- lifetime ends in close()
        self.previous_time = None
        self.last_update = None
        self.mode = "initializing"
        self.history = {s: deque(maxlen=5) for s in ("left", "right")}
        self.anchors = {}
        self.weight = 0.0
        self.velocity = np.zeros(3)
        self.criteria = RotationFootPoseCriteria()
        self.count = self.accepted = 0
        self.pending_rotation = np.eye(3)
        self.missing_rotation = 0
        self.frames_since_update = 0
        self.world_chain_complete = True

    def consume(self, payload, stage_status, rotation_status=None, visual_pose=None):
        started = time.perf_counter()
        timestamp = float(payload["pair_timestamp_sec"])
        dt = 0.0 if self.previous_time is None else timestamp - self.previous_time
        if self.previous_time is not None and dt <= 0:
            raise ValueError("timestamps must increase")
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        stage = stage_status["stage"]
        background = stage.get("background", {}).get("fused", {})
        moving = bool(background.get("available") and background.get("motion_active"))
        static = bool(
            background.get("available")
            and background.get("static_confident")
            and not moving
        )
        points = RealtimeDynamicGroundWriter._strict_points(payload)
        feet = {
            side: points[j] for side, j in [("left", 15), ("right", 16)] if j in points
        }
        old_pose = self.pose
        was_static = self.mode in ("walker_fixed", "settling")
        reason = None
        source = "bounded_prediction"
        status = "predicted"
        self.frames_since_update += 1
        if self.previous_time is None and not static:
            # The first measured pose needs no previous-to-current rotation edge.
            # This is initialization, not observed walker stationarity.
            source, status = "measured_static_reference_initialization", "held"
            self.last_update = timestamp
            self.frames_since_update = 0
        elif static:
            self.mode = "walker_fixed"
            self.weight = max(0.0, self.weight - dt / 0.15)
            self.pending_rotation = np.eye(3)
            self.missing_rotation = 0
            self.velocity *= 0.0
            for side, point in feet.items():
                self.history[side].append((timestamp, self.pose.transform(point)))
            source, status = "static_background_rigid_walker_hold", "held"
            self.last_update = timestamp
            self.frames_since_update = 0
        else:
            if was_static:
                self.anchors = {}
                for side, hist in self.history.items():
                    recent = [x[1] for x in hist if timestamp - x[0] <= 0.25]
                    if len(recent) >= 2:
                        center = np.median(recent, axis=0)
                        if (
                            np.max(np.linalg.norm(np.asarray(recent) - center, axis=1))
                            <= 20.0
                        ):
                            self.anchors[side] = center
                for hist in self.history.values():
                    hist.clear()
            self.mode = (
                "feet_supported" if stage.get("operational") == STAGE2 else "handoff"
            )
            candidate_support = (
                stage.get("candidate") == STAGE2 or stage.get("operational") == STAGE2
            )
            # Confidence in the rule ramps causally; this is not a calibrated probability.
            self.weight = (
                min(1.0, self.weight + dt / 0.15)
                if candidate_support
                else max(0.0, self.weight - dt / 0.25)
            )
            edge_valid = (
                rotation_status is not None
                and rotation_status.get("status") == "accepted"
            )
            if edge_valid:
                edge = np.asarray(rotation_status["R_to_from"], float)
                edge_valid = (
                    edge.shape == (3, 3)
                    and np.isfinite(edge).all()
                    and np.linalg.norm(edge.T @ edge - np.eye(3)) < 1e-3
                    and np.linalg.det(edge) > 0.999
                )
            if edge_valid:
                self.pending_rotation = edge @ self.pending_rotation
            else:
                self.missing_rotation += 1
                self.world_chain_complete = False
            if (
                visual_pose is not None
                and visual_pose.get("status") == "accepted"
                and visual_pose.get("metric_pose_usable") is True
                and self.frames_since_update == 1
            ):
                relative = np.asarray(visual_pose["R_to_from"])
                new_rotation = self.pose.rotation_world_from_camera @ relative.T
                new_translation = (
                    self.pose.translation_world_from_camera_mm
                    - new_rotation @ np.asarray(visual_pose["t_to_from_mm"])
                )
                self.pose = WorldPose(new_rotation, new_translation)
                status, source = "accepted", "independent_stereo_background"
                self.pending_rotation = np.eye(3)
                self.missing_rotation = 0
            elif self.missing_rotation:
                reason = "incomplete_rotation_chain"
            elif (
                self.weight > 0
                and set(feet) == {"left", "right"}
                and set(self.anchors) == {"left", "right"}
            ):
                solved, audit = solve_pose_from_relative_rotation_and_feet(
                    self.pose,
                    self.pending_rotation,
                    feet,
                    self.anchors,
                    replace(
                        self.criteria,
                        smoothing_gain=self.criteria.smoothing_gain * self.weight,
                    ),
                    elapsed_frames_since_update=max(1, self.frames_since_update),
                )
                if solved is not None:
                    self.pose = solved
                    self.pending_rotation = np.eye(3)
                    status, source = "accepted", "current_strict_two_ankles"
                else:
                    reason = audit.get("reason")
            elif (
                self.weight > 0
                and len(feet) == 1
                and self.missing_rotation == 0
                and self.last_update is not None
            ):
                side, point = next(iter(feet.items()))
                if side in self.anchors and timestamp - self.last_update <= 0.2:
                    rot = self.pose.rotation_world_from_camera @ self.pending_rotation.T
                    proposed = self.anchors[side] - rot @ point
                    prediction = (
                        self.pose.translation_world_from_camera_mm + self.velocity * dt
                    )
                    angle = np.degrees(
                        np.arccos(
                            np.clip((np.trace(self.pending_rotation) - 1) / 2, -1, 1)
                        )
                    )
                    if (
                        np.linalg.norm(proposed - prediction) <= 80.0
                        and angle <= 5.0
                        and np.linalg.norm(
                            proposed - self.pose.translation_world_from_camera_mm
                        )
                        <= 120.0
                    ):
                        self.pose = WorldPose(
                            rot,
                            self.pose.translation_world_from_camera_mm
                            + 0.45
                            * self.weight
                            * (proposed - self.pose.translation_world_from_camera_mm),
                        )
                        self.pending_rotation = np.eye(3)
                        status, source = (
                            "accepted",
                            "single_ankle_with_rotation_and_motion_gate",
                        )
                    else:
                        reason = "single_ankle_prediction_disagreement"
            else:
                reason = "support_or_strict_ankles_unavailable"
            if status == "accepted":
                self.velocity = (
                    self.pose.translation_world_from_camera_mm
                    - old_pose.translation_world_from_camera_mm
                ) / max(dt, 1e-6)
                speed = np.linalg.norm(self.velocity)
                if speed > 600.0:
                    self.velocity *= 600.0 / speed
                self.last_update = timestamp
                self.frames_since_update = 0
                self.accepted += 1
            elif self.last_update is not None and timestamp - self.last_update <= 0.2:
                self.pose = WorldPose(
                    self.pose.rotation_world_from_camera,
                    self.pose.translation_world_from_camera_mm + self.velocity * dt,
                )
            else:
                status, source = "unavailable", "prediction_horizon_exhausted"
        targets = {
            side: self.anchors[side].tolist() for side in feet if side in self.anchors
        }
        row = {
            "schema_version": "continuous_stage_ground_candidate_v1",
            "pair_id": payload["pair_id"],
            "stage": stage.get("operational"),
            "pair_timestamp_sec": timestamp,
            "length_unit": "millimeter",
            "rotation_world_from_left_camera": self.pose.rotation_world_from_camera.tolist(),
            "translation_world_from_left_camera_mm": self.pose.translation_world_from_camera_mm.tolist(),
            "pose_audit": {
                "status": status,
                "source": source,
                "reason": reason,
                "mode": self.mode,
                "missing_rotation_edges": self.missing_rotation,
                "support_weight": self.weight,
                "estimate_available": status != "unavailable",
                "world_rotation_chain_complete": self.world_chain_complete,
                "absolute_world_estimate_available": status != "unavailable"
                and self.world_chain_complete,
            },
            "rotation_audit": rotation_status,
            "strict_points_ground_mm": {
                str(j): self.pose.transform(x).tolist() for j, x in points.items()
            },
            "ankle_anchor_world_mm": targets,
            "ankle_constraint_weight": self.weight
            if status == "accepted" and self.world_chain_complete
            else 0.0,
            "visual_pose_audit": visual_pose,
            "causal": True,
            "processing_ms": (time.perf_counter() - started) * 1000,
        }
        self.file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.previous_time = timestamp
        self.count += 1
        return row

    def close(self, *, completed):
        self.file.close()
        return {
            "completed": completed,
            "record_count": self.count,
            "accepted_updates": self.accepted,
            "evidence": "operational ankle anchors; no measured physical contact",
        }
