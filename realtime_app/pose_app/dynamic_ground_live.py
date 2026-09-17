"""Incremental fixed-ground output for the live stereo pipeline."""

from __future__ import annotations

from collections import Counter, deque
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import cv2

from .dynamic_ground_pose import (
    PlanarFootPoseCriteria,
    RotationFootPoseCriteria,
    solve_planar_translation_from_feet,
    solve_pose_from_relative_rotation_and_feet,
)
from .static_ground_reference import StaticGroundReference
from .two_stage_motion import WorldPose


STAGE2 = "stage2_feet_static_walker_moving"
STAGE1 = "stage1_walker_static_human_moving"
TRANSITION = "transition"


class RealtimeDynamicGroundWriter:
    """Causal O(1) Stage-2 planar translation tracker and JSONL writer."""

    def __init__(
        self, output_path: str | Path, ground_reference_path: str | Path, *, foot_window: int = 5,
        pose_mode: str = "planar_xy", foot_observation_mode: str = "causal_median",
        minimum_stage2_vote_confidence: float = 0.5, stage1_ground_settle: bool = False,
        maximum_stage2_vote_dropout_frames: int = 2, maximum_recovery_gap_frames: int = 4,
        maximum_stage2_transition_carry_frames: int = 4,
        allow_single_foot_recovery: bool = True, stage1_landed_support_reanchor: bool = False,
    ) -> None:
        if foot_window < 1:
            raise ValueError("foot_window must be positive")
        reference = StaticGroundReference.load(ground_reference_path)
        self.initial_pose = WorldPose(reference.rotation_ground_from_left, reference.translation_ground_from_left_mm)
        self.pose = self.initial_pose
        if foot_observation_mode not in {"current", "causal_median"}:
            raise ValueError("foot_observation_mode must be current or causal_median")
        if not 0.0 <= minimum_stage2_vote_confidence <= 1.0:
            raise ValueError("minimum_stage2_vote_confidence must be in [0, 1]")
        if maximum_stage2_vote_dropout_frames < 0:
            raise ValueError("maximum_stage2_vote_dropout_frames must be non-negative")
        if maximum_stage2_transition_carry_frames < 0:
            raise ValueError("maximum_stage2_transition_carry_frames must be non-negative")
        self.foot_observation_mode = foot_observation_mode
        self.minimum_stage2_vote_confidence = float(minimum_stage2_vote_confidence)
        self.maximum_stage2_vote_dropout_frames = int(maximum_stage2_vote_dropout_frames)
        self.maximum_stage2_transition_carry_frames = int(maximum_stage2_transition_carry_frames)
        self.allow_single_foot_recovery = bool(allow_single_foot_recovery)
        self.stage1_ground_settle = bool(stage1_ground_settle)
        self.stage1_landed_support_reanchor = bool(stage1_landed_support_reanchor)
        self.criteria = PlanarFootPoseCriteria(smoothing_gain=0.45, maximum_translation_step_mm=120.0)
        if pose_mode not in {"planar_xy", "full_se3"}:
            raise ValueError("pose_mode must be planar_xy or full_se3")
        self.pose_mode = pose_mode
        self.full_criteria = RotationFootPoseCriteria(
            smoothing_gain=0.45, maximum_translation_step_mm=120.0,
            maximum_two_foot_translation_disagreement_mm=60.0,
            maximum_recovery_gap_frames=maximum_recovery_gap_frames,
        )
        self.recent_feet: deque[dict[str, np.ndarray]] = deque(maxlen=foot_window)
        self.recent_anchor_feet_world: deque[dict[str, np.ndarray]] = deque(maxlen=foot_window)
        self.latest_stationary_feet_world: dict[str, np.ndarray] | None = None
        self.anchors: dict[str, np.ndarray] | None = None
        self.previous_stage: str | None = None
        self.ground_settling = False
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.output_path.open("w", encoding="utf-8", newline="\n")
        self.count = 0
        self.accepted_updates = 0
        self.stage_counts: Counter[str] = Counter()
        self.processing_ms: list[float] = []
        self.stage2_low_vote_streak = 0
        self.frames_since_pose_update = 0
        self.accepted_translation_history: deque[tuple[int, np.ndarray]] = deque(maxlen=2)
        self.stage2_pose_latched = False
        self.stage2_transition_carry_frames = 0
        self.previous_pose_active = False

    @staticmethod
    def _strict_points(payload: dict[str, Any]) -> dict[int, np.ndarray]:
        people = payload.get("persons_3d") or []
        if len(people) != 1:
            return {}
        result = {}
        for item in people[0].get("keypoints_3d", []):
            xyz = np.asarray(item.get("xyz", []), dtype=np.float64)
            if bool(item.get("valid")) and xyz.shape == (3,) and np.all(np.isfinite(xyz)):
                result[int(item["index"])] = xyz
        return result

    def _settle_ground(self, gain: float) -> None:
        relative = self.pose.rotation_world_from_camera @ self.initial_pose.rotation_world_from_camera.T
        yaw = float(np.arctan2(relative[1, 0], relative[0, 0]))
        c, s = np.cos(yaw), np.sin(yaw)
        target = np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))) @ self.initial_pose.rotation_world_from_camera
        vector, _ = cv2.Rodrigues(target @ self.pose.rotation_world_from_camera.T)
        delta, _ = cv2.Rodrigues(gain * vector)
        translation = self.pose.translation_world_from_camera_mm.copy()
        translation[2] += gain * (self.initial_pose.translation_world_from_camera_mm[2] - translation[2])
        self.pose = WorldPose(delta @ self.pose.rotation_world_from_camera, translation)

    def consume(
        self, payload: dict[str, Any], stage_status: dict[str, Any],
        rotation_status: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        stage_data = stage_status.get("stage") or {}
        stage = str(stage_data.get("operational") or stage_data.get("confirmed") or "unknown")
        if stage == STAGE2:
            pose_active = True
            self.stage2_pose_latched = True
            self.stage2_transition_carry_frames = 0
        elif (
            stage == TRANSITION
            and self.stage2_pose_latched
            and self.stage2_transition_carry_frames < self.maximum_stage2_transition_carry_frames
        ):
            self.stage2_transition_carry_frames += 1
            pose_active = True
        else:
            pose_active = False
            if stage != TRANSITION or self.stage2_transition_carry_frames >= self.maximum_stage2_transition_carry_frames:
                self.stage2_pose_latched = False
        points = self._strict_points(payload)
        feet = {"left": points[15], "right": points[16]} if 15 in points and 16 in points else None
        if not pose_active:
            self.stage2_low_vote_streak = 0
            self.frames_since_pose_update = 0
            self.accepted_translation_history.clear()
            if self.stage1_ground_settle and self.pose_mode == "full_se3" and self.previous_stage == STAGE2 and stage == TRANSITION:
                self.ground_settling = True
            if self.stage1_ground_settle and self.pose_mode == "full_se3" and self.ground_settling and stage == TRANSITION:
                self._settle_ground(0.35)
            if self.stage1_ground_settle and self.pose_mode == "full_se3" and stage == STAGE1 and self.previous_stage != STAGE1:
                self._settle_ground(1.0)
                self.ground_settling = False
            if (
                self.stage1_landed_support_reanchor and self.pose_mode == "full_se3"
                and stage == STAGE1 and self.previous_stage != STAGE1
            ):
                self._settle_ground(1.0)
                self.ground_settling = False
            self.anchors = None
            self.recent_feet.clear()
            if feet is not None:
                self.latest_stationary_feet_world = {name: self.pose.transform(point) for name, point in feet.items()}
                if self.pose_mode == "full_se3":
                    self.recent_anchor_feet_world.append(self.latest_stationary_feet_world)
            audit: dict[str, Any] = {"status": "held", "source": "stage1_or_transition_camera_pose_hold"}
            if (
                self.stage1_landed_support_reanchor and self.pose_mode == "full_se3"
                and stage == STAGE1 and self.previous_stage != STAGE1
            ):
                audit = {
                    "status": "landed_support_reanchored",
                    "source": "confirmed_stage1_rigid_no_wheel_support_height_roll_pitch",
                }
        else:
            self.ground_settling = False
            if not self.previous_pose_active:
                self.stage2_low_vote_streak = 0
                self.frames_since_pose_update = 0
                self.accepted_translation_history.clear()
                self.accepted_translation_history.append(
                    (int(payload["pair_id"]), self.pose.translation_world_from_camera_mm.copy())
                )
                self.recent_feet.clear()
                if self.pose_mode == "full_se3" and self.recent_anchor_feet_world:
                    self.anchors = {
                        name: np.median(
                            np.asarray([item[name] for item in self.recent_anchor_feet_world]), axis=0
                        ) for name in ("left", "right")
                    }
                    self.recent_anchor_feet_world.clear()
                else:
                    self.anchors = self.latest_stationary_feet_world
                if self.pose_mode != "full_se3" and self.anchors is None and feet is not None:
                    self.anchors = {name: self.pose.transform(point) for name, point in feet.items()}
            if feet is not None:
                self.recent_feet.append(feet)
            filtered = None
            if feet is not None and self.foot_observation_mode == "current":
                filtered = feet
            elif feet is not None and self.recent_feet:
                filtered = {
                    name: np.median(np.asarray([item[name] for item in self.recent_feet]), axis=0)
                    for name in ("left", "right")
                }
            stage_vote_confidence = (
                float(stage_data.get("vote_confidence") or 0.0) if stage == STAGE2 else 1.0
            )
            low_vote = stage == STAGE2 and stage_vote_confidence < self.minimum_stage2_vote_confidence
            if low_vote:
                self.stage2_low_vote_streak += 1
            else:
                self.stage2_low_vote_streak = 0
            vote_grace = bool(
                low_vote
                and self.previous_pose_active
                and self.stage2_low_vote_streak <= self.maximum_stage2_vote_dropout_frames
            )
            if feet is None or filtered is None or self.anchors is None:
                audit = {"status": "unavailable", "source": "pose_hold", "reason": "strict_two_foot_input_missing"}
            elif low_vote and not vote_grace:
                audit = {
                    "status": "rejected", "source": "pose_hold",
                    "reason": "stage2_vote_confidence_below_gate",
                    "stage2_vote_confidence": stage_vote_confidence,
                    "minimum_stage2_vote_confidence": self.minimum_stage2_vote_confidence,
                }
            elif self.pose_mode == "full_se3":
                accepted_rotation = rotation_status is not None and rotation_status.get("status") == "accepted"
                relative_rotation = (
                    np.asarray(rotation_status["R_to_from"], dtype=np.float64)
                    if accepted_rotation else np.eye(3, dtype=np.float64)
                )
                translation_prediction = None
                pair_id = int(payload["pair_id"])
                if len(self.accepted_translation_history) == 2:
                    first_pair, first_translation = self.accepted_translation_history[0]
                    last_pair, last_translation = self.accepted_translation_history[1]
                    interval = max(1, last_pair - first_pair)
                    velocity = (last_translation - first_translation) / interval
                    translation_prediction = last_translation + velocity * max(1, pair_id - last_pair)
                solved, audit = solve_pose_from_relative_rotation_and_feet(
                    self.pose, relative_rotation, filtered, self.anchors, self.full_criteria,
                    elapsed_frames_since_update=self.frames_since_pose_update + 1,
                    translation_prediction_world_mm=translation_prediction,
                    allow_single_foot_recovery=self.allow_single_foot_recovery,
                )
                audit["causal_foot_window_frames"] = len(self.recent_feet)
                audit["foot_observation_mode"] = self.foot_observation_mode
                audit["current_frame_two_feet_required"] = True
                audit["stage2_vote_confidence"] = stage_vote_confidence
                audit["stage2_vote_grace_applied"] = vote_grace
                audit["stage2_low_vote_streak"] = self.stage2_low_vote_streak
                audit["stage2_transition_carry_applied"] = stage == TRANSITION
                audit["stage2_transition_carry_frame"] = (
                    self.stage2_transition_carry_frames if stage == TRANSITION else 0
                )
                if solved is not None:
                    self.pose = solved
                    self.accepted_updates += 1
                    self.frames_since_pose_update = 0
                    self.accepted_translation_history.append(
                        (pair_id, self.pose.translation_world_from_camera_mm.copy())
                    )
                    if not accepted_rotation:
                        audit["status"] = "accepted_rotation_held"
                        audit["source"] = "two_stationary_feet_xyz_with_previous_rotation_held"
                        audit["rotation_hold_reason"] = None if rotation_status is None else rotation_status.get("reason")
            else:
                solved, audit = solve_planar_translation_from_feet(
                    self.initial_pose, self.pose, filtered, self.anchors, self.criteria
                )
                audit["causal_foot_window_frames"] = len(self.recent_feet)
                audit["foot_observation_mode"] = self.foot_observation_mode
                audit["current_frame_two_feet_required"] = True
                audit["stage2_vote_confidence"] = stage_vote_confidence
                if solved is not None:
                    self.pose = solved
                    self.accepted_updates += 1
                    self.frames_since_pose_update = 0
            if audit.get("status") in {"unavailable", "rejected"}:
                self.frames_since_pose_update += 1
        transformed = {str(index): self.pose.transform(point).tolist() for index, point in points.items()}
        constrained = {index: list(point) for index, point in transformed.items()}
        projection = {"applied": False, "type": None}
        if pose_active and self.anchors is not None:
            projected = []
            for name, index in (("left", 15), ("right", 16)):
                key = str(index)
                if key in constrained:
                    constrained[key] = self.anchors[name].tolist()
                    projected.append(index)
            if projected:
                projection = {
                    "applied": True,
                    "type": "stage2_stationary_ankle_constraint_for_output",
                    "projected_joint_indices": projected,
                    "raw_measurements_preserved": True,
                }
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        record = {
            "schema_version": "realtime_dynamic_ground_pose_v4", "pair_id": int(payload["pair_id"]),
            "stage": stage, **self.pose.as_mapping(), "pose_audit": audit,
            "rotation_audit": rotation_status,
            "strict_points_ground_mm": transformed, "processing_ms": elapsed_ms, "causal": True,
            "constrained_points_ground_mm": constrained,
            "constraint_projection": projection,
            "estimator_policy": {
                "foot_observation_mode": self.foot_observation_mode,
                "minimum_stage2_vote_confidence": self.minimum_stage2_vote_confidence,
                "maximum_stage2_vote_dropout_frames": self.maximum_stage2_vote_dropout_frames,
                "maximum_stage2_transition_carry_frames": self.maximum_stage2_transition_carry_frames,
                "maximum_recovery_gap_frames": self.full_criteria.maximum_recovery_gap_frames,
                "single_foot_recovery": self.allow_single_foot_recovery,
                "stage1_ground_settle": self.stage1_ground_settle,
                "stage1_landed_support_reanchor": self.stage1_landed_support_reanchor,
                "pose_smoothing_gain": self.full_criteria.smoothing_gain if self.pose_mode == "full_se3" else self.criteria.smoothing_gain,
            },
        }
        self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.handle.flush()
        self.count += 1
        self.stage_counts[stage] += 1
        self.processing_ms.append(elapsed_ms)
        self.previous_stage = stage
        self.previous_pose_active = pose_active
        return record

    def close(self, *, completed: bool) -> dict[str, Any]:
        if not self.handle.closed:
            self.handle.close()
        return {
            "completed": bool(completed), "record_count": self.count,
            "accepted_dynamic_pose_updates": self.accepted_updates, "stage_counts": dict(self.stage_counts),
            "mean_processing_ms": float(np.mean(self.processing_ms)) if self.processing_ms else None,
            "p95_processing_ms": float(np.percentile(self.processing_ms, 95)) if self.processing_ms else None,
            "output_jsonl": str(self.output_path.resolve()),
            "pose_model": (
                "stage_conditioned_stereo_rotation_plus_two_foot_xyz_full_se3"
                if self.pose_mode == "full_se3" else "fixed_ground_plus_stage2_two_foot_planar_translation"
            ),
        }
