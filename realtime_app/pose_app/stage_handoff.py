"""Experimental camera-regime hysteresis and explicit transition actions.

Stage2 foot stationarity is the controlled capture protocol, not an inferred
physical contact label. The original strict geometry gates remain mandatory.
"""

import copy
import json
from pathlib import Path

import numpy as np

from .anchored_ankle_translation import solve_ankle_bearing_translation
from .dynamic_ground_live import STAGE1, STAGE2, TRANSITION, RealtimeDynamicGroundWriter
from .landing_pose_recovery import LandingPoseRecovery
from .two_stage_motion import WorldPose


class StageHandoffPolicy:
    def __init__(self):
        self.regime = None
        self.moving_streak = self.static_streak = 0

    def update(self, original):
        row = copy.deepcopy(original)
        stage = row["stage"]
        bg = stage.get("background", {}).get("fused", {})
        available = bool(bg.get("available"))
        reliable = [
            b
            for s, b in stage.get("background", {}).items()
            if s != "fused" and b.get("available") and b.get("inlier_ratio", 0) >= 0.35
        ]
        instantaneous_stop = bool(reliable) and all(
            b.get("median_motion_px", float("inf")) <= 0.2 for b in reliable
        )
        static = available and (bool(bg.get("static_confident")) or instantaneous_stop)
        self.static_streak = self.static_streak + 1 if static else 0
        moving = available and bool(bg.get("motion_active")) and not instantaneous_stop
        self.moving_streak = self.moving_streak + 1 if moving else 0
        if self.static_streak >= 2:
            self.regime = STAGE1
            confirmed = STAGE1
            action = "walker_pose_hold"
            ground_stage = STAGE1
        elif self.moving_streak >= 2:
            self.regime = STAGE2
            confirmed = STAGE2
            action = "strict_foot_pose_update"
            ground_stage = STAGE2
        elif moving:
            confirmed = TRANSITION
            action = "motion_onset_strict_foot_pose_update"
            ground_stage = STAGE2
        elif static:
            confirmed = STAGE1 if self.regime == STAGE1 else TRANSITION
            action = "stop_confirmation_pose_hold"
            ground_stage = STAGE1
        else:
            confirmed = TRANSITION if self.regime else "warming_up"
            # A short uncertain edge after confirmed motion retains the original
            # four-frame carry and all strict point/rotation gates.
            action = (
                "bounded_support_carry"
                if self.regime == STAGE2
                else "uncertain_pose_hold"
            )
            ground_stage = TRANSITION
        stage["legacy_operational"] = stage["operational"]
        stage["legacy_confirmed"] = stage.get("confirmed")
        stage["confirmed"] = confirmed
        stage["operational"] = confirmed
        stage["handoff"] = {
            "action": action,
            "ground_stage": ground_stage,
            "moving_streak": self.moving_streak,
            "static_streak": self.static_streak,
            "instantaneous_stop": instantaneous_stop,
            "feet_stationarity_source": "controlled_capture_protocol_not_contact_truth",
            "classification_basis": "background_camera_motion; body backward displacement is not a vote",
        }
        return row


class StageHandoffGroundWriter:
    def __init__(
        self,
        output,
        reference,
        *,
        foot_observation_mode="causal_median",
        reanchor=True,
        prediction=False,
        calibration=None,
        bearing_fallback=False,
        single_ankle_fallback=False,
        latest_anchor=False,
        recovery_mode="none",
        landing_protocol=None,
    ):
        output = Path(output)
        self.estimator = RealtimeDynamicGroundWriter(
            output.with_name("ground_measurement_updates.jsonl"),
            reference,
            pose_mode="full_se3",
            foot_observation_mode=foot_observation_mode,
            stage1_landed_support_reanchor=reanchor and recovery_mode == "none",
        )
        self.recovery = (
            LandingPoseRecovery(
                self.estimator.initial_pose,
                mode=recovery_mode,
                protocol=landing_protocol,
            )
            if recovery_mode != "none"
            else None
        )
        self.file = output.open("x", encoding="utf-8")
        self.prediction = prediction
        self.calibration = calibration
        self.bearing_fallback = bearing_fallback
        self.single_ankle_fallback = single_ankle_fallback
        self.latest_anchor = latest_anchor
        self.last_time = self.previous_time = None
        self.velocity = np.zeros(3)
        self.last_measured_translation = None

    def consume(
        self, payload, stage_status, rotation_status=None, ankle_observations=None
    ):
        proxy = copy.deepcopy(stage_status)
        handoff = proxy["stage"]["handoff"]
        proxy["stage"]["operational"] = handoff["ground_stage"]
        if handoff["ground_stage"] == STAGE2:
            # The action is triggered by camera-motion evidence, not legacy votes.
            proxy["stage"]["vote_confidence"] = 1.0
        time = float(payload["pair_timestamp_sec"])
        dt = 0.0 if self.previous_time is None else time - self.previous_time
        if not np.isfinite(time) or (self.previous_time is not None and dt <= 0):
            raise ValueError("timestamps must increase")
        if (
            self.latest_anchor
            and handoff["ground_stage"] == STAGE2
            and not self.estimator.previous_pose_active
        ):
            self.estimator.recent_anchor_feet_world.clear()
        row = self.estimator.consume(payload, proxy, rotation_status)
        measured = row["pose_audit"]["status"] in ("accepted", "accepted_rotation_held")
        points = self.estimator._strict_points(payload)
        feet = {s: points[j] for s, j in [("left", 15), ("right", 16)] if j in points}
        if (
            not measured
            and self.single_ankle_fallback
            and len(feet) == 1
            and self.estimator.anchors is not None
            and handoff["ground_stage"] != STAGE1
            and self.estimator.pending_rotation_missing == 0
        ):
            foot, point = next(iter(feet.items()))
            rotation = (
                self.estimator.pose.rotation_world_from_camera
                @ self.estimator.pending_rotation.T
            )
            translation = self.estimator.anchors[foot] - rotation @ point
            prediction = (
                self.estimator.pose.translation_world_from_camera_mm
                + self.velocity * dt
            )
            error = float(np.linalg.norm(translation - prediction))
            angle = np.degrees(
                np.arccos(
                    np.clip((np.trace(self.estimator.pending_rotation) - 1) / 2, -1, 1)
                )
            )
            row["pose_audit"]["single_ankle_audit"] = {
                "foot": foot,
                "prediction_error_mm": error,
                "rotation_step_deg": float(angle),
                "strict_current_point_required": True,
            }
            if (
                error <= 80
                and angle <= 5
                and np.linalg.norm(
                    translation - self.estimator.pose.translation_world_from_camera_mm
                )
                <= 120
            ):
                pose = WorldPose(
                    rotation,
                    self.estimator.pose.translation_world_from_camera_mm
                    + 0.45
                    * (
                        translation
                        - self.estimator.pose.translation_world_from_camera_mm
                    ),
                )
                self._commit_fallback(row, pose, "accepted_single_ankle", payload)
                measured = True
        if (
            not measured
            and self.bearing_fallback
            and ankle_observations is not None
            and handoff["ground_stage"] != STAGE1
            and self.estimator.anchors is not None
            and self.estimator.pending_rotation_missing == 0
        ):
            rotation = (
                self.estimator.pose.rotation_world_from_camera
                @ self.estimator.pending_rotation.T
            )
            translation, audit = solve_ankle_bearing_translation(
                self.calibration, rotation, self.estimator.anchors, ankle_observations
            )
            angle = np.degrees(
                np.arccos(
                    np.clip((np.trace(self.estimator.pending_rotation) - 1) / 2, -1, 1)
                )
            )
            limit = 120 * min(4, max(1, self.estimator.frames_since_pose_update))
            if (
                translation is not None
                and angle <= 5
                and np.linalg.norm(
                    translation - self.estimator.pose.translation_world_from_camera_mm
                )
                <= limit
            ):
                self.estimator.pose = WorldPose(
                    rotation,
                    self.estimator.pose.translation_world_from_camera_mm
                    + 0.45
                    * (
                        translation
                        - self.estimator.pose.translation_world_from_camera_mm
                    ),
                )
                self.estimator.pending_rotation = np.eye(3)
                self.estimator.pending_rotation_edges = 0
                self.estimator.pending_rotation_missing = 0
                self.estimator.frames_since_pose_update = 0
                self.estimator.accepted_updates += 1
                self.estimator.accepted_translation_history.append(
                    (
                        int(payload["pair_id"]),
                        self.estimator.pose.translation_world_from_camera_mm.copy(),
                    )
                )
                row.update(self.estimator.pose.as_mapping())
                row["pose_audit"]["measurement_status"] = row["pose_audit"]["status"]
                row["pose_audit"]["measurement_reason"] = row["pose_audit"].get(
                    "reason"
                )
                row["pose_audit"]["reason"] = None
                row["pose_audit"]["status"] = "accepted_ankle_bearings"
                measured = True
                row["strict_points_ground_mm"] = {
                    str(j): self.estimator.pose.transform(p).tolist()
                    for j, p in self.estimator._strict_points(payload).items()
                }
                row["constrained_points_ground_mm"] = dict(
                    row["strict_points_ground_mm"]
                )
                row["constraint_projection"] = {"applied": False, "type": None}
            row["pose_audit"]["ankle_bearing_audit"] = audit
            if translation is not None and not measured:
                audit.update(
                    status="rejected",
                    reason="bearing_pose_step_exceeds_original_limits",
                )
        row["stage"] = stage_status["stage"]["operational"]
        row["pose_audit"]["handoff_action"] = handoff["action"]
        if measured:
            translation = self.estimator.pose.translation_world_from_camera_mm.copy()
            if (
                self.last_time is not None
                and self.last_measured_translation is not None
            ):
                self.velocity = (translation - self.last_measured_translation) / (
                    time - self.last_time
                )
                speed = np.linalg.norm(self.velocity)
                if speed > 600:
                    self.velocity *= 600 / speed
            self.last_time = time
            self.last_measured_translation = translation
        elif handoff["ground_stage"] == STAGE1:
            self.velocity *= 0
            self.last_time = None
            self.last_measured_translation = None
        elif (
            self.prediction
            and self.last_time is not None
            and time - self.last_time <= 0.15
        ):
            self.estimator.pose = WorldPose(
                self.estimator.pose.rotation_world_from_camera,
                self.estimator.pose.translation_world_from_camera_mm
                + self.velocity * dt,
            )
            row.update(self.estimator.pose.as_mapping())
            row["pose_audit"]["measurement_status"] = row["pose_audit"]["status"]
            row["pose_audit"]["status"] = "predicted"
            row["pose_audit"]["prediction_age_sec"] = time - self.last_time
            points = self.estimator._strict_points(payload)
            row["strict_points_ground_mm"] = {
                str(j): self.estimator.pose.transform(p).tolist()
                for j, p in points.items()
            }
            row["constrained_points_ground_mm"] = dict(row["strict_points_ground_mm"])
            row["constraint_projection"] = {"applied": False, "type": None}
        row["pose_audit"]["world_pose_certified"] = False
        if self.recovery:
            edge = None
            if rotation_status and rotation_status.get("status") == "accepted":
                value = np.asarray(rotation_status.get("R_to_from"), dtype=float)
                if (
                    value.shape == (3, 3)
                    and np.isfinite(value).all()
                    and np.linalg.norm(value.T @ value - np.eye(3)) <= 1e-3
                    and np.linalg.det(value) >= 0.999
                ):
                    edge = value.tolist()
            row["pose_audit"]["window_observation"] = {
                "ground_stage": handoff["ground_stage"],
                "pose_update_active": bool(self.estimator.previous_pose_active),
                "rotation_to_from": edge,
                "rotation_status": rotation_status.get("status")
                if rotation_status
                else "unavailable",
            }
            row["operational_ankle_anchors_mm"] = (
                {s: p.tolist() for s, p in self.estimator.anchors.items()}
                if self.estimator.anchors
                else None
            )
            if self.recovery.update(self.estimator, row, payload, stage_status):
                self.velocity *= 0
                self.last_time = self.last_measured_translation = None
            self.recovery.flush(self.file, time)
        else:
            self.file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.previous_time = time
        return row

    def close(self, *, completed):
        if self.recovery:
            self.recovery.flush(self.file)
        self.file.close()
        summary = self.estimator.close(completed=completed)
        if self.recovery:
            summary["landing_recovery"] = {
                "mode": self.recovery.mode,
                "events": self.recovery.events,
                "support_source": "explicit_capture_protocol_not_contact_truth",
            }
        return summary

    def _commit_fallback(self, row, pose, status, payload):
        self.estimator.pose = pose
        self.estimator.pending_rotation = np.eye(3)
        self.estimator.pending_rotation_edges = (
            self.estimator.pending_rotation_missing
        ) = self.estimator.frames_since_pose_update = 0
        self.estimator.accepted_updates += 1
        self.estimator.accepted_translation_history.append(
            (int(payload["pair_id"]), pose.translation_world_from_camera_mm.copy())
        )
        row.update(pose.as_mapping())
        row["pose_audit"]["measurement_status"] = row["pose_audit"]["status"]
        row["pose_audit"]["measurement_reason"] = row["pose_audit"].get("reason")
        row["pose_audit"].update(status=status, reason=None)
        row["strict_points_ground_mm"] = {
            str(j): pose.transform(p).tolist()
            for j, p in self.estimator._strict_points(payload).items()
        }
        row["constrained_points_ground_mm"] = dict(row["strict_points_ground_mm"])
        row["constraint_projection"] = {"applied": False, "type": None}
