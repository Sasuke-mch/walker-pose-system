"""Protocol-qualified support restoration in the pose estimator, not rendering.

Fixed-lag outputs can change only while buffered. A static image alone never
triggers recovery: preceding motion, sustained stop and resumed human motion
are required, together with an explicit flat-ground capture-protocol prior.
"""

import json
from collections import deque
from time import perf_counter

import cv2
import numpy as np

from .dynamic_ground_live import STAGE1, STAGE2
from .two_stage_motion import WorldPose


def supported_pose(pose, initial):
    relative = pose.rotation_world_from_camera @ initial.rotation_world_from_camera.T
    yaw = np.arctan2(relative[1, 0], relative[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = (
        np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        @ initial.rotation_world_from_camera
    )
    translation = pose.translation_world_from_camera_mm.copy()
    translation[2] = initial.translation_world_from_camera_mm[2]
    return WorldPose(rotation, translation)


def blend_support(pose, initial, weight):
    target = supported_pose(pose, initial)
    vector = cv2.Rodrigues(
        target.rotation_world_from_camera @ pose.rotation_world_from_camera.T
    )[0]
    rotation = cv2.Rodrigues(weight * vector)[0] @ pose.rotation_world_from_camera
    translation = pose.translation_world_from_camera_mm.copy()
    translation[2] += weight * (
        target.translation_world_from_camera_mm[2] - translation[2]
    )
    return WorldPose(rotation, translation)


class LandingPoseRecovery:
    def __init__(self, initial, *, mode, protocol, lag_sec=0.5):
        if mode not in ("causal", "fixed_lag", "joint_window"):
            raise ValueError("invalid landing recovery mode")
        if protocol != "flat_ground_supported_after_motion":
            raise ValueError(
                "landing recovery requires explicit capture-protocol prior"
            )
        if not np.isfinite(lag_sec) or lag_sec < 0.5:
            raise ValueError("landing buffer must be at least 0.5 seconds")
        self.initial, self.mode, self.lag = initial, mode, lag_sec
        self.buffer = deque()
        self.motion_start = self.stop_start = None
        self.motion_samples = self.human_samples = 0
        self.recovering = False
        self.events = []
        self.previous_time = None

    def update(self, estimator, row, payload, stage):
        time = float(payload["pair_timestamp_sec"])
        dt = 0 if self.previous_time is None else time - self.previous_time
        if not np.isfinite(time) or (self.previous_time is not None and dt <= 0):
            raise ValueError("landing timestamps must increase")
        self.previous_time = time
        row["causal"] = self.mode == "causal"
        self.buffer.append((row, payload))
        action = stage["stage"]["handoff"]["ground_stage"]
        audit = {
            "applied": False,
            "mode": self.mode,
            "support_source": "capture_protocol_with_motion_stop_human_resume_not_contact_truth",
            "nominal_output_lag_sec": self.lag if self.mode != "causal" else 0.0,
        }
        row["pose_audit"]["landing_recovery"] = audit
        if action == STAGE2:
            if self.motion_start is None or self.stop_start is not None:
                self.motion_start = time
                self.motion_samples = 0
            self.motion_samples += 1
            self.stop_start = None
            self.human_samples = 0
            self.recovering = False
        elif action == STAGE1 and self.motion_start is not None:
            if self.stop_start is None:
                self.stop_start = time
            human = any(
                p.get("upper_body_keypoints", 0) >= 3
                and p.get("upper_body_compensated_motion_px", 0) is not None
                and p.get("upper_body_compensated_motion_px", 0) >= 1.2
                and p.get("ankle_count", 0) >= 1
                and p.get("ankle_background_residual_median_px") is not None
                and p["ankle_background_residual_median_px"] >= 1.2
                for p in stage["stage"].get("pose_motion", {}).values()
            )
            self.human_samples = self.human_samples + 1 if human else 0
            qualified = (
                self.motion_samples >= 2
                and self.stop_start - self.motion_start >= 0.1
                and time - self.stop_start >= 0.12
                and self.human_samples >= 2
            )
            audit.update(
                stop_duration_sec=time - self.stop_start,
                human_resume_samples=self.human_samples,
                qualified=qualified,
            )
            if qualified or self.recovering:
                target = supported_pose(estimator.pose, self.initial)
                vector = cv2.Rodrigues(
                    target.rotation_world_from_camera
                    @ estimator.pose.rotation_world_from_camera.T
                )[0]
                height_error = float(
                    estimator.pose.translation_world_from_camera_mm[2]
                    - target.translation_world_from_camera_mm[2]
                )
                tilt_error = float(np.degrees(np.linalg.norm(vector)))
                audit.update(
                    height_error_before_mm=height_error,
                    tilt_error_before_deg=tilt_error,
                )
                if abs(height_error) > 150 or tilt_error > 8:
                    audit["reason"] = "support_correction_exceeds_150mm_or_8deg"
                elif self.mode == "joint_window":
                    if self._joint_update(estimator, support=True):
                        self.events.append(
                            {
                                "pair_id": payload["pair_id"],
                                "confirmation_sec": time,
                                "solver": "joint_stage_window",
                            }
                        )
                        self.motion_start = None
                        return True
                elif self.mode == "fixed_lag":
                    start = max(self.motion_start, self.stop_start - 0.3)
                    eligible = [
                        (r, p)
                        for r, p in self.buffer
                        if p["pair_timestamp_sec"] >= start
                    ]
                    if (
                        not eligible
                        or eligible[0][1]["pair_timestamp_sec"] >= self.stop_start
                    ):
                        audit["reason"] = "insufficient_unreleased_motion_buffer"
                    else:
                        start = eligible[0][1]["pair_timestamp_sec"]
                        for saved, source in eligible:
                            u = np.clip(
                                (source["pair_timestamp_sec"] - start)
                                / (self.stop_start - start),
                                0,
                                1,
                            )
                            weight = float(u * u * (3 - 2 * u))
                            old = WorldPose(
                                np.asarray(saved["rotation_world_from_left_camera"]),
                                np.asarray(
                                    saved["translation_world_from_left_camera_mm"]
                                ),
                            )
                            self._replace(
                                saved,
                                source,
                                blend_support(old, self.initial, weight),
                                weight,
                            )
                        estimator.pose = target
                        self._reanchor(estimator, payload)
                        self.events.append(
                            {
                                "pair_id": payload["pair_id"],
                                "motion_window_start_sec": start,
                                "stop_start_sec": self.stop_start,
                                "confirmation_sec": time,
                                "height_error_mm": height_error,
                                "tilt_error_deg": tilt_error,
                            }
                        )
                        self.motion_start = None
                        return True
                else:
                    weight = 1 - np.exp(-dt / 0.12)
                    if abs(height_error) > 1e-9:
                        weight = min(weight, 200 * dt / abs(height_error))
                    if tilt_error > 1e-9:
                        weight = min(weight, 20 * dt / tilt_error)
                    estimator.pose = blend_support(estimator.pose, self.initial, weight)
                    self._replace(row, payload, estimator.pose, float(weight))
                    self._reanchor(estimator, payload)
                    self.recovering = True
                    if abs(height_error) < 0.2 and tilt_error < 0.02:
                        self.motion_start = None
                        self.recovering = False
                    return True
        else:
            # An unavailable/ambiguous action breaks the landing evidence chain.
            # Do not use elapsed wall time across a gap as observed motion.
            self.motion_start = None
            self.motion_samples = 0
            self.stop_start = None
            self.human_samples = 0
            self.recovering = False
        if (
            self.mode == "joint_window"
            and action != STAGE1
            and row["pose_audit"]["window_observation"]["pose_update_active"]
        ):
            return self._joint_update(estimator, support=False)
        return False

    def _joint_update(self, estimator, *, support):
        from .joint_stage_window import solve_stage_window

        entries = list(self.buffer)
        started = perf_counter()
        poses, audit = solve_stage_window(entries, self.initial, support=support)
        audit["processing_ms"] = (perf_counter() - started) * 1000
        entries[-1][0]["pose_audit"]["joint_window_attempt"] = audit
        if poses is None:
            return False
        for (saved, source), pose in zip(entries[1:], poses[1:]):
            self._replace(saved, source, pose, 1.0)
            saved["pose_audit"]["landing_recovery"]["correction_type"] = (
                "joint_protocol_support" if support else "joint_motion_estimate"
            )
            if not support:
                saved["pose_audit"]["status"] = "joint_window_updated"
            saved["pose_audit"]["joint_window_solution"] = audit
        estimator.pose = poses[-1]
        estimator.pending_rotation = np.eye(3)
        estimator.pending_rotation_edges = estimator.pending_rotation_missing = 0
        estimator.frames_since_pose_update = 0
        estimator.accepted_translation_history.clear()
        estimator.accepted_translation_history.append(
            (
                int(entries[-1][1]["pair_id"]),
                poses[-1].translation_world_from_camera_mm.copy(),
            )
        )
        if support:
            self._reanchor(estimator, entries[-1][1])
        return True

    @staticmethod
    def _replace(row, payload, pose, weight):
        if weight <= 0:
            row["pose_audit"]["landing_recovery"].update(
                applied=False, correction_weight=0.0
            )
            return
        row["pose_audit"].setdefault(
            "measurement_pose_before_support",
            {
                k: row[k]
                for k in (
                    "rotation_world_from_left_camera",
                    "translation_world_from_left_camera_mm",
                )
            },
        )
        row["pose_audit"]["measurement_audit_scope"] = "before_support_restoration"
        row.update(pose.as_mapping())
        row["pose_audit"].setdefault("landing_recovery", {}).update(
            applied=True, correction_weight=weight
        )
        if weight > 0:
            row["pose_audit"].setdefault(
                "pre_support_status", row["pose_audit"]["status"]
            )
            if row["stage"] == STAGE1:
                row["pose_audit"]["status"] = "support_restored"
        points = {
            str(p["index"]): pose.transform(np.asarray(p["xyz"])).tolist()
            for person in payload.get("persons_3d", [])
            for p in person["keypoints_3d"]
            if p["valid"]
        }
        anchors = row.get("operational_ankle_anchors_mm") or {}
        residuals = {
            s: float(
                np.linalg.norm(np.asarray(points[str(j)]) - np.asarray(anchors[s]))
            )
            for s, j in [("left", 15), ("right", 16)]
            if s in anchors and str(j) in points
        }
        row["pose_audit"]["landing_recovery"]["post_support_ankle_residual_mm"] = (
            residuals
        )
        if len(residuals) == 2:
            disagreement = np.linalg.norm(
                (np.asarray(points["15"]) - np.asarray(anchors["left"]))
                - (np.asarray(points["16"]) - np.asarray(anchors["right"]))
            )
            row["pose_audit"]["landing_recovery"][
                "post_support_two_foot_disagreement_mm"
            ] = float(disagreement)
        row["strict_points_ground_mm"] = points
        row["constrained_points_ground_mm"] = dict(points)
        row["constraint_projection"] = {"applied": False, "type": None}

    @staticmethod
    def _reanchor(estimator, payload):
        points = estimator._strict_points(payload)
        estimator.anchors = None
        estimator.recent_feet.clear()
        estimator.recent_anchor_feet_world.clear()
        estimator.accepted_translation_history.clear()
        estimator.latest_stationary_feet_world = (
            {
                s: estimator.pose.transform(points[j])
                for s, j in [("left", 15), ("right", 16)]
            }
            if 15 in points and 16 in points
            else None
        )
        if estimator.latest_stationary_feet_world is not None:
            estimator.recent_anchor_feet_world.append(
                estimator.latest_stationary_feet_world
            )

    def flush(self, file, now=None):
        delay = self.lag if self.mode != "causal" else 0
        while self.buffer and (
            now is None or now - self.buffer[0][1]["pair_timestamp_sec"] >= delay
        ):
            row, source = self.buffer.popleft()
            release_time = self.previous_time if now is None else now
            row["pose_audit"]["landing_recovery"]["release_delay_sec"] = (
                release_time - source["pair_timestamp_sec"]
            )
            row["pose_audit"]["landing_recovery"]["stream_tail_flush"] = now is None
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
