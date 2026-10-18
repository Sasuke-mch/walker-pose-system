"""Causal fixed-lag kinematic skeleton tracker (real-time mainline candidate).

A 15-frame sliding window over the *current and past* frames only. Each update
optimises the window state for a fixed small number of iterations (5/10/20)
with Adam on the per-frame root pose, root rotation and the 8 joint rotations;
bone lengths are FROZEN (shared across the whole sequence, estimated offline
from six clips and never optimised at runtime).

Causality contract:
  - the window contains the current frame and at most 14 history frames;
  - no future frame is ever read;
  - the first window cold-starts from the offline prefit (allowed for
    offline cold-start / comparison only; the full-sequence prefit is never a
    runtime soft target);
  - later windows warm-start from the previous window: overlapping frames
    keep their fitted state, the new frame root translation is initialised by
    constant velocity from the last two fitted frames and its root rotation /
    joint rotations are copied from the previous frame;
  - only the current frame is output; history outputs are never rewritten.

Runtime soft-3D (optional, off by default) may only read the CURRENT frame
strict stereo triangulated points (strict_stereo_person), never the full-
sequence prefit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from pose_app.fisheye_camera import (
    StereoFisheyeCalibration,
    fisheye_project_torch,
    world_to_cameras_torch_batch,
)
from pose_app.pose_observation_partition import (
    REASON_ACCEPTED_RELAXED,
    REASON_ACCEPTED_STRICT,
    classify_view_observations_batch,
)
from pose_app.reprojection_common import view_2d_weights
from pose_app.skeleton_factor_graph import (
    BODY_JOINTS,
    FixedBoneSkeleton,
    HINGE_PARAM_INDICES,
    compute_init_from_points,
    hinge_soft_prior,
)


@dataclass(frozen=True)
class CausalSkeletonConfig:
    # --- frozen contract fields (specified defaults) ---
    window_size: int = 15
    iterations_per_update: int = 10
    learning_rate: float = 0.0005
    gm_scale_px: float = 20.0
    temporal_weight: float = 1000.0
    foot_static_weight: float = 4000.0
    ground_weight: float = 100.0
    hand_weight: float = 60.0
    rotation_prior_weight: float = 0.02
    # --- supplementary fields (defaults used by the causal experiment) ---
    data2d_weight: float = 40.0
    camera_barrier_weight: float = 50.0
    hinge_prior_weight: float = 10.0
    max_hinge_rotation_rad: float = 2.9
    min_segment_angle_deg: float = 40.0
    relaxed_weight: float = 0.25
    minimum_score: float = 0.20
    # Optional runtime soft-3D anchor from CURRENT-frame strict triangulated
    # points only. 0.0 disables it (default).
    soft3d_current_frame_weight: float = 0.0


# COCO joint indices -> positions in the 12-vector skeleton output.
_COCO_TO_POS = {joint: position for position, joint in enumerate(BODY_JOINTS)}
_POS_ANKLES = [_COCO_TO_POS[15], _COCO_TO_POS[16]]
_POS_WRISTS = [_COCO_TO_POS[9], _COCO_TO_POS[10]]


class CausalSkeletonTracker:
    """Causal fixed-lag tracker. One instance owns the sliding window state."""

    def __init__(
        self,
        config: CausalSkeletonConfig,
        cal: StereoFisheyeCalibration,
        frozen_bone_lengths_mm: np.ndarray,
        prefit_points_mm: Optional[np.ndarray] = None,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.cal = cal
        self.frozen_bone_lengths_mm = np.asarray(frozen_bone_lengths_mm, dtype=np.float64)
        if not np.all(np.isfinite(self.frozen_bone_lengths_mm)) or np.any(self.frozen_bone_lengths_mm <= 0.0):
            raise ValueError("frozen bone lengths must be finite and positive")
        if len(self.frozen_bone_lengths_mm) != 11:
            raise ValueError("expected 11 frozen bone lengths")
        self.prefit_points_mm = prefit_points_mm
        self.device = device
        self._window: list[dict[str, Any]] = []  # frame entries, current at the end
        # Internal fitted state per processed frame (used for warm-start only).
        self._param_by_frame: dict[int, dict[str, np.ndarray]] = {}
        # Output history: frame_index -> (joints 12x3 mm, metadata). Never rewritten.
        self.output_history: dict[int, dict[str, Any]] = {}

    # ------------------------------------------------------------------ state
    def _append_frame(
        self,
        frame_index: int,
        motion_frame: dict,
        stereo_row: dict | None,
        strict_stereo_person: dict | None,
        interaction_record: dict | None,
    ) -> None:
        left_kp = right_kp = None
        if stereo_row is not None:
            left_persons = stereo_row.get("left", {}).get("persons") or []
            right_persons = stereo_row.get("right", {}).get("persons") or []
            if left_persons:
                left_kp = np.asarray(left_persons[0]["keypoints"], dtype=np.float64)
            if right_persons:
                right_kp = np.asarray(right_persons[0]["keypoints"], dtype=np.float64)
        entry = {
            "frame_index": int(frame_index),
            "motion_frame": motion_frame,
            "stereo_row": stereo_row,
            "strict_stereo_person": strict_stereo_person,
            "interaction_record": interaction_record,
            "left_kp": left_kp,
            "right_kp": right_kp,
        }
        self._window.append(entry)
        if len(self._window) > self.config.window_size:
            del self._window[0]

    def _build_obs(self) -> dict[str, np.ndarray]:
        """Per-view observation arrays for the current window (causal)."""
        n = len(self._window)
        kp_left = np.full((n, 12, 2), np.nan)
        kp_right = np.full_like(kp_left, np.nan)
        score_left = np.full((n, 12), np.nan)
        score_right = np.full_like(score_left, np.nan)
        strict = np.zeros((n, 12), dtype=bool)
        errors = np.full((n, 12), np.nan)
        for local, entry in enumerate(self._window):
            motion_frame = entry["motion_frame"]
            for position, joint in enumerate(BODY_JOINTS):
                if entry["left_kp"] is not None:
                    lv = entry["left_kp"][joint]
                    if np.all(np.isfinite(lv[:2])):
                        kp_left[local, position] = lv[:2]
                        score_left[local, position] = float(lv[2])
                if entry["right_kp"] is not None:
                    rv = entry["right_kp"][joint]
                    if np.all(np.isfinite(rv[:2])):
                        kp_right[local, position] = rv[:2]
                        score_right[local, position] = float(rv[2])
                strict[local, position] = bool(motion_frame["strict"][joint])
                err = motion_frame["errors_px"][joint]
                if err is not None:
                    errors[local, position] = float(err)
        part_l = classify_view_observations_batch(
            kp_left, score_left, strict, errors, self.cal.image_width, self.cal.image_height,
            self.config.minimum_score,
        )
        part_r = classify_view_observations_batch(
            kp_right, score_right, strict, errors, self.cal.image_width, self.cal.image_height,
            self.config.minimum_score,
        )
        w_l = view_2d_weights(score_left, part_l["accepted"], strict, errors, relaxed_weight=self.config.relaxed_weight)
        w_r = view_2d_weights(score_right, part_r["accepted"], strict, errors, relaxed_weight=self.config.relaxed_weight)
        return {
            "kp_left_px": kp_left, "kp_right_px": kp_right,
            "accepted_left": part_l["accepted"], "accepted_right": part_r["accepted"],
            "reasons_left": part_l["reasons"], "reasons_right": part_r["reasons"],
            "rejection_counts_left": part_l["rejection_counts"],
            "rejection_counts_right": part_r["rejection_counts"],
            "strict": strict, "errors_px": errors,
            "weights_left": w_l, "weights_right": w_r,
        }

    def _build_init(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Warm-start init for the current window (causal; no future frames)."""
        indices = np.asarray([entry["frame_index"] for entry in self._window], dtype=np.int64)
        n = len(indices)
        root_pos = np.zeros((n, 3), dtype=np.float64)
        root_rot = np.zeros((n, 3), dtype=np.float64)
        rotations = np.zeros((n, 8, 3), dtype=np.float64)

        prev_indices = sorted(self._param_by_frame)
        for local, frame_index in enumerate(indices):
            if frame_index in self._param_by_frame:
                state = self._param_by_frame[frame_index]
                # The stored fitted state is in metres; FixedBoneSkeleton expects
                # millimetres for the init (it divides by 1000 internally).
                root_pos[local] = state["root_pos"] * 1000.0
                root_rot[local] = state["root_rot"]
                rotations[local] = state["rotations"]
            else:
                # The newest frame: constant velocity root translation, copy
                # rotations from the previous frame (never from the future).
                if local > 0:
                    if local >= 2:
                        velocity = root_pos[local - 1] - root_pos[local - 2]
                    else:
                        velocity = np.zeros(3)
                    root_pos[local] = root_pos[local - 1] + velocity
                    root_rot[local] = root_rot[local - 1]
                    rotations[local] = rotations[local - 1]
                else:
                    # Cold start (first frame ever): offline prefit init.
                    if self.prefit_points_mm is not None:
                        single = self.prefit_points_mm[frame_index][list(BODY_JOINTS)][None]
                        init = compute_init_from_points(single)
                        root_pos[local] = init["root_pos_mm"][0]
                        root_rot[local] = init["root_rot"][0]
                        rotations[local] = init["joint_rotation_init"][0]
        return {"root_pos_mm": root_pos, "root_rot": root_rot, "joint_rotation_init": rotations}

    def _loss_terms(
        self, skeleton: FixedBoneSkeleton, obs: dict[str, np.ndarray], torch: Any,
    ) -> tuple[Any, dict[str, Any]]:
        cfg = self.config
        n = skeleton.rotations.shape[0]
        joints = skeleton.forward()  # (N,12,3) world m
        per_frame_rotation = [
            np.asarray(entry["motion_frame"]["rotation_world_from_left"], dtype=np.float64)
            for entry in self._window
        ]
        per_frame_translation = [
            np.asarray(entry["motion_frame"]["translation_world_from_left_mm"], dtype=np.float64)
            for entry in self._window
        ]
        left_m, right_m = world_to_cameras_torch_batch(
            joints, per_frame_rotation, per_frame_translation, self.cal
        )
        left_barrier = torch.clamp(0.3 - left_m[..., 2], min=0.0).square().mean()
        right_barrier = torch.clamp(0.3 - right_m[..., 2], min=0.0).square().mean()
        camera_barrier = 0.5 * (left_barrier + right_barrier)
        left_m = torch.cat([left_m[..., :2], torch.clamp(left_m[..., 2:], min=0.05)], dim=-1)
        right_m = torch.cat([right_m[..., :2], torch.clamp(right_m[..., 2:], min=0.05)], dim=-1)
        proj_l = fisheye_project_torch(left_m, self.cal.K0, self.cal.D0)
        proj_r = fisheye_project_torch(right_m, self.cal.K1, self.cal.D1)
        kp_l = torch.as_tensor(np.nan_to_num(obs["kp_left_px"]), dtype=torch.float32, device=self.device)
        kp_r = torch.as_tensor(np.nan_to_num(obs["kp_right_px"]), dtype=torch.float32, device=self.device)
        w_l = torch.as_tensor(obs["weights_left"], dtype=torch.float32, device=self.device)
        w_r = torch.as_tensor(obs["weights_right"], dtype=torch.float32, device=self.device)
        residual_l = torch.linalg.vector_norm(proj_l - kp_l, dim=2)
        residual_r = torch.linalg.vector_norm(proj_r - kp_r, dim=2)
        scale_px2 = float(cfg.gm_scale_px) ** 2
        loss_l = torch.sum(w_l * (residual_l * residual_l) / (residual_l * residual_l + scale_px2)) / torch.clamp(w_l.sum(), min=1.0)
        loss_r = torch.sum(w_r * (residual_r * residual_r) / (residual_r * residual_r + scale_px2)) / torch.clamp(w_r.sum(), min=1.0)
        data2d = 0.5 * (loss_l + loss_r)

        temporal = torch.zeros((), dtype=torch.float32, device=self.device)
        if n >= 3:
            acc = joints[2:] - 2.0 * joints[1:-1] + joints[:-2]
            temporal = torch.mean(acc.square())

        # Stage-2 foot staticness: ankle velocity during active (Stage 2) pairs.
        stages = np.asarray([entry["motion_frame"]["stage"] for entry in self._window], dtype=object)
        stage2 = np.asarray([str(s) == "stage2_feet_static_walker_moving" for s in stages], dtype=np.float32)
        foot_static = torch.as_tensor(stage2, dtype=torch.float32, device=self.device).reshape(n, 1)
        foot_static_pair = (foot_static[1:] * foot_static[:-1]).reshape(-1)
        ankle_vel = torch.linalg.vector_norm(joints[1:, _POS_ANKLES] - joints[:-1, _POS_ANKLES], dim=2).sum(dim=1)
        foot_static_loss = torch.sum(foot_static_pair * ankle_vel.square()) / torch.clamp(foot_static_pair.sum(), min=1.0)

        ankle_z_mm = joints[:, _POS_ANKLES][..., 2] * 1000.0
        ground_floor_mm = 5.0
        penetration = torch.clamp(ground_floor_mm - ankle_z_mm, min=0.0)
        ground_loss = torch.sum(penetration.square()) / float(n * 2)

        hand_target = np.full((n, 2, 3), np.nan)
        hand_valid = np.zeros((n, 2), dtype=np.float32)
        for local, entry in enumerate(self._window):
            record = entry["interaction_record"] or {}
            for side_index, side in enumerate(("left", "right")):
                proxy = (record.get("hands") or {}).get(side, {}).get("closest_proxy_point_world_mm")
                if proxy is not None and np.all(np.isfinite(proxy)):
                    hand_target[local, side_index] = proxy
                    hand_valid[local, side_index] = 1.0
        hand_target = torch.as_tensor(np.nan_to_num(hand_target / 1000.0, nan=0.0), dtype=torch.float32, device=self.device)
        hand_valid_t = torch.as_tensor(hand_valid, dtype=torch.float32, device=self.device)
        wrist = joints[:, _POS_WRISTS]
        hand_residual = torch.linalg.vector_norm(wrist - hand_target, dim=2)
        hand_loss = torch.sum(hand_valid_t * hand_residual.square()) / torch.clamp(hand_valid_t.sum(), min=1.0)

        rotation_prior = torch.mean((skeleton.rotations - skeleton.joint_rotation_init[:n]).square())
        hinge_prior, hinge_frame_penalty = hinge_soft_prior(
            joints,
            max_hinge_rotation_rad=cfg.max_hinge_rotation_rad,
            min_segment_angle_deg=cfg.min_segment_angle_deg,
        )
        hinge_rot_norm = torch.linalg.vector_norm(skeleton.rotations[:, HINGE_PARAM_INDICES], dim=2)
        hinge_rotation_cap = torch.clamp(hinge_rot_norm - cfg.max_hinge_rotation_rad, min=0.0).square().mean()
        hinge_frame_triggered = (
            (hinge_frame_penalty > 1e-6) | (hinge_rot_norm > cfg.max_hinge_rotation_rad).any(dim=1)
        )

        soft3d_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        if cfg.soft3d_current_frame_weight > 0.0 and self._window:
            # CURRENT-frame strict triangulated points in the ground frame
            # (motion.json points_mm), same source stream as the 2D keypoints.
            last_motion = self._window[-1]["motion_frame"]
            points_mm = last_motion.get("points_mm")
            strict_flags = last_motion.get("strict")
            if points_mm is not None and strict_flags is not None and len(points_mm) >= 17:
                pts = np.full((17, 3), np.nan, dtype=np.float64)
                for joint in range(17):
                    value = points_mm[joint]
                    if value is not None:
                        pts[joint] = np.asarray(value, dtype=np.float64)
                valid = np.asarray(strict_flags, dtype=bool)
                finite = np.all(np.isfinite(pts), axis=1)
                mask = (valid & finite).astype(np.float32)
                target = torch.as_tensor(np.nan_to_num(pts[list(BODY_JOINTS)] / 1000.0, nan=0.0), dtype=torch.float32, device=self.device)
                mask_t = torch.as_tensor(mask[list(BODY_JOINTS)], dtype=torch.float32, device=self.device)
                diff = (joints[-1] - target).square().sum(dim=-1)
                soft3d_loss = torch.sum(mask_t * diff) / torch.clamp(mask_t.sum() * 3.0, min=1.0)

        loss = (
            cfg.data2d_weight * data2d
            + cfg.camera_barrier_weight * camera_barrier
            + cfg.hinge_prior_weight * (hinge_prior + hinge_rotation_cap)
            + cfg.rotation_prior_weight * rotation_prior
            + cfg.temporal_weight * temporal
            + cfg.foot_static_weight * foot_static_loss
            + cfg.ground_weight * ground_loss
            + cfg.hand_weight * hand_loss
            + cfg.soft3d_current_frame_weight * soft3d_loss
        )
        terms = {
            "data2d": float(data2d.detach().cpu()),
            "camera_barrier": float(camera_barrier.detach().cpu()),
            "temporal": float(temporal.detach().cpu()),
            "foot_static": float(foot_static_loss.detach().cpu()),
            "ground": float(ground_loss.detach().cpu()),
            "hand": float(hand_loss.detach().cpu()),
            "rotation_prior": float(rotation_prior.detach().cpu()),
            "hinge_prior": float(hinge_prior.detach().cpu()),
            "hinge_rotation_cap": float(hinge_rotation_cap.detach().cpu()),
            "soft3d_current_frame": float(soft3d_loss.detach().cpu()),
            "hinge_frame_triggered": hinge_frame_triggered.detach().cpu().numpy(),
        }
        return loss, terms

    # ------------------------------------------------------------------ update
    def update(
        self,
        frame_index: int,
        motion_frame: dict,
        stereo_row: dict,
        strict_stereo_person: dict | None,
        interaction_record: dict,
    ) -> dict:
        """Process one new frame (the current frame). Returns the current-frame result."""
        import time as _time
        import torch

        self._append_frame(frame_index, motion_frame, stereo_row, strict_stereo_person, interaction_record)
        obs = self._build_obs()
        n = len(self._window)
        if n > self.config.window_size:
            raise RuntimeError("window exceeded its configured size")
        if self.config.iterations_per_update not in (5, 10, 20):
            raise ValueError("iterations_per_update must be one of 5, 10, 20")

        init = self._build_init(obs)
        skeleton = FixedBoneSkeleton(
            bone_lengths_mm=self.frozen_bone_lengths_mm,
            joint_rotation_init=init["joint_rotation_init"],
            root_rot_init=init["root_rot"],
            root_pos_init_mm=init["root_pos_mm"],
            device=self.device,
        )
        skeleton.initialise(n, requires_grad_bones=False)
        optimizer = torch.optim.Adam(
            [skeleton.rotations, skeleton.root_pos, skeleton.root_rot],
            lr=self.config.learning_rate,
        )
        started = _time.perf_counter()
        last_terms = {}
        for _ in range(self.config.iterations_per_update):
            optimizer.zero_grad()
            loss, terms = self._loss_terms(skeleton, obs, torch)
            last_terms = terms
            loss.backward()
            optimizer.step()
        elapsed = _time.perf_counter() - started

        with torch.no_grad():
            fitted_m = skeleton.forward().cpu().numpy()  # (N,12,3) m
        joints_mm = fitted_m * 1000.0  # (N,12,3) mm world
        current_joints_mm = joints_mm[-1]

        # Store fitted state for warm-starting the next window.
        self._param_by_frame[frame_index] = {
            "root_pos": skeleton.root_pos.detach().cpu().numpy()[-1],
            "root_rot": skeleton.root_rot.detach().cpu().numpy()[-1],
            "rotations": skeleton.rotations.detach().cpu().numpy()[-1],
        }
        for local, entry in enumerate(self._window):
            self._param_by_frame[entry["frame_index"]] = {
                "root_pos": skeleton.root_pos.detach().cpu().numpy()[local],
                "root_rot": skeleton.root_rot.detach().cpu().numpy()[local],
                "rotations": skeleton.rotations.detach().cpu().numpy()[local],
            }

        nan_count = int(np.count_nonzero(~np.isfinite(joints_mm)))
        behind_camera = self._count_behind_camera(joints_mm)
        hinge_frame_triggered_np = np.asarray(last_terms["hinge_frame_triggered"], dtype=bool)
        current_hinge_triggered = bool(hinge_frame_triggered_np[-1]) if n > 0 else False

        result = {
            "frame_index": int(frame_index),
            "pair_id": int(motion_frame["pair"]),
            "stage": motion_frame["stage"],
            "joints_ground_mm": current_joints_mm,
            "update_time_s": elapsed,
            "iterations_performed": self.config.iterations_per_update,
            "algorithmic_lookahead_frames": 0,
            "window_history_frames": n - 1,
            "nan_count": nan_count,
            "behind_camera_points": behind_camera,
            "negative_bone_count": int(np.count_nonzero(self.frozen_bone_lengths_mm <= 0.0)),
            "loss_terms": last_terms,
            "obs_partition_left": dict(obs["rejection_counts_left"]),
            "obs_partition_right": dict(obs["rejection_counts_right"]),
            "hinge_penalty_triggered": current_hinge_triggered,
        }
        # The output history is append-only.
        self.output_history[frame_index] = {
            "frame_index": int(frame_index),
            "pair_id": result["pair_id"],
            "stage": result["stage"],
            "joints_ground_mm": current_joints_mm.round(4).tolist(),
            "update_time_s": elapsed,
            "iterations_performed": self.config.iterations_per_update,
            "algorithmic_lookahead_frames": 0,
            "window_history_frames": n - 1,
            "nan_count": nan_count,
            "behind_camera_points": behind_camera,
            "hinge_penalty_triggered": current_hinge_triggered,
        }
        return result

    def _count_behind_camera(self, joints_mm: np.ndarray) -> int:
        count = 0
        for local, entry in enumerate(self._window):
            rotation = np.asarray(entry["motion_frame"]["rotation_world_from_left"], dtype=np.float64)
            translation = np.asarray(entry["motion_frame"]["translation_world_from_left_mm"], dtype=np.float64)
            left = (rotation.T @ (joints_mm[local] - translation).T).T
            right = (self.cal.R_cam0_to_cam1 @ left.T + self.cal.T_cam0_to_cam1_mm.reshape(3, 1)).T
            count += int(np.count_nonzero(left[:, 2] <= 0.0)) + int(np.count_nonzero(right[:, 2] <= 0.0))
        return count
