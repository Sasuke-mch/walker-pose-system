#!/usr/bin/env python3
"""Fit the fixed-bone-length skeleton factor graph on one window (real-time mainline candidate).

Primary observations are the frozen L/R raw-fisheye 2D keypoints reprojected
through the stereo calibration (GM-robust direct pixel residual).
Shared fixed bone lengths are optimised jointly; temporal smoothness, contact
foot staticness, ground non-penetration, hand-to-handle proximity and a weak
prior toward the initial skeleton keep the result stable. No mesh is produced.

The output metrics intentionally mirror fit_smplx_window_2d_reproj.py so the
two experiments can be compared directly on the same window.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from pose_app.fisheye_camera import (  # noqa: E402
    fisheye_project_torch,
    load_stereo_fisheye,
    world_to_cameras_torch_batch,
)
from pose_app.reprojection_common import (  # noqa: E402
    acceleration_p95_mm,
    build_2d_observations,
    build_view_2d_weights_from_obs,
)
from pose_app.skeleton_factor_graph import (  # noqa: E402
    BODY_JOINTS,
    DEFAULT_BONE_NAMES,
    FixedBoneSkeleton,
    HINGE_PARAM_INDICES,
    compute_init_from_points,
    estimate_bone_lengths_mm,
    hinge_soft_prior,
)
from pose_app.smplx_fitting import read_jsonl  # noqa: E402
from pose_app.temporal_body_prefit import (  # noqa: E402
    TemporalBodyPrefitConfig,
    build_contact_targets,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-json", type=Path, required=True)
    parser.add_argument("--prefit-jsonl", type=Path, required=True)
    parser.add_argument("--interaction-jsonl", type=Path, required=True)
    parser.add_argument("--stereo-jsonl", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=60)
    parser.add_argument("--end-frame", type=int, default=90)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--phase-split", type=int, default=300,
                        help="iterations spent in phase 1 (data terms only, stable) before "
                             "adding temporal/contact terms in phase 2 at a lower learning rate")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--phase2-learning-rate", type=float, default=0.005)
    parser.add_argument("--scheme", choices=("2d_only", "2d_plus_soft3d"), default="2d_plus_soft3d")
    parser.add_argument("--data2d-weight", type=float, default=40.0)
    parser.add_argument("--soft3d-weight", type=float, default=15.0)
    parser.add_argument("--temporal-weight", type=float, default=300.0)
    parser.add_argument("--foot-static-weight", type=float, default=200.0)
    parser.add_argument("--ground-weight", type=float, default=100.0)
    parser.add_argument("--hand-weight", type=float, default=60.0)
    parser.add_argument("--bone-prior-weight", type=float, default=0.2)
    parser.add_argument("--rotation-prior-weight", type=float, default=0.02)
    parser.add_argument("--gm-scale-px", type=float, default=20.0)
    parser.add_argument("--hold-out-joints", type=str, default="",
                        help="comma-separated COCO body joints excluded from all data terms")
    parser.add_argument("--frozen-bones-json", type=Path, default=None,
                        help="frozen_skeleton_shape.json with frozen_median_lengths_mm; "
                             "when provided bone lengths are NOT optimised")
    parser.add_argument("--hinge-prior-weight", type=float, default=0.0,
                        help="soft elbow/knee angle-range and fold-back prior weight")
    parser.add_argument("--max-hinge-rotation-rad", type=float, default=2.9)
    parser.add_argument("--min-segment-angle-deg", type=float, default=40.0)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)

    import torch

    motion = json.loads(args.motion_json.read_text(encoding="utf-8-sig"))
    prefit = read_jsonl(args.prefit_jsonl)
    interactions = read_jsonl(args.interaction_jsonl)
    stereo_rows = read_jsonl(args.stereo_jsonl)
    stereo_by_pair = {int(row["pair_id"]): row for row in stereo_rows}
    cal = load_stereo_fisheye(args.calibration_dir)

    frame_count = len(motion["frames"])
    if not (len(prefit) == len(interactions) == frame_count):
        raise ValueError("motion, prefit and interaction streams must align")
    if not (0 <= args.start_frame <= args.end_frame < frame_count):
        raise ValueError("invalid inclusive frame window")
    indices = np.arange(args.start_frame, args.end_frame + 1)
    frames = motion["frames"]
    window_frames = [frames[index] for index in indices]
    window_size = len(indices)
    device = torch.device(args.device)

    raw_points = np.asarray([
        [point if point is not None else [np.nan, np.nan, np.nan] for point in frame["points_mm"]]
        for frame in frames
    ], dtype=np.float64)
    prefit_points_mm = np.asarray([
        [row["points_ground_mm"][str(joint)] for joint in range(17)] for row in prefit
    ], dtype=np.float64)
    errors = np.asarray([
        [value if value is not None else np.nan for value in frame["errors_px"]] for frame in frames
    ], dtype=np.float64)
    strict = np.asarray([frame["strict"] for frame in frames], dtype=bool)
    _, contact_weights, _ = build_contact_targets(
        raw_points, frames, interactions, motion["walker"], TemporalBodyPrefitConfig()
    )

    body_indices = np.asarray(BODY_JOINTS, dtype=np.int64)  # COCO 5..16
    target_body_mm = prefit_points_mm[indices][:, body_indices]  # (N,12,3) mm
    prefit_body = target_body_mm
    init = compute_init_from_points(prefit_body)
    bone_lengths_init = estimate_bone_lengths_mm(prefit_body)

    frozen_bones = None
    if args.frozen_bones_json is not None:
        frozen_record = json.loads(args.frozen_bones_json.read_text(encoding="utf-8"))
        frozen_bones = np.asarray(frozen_record["frozen_median_lengths_mm"], dtype=np.float64)
        if frozen_bones.shape != (len(DEFAULT_BONE_NAMES),):
            raise ValueError("frozen bone length vector has the wrong shape")
        bone_lengths_init = frozen_bones.copy()
        if not np.all(np.isfinite(bone_lengths_init)) or np.any(bone_lengths_init <= 0.0):
            raise ValueError("frozen bone lengths must be finite and positive")

    hold_out = {int(part) for part in args.hold_out_joints.split(",") if part.strip()}
    if not hold_out.issubset(set(BODY_JOINTS)):
        raise ValueError(f"hold-out joints must be body joints {list(BODY_JOINTS)}")
    hold_out_positions = [position for position, joint in enumerate(BODY_JOINTS) if joint in hold_out]
    child_rotation_parameter = {7: 0, 8: 1, 9: 2, 10: 3, 13: 4, 14: 5, 15: 6, 16: 7}
    for joint in hold_out:
        parameter = child_rotation_parameter.get(joint)
        if parameter is not None:
            init["joint_rotation_init"][:, parameter] = 0.0

    skeleton = FixedBoneSkeleton(
        bone_lengths_mm=bone_lengths_init,
        joint_rotation_init=init["joint_rotation_init"],
        root_rot_init=init["root_rot"],
        root_pos_init_mm=init["root_pos_mm"],
        device=args.device,
    )
    skeleton.initialise(window_size, requires_grad_bones=args.frozen_bones_json is None)

    obs2d = build_2d_observations(
        stereo_by_pair, frames, indices, body_indices, cal.image_width, cal.image_height
    )
    kp_left = torch.as_tensor(np.nan_to_num(obs2d["kp_left_px"]), dtype=torch.float32, device=device)
    kp_right = torch.as_tensor(np.nan_to_num(obs2d["kp_right_px"]), dtype=torch.float32, device=device)
    w2d_left_np, w2d_right_np = build_view_2d_weights_from_obs(obs2d)
    for position in hold_out_positions:
        w2d_left_np[:, position] = 0.0
        w2d_right_np[:, position] = 0.0
    w2d_left = torch.as_tensor(w2d_left_np, dtype=torch.float32, device=device)
    w2d_right = torch.as_tensor(w2d_right_np, dtype=torch.float32, device=device)
    kp_valid_l = torch.as_tensor(obs2d["valid_left"], dtype=torch.float32, device=device)
    kp_valid_r = torch.as_tensor(obs2d["valid_right"], dtype=torch.float32, device=device)

    per_frame_rotation = [np.asarray(frames[index]["rotation_world_from_left"], dtype=np.float64) for index in indices]
    per_frame_translation = [np.asarray(frames[index]["translation_world_from_left_mm"], dtype=np.float64) for index in indices]

    # Contact-phase foot static mask: ankle joints during active contact.
    foot_static = torch.as_tensor(
        (contact_weights[indices][:, [15, 16]] > 0).astype(np.float32), dtype=torch.float32, device=device
    )
    for side, joint in enumerate((15, 16)):
        if joint in hold_out:
            foot_static[:, side] = 0.0
    # Ground non-penetration: world z is height above ground; penalise below 5 mm.
    ankle_positions = [10, 11]  # positions of COCO 15,16 in the 12-vector
    ground_floor_mm = 5.0

    hand_proxy = np.full((window_size, 2, 3), np.nan)
    for local, index in enumerate(indices):
        for side, side_index in (("left", 0), ("right", 1)):
            proxy = interactions[index]["hands"][side].get("closest_proxy_point_world_mm")
            if proxy is not None:
                hand_proxy[local, side_index] = proxy
    hand_proxy = np.nan_to_num(hand_proxy, nan=0.0) / 1000.0
    hand_valid = np.asarray(
        [[bool(np.all(np.isfinite(interactions[index]["hands"][side].get("closest_proxy_point_world_mm", (np.nan, np.nan, np.nan))))) for side in ("left", "right")]
        for index in indices
    ], dtype=np.float32)
    hand_target = torch.as_tensor(hand_proxy, dtype=torch.float32, device=device)
    hand_valid_t = torch.as_tensor(hand_valid, dtype=torch.float32, device=device)
    for side, joint in enumerate((9, 10)):
        if joint in hold_out:
            hand_valid_t[:, side] = 0.0
    wrist_positions = [4, 5]  # positions of COCO 9,10 in the 12-vector

    bone_init = torch.as_tensor(bone_lengths_init / 1000.0, dtype=torch.float32, device=device)
    rotation_init = torch.as_tensor(init["joint_rotation_init"], dtype=torch.float32, device=device)
    rotation_prior_mask = torch.ones((1, 8, 1), dtype=torch.float32, device=device)
    for joint in hold_out:
        parameter = child_rotation_parameter.get(joint)
        if parameter is not None:
            rotation_prior_mask[:, parameter] = 0.0
    soft3d_mask = torch.ones((1, len(BODY_JOINTS), 1), dtype=torch.float32, device=device)
    for position in hold_out_positions:
        soft3d_mask[:, position] = 0.0
    ground_mask = torch.ones((1, 2), dtype=torch.float32, device=device)
    for side, joint in enumerate((15, 16)):
        if joint in hold_out:
            ground_mask[:, side] = 0.0

    optimize_bones = args.frozen_bones_json is None
    bone_param_list = [skeleton.bone_lengths] if optimize_bones else []
    if args.optimizer == "adam":
        optimizer_phase1 = torch.optim.Adam(
            [skeleton.rotations, skeleton.root_pos, skeleton.root_rot] + bone_param_list,
            lr=args.learning_rate,
        )
        optimizer_phase2 = torch.optim.Adam(
            [skeleton.rotations, skeleton.root_pos, skeleton.root_rot] + bone_param_list,
            lr=args.phase2_learning_rate,
        )
    else:
        optimizer_phase1 = torch.optim.SGD(
            [skeleton.rotations, skeleton.root_pos, skeleton.root_rot] + bone_param_list,
            lr=args.learning_rate, momentum=0.9,
        )
        optimizer_phase2 = torch.optim.SGD(
            [skeleton.rotations, skeleton.root_pos, skeleton.root_rot] + bone_param_list,
            lr=args.phase2_learning_rate, momentum=0.9,
        )
    optimizer = optimizer_phase1
    final_terms = {}
    started = time.perf_counter()
    for iteration in range(args.iterations):
        optimizer.zero_grad()
        joints = skeleton.forward()  # (N,12,3) world m

        left_m, right_m = world_to_cameras_torch_batch(joints, per_frame_rotation, per_frame_translation, cal)
        # Camera-plane barrier: keep joints in front of the camera so the fisheye
        # division cannot push them behind and trap them there. The barrier acts on
        # the UNCLAMPED Z so the gradient flows back; the projection clamp only
        # keeps the loss finite mid-step.
        left_barrier = torch.clamp(0.3 - left_m[..., 2], min=0.0).square().mean()
        right_barrier = torch.clamp(0.3 - right_m[..., 2], min=0.0).square().mean()
        camera_barrier_loss = 0.5 * (left_barrier + right_barrier)
        left_m = torch.cat([left_m[..., :2], torch.clamp(left_m[..., 2:], min=0.05)], dim=-1)
        right_m = torch.cat([right_m[..., :2], torch.clamp(right_m[..., 2:], min=0.05)], dim=-1)
        proj_l = fisheye_project_torch(left_m, cal.K0, cal.D0)
        proj_r = fisheye_project_torch(right_m, cal.K1, cal.D1)
        residual_l = torch.linalg.vector_norm(proj_l - kp_left, dim=2) * kp_valid_l
        residual_r = torch.linalg.vector_norm(proj_r - kp_right, dim=2) * kp_valid_r
        scale_px2 = args.gm_scale_px * args.gm_scale_px
        loss_l = torch.sum(w2d_left * (residual_l * residual_l) / (residual_l * residual_l + scale_px2)) / torch.clamp(w2d_left.sum(), min=1.0)
        loss_r = torch.sum(w2d_right * (residual_r * residual_r) / (residual_r * residual_r + scale_px2)) / torch.clamp(w2d_right.sum(), min=1.0)
        data2d_loss = 0.5 * (loss_l + loss_r)

        acceleration = joints[2:] - 2.0 * joints[1:-1] + joints[:-2]
        temporal_loss = torch.mean(acceleration.square())

        ankle_velocity = joints[1:, ankle_positions] - joints[:-1, ankle_positions]
        foot_static_pairs = foot_static[1:] * foot_static[:-1]
        foot_static_loss = torch.sum(foot_static_pairs * ankle_velocity.square().sum(dim=-1)) / torch.clamp(
            foot_static_pairs.sum(), min=1.0
        )

        ankle_z = joints[:, ankle_positions][..., 2] * 1000.0
        penetration = torch.clamp(ground_floor_mm - ankle_z, min=0.0)
        ground_loss = torch.sum(ground_mask * penetration.square()) / torch.clamp(
            ground_mask.sum() * window_size, min=1.0
        )

        wrist = joints[:, wrist_positions]
        hand_residual = torch.linalg.vector_norm(wrist - hand_target, dim=2)
        hand_loss = torch.sum(hand_valid_t * hand_residual.square()) / torch.clamp(hand_valid_t.sum(), min=1.0)

        # Soft 3D anchor to the fixed prefit (scheme 2d_plus_soft3d only): keeps
        # the skeleton from collapsing while the 2D term remains the primary
        # observation. Mirrors scheme C of the SMPL-X comparison.
        prefit_target = torch.as_tensor(target_body_mm / 1000.0, dtype=torch.float32, device=device)
        soft3d_squared = (joints - prefit_target).square()
        soft3d_loss = torch.sum(soft3d_mask * soft3d_squared) / torch.clamp(
            soft3d_mask.sum() * window_size * 3.0, min=1.0
        )

        bone_prior = torch.mean((skeleton.bone_lengths - bone_init).square())
        # Soft clamp: discourage bones drifting more than 50% from the init.
        bone_drift = torch.clamp((skeleton.bone_lengths / bone_init - 1.0).abs() - 0.5, min=0.0)
        bone_clamp = torch.mean(bone_drift.square())
        rotation_prior_squared = (skeleton.rotations - rotation_init).square()
        rotation_prior = torch.sum(rotation_prior_mask * rotation_prior_squared) / torch.clamp(
            rotation_prior_mask.sum() * window_size * 3.0, min=1.0
        )

        hinge_prior, hinge_frame_penalty = hinge_soft_prior(
            joints,
            max_hinge_rotation_rad=args.max_hinge_rotation_rad,
            min_segment_angle_deg=args.min_segment_angle_deg,
        )
        hinge_rotation_cap = torch.clamp(
            torch.linalg.vector_norm(skeleton.rotations[:, HINGE_PARAM_INDICES], dim=2)
            - args.max_hinge_rotation_rad,
            min=0.0,
        ).square().mean()

        loss = (
            args.data2d_weight * data2d_loss
            + 50.0 * camera_barrier_loss
            + args.hinge_prior_weight * (hinge_prior + hinge_rotation_cap)
            + args.rotation_prior_weight * rotation_prior
        )
        if optimize_bones:
            loss = loss + args.bone_prior_weight * (bone_prior + 5.0 * bone_clamp)
        if iteration >= args.phase_split:
            loss = loss + (
                args.temporal_weight * temporal_loss
                + args.foot_static_weight * foot_static_loss
                + args.ground_weight * ground_loss
                + args.hand_weight * hand_loss
            )
        loss = loss + (args.soft3d_weight * soft3d_loss if args.scheme == "2d_plus_soft3d" else 0.0)
        loss.backward()
        optimizer.step()
        if iteration == args.phase_split - 1:
            optimizer = optimizer_phase2
        if iteration == args.iterations - 1:
            final_terms = {
                "total": float(loss.detach().cpu()),
                "data2d": float(data2d_loss.detach().cpu()),
                "camera_barrier": float(camera_barrier_loss.detach().cpu()),
                "temporal_m2": float(temporal_loss.detach().cpu()),
                "foot_static_m2": float(foot_static_loss.detach().cpu()),
                "ground_m2": float(ground_loss.detach().cpu()),
                "hand_m2": float(hand_loss.detach().cpu()),
                "soft3d_m2": float(soft3d_loss.detach().cpu()),
                "rotation_prior": float(rotation_prior.detach().cpu()),
                "hinge_prior": float(hinge_prior.detach().cpu()),
                "hinge_rotation_cap": float(hinge_rotation_cap.detach().cpu()),
            }
            if optimize_bones:
                final_terms["bone_prior"] = float(bone_prior.detach().cpu())
                final_terms["bone_clamp"] = float(bone_clamp.detach().cpu())
            hinge_frame_penalty_np = hinge_frame_penalty.detach().cpu().numpy()
            hinge_rot_norm = torch.linalg.vector_norm(
                skeleton.rotations[:, HINGE_PARAM_INDICES], dim=2
            ).detach().cpu().numpy()
            hinge_triggered_local = np.flatnonzero(
                (hinge_frame_penalty_np > 1e-6) | (hinge_rot_norm > args.max_hinge_rotation_rad).any(axis=1)
            )
            final_terms["hinge_triggered_frames"] = [int(indices[index]) for index in hinge_triggered_local]
        if iteration % 50 == 0 or iteration == args.iterations - 1:
            with torch.no_grad():
                trace_reproj = np.nanmedian(np.concatenate([
                    torch.linalg.vector_norm(proj_l - kp_left, dim=2).cpu().numpy()[kp_valid_l.cpu().numpy() > 0],
                    torch.linalg.vector_norm(proj_r - kp_right, dim=2).cpu().numpy()[kp_valid_r.cpu().numpy() > 0],
                ]))
            print(f"iter={iteration:4d} total={loss.item():9.4f} data2d={data2d_loss.item():8.4f} "
                  f"temporal={temporal_loss.item():10.6f} soft3d={soft3d_loss.item():10.6f} "
                  f"reproj_med_px={trace_reproj:7.2f} bones={skeleton.bone_lengths.detach().cpu().numpy().round(1).tolist()}",
                  flush=True)
    elapsed = time.perf_counter() - started

    with torch.no_grad():
        fitted = skeleton.forward().cpu().numpy() * 1000.0  # (N,12,3) mm world
    deviations_mm = np.linalg.norm(fitted - target_body_mm, axis=2)

    # Reprojection of the final fit.
    reproj_l = np.full((window_size, 12), np.nan)
    reproj_r = np.full_like(reproj_l, np.nan)
    for local in range(window_size):
        left_pts = np.asarray([
            np.asarray(per_frame_rotation[local]).T @ (fitted[local, position] - np.asarray(per_frame_translation[local]))
            for position in range(12)
        ])
        right_pts = (cal.R_cam0_to_cam1 @ left_pts.T + cal.T_cam0_to_cam1_mm.reshape(3, 1)).T
        valid_l = obs2d["valid_left"][local]
        valid_r = obs2d["valid_right"][local]
        if valid_l.any():
            proj_l = _project_np(left_pts, cal.K0, cal.D0)
            reproj_l[local, valid_l] = np.linalg.norm(proj_l[valid_l] - obs2d["kp_left_px"][local][valid_l], axis=1)
        if valid_r.any():
            proj_r = _project_np(right_pts, cal.K1, cal.D1)
            reproj_r[local, valid_r] = np.linalg.norm(proj_r[valid_r] - obs2d["kp_right_px"][local][valid_r], axis=1)

    pelvis_net_input = float(np.linalg.norm(
        0.5 * (prefit_points_mm[indices[-1], [11, 12]].sum(axis=0) - prefit_points_mm[indices[0], [11, 12]].sum(axis=0))
    ))
    fitted_hips = 0.5 * (fitted[:, 6] + fitted[:, 7])  # COCO 11,12 positions 6,7
    pelvis_net_fitted = float(np.linalg.norm(fitted_hips[-1] - fitted_hips[0]))

    active_foot = contact_weights[indices][:, 15:17] > 0
    prefit_ankles = prefit_points_mm[indices][:, [15, 16]]
    fitted_ankles = fitted[:, [10, 11]]
    prefit_foot_vel = np.linalg.norm(prefit_ankles[1:] - prefit_ankles[:-1], axis=2)
    fitted_foot_vel = np.linalg.norm(fitted_ankles[1:] - fitted_ankles[:-1], axis=2)
    active_pairs = active_foot[1:]
    prefit_foot_vel_active = prefit_foot_vel[active_pairs]
    fitted_foot_vel_active = fitted_foot_vel[active_pairs]

    hand_input_distances = []
    hand_fitted_distances = []
    for local, index in enumerate(indices):
        for side, side_index in (("left", 0), ("right", 1)):
            hand = interactions[index]["hands"][side]
            proxy = np.asarray(hand["closest_proxy_point_world_mm"], dtype=np.float64)
            hand_input_distances.append(float(hand["wrist_to_handle_axis_distance_mm"]))
            hand_fitted_distances.append(float(np.linalg.norm(fitted[local, wrist_positions[side_index]] - proxy)))

    prefit_acceleration = acceleration_p95_mm(target_body_mm / 1000.0)
    fitted_acceleration = acceleration_p95_mm(fitted / 1000.0)
    metrics = {
        "joint_residual_median_mm": float(np.median(deviations_mm)),
        "joint_residual_p95_mm": float(np.percentile(deviations_mm, 95)),
        "prefit_body_acceleration_p95_mm_per_frame2": prefit_acceleration,
        "skeleton_body_acceleration_p95_mm_per_frame2": fitted_acceleration,
        "foot_contact_velocity_input_p95_mm_per_frame": (
            float(np.percentile(prefit_foot_vel_active, 95)) if prefit_foot_vel_active.size else None
        ),
        "foot_contact_velocity_fitted_p95_mm_per_frame": (
            float(np.percentile(fitted_foot_vel_active, 95)) if fitted_foot_vel_active.size else None
        ),
        "hand_handle_distance_input_median_mm": float(np.median(hand_input_distances)),
        "hand_handle_distance_fitted_median_mm": float(np.median(hand_fitted_distances)),
        "reprojection_left_median_px": float(np.nanmedian(reproj_l)),
        "reprojection_left_p95_px": float(np.nanpercentile(reproj_l, 95)),
        "reprojection_right_median_px": float(np.nanmedian(reproj_r)),
        "reprojection_right_p95_px": float(np.nanpercentile(reproj_r, 95)),
        "pelvis_net_displacement_input_mm": pelvis_net_input,
        "pelvis_net_displacement_fitted_mm": pelvis_net_fitted,
        "pelvis_net_displacement_change_fraction": float(
            abs(pelvis_net_fitted - pelvis_net_input) / max(pelvis_net_input, 1e-6)
        ),
        "valid_2d_observations_left": int(np.count_nonzero(obs2d["valid_left"])),
        "valid_2d_observations_right": int(np.count_nonzero(obs2d["valid_right"])),
    }
    hold_out_rows = []
    joint_names = {
        5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
        9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
        13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle",
    }
    for position in hold_out_positions:
        coco_index = int(BODY_JOINTS[position])
        hold_out_rows.append({
            "joint_index": coco_index,
            "joint_name": joint_names[coco_index],
            "deviation_median_mm": float(np.median(deviations_mm[:, position])),
            "deviation_p95_mm": float(np.percentile(deviations_mm[:, position], 95)),
            "reprojection_left_median_px": float(np.nanmedian(reproj_l[:, position])),
            "reprojection_right_median_px": float(np.nanmedian(reproj_r[:, position])),
        })
    metrics["hold_out"] = hold_out_rows

    gates = {
        "joint_residual_median_at_most_40_mm": metrics["joint_residual_median_mm"] <= 40.0,
        "joint_residual_p95_at_most_75_mm": metrics["joint_residual_p95_mm"] <= 75.0,
        "left_reprojection_median_at_most_20_px": metrics["reprojection_left_median_px"] <= 20.0,
        "right_reprojection_median_at_most_20_px": metrics["reprojection_right_median_px"] <= 20.0,
        "left_reprojection_p95_at_most_50_px": metrics["reprojection_left_p95_px"] <= 50.0,
        "right_reprojection_p95_at_most_50_px": metrics["reprojection_right_p95_px"] <= 50.0,
        "acceleration_at_most_1_2x_prefit": (
            metrics["skeleton_body_acceleration_p95_mm_per_frame2"]
            <= 1.2 * metrics["prefit_body_acceleration_p95_mm_per_frame2"]
        ),
        "foot_contact_velocity_fitted_at_most_2x_input": (
            metrics["foot_contact_velocity_input_p95_mm_per_frame"] is not None
            and metrics["foot_contact_velocity_fitted_p95_mm_per_frame"] is not None
            and metrics["foot_contact_velocity_fitted_p95_mm_per_frame"]
            <= 2.0 * metrics["foot_contact_velocity_input_p95_mm_per_frame"] + 1e-9
        ),
        "hand_handle_distance_fitted_within_30mm_of_input": (
            metrics["hand_handle_distance_fitted_median_mm"]
            <= metrics["hand_handle_distance_input_median_mm"] + 30.0
        ),
        "pelvis_net_displacement_change_at_most_25_percent": (
            metrics["pelvis_net_displacement_change_fraction"] <= 0.25
        ),
    }

    rows_out = []
    for local, frame_index in enumerate(indices):
        body_joints = np.full((17, 3), np.nan)
        body_joints[body_indices] = fitted[local]
        rows_out.append({
            "frame_index": int(frame_index),
            "pair_id": int(frames[frame_index]["pair"]),
            "stage": frames[frame_index]["stage"],
            "body_joints_ground_mm": body_joints.round(4).tolist(),
            "reprojection_left_px": np.nan_to_num(reproj_l[local], nan=-1.0).round(3).tolist(),
            "reprojection_right_px": np.nan_to_num(reproj_r[local], nan=-1.0).round(3).tolist(),
        })
    (output / "skeleton_window_joints.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows_out), encoding="utf-8"
    )

    summary = {
        "schema_version": "skeleton_factor_graph_window_fit_v1",
        "status": "bounded_window_engineering_candidate_not_full_sequence",
        "scheme": f"SKELE_FIXED_BONE_{args.scheme.upper()}",
        "inclusive_frame_window": [args.start_frame, args.end_frame],
        "pair_window": [int(window_frames[0]["pair"]), int(window_frames[-1]["pair"])],
        "frame_count": window_size,
        "stage_counts": {
            stage: sum(frame["stage"] == stage for frame in window_frames)
            for stage in sorted({frame["stage"] for frame in window_frames})
        },
        "iterations": args.iterations, "device": args.device, "elapsed_seconds": elapsed,
        "optimizer": args.optimizer, "learning_rate": args.learning_rate,
        "phase_split": args.phase_split, "phase2_learning_rate": args.phase2_learning_rate,
        "bone_lengths_initial_mm": bone_lengths_init.round(3).tolist(),
        "bone_names": DEFAULT_BONE_NAMES,
        "bone_lengths_fitted_mm": (skeleton.bone_lengths.detach().cpu().numpy() * 1000.0).round(3).tolist(),
        "bone_lengths_frozen": not optimize_bones,
        "frozen_bones_source": None if args.frozen_bones_json is None else str(args.frozen_bones_json.resolve()),
        "bone_lengths_changed_from_frozen": (
            None if frozen_bones is None
            else float(np.max(np.abs(skeleton.bone_lengths.detach().cpu().numpy() * 1000.0 - frozen_bones)))
        ),
        "observation_partition": {
            "left": {
                "strict": int(np.count_nonzero(obs2d["accepted_left"] & obs2d["strict_left"])),
                "relaxed": int(np.count_nonzero(obs2d["accepted_left"] & ~obs2d["strict_left"])),
                "rejected": int(np.count_nonzero(~obs2d["accepted_left"])),
                "reasons": obs2d["rejection_counts_left"],
            },
            "right": {
                "strict": int(np.count_nonzero(obs2d["accepted_right"] & obs2d["strict_right"])),
                "relaxed": int(np.count_nonzero(obs2d["accepted_right"] & ~obs2d["strict_right"])),
                "rejected": int(np.count_nonzero(~obs2d["accepted_right"])),
                "reasons": obs2d["rejection_counts_right"],
            },
        },
        "hinge_prior": {
            "weight": args.hinge_prior_weight,
            "max_hinge_rotation_rad": args.max_hinge_rotation_rad,
            "min_segment_angle_deg": args.min_segment_angle_deg,
        },
        "hold_out_joints": sorted(hold_out),
        "weights": {
            "scheme": args.scheme, "data2d": args.data2d_weight, "soft3d": args.soft3d_weight,
            "temporal": args.temporal_weight, "foot_static": args.foot_static_weight,
            "ground": args.ground_weight, "hand": args.hand_weight,
            "bone_prior": args.bone_prior_weight, "rotation_prior": args.rotation_prior_weight,
            "hinge_prior": args.hinge_prior_weight,
            "gm_scale_px": args.gm_scale_px,
        },
        "loss_terms": final_terms,
        **metrics,
        "acceptance_audit": {
            "status": "passed" if all(gates.values()) else "failed",
            "gates": gates,
            "interpretation": (
                "Internal engineering gates on the same short window. The skeleton is a "
                "lightweight real-time mainline candidate: fixed shared bone lengths, "
                "L/R 2D reprojection as the primary observation, no mesh and no VPoser."
            ),
        },
        "boundary": (
            "Short offline window. Skeleton is not a full anatomical rig; head/neck/feet "
            "surface are not modelled; bone lengths are fixed across the window by design."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _project_np(points_camera_mm: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    from pose_app.fisheye_camera import fisheye_project_numpy
    return fisheye_project_numpy(points_camera_mm, K, D)


if __name__ == "__main__":
    raise SystemExit(main())
