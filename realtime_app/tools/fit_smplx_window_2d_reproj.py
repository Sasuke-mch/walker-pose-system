#!/usr/bin/env python3
"""Fit one SMPL-X/VPoser window against L/R raw-fisheye 2D reprojection (schemes A/B/C).

Schemes (all share the same window, initialization, shared betas and iteration
budget; the only difference is the observation term):
    A  current 3D-only fitting (legacy; with --apply-offset it uses the COCO
       joint-offset observation model in the 3D residual).
    B  direct L/R raw-fisheye 2D reprojection only, with the body-local joint
       offset observation model and a robust kernel on the direct pixel
       residual.
    C  L/R 2D reprojection plus a robust soft 3D constraint (both through the
       offset observation model).

The 2D term stays in pixels. It does not use a depth copied from the 3D target;
its bounded robust loss is balanced against the other terms by an explicit
weight.

Hold-out joints (--hold-out-joints) are excluded from every data term and
reported separately as a predictive check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from pose_app.fisheye_camera import (  # noqa: E402
    fisheye_project_torch,
    load_stereo_fisheye,
    world_to_cameras_torch_batch,
)
from pose_app.joint_offset_model import JointOffsetModel  # noqa: E402
from pose_app.reprojection_common import (  # noqa: E402
    acceleration_p95_mm,
    build_2d_observations,
    build_view_2d_weights_from_obs,
    robust_gm,
)
from pose_app.smplx_fitting import (  # noqa: E402
    BODY_COCO_INDICES,
    COCO17_TO_SMPLX,
    initial_world_from_model_rotation,
    load_vposer_explicit,
    observation_weights,
    read_jsonl,
)
from pose_app.temporal_body_prefit import (  # noqa: E402
    TemporalBodyPrefitConfig,
    build_contact_targets,
)


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.7f} {vertex[1]:.7f} {vertex[2]:.7f}\n")
        for face in faces + 1:
            handle.write(f"f {face[0]} {face[1]} {face[2]}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--vposer-dir", type=Path, required=True)
    parser.add_argument("--motion-json", type=Path, required=True)
    parser.add_argument("--prefit-jsonl", type=Path, required=True)
    parser.add_argument("--interaction-jsonl", type=Path, required=True)
    parser.add_argument("--initial-fit-summary", type=Path, required=True)
    parser.add_argument("--stereo-jsonl", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--offset-audit-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=60)
    parser.add_argument("--end-frame", type=int, default=90)
    parser.add_argument("--scheme", choices=("A", "B", "C"), required=True)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--data3d-weight", type=float, default=120.0)
    parser.add_argument("--data2d-weight", type=float, default=40.0)
    parser.add_argument("--temporal-weight", type=float, default=220.0)
    parser.add_argument("--foot-contact-weight", type=float, default=100.0)
    parser.add_argument("--hand-contact-weight", type=float, default=30.0)
    parser.add_argument("--apply-offset", action="store_true",
                        help="apply the body-local COCO observation offset in the 3D term (scheme A)")
    parser.add_argument("--robust-kernel", choices=("none", "gm"), default="gm",
                        help="robust kernel for data terms (gm = Geman-McClure, default)")
    parser.add_argument("--gm-scale-px", type=float, default=20.0,
                        help="Geman-McClure scale for the direct pixel residual")
    parser.add_argument("--hold-out-joints", type=str, default="",
                        help="comma-separated COCO body joint indices excluded from all data terms")
    parser.add_argument(
        "--fixed-betas-json", type=Path,
        help="shape audit JSON containing beta_statistics.coordinatewise_median_shared_beta.",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)

    import smplx
    import torch
    from human_body_prior.tools.rotation_tools import matrot2aa

    motion = json.loads(args.motion_json.read_text(encoding="utf-8-sig"))
    prefit = read_jsonl(args.prefit_jsonl)
    interactions = read_jsonl(args.interaction_jsonl)
    initial_fit = json.loads(args.initial_fit_summary.read_text(encoding="utf-8"))
    stereo_rows = read_jsonl(args.stereo_jsonl)
    stereo_by_pair = {int(row["pair_id"]): row for row in stereo_rows}
    cal = load_stereo_fisheye(args.calibration_dir)
    offset_model = JointOffsetModel.from_audit_json(args.offset_audit_json)

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
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    hold_out = {int(part) for part in args.hold_out_joints.split(",") if part.strip()}
    if not hold_out.issubset(set(BODY_COCO_INDICES)):
        raise ValueError(f"hold-out joints must be body joints {list(BODY_COCO_INDICES)}")

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
    contact_targets_mm, contact_weights, contact_audit = build_contact_targets(
        raw_points, frames, interactions, motion["walker"], TemporalBodyPrefitConfig()
    )

    body_model = smplx.create(
        str(args.model_root.resolve()), model_type="smplx", gender="male",
        ext="npz", use_pca=False, batch_size=window_size,
    ).to(device)
    vposer, _, checkpoint_path = load_vposer_explicit(args.vposer_dir, args.device)
    zero = smplx.create(
        str(args.model_root.resolve()), model_type="smplx", gender="male",
        ext="npz", use_pca=False, batch_size=1,
    ).to(device)()
    zero_pelvis = zero.joints[0, 0].detach().cpu().numpy()

    rotations = np.stack([initial_world_from_model_rotation(prefit_points_mm[index]) for index in indices])
    rotation_tensor = torch.as_tensor(rotations, dtype=torch.float32, device=device)
    root_orient = torch.nn.Parameter(matrot2aa(rotation_tensor).reshape(window_size, 3))
    translation_init = []
    for local, index in enumerate(indices):
        hip_mid = prefit_points_mm[index, [11, 12]].mean(axis=0) / 1000.0
        translation_init.append(hip_mid - rotations[local] @ zero_pelvis)
    translation = torch.nn.Parameter(
        torch.as_tensor(np.asarray(translation_init), dtype=torch.float32, device=device)
    )
    initial_latent = np.asarray(initial_fit["parameters"]["vposer_latent"], dtype=np.float32)
    latent = torch.nn.Parameter(torch.as_tensor(np.tile(initial_latent, (window_size, 1)), device=device))
    if args.fixed_betas_json is None:
        initial_betas = np.asarray(initial_fit["parameters"]["betas"], dtype=np.float32)
        betas = torch.nn.Parameter(torch.as_tensor(initial_betas, device=device).reshape(1, 10))
        shape_mode = "optimized_within_window"
    else:
        beta_record = json.loads(args.fixed_betas_json.read_text(encoding="utf-8"))
        initial_betas = np.asarray(
            beta_record["beta_statistics"]["coordinatewise_median_shared_beta"], dtype=np.float32
        )
        betas = torch.as_tensor(initial_betas, device=device).reshape(1, 10)
        shape_mode = "fixed_cross_clip_coordinatewise_median"

    coco_indices = np.asarray(BODY_COCO_INDICES, dtype=np.int64)
    smplx_indices = np.asarray([COCO17_TO_SMPLX[index] for index in BODY_COCO_INDICES], dtype=np.int64)
    target = torch.as_tensor(
        prefit_points_mm[indices][:, coco_indices] / 1000.0, dtype=torch.float32, device=device
    )
    weights_np = np.stack([
        observation_weights(errors[index], strict[index])[coco_indices] for index in indices
    ])
    if hold_out:
        for position, coco_index in enumerate(BODY_COCO_INDICES):
            if coco_index in hold_out:
                weights_np[:, position] = 0.0
    weights = torch.as_tensor(weights_np, dtype=torch.float32, device=device)
    contacts = torch.as_tensor(contact_targets_mm[indices] / 1000.0, dtype=torch.float32, device=device)
    contact_mask = torch.as_tensor(contact_weights[indices], dtype=torch.float32, device=device)
    for coco_index in hold_out:
        contact_mask[:, coco_index] = 0.0
    all_smplx_indices = torch.as_tensor(
        [COCO17_TO_SMPLX[index] for index in range(17)], dtype=torch.long, device=device
    )

    # 2D observations (body joints only), raw fisheye pixels, per-view partition.
    obs2d = build_2d_observations(
        stereo_by_pair, frames, indices, coco_indices, cal.image_width, cal.image_height
    )
    kp_left = torch.as_tensor(np.nan_to_num(obs2d["kp_left_px"]), dtype=torch.float32, device=device)
    kp_right = torch.as_tensor(np.nan_to_num(obs2d["kp_right_px"]), dtype=torch.float32, device=device)
    w2d_left_np, w2d_right_np = build_view_2d_weights_from_obs(obs2d)
    if hold_out:
        for position, coco_index in enumerate(BODY_COCO_INDICES):
            if coco_index in hold_out:
                w2d_left_np[:, position] = 0.0
                w2d_right_np[:, position] = 0.0
    w2d_left = torch.as_tensor(w2d_left_np, dtype=torch.float32, device=device)
    w2d_right = torch.as_tensor(w2d_right_np, dtype=torch.float32, device=device)

    # Mask for 2D terms: weight zero where the keypoint is missing.
    kp_valid_l = torch.as_tensor(obs2d["valid_left"], dtype=torch.float32, device=device)
    kp_valid_r = torch.as_tensor(obs2d["valid_right"], dtype=torch.float32, device=device)

    per_frame_rotation = [np.asarray(frames[index]["rotation_world_from_left"], dtype=np.float64) for index in indices]
    per_frame_translation = [np.asarray(frames[index]["translation_world_from_left_mm"], dtype=np.float64) for index in indices]
    scale_m = 0.03
    optimized_parameters = [root_orient, translation, latent]
    if isinstance(betas, torch.nn.Parameter):
        optimized_parameters.append(betas)
    optimizer = torch.optim.Adam(optimized_parameters, lr=0.02)
    final_terms = {}
    started = time.perf_counter()
    for iteration in range(args.iterations):
        optimizer.zero_grad()
        body_pose = vposer.decode(latent)["pose_body"].reshape(window_size, -1)
        result = body_model(
            betas=betas.expand(window_size, -1), body_pose=body_pose,
            global_orient=root_orient, transl=translation, return_verts=True,
        )
        predicted_body = result.joints[:, smplx_indices]  # (N,12,3) world m
        predicted_coco = offset_model.predict_coco_torch(result.joints[:, all_smplx_indices])
        predicted_coco_body = predicted_coco[:, coco_indices]  # (N,12,3)

        data3d_loss = None
        data2d_loss = None
        if args.scheme in ("A", "C"):
            residual3d = predicted_coco_body - target
            if args.robust_kernel == "gm":
                data3d_loss = torch.sum(weights * robust_gm(residual3d, scale_m)) / torch.clamp(weights.sum(), min=1.0)
            else:
                data3d_loss = torch.sum(weights * residual3d.square().sum(dim=-1)) / torch.clamp(weights.sum(), min=1.0)
        if args.scheme in ("B", "C"):
            left_m, right_m = world_to_cameras_torch_batch(
                predicted_coco_body, per_frame_rotation, per_frame_translation, cal
            )
            proj_l = fisheye_project_torch(left_m, cal.K0, cal.D0)
            proj_r = fisheye_project_torch(right_m, cal.K1, cal.D1)
            # Note: fisheye_project_torch divides X,Y by Z; X/Y/Z share units (metres),
            # so the projected pixel coordinates are unit-consistent.
            residual_l = torch.linalg.vector_norm(proj_l - kp_left, dim=2) * kp_valid_l
            residual_r = torch.linalg.vector_norm(proj_r - kp_right, dim=2) * kp_valid_r
            if args.robust_kernel == "gm":
                scale_px2 = args.gm_scale_px * args.gm_scale_px
                loss_l = torch.sum(w2d_left * (residual_l * residual_l) / (residual_l * residual_l + scale_px2)) / torch.clamp(w2d_left.sum(), min=1.0)
                loss_r = torch.sum(w2d_right * (residual_r * residual_r) / (residual_r * residual_r + scale_px2)) / torch.clamp(w2d_right.sum(), min=1.0)
            else:
                loss_l = torch.sum(w2d_left * residual_l.square()) / torch.clamp(w2d_left.sum(), min=1.0)
                loss_r = torch.sum(w2d_right * residual_r.square()) / torch.clamp(w2d_right.sum(), min=1.0)
            data2d_loss = 0.5 * (loss_l + loss_r)

        acceleration = predicted_body[2:] - 2.0 * predicted_body[1:-1] + predicted_body[:-2]
        temporal_loss = torch.mean(acceleration.square())
        predicted_all = result.joints[:, all_smplx_indices]
        contact_residual = torch.linalg.vector_norm(predicted_all - contacts, dim=2)
        foot_mask = torch.zeros_like(contact_mask)
        foot_mask[:, 15:17] = contact_mask[:, 15:17]
        hand_mask = torch.zeros_like(contact_mask)
        hand_mask[:, 9:11] = contact_mask[:, 9:11]
        foot_loss = torch.sum(foot_mask * contact_residual.square()) / torch.clamp(foot_mask.sum(), min=1.0)
        hand_loss = torch.sum(hand_mask * contact_residual.square()) / torch.clamp(hand_mask.sum(), min=1.0)
        latent_loss = torch.mean(latent.square())
        beta_loss = torch.mean(betas.square()) if isinstance(betas, torch.nn.Parameter) else torch.zeros((), device=device)

        loss = (
            args.temporal_weight * temporal_loss
            + 0.02 * latent_loss
            + 0.04 * beta_loss
            + args.foot_contact_weight * foot_loss
            + args.hand_contact_weight * hand_loss
        )
        if data3d_loss is not None:
            loss = loss + args.data3d_weight * data3d_loss
        if data2d_loss is not None:
            loss = loss + args.data2d_weight * data2d_loss
        loss.backward()
        optimizer.step()
        if iteration == args.iterations - 1:
            final_terms = {
                "total": float(loss.detach().cpu()),
                "temporal_m2": float(temporal_loss.detach().cpu()),
                "foot_contact_m2": float(foot_loss.detach().cpu()),
                "hand_contact_m2": float(hand_loss.detach().cpu()),
                "latent": float(latent_loss.detach().cpu()),
                "betas": float(beta_loss.detach().cpu()),
            }
            if data3d_loss is not None:
                final_terms["data3d"] = float(data3d_loss.detach().cpu())
            if data2d_loss is not None:
                final_terms["data2d"] = float(data2d_loss.detach().cpu())
    elapsed = time.perf_counter() - started

    with torch.no_grad():
        body_pose = vposer.decode(latent)["pose_body"].reshape(window_size, -1)
        result = body_model(
            betas=betas.expand(window_size, -1), body_pose=body_pose,
            global_orient=root_orient, transl=translation, return_verts=True,
        )
    fitted_all_m = result.joints[:, all_smplx_indices].cpu().numpy()
    fitted_body_m = result.joints[:, smplx_indices].cpu().numpy()
    predicted_coco = offset_model.predict_coco_numpy(fitted_all_m * 1000.0) / 1000.0
    predicted_coco_body = predicted_coco[:, coco_indices]
    target_body_m = prefit_points_mm[indices][:, coco_indices] / 1000.0
    deviations_mm = np.linalg.norm(predicted_coco_body - target_body_m, axis=2) * 1000.0
    active_contacts = contact_weights[indices] > 0
    before_contact = np.linalg.norm(prefit_points_mm[indices] - contact_targets_mm[indices], axis=2)[active_contacts]
    after_contact = np.linalg.norm(fitted_all_m * 1000.0 - contact_targets_mm[indices], axis=2)[active_contacts]

    # 2D reprojection metrics on the final fit.
    reproj_l = np.full((window_size, len(coco_indices)), np.nan)
    reproj_r = np.full_like(reproj_l, np.nan)
    pelvis_net_input = float(np.linalg.norm(
        0.5 * (prefit_points_mm[indices[-1], [11, 12]].sum(axis=0) - prefit_points_mm[indices[0], [11, 12]].sum(axis=0))
    ))
    fitted_pelvis = 0.5 * (predicted_coco[:, 11] + predicted_coco[:, 12])
    pelvis_net_fitted = float(np.linalg.norm(fitted_pelvis[-1] - fitted_pelvis[0]) * 1000.0)
    predicted_coco_mm = predicted_coco * 1000.0
    for local in range(window_size):
        left_pts = np.asarray([
            np.asarray(per_frame_rotation[local]).T @ (predicted_coco_mm[local, joint] - np.asarray(per_frame_translation[local]))
            for joint in coco_indices
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

    reproj_l_flat = reproj_l[np.isfinite(reproj_l)]
    reproj_r_flat = reproj_r[np.isfinite(reproj_r)]
    hold_out_rows = []
    if hold_out:
        for position, coco_index in enumerate(BODY_COCO_INDICES):
            if coco_index not in hold_out:
                continue
            hold_out_rows.append({
                "joint_index": int(coco_index),
                "joint_name": {
                    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
                    9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
                    13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle",
                }[int(coco_index)],
                "deviation_median_mm": float(np.median(deviations_mm[:, position])),
                "deviation_p95_mm": float(np.percentile(deviations_mm[:, position], 95)),
                "reprojection_left_median_px": float(np.nanmedian(reproj_l[:, position])),
                "reprojection_right_median_px": float(np.nanmedian(reproj_r[:, position])),
            })

    vertices = result.vertices.cpu().numpy()
    for local_index, label in ((0, "first"), (window_size // 2, "middle"), (window_size - 1, "last")):
        write_obj(output / f"fitted_{label}.obj", vertices[local_index], body_model.faces)
    rows_out = []
    for local, frame_index in enumerate(indices):
        rows_out.append({
            "frame_index": int(frame_index),
            "pair_id": int(frames[frame_index]["pair"]),
            "stage": frames[frame_index]["stage"],
            "root_orient_axis_angle_rad": root_orient.detach().cpu().numpy()[local].round(6).tolist(),
            "smplx_coco17_joints_ground_mm": (predicted_coco[local] * 1000.0).round(4).tolist(),
            "smplx_raw17_joints_ground_mm": (fitted_all_m[local] * 1000.0).round(4).tolist(),
            "reprojection_left_px": np.nan_to_num(reproj_l[local], nan=-1.0).round(3).tolist(),
            "reprojection_right_px": np.nan_to_num(reproj_r[local], nan=-1.0).round(3).tolist(),
            "foot_contact_regularizer_active": {
                "left": bool(contact_weights[frame_index, 15] > 0),
                "right": bool(contact_weights[frame_index, 16] > 0),
            },
            "hand_contact_regularizer_active": {
                "left": bool(contact_weights[frame_index, 9] > 0),
                "right": bool(contact_weights[frame_index, 10] > 0),
            },
        })
    (output / "smplx_window_joints.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows_out), encoding="utf-8"
    )

    prefit_acceleration = acceleration_p95_mm(target_body_m)
    fitted_acceleration = acceleration_p95_mm(fitted_body_m)

    # Contact-phase foot velocity and hand-to-handle distance (input vs fitted).
    active_foot = contact_weights[indices][:, 15:17] > 0
    prefit_ankles = prefit_points_mm[indices][:, [15, 16]]
    fitted_ankles = predicted_coco_mm[:, [15, 16]]
    prefit_foot_vel = np.linalg.norm(prefit_ankles[1:] - prefit_ankles[:-1], axis=2)
    fitted_foot_vel = np.linalg.norm(fitted_ankles[1:] - fitted_ankles[:-1], axis=2)
    active_foot_vel_pairs = active_foot[1:]
    prefit_foot_vel_active = prefit_foot_vel[active_foot_vel_pairs]
    fitted_foot_vel_active = fitted_foot_vel[active_foot_vel_pairs]
    hand_input_distances = []
    hand_fitted_distances = []
    for local, index in enumerate(indices):
        record = interactions[index]
        for side in ("left", "right"):
            hand = record["hands"][side]
            proxy = np.asarray(hand["closest_proxy_point_world_mm"], dtype=np.float64)
            input_distance = float(hand["wrist_to_handle_axis_distance_mm"])
            side_index = 9 if side == "left" else 10
            fitted_distance = float(np.linalg.norm(predicted_coco_mm[local, side_index] - proxy))
            hand_input_distances.append(input_distance)
            hand_fitted_distances.append(fitted_distance)

    metrics = {
        "joint_residual_median_mm": float(np.median(deviations_mm)),
        "joint_residual_p95_mm": float(np.percentile(deviations_mm, 95)),
        "prefit_body_acceleration_p95_mm_per_frame2": prefit_acceleration,
        "smplx_body_acceleration_p95_mm_per_frame2": fitted_acceleration,
        "foot_contact_velocity_input_p95_mm_per_frame": (
            float(np.percentile(prefit_foot_vel_active, 95)) if prefit_foot_vel_active.size else None
        ),
        "foot_contact_velocity_fitted_p95_mm_per_frame": (
            float(np.percentile(fitted_foot_vel_active, 95)) if fitted_foot_vel_active.size else None
        ),
        "hand_handle_distance_input_median_mm": float(np.median(hand_input_distances)),
        "hand_handle_distance_fitted_median_mm": float(np.median(hand_fitted_distances)),
        "contact_residual_before_median_mm": float(np.median(before_contact)) if len(before_contact) else None,
        "contact_residual_after_median_mm": float(np.median(after_contact)) if len(after_contact) else None,
        "reprojection_left_median_px": float(np.median(reproj_l_flat)) if len(reproj_l_flat) else None,
        "reprojection_left_p95_px": float(np.percentile(reproj_l_flat, 95)) if len(reproj_l_flat) else None,
        "reprojection_right_median_px": float(np.median(reproj_r_flat)) if len(reproj_r_flat) else None,
        "reprojection_right_p95_px": float(np.percentile(reproj_r_flat, 95)) if len(reproj_r_flat) else None,
        "pelvis_net_displacement_input_mm": float(pelvis_net_input),
        "pelvis_net_displacement_fitted_mm": pelvis_net_fitted,
        "pelvis_net_displacement_change_fraction": float(
            abs(pelvis_net_fitted - pelvis_net_input) / max(pelvis_net_input, 1e-6)
        ),
        "valid_2d_observations_left": int(np.count_nonzero(obs2d["valid_left"])),
        "valid_2d_observations_right": int(np.count_nonzero(obs2d["valid_right"])),
        "hold_out": hold_out_rows,
    }
    gates = {
        "joint_residual_median_at_most_40_mm": metrics["joint_residual_median_mm"] <= 40.0,
        "joint_residual_p95_at_most_75_mm": metrics["joint_residual_p95_mm"] <= 75.0,
        "acceleration_not_worse_than_prefit": fitted_acceleration <= prefit_acceleration + 1e-9,
        "active_contact_residual_median_at_most_20_mm": (
            metrics["contact_residual_after_median_mm"] is not None
            and metrics["contact_residual_after_median_mm"] <= 20.0
        ),
        "left_reprojection_median_at_most_20_px": (
            metrics["reprojection_left_median_px"] is not None and metrics["reprojection_left_median_px"] <= 20.0
        ),
        "right_reprojection_median_at_most_20_px": (
            metrics["reprojection_right_median_px"] is not None and metrics["reprojection_right_median_px"] <= 20.0
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
    summary = {
        "schema_version": "smplx_window_2d_reprojection_fit_v1",
        "status": "bounded_window_engineering_candidate_not_full_sequence",
        "scheme": args.scheme,
        "inclusive_frame_window": [args.start_frame, args.end_frame],
        "pair_window": [int(window_frames[0]["pair"]), int(window_frames[-1]["pair"])],
        "frame_count": window_size,
        "stage_counts": {
            stage: sum(frame["stage"] == stage for frame in window_frames)
            for stage in sorted({frame["stage"] for frame in window_frames})
        },
        "iterations": args.iterations, "device": args.device, "elapsed_seconds": elapsed,
        "shape_mode": shape_mode,
        "apply_offset": bool(args.apply_offset) or args.scheme in ("B", "C"),
        "offset_applied_joints": offset_model.applied_joint_indices,
        "robust_kernel": args.robust_kernel,
        "gm_scale_px": args.gm_scale_px,
        "hold_out_joints": sorted(hold_out),
        "weights": {
            "data3d": args.data3d_weight, "data2d": args.data2d_weight,
            "temporal": args.temporal_weight,
            "foot_contact": args.foot_contact_weight, "hand_contact": args.hand_contact_weight,
            "latent": 0.02, "betas": 0.04,
        },
        "fixed_betas_source": None if args.fixed_betas_json is None else str(args.fixed_betas_json.resolve()),
        "vposer_checkpoint": str(checkpoint_path),
        "loss_terms": final_terms,
        **metrics,
        "acceptance_audit": {
            "status": "passed" if all(gates.values()) else "failed",
            "gates": gates,
            "interpretation": (
                "Internal engineering gates on the same short window. They test optimisation "
                "consistency and the offset observation model, not physical accuracy truth. "
                "2D reprojection uses the frozen raw-fisheye keypoints; per-joint quality "
                "combines the frozen strict flag, triangulation error and view score."
            ),
        },
        "shared_betas": betas.detach().cpu().numpy()[0].tolist(),
        "contact_audit_source": contact_audit,
        "boundary": (
            "Short offline window comparing 3D-only, 2D-only and 2D+3D observations on the "
            "frozen people_1 keypoints. It is not full-sequence, realtime, physical accuracy "
            "or contact truth."
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
