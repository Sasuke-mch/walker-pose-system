#!/usr/bin/env python3
"""Fit male SMPL with explicit triangulation and temporal-prefit trajectory losses.

This stage uses low-dimensional 5-frame knots.  Triangulated COCO points are
always present in a quality-weighted robust 3-D term and additionally define a
large-error barrier.  The temporal prefit is compared in fixed-ground space by
velocity and acceleration, so a persistent COCO/SMPL positional bias is not
mistaken for motion.

Optionally, the same fixed-ground temporal-prefit COCO-17 points can also be
used as an absolute robust 3-D fitting target.  This is disabled by default so
the original trajectory-only experiment remains reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_app.fisheye_camera import fisheye_project_torch, load_stereo_fisheye  # noqa: E402
from pose_app.smpl_coco_observation import load_coco17_regressor, load_smpl_male, regress_coco17_torch  # noqa: E402
from tools.fit_smpl_coco_single_frame import ACTIVE_POSE_COMPONENTS, RIGHT_KNEE_INDEX, huber_vector  # noqa: E402
from tools.fit_smpl_coco_temporal_guardrail import (  # noqa: E402
    load_force_all_guardrails,
    percentile,
    read_jsonl_by_pair,
    top_tail_mean,
)


def load_prefit(path: Path, pair_ids: list[int]) -> np.ndarray:
    rows: dict[int, dict] = {}
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                pair = int(row.get("pair_id", -1))
                if pair in pair_ids:
                    rows[pair] = row
    missing = [pair for pair in pair_ids if pair not in rows]
    if missing:
        raise KeyError(f"temporal prefit misses pairs: {missing}")
    values = np.asarray(
        [[rows[pair]["points_ground_mm"][str(joint)] for joint in range(17)] for pair in pair_ids],
        dtype=np.float32,
    )
    if values.shape != (len(pair_ids), 17, 3) or not np.isfinite(values).all():
        raise ValueError(f"invalid temporal prefit points: {values.shape}")
    return values / 1000.0


def load_ground_transforms(path: Path, pair_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    motion = json.loads(path.read_text(encoding="utf-8"))
    frames = {int(frame["pair"]): frame for frame in motion["frames"]}
    missing = [pair for pair in pair_ids if pair not in frames]
    if missing:
        raise KeyError(f"ground-transform template misses pairs: {missing}")
    rotations = np.asarray([frames[pair]["rotation_world_from_left"] for pair in pair_ids], dtype=np.float32)
    translations = np.asarray([frames[pair]["translation_world_from_left_mm"] for pair in pair_ids], dtype=np.float32) / 1000.0
    return rotations, translations


def knot_layout(frame_count: int, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    knots = list(range(0, frame_count, stride))
    if knots[-1] != frame_count - 1:
        knots.append(frame_count - 1)
    left = np.empty(frame_count, dtype=np.int64)
    right = np.empty(frame_count, dtype=np.int64)
    alpha = np.empty(frame_count, dtype=np.float32)
    for frame in range(frame_count):
        upper = int(np.searchsorted(knots, frame, side="right"))
        if upper == 0:
            left[frame] = right[frame] = 0
            alpha[frame] = 0.0
        elif upper == len(knots):
            left[frame] = right[frame] = len(knots) - 1
            alpha[frame] = 0.0
        else:
            lo, hi = upper - 1, upper
            left[frame], right[frame] = lo, hi
            alpha[frame] = (frame - knots[lo]) / float(knots[hi] - knots[lo])
    return np.asarray(knots, dtype=np.int64), np.stack([left, right], axis=1), alpha


def pseudo_huber(torch: "object", norm: "object", scale: float) -> "object":
    ratio = norm / scale
    return torch.sqrt(1.0 + ratio.square()) - 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent-window", type=Path, required=True)
    parser.add_argument("--force-all-jsonl", type=Path, required=True)
    parser.add_argument("--prefit-jsonl", type=Path, required=True)
    parser.add_argument("--ground-motion", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--regressor", type=Path, required=True)
    parser.add_argument("--start-pair", type=int, required=True)
    parser.add_argument("--end-pair", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--knot-stride", type=int, default=5)
    parser.add_argument(
        "--knee-anatomy",
        action="store_true",
        help="Enable a VIBE-style knee bend-direction prior and a soft non-hinge swing limit.",
    )
    parser.add_argument("--knee-angle-prior-weight", type=float, default=15.0)
    parser.add_argument("--knee-swing-weight", type=float, default=40.0)
    parser.add_argument("--knee-swing-free-rad", type=float, default=0.20)
    parser.add_argument(
        "--prefit-position-weight",
        type=float,
        default=0.0,
        help="Weight of the always-active absolute temporal-prefit COCO-17 position loss.",
    )
    parser.add_argument(
        "--prefit-position-scale-mm",
        type=float,
        default=75.0,
        help="Pseudo-Huber scale in millimetres for the absolute temporal-prefit position loss.",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "frames").mkdir()

    pair_ids = list(range(args.start_pair, args.end_pair + 1))
    frame_dirs = [args.independent_window.resolve() / "frames" / f"frame_{pair:04d}" for pair in pair_ids]
    initial = []
    for directory in frame_dirs:
        source = directory / "result.npz"
        if not source.is_file():
            raise FileNotFoundError(source)
        initial.append(np.load(source))
    roots_np = np.concatenate([item["root_orient"] for item in initial], axis=0).astype(np.float32)
    translations_np = np.concatenate([item["translation_m"] for item in initial], axis=0).astype(np.float32)
    full_pose_np = np.concatenate([item["body_pose"] for item in initial], axis=0).astype(np.float32)
    active_pose_np = full_pose_np[:, ACTIVE_POSE_COMPONENTS]
    observed_left_np = np.stack([item["observed_left"] for item in initial]).astype(np.float32)
    observed_right_np = np.stack([item["observed_right"] for item in initial]).astype(np.float32)
    valid_left_np = np.stack([item["valid_left"] for item in initial]).astype(bool)
    valid_right_np = np.stack([item["valid_right"] for item in initial]).astype(bool)
    faces = np.asarray(initial[0]["faces"], dtype=np.uint32)
    for item in initial:
        item.close()

    force_rows = read_jsonl_by_pair(args.force_all_jsonl.resolve(), pair_ids[0], pair_ids[-1])
    tri_np, _, tri_reprojection_np, tri_available_np, tri_flags = load_force_all_guardrails(force_rows, pair_ids)
    prefit_world_np = load_prefit(args.prefit_jsonl.resolve(), pair_ids)
    world_rotations_np, world_translations_np = load_ground_transforms(args.ground_motion.resolve(), pair_ids)
    knot_frames, interpolation_indices, interpolation_alpha = knot_layout(len(pair_ids), args.knot_stride)

    import torch

    device = args.device
    count = len(pair_ids)
    model = load_smpl_male(args.model_path, batch_size=count, device=device)
    regressor = load_coco17_regressor(args.regressor)
    roots_k = torch.nn.Parameter(torch.tensor(roots_np[knot_frames], device=device))
    translations_k = torch.nn.Parameter(torch.tensor(translations_np[knot_frames], device=device))
    active_pose_k = torch.nn.Parameter(torch.tensor(active_pose_np[knot_frames], device=device))
    interpolation_lr = torch.tensor(interpolation_indices, dtype=torch.long, device=device)
    alpha = torch.tensor(interpolation_alpha[:, None], device=device)
    active_indices = torch.tensor(ACTIVE_POSE_COMPONENTS, dtype=torch.long, device=device)
    betas = torch.zeros(count, 10, dtype=torch.float32, device=device)
    observed_left = torch.tensor(observed_left_np[:, :, :2], device=device)
    observed_right = torch.tensor(observed_right_np[:, :, :2], device=device)
    weight_left = torch.tensor(valid_left_np * np.square(observed_left_np[:, :, 2]), device=device)
    weight_right = torch.tensor(valid_right_np * np.square(observed_right_np[:, :, 2]), device=device)
    tri = torch.tensor(tri_np, device=device)
    tri_available = torch.tensor(tri_available_np, dtype=torch.bool, device=device)
    tri_reprojection = torch.tensor(np.nan_to_num(tri_reprojection_np, nan=1000.0), device=device)
    prefit_world = torch.tensor(prefit_world_np, device=device)
    world_rotations = torch.tensor(world_rotations_np, device=device)
    world_translations = torch.tensor(world_translations_np, device=device)
    rotation_right = torch.tensor(load_stereo_fisheye(args.calibration_dir).R_cam0_to_cam1, dtype=torch.float32, device=device)
    cal = load_stereo_fisheye(args.calibration_dir)
    translation_right = torch.tensor(cal.T_cam0_to_cam1_mm / 1000.0, dtype=torch.float32, device=device)

    def interpolate(knots: "object") -> "object":
        left = knots[interpolation_lr[:, 0]]
        right = knots[interpolation_lr[:, 1]]
        return left * (1.0 - alpha) + right * alpha

    def forward() -> tuple:
        roots = interpolate(roots_k)
        translations = interpolate(translations_k)
        active_pose = interpolate(active_pose_k)
        full_pose = torch.zeros(count, 69, dtype=active_pose.dtype, device=device).index_copy(1, active_indices, active_pose)
        result = model(betas=betas, body_pose=full_pose, global_orient=roots, transl=translations, return_verts=True)
        coco_left = regress_coco17_torch(result.vertices, regressor)
        coco_right = torch.einsum("ij,bkj->bki", rotation_right, coco_left) + translation_right
        projected_left = fisheye_project_torch(coco_left, cal.K0, cal.D0)
        projected_right = fisheye_project_torch(coco_right, cal.K1, cal.D1)
        coco_world = torch.einsum("nij,nkj->nki", world_rotations, coco_left) + world_translations[:, None, :]
        return result, roots, translations, active_pose, full_pose, coco_left, coco_right, coco_world, projected_left, projected_right

    # Absolute 3-D term is always active.  Quality affects weight and the gross
    # rejection boundary, but does not remove a finite force-all observation.
    tri_weight_np = np.where(tri_reprojection_np <= 10.0, 1.0, 0.25).astype(np.float32)
    tri_weight_np[~tri_available_np] = 0.0
    tri_weight_np[:, RIGHT_KNEE_INDEX] = np.where(tri_available_np[:, RIGHT_KNEE_INDEX], 0.10, 0.0)
    gross_threshold_np = np.where(tri_reprojection_np <= 10.0, 0.30, 0.45).astype(np.float32)
    gross_threshold_np[:, RIGHT_KNEE_INDEX] = 0.55
    tri_weight = torch.tensor(tri_weight_np, device=device)
    gross_threshold = torch.tensor(gross_threshold_np, device=device)
    trajectory_weight_np = np.ones(17, dtype=np.float32)
    trajectory_weight_np[:5] = 0.25
    trajectory_weight_np[RIGHT_KNEE_INDEX] = 0.25
    trajectory_weight = torch.tensor(trajectory_weight_np, device=device)

    def compute_losses(coco_left: "object", coco_world: "object", projected_left: "object", projected_right: "object", active_pose: "object", full_pose: "object", delta_px: float) -> dict:
        left = (huber_vector(torch, projected_left - observed_left, delta_px) * weight_left).sum()
        right = (huber_vector(torch, projected_right - observed_right, delta_px) * weight_right).sum()
        loss_2d = (left + right) / (weight_left.sum() + weight_right.sum()).clamp_min(1.0)

        distance_3d = torch.linalg.vector_norm(coco_left - tri, dim=2)
        loss_3d = (pseudo_huber(torch, distance_3d, 0.10) * tri_weight).sum() / tri_weight.sum().clamp_min(1.0)
        gross_excess = torch.where(
            tri_available,
            torch.relu(distance_3d - gross_threshold) / 0.05,
            torch.zeros_like(distance_3d),
        )
        loss_gross = top_tail_mean(torch, gross_excess.square(), fraction=0.04)

        predicted_velocity = coco_world[1:] - coco_world[:-1]
        prefit_velocity = prefit_world[1:] - prefit_world[:-1]
        velocity_error = torch.linalg.vector_norm(predicted_velocity - prefit_velocity, dim=2)
        loss_velocity = (pseudo_huber(torch, velocity_error, 0.03) * trajectory_weight[None]).mean()
        velocity_excess = torch.relu(velocity_error - 0.06) / 0.02
        loss_velocity_barrier = top_tail_mean(torch, velocity_excess.square(), fraction=0.06)

        predicted_acc = predicted_velocity[1:] - predicted_velocity[:-1]
        prefit_acc = prefit_velocity[1:] - prefit_velocity[:-1]
        acceleration_error = torch.linalg.vector_norm(predicted_acc - prefit_acc, dim=2)
        loss_acceleration = (pseudo_huber(torch, acceleration_error, 0.03) * trajectory_weight[None]).mean()
        prefit_position_error = torch.linalg.vector_norm(coco_world - prefit_world, dim=2)
        loss_prefit_position = pseudo_huber(
            torch,
            prefit_position_error,
            args.prefit_position_scale_mm / 1000.0,
        ).mean()
        loss_pose = active_pose.square().mean()
        # Standard SMPL knee components are body-pose offsets 9 and 12.  VIBE's
        # SMPLify loss applies exp(-knee_bend)^2 to discourage the anatomically
        # wrong bend direction.  It does not constrain the other two axes, so
        # add a separate soft swing tube rather than hard-locking them to zero.
        knee_bend = full_pose[:, [9, 12]]
        knee_swing = torch.stack(
            [
                torch.linalg.vector_norm(full_pose[:, 10:12], dim=1),
                torch.linalg.vector_norm(full_pose[:, 13:15], dim=1),
            ],
            dim=1,
        )
        loss_knee_angle = torch.exp(-knee_bend).square().mean()
        knee_swing_excess = torch.relu(knee_swing - args.knee_swing_free_rad) / 0.10
        loss_knee_swing = knee_swing_excess.square().mean()
        return {
            "loss_2d": loss_2d,
            "loss_3d": loss_3d,
            "loss_gross": loss_gross,
            "loss_velocity": loss_velocity,
            "loss_velocity_barrier": loss_velocity_barrier,
            "loss_acceleration": loss_acceleration,
            "loss_prefit_position": loss_prefit_position,
            "loss_pose": loss_pose,
            "loss_knee_angle": loss_knee_angle,
            "loss_knee_swing": loss_knee_swing,
            "distance_3d": distance_3d,
            "gross_excess_m": torch.where(tri_available, torch.relu(distance_3d - gross_threshold), torch.zeros_like(distance_3d)),
            "velocity_error_m": velocity_error,
            "acceleration_error_m": acceleration_error,
            "prefit_position_error_m": prefit_position_error,
            "predicted_velocity_m": torch.linalg.vector_norm(predicted_velocity, dim=2),
            "knee_bend_rad": knee_bend,
            "knee_swing_rad": knee_swing,
        }

    optimizer = torch.optim.Adam(
        [
            {"params": [roots_k, active_pose_k], "lr": 0.0030},
            {"params": [translations_k], "lr": 0.0015},
        ]
    )
    history: list[dict] = []
    for step in range(args.iterations):
        forward_values = forward()
        result, roots, translations, active_pose, full_pose, coco_left, coco_right, coco_world, projected_left, projected_right = forward_values
        delta_px = 50.0 if step < args.iterations // 3 else 20.0
        terms = compute_losses(coco_left, coco_world, projected_left, projected_right, active_pose, full_pose, delta_px)
        ramp = min(1.0, (step + 1) / max(1, args.iterations // 5))
        total = (
            terms["loss_2d"]
            + 40.0 * terms["loss_3d"]
            + 5.0 * terms["loss_pose"]
            + ramp * (
                250.0 * terms["loss_gross"]
                + 100.0 * terms["loss_velocity"]
                + 250.0 * terms["loss_velocity_barrier"]
                + 30.0 * terms["loss_acceleration"]
                + args.prefit_position_weight * terms["loss_prefit_position"]
            )
        )
        if args.knee_anatomy:
            total = total + ramp * (
                args.knee_angle_prior_weight * terms["loss_knee_angle"]
                + args.knee_swing_weight * terms["loss_knee_swing"]
            )
        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_([roots_k, active_pose_k, translations_k], 2000.0)
        optimizer.step()
        with torch.no_grad():
            active_pose_k.clamp_(-2.5, 2.5)
            translations_k[:, 0:2].clamp_(-3.0, 3.0)
            translations_k[:, 2].clamp_(0.25, 3.0)
        if step in {0, 49, 99, 199, 399, 599, args.iterations - 1}:
            history.append({
                "iteration": step + 1,
                "total": float(total.detach()),
                **{key: float(value.detach()) for key, value in terms.items() if key.startswith("loss_")},
                "max_3d_distance_mm": float(terms["distance_3d"][tri_available].max().detach() * 1000.0),
                "max_gross_excess_mm": float(terms["gross_excess_m"].max().detach() * 1000.0),
                "max_velocity_error_mm": float(terms["velocity_error_m"].max().detach() * 1000.0),
                "max_prefit_position_error_mm": float(terms["prefit_position_error_m"].max().detach() * 1000.0),
                "max_knee_swing_rad": float(terms["knee_swing_rad"].max().detach()),
            })
            print(json.dumps(history[-1]), flush=True)

    with torch.no_grad():
        result, roots, translations, active_pose, full_pose, coco_left, coco_right, coco_world, projected_left, projected_right = forward()
        terms = compute_losses(coco_left, coco_world, projected_left, projected_right, active_pose, full_pose, 20.0)
    arrays = {name: value.detach().cpu().numpy() for name, value in {
        "vertices": result.vertices, "roots": roots, "translations": translations,
        "full_pose": full_pose, "coco_left": coco_left, "coco_right": coco_right,
        "coco_world": coco_world, "projected_left": projected_left, "projected_right": projected_right,
        "distance_3d": terms["distance_3d"], "gross_excess": terms["gross_excess_m"],
        "velocity_error": terms["velocity_error_m"], "predicted_velocity": terms["predicted_velocity_m"],
        "prefit_position_error": terms["prefit_position_error_m"],
        "knee_bend": terms["knee_bend_rad"], "knee_swing": terms["knee_swing_rad"],
    }.items()}

    left_error = np.linalg.norm(arrays["projected_left"] - observed_left_np[:, :, :2], axis=2)
    right_error = np.linalg.norm(arrays["projected_right"] - observed_right_np[:, :, :2], axis=2)
    used_left, used_right = left_error[valid_left_np], right_error[valid_right_np]
    acceleration = np.linalg.norm(np.diff(arrays["coco_world"], n=2, axis=0), axis=2) * 1000.0
    joint_steps = arrays["predicted_velocity"] * 1000.0
    velocity_error_mm = arrays["velocity_error"] * 1000.0
    reliable = tri_available_np & (tri_reprojection_np <= 10.0)
    high_error = tri_available_np & ~(tri_reprojection_np <= 10.0)
    negative_left = int(np.sum(arrays["coco_left"][:, :, 2] <= 0.0))
    negative_right = int(np.sum(arrays["coco_right"][:, :, 2] <= 0.0))

    rejection_reasons = []
    max_velocity_error = float(velocity_error_mm.max())
    max_joint_step = float(joint_steps.max())
    reliable_max = float((arrays["distance_3d"] * 1000.0)[reliable].max())
    high_error_max = float((arrays["distance_3d"] * 1000.0)[high_error].max())
    if max_velocity_error > 80.0:
        rejection_reasons.append(f"prefit_velocity_error_gt_80mm:{max_velocity_error:.3f}")
    if max_joint_step > 110.0:
        rejection_reasons.append(f"model_joint_step_gt_110mm:{max_joint_step:.3f}")
    if reliable_max > 310.0:
        rejection_reasons.append(f"reliable_triangulation_distance_gt_310mm:{reliable_max:.3f}")
    if high_error_max > 460.0:
        rejection_reasons.append(f"high_error_triangulation_distance_gt_460mm:{high_error_max:.3f}")
    if negative_left or negative_right:
        rejection_reasons.append(f"negative_depth:{negative_left}/{negative_right}")
    max_knee_swing = float(arrays["knee_swing"].max())
    if args.knee_anatomy and max_knee_swing > 0.35:
        rejection_reasons.append(f"knee_nonhinge_swing_gt_0.35rad:{max_knee_swing:.6f}")
    knee_bend_p05 = np.percentile(arrays["knee_bend"], 5, axis=0)
    if args.knee_anatomy and float(knee_bend_p05.min()) < -0.25:
        rejection_reasons.append(
            "knee_reverse_bend_p05_lt_-0.25rad:"
            f"{float(knee_bend_p05[0]):.6f}/{float(knee_bend_p05[1]):.6f}"
        )

    wrist_64_65 = None
    if 64 in pair_ids and 65 in pair_ids:
        a, b = pair_ids.index(64), pair_ids.index(65)
        wrist_64_65 = float(np.linalg.norm(arrays["coco_world"][b, 9] - arrays["coco_world"][a, 9]) * 1000.0)
    wrist_65_66 = None
    if 65 in pair_ids and 66 in pair_ids:
        a, b = pair_ids.index(65), pair_ids.index(66)
        wrist_65_66 = float(np.linalg.norm(arrays["coco_world"][b, 9] - arrays["coco_world"][a, 9]) * 1000.0)

    summary = {
        "schema_version": "pmpose_smpl_coco_prefit_trajectory_v1",
        "status": "rejected_gross_solution" if rejection_reasons else "accepted_development_candidate",
        "rejection_reasons": rejection_reasons,
        "window": [pair_ids[0], pair_ids[-1]],
        "frame_count": count,
        "knot_frames_relative": knot_frames.tolist(),
        "fit_contract": {
            "model": "official male SMPL 6890; beta fixed zero",
            "model_observation": "Pose2Mesh 17x6890 COCO regressor",
            "primary_2d": "stereo PMPose COCO-17 saved in independent fit inputs",
            "absolute_3d": "all finite force-all triangulation in always-active robust loss",
            "absolute_3d_weights": {"reprojection_le_10px": 1.0, "high_reprojection": 0.25, "right_knee": 0.10},
            "gross_3d_barriers_m": {"reprojection_le_10px": 0.30, "high_reprojection": 0.45, "right_knee": 0.55},
            "temporal_reference": str(args.prefit_jsonl.resolve()),
            "temporal_comparison": "fixed-ground COCO velocity and acceleration",
            "absolute_prefit_position": {
                "enabled": bool(args.prefit_position_weight > 0.0),
                "weight": args.prefit_position_weight,
                "pseudo_huber_scale_mm": args.prefit_position_scale_mm,
                "joint_weights": "uniform over all 17 temporal-prefit COCO points",
            },
            "velocity_soft_scale_mm": 30.0,
            "velocity_barrier_mm": 60.0,
            "parameterization": f"linear temporal knots every {args.knot_stride} frames",
            "knee_anatomy": {
                "enabled": bool(args.knee_anatomy),
                "natural_bend_prior": "VIBE-style exp(-SMPL_knee_x)^2",
                "nonhinge_swing_free_rad": args.knee_swing_free_rad,
                "angle_prior_weight": args.knee_angle_prior_weight,
                "swing_weight": args.knee_swing_weight,
            },
            "excluded": ["contact loss in SMPL fit", "beta optimization", "SMPL-X", "VPoser", "display smoothing"],
        },
        "weights": {"2d": 1.0, "absolute_3d": 40.0, "gross_3d": 250.0, "prefit_position": args.prefit_position_weight, "prefit_velocity": 100.0, "velocity_barrier": 250.0, "prefit_acceleration": 30.0, "pose": 5.0, "knee_angle_prior": args.knee_angle_prior_weight if args.knee_anatomy else 0.0, "knee_swing": args.knee_swing_weight if args.knee_anatomy else 0.0},
        "reprojection": {
            "combined_median_px": percentile(np.concatenate([used_left, used_right]), 50),
            "combined_p95_px": percentile(np.concatenate([used_left, used_right]), 95),
            "left_p95_px": percentile(used_left, 95),
            "right_p95_px": percentile(used_right, 95),
        },
        "absolute_3d": {
            "available_points": int(tri_available_np.sum()),
            "high_reprojection_points_retained": int(high_error.sum()),
            "distance_median_mm": percentile((arrays["distance_3d"] * 1000.0)[tri_available_np], 50),
            "distance_p95_mm": percentile((arrays["distance_3d"] * 1000.0)[tri_available_np], 95),
            "reliable_max_mm": reliable_max,
            "high_error_max_mm": high_error_max,
            "gross_violation_count": int(np.sum(arrays["gross_excess"] > 1e-6)),
            "gross_violation_max_mm": float(arrays["gross_excess"].max() * 1000.0),
        },
        "temporal": {
            "joint_step_p95_mm": percentile(joint_steps, 95),
            "joint_step_max_mm": max_joint_step,
            "prefit_velocity_error_p95_mm": percentile(velocity_error_mm, 95),
            "prefit_velocity_error_max_mm": max_velocity_error,
            "coco_acceleration_p95_mm_per_frame2": percentile(acceleration, 95),
            "left_wrist_step_64_65_mm": wrist_64_65,
            "left_wrist_step_65_66_mm": wrist_65_66,
        },
        "absolute_prefit_position": {
            "distance_median_mm": percentile(arrays["prefit_position_error"] * 1000.0, 50),
            "distance_p95_mm": percentile(arrays["prefit_position_error"] * 1000.0, 95),
            "distance_max_mm": float(arrays["prefit_position_error"].max() * 1000.0),
        },
        "knee_anatomy": {
            "enabled": bool(args.knee_anatomy),
            "left_bend_median_rad": float(np.median(arrays["knee_bend"][:, 0])),
            "right_bend_median_rad": float(np.median(arrays["knee_bend"][:, 1])),
            "left_bend_p05_rad": float(knee_bend_p05[0]),
            "right_bend_p05_rad": float(knee_bend_p05[1]),
            "left_swing_p95_rad": percentile(arrays["knee_swing"][:, 0], 95),
            "right_swing_p95_rad": percentile(arrays["knee_swing"][:, 1], 95),
            "max_swing_rad": max_knee_swing,
        },
        "negative_depth_points_left": negative_left,
        "negative_depth_points_right": negative_right,
        "history": history,
        "conclusion_boundary": "Temporal prefit and triangulation are engineering references, not anatomical or physical ground truth.",
    }

    for frame_index, pair in enumerate(pair_ids):
        frame_output = output / "frames" / f"frame_{pair:04d}"
        frame_output.mkdir()
        np.savez_compressed(
            frame_output / "result.npz",
            vertices_m=arrays["vertices"][frame_index].astype(np.float32), faces=faces,
            coco_left_m=arrays["coco_left"][frame_index].astype(np.float32),
            coco_right_m=arrays["coco_right"][frame_index].astype(np.float32),
            coco_world_m=arrays["coco_world"][frame_index].astype(np.float32),
            projected_left_px=arrays["projected_left"][frame_index].astype(np.float32),
            projected_right_px=arrays["projected_right"][frame_index].astype(np.float32),
            observed_left=observed_left_np[frame_index], observed_right=observed_right_np[frame_index],
            valid_left=valid_left_np[frame_index], valid_right=valid_right_np[frame_index],
            root_orient=arrays["roots"][frame_index:frame_index + 1].astype(np.float32),
            body_pose=arrays["full_pose"][frame_index:frame_index + 1].astype(np.float32),
            translation_m=arrays["translations"][frame_index:frame_index + 1].astype(np.float32),
            betas=np.zeros((1, 10), dtype=np.float32),
            triangulated_force_all_m=tri_np[frame_index], triangulation_available=tri_available_np[frame_index],
            temporal_prefit_world_m=prefit_world_np[frame_index],
        )
        (frame_output / "metrics.json").write_text(json.dumps({
            "pair_id": pair,
            "left_p95_px": percentile(left_error[frame_index, valid_left_np[frame_index]], 95),
            "right_p95_px": percentile(right_error[frame_index, valid_right_np[frame_index]], 95),
            "triangulation_quality_flags": tri_flags[frame_index],
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 2 if rejection_reasons else 0


if __name__ == "__main__":
    raise SystemExit(main())
