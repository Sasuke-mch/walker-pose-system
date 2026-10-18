#!/usr/bin/env python3
"""Repair clean-male 2-D convergence with one deterministic coarse-to-fine route."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from pose_app.fisheye_camera import (  # noqa: E402
    fisheye_project_torch, world_to_cameras_torch_batch,
)
from tools.benchmark_smplx_r3_integration import (  # noqa: E402
    DEV_WINDOW, FitEnv, FitRouteConfig, build_per_frame_rows, write_jsonl,
)
from tools.diagnose_smplx_loss_gradients import parser  # noqa: E402

COARSE_DELTA_PX = 100.0
POSE_ONLY_STEPS = 200
JOINT_COARSE_STEPS = 200
FINE_STEPS = 400
BRANCH_REPAIR_STEPS = 300
JOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)


def balanced_huber_2d(env: FitEnv, root, translation, latent, delta_px: float = COARSE_DELTA_PX, betas=None):
    """Accepted-view loss with equal influence per observed joint and live tail gradients."""
    torch = env.torch
    _, _, predicted_body, _ = env.forward(root, translation, latent, betas=betas)
    left_m, right_m = world_to_cameras_torch_batch(
        predicted_body, env.per_frame_rotation, env.per_frame_translation, env.cal
    )
    projected = (
        fisheye_project_torch(left_m, env.cal.K0, env.cal.D0),
        fisheye_project_torch(right_m, env.cal.K1, env.cal.D1),
    )
    observations = (env.kp_left, env.kp_right)
    accepted = (env.kp_valid_l, env.kp_valid_r)
    weights = (
        torch.as_tensor(env.w2d_left_np, dtype=torch.float32, device=env.device),
        torch.as_tensor(env.w2d_right_np, dtype=torch.float32, device=env.device),
    )
    groups = []
    delta = float(delta_px)
    for prediction, target, valid, weight in zip(projected, observations, accepted, weights):
        residual = torch.linalg.vector_norm(prediction - target, dim=2)
        huber = torch.where(
            residual <= delta,
            0.5 * residual.square() / delta,
            residual - 0.5 * delta,
        )
        effective = valid * weight
        for joint in range(huber.shape[1]):
            denominator = effective[:, joint].sum()
            if float(denominator.detach().cpu()) > 0.0:
                groups.append((effective[:, joint] * huber[:, joint]).sum() / denominator)
    if not groups:
        raise RuntimeError("no accepted 2-D observations for coarse fit")
    return torch.stack(groups).mean()


def snapshot(env, config, root, translation, latent, phase: str, step: int) -> dict:
    masks = env.masks_with_holdout(set())
    metric = env.snapshot(root, translation, latent, masks, config, set())
    metric.update({"phase": phase, "phase_step": step})
    return metric


def joint_metrics(rows: list[dict]) -> dict:
    result = {}
    for joint in range(5, 17):
        record = {"joint_index": joint, "joint_name": JOINT_NAMES[joint]}
        for side, error_key, accepted_key in (
            ("left", "reprojection_left_px", "left_accepted"),
            ("right", "reprojection_right_px", "right_accepted"),
        ):
            values = [
                float(row[error_key][joint]) for row in rows
                if row["observation_state"][joint][accepted_key]
            ]
            record[side] = {
                "accepted_count": len(values),
                "median_px": float(np.median(values)) if values else None,
                "p95_px": float(np.percentile(values, 95)) if values else None,
                "max_px": float(np.max(values)) if values else None,
                "count_over_100px": int(np.count_nonzero(np.asarray(values) > 100.0)),
            }
        result[str(joint)] = record
    return result


def failed_frame_indices(rows: list[dict]) -> list[int]:
    failed = []
    for local, row in enumerate(rows):
        high_accepted = any(
            row["observation_state"][joint][accepted_key]
            and float(row[error_key][joint]) > 100.0
            for joint in range(5, 17)
            for error_key, accepted_key in (
                ("reprojection_left_px", "left_accepted"),
                ("reprojection_right_px", "right_accepted"),
            )
        )
        nonpositive_depth = any(
            float(row["depth_left_mm"][joint]) <= 0.0
            or float(row["depth_right_mm"][joint]) <= 0.0
            for joint in range(5, 17)
        )
        if high_accepted or nonpositive_depth:
            failed.append(local)
    return failed


def interpolate_failed_parameters(torch, parameters, failed: list[int]) -> None:
    failed_set = set(failed)
    good = [index for index in range(parameters[0].shape[0]) if index not in failed_set]
    if not good:
        raise RuntimeError("cannot repair when every frame failed")
    with torch.no_grad():
        for index in failed:
            before = max((value for value in good if value < index), default=None)
            after = min((value for value in good if value > index), default=None)
            for parameter in parameters:
                if before is None:
                    parameter[index].copy_(parameter[after])
                elif after is None:
                    parameter[index].copy_(parameter[before])
                else:
                    alpha = float(index - before) / float(after - before)
                    parameter[index].copy_((1.0 - alpha) * parameter[before] + alpha * parameter[after])


def main() -> int:
    args = parser().parse_args()
    if args.smplx_gender != "male" or args.smplx_prior_mode != "clean_zero":
        raise ValueError("route requires --smplx-gender male --smplx-prior-mode clean_zero")
    out = args.output_dir.resolve()
    metrics_path = out / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError(f"refuse to overwrite coarse-to-fine fit: {metrics_path}")
    out.mkdir(parents=True, exist_ok=True)

    env = FitEnv(args, *DEV_WINDOW, device=args.device)
    root, translation, latent = env.make_params(set())
    config = FitRouteConfig(
        "M1-clean-male-coarse-to-fine-2d", init="prefit", data3d=False,
        r3soft_weight=None, temporal_weight=0.0, anatomical_temporal_weight=0.0,
        foot_contact_weight=0.0, hand_contact_weight=0.0,
        pelvis_anchor_weight=0.0, pelvis_net_weight=0.0, pelvis_direction_weight=0.0,
        temporal_point_space="coco",
    )
    history = [snapshot(env, config, root, translation, latent, "initial", 0)]

    # Phase 1: preserve the corrected root/translation contract and move pose
    # out of the neutral-latent basin. Equal-per-joint aggregation prevents the
    # fully observed ankles from overwhelming partially observed arms.
    optimizer = env.torch.optim.Adam([latent], lr=0.02)
    for step in range(1, POSE_ONLY_STEPS + 1):
        optimizer.zero_grad()
        loss = balanced_huber_2d(env, root.detach(), translation.detach(), latent)
        loss = loss + 0.02 * env.torch.mean(latent.square())
        loss.backward()
        optimizer.step()
    history.append(snapshot(env, config, root, translation, latent, "pose_only_wide", POSE_ONLY_STEPS))

    # Phase 2: after articulated pose has moved, jointly correct remaining
    # root/translation errors under the same non-saturating, joint-balanced loss.
    optimizer = env.torch.optim.Adam([root, translation, latent], lr=0.01)
    for step in range(1, JOINT_COARSE_STEPS + 1):
        optimizer.zero_grad()
        loss = balanced_huber_2d(env, root, translation, latent)
        loss = loss + 0.02 * env.torch.mean(latent.square())
        loss.backward()
        optimizer.step()
    history.append(snapshot(env, config, root, translation, latent, "joint_wide", JOINT_COARSE_STEPS))

    # Phase 3: narrow the Huber transition to 20 px while preserving the
    # equal-per-joint aggregation and non-zero tail gradient. The v1 audit
    # showed that restoring globally averaged GM here re-created the tail.
    masks = env.masks_with_holdout(set())
    optimizer = env.torch.optim.Adam([root, translation, latent], lr=0.005)
    for step in range(1, FINE_STEPS + 1):
        optimizer.zero_grad()
        loss = balanced_huber_2d(env, root, translation, latent, delta_px=20.0)
        loss = loss + 0.02 * env.torch.mean(latent.square())
        loss.backward()
        optimizer.step()
    fine = snapshot(env, config, root, translation, latent, "fine_balanced_huber20", FINE_STEPS)
    history.append(fine)
    fine_rows = build_per_frame_rows(
        env, config, root, translation, latent, set(), "clean_male_coarse_to_fine", 0, False
    )
    failed_before_repair = failed_frame_indices(fine_rows)
    interpolate_failed_parameters(env.torch, (root, translation, latent), failed_before_repair)

    failed_mask = env.torch.zeros(env.window_size, dtype=env.torch.bool, device=env.device)
    failed_mask[failed_before_repair] = True
    optimizer = env.torch.optim.Adam([root, translation, latent], lr=0.005)
    for step in range(1, BRANCH_REPAIR_STEPS + 1):
        optimizer.zero_grad()
        loss = balanced_huber_2d(env, root, translation, latent, delta_px=20.0)
        loss = loss + 0.02 * env.torch.mean(latent[failed_mask].square())
        loss.backward()
        for parameter in (root, translation, latent):
            parameter.grad[~failed_mask] = 0.0
        optimizer.step()

    final = snapshot(env, config, root, translation, latent, "failed_branch_repair", BRANCH_REPAIR_STEPS)
    final_terms = env.loss_terms(root, translation, latent, masks, config)
    final["loss_terms"] = env.terms_to_float(final_terms)
    final["loss_total"] = float(env.total_loss(final_terms, config).detach().cpu())
    history.append(final)
    rows = build_per_frame_rows(
        env, config, root, translation, latent, set(), "clean_male_coarse_to_fine", 0, False
    )
    failed_after_repair = failed_frame_indices(rows)
    per_joint = joint_metrics(rows)
    observed_joint_views = [
        side for record in per_joint.values() for side in (record["left"], record["right"])
        if side["accepted_count"] > 0
    ]
    checks = {
        "left_strict_p95_at_most_80px": final["reproj_left_strict_p95_px"] <= 80.0,
        "right_strict_p95_at_most_80px": final["reproj_right_strict_p95_px"] <= 80.0,
        "all_observed_joint_view_medians_at_most_50px": all(
            side["median_px"] <= 50.0 for side in observed_joint_views
        ),
        "no_accepted_residual_over_100px": all(
            side["count_over_100px"] == 0 for side in observed_joint_views
        ),
        "no_nan": final["nan_joint_count"] == 0,
        "no_negative_depth": final["negative_depth_count"] == 0,
        "full_body_availability": final["available_joint_count"] == env.window_size * 12,
        "right_knee_unobserved_is_explicit": (
            per_joint["14"]["left"]["accepted_count"] == 0
            and per_joint["14"]["right"]["accepted_count"] == 0
        ),
    }
    payload = {
        "schema_version": "smplx_clean_male_coarse_to_fine_v1",
        "status": "passed_pose_tail_gate" if all(checks.values()) else "failed_pose_tail_gate",
        "scope": "accepted dual-fisheye 2-D convergence repair only",
        "frame_range_inclusive": list(DEV_WINDOW),
        "fixed_design": {
            "coarse_delta_px": COARSE_DELTA_PX,
            "pose_only_steps": POSE_ONLY_STEPS,
            "joint_coarse_steps": JOINT_COARSE_STEPS,
            "fine_steps": FINE_STEPS,
            "fine_loss": "equal_per_observed_joint_view_huber20",
            "failed_branch_repair_steps": BRANCH_REPAIR_STEPS,
            "betas": "fixed_zero",
            "soft_3d": False,
            "temporal": False,
            "foot_contact": False,
            "hand_contact": False,
        },
        "failed_frame_indices_before_branch_repair": failed_before_repair,
        "failed_frame_indices_after_branch_repair": failed_after_repair,
        "checks": checks,
        "history": history,
        "final": final,
        "per_joint": per_joint,
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(out / "per_frame_metrics.jsonl", rows)
    np.savez_compressed(
        out / "parameters.npz",
        root_orient=root.detach().cpu().numpy(),
        translation=translation.detach().cpu().numpy(),
        vposer_latent=latent.detach().cpu().numpy(),
        betas=env.initial_betas.copy(),
    )
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
