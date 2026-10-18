#!/usr/bin/env python3
"""Audit weighted SMPL-X loss gradients at frozen clean checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from tools.benchmark_smplx_r3_integration import (  # noqa: E402
    DEV_WINDOW, GM_SCALE_M, GM_SCALE_PX, JOINT_NAMES, WEIGHTS, FitEnv,
    FitRouteConfig, parameters_from_state, run_fit,
)
from pose_app.fisheye_camera import fisheye_project_torch, world_to_cameras_torch_batch  # noqa: E402
from pose_app.reprojection_common import robust_gm  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--smplx-gender", choices=("male", "female"), default="male")
    p.add_argument(
        "--smplx-prior-mode",
        choices=("inherited", "clean_zero"),
        default="inherited",
        help=(
            "inherited reproduces the historical fixed beta/offset/latent inputs; "
            "clean_zero starts from zero beta, zero VPoser latent and identity COCO mapping"
        ),
    )
    p.add_argument("--model-root", type=Path, default=Path(r"D:\my_works\walker_pose_system\models"))
    p.add_argument("--vposer-dir", type=Path, default=Path(r"D:\my_works\walker_pose_system\models\VPoser02_05\V02_05"))
    p.add_argument("--motion-json", type=Path, default=Path(r"D:\my_works\walker_pose_system\docs\walker-viewer\data\motion.json"))
    p.add_argument("--prefit-jsonl", type=Path, default=Path(r"D:\my_works\walker_pose_system\research_records\engineering_validation\G20260918_temporal_body_model_prefit_v1\fixed_bone_contact_prefit_v2_gated\temporal_body_prefit.jsonl"))
    p.add_argument("--interaction-jsonl", type=Path, default=Path(r"D:\my_works\walker_pose_system\research_records\engineering_validation\G20260918_offline_walker_frame_evidence_v1\coarse_complete_model_visual_v3_camera_pair\interaction_distance_records.jsonl"))
    p.add_argument("--initial-fit-summary", type=Path, default=Path(r"D:\my_works\walker_pose_system\research_records\engineering_validation\G20260918_smplx_temporal_fit_v1\single_frame_quality_selected_v2\summary.json"))
    p.add_argument("--stereo-jsonl", type=Path, default=Path(r"D:\my_works\walker_pose_system\research_records\engineering_validation\V20260908_people0_1_2_pmpose_c3_chain\people_1\pmpose_strict_stereo\offline_stereo_results.jsonl"))
    p.add_argument("--calibration-dir", type=Path, default=Path(r"D:\my_works\walker_pose_system\realtime_app\calibration\results"))
    p.add_argument("--offset-audit-json", type=Path, default=Path(r"D:\my_works\walker_pose_system\research_records\engineering_validation\G20260918_smplx_shape_offset_gate_v1\final_shape_offset_audit_v3.json"))
    p.add_argument("--fixed-betas-json", type=Path, default=Path(r"D:\my_works\walker_pose_system\research_records\engineering_validation\G20260918_smplx_shape_offset_gate_v1\final_shape_offset_audit_v3.json"))
    p.add_argument("--r3-jsonl", type=Path, default=Path(r"D:\my_works\walker_pose_system\research_records\engineering_validation\G20260919_alternative_body_routes_final_v8\R3_kalman_fixed_bone_ik\config_IK3_R030\per_frame_metrics.jsonl"))
    return p


def replace_unused(grads, params, torch):
    return tuple(torch.zeros_like(param) if grad is None else grad for grad, param in zip(grads, params))


def flatten(grads, torch):
    return torch.cat([g.reshape(-1) for g in grads])


def gradient_audit(env: FitEnv, config: FitRouteConfig, state: dict, iteration: int) -> dict:
    torch = env.torch
    params = parameters_from_state(env, state)
    masks = env.masks_with_holdout(set())
    terms = env.loss_terms(*params, masks, config)
    weighted = {
        "data2d": WEIGHTS["data2d"] * terms["data2d"],
        "data3d": WEIGHTS["data3d"] * terms["data3d"],
        "temporal": WEIGHTS["temporal"] * terms["temporal"],
        "foot_contact": WEIGHTS["foot_contact"] * terms["foot_contact"],
        "hand_contact": WEIGHTS["hand_contact"] * terms["hand_contact"],
        "latent": WEIGHTS["latent"] * terms["latent"],
    }
    vectors = {}
    group_vectors = {}
    groups = {}
    values = {}
    for name, value in weighted.items():
        grads = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
        grads = replace_unused(grads, params, torch)
        vectors[name] = flatten(grads, torch).detach()
        group_vectors[name] = {
            key: grad.reshape(-1).detach()
            for key, grad in zip(("root_orient", "translation", "latent"), grads)
        }
        groups[name] = {
            "root_orient": float(torch.linalg.vector_norm(grads[0]).detach().cpu()),
            "translation": float(torch.linalg.vector_norm(grads[1]).detach().cpu()),
            "latent": float(torch.linalg.vector_norm(grads[2]).detach().cpu()),
            "all": float(torch.linalg.vector_norm(vectors[name]).cpu()),
        }
        values[name] = float(value.detach().cpu())
    cosines = {}
    group_cosines = {key: {} for key in ("root_orient", "translation", "latent")}
    names = list(vectors)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            denom = torch.linalg.vector_norm(vectors[left]) * torch.linalg.vector_norm(vectors[right])
            cosine = torch.dot(vectors[left], vectors[right]) / torch.clamp(denom, min=1e-12)
            cosines[f"{left}__{right}"] = float(cosine.cpu())
            for group_name in group_cosines:
                left_g = group_vectors[left][group_name]
                right_g = group_vectors[right][group_name]
                group_denom = torch.linalg.vector_norm(left_g) * torch.linalg.vector_norm(right_g)
                group_value = torch.dot(left_g, right_g) / torch.clamp(group_denom, min=1e-12)
                group_cosines[group_name][f"{left}__{right}"] = float(group_value.cpu())
    antagonistic_to_temporal = sorted(
        ({"term": name, "cosine": cosines.get(f"{name}__temporal", cosines.get(f"temporal__{name}"))}
         for name in names if name != "temporal"),
        key=lambda row: row["cosine"],
    )
    per_joint = []
    if iteration in (200, 400):
        _, _, predicted_coco_body, _ = env.forward(*params)
        left_m, right_m = world_to_cameras_torch_batch(
            predicted_coco_body, env.per_frame_rotation, env.per_frame_translation, env.cal
        )
        proj_l = fisheye_project_torch(left_m, env.cal.K0, env.cal.D0)
        proj_r = fisheye_project_torch(right_m, env.cal.K1, env.cal.D1)
        residual_l = torch.linalg.vector_norm(proj_l - env.kp_left, dim=2) * env.kp_valid_l
        residual_r = torch.linalg.vector_norm(proj_r - env.kp_right, dim=2) * env.kp_valid_r
        denom_l = torch.clamp(masks["w2d_l"].sum(), min=1.0)
        denom_r = torch.clamp(masks["w2d_r"].sum(), min=1.0)
        denom_3d = torch.clamp(masks["prefit_w"].sum(), min=1.0)
        for position, coco_index in enumerate(env.coco_indices):
            loss2d = WEIGHTS["data2d"] * 0.5 * (
                torch.sum(masks["w2d_l"][:, position] * residual_l[:, position].square()
                          / (residual_l[:, position].square() + GM_SCALE_PX**2)) / denom_l
                + torch.sum(masks["w2d_r"][:, position] * residual_r[:, position].square()
                            / (residual_r[:, position].square() + GM_SCALE_PX**2)) / denom_r
            )
            loss3d = WEIGHTS["data3d"] * torch.sum(
                masks["prefit_w"][:, position, None]
                * robust_gm(predicted_coco_body[:, position] - env.prefit_target_m[:, position], GM_SCALE_M)
            ) / denom_3d
            grad2d = torch.autograd.grad(loss2d, params[0], retain_graph=True)[0].reshape(-1)
            grad3d = torch.autograd.grad(loss3d, params[0], retain_graph=True)[0].reshape(-1)
            denom = torch.linalg.vector_norm(grad2d) * torch.linalg.vector_norm(grad3d)
            cosine = torch.dot(grad2d, grad3d) / torch.clamp(denom, min=1e-12)
            per_joint.append({
                "joint_index": int(coco_index), "joint_name": JOINT_NAMES[int(coco_index)],
                "root_cosine_2d_vs_3d": float(cosine.detach().cpu()),
                "root_grad_norm_2d": float(torch.linalg.vector_norm(grad2d).detach().cpu()),
                "root_grad_norm_3d": float(torch.linalg.vector_norm(grad3d).detach().cpu()),
                "weighted_loss_2d": float(loss2d.detach().cpu()),
                "weighted_loss_3d": float(loss3d.detach().cpu()),
            })
        per_joint.sort(key=lambda row: row["root_cosine_2d_vs_3d"])
    return {"iteration": iteration, "weighted_values": values, "gradient_norms": groups,
            "pairwise_cosines": cosines, "group_pairwise_cosines": group_cosines,
            "temporal_antagonists": antagonistic_to_temporal,
            "per_joint_root_conflicts": per_joint}


def main() -> int:
    args = parser().parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = FitEnv(args, *DEV_WINDOW, device=args.device)
    config = FitRouteConfig("A0-clean-gradient-audit", init="prefit", data3d=True,
                            r3soft_weight=None, temporal_point_space="body",
                            anatomical_temporal_weight=0.0)
    run = run_fit(env, config, set(), (0, 100, 200, 400), "gradient_audit", iterations=400)
    audits = [gradient_audit(env, config, run.checkpoint_states[i], i) for i in (0, 100, 200, 400)]
    payload = {"window": list(DEV_WINDOW), "route": config.name, "audits": audits}
    (out / "gradient_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out / "gradient_audit.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
