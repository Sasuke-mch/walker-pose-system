#!/usr/bin/env python3
"""Jointly fit a short male-SMPL sequence with broad 3-D guardrails.

The stereo 2-D observations remain the data term.  Force-all triangulation is
not treated as anatomical truth: every finite point is retained, but incurs no
loss inside a deliberately broad, quality-dependent radius.  Temporal terms
and top-tail hinge penalties only prevent gross branch switches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_app.fisheye_camera import fisheye_project_torch, load_stereo_fisheye  # noqa: E402
from pose_app.smpl_coco_observation import (  # noqa: E402
    COCO17_NAMES,
    load_coco17_regressor,
    load_smpl_male,
    regress_coco17_torch,
)
from tools.fit_smpl_coco_single_frame import (  # noqa: E402
    ACTIVE_POSE_COMPONENTS,
    RIGHT_KNEE_INDEX,
    huber_vector,
)


def read_jsonl_by_pair(path: Path, start_pair: int, end_pair: int) -> dict[int, dict]:
    wanted = set(range(start_pair, end_pair + 1))
    rows: dict[int, dict] = {}
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            pair = int(row.get("pair_id", -1))
            if pair in wanted:
                if row.get("max_matches") != 1:
                    raise ValueError(f"pair {pair} does not preserve max_matches=1")
                rows[pair] = row
    missing = sorted(wanted - rows.keys())
    if missing:
        raise KeyError(f"missing pairs in {path}: {missing}")
    return rows


def load_force_all_guardrails(
    rows: dict[int, dict], pair_ids: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[list[str]]]]:
    points = np.full((len(pair_ids), 17, 3), np.nan, dtype=np.float32)
    radii = np.full((len(pair_ids), 17), 0.55, dtype=np.float32)
    reprojection = np.full((len(pair_ids), 17), np.nan, dtype=np.float32)
    flags: list[list[list[str]]] = []
    for frame_index, pair in enumerate(pair_ids):
        people = rows[pair].get("persons_3d", [])
        if len(people) != 1:
            raise ValueError(f"force-all pair {pair} must contain exactly one stereo person")
        joints = people[0].get("keypoints_3d", [])
        if len(joints) != 17:
            raise ValueError(f"force-all pair {pair} must contain 17 triangulated points")
        frame_flags: list[list[str]] = [[] for _ in range(17)]
        for joint in joints:
            index = int(joint["index"])
            xyz = np.asarray(joint.get("xyz"), dtype=np.float64)
            if xyz.shape == (3,) and np.isfinite(xyz).all():
                points[frame_index, index] = (xyz / 1000.0).astype(np.float32)
            raw_error = joint.get("reprojection_error_mean_px")
            error = float(raw_error) if raw_error is not None else float("nan")
            reprojection[frame_index, index] = error
            # Good stereo geometry still gets a 350 mm dead zone.  A point with
            # large reprojection error is retained, never rejected, but receives
            # a 550 mm dead zone.  The known-unreliable right knee gets 650 mm.
            radii[frame_index, index] = 0.35 if np.isfinite(error) and error <= 10.0 else 0.55
            if index == RIGHT_KNEE_INDEX:
                radii[frame_index, index] = 0.65
            frame_flags[index] = [str(value) for value in joint.get("quality_flags", [])]
        flags.append(frame_flags)
    available = np.isfinite(points).all(axis=2)
    # Missing xyz is not a large-error observation and cannot define a distance.
    # Keep an explicit mask and a finite placeholder solely for tensor arithmetic.
    points[~available] = 0.0
    return points, radii, reprojection, available, flags


def top_tail_mean(torch: "object", values: "object", fraction: float = 0.06) -> "object":
    flat = values.reshape(-1)
    count = max(1, int(np.ceil(flat.numel() * fraction)))
    return torch.topk(flat, count, largest=True).values.mean()


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent-window", type=Path, required=True)
    parser.add_argument("--force-all-jsonl", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--regressor", type=Path, required=True)
    parser.add_argument("--start-pair", type=int, required=True)
    parser.add_argument("--end-pair", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "frames").mkdir()

    pair_ids = list(range(args.start_pair, args.end_pair + 1))
    frame_dirs = [args.independent_window.resolve() / "frames" / f"frame_{pair:04d}" for pair in pair_ids]
    for directory in frame_dirs:
        if not (directory / "result.npz").is_file():
            raise FileNotFoundError(f"missing independent initialization: {directory / 'result.npz'}")

    initial = [np.load(directory / "result.npz") for directory in frame_dirs]
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

    force_rows = read_jsonl_by_pair(args.force_all_jsonl.resolve(), args.start_pair, args.end_pair)
    tri_np, radius_np, tri_reprojection_np, tri_available_np, tri_flags = load_force_all_guardrails(force_rows, pair_ids)
    cal = load_stereo_fisheye(args.calibration_dir)

    import torch

    count = len(pair_ids)
    model = load_smpl_male(args.model_path, batch_size=count, device=args.device)
    regressor = load_coco17_regressor(args.regressor)
    roots = torch.nn.Parameter(torch.tensor(roots_np, device=args.device))
    translations = torch.nn.Parameter(torch.tensor(translations_np, device=args.device))
    active_pose = torch.nn.Parameter(torch.tensor(active_pose_np, device=args.device))
    active_indices = torch.tensor(ACTIVE_POSE_COMPONENTS, dtype=torch.long, device=args.device)
    betas = torch.zeros(count, 10, dtype=torch.float32, device=args.device)
    observed_left = torch.tensor(observed_left_np[:, :, :2], device=args.device)
    observed_right = torch.tensor(observed_right_np[:, :, :2], device=args.device)
    weight_left = torch.tensor(valid_left_np * np.square(observed_left_np[:, :, 2]), device=args.device)
    weight_right = torch.tensor(valid_right_np * np.square(observed_right_np[:, :, 2]), device=args.device)
    tri = torch.tensor(tri_np, device=args.device)
    guard_radius = torch.tensor(radius_np, device=args.device)
    guard_available = torch.tensor(tri_available_np, dtype=torch.bool, device=args.device)
    rotation_right = torch.tensor(cal.R_cam0_to_cam1, dtype=torch.float32, device=args.device)
    translation_right = torch.tensor(cal.T_cam0_to_cam1_mm / 1000.0, dtype=torch.float32, device=args.device)

    def expand_pose() -> "object":
        full = torch.zeros(count, 69, dtype=active_pose.dtype, device=active_pose.device)
        return full.index_copy(1, active_indices, active_pose)

    def forward() -> tuple:
        result = model(
            betas=betas,
            body_pose=expand_pose(),
            global_orient=roots,
            transl=translations,
            return_verts=True,
        )
        coco_left = regress_coco17_torch(result.vertices, regressor)
        coco_right = torch.einsum("ij,bkj->bki", rotation_right, coco_left) + translation_right
        projected_left = fisheye_project_torch(coco_left, cal.K0, cal.D0)
        projected_right = fisheye_project_torch(coco_right, cal.K1, cal.D1)
        return result, coco_left, coco_right, projected_left, projected_right

    def losses(coco_left: "object", projected_left: "object", projected_right: "object", delta: float) -> dict:
        left = (huber_vector(torch, projected_left - observed_left, delta) * weight_left).sum()
        right = (huber_vector(torch, projected_right - observed_right, delta) * weight_right).sum()
        loss_2d = (left + right) / (weight_left.sum() + weight_right.sum()).clamp_min(1.0)

        distance = torch.linalg.vector_norm(coco_left - tri, dim=2)
        guard_excess = torch.where(
            guard_available,
            torch.relu(distance - guard_radius) / 0.10,
            torch.zeros_like(distance),
        )
        loss_guard = top_tail_mean(torch, guard_excess.square(), fraction=0.06)

        steps = torch.linalg.vector_norm(coco_left[1:] - coco_left[:-1], dim=2)
        step_excess = torch.relu(steps - 0.15) / 0.10
        loss_step = top_tail_mean(torch, step_excess.square(), fraction=0.06)

        pelvis = 0.5 * (coco_left[:, 11] + coco_left[:, 12])
        local = coco_left - pelvis[:, None, :]
        local_acc = local[2:] - 2.0 * local[1:-1] + local[:-2]
        translation_acc = translations[2:] - 2.0 * translations[1:-1] + translations[:-2]
        root_acc = roots[2:] - 2.0 * roots[1:-1] + roots[:-2]
        loss_acc = (local_acc / 0.10).square().mean()
        loss_translation_acc = (translation_acc / 0.10).square().mean()
        loss_root_acc = root_acc.square().mean()
        loss_pose = active_pose.square().mean()
        return {
            "loss_2d": loss_2d,
            "loss_guard": loss_guard,
            "loss_step": loss_step,
            "loss_local_acc": loss_acc,
            "loss_translation_acc": loss_translation_acc,
            "loss_root_acc": loss_root_acc,
            "loss_pose": loss_pose,
            "guard_distance": distance,
            "guard_excess_m": torch.where(
                guard_available,
                torch.relu(distance - guard_radius),
                torch.zeros_like(distance),
            ),
            "steps_m": steps,
        }

    optimizer = torch.optim.Adam(
        [
            {"params": [roots, active_pose], "lr": 0.0025},
            {"params": [translations], "lr": 0.0015},
        ]
    )
    history: list[dict] = []
    for step in range(args.iterations):
        result, coco_left, coco_right, projected_left, projected_right = forward()
        delta = 50.0 if step < args.iterations // 3 else 20.0
        terms = losses(coco_left, projected_left, projected_right, delta)
        ramp = min(1.0, (step + 1) / max(1, args.iterations // 4))
        total = (
            terms["loss_2d"]
            + 5.0 * terms["loss_pose"]
            + ramp * (
                500.0 * terms["loss_guard"]
                + 200.0 * terms["loss_step"]
                + 20.0 * terms["loss_local_acc"]
                + 20.0 * terms["loss_translation_acc"]
                + 5.0 * terms["loss_root_acc"]
            )
        )
        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_([roots, active_pose, translations], max_norm=2000.0)
        optimizer.step()
        with torch.no_grad():
            active_pose.clamp_(-2.5, 2.5)
            translations[:, 0:2].clamp_(-3.0, 3.0)
            translations[:, 2].clamp_(0.25, 3.0)
        if step in {0, 49, 99, 199, 399, args.iterations - 1}:
            history.append({
                "iteration": step + 1,
                "total": float(total.detach()),
                **{key: float(value.detach()) for key, value in terms.items() if key.startswith("loss_")},
                "max_guard_excess_mm": float(terms["guard_excess_m"].max().detach() * 1000.0),
                "max_joint_step_mm": float(terms["steps_m"].max().detach() * 1000.0),
            })
            print(json.dumps(history[-1]), flush=True)

    with torch.no_grad():
        result, coco_left, coco_right, projected_left, projected_right = forward()
        terms = losses(coco_left, projected_left, projected_right, 20.0)
    vertices_np = result.vertices.cpu().numpy()
    coco_left_np = coco_left.cpu().numpy()
    coco_right_np = coco_right.cpu().numpy()
    projected_left_np = projected_left.cpu().numpy()
    projected_right_np = projected_right.cpu().numpy()
    full_pose_final = expand_pose().detach().cpu().numpy()
    roots_final = roots.detach().cpu().numpy()
    translations_final = translations.detach().cpu().numpy()
    guard_distance_np = terms["guard_distance"].cpu().numpy()
    guard_excess_np = terms["guard_excess_m"].cpu().numpy()
    steps_np = terms["steps_m"].cpu().numpy()

    left_error = np.linalg.norm(projected_left_np - observed_left_np[:, :, :2], axis=2)
    right_error = np.linalg.norm(projected_right_np - observed_right_np[:, :, :2], axis=2)
    used_left = left_error[valid_left_np]
    used_right = right_error[valid_right_np]
    acceleration = np.linalg.norm(np.diff(coco_left_np, n=2, axis=0), axis=2) * 1000.0
    left_wrist_step_65_66 = None
    if 65 in pair_ids and 66 in pair_ids:
        a, b = pair_ids.index(65), pair_ids.index(66)
        left_wrist_step_65_66 = float(np.linalg.norm(coco_left_np[b, 9] - coco_left_np[a, 9]) * 1000.0)

    baseline_coco = np.stack([
        np.load(directory / "result.npz")["coco_left_m"] for directory in frame_dirs
    ])
    baseline_steps = np.linalg.norm(np.diff(baseline_coco, axis=0), axis=2) * 1000.0
    baseline_wrist_65_66 = None
    if 65 in pair_ids and 66 in pair_ids:
        a, b = pair_ids.index(65), pair_ids.index(66)
        baseline_wrist_65_66 = float(np.linalg.norm(baseline_coco[b, 9] - baseline_coco[a, 9]) * 1000.0)

    summary = {
        "schema_version": "pmpose_smpl_coco_temporal_broad_guardrail_v1",
        "window": [args.start_pair, args.end_pair],
        "frame_count": count,
        "fit_contract": {
            "initialization": "existing independent male-SMPL 2-D fits for the same window",
            "model": "official male SMPL 6890; beta fixed zero",
            "observation_model": "Pose2Mesh COCO17 17x6890 regressor",
            "primary_observation": "saved raw stereo PMPose 2-D from the independent result files",
            "triangulation": "all finite PMPose force-all points retained as broad dead-zone guardrails",
            "guardrail_radii_m": {"reprojection_le_10px": 0.35, "high_reprojection": 0.55, "right_knee": 0.65},
            "temporal": "150 mm/frame dead-zone step barrier plus root-relative joint and root/translation acceleration",
            "excluded": ["contact", "beta optimization", "SMPL-X", "VPoser", "display smoothing"],
        },
        "weights": {
            "pose": 5.0, "guard_top_tail": 500.0, "step_top_tail": 200.0,
            "local_acceleration": 20.0, "translation_acceleration": 20.0, "root_acceleration": 5.0,
        },
        "reprojection": {
            "combined_median_px": percentile(np.concatenate([used_left, used_right]), 50),
            "combined_p95_px": percentile(np.concatenate([used_left, used_right]), 95),
            "left_median_px": percentile(used_left, 50),
            "left_p95_px": percentile(used_left, 95),
            "right_median_px": percentile(used_right, 50),
            "right_p95_px": percentile(used_right, 95),
        },
        "temporal": {
            "joint_step_p95_mm": percentile(steps_np * 1000.0, 95),
            "joint_step_max_mm": float(steps_np.max() * 1000.0),
            "left_wrist_step_65_66_mm": left_wrist_step_65_66,
            "coco_acceleration_p95_mm_per_frame2": percentile(acceleration, 95),
        },
        "baseline_independent": {
            "joint_step_p95_mm": percentile(baseline_steps, 95),
            "joint_step_max_mm": float(baseline_steps.max()),
            "left_wrist_step_65_66_mm": baseline_wrist_65_66,
        },
        "guardrail": {
            "all_finite_force_all_points_retained": True,
            "available_point_count": int(tri_available_np.sum()),
            "unavailable_xyz_count": int((~tri_available_np).sum()),
            "high_reprojection_point_count": int(np.sum((tri_reprojection_np > 10.0) & tri_available_np)),
            "violation_count": int(np.sum(guard_excess_np > 1e-6)),
            "max_violation_mm": float(guard_excess_np.max() * 1000.0),
            "p95_distance_to_triangulation_mm": percentile(guard_distance_np * 1000.0, 95),
        },
        "negative_depth_points_left": int(np.sum(coco_left_np[:, :, 2] <= 0.0)),
        "negative_depth_points_right": int(np.sum(coco_right_np[:, :, 2] <= 0.0)),
        "history": history,
        "conclusion_boundary": (
            "Triangulation is a broad branch guardrail, not anatomical ground truth. "
            "Reprojection and temporal metrics are internal consistency diagnostics, not true accuracy."
        ),
    }

    for frame_index, pair in enumerate(pair_ids):
        frame_output = output / "frames" / f"frame_{pair:04d}"
        frame_output.mkdir()
        np.savez_compressed(
            frame_output / "result.npz",
            vertices_m=vertices_np[frame_index].astype(np.float32),
            faces=faces,
            coco_left_m=coco_left_np[frame_index].astype(np.float32),
            coco_right_m=coco_right_np[frame_index].astype(np.float32),
            projected_left_px=projected_left_np[frame_index].astype(np.float32),
            projected_right_px=projected_right_np[frame_index].astype(np.float32),
            observed_left=observed_left_np[frame_index],
            observed_right=observed_right_np[frame_index],
            valid_left=valid_left_np[frame_index],
            valid_right=valid_right_np[frame_index],
            root_orient=roots_final[frame_index:frame_index + 1],
            body_pose=full_pose_final[frame_index:frame_index + 1],
            translation_m=translations_final[frame_index:frame_index + 1],
            betas=np.zeros((1, 10), dtype=np.float32),
            triangulated_force_all_m=tri_np[frame_index],
            guardrail_radius_m=radius_np[frame_index],
            guardrail_distance_m=guard_distance_np[frame_index],
            guardrail_available=tri_available_np[frame_index],
        )
        frame_metrics = {
            "pair_id": pair,
            "left_median_px": percentile(left_error[frame_index, valid_left_np[frame_index]], 50),
            "left_p95_px": percentile(left_error[frame_index, valid_left_np[frame_index]], 95),
            "right_median_px": percentile(right_error[frame_index, valid_right_np[frame_index]], 50),
            "right_p95_px": percentile(right_error[frame_index, valid_right_np[frame_index]], 95),
            "guardrail_violation_count": int(np.sum(guard_excess_np[frame_index] > 1e-6)),
            "guardrail_max_violation_mm": float(guard_excess_np[frame_index].max() * 1000.0),
            "triangulation_reprojection_mean_px": tri_reprojection_np[frame_index].tolist(),
            "triangulation_quality_flags": tri_flags[frame_index],
        }
        (frame_output / "metrics.json").write_text(
            json.dumps(frame_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
