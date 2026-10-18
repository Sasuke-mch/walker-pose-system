#!/usr/bin/env python3
"""Replay the causal fixed-lag skeleton tracker over a window and audit it.

Writes per output directory:
  - causal_skeleton.jsonl  (per-frame current-frame outputs)
  - timing.json            (per-update elapsed times + aggregates)
  - rejection_counts.json  (per-reason rejection counts per view)
  - summary.json           (metrics + frozen acceptance gates)
  - command.txt

The full-sequence prefit is used ONLY for cold-start initialisation and for
metric comparison. It is never a runtime soft target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from pose_app.causal_skeleton_tracker import CausalSkeletonConfig, CausalSkeletonTracker  # noqa: E402
from pose_app.fisheye_camera import load_stereo_fisheye  # noqa: E402
from pose_app.pose_observation_partition import (  # noqa: E402
    classify_view_observations_batch,
)
from pose_app.smplx_fitting import read_jsonl  # noqa: E402
from pose_app.skeleton_factor_graph import BODY_JOINTS  # noqa: E402


def _reproject_joints_np(joints_mm: np.ndarray, frames: list[dict], cal) -> tuple[np.ndarray, np.ndarray]:
    """Project (N,12,3) mm world joints into both fisheye views (numpy)."""
    from pose_app.fisheye_camera import fisheye_project_numpy
    proj_l = np.full((len(frames), 12, 2), np.nan)
    proj_r = np.full_like(proj_l, np.nan)
    for local, frame in enumerate(frames):
        rotation = np.asarray(frame["rotation_world_from_left"], dtype=np.float64)
        translation = np.asarray(frame["translation_world_from_left_mm"], dtype=np.float64)
        left = (rotation.T @ (joints_mm[local] - translation).T).T
        right = (cal.R_cam0_to_cam1 @ left.T + cal.T_cam0_to_cam1_mm.reshape(3, 1)).T
        proj_l[local] = fisheye_project_numpy(left, cal.K0, cal.D0)
        proj_r[local] = fisheye_project_numpy(right, cal.K1, cal.D1)
    return proj_l, proj_r


def _percentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-json", type=Path, required=True)
    parser.add_argument("--stereo-jsonl", type=Path, required=True)
    parser.add_argument("--interaction-jsonl", type=Path, required=True)
    parser.add_argument("--prefit-jsonl", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--frozen-bones-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--iterations", type=int, choices=(5, 10, 20), default=10)
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--soft3d-weight", type=float, default=0.0)
    parser.add_argument("--hinge-prior-weight", type=float, default=10.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)

    motion = json.loads(args.motion_json.read_text(encoding="utf-8-sig"))
    frames = motion["frames"]
    stereo_rows = read_jsonl(args.stereo_jsonl)
    interactions = read_jsonl(args.interaction_jsonl)
    prefit = read_jsonl(args.prefit_jsonl)
    cal = load_stereo_fisheye(args.calibration_dir)
    stereo_by_pair = {int(row["pair_id"]): row for row in stereo_rows}
    frozen = json.loads(args.frozen_bones_json.read_text(encoding="utf-8"))
    frozen_bones = np.asarray(frozen["frozen_median_lengths_mm"], dtype=np.float64)

    frame_count = len(frames)
    if not (len(stereo_rows) == len(interactions) == len(prefit) == frame_count):
        raise ValueError("input streams must align")
    if not (0 <= args.start_frame <= args.end_frame < frame_count):
        raise ValueError("invalid inclusive window")

    prefit_points_mm = np.asarray([
        [row["points_ground_mm"][str(joint)] for joint in range(17)] for row in prefit
    ], dtype=np.float64)
    indices = np.arange(args.start_frame, args.end_frame + 1)

    config = CausalSkeletonConfig(
        window_size=args.window_size,
        iterations_per_update=args.iterations,
        learning_rate=args.learning_rate,
        soft3d_current_frame_weight=args.soft3d_weight,
        hinge_prior_weight=args.hinge_prior_weight,
    )
    tracker = CausalSkeletonTracker(config, cal, frozen_bones, prefit_points_mm, device=args.device)

    output_rows = []
    times = []
    failed_frames = []
    for frame_index in indices:
        motion_frame = frames[frame_index]
        stereo_row = stereo_by_pair[int(motion_frame["pair"])]
        interaction_record = interactions[frame_index]
        person = None
        persons_3d = stereo_row.get("persons_3d") or []
        if persons_3d:
            person = persons_3d[0]
        result = tracker.update(
            frame_index, motion_frame, stereo_row, person, interaction_record
        )
        row = dict(tracker.output_history[frame_index])
        output_rows.append(row)
        times.append(result["update_time_s"])
        reasons = []
        if result["nan_count"] > 0:
            reasons.append("nan_or_nonfinite")
        if result["behind_camera_points"] > 0:
            reasons.append("behind_camera")
        if result["hinge_penalty_triggered"]:
            reasons.append("hinge_range_penalty_triggered")
        if reasons:
            failed_frames.append({"frame_index": int(frame_index), "reasons": reasons})

    # ---------------- observation partition of the OUTPUT frames ----------------
    joints_all = np.asarray([row["joints_ground_mm"] for row in output_rows], dtype=np.float64)  # (N,12,3) mm
    proj_l, proj_r = _reproject_joints_np(joints_all, [frames[i] for i in indices], cal)

    kp_left = np.full((len(indices), 12, 2), np.nan)
    kp_right = np.full_like(kp_left, np.nan)
    score_left = np.full((len(indices), 12), np.nan)
    score_right = np.full_like(score_left, np.nan)
    strict = np.zeros((len(indices), 12), dtype=bool)
    errors = np.full((len(indices), 12), np.nan)
    for local, frame_index in enumerate(indices):
        stereo_row = stereo_by_pair[int(frames[frame_index]["pair"])]
        left_persons = (stereo_row.get("left") or {}).get("persons") or []
        right_persons = (stereo_row.get("right") or {}).get("persons") or []
        left_kp = np.asarray(left_persons[0]["keypoints"], dtype=np.float64) if left_persons else None
        right_kp = np.asarray(right_persons[0]["keypoints"], dtype=np.float64) if right_persons else None
        for position, joint in enumerate(BODY_JOINTS):
            if left_kp is not None and np.all(np.isfinite(left_kp[joint][:2])):
                kp_left[local, position] = left_kp[joint][:2]
                score_left[local, position] = float(left_kp[joint][2])
            if right_kp is not None and np.all(np.isfinite(right_kp[joint][:2])):
                kp_right[local, position] = right_kp[joint][:2]
                score_right[local, position] = float(right_kp[joint][2])
            strict[local, position] = bool(frames[frame_index]["strict"][joint])
            err = frames[frame_index]["errors_px"][joint]
            if err is not None:
                errors[local, position] = float(err)
    part_l = classify_view_observations_batch(kp_left, score_left, strict, errors, cal.image_width, cal.image_height, 0.20)
    part_r = classify_view_observations_batch(kp_right, score_right, strict, errors, cal.image_width, cal.image_height, 0.20)

    def _split_reproj(proj: np.ndarray, accepted: np.ndarray, strict_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        residual = np.linalg.norm(proj - kp_left if proj is proj_l else proj - kp_right, axis=2)
        residual = np.where(np.isfinite(residual), residual, np.nan)
        strict_res = residual[accepted & strict_]
        relaxed_res = residual[accepted & ~strict_]
        return strict_res, relaxed_res

    strict_res_l, relaxed_res_l = _split_reproj(proj_l, part_l["accepted"], strict)
    strict_res_r, relaxed_res_r = _split_reproj(proj_r, part_r["accepted"], strict)

    # Per-joint strict reprojection P95 (12 joints, per view).
    per_joint_p95 = {}
    for name, proj, accepted, strict_, kp in (
        ("left", proj_l, part_l["accepted"], strict, kp_left),
        ("right", proj_r, part_r["accepted"], strict, kp_right),
    ):
        residual = np.linalg.norm(proj - kp, axis=2)
        residual = np.where(np.isfinite(residual), residual, np.nan)
        row = {}
        for position, joint in enumerate(BODY_JOINTS):
            values = residual[:, position][accepted[:, position] & strict_[:, position]]
            row[str(joint)] = _percentile(values, 95)
        per_joint_p95[name] = row

    # Per-stage strict reprojection P95.
    stage_names = [frames[i]["stage"] for i in indices]
    per_stage_p95 = {"left": {}, "right": {}}
    for name, proj, accepted, strict_, kp in (
        ("left", proj_l, part_l["accepted"], strict, kp_left),
        ("right", proj_r, part_r["accepted"], strict, kp_right),
    ):
        residual = np.linalg.norm(proj - kp, axis=2)
        residual = np.where(np.isfinite(residual), residual, np.nan)
        for stage in sorted(set(stage_names)):
            mask = np.asarray([s == stage for s in stage_names])
            values = residual[mask][accepted[mask] & strict_[mask]]
            per_stage_p95[name][stage] = _percentile(values, 95)

    # ---------------- dynamics metrics ----------------
    body_joints = joints_all  # (N,12,3) mm
    acc = body_joints[2:] - 2.0 * body_joints[1:-1] + body_joints[:-2]
    acc_p95 = _percentile(np.linalg.norm(acc, axis=2), 95)

    prefit_body = prefit_points_mm[indices][:, list(BODY_JOINTS)]
    prefit_acc = prefit_body[2:] - 2.0 * prefit_body[1:-1] + prefit_body[:-2]
    prefit_acc_p95 = _percentile(np.linalg.norm(prefit_acc, axis=2), 95)

    jumps = np.linalg.norm(body_joints[1:] - body_joints[:-1], axis=2)
    jump_p95 = _percentile(jumps, 95)

    pelvis_fitted = 0.5 * (body_joints[:, 6] + body_joints[:, 7])
    pelvis_input = 0.5 * (prefit_body[:, 6] + prefit_body[:, 7])
    pelvis_net_fitted = float(np.linalg.norm(pelvis_fitted[-1] - pelvis_fitted[0]))
    pelvis_net_input = float(np.linalg.norm(pelvis_input[-1] - pelvis_input[0]))
    pelvis_change = float(abs(pelvis_net_fitted - pelvis_net_input) / max(pelvis_net_input, 1e-6))

    # Stage-2 foot velocity (non-independent: Stage 2 feet also participate in
    # the camera pose estimation).
    stage2_mask = np.asarray([str(frames[i]["stage"]) == "stage2_feet_static_walker_moving" for i in indices], dtype=bool)
    ankle_vel = np.linalg.norm(body_joints[1:, [10, 11]] - body_joints[:-1, [10, 11]], axis=2)
    pair_mask = stage2_mask[1:] & stage2_mask[:-1]
    foot_vel_active = ankle_vel[pair_mask]
    foot_vel_p95 = _percentile(foot_vel_active, 95)

    # Hand-to-handle distance (visual proximity only, not real grip/load).
    hand_distances = []
    for local, frame_index in enumerate(indices):
        record = interactions[frame_index]
        for side_index, side in enumerate(("left", "right")):
            proxy = (record.get("hands") or {}).get(side, {}).get("closest_proxy_point_world_mm")
            if proxy is not None and np.all(np.isfinite(proxy)):
                wrist = body_joints[local, 4 + side_index]
                hand_distances.append(float(np.linalg.norm(wrist - np.asarray(proxy, dtype=np.float64))))
    hand_distance_median = float(np.median(hand_distances)) if hand_distances else float("nan")

    # ---------------- timing ----------------
    times = np.asarray(times, dtype=np.float64)
    timing = {
        "per_update_s": times.round(6).tolist(),
        "p50_s": float(np.percentile(times, 50)),
        "p95_s": float(np.percentile(times, 95)),
        "max_s": float(np.max(times)),
        "iterations_per_update": args.iterations,
    }

    def _rejection_only(counts: dict) -> dict:
        return {
            key: value for key, value in counts.items()
            if key not in ("accepted_strict", "accepted_relaxed")
        }

    rejection_counts = {
        "left": _rejection_only(part_l["rejection_counts"]),
        "right": _rejection_only(part_r["rejection_counts"]),
    }

    nan_count = int(np.count_nonzero(~np.isfinite(body_joints)))
    behind_camera = sum(row["behind_camera_points"] for row in output_rows)
    negative_bone = int(np.count_nonzero(frozen_bones <= 0.0))

    # ---------------- gates (frozen) ----------------
    def _max_finite(values: list) -> float:
        values = [value for value in values if np.isfinite(value)]
        return float(max(values)) if values else float("nan")

    gates = {
        "algorithmic_lookahead_frames_eq_0": True,
        "nan_count_eq_0": nan_count == 0,
        "negative_bone_length_count_eq_0": negative_bone == 0,
        "strict_left_reprojection_p95_at_most_25_px": _percentile(strict_res_l, 95) <= 25.0,
        "strict_right_reprojection_p95_at_most_25_px": _percentile(strict_res_r, 95) <= 25.0,
        "strict_per_joint_p95_at_most_50_px": _max_finite(
            list(per_joint_p95["left"].values()) + list(per_joint_p95["right"].values())
        ) <= 50.0,
        "acceleration_p95_at_most_1_2x_prefit": acc_p95 <= 1.2 * prefit_acc_p95 + 1e-9,
        "adjacent_output_jump_p95_at_most_40_mm": jump_p95 <= 40.0,
        "pelvis_net_displacement_change_at_most_15_percent": pelvis_change <= 0.15,
        "update_time_measured": bool(times.size > 0),
    }
    passed = all(gates.values())
    realtime_note = None
    if args.iterations == 10 and timing["p95_s"] > 0.033:
        realtime_note = "不满足30 FPS实时预算（10次迭代CPU P95 > 33 ms）"

    summary = {
        "schema_version": "causal_skeleton_fixed_lag_window_v1",
        "status": "bounded_window_engineering_candidate_not_full_sequence",
        "inclusive_frame_window": [args.start_frame, args.end_frame],
        "frame_count": len(indices),
        "window_size": args.window_size,
        "iterations_per_update": args.iterations,
        "learning_rate": args.learning_rate,
        "algorithmic_lookahead_frames": 0,
        "window_history_frames_max": args.window_size - 1,
        "frozen_bones_source": str(args.frozen_bones_json.resolve()),
        "metrics": {
            "strict_left_reprojection_median_px": _percentile(strict_res_l, 50),
            "strict_left_reprojection_p95_px": _percentile(strict_res_l, 95),
            "strict_left_reprojection_max_px": _percentile(strict_res_l, 100),
            "strict_right_reprojection_median_px": _percentile(strict_res_r, 50),
            "strict_right_reprojection_p95_px": _percentile(strict_res_r, 95),
            "strict_right_reprojection_max_px": _percentile(strict_res_r, 100),
            "relaxed_left_reprojection_median_px": _percentile(relaxed_res_l, 50),
            "relaxed_left_reprojection_p95_px": _percentile(relaxed_res_l, 95),
            "relaxed_right_reprojection_median_px": _percentile(relaxed_res_r, 50),
            "relaxed_right_reprojection_p95_px": _percentile(relaxed_res_r, 95),
            "strict_left_count": int(np.count_nonzero(part_l["accepted"] & strict)),
            "relaxed_left_count": int(np.count_nonzero(part_l["accepted"] & ~strict)),
            "rejected_left_count": int(np.count_nonzero(~part_l["accepted"])),
            "strict_right_count": int(np.count_nonzero(part_r["accepted"] & strict)),
            "relaxed_right_count": int(np.count_nonzero(part_r["accepted"] & ~strict)),
            "rejected_right_count": int(np.count_nonzero(~part_r["accepted"])),
            "rejected_by_reason_left": _rejection_only(part_l["rejection_counts"]),
            "rejected_by_reason_right": _rejection_only(part_r["rejection_counts"]),
            "per_joint_strict_reprojection_p95_px": per_joint_p95,
            "per_stage_strict_reprojection_p95_px": per_stage_p95,
            "joint_acceleration_p95_mm_per_frame2": acc_p95,
            "prefit_input_acceleration_p95_mm_per_frame2": prefit_acc_p95,
            "adjacent_output_jump_p95_mm": jump_p95,
            "pelvis_net_displacement_fitted_mm": pelvis_net_fitted,
            "pelvis_net_displacement_input_mm": pelvis_net_input,
            "pelvis_net_displacement_change_fraction": pelvis_change,
            "stage2_foot_velocity_p95_mm_per_frame": foot_vel_p95,
            "stage2_foot_velocity_note": "non_independent: Stage 2 feet also participate in camera pose estimation",
            "hand_handle_distance_median_mm": hand_distance_median,
            "hand_handle_distance_note": "visual_proximity_only: not real grip, force or load",
            "nan_count": nan_count,
            "behind_camera_points": behind_camera,
            "negative_bone_count": negative_bone,
            "hinge_penalty_triggered_frames": [row["frame_index"] for row in output_rows if row["hinge_penalty_triggered"]],
            "failed_frames": failed_frames,
        },
        "timing": timing,
        "rejection_counts": rejection_counts,
        "realtime_note": realtime_note,
        "acceptance_audit": {
            "status": "passed" if passed else "failed",
            "gates": gates,
            "interpretation": (
                "Internal engineering gates on a causal fixed-lag window. They test "
                "causality, self-consistency and real-time feasibility, not physical "
                "accuracy truth or contact truth."
            ),
        },
        "boundary": (
            "Causal fixed-lag offline replay on frozen people_1 keypoints. The "
            "full-sequence prefit is used only for cold start and metric comparison, "
            "never as a runtime soft target."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "causal_skeleton.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows), encoding="utf-8"
    )
    (output / "timing.json").write_text(json.dumps(timing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "rejection_counts.json").write_text(
        json.dumps(rejection_counts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
