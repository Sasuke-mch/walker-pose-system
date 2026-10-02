#!/usr/bin/env python3
"""Fuse stage-conditioned visual rotation and stationary feet into full SE(3)."""

from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
from walker_tools._compat import APP_ROOT as _tool_app_root
_tool_prepare_imports()

import argparse
from collections import Counter, deque
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


APP_ROOT = _tool_app_root

from pose_app.dynamic_ground_pose import (  # noqa: E402
    RotationFootPoseCriteria,
    solve_pose_from_relative_rotation_and_feet,
)
from pose_app.static_ground_reference import StaticGroundReference  # noqa: E402
from pose_app.two_stage_motion import WorldPose  # noqa: E402


STAGE2 = "stage2_feet_static_walker_moving"
STAGE1 = "stage1_walker_static_human_moving"
TRANSITION = "transition"
SCHEMA = "stage_conditioned_visual_rotation_two_foot_full_se3_v3"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def finite_joint_map(row: dict, *, valid_only: bool) -> dict[int, np.ndarray]:
    people = row.get("persons_3d") or []
    if len(people) != 1:
        return {}
    result = {}
    for point in people[0].get("keypoints_3d", []):
        xyz = np.asarray(point.get("xyz", []), dtype=np.float64)
        if xyz.shape == (3,) and np.all(np.isfinite(xyz)) and (not valid_only or bool(point.get("valid"))):
            result[int(point["index"])] = xyz
    return result


def joint_rows(row: dict) -> dict[int, dict]:
    people = row.get("persons_3d") or []
    if len(people) != 1:
        return {}
    return {int(point["index"]): point for point in people[0].get("keypoints_3d", [])}


def stage_name(row: dict) -> str:
    stage = row.get("stage") or {}
    return str(stage.get("operational") or stage.get("confirmed") or "unknown")


def rotation_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first @ second.T
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))))


def settle_pose_on_ground(pose: WorldPose, reference: WorldPose, gain: float = 1.0) -> WorldPose:
    """Restore measured height/roll/pitch while preserving ground XY and yaw."""

    relative = pose.rotation_world_from_camera @ reference.rotation_world_from_camera.T
    yaw = float(np.arctan2(relative[1, 0], relative[0, 0]))
    cosine, sine = np.cos(yaw), np.sin(yaw)
    yaw_rotation = np.asarray(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))
    target_rotation = yaw_rotation @ reference.rotation_world_from_camera
    relative_to_target = target_rotation @ pose.rotation_world_from_camera.T
    vector, _ = cv2.Rodrigues(relative_to_target)
    delta, _ = cv2.Rodrigues(gain * vector)
    translation = pose.translation_world_from_camera_mm.copy()
    translation[2] += gain * (reference.translation_world_from_camera_mm[2] - translation[2])
    return WorldPose(delta @ pose.rotation_world_from_camera, translation)


def accepted_rotation(edge: dict, maximum_agreement_deg: float) -> tuple[np.ndarray | None, dict]:
    fast_status = edge.get("rotation_status")
    if isinstance(fast_status, dict):
        if fast_status.get("status") != "accepted" or edge.get("R_to_from") is None:
            return None, dict(fast_status)
        rotation = np.asarray(edge["R_to_from"], dtype=np.float64)
        return rotation, {
            **fast_status,
            "source": "independent_left_right_fisheye_rotation_consensus",
            "fast_rotation_edge_processing_ms": edge.get("edge_processing_ms"),
        }
    pnp = edge.get("three_d_two_d_pnp") or {}
    three_d = edge.get("three_d_three_d") or {}
    if pnp.get("status") != "candidate_accepted" or three_d.get("status") != "candidate_accepted":
        return None, {"status": "unavailable", "reason": "both_visual_rotation_backends_required"}
    pnp_rotation = np.asarray(pnp.get("R_to_from"), dtype=np.float64)
    three_d_rotation = np.asarray(three_d.get("R_to_from"), dtype=np.float64)
    if pnp_rotation.shape != (3, 3) or three_d_rotation.shape != (3, 3):
        return None, {"status": "unavailable", "reason": "rotation_matrix_missing"}
    agreement = rotation_angle_deg(pnp_rotation, three_d_rotation)
    if agreement > maximum_agreement_deg:
        return None, {
            "status": "rejected", "reason": "visual_rotation_backends_disagree",
            "rotation_backend_disagreement_deg": agreement,
        }
    return pnp_rotation, {
        "status": "accepted", "source": "rectified_pnp_rotation_cross_checked_by_3d3d",
        "rotation_backend_disagreement_deg": agreement,
        "pnp_inlier_count": pnp.get("ransac_inlier_count"),
        "pnp_inlier_fraction": pnp.get("ransac_inlier_fraction"),
        "three_d_inlier_count": three_d.get("ransac_inlier_count"),
        "three_d_inlier_fraction": three_d.get("ransac_inlier_fraction"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-stereo-jsonl", type=Path, required=True)
    parser.add_argument("--display-completion-jsonl", type=Path, required=True)
    parser.add_argument("--stage-jsonl", type=Path, required=True)
    parser.add_argument("--visual-se3-jsonl", type=Path, required=True)
    parser.add_argument("--ground-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-window", type=int, default=5)
    parser.add_argument("--maximum-rotation-agreement-deg", type=float, default=1.5)
    parser.add_argument("--maximum-translation-step-mm", type=float, default=120.0)
    parser.add_argument("--maximum-rotation-step-deg", type=float, default=5.0)
    parser.add_argument("--smoothing-gain", type=float, default=0.45)
    parser.add_argument("--foot-observation-mode", choices=("current", "causal_median"), default="causal_median")
    parser.add_argument("--maximum-two-foot-translation-disagreement-mm", type=float, default=60.0)
    parser.add_argument("--minimum-stage2-vote-confidence", type=float, default=0.0)
    parser.add_argument("--maximum-stage2-vote-dropout-frames", type=int, default=2)
    parser.add_argument("--maximum-stage2-transition-carry-frames", type=int, default=4)
    parser.add_argument("--maximum-recovery-gap-frames", type=int, default=4)
    parser.add_argument("--maximum-single-foot-prediction-error-mm", type=float, default=80.0)
    parser.add_argument("--minimum-single-foot-prediction-advantage-mm", type=float, default=20.0)
    parser.add_argument("--disable-single-foot-recovery", action="store_true")
    parser.add_argument("--rotation-mode", choices=("visual_cross_checked", "fixed_reference"), default="visual_cross_checked")
    parser.add_argument("--no-stage1-ground-settle", action="store_true")
    parser.add_argument(
        "--stage1-landed-support-reanchor", action="store_true",
        help="For a rigid no-wheel walker, restore measured support height/roll/pitch only on confirmed Stage-1 entry.",
    )
    parser.add_argument("--allow-rotation-hold-fallback", action="store_true")
    parser.add_argument("--transition-settle-gain", type=float, default=0.35)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.transition_settle_gain <= 1.0:
        raise ValueError("transition settle gain must be in (0, 1]")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    strict_rows = read_jsonl(args.strict_stereo_jsonl)
    complete_rows = read_jsonl(args.display_completion_jsonl)
    stage_rows = read_jsonl(args.stage_jsonl)
    visual_rows = read_jsonl(args.visual_se3_jsonl)
    if not (len(strict_rows) == len(complete_rows) == len(stage_rows) == len(visual_rows)):
        raise ValueError("all four frame streams must have identical length")
    if args.anchor_window < 1:
        raise ValueError("anchor window must be positive")
    reference = StaticGroundReference.load(args.ground_reference)
    initial = WorldPose(reference.rotation_ground_from_left, reference.translation_ground_from_left_mm)
    pose = initial
    criteria = RotationFootPoseCriteria(
        maximum_translation_step_mm=args.maximum_translation_step_mm,
        maximum_rotation_step_deg=args.maximum_rotation_step_deg,
        maximum_two_foot_translation_disagreement_mm=args.maximum_two_foot_translation_disagreement_mm,
        maximum_recovery_gap_frames=args.maximum_recovery_gap_frames,
        maximum_single_foot_prediction_error_mm=args.maximum_single_foot_prediction_error_mm,
        minimum_single_foot_prediction_advantage_mm=args.minimum_single_foot_prediction_advantage_mm,
        smoothing_gain=args.smoothing_gain,
    )
    if not 0.0 <= args.minimum_stage2_vote_confidence <= 1.0:
        raise ValueError("minimum Stage-2 vote confidence must be in [0, 1]")
    if args.maximum_stage2_vote_dropout_frames < 0:
        raise ValueError("maximum Stage-2 vote dropout frames must be non-negative")
    if args.maximum_stage2_transition_carry_frames < 0:
        raise ValueError("maximum Stage-2 transition carry frames must be non-negative")
    recent_non_stage2_feet: deque[dict[str, np.ndarray]] = deque(maxlen=args.anchor_window)
    recent_stage2_feet: deque[dict[str, np.ndarray]] = deque(maxlen=args.anchor_window)
    anchors = None
    previous_stage = None
    ground_settling = False
    records = []
    statuses: Counter[str] = Counter()
    stage2_reasons: Counter[str] = Counter()
    transition_carry_statuses: Counter[str] = Counter()
    stage2_low_vote_streak = 0
    frames_since_pose_update = 0
    accepted_translation_history: deque[tuple[int, np.ndarray]] = deque(maxlen=2)
    stage2_pose_latched = False
    stage2_transition_carry_frames = 0
    previous_pose_active = False
    started = time.perf_counter()
    for strict, complete, stage_row, edge in zip(strict_rows, complete_rows, stage_rows, visual_rows):
        pair_id = int(strict["pair_id"])
        if len({pair_id, int(complete["pair_id"]), int(stage_row["pair_id"]), int(edge["to_pair_id"])}) != 1:
            raise ValueError(f"unaligned pair at {pair_id}")
        stage = stage_name(stage_row)
        if stage == STAGE2:
            pose_active = True
            stage2_pose_latched = True
            stage2_transition_carry_frames = 0
        elif (
            stage == TRANSITION
            and stage2_pose_latched
            and stage2_transition_carry_frames < args.maximum_stage2_transition_carry_frames
        ):
            stage2_transition_carry_frames += 1
            pose_active = True
        else:
            pose_active = False
            if stage != TRANSITION or stage2_transition_carry_frames >= args.maximum_stage2_transition_carry_frames:
                stage2_pose_latched = False
        strict_points = finite_joint_map(strict, valid_only=True)
        feet = {"left": strict_points[15], "right": strict_points[16]} if 15 in strict_points and 16 in strict_points else None
        if args.rotation_mode == "fixed_reference":
            rotation = np.eye(3, dtype=np.float64)
            rotation_audit = {
                "status": "accepted", "source": "fixed_measured_reference_orientation",
                "rotation_backend_disagreement_deg": None,
            }
        else:
            rotation, rotation_audit = accepted_rotation(edge, args.maximum_rotation_agreement_deg)
        audit: dict = {"status": "held", "reason": "stage1_or_transition_pose_hold"}
        if not pose_active:
            stage2_low_vote_streak = 0
            frames_since_pose_update = 0
            accepted_translation_history.clear()
            recent_stage2_feet.clear()
            if previous_pose_active and stage == TRANSITION and not args.stage1_landed_support_reanchor:
                ground_settling = True
            if ground_settling and stage == TRANSITION and not args.no_stage1_ground_settle and not args.stage1_landed_support_reanchor:
                pose = settle_pose_on_ground(pose, initial, args.transition_settle_gain)
                audit = {"status": "settling", "reason": None, "source": "transition_ground_height_roll_pitch_settle"}
            if stage == STAGE1 and previous_stage != STAGE1 and args.stage1_landed_support_reanchor:
                pose = settle_pose_on_ground(pose, initial)
                audit = {
                    "status": "landed_support_reanchored", "reason": None,
                    "source": "confirmed_stage1_rigid_no_wheel_support_height_roll_pitch",
                }
                ground_settling = False
            elif stage == STAGE1 and previous_stage != STAGE1 and not args.no_stage1_ground_settle:
                pose = settle_pose_on_ground(pose, initial)
                audit = {"status": "settled", "reason": None, "source": "confirmed_stage1_ground_height_roll_pitch_reanchor"}
                ground_settling = False
            anchors = None
            if feet is not None:
                recent_non_stage2_feet.append({name: pose.transform(point) for name, point in feet.items()})
        else:
            ground_settling = False
            stage_vote_confidence = (
                float((stage_row.get("stage") or {}).get("vote_confidence") or 0.0)
                if stage == STAGE2 else 1.0
            )
            if not previous_pose_active:
                stage2_low_vote_streak = 0
                frames_since_pose_update = 0
                accepted_translation_history.clear()
                accepted_translation_history.append((pair_id, pose.translation_world_from_camera_mm.copy()))
                recent_stage2_feet.clear()
                if recent_non_stage2_feet:
                    anchors = {
                        name: np.median(np.asarray([item[name] for item in recent_non_stage2_feet]), axis=0)
                        for name in ("left", "right")
                    }
                recent_non_stage2_feet.clear()
            if feet is not None:
                recent_stage2_feet.append(feet)
            filtered_feet = None
            if feet is not None and args.foot_observation_mode == "current":
                filtered_feet = feet
            elif feet is not None and recent_stage2_feet:
                filtered_feet = {
                    name: np.median(np.asarray([item[name] for item in recent_stage2_feet]), axis=0)
                    for name in ("left", "right")
                }
            low_vote = stage == STAGE2 and stage_vote_confidence < args.minimum_stage2_vote_confidence
            if low_vote:
                stage2_low_vote_streak += 1
            else:
                stage2_low_vote_streak = 0
            vote_grace = bool(
                low_vote
                and previous_pose_active
                and stage2_low_vote_streak <= args.maximum_stage2_vote_dropout_frames
            )
            if anchors is None:
                audit = {"status": "unavailable", "reason": "no_pre_stage2_two_foot_anchor"}
            elif feet is None or filtered_feet is None:
                audit = {"status": "unavailable", "reason": "strict_two_foot_input_missing"}
            elif low_vote and not vote_grace:
                audit = {
                    "status": "rejected", "reason": "stage2_vote_confidence_below_gate",
                    "stage2_vote_confidence": stage_vote_confidence,
                    "minimum_stage2_vote_confidence": args.minimum_stage2_vote_confidence,
                }
            elif rotation is None and not args.allow_rotation_hold_fallback:
                audit = dict(rotation_audit)
            else:
                used_rotation_hold = rotation is None
                translation_prediction = None
                if len(accepted_translation_history) == 2:
                    first_pair, first_translation = accepted_translation_history[0]
                    last_pair, last_translation = accepted_translation_history[1]
                    interval = max(1, last_pair - first_pair)
                    velocity = (last_translation - first_translation) / interval
                    translation_prediction = last_translation + velocity * max(1, pair_id - last_pair)
                solved, pose_audit = solve_pose_from_relative_rotation_and_feet(
                    pose, np.eye(3) if used_rotation_hold else rotation, filtered_feet, anchors, criteria,
                    elapsed_frames_since_update=frames_since_pose_update + 1,
                    translation_prediction_world_mm=translation_prediction,
                    allow_single_foot_recovery=not args.disable_single_foot_recovery,
                )
                audit = {**rotation_audit, **pose_audit}
                audit["causal_foot_window_frames"] = len(recent_stage2_feet)
                audit["foot_observation_mode"] = args.foot_observation_mode
                audit["current_frame_two_feet_required"] = True
                audit["stage2_vote_confidence"] = stage_vote_confidence
                audit["stage2_vote_grace_applied"] = vote_grace
                audit["stage2_low_vote_streak"] = stage2_low_vote_streak
                audit["stage2_transition_carry_applied"] = stage == TRANSITION
                audit["stage2_transition_carry_frame"] = (
                    stage2_transition_carry_frames if stage == TRANSITION else 0
                )
                if solved is not None and used_rotation_hold:
                    audit["status"] = "accepted_rotation_held"
                    audit["source"] = "two_stationary_feet_xyz_with_previous_rotation_held"
                    audit["rotation_hold_reason"] = rotation_audit.get("reason")
                if solved is not None:
                    pose = solved
                    frames_since_pose_update = 0
                    accepted_translation_history.append((pair_id, pose.translation_world_from_camera_mm.copy()))
            if audit.get("status") in {"unavailable", "rejected"}:
                frames_since_pose_update += 1
            if stage == STAGE2:
                statuses[audit["status"]] += 1
                if audit["status"] != "accepted":
                    stage2_reasons[str(audit.get("reason"))] += 1
            else:
                transition_carry_statuses[audit["status"]] += 1
        display = finite_joint_map(complete, valid_only=False)
        camera_points = dict(display)
        camera_points.update(strict_points)
        strict_joint_rows = joint_rows(strict)
        complete_joint_rows = joint_rows(complete)
        point_provenance = {}
        for index in sorted(camera_points):
            strict_item = strict_joint_rows.get(index)
            complete_item = complete_joint_rows.get(index)
            selected = strict_item if index in strict_points else complete_item
            point_provenance[str(index)] = {
                "source": "strict_stereo" if index in strict_points else "force_all_stereo_display",
                "strict_valid": bool(strict_item and strict_item.get("valid")),
                "strict_rejection_reason": None if strict_item is None else strict_item.get("reason"),
                "quality_flags": [] if selected is None else list(selected.get("quality_flags") or []),
                "reprojection_error_mean_px": None if selected is None else selected.get("reprojection_error_mean_px"),
                "used_for_pose": bool(index in (15, 16) and index in strict_points and pose_active),
            }
        ground_points = {index: pose.transform(point) for index, point in camera_points.items()}
        raw_ground_points = {index: point.copy() for index, point in ground_points.items()}
        projection = {"applied": False, "type": None}
        if pose_active and anchors is not None:
            projected = []
            for name, index in (("left", 15), ("right", 16)):
                if index in ground_points:
                    ground_points[index] = anchors[name].copy()
                    projected.append(index)
            if projected:
                projection = {
                    "applied": True,
                    "type": "stage2_stationary_ankle_constraint_for_output",
                    "projected_joint_indices": projected,
                    "estimator_input": "causal_median_strict_feet",
                    "raw_measurements_preserved": True,
                }
        records.append({
            "schema_version": SCHEMA, "pair_id": pair_id, "stage": stage,
            **pose.as_mapping(), "pose_audit": audit, "rotation_audit": rotation_audit,
            "strict_feet_available": feet is not None,
            "raw_points_ground_mm": {str(index): point.tolist() for index, point in raw_ground_points.items()},
            "points_ground_mm": {str(index): point.tolist() for index, point in ground_points.items()},
            "point_provenance": point_provenance,
            "constraint_projection": projection,
            "estimator_policy": {
                "foot_observation_mode": args.foot_observation_mode,
                "current_frame_two_feet_required": True,
                "minimum_stage2_vote_confidence": args.minimum_stage2_vote_confidence,
                "maximum_stage2_vote_dropout_frames": args.maximum_stage2_vote_dropout_frames,
                "maximum_stage2_transition_carry_frames": args.maximum_stage2_transition_carry_frames,
                "maximum_two_foot_translation_disagreement_mm": args.maximum_two_foot_translation_disagreement_mm,
                "maximum_recovery_gap_frames": args.maximum_recovery_gap_frames,
                "single_foot_recovery": not args.disable_single_foot_recovery,
                "smoothing_gain": args.smoothing_gain,
                "stage1_ground_settle": not args.no_stage1_ground_settle,
                "stage1_landed_support_reanchor": args.stage1_landed_support_reanchor,
            },
            "production_eligible": False,
            "production_blocker": "visual walker exclusion is not identity-certified and no external trajectory truth exists",
        })
        previous_stage = stage
        previous_pose_active = pose_active
    elapsed = time.perf_counter() - started
    records_path = output / "full_se3_dynamic_ground.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records), encoding="utf-8"
    )
    cameras = np.asarray([row["translation_world_from_left_camera_mm"] for row in records])
    accepted = [
        row for row in records
        if row["pose_audit"].get("status") in {"accepted", "accepted_rotation_held"}
    ]
    accepted_z = np.asarray([row["translation_world_from_left_camera_mm"][2] for row in accepted])
    rotations = [np.asarray(row["rotation_world_from_left_camera"]) for row in records]
    net_rotation = rotation_angle_deg(rotations[-1], rotations[0])
    summary = {
        "schema_version": SCHEMA, "status": "engineering_candidate_not_production_eligible",
        "frame_count": len(records), "stage2_status_counts": dict(statuses),
        "stage2_transition_carry_status_counts": dict(transition_carry_statuses),
        "stage2_failure_reasons": dict(stage2_reasons),
        "stage2_pose_updated_fraction": (
            statuses["accepted"] + statuses["accepted_rotation_held"]
        ) / max(1, sum(statuses.values())),
        "camera_net_translation_mm": (cameras[-1] - cameras[0]).tolist(),
        "camera_net_horizontal_displacement_mm": float(np.linalg.norm((cameras[-1] - cameras[0])[:2])),
        "camera_z_min_max_mm": [float(cameras[:, 2].min()), float(cameras[:, 2].max())],
        "accepted_camera_z_median_mm": float(np.median(accepted_z)) if len(accepted_z) else None,
        "accepted_camera_z_p05_p95_mm": [float(np.percentile(accepted_z, 5)), float(np.percentile(accepted_z, 95))] if len(accepted_z) else None,
        "camera_net_rotation_deg": net_rotation,
        "processing_ms_per_frame_excluding_visual_frontend": elapsed * 1000.0 / max(1, len(records)),
        "production_eligible": False,
        "production_blocker": "visual walker exclusion is not identity-certified and no external trajectory truth exists",
        "rotation_mode": args.rotation_mode,
        "estimator_policy": {
            "foot_observation_mode": args.foot_observation_mode,
            "current_frame_two_feet_required": True,
            "minimum_stage2_vote_confidence": args.minimum_stage2_vote_confidence,
            "maximum_stage2_vote_dropout_frames": args.maximum_stage2_vote_dropout_frames,
            "maximum_stage2_transition_carry_frames": args.maximum_stage2_transition_carry_frames,
            "maximum_two_foot_translation_disagreement_mm": args.maximum_two_foot_translation_disagreement_mm,
            "maximum_recovery_gap_frames": args.maximum_recovery_gap_frames,
            "single_foot_recovery": not args.disable_single_foot_recovery,
            "smoothing_gain": args.smoothing_gain,
            "stage1_ground_settle": not args.no_stage1_ground_settle,
            "stage1_landed_support_reanchor": args.stage1_landed_support_reanchor,
        },
        "interpretation": (
            "Confirmed Stage 1 holds ground XY/yaw. For the rigid no-wheel configuration, an enabled landed-support "
            "reanchor restores only the measured support height/roll/pitch on confirmed Stage-1 entry; it is not a "
            "display smoother. A bounded post-Stage-2 transition carry continues the stationary-feet solver during "
            "the physical landing interval. Stage 2 requires current-frame strict feet and solves XYZ after a "
            "short vote-dropout grace and per-foot consistency checks. Translation step limits scale only with the "
            "number of frames since the last accepted update, preventing stale-pose deadlock. A single-foot recovery "
            "is allowed only when the recent accepted motion prediction clearly selects one foot. "
            + ("Rotation comes from the supplied cross-checked visual rotation stream when available. " if args.rotation_mode == "visual_cross_checked" else
               "Rotation is held at the last measured orientation, so this is 3-D translation rather than unconstrained 6-DoF. ")
            + ("Missing rotation edges explicitly hold the previous rotation while still updating XYZ. " if args.allow_rotation_hold_fallback else "")
            + "Feet are estimator inputs and not independent validation. Raw transformed points are preserved; "
            + "the Stage-2 display/output ankle points are explicitly projected to their stationary anchors."
        ),
        "records": str(records_path.resolve()),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "run_metadata.json").write_text(json.dumps({
        "inputs": {name: str(getattr(args, name).resolve()) for name in (
            "strict_stereo_jsonl", "display_completion_jsonl", "stage_jsonl", "visual_se3_jsonl", "ground_reference"
        )}, "parameters": {key: value for key, value in vars(args).items() if not isinstance(value, Path)},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
