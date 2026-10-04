"""Frozen-body comparison of transition treatment and background links."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def read(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]


def stats(values):
    values = np.asarray(values, float)
    return (
        {
            "count": len(values),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "maximum": float(np.max(values)),
        }
        if len(values)
        else {"count": 0}
    )


def audit(root, experiment):
    original = root / "scene_rotation_fix_v1"
    legacy = read(original / "stage1_stage2.jsonl")
    fit = np.load(root / "pmpose/fit/result.npz", allow_pickle=True)
    baseline = np.load(original / "scene_transforms.npz", allow_pickle=True)
    moving_transition = np.array(
        [
            i
            for i, r in enumerate(legacy)
            if i
            and r["stage"]["operational"] == "transition"
            and r["stage"]["background"]["fused"].get("motion_active")
        ]
    )
    transition = np.array(
        [
            i
            for i, r in enumerate(legacy)
            if i and r["stage"]["operational"] == "transition"
        ]
    )
    forward = baseline["rotation_ground_from_left"][0, :, 2].copy()
    forward[2] = 0
    forward /= np.linalg.norm(forward)
    output = {
        "evidence": "internal frozen-body diagnostics, no stage/contact/external trajectory truth",
        "backward_axis": "negative horizontal projection of initial left-camera optical forward",
        "variants": {},
    }
    for name, path in [("control", original)] + [
        (name, experiment / name)
        for name in [
            "stage_policy",
            "stage_no_reanchor",
            "stage_current",
            "stage_predict",
            "stage_bearing",
            "stage_single",
            "stage_latest_single",
            "background_wide",
            "stage_final",
        ]
    ]:
        if not (path / "scene_transforms.npz").exists():
            raise ValueError(f"missing required candidate {name}")
        data = np.load(path / "scene_transforms.npz", allow_pickle=True)
        rows = read(path / "dynamic_ground_pose.jsonl")
        read(path / "stage1_stage2.jsonl")
        if [r["pair_id"] for r in rows] != fit["pair_id"].tolist():
            raise ValueError("frame mismatch")
        if not np.array_equal(
            data["accepted"], baseline["accepted"]
        ) or not np.allclose(
            data["raw_triangulated_points_mm"],
            baseline["raw_triangulated_points_mm"],
            equal_nan=True,
        ):
            raise ValueError("body input changed")
        R, t = (
            data["rotation_ground_from_left"],
            data["translation_ground_from_left_mm"],
        )
        if (
            not np.isfinite(R).all()
            or not np.isfinite(t).all()
            or np.max(abs(R @ np.transpose(R, (0, 2, 1)) - np.eye(3))) > 1e-6
        ):
            raise ValueError("bad transforms")
        world = np.einsum("fij,fkj->fki", R, fit["predicted_coco"] * 1000) + t[:, None]
        delta = np.diff(world[:, [11, 12]].mean(1), axis=0)
        camera = np.linalg.norm(np.diff(t, axis=0), axis=1)
        same_static = (data["stage"][1:] == "stage1_walker_static_human_moving") & (
            data["stage"][:-1] == "stage1_walker_static_human_moving"
        )
        report = {
            "stages": dict(Counter(data["stage"].tolist())),
            "ground_status": dict(Counter(r["pose_audit"]["status"] for r in rows)),
            "moving_transition_backward_sum_mm": float(
                np.maximum(0, -delta[moving_transition - 1] @ forward).sum()
            ),
            "moving_transition_pelvis_step_mm": stats(
                np.linalg.norm(delta[moving_transition - 1], axis=1)
            ),
            "all_transition_pelvis_step_mm": stats(
                np.linalg.norm(delta[transition - 1], axis=1)
            ),
            "all_pelvis_step_mm": stats(np.linalg.norm(delta, axis=1)),
            "all_fitted_ankle_step_mm": stats(
                np.linalg.norm(np.diff(world[:, [15, 16]], axis=0), axis=2).ravel()
            ),
            "camera_step_mm": stats(camera),
            "consecutive_stage1_camera_step_mm": stats(camera[same_static]),
            "baseline_moving_transition_treated": sum(
                rows[i]["pose_audit"]["status"]
                in [
                    "accepted",
                    "accepted_rotation_held",
                    "predicted",
                    "accepted_single_ankle",
                    "accepted_ankle_bearings",
                ]
                for i in moving_transition
            ),
            "actions": dict(
                Counter(r["pose_audit"].get("handoff_action", "legacy") for r in rows)
            ),
            "moving_transition_frames": [
                {
                    "frame": int(i),
                    "pelvis_step_mm": float(np.linalg.norm(delta[i - 1])),
                    "backward_step_mm": float(max(0, -delta[i - 1] @ forward)),
                    "status": rows[i]["pose_audit"]["status"],
                    "reason": rows[i]["pose_audit"].get("reason"),
                }
                for i in moving_transition
            ],
        }
        if (path / "background_vo_diagnostic.jsonl").exists():
            vo = read(path / "background_vo_diagnostic.jsonl")
            report["vo"] = {
                "status": dict(Counter(r["status"] for r in vo)),
                "reasons": dict(Counter(r["reason"] for r in vo)),
                "max_links": max(
                    r.get("available_count", r.get("correspondence_count", 0))
                    for r in vo
                ),
                "max_solver_points": max(
                    r.get("final_solver_point_count", 0) for r in vo
                ),
                "processing_ms": stats([r["processing_ms"] for r in vo]),
                "pose_injected": False,
            }
        output["variants"][name] = report
    (experiment / "comparison.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def audit_landing(root, experiment):
    fit = np.load(root / "pmpose/fit/result.npz", allow_pickle=False)
    surface = (
        root.parents[0]
        / "G20260927_wilor_smplh_full448_v1/contact_surface_sets_smplh_v1/contact_vertex_sets.json"
    )
    sets = json.loads(surface.read_text(encoding="utf-8"))["sets"]
    soles = np.unique(
        [
            i
            for side in ("left", "right")
            for region in sets[f"{side}_sole_surface_candidate"].values()
            for i in region
        ]
    )
    paths = {
        "no_restore": root / "stage_transition_diagnosis_v1/stage_final",
        "hard_restore": experiment / "hard_restore",
        "causal_restore": experiment / "causal_lowerbody",
        "lag_restore": experiment / "lag_lowerbody",
    }
    report = {
        "evidence": "protocol-qualified support consistency; no contact or external trajectory truth",
        "variants": {},
    }
    baseline = np.load(paths["no_restore"] / "scene_transforms.npz", allow_pickle=True)
    stages = None
    for name, path in paths.items():
        data = np.load(path / "scene_transforms.npz", allow_pickle=True)
        for key in ("accepted", "raw_triangulated_points_mm"):
            if not np.array_equal(data[key], baseline[key], equal_nan=True):
                raise ValueError(f"changed body observations in {name}")
        if stages is None:
            stages = data["stage"]
        elif not np.array_equal(stages, data["stage"]):
            raise ValueError("stage evidence differs between recovery controls")
        R, t = (
            data["rotation_ground_from_left"],
            data["translation_ground_from_left_mm"],
        )
        if (
            np.max(abs(R @ R.transpose(0, 2, 1) - np.eye(3))) > 1e-6
            or not np.isfinite(t).all()
        ):
            raise ValueError("invalid recovered transforms")
        rows = read(path / "dynamic_ground_pose.jsonl")
        if len(rows) != 360 or [r["pair_id"] for r in rows] != fit["pair_id"].tolist():
            raise ValueError("missing or reordered recovery frames")
        if not np.allclose(R, [r["rotation_world_from_left_camera"] for r in rows]):
            raise ValueError("JSONL and NPZ rotations disagree")
        if not np.allclose(
            t, [r["translation_world_from_left_camera_mm"] for r in rows]
        ):
            raise ValueError("JSONL and NPZ translations disagree")
        world = np.einsum("fij,fkj->fki", R, fit["predicted_coco"] * 1000) + t[:, None]
        pelvis = world[:, [11, 12]].mean(1)
        steps = np.linalg.norm(np.diff(pelvis, axis=0), axis=1)
        sole_z = (
            np.einsum("fij,fkj->fki", R, fit["vertices"][:, soles] * 1000) + t[:, None]
        )[:, :, 2].min(1)
        tilt = np.degrees(
            np.arccos(np.clip(np.einsum("fi,i->f", R[:, 2], R[0, 2]), -1, 1))
        )
        static = data["stage"] == "stage1_walker_static_human_moving"
        indices = np.flatnonzero(static)
        ends = indices[np.r_[np.diff(indices) > 1, True]]
        recoveries = [r["pose_audit"].get("landing_recovery", {}) for r in rows]
        post_residuals = [
            v
            for a in recoveries
            if a.get("applied")
            for v in a.get("post_support_ankle_residual_mm", {}).values()
        ]
        pre_residuals = []
        for row in rows:
            landing = row["pose_audit"].get("landing_recovery", {})
            if not landing.get("applied"):
                continue
            pre = row["pose_audit"].get("measurement_pose_before_support")
            anchors = row.get("operational_ankle_anchors_mm") or {}
            if pre is None:
                continue
            oldR = np.asarray(pre["rotation_world_from_left_camera"])
            oldt = np.asarray(pre["translation_world_from_left_camera_mm"])
            newR = np.asarray(row["rotation_world_from_left_camera"])
            newt = np.asarray(row["translation_world_from_left_camera_mm"])
            for side, j in (("left", 15), ("right", 16)):
                if side not in landing.get("post_support_ankle_residual_mm", {}):
                    continue
                camera = newR.T @ (
                    np.asarray(row["strict_points_ground_mm"][str(j)]) - newt
                )
                pre_residuals.append(
                    float(np.linalg.norm(oldR @ camera + oldt - anchors[side]))
                )
        disagreements = [
            a["post_support_two_foot_disagreement_mm"]
            for a in recoveries
            if a.get("applied") and "post_support_two_foot_disagreement_mm" in a
        ]
        dR = R[1:] @ R[:-1].transpose(0, 2, 1)
        angle_step = np.degrees(
            np.arccos(np.clip((np.trace(dR, axis1=1, axis2=2) - 1) / 2, -1, 1))
        )
        report["variants"][name] = {
            "ground_status": dict(Counter(r["pose_audit"]["status"] for r in rows)),
            "static_ends": [
                {
                    "frame": int(i),
                    "height_bias_mm": float(t[i, 2] - t[0, 2]),
                    "tilt_deg": float(tilt[i]),
                }
                for i in ends
            ],
            "camera_height_mm": stats(t[:, 2]),
            "maximum_tilt_deg": float(tilt.max()),
            "pelvis_step_mm": stats(steps),
            "fitted_ankle_step_mm": stats(
                np.linalg.norm(np.diff(world[:, [15, 16]], axis=0), axis=2).ravel()
            ),
            "camera_step_mm": stats(np.linalg.norm(np.diff(t, axis=0), axis=1)),
            "static_camera_step_mm": stats(
                np.linalg.norm(np.diff(t, axis=0), axis=1)[static[1:] & static[:-1]]
            ),
            "rotation_step_deg": stats(angle_step),
            "sole_minimum_z_mm": stats(sole_z),
            "segments": {
                f"{a}-{b}": float(
                    np.linalg.norm(pelvis[a : b + 1] - pelvis[a], axis=1).max()
                )
                for a, b in [
                    (85, 94),
                    (123, 136),
                    (189, 206),
                    (268, 284),
                    (288, 295),
                    (346, 359),
                ]
            },
            "support_ankle_residual_mm": stats(post_residuals),
            "same_corrected_frames_before_support_ankle_residual_mm": stats(
                pre_residuals
            ),
            "sole_minimum_over_sequence_mm": float(sole_z.min()),
            "support_two_foot_disagreement_mm": stats(disagreements),
            "support_rejection_reasons": dict(
                Counter(a["reason"] for a in recoveries if a.get("reason"))
            ),
            "release_delay_sec": stats(
                [a["release_delay_sec"] for a in recoveries if "release_delay_sec" in a]
            ),
            "all_rows_causal": all(r.get("causal", True) for r in rows),
        }
    report["strict_masks_and_stage_sequences_equal"] = True
    (experiment / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--landing", action="store_true")
    a = p.parse_args()
    print(
        json.dumps(
            (audit_landing if a.landing else audit)(a.root, a.experiment),
            ensure_ascii=False,
            indent=2,
        )
    )
