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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--experiment", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(audit(a.root, a.experiment), ensure_ascii=False, indent=2))
