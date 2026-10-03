"""Compare frozen-body scene transforms without claiming external accuracy."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def groups(mask):
    indexes = np.flatnonzero(mask)
    return (
        np.split(indexes, np.flatnonzero(np.diff(indexes) > 1) + 1)
        if len(indexes)
        else []
    )


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


def audit(root, output, experiment_root=None):
    experiment_root = experiment_root or root / "stage_continuous_v1"
    fit = np.load(root / "pmpose/fit/result.npz", allow_pickle=True)
    control = np.load(
        root / "scene_rotation_fix_v1/scene_transforms.npz", allow_pickle=True
    )
    reference_stage = control["stage"]
    intervals = groups(reference_stage == "stage2_feet_static_walker_moving")
    result = {
        "evidence": "frozen body and internal operational stage intervals; not physical or 3D accuracy",
        "variants": {},
    }
    for name, path in [("control", root / "scene_rotation_fix_v1")] + [
        (s, experiment_root / s)
        for s in ["nominal_current", "host_current", "continuous_checked"]
    ]:
        data = np.load(path / "scene_transforms.npz", allow_pickle=True)
        rows = [
            json.loads(s)
            for s in (path / "dynamic_ground_pose.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        if (
            len(rows) != len(fit["pair_id"])
            or [r["pair_id"] for r in rows] != fit["pair_id"].tolist()
        ):
            raise ValueError("missing/reordered output frames")
        R, t = (
            data["rotation_ground_from_left"],
            data["translation_ground_from_left_mm"],
        )
        if (
            not np.isfinite(R).all()
            or not np.isfinite(t).all()
            or np.max(abs(np.linalg.det(R) - 1)) > 1e-6
        ):
            raise ValueError("invalid world transforms")
        if not np.array_equal(control["accepted"], data["accepted"]) or not np.allclose(
            control["raw_triangulated_points_mm"],
            data["raw_triangulated_points_mm"],
            equal_nan=True,
        ):
            raise ValueError("raw body geometry changed")
        world = np.einsum("fij,fkj->fki", R, fit["predicted_coco"] * 1000) + t[:, None]
        pelvis = world[:, [11, 12]].mean(1)
        step = np.linalg.norm(np.diff(t, axis=0), axis=1)
        switch = np.flatnonzero(reference_stage[1:] != reference_stage[:-1])
        details = []
        for ids in intervals:
            if len(ids) < 2:
                continue
            foot_span = [
                float(np.max(np.linalg.norm(world[ids, j] - world[ids[0], j], axis=1)))
                for j in [15, 16]
            ]
            pelvis_span = float(
                np.max(np.linalg.norm(pelvis[ids] - pelvis[ids[0]], axis=1))
            )
            details.append(
                {
                    "first": int(ids[0]),
                    "last": int(ids[-1]),
                    "frames": len(ids),
                    "fitted_ankle_excursion_mm": foot_span,
                    "fitted_pelvis_excursion_mm": pelvis_span,
                    "unavailable": sum(
                        rows[i]["pose_audit"]["status"] == "unavailable" for i in ids
                    ),
                }
            )
        stage1_step = step[
            (data["stage"][1:] == "stage1_walker_static_human_moving")
            & (data["stage"][:-1] == "stage1_walker_static_human_moving")
        ]
        result["variants"][name] = {
            "frame_count": len(rows),
            "status": dict(Counter(r["pose_audit"]["status"] for r in rows)),
            "reasons": dict(Counter(str(r["pose_audit"].get("reason")) for r in rows)),
            "stage_counts": dict(Counter(data["stage"].tolist())),
            "camera_translation_step_mm": stats(step),
            "baseline_stage_switch_translation_mm": stats(step[switch]),
            "consecutive_stage1_camera_step_mm": stats(stage1_step),
            "baseline_stage2_intervals": details,
            "world_fitted_ankle_step_mm": stats(
                np.linalg.norm(np.diff(world[:, [15, 16]], axis=0), axis=2).ravel()
            ),
            "estimate_processing_ms": stats([r["processing_ms"] for r in rows]),
        }
        if "world_rotation_chain_complete" in rows[0]["pose_audit"]:
            result["variants"][name]["world_chain_complete_frames"] = sum(
                r["pose_audit"]["world_rotation_chain_complete"] for r in rows
            )
            result["variants"][name]["world_valid_ankle_constraint_frames"] = sum(
                r["ankle_constraint_weight"] > 0 for r in rows
            )
            result["variants"][name]["sources"] = dict(
                Counter(r["pose_audit"]["source"] for r in rows)
            )
    timing = json.loads((root / "frames.json").read_text())
    pair_times = np.asarray(
        [
            (
                int(r["left_host_return_timestamp_ns"])
                + int(r["right_host_return_timestamp_ns"])
            )
            / 2e9
            for r in timing
        ]
    )
    result["timing"] = {
        "nominal_span_sec": (len(fit["pair_id"]) - 1) / 30,
        "host_span_sec": float(pair_times[-1] - pair_times[0]),
        "host_interval_ms": stats(np.diff(pair_times) * 1000),
        "host_skew_ms": stats([r["abs_host_delta_ms"] for r in timing]),
        "exposure_sync_verified": False,
    }
    vo = [
        json.loads(s)
        for s in (experiment_root / "continuous_checked/background_vo_diagnostic.jsonl")
        .read_text()
        .splitlines()
    ]
    result["background_vo"] = {
        "status": dict(Counter(r["status"] for r in vo)),
        "reasons": dict(Counter(str(r.get("reason")) for r in vo)),
        "processing_ms": stats([r["processing_ms"] for r in vo]),
        "withheld_p95_normalized": stats(
            [r["withheld_p95_normalized"] for r in vo if "withheld_p95_normalized" in r]
        ),
        "injected_into_ground": False,
        "mask_certified": False,
        "accepted_for_mainline": False,
        "maximum_correspondence_count": max(
            r.get("correspondence_count", 0) for r in vo
        ),
        "geometry_inlier_count": stats([r["frontend"]["geometry_inliers"] for r in vo]),
    }
    result["accepted_for_mainline"] = False
    result["experiment_root"] = str(experiment_root)
    rectified_path = (
        experiment_root / "continuous_rectified/background_vo_diagnostic.jsonl"
    )
    if rectified_path.exists():
        rows = [json.loads(s) for s in rectified_path.read_text().splitlines()]
        result["background_vo_rectified"] = {
            "status": dict(Counter(r["status"] for r in rows)),
            "reasons": dict(Counter(r["reason"] for r in rows)),
            "maximum_correspondence_count": max(
                r.get("correspondence_count", 0) for r in rows
            ),
            "processing_ms": stats([r["processing_ms"] for r in rows]),
            "injected_into_ground": False,
            "accepted_for_mainline": False,
        }
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--experiment-root", type=Path)
    a = p.parse_args()
    print(
        json.dumps(
            audit(a.root, a.output, a.experiment_root), ensure_ascii=False, indent=2
        )
    )
