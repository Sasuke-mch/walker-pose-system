#!/usr/bin/env python3
"""Replay Stage 1/2 and T_G<-C_t from the exact fitted-body image pairs.

Only raw body observations, corresponding images, calibration, the measured static ground
reference, and the static coarse walker topology are inputs. Previous stage JSONL
files are never read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP))

from pose_app import body_observations as raw_clean
from pose_app.calibration import StereoCalibration
from pose_app.causal_stereo_background_pose import CausalStereoBackgroundPose
from pose_app.continuous_stage_ground import ContinuousStageGroundWriter
from pose_app.dynamic_ground_live import RealtimeDynamicGroundWriter
from pose_app.fisheye_camera import load_stereo_fisheye
from pose_app.realtime_stage_walker import STAGE_TWO, RealtimeStageWalkerWriter
from pose_app.replay_pair_timing import load_pair_timing
from pose_app.schema import InferenceResult, PersonPose
from pose_app.sources import SourceFrame
from pose_app.stage_handoff import StageHandoffGroundWriter, StageHandoffPolicy
from pose_app.static_background_rotation import RealtimeStereoRotationTracker
from pose_app.stereo_sources import StereoFramePair


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(allow_abbrev=False)
    profiles = json.loads(
        (APP / "configs" / "stage_ground_profiles.json").read_text(encoding="utf-8")
    )
    p.add_argument("--scene-profile", choices=["mainline", *profiles["profiles"]])
    p.add_argument("--left-video", type=Path, required=True)
    p.add_argument("--right-video", type=Path, required=True)
    p.add_argument(
        "--input-pair-dir",
        type=Path,
        default=None,
        help="The selected original stereo pair images used by PMPose. When set, this is the authoritative image stream.",
    )
    p.add_argument("--left-pmpose", type=Path, required=True)
    p.add_argument("--right-pmpose", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--ground-reference", type=Path, required=True)
    p.add_argument("--walker-model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--strict-jsonl", type=Path, required=True)
    p.add_argument("--processing-width", type=int, default=480)
    p.add_argument("--frames-metadata", type=Path)
    p.add_argument(
        "--foot-observation-mode",
        choices=["current", "causal_median"],
        default="causal_median",
    )
    p.add_argument(
        "--continuous-stage",
        action="store_true",
        help="Experimental continuous handoff; does not replace the baseline",
    )
    p.add_argument("--background-vo-diagnostic", action="store_true")
    p.add_argument("--background-vo-rectified", action="store_true")
    p.add_argument("--stage-handoff", action="store_true")
    p.add_argument("--disable-stage1-reanchor", action="store_true")
    p.add_argument("--predict-camera-dropout", action="store_true")
    p.add_argument("--background-processing-width", type=int, default=None)
    p.add_argument("--background-focal-fraction", type=float, default=0.65)
    p.add_argument("--ankle-bearing-fallback", action="store_true")
    p.add_argument("--single-ankle-fallback", action="store_true")
    p.add_argument("--latest-anchor", action="store_true")
    p.add_argument(
        "--landing-recovery",
        choices=["none", "causal", "fixed_lag", "joint_window"],
        default="none",
    )
    p.add_argument("--landing-protocol", choices=["flat_ground_supported_after_motion", "supported_at_camera_stop"])
    parsed = p.parse_args()
    parsed.resolved_scene_profile = parsed.scene_profile
    if parsed.scene_profile == "mainline":
        route = json.loads(
            (APP / "configs" / "current_mainline.json").read_text(encoding="utf-8")
        )
        parsed.resolved_scene_profile = route["scene_route"]["default_profile"]
        if parsed.resolved_scene_profile not in profiles["profiles"]:
            p.error("mainline refers to an unknown scene profile")
    if parsed.scene_profile:
        options = profiles["profiles"][parsed.resolved_scene_profile]["options"]
        supplied = {v.split("=", 1)[0] for v in sys.argv[1:] if v.startswith("--")}
        for key, value in options.items():
            flag = "--" + key.replace("_", "-")
            if flag in supplied and getattr(parsed, key) != value:
                p.error(f"{flag} conflicts with --scene-profile {parsed.scene_profile}")
            setattr(parsed, key, value)
    if parsed.landing_recovery != "none" and (
        not parsed.stage_handoff or parsed.landing_protocol is None
    ):
        p.error(
            "landing recovery requires --stage-handoff and explicit --landing-protocol"
        )
    if parsed.landing_protocol == "supported_at_camera_stop" and parsed.landing_recovery != "fixed_lag":
        p.error("supported-at-camera-stop protocol requires verified fixed-lag recovery")
    if parsed.stage_handoff and (
        (parsed.frames_metadata is None and parsed.landing_protocol != "supported_at_camera_stop")
        or parsed.continuous_stage
    ):
        p.error(
            "--stage-handoff requires frames metadata and cannot combine with --continuous-stage"
        )
    if parsed.continuous_stage and parsed.frames_metadata is None:
        p.error(
            "--continuous-stage requires actual --frames-metadata for prediction horizons"
        )
    if parsed.background_vo_rectified and not parsed.background_vo_diagnostic:
        p.error("--background-vo-rectified requires --background-vo-diagnostic")
    if (
        parsed.ankle_bearing_fallback
        or parsed.single_ankle_fallback
        or parsed.latest_anchor
        or parsed.disable_stage1_reanchor
        or parsed.predict_camera_dropout
    ) and not parsed.stage_handoff:
        p.error("handoff controls require --stage-handoff")
    if (
        parsed.background_processing_width is not None
        and parsed.background_processing_width <= 0
    ):
        p.error("background processing width must be positive")
    if not 0 < parsed.background_focal_fraction <= 1:
        p.error("background focal fraction must be in (0, 1]")
    if parsed.processing_width <= 0:
        p.error("--processing-width must be positive")
    return parsed


def read_selected_pairs(
    path: Path, count: int
) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    """Read the same ordered pair images that generated the saved PMPose rows."""
    left_dir, right_dir = path / "left_ccw90", path / "right_cw90"
    left, right = [], []
    for i in range(count):
        name = f"pair_{i:04d}.png"
        lu = cv2.imread(str(left_dir / name), cv2.IMREAD_COLOR)
        ru = cv2.imread(str(right_dir / name), cv2.IMREAD_COLOR)
        if lu is None or ru is None:
            raise FileNotFoundError(f"selected pair image missing: {name}")
        # Stage/KLT works in the original raw-camera pixel convention.
        left.append(cv2.rotate(lu, cv2.ROTATE_90_CLOCKWISE))
        right.append(cv2.rotate(ru, cv2.ROTATE_90_COUNTERCLOCKWISE))
    return left, right, 30.0


def person(points: np.ndarray, width: int, height: int) -> PersonPose:
    keypoints = np.asarray(points, dtype=np.float64).tolist()
    score = float(np.nanmean(points[:, 2])) if np.isfinite(points[:, 2]).any() else 0.0
    return PersonPose(
        0, [0.0, 0.0, float(width - 1), float(height - 1)], score, score, keypoints
    )


def strict_scene_arrays(rows, ids):
    """Export exactly the accepted first-instance input consumed by the writer."""
    if [r["index"] for r in rows] != ids:
        raise ValueError("strict frame order differs from fit")
    points = np.full((len(ids), 17, 3), np.nan, dtype=np.float64)
    mask = np.zeros((len(ids), 17), dtype=bool)
    for i, row in enumerate(rows):
        people = row["persons_3d"]
        if len(people) > 1:
            raise ValueError("strict replay requires max_matches=1")
        for person3d in people:
            if (person3d["left_person_id"], person3d["right_person_id"]) != (0, 0):
                raise ValueError("strict replay must match fitter first instances")
            seen = set()
            for point in person3d["keypoints_3d"]:
                j = point["index"]
                if not isinstance(j, int) or j not in range(17) or j in seen:
                    raise ValueError("invalid or duplicate strict joint index")
                seen.add(j)
                if point["valid"]:
                    xyz = np.asarray(point["xyz"], dtype=float)
                    if (
                        xyz.shape != (3,)
                        or not np.isfinite(xyz).all()
                        or not np.isfinite(
                            [point["depth_left"], point["depth_right"]]
                        ).all()
                        or point["depth_left"] <= 0
                        or point["depth_right"] <= 0
                    ):
                        raise ValueError("invalid accepted strict point")
                    points[i, j] = xyz
                    mask[i, j] = True
    return points, mask


def main() -> int:
    a = args()
    out = a.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refuse to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    cal = StereoCalibration.load(a.calibration)
    cal_geo = load_stereo_fisheye(a.calibration.parent)
    left_rows = raw_clean.raw_side(a.left_pmpose, "left")
    right_rows = raw_clean.raw_side(a.right_pmpose, "right")
    if set(left_rows) != set(right_rows):
        raise ValueError(
            "left/right frame identities differ; do not silently drop frames"
        )
    ids = sorted(left_rows)
    if ids != list(range(len(ids))):
        raise ValueError("raw PMPose frame ids must be contiguous from zero")
    left_points = np.stack([left_rows[i] for i in ids]).astype(np.float32)
    right_points = np.stack([right_rows[i] for i in ids]).astype(np.float32)
    (
        detector_raw,
        _el,
        _er,
        _gap,
        _dl,
        _dr,
        detector_accepted,
        _reason,
        _confidence,
        _parts,
    ) = raw_clean.raw_triangulate(left_points, right_points, cal_geo)
    strict_rows = [
        json.loads(line)
        for line in a.strict_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    raw, accepted = strict_scene_arrays(strict_rows, ids)
    if a.input_pair_dir is not None:
        left_frames, right_frames, fps_l = read_selected_pairs(
            a.input_pair_dir, len(ids)
        )
        fps_r = fps_l
        video_truncated = None
        image_source = (
            "selected original stereo pair images, inverse-rotated to raw camera pixels"
        )
    else:
        raise ValueError(
            "use --input-pair-dir: video replay requires an explicit frame mapping; prefix truncation is forbidden"
        )
    stage_writer = RealtimeStageWalkerWriter(
        out / "stage1_stage2.jsonl",
        cal,
        processing_width=a.processing_width,
        reconstruction_interval=10,
    )
    rotation_tracker = RealtimeStereoRotationTracker(
        cal, processing_width=a.processing_width
    )
    ground_writer = (
        StageHandoffGroundWriter(
            out / "dynamic_ground_pose.jsonl",
            a.ground_reference,
            foot_observation_mode=a.foot_observation_mode,
            reanchor=not a.disable_stage1_reanchor,
            prediction=a.predict_camera_dropout,
            calibration=cal,
            bearing_fallback=a.ankle_bearing_fallback,
            single_ankle_fallback=a.single_ankle_fallback,
            latest_anchor=a.latest_anchor,
            recovery_mode=a.landing_recovery,
            landing_protocol=a.landing_protocol,
        )
        if a.stage_handoff
        else ContinuousStageGroundWriter(
            out / "dynamic_ground_pose.jsonl", a.ground_reference
        )
        if a.continuous_stage
        else RealtimeDynamicGroundWriter(
            out / "dynamic_ground_pose.jsonl",
            a.ground_reference,
            pose_mode="full_se3",
            stage1_landed_support_reanchor=not a.disable_stage1_reanchor,
            foot_observation_mode=a.foot_observation_mode,
        )
    )
    handoff_policy = StageHandoffPolicy(
        preserve_motion_labels=a.landing_protocol == "supported_at_camera_stop"
    ) if a.stage_handoff else None
    timing = (
        load_pair_timing(a.frames_metadata, len(ids), input_pair_dir=a.input_pair_dir)
        if a.frames_metadata
        else [(i / fps_l, i / fps_r) for i in ids]
    )
    visual_tracker = (
        CausalStereoBackgroundPose(
            cal,
            a.background_processing_width or a.processing_width,
            rectified=a.background_vo_rectified,
            focal_fraction=a.background_focal_fraction,
        )
        if a.background_vo_diagnostic
        else None
    )
    visual_rows = []
    inference_pairs = []
    stage_rows: list[dict] = []
    ground_rows: list[dict] = []
    for i, (lf, rf) in enumerate(zip(left_frames, right_frames)):
        h, w = lf.shape[:2]
        tl, tr = timing[i]
        ts = (tl + tr) / 2
        left_result = InferenceResult(
            i, tl, w, h, "raw_pmpose_replay", 0.0, 0.0, [person(left_points[i], w, h)]
        )
        right_result = InferenceResult(
            i, tr, w, h, "raw_pmpose_replay", 0.0, 0.0, [person(right_points[i], w, h)]
        )
        inference_pairs.append((left_result, right_result))
        pair = StereoFramePair(
            pair_id=i,
            left=SourceFrame(i, tl, lf),
            right=SourceFrame(i, tr, rf),
            timestamp_skew_sec=abs(tr - tl),
            timestamp_type="host_return_not_exposure"
            if a.frames_metadata
            else "video_frame_index_over_fps",
        )
        stage_row = stage_writer.consume(pair, left_result, right_result)
        if handoff_policy:
            stage_row = handoff_policy.update(stage_row)
        stage_rows.append(stage_row)
        points = []
        people = strict_rows[i]["persons_3d"]
        if len(people) > 1:
            raise ValueError("strict replay requires max_matches=1")
        for person3d in people:
            if (person3d["left_person_id"], person3d["right_person_id"]) != (0, 0):
                raise ValueError("strict replay must match fitter first instances")
            for point in person3d["keypoints_3d"]:
                if point["valid"]:
                    if (
                        not np.isfinite(point["xyz"]).all()
                        or point["depth_left"] <= 0
                        or point["depth_right"] <= 0
                    ):
                        raise ValueError("invalid accepted strict point")
                    points.append(
                        {"index": point["index"], "xyz": point["xyz"], "valid": True}
                    )
        payload = {
            "pair_id": i,
            "pair_timestamp_sec": ts,
            "persons_3d": [{"keypoints_3d": points}],
        }
        stage_name = str(stage_row["stage"].get("operational", "unknown"))
        rotation = rotation_tracker.update(
            lf,
            rf,
            left_result,
            right_result,
            estimate=a.continuous_stage
            or a.stage_handoff
            or stage_name in {STAGE_TWO, "transition"},
        )
        ground_rows.append(
            ground_writer.consume(
                payload,
                stage_row,
                rotation,
                {"left": left_points[i], "right": right_points[i]},
            )
            if a.stage_handoff
            else ground_writer.consume(payload, stage_row, rotation)
        )
    # Diagnostic OpenCV RANSAC must not consume the stage tracker's RNG stream.
    if visual_tracker:
        for i, (lf, rf, (left_result, right_result)) in enumerate(
            zip(left_frames, right_frames, inference_pairs)
        ):
            visual_rows.append(
                dict(
                    pair_id=i,
                    **visual_tracker.update(lf, rf, left_result, right_result),
                )
            )
    stage_summary = stage_writer.close(completed=True)
    ground_summary = ground_writer.close(completed=True)
    # The fixed-lag writer returns provisional mutable rows during consume().
    # Only the closed, published stream is authoritative for every export mode.
    ground_rows = [
        json.loads(line)
        for line in (out / "dynamic_ground_pose.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    if [row["pair_id"] for row in ground_rows] != ids:
        raise ValueError("published ground frame order differs from input")
    if handoff_policy:
        stage_summary = {
            "legacy_summary": stage_summary,
            "candidate_stage_counts": {
                s: sum(r["stage"]["operational"] == s for r in stage_rows)
                for s in {r["stage"]["operational"] for r in stage_rows}
            },
        }
        (out / "stage1_stage2.jsonl").rename(out / "legacy_stage1_stage2.jsonl")
        (out / "stage1_stage2.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in stage_rows),
            encoding="utf-8",
        )
    if visual_tracker:
        (out / "background_vo_diagnostic.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in visual_rows), encoding="utf-8"
        )
    static_walker = json.loads(a.walker_model.read_text(encoding="utf-8-sig"))
    rotations = np.asarray(
        [r["rotation_world_from_left_camera"] for r in ground_rows], dtype=np.float64
    )
    translations = np.asarray(
        [r["translation_world_from_left_camera_mm"] for r in ground_rows],
        dtype=np.float64,
    )
    stages = np.asarray(
        [r["stage"].get("operational", "unknown") for r in stage_rows], dtype=object
    )
    np.savez_compressed(
        out / "scene_transforms.npz",
        rotation_ground_from_left=rotations,
        translation_ground_from_left_mm=translations,
        stage=stages,
        accepted=accepted,
        raw_triangulated_points_mm=raw,
        detector_only_triangulated_points_mm=detector_raw,
        detector_only_accepted=detector_accepted,
        pair_timestamp_sec=np.asarray(timing).mean(axis=1),
        left_timestamp_sec=np.asarray(timing)[:, 0],
        right_timestamp_sec=np.asarray(timing)[:, 1],
    )
    (out / "scene_sources.json").write_text(
        json.dumps(
            {
                "raw_left_video": str(a.left_video.resolve())
                if a.input_pair_dir is None
                else None,
                "raw_right_video": str(a.right_video.resolve())
                if a.input_pair_dir is None
                else None,
                "raw_video_consumed": a.input_pair_dir is None,
                "input_pair_dir": str(a.input_pair_dir.resolve())
                if a.input_pair_dir
                else None,
                "raw_left_pmpose": str(a.left_pmpose.resolve()),
                "raw_right_pmpose": str(a.right_pmpose.resolve()),
                "calibration": str(a.calibration.resolve()),
                "static_ground_reference": str(a.ground_reference.resolve()),
                "static_walker_model": str(a.walker_model.resolve()),
                "strict_input": str(a.strict_jsonl.resolve()),
                "strict_scope": "same first instances as fitter; original association and per-point gates",
                "export_mask_source": "strict_input_not_detector_only_retriangulation",
                "ground_export_source": "closed_published_dynamic_ground_pose_jsonl",
                "scene_profile": a.scene_profile,
                "resolved_scene_profile": a.resolved_scene_profile,
                "old_stage_jsonl_read": False,
                "old_dynamic_ground_read": False,
                "old_dynamic_walker_pose_read": False,
                "static_installation_pose_assumption_consumed": True,
                "stage_algorithm": "StageHandoffPolicy / RealtimeStageWalkerWriter evidence"
                if a.stage_handoff
                else "RealtimeStageWalkerWriter / TwoStageRecognizer",
                "ground_algorithm": "ContinuousStageGroundWriter candidate"
                if a.continuous_stage
                else (
                    "StageHandoffGroundWriter candidate"
                    if a.stage_handoff
                    else "RealtimeDynamicGroundWriter full_se3 with current-run Stage 1/2"
                ),
                "stage_handoff_candidate": a.stage_handoff,
                "single_ankle_fallback": a.single_ankle_fallback,
                "latest_stationary_anchor": a.latest_anchor,
                "ankle_bearing_fallback": a.ankle_bearing_fallback,
                "stage1_reanchor": not a.disable_stage1_reanchor
                and a.landing_recovery == "none",
                "landing_recovery": a.landing_recovery,
                "landing_protocol": a.landing_protocol,
                "dropout_prediction": a.predict_camera_dropout,
                "frames_metadata": str(a.frames_metadata.resolve())
                if a.frames_metadata
                else None,
                "timestamp_kind": "host_return_not_exposure"
                if a.frames_metadata
                else "nominal_fps",
                "foot_observation_mode": "current"
                if a.continuous_stage
                else a.foot_observation_mode,
                "fps_left": fps_l,
                "fps_right": fps_r,
                "video_frame_counts_before_truncation": video_truncated,
                "frames": len(ids),
                "opencv_threads": cv2.getNumThreads(),
                "image_source": image_source,
                "stage_summary": stage_summary,
                "ground_summary": ground_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "static_walker_model.json").write_text(
        json.dumps(static_walker, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"frames": len(ids), "stage": stage_summary, "ground": ground_summary},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
