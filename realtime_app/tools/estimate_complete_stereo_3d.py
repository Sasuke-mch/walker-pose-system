#!/usr/bin/env python3
"""Create a separate, provenance-labelled complete 3-D estimate sequence.

This tool deliberately does not alter the production triangulation pipeline.
It replays saved PMPose or ProbPose 2-D predictions for one ordered sequence.
Any finite, raw-image-in-bounds left/right joint pair is triangulated without a
2-D-score, association-cost, or reprojection-error rejection.  Reprojection
error is retained exclusively as a statistic.  A missing one-view observation
is estimated from its camera ray and same-joint direct-stereo temporal anchor;
a missing two-view observation uses only that temporal anchor.  Bone lengths
are emitted after estimation and never enter the estimator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_app.calibration import StereoCalibration
from pose_app.geometry_input import raw_point_rejection_reason
from pose_app.rotation import ROTATION_CHOICES
from pose_app.triangulation import COCO17_NAMES
from tools.evaluate_offline_stereo_predictions import _records, _result


BONES = {
    "left_thigh": ("left_hip", "left_knee"),
    "right_thigh": ("right_hip", "right_knee"),
    "left_shank": ("left_knee", "left_ankle"),
    "right_shank": ("right_knee", "right_ankle"),
}


@dataclass(frozen=True)
class DirectStereoCandidate:
    """Raw two-view candidate; reprojection error does not decide retention."""

    raw_xyz: np.ndarray | None
    usable_xyz: np.ndarray | None
    state: str
    depth_left: float | None
    depth_right: float | None
    reprojection_left_px: float | None
    reprojection_right_px: float | None
    reprojection_mean_px: float | None


@dataclass(frozen=True)
class JointFrameInput:
    pair_id: int
    file_name: str
    left_point: list[float] | None
    right_point: list[float] | None
    left_reason: str | None
    right_reason: str | None
    direct: DirectStereoCandidate | None


@dataclass(frozen=True)
class TemporalAnchor:
    xyz: np.ndarray
    source: str
    anchor_pair_ids: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("pmpose", "probpose"), required=True)
    parser.add_argument("--left-json", required=True, type=Path)
    parser.add_argument("--right-json", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--left-model-rotation", choices=ROTATION_CHOICES, default="ccw90")
    parser.add_argument("--right-model-rotation", choices=ROTATION_CHOICES, default="cw90")
    parser.add_argument(
        "--topdown-score-source",
        choices=("raw_keypoint", "presence"),
        default="raw_keypoint",
    )
    return parser.parse_args()


def _float_list(value: np.ndarray | None) -> list[float] | None:
    if value is None:
        return None
    return [float(item) for item in value]


def _statistics(values: list[float]) -> dict[str, int | float | None]:
    if not values:
        return {"n": 0, "median": None, "mad": None, "p10": None, "p90": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    return {
        "n": int(array.size),
        "median": median,
        "mad": float(np.median(np.abs(array - median))),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _triangulate_unfiltered(
    calibration: StereoCalibration,
    left_point: list[float],
    right_point: list[float],
) -> DirectStereoCandidate:
    """Triangulate valid-domain points; keep reprojection error as evidence only."""

    left_pixels = np.asarray([[left_point[0], left_point[1]]], dtype=np.float64)
    right_pixels = np.asarray([[right_point[0], right_point[1]]], dtype=np.float64)
    left_normalized = calibration.undistort_normalized(left_pixels, "left")
    right_normalized = calibration.undistort_normalized(right_pixels, "right")
    projection_left = np.concatenate(
        [np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)], axis=1
    )
    projection_right = np.concatenate(
        [calibration.R, calibration.T.reshape(3, 1)], axis=1
    )
    homogeneous = cv2.triangulatePoints(
        projection_left,
        projection_right,
        left_normalized.T,
        right_normalized.T,
    )
    denominator = float(homogeneous[3, 0])
    if not math.isfinite(denominator) or abs(denominator) <= 1e-12:
        return DirectStereoCandidate(
            raw_xyz=None,
            usable_xyz=None,
            state="non_finite_triangulation",
            depth_left=None,
            depth_right=None,
            reprojection_left_px=None,
            reprojection_right_px=None,
            reprojection_mean_px=None,
        )

    xyz = (homogeneous[:3, 0] / denominator).astype(np.float64)
    if not np.all(np.isfinite(xyz)):
        return DirectStereoCandidate(
            raw_xyz=None,
            usable_xyz=None,
            state="non_finite_triangulation",
            depth_left=None,
            depth_right=None,
            reprojection_left_px=None,
            reprojection_right_px=None,
            reprojection_mean_px=None,
        )

    right_xyz = calibration.R @ xyz + calibration.T
    depth_left = float(xyz[2])
    depth_right = float(right_xyz[2])
    projected_left = calibration.project_left(xyz.reshape(1, 3))[0]
    projected_right = calibration.project_right(xyz.reshape(1, 3))[0]
    left_error = float(np.linalg.norm(projected_left - left_pixels[0]))
    right_error = float(np.linalg.norm(projected_right - right_pixels[0]))
    mean_error = 0.5 * (left_error + right_error)
    finite_errors = all(math.isfinite(value) for value in (left_error, right_error, mean_error))
    if not finite_errors:
        state = "non_finite_reprojection"
        usable = None
    elif depth_left <= 0.0 or depth_right <= 0.0:
        state = "negative_or_zero_depth"
        usable = None
    else:
        state = "stereo_raw"
        usable = xyz.copy()
    return DirectStereoCandidate(
        raw_xyz=xyz,
        usable_xyz=usable,
        state=state,
        depth_left=depth_left,
        depth_right=depth_right,
        reprojection_left_px=left_error if finite_errors else None,
        reprojection_right_px=right_error if finite_errors else None,
        reprojection_mean_px=mean_error if finite_errors else None,
    )


def _temporal_anchor(
    samples: list[JointFrameInput], current_index: int
) -> TemporalAnchor | None:
    """Use direct physical stereo candidates only; never use a bone prior."""

    previous = [
        index
        for index in range(current_index - 1, -1, -1)
        if samples[index].direct is not None and samples[index].direct.usable_xyz is not None
    ]
    following = [
        index
        for index in range(current_index + 1, len(samples))
        if samples[index].direct is not None and samples[index].direct.usable_xyz is not None
    ]
    before = previous[0] if previous else None
    after = following[0] if following else None
    if before is not None and after is not None:
        before_xyz = samples[before].direct.usable_xyz
        after_xyz = samples[after].direct.usable_xyz
        fraction = (current_index - before) / (after - before)
        return TemporalAnchor(
            xyz=(1.0 - fraction) * before_xyz + fraction * after_xyz,
            source="interpolated",
            anchor_pair_ids=(samples[before].pair_id, samples[after].pair_id),
        )
    if before is not None:
        before_xyz = samples[before].direct.usable_xyz
        if len(previous) >= 2:
            older = previous[1]
            older_xyz = samples[older].direct.usable_xyz
            velocity = (before_xyz - older_xyz) / (before - older)
            xyz = before_xyz + (current_index - before) * velocity
            source = "forward_extrapolated"
            anchor_ids = (samples[older].pair_id, samples[before].pair_id)
        else:
            xyz = before_xyz.copy()
            source = "forward_held"
            anchor_ids = (samples[before].pair_id,)
        return TemporalAnchor(xyz=xyz, source=source, anchor_pair_ids=anchor_ids)
    if after is not None:
        after_xyz = samples[after].direct.usable_xyz
        if len(following) >= 2:
            later = following[1]
            later_xyz = samples[later].direct.usable_xyz
            velocity = (later_xyz - after_xyz) / (later - after)
            xyz = after_xyz - (after - current_index) * velocity
            source = "backward_extrapolated"
            anchor_ids = (samples[after].pair_id, samples[later].pair_id)
        else:
            xyz = after_xyz.copy()
            source = "backward_held"
            anchor_ids = (samples[after].pair_id,)
        return TemporalAnchor(xyz=xyz, source=source, anchor_pair_ids=anchor_ids)
    return None


def _project_anchor_to_ray(
    calibration: StereoCalibration,
    anchor: np.ndarray,
    point: list[float],
    side: str,
) -> tuple[np.ndarray | None, str]:
    """Find the point on an observed camera ray nearest a temporal anchor."""

    normalized = calibration.undistort_normalized(
        np.asarray([[point[0], point[1]]], dtype=np.float64), side
    )[0]
    direction_local = np.asarray([normalized[0], normalized[1], 1.0], dtype=np.float64)
    if side == "left":
        origin = np.zeros(3, dtype=np.float64)
        direction = direction_local
    elif side == "right":
        origin = -calibration.R.T @ calibration.T
        direction = calibration.R.T @ direction_local
    else:
        raise ValueError(f"Unknown camera side: {side}")
    scale = float(np.dot(anchor - origin, direction) / np.dot(direction, direction))
    if not math.isfinite(scale) or scale <= 0.0:
        return None, "ray_behind_camera"
    estimate = origin + scale * direction
    if not np.all(np.isfinite(estimate)):
        return None, "non_finite_ray_projection"
    return estimate, "ray_projected"


def estimate_joint_series(
    samples: list[JointFrameInput], calibration: StereoCalibration
) -> list[dict[str, Any]]:
    """Estimate one joint over time without using any reprojection threshold."""

    output: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        direct = sample.direct
        left_available = sample.left_reason is None and sample.left_point is not None
        right_available = sample.right_reason is None and sample.right_point is not None
        anchor = None if direct and direct.usable_xyz is not None else _temporal_anchor(samples, index)
        xyz: np.ndarray | None = None
        source: str
        ray_side: str | None = None
        ray_state: str | None = None

        if direct is not None and direct.usable_xyz is not None:
            xyz = direct.usable_xyz.copy()
            source = "stereo_raw"
        elif left_available ^ right_available:
            selected_side = "left" if left_available else "right"
            selected_point = sample.left_point if left_available else sample.right_point
            if anchor is None:
                source = "unavailable_no_stereo_anchor"
            else:
                ray_side = selected_side
                ray_xyz, ray_state = _project_anchor_to_ray(
                    calibration, anchor.xyz, selected_point, selected_side
                )
                if ray_xyz is not None:
                    xyz = ray_xyz
                    source = f"single_view_temporal_{anchor.source}"
                else:
                    xyz = anchor.xyz.copy()
                    source = f"temporal_{anchor.source}_ray_unusable"
        elif anchor is not None:
            xyz = anchor.xyz.copy()
            source = f"temporal_{anchor.source}"
        else:
            source = "unavailable_no_stereo_anchor"

        output.append({
            "pair_id": sample.pair_id,
            "file_name": sample.file_name,
            "estimate_source": source,
            "has_estimate": xyz is not None,
            "xyz": _float_list(xyz),
            "temporal_anchor_source": None if anchor is None else anchor.source,
            "temporal_anchor_pair_ids": [] if anchor is None else list(anchor.anchor_pair_ids),
            "single_view_ray_side": ray_side,
            "single_view_ray_state": ray_state,
        })
    return output


def _joint_input(
    pair_id: int,
    file_name: str,
    left_point: list[float] | None,
    right_point: list[float] | None,
    calibration: StereoCalibration,
    *,
    target_is_unique: bool,
) -> JointFrameInput:
    if not target_is_unique:
        return JointFrameInput(
            pair_id=pair_id,
            file_name=file_name,
            left_point=None,
            right_point=None,
            left_reason="not_unique_target_person",
            right_reason="not_unique_target_person",
            direct=None,
        )
    left_reason = raw_point_rejection_reason(left_point, calibration.left_image_size)
    right_reason = raw_point_rejection_reason(right_point, calibration.right_image_size)
    direct = None
    if left_reason is None and right_reason is None:
        direct = _triangulate_unfiltered(calibration, left_point, right_point)
    return JointFrameInput(
        pair_id=pair_id,
        file_name=file_name,
        left_point=left_point,
        right_point=right_point,
        left_reason=left_reason,
        right_reason=right_reason,
        direct=direct,
    )


def _load_joint_inputs(args: argparse.Namespace, calibration: StereoCalibration) -> dict[str, list[JointFrameInput]]:
    left_records = _records(args.left_json, args.model, args.topdown_score_source)
    right_records = _records(args.right_json, args.model, args.topdown_score_source)
    left_by_name = {record["name"]: record for record in left_records}
    right_by_name = {record["name"]: record for record in right_records}
    if set(left_by_name) != set(right_by_name):
        raise ValueError("Left/right saved prediction file names do not match exactly")
    if len(left_records) != len(left_by_name):
        raise ValueError("Left prediction sequence has duplicate file names")

    joints: dict[str, list[JointFrameInput]] = {name: [] for name in COCO17_NAMES}
    for pair_id, left_record in enumerate(left_records):
        file_name = left_record["name"]
        right_record = right_by_name[file_name]
        left_result = _result(
            left_record,
            args.model,
            calibration.left_image_size,
            args.left_model_rotation,
            pair_id / args.fps,
        )
        right_result = _result(
            right_record,
            args.model,
            calibration.right_image_size,
            args.right_model_rotation,
            pair_id / args.fps,
        )
        target_is_unique = len(left_result.persons) == 1 and len(right_result.persons) == 1
        for joint_index, joint_name in enumerate(COCO17_NAMES):
            left_point = (
                list(left_result.persons[0].keypoints[joint_index])
                if target_is_unique else None
            )
            right_point = (
                list(right_result.persons[0].keypoints[joint_index])
                if target_is_unique else None
            )
            joints[joint_name].append(
                _joint_input(
                    pair_id,
                    file_name,
                    left_point,
                    right_point,
                    calibration,
                    target_is_unique=target_is_unique,
                )
            )
    return joints


def _direct_payload(candidate: DirectStereoCandidate | None) -> dict[str, Any]:
    if candidate is None:
        return {
            "state": "not_triangulated",
            "raw_xyz": None,
            "depth_left": None,
            "depth_right": None,
            "reprojection_error_left_px": None,
            "reprojection_error_right_px": None,
            "reprojection_error_mean_px": None,
        }
    return {
        "state": candidate.state,
        "raw_xyz": _float_list(candidate.raw_xyz),
        "depth_left": candidate.depth_left,
        "depth_right": candidate.depth_right,
        "reprojection_error_left_px": candidate.reprojection_left_px,
        "reprojection_error_right_px": candidate.reprojection_right_px,
        "reprojection_error_mean_px": candidate.reprojection_mean_px,
    }


def _make_frame_rows(
    joint_inputs: dict[str, list[JointFrameInput]],
    estimates: dict[str, list[dict[str, Any]]],
    model: str,
    calibration: StereoCalibration,
) -> list[dict[str, Any]]:
    frame_count = len(next(iter(joint_inputs.values())))
    frames: list[dict[str, Any]] = []
    for index in range(frame_count):
        reference = joint_inputs[COCO17_NAMES[0]][index]
        keypoints = []
        for joint_index, joint_name in enumerate(COCO17_NAMES):
            source = joint_inputs[joint_name][index]
            estimate = estimates[joint_name][index]
            keypoints.append({
                "index": joint_index,
                "name": joint_name,
                "has_estimate": estimate["has_estimate"],
                "xyz": estimate["xyz"],
                "estimate_source": estimate["estimate_source"],
                "temporal_anchor_source": estimate["temporal_anchor_source"],
                "temporal_anchor_pair_ids": estimate["temporal_anchor_pair_ids"],
                "single_view_ray_side": estimate["single_view_ray_side"],
                "single_view_ray_state": estimate["single_view_ray_state"],
                "left_2d": source.left_point,
                "right_2d": source.right_point,
                "left_2d_status": source.left_reason or "in_raw_image_bounds",
                "right_2d_status": source.right_reason or "in_raw_image_bounds",
                "left_score": None if source.left_point is None else float(source.left_point[2]),
                "right_score": None if source.right_point is None else float(source.right_point[2]),
                "direct_stereo": _direct_payload(source.direct),
            })
        frames.append({
            "pair_id": reference.pair_id,
            "file_name": reference.file_name,
            "model": model,
            "coordinate_frame": "left_camera",
            "length_unit": calibration.length_unit,
            "estimation_policy": "no score or reprojection-error rejection; raw-bounds remains a hard geometry domain rule",
            "keypoints_3d_estimated": keypoints,
        })
    return frames


def _source_summary(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    for frame in frames:
        for point in frame["keypoints_3d_estimated"]:
            counts[(point["name"], point["estimate_source"])] += 1
    rows = []
    for (joint, source), count in sorted(counts.items()):
        rows.append({"joint": joint, "estimate_source": source, "frames": count})
    return rows


def _reprojection_summary(
    joint_inputs: dict[str, list[JointFrameInput]],
) -> list[dict[str, Any]]:
    rows = []
    for joint, samples in joint_inputs.items():
        states = Counter(
            "not_triangulated" if sample.direct is None else sample.direct.state
            for sample in samples
        )
        errors = [
            sample.direct.reprojection_mean_px
            for sample in samples
            if sample.direct is not None and sample.direct.reprojection_mean_px is not None
        ]
        statistics = _statistics(errors)
        rows.append({
            "joint": joint,
            "frames": len(samples),
            "raw_domain_two_view_triangulations": sum(
                sample.direct is not None for sample in samples
            ),
            "stereo_raw_positive_depth": states["stereo_raw"],
            "negative_or_zero_depth": states["negative_or_zero_depth"],
            "non_finite_triangulation": states["non_finite_triangulation"],
            "non_finite_reprojection": states["non_finite_reprojection"],
            "not_triangulated": states["not_triangulated"],
            "reprojection_n": statistics["n"],
            "reprojection_median_px": statistics["median"],
            "reprojection_mad_px": statistics["mad"],
            "reprojection_p10_px": statistics["p10"],
            "reprojection_p90_px": statistics["p90"],
            "reprojection_max_px": statistics["max"],
        })
    return rows


def _bone_rows(frames: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    point_by_frame = [
        {point["name"]: point for point in frame["keypoints_3d_estimated"]}
        for frame in frames
    ]
    per_frame: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for bone, (start, end) in BONES.items():
        values: list[float] = []
        deltas: list[float] = []
        direct_endpoint_count = 0
        previous_length: float | None = None
        previous_pair_id: int | None = None
        for frame, points in zip(frames, point_by_frame):
            start_point, end_point = points[start], points[end]
            start_xyz = start_point["xyz"]
            end_xyz = end_point["xyz"]
            available = start_xyz is not None and end_xyz is not None
            length = (
                float(np.linalg.norm(np.asarray(start_xyz) - np.asarray(end_xyz)))
                if available
                else None
            )
            adjacent = (
                length is not None
                and previous_length is not None
                and previous_pair_id is not None
                and frame["pair_id"] == previous_pair_id + 1
            )
            delta = abs(length - previous_length) if adjacent else None
            if length is not None:
                values.append(length)
                if start_point["estimate_source"] == "stereo_raw" and end_point["estimate_source"] == "stereo_raw":
                    direct_endpoint_count += 1
            if delta is not None:
                deltas.append(delta)
            per_frame.append({
                "pair_id": frame["pair_id"],
                "file_name": frame["file_name"],
                "bone": bone,
                "start_joint": start,
                "end_joint": end,
                "start_source": start_point["estimate_source"],
                "end_source": end_point["estimate_source"],
                "length_available": available,
                "estimated_length_mm": length,
                "adjacent_length_delta_available": delta is not None,
                "adjacent_abs_length_delta_mm": delta,
            })
            previous_length = length
            previous_pair_id = frame["pair_id"]
        length_statistics = _statistics(values)
        delta_statistics = _statistics(deltas)
        summary.append({
            "bone": bone,
            "sequence_frames": len(frames),
            "frames_with_estimated_length": len(values),
            "both_endpoints_stereo_raw": direct_endpoint_count,
            "at_least_one_temporal_or_other_estimate": len(values) - direct_endpoint_count,
            "adjacent_length_delta_count": len(deltas),
            "length_median_mm": length_statistics["median"],
            "length_mad_mm": length_statistics["mad"],
            "length_p10_mm": length_statistics["p10"],
            "length_p90_mm": length_statistics["p90"],
            "length_max_mm": length_statistics["max"],
            "abs_delta_median_mm": delta_statistics["median"],
            "abs_delta_mad_mm": delta_statistics["mad"],
            "abs_delta_p90_mm": delta_statistics["p90"],
            "abs_delta_max_mm": delta_statistics["max"],
        })
    return per_frame, summary


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    for field in ("left_json", "right_json", "calibration"):
        path = getattr(args, field).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        setattr(args, field, path)
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    cwd = Path.cwd()
    try:
        os.chdir(ROOT)
        source_calibration = StereoCalibration.load(args.calibration)
    finally:
        os.chdir(cwd)
    calibration = source_calibration.for_runtime_sizes(
        source_calibration.left_image_size, source_calibration.right_image_size
    )
    joint_inputs = _load_joint_inputs(args, calibration)
    estimates = {
        joint: estimate_joint_series(samples, calibration)
        for joint, samples in joint_inputs.items()
    }
    frames = _make_frame_rows(joint_inputs, estimates, args.model, calibration)
    source_rows = _source_summary(frames)
    reprojection_rows = _reprojection_summary(joint_inputs)
    bone_per_frame, bone_summary = _bone_rows(frames)

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "complete_3d_estimates.jsonl").open("w", encoding="utf-8") as handle:
        for frame in frames:
            handle.write(json.dumps(frame, ensure_ascii=False, allow_nan=False) + "\n")
    _write_csv(args.output_dir / "estimate_source_summary.csv", source_rows)
    _write_csv(args.output_dir / "direct_reprojection_statistics.csv", reprojection_rows)
    _write_csv(args.output_dir / "estimated_bone_per_frame.csv", bone_per_frame)
    _write_csv(args.output_dir / "estimated_bone_summary.csv", bone_summary)
    metadata = {
        "model": args.model,
        "frames": len(frames),
        "coordinate_frame": "left_camera",
        "length_unit": calibration.length_unit,
        "inputs": {"left": str(args.left_json), "right": str(args.right_json)},
        "calibration": str(args.calibration),
        "rotations": {"left": args.left_model_rotation, "right": args.right_model_rotation},
        "score_or_reprojection_rejection": "none",
        "raw_image_bounds_policy": "out-of-bounds or non-finite 2-D points do not enter calibration; they are treated as missing observations for temporal estimation",
        "bone_length_usage": "post-estimation statistics only; never an estimation input",
        "identity_policy": "only frames with exactly one left and one right saved person use image observations; non-unique frames use temporal estimation only",
        "interpretation": "The output contains raw stereo candidates and temporal estimates with explicit provenance. It is not a replacement for strict triangulation, an accuracy benchmark, or a clinical gait result.",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    total = len(frames) * len(COCO17_NAMES)
    available = sum(
        point["has_estimate"]
        for frame in frames
        for point in frame["keypoints_3d_estimated"]
    )
    print(json.dumps({"output": str(args.output_dir), "model": args.model, "frames": len(frames), "estimated_points": available, "total_points": total}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
