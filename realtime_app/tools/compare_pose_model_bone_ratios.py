from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from pose_app.calibration import StereoCalibration


COCO17 = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

BONES = {
    "left_upper_arm": ("left_shoulder", "left_elbow"),
    "right_upper_arm": ("right_shoulder", "right_elbow"),
    "left_forearm": ("left_elbow", "left_wrist"),
    "right_forearm": ("right_elbow", "right_wrist"),
    "left_thigh": ("left_hip", "left_knee"),
    "right_thigh": ("right_hip", "right_knee"),
    "left_shin": ("left_knee", "left_ankle"),
    "right_shin": ("right_knee", "right_ankle"),
}

RATIOS = {
    "left_thigh_over_shin": ("left_thigh", "left_shin"),
    "right_thigh_over_shin": ("right_thigh", "right_shin"),
    "left_upper_arm_over_forearm": ("left_upper_arm", "left_forearm"),
    "right_upper_arm_over_forearm": ("right_upper_arm", "right_forearm"),
}

FRAME_RE = re.compile(r"(\d+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare PMPose and Sapiens2 2-D offsets and triangulated bone ratios."
    )
    p.add_argument("--pmpose-strict", type=Path, required=True)
    p.add_argument("--pmpose-force-all", type=Path, required=True)
    p.add_argument("--sapiens-strict", type=Path, required=True)
    p.add_argument("--sapiens-force-all", type=Path, required=True)
    p.add_argument("--start-frame", type=int, default=60)
    p.add_argument("--end-frame", type=int, default=90)
    p.add_argument("--reprojection-threshold-px", type=float, default=10.0)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--output-2d-csv", type=Path, required=True)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def frame_number(row: dict[str, Any]) -> int:
    name = str(row.get("file_name", ""))
    matches = FRAME_RE.findall(name)
    if matches:
        return int(matches[-1])
    return int(row.get("pair_id", -1))


def select_window(rows: Iterable[dict[str, Any]], start: int, end: int) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for row in rows:
        frame = frame_number(row)
        if start <= frame <= end:
            if frame in selected:
                raise ValueError(f"duplicate row for frame {frame}")
            selected[frame] = row
    return selected


def first_person_points(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    persons = row.get("persons_3d") or []
    if not persons:
        return {}
    return {
        str(point.get("name")): point
        for point in persons[0].get("keypoints_3d", [])
    }


def finite_xyz(point: dict[str, Any] | None) -> np.ndarray | None:
    if point is None:
        return None
    xyz = np.asarray(point.get("xyz", []), dtype=np.float64)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        return None
    return xyz


def point_passes_strict(point: dict[str, Any] | None) -> bool:
    return bool(point and point.get("valid") and finite_xyz(point) is not None)


def point_passes_reprojection(point: dict[str, Any] | None, threshold: float) -> bool:
    if finite_xyz(point) is None:
        return False
    try:
        error = float(point.get("reprojection_error_mean_px"))
        depth_left = float(point.get("depth_left"))
        depth_right = float(point.get("depth_right"))
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(error)
        and error <= threshold
        and depth_left > 0.0
        and depth_right > 0.0
    )


def summary(values: list[float], frames: list[int] | None = None) -> dict[str, Any]:
    if not values:
        return {
            "count": 0, "median": None, "p05": None, "p95": None,
            "mad": None, "adjacent_delta_median": None,
            "adjacent_delta_p95": None, "adjacent_delta_count": 0,
        }
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    adjacent: list[float] = []
    if frames is not None:
        ordered = sorted(zip(frames, values))
        for (frame_a, value_a), (frame_b, value_b) in zip(ordered, ordered[1:]):
            if frame_b == frame_a + 1:
                adjacent.append(abs(float(value_b) - float(value_a)))
    return {
        "count": int(array.size),
        "median": median,
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "mad": float(np.median(np.abs(array - median))),
        "adjacent_delta_median": None if not adjacent else float(np.median(adjacent)),
        "adjacent_delta_p95": None if not adjacent else float(np.percentile(adjacent, 95)),
        "adjacent_delta_count": len(adjacent),
    }


def extract_bones(
    rows: dict[int, dict[str, Any]],
    mode: str,
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    per_frame: list[dict[str, Any]] = []
    bone_values: dict[str, list[float]] = defaultdict(list)
    bone_frames: dict[str, list[int]] = defaultdict(list)
    ratio_values: dict[str, list[float]] = defaultdict(list)
    ratio_frames: dict[str, list[int]] = defaultdict(list)

    for frame in sorted(rows):
        points = first_person_points(rows[frame])
        lengths: dict[str, float | None] = {}
        qualities: dict[str, str] = {}
        for bone, (joint_a, joint_b) in BONES.items():
            point_a, point_b = points.get(joint_a), points.get(joint_b)
            xyz_a, xyz_b = finite_xyz(point_a), finite_xyz(point_b)
            if xyz_a is None or xyz_b is None:
                value = None
                quality = "missing"
            else:
                value = float(np.linalg.norm(xyz_a - xyz_b))
                both_pass = point_passes_reprojection(point_a, threshold) and point_passes_reprojection(point_b, threshold)
                if mode == "strict":
                    accepted = point_passes_strict(point_a) and point_passes_strict(point_b)
                    quality = "strict_accepted" if accepted else "strict_rejected"
                    if not accepted:
                        value = None
                else:
                    quality = "both_endpoints_le_10px" if both_pass else "any_endpoint_high_reprojection"
                if value is not None:
                    bone_values[bone].append(value)
                    bone_frames[bone].append(frame)
            lengths[bone] = value
            qualities[bone] = quality

        ratios: dict[str, float | None] = {}
        for ratio, (numerator, denominator) in RATIOS.items():
            a, b = lengths.get(numerator), lengths.get(denominator)
            value = None if a is None or b is None or b <= 0.0 else float(a / b)
            ratios[ratio] = value
            if value is not None:
                ratio_values[ratio].append(value)
                ratio_frames[ratio].append(frame)

        per_frame.append({
            "frame": frame,
            "lengths_mm": lengths,
            "bone_quality": qualities,
            "ratios": ratios,
        })

    bone_summary = {
        bone: summary(bone_values[bone], bone_frames[bone]) for bone in BONES
    }
    ratio_summary = {
        ratio: summary(ratio_values[ratio], ratio_frames[ratio]) for ratio in RATIOS
    }
    return per_frame, bone_summary, ratio_summary


def extract_force_layers(
    rows: dict[int, dict[str, Any]], threshold: float
) -> dict[str, dict[str, dict[str, Any]]]:
    layers: dict[str, dict[str, list[float]]] = {
        "both_endpoints_le_10px": defaultdict(list),
        "any_endpoint_high_reprojection": defaultdict(list),
    }
    layer_frames: dict[str, dict[str, list[int]]] = {
        key: defaultdict(list) for key in layers
    }
    for frame in sorted(rows):
        points = first_person_points(rows[frame])
        for bone, (joint_a, joint_b) in BONES.items():
            pa, pb = points.get(joint_a), points.get(joint_b)
            xa, xb = finite_xyz(pa), finite_xyz(pb)
            if xa is None or xb is None:
                continue
            layer = (
                "both_endpoints_le_10px"
                if point_passes_reprojection(pa, threshold) and point_passes_reprojection(pb, threshold)
                else "any_endpoint_high_reprojection"
            )
            layers[layer][bone].append(float(np.linalg.norm(xa - xb)))
            layer_frames[layer][bone].append(frame)
    return {
        layer: {
            bone: summary(layers[layer][bone], layer_frames[layer][bone])
            for bone in BONES
        }
        for layer in layers
    }


def two_d_comparison(
    pmpose: dict[int, dict[str, Any]], sapiens: dict[int, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    common_frames = sorted(set(pmpose) & set(sapiens))
    for frame in common_frames:
        for view in ("left", "right"):
            pp = (((pmpose[frame].get(view) or {}).get("persons") or [{}])[0].get("keypoints") or [])
            sp = (((sapiens[frame].get(view) or {}).get("persons") or [{}])[0].get("keypoints") or [])
            for index, name in enumerate(COCO17):
                if index >= len(pp) or index >= len(sp):
                    continue
                p = np.asarray(pp[index][:2], dtype=np.float64)
                s = np.asarray(sp[index][:2], dtype=np.float64)
                if not np.isfinite(p).all() or not np.isfinite(s).all():
                    continue
                dx, dy = (s - p).tolist()
                distance = float(np.linalg.norm(s - p))
                rows.append({
                    "frame": frame, "view": view, "joint": name,
                    "pmpose_x": float(p[0]), "pmpose_y": float(p[1]),
                    "sapiens_x": float(s[0]), "sapiens_y": float(s[1]),
                    "sapiens_minus_pmpose_dx": float(dx),
                    "sapiens_minus_pmpose_dy": float(dy),
                    "distance_px": distance,
                })
                grouped[(view, name)].append((float(dx), float(dy), distance))
    stats: dict[str, Any] = {}
    for (view, name), values in grouped.items():
        arr = np.asarray(values, dtype=np.float64)
        stats[f"{view}:{name}"] = {
            "count": int(arr.shape[0]),
            "median_dx_px": float(np.median(arr[:, 0])),
            "median_dy_px": float(np.median(arr[:, 1])),
            "median_distance_px": float(np.median(arr[:, 2])),
            "p95_distance_px": float(np.percentile(arr[:, 2], 95)),
        }
    return rows, stats


def pair_geometry(
    left_xy: np.ndarray, right_xy: np.ndarray, calibration: StereoCalibration
) -> tuple[np.ndarray, float] | None:
    import cv2

    if left_xy.shape != (2,) or right_xy.shape != (2,):
        return None
    if not np.isfinite(left_xy).all() or not np.isfinite(right_xy).all():
        return None
    left_normalized = calibration.undistort_normalized(left_xy[None, :], "left")
    right_normalized = calibration.undistort_normalized(right_xy[None, :], "right")
    p_left = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)
    p_right = np.concatenate([calibration.R, calibration.T.reshape(3, 1)], axis=1)
    homogeneous = cv2.triangulatePoints(
        p_left, p_right, left_normalized.T, right_normalized.T
    )
    if abs(float(homogeneous[3, 0])) <= 1e-12:
        return None
    xyz = (homogeneous[:3, 0] / homogeneous[3, 0])[None, :]
    right_xyz = (calibration.R @ xyz.T + calibration.T.reshape(3, 1)).T
    if not np.isfinite(xyz).all() or xyz[0, 2] <= 0.0 or right_xyz[0, 2] <= 0.0:
        return None
    projected_left = calibration.project_left(xyz)[0]
    projected_right = calibration.project_right(xyz)[0]
    error = float(
        0.5
        * (
            np.linalg.norm(projected_left - left_xy)
            + np.linalg.norm(projected_right - right_xy)
        )
    )
    return xyz[0], error


def pair_reprojection_error(
    left_xy: np.ndarray, right_xy: np.ndarray, calibration: StereoCalibration
) -> float | None:
    geometry = pair_geometry(left_xy, right_xy, calibration)
    return None if geometry is None else geometry[1]


def left_right_assignment_diagnostic(
    rows: dict[int, dict[str, Any]], calibration: StereoCalibration
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for body_part, indices in {
        "hip": (11, 12), "knee": (13, 14), "ankle": (15, 16)
    }.items():
        same_values: list[float] = []
        crossed_values: list[float] = []
        crossed_wins = 0
        comparable = 0
        per_frame = []
        for frame in sorted(rows):
            left = (((rows[frame].get("left") or {}).get("persons") or [{}])[0].get("keypoints") or [])
            right = (((rows[frame].get("right") or {}).get("persons") or [{}])[0].get("keypoints") or [])
            if max(indices) >= len(left) or max(indices) >= len(right):
                continue
            li, ri = indices
            same_errors = [
                pair_reprojection_error(
                    np.asarray(left[index][:2], dtype=np.float64),
                    np.asarray(right[index][:2], dtype=np.float64), calibration,
                )
                for index in indices
            ]
            crossed_errors = [
                pair_reprojection_error(
                    np.asarray(left[li][:2], dtype=np.float64),
                    np.asarray(right[ri][:2], dtype=np.float64), calibration,
                ),
                pair_reprojection_error(
                    np.asarray(left[ri][:2], dtype=np.float64),
                    np.asarray(right[li][:2], dtype=np.float64), calibration,
                ),
            ]
            same = None if any(value is None for value in same_errors) else float(np.mean(same_errors))
            crossed = None if any(value is None for value in crossed_errors) else float(np.mean(crossed_errors))
            if same is not None:
                same_values.append(same)
            if crossed is not None:
                crossed_values.append(crossed)
            if same is not None and crossed is not None:
                comparable += 1
                crossed_wins += int(crossed < same)
            per_frame.append({"frame": frame, "same_mean_px": same, "crossed_mean_px": crossed})
        output[body_part] = {
            "same_label": summary(same_values),
            "crossed_label": summary(crossed_values),
            "comparable_frames": comparable,
            "crossed_lower_error_frames": crossed_wins,
            "per_frame": per_frame,
            "interpretation": (
                "A lower crossed-label error indicates possible per-view anatomical left/right confusion. "
                "It does not prove that the crossed assignment is anatomically correct."
            ),
        }
    return output


def crossed_knee_bone_diagnostic(
    rows: dict[int, dict[str, Any]], calibration: StereoCalibration, threshold: float
) -> dict[str, Any]:
    definitions = {
        "left": {"hip": (11, 11), "knee": (13, 14), "ankle": (15, 15)},
        "right": {"hip": (12, 12), "knee": (14, 13), "ankle": (16, 16)},
    }
    output: dict[str, Any] = {}
    for side, mapping in definitions.items():
        buckets: dict[str, list[float]] = defaultdict(list)
        bucket_frames: dict[str, list[int]] = defaultdict(list)
        strict_buckets: dict[str, list[float]] = defaultdict(list)
        strict_frames: dict[str, list[int]] = defaultdict(list)
        per_frame: list[dict[str, Any]] = []
        for frame in sorted(rows):
            left = (((rows[frame].get("left") or {}).get("persons") or [{}])[0].get("keypoints") or [])
            right = (((rows[frame].get("right") or {}).get("persons") or [{}])[0].get("keypoints") or [])
            geometries: dict[str, tuple[np.ndarray, float] | None] = {}
            for joint, (left_index, right_index) in mapping.items():
                if left_index >= len(left) or right_index >= len(right):
                    geometries[joint] = None
                    continue
                geometries[joint] = pair_geometry(
                    np.asarray(left[left_index][:2], dtype=np.float64),
                    np.asarray(right[right_index][:2], dtype=np.float64),
                    calibration,
                )
            if any(value is None for value in geometries.values()):
                per_frame.append({"frame": frame, "available": False})
                continue
            hip_xyz, hip_error = geometries["hip"]  # type: ignore[misc]
            knee_xyz, knee_error = geometries["knee"]  # type: ignore[misc]
            ankle_xyz, ankle_error = geometries["ankle"]  # type: ignore[misc]
            thigh = float(np.linalg.norm(hip_xyz - knee_xyz))
            shin = float(np.linalg.norm(knee_xyz - ankle_xyz))
            ratio = thigh / shin if shin > 0.0 else math.nan
            strict = max(hip_error, knee_error, ankle_error) <= threshold
            values = {"thigh_mm": thigh, "shin_mm": shin, "thigh_over_shin": ratio}
            for key, value in values.items():
                if math.isfinite(value):
                    buckets[key].append(value)
                    bucket_frames[key].append(frame)
                    if strict:
                        strict_buckets[key].append(value)
                        strict_frames[key].append(frame)
            per_frame.append({
                "frame": frame, "available": True, "strict": strict,
                "hip_reprojection_px": hip_error,
                "crossed_knee_reprojection_px": knee_error,
                "ankle_reprojection_px": ankle_error,
                **values,
            })
        output[side] = {
            "force_all": {
                key: summary(buckets[key], bucket_frames[key])
                for key in ("thigh_mm", "shin_mm", "thigh_over_shin")
            },
            "all_three_points_le_10px": {
                key: summary(strict_buckets[key], strict_frames[key])
                for key in ("thigh_mm", "shin_mm", "thigh_over_shin")
            },
            "per_frame": per_frame,
        }
    output["interpretation"] = (
        "Diagnostic only: the right-view knee labels are exchanged while hip and ankle labels remain unchanged. "
        "This tests per-view knee identity inconsistency and is not an authorized relabeling rule."
    )
    return output


def radial_reprojection_diagnostic(
    rows: dict[int, dict[str, Any]], calibration: StereoCalibration
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    all_values: list[tuple[float, float]] = []
    centers = {
        "left": np.asarray(
            [calibration.left_K[0, 2], calibration.left_K[1, 2]], dtype=np.float64
        ),
        "right": np.asarray(
            [calibration.right_K[0, 2], calibration.right_K[1, 2]], dtype=np.float64
        ),
    }
    for frame in sorted(rows):
        record = rows[frame]
        points = first_person_points(record)
        view_points = {
            view: (((record.get(view) or {}).get("persons") or [{}])[0].get("keypoints") or [])
            for view in ("left", "right")
        }
        for index, name in enumerate(COCO17):
            point = points.get(name)
            try:
                error = float(point.get("reprojection_error_mean_px"))  # type: ignore[union-attr]
            except (AttributeError, TypeError, ValueError):
                continue
            radii = []
            for view in ("left", "right"):
                if index >= len(view_points[view]):
                    continue
                xy = np.asarray(view_points[view][index][:2], dtype=np.float64)
                if np.isfinite(xy).all():
                    radii.append(float(np.linalg.norm(xy - centers[view])))
            if not radii or not math.isfinite(error):
                continue
            item = (float(np.mean(radii)), error)
            grouped[name].append(item)
            all_values.append(item)

    def summarize_pairs(values: list[tuple[float, float]]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "median_radius_px": None, "median_error_px": None, "p95_error_px": None}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.shape[0]),
            "median_radius_px": float(np.median(array[:, 0])),
            "median_error_px": float(np.median(array[:, 1])),
            "p95_error_px": float(np.percentile(array[:, 1], 95)),
        }

    overall = summarize_pairs(all_values)
    if len(all_values) >= 2:
        array = np.asarray(all_values, dtype=np.float64)
        radius_ranks = np.argsort(np.argsort(array[:, 0]))
        error_ranks = np.argsort(np.argsort(array[:, 1]))
        overall["rank_correlation_radius_vs_error"] = float(
            np.corrcoef(radius_ranks, error_ranks)[0, 1]
        )
    else:
        overall["rank_correlation_radius_vs_error"] = None
    return {
        "overall": overall,
        "by_joint": {name: summarize_pairs(grouped[name]) for name in COCO17},
        "interpretation": (
            "Image radius is measured from each calibrated principal point and averaged over views. "
            "Weak rank correlation does not exclude pose-network fisheye domain shift; it only argues against a simple monotonic radial calibration-error explanation."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    inputs = {
        "pmpose_strict": select_window(read_jsonl(args.pmpose_strict), args.start_frame, args.end_frame),
        "pmpose_force_all": select_window(read_jsonl(args.pmpose_force_all), args.start_frame, args.end_frame),
        "sapiens_strict": select_window(read_jsonl(args.sapiens_strict), args.start_frame, args.end_frame),
        "sapiens_force_all": select_window(read_jsonl(args.sapiens_force_all), args.start_frame, args.end_frame),
    }
    calibration = StereoCalibration.load(args.calibration)
    expected = set(range(args.start_frame, args.end_frame + 1))
    missing = {name: sorted(expected - set(rows)) for name, rows in inputs.items()}

    result: dict[str, Any] = {
        "scope": {
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "expected_frame_count": len(expected),
            "reprojection_threshold_px": args.reprojection_threshold_px,
            "missing_frames": missing,
            "interpretation_boundary": (
                "Sapiens2 is an operational pseudo-reference, not anatomical ground truth. "
                "Force-all high-reprojection bones are diagnostics, not accepted 3-D."
            ),
        },
        "models": {},
    }
    flat_rows: list[dict[str, Any]] = []
    for model in ("pmpose", "sapiens"):
        strict_frames, strict_bones, strict_ratios = extract_bones(
            inputs[f"{model}_strict"], "strict", args.reprojection_threshold_px
        )
        force_frames, force_bones, force_ratios = extract_bones(
            inputs[f"{model}_force_all"], "force_all", args.reprojection_threshold_px
        )
        result["models"][model] = {
            "strict": {"bones_mm": strict_bones, "ratios": strict_ratios},
            "force_all": {
                "bones_mm": force_bones,
                "ratios": force_ratios,
                "quality_layers": extract_force_layers(
                    inputs[f"{model}_force_all"], args.reprojection_threshold_px
                ),
            },
            "left_right_assignment_diagnostic": left_right_assignment_diagnostic(
                inputs[f"{model}_force_all"], calibration
            ),
            "crossed_knee_bone_diagnostic": crossed_knee_bone_diagnostic(
                inputs[f"{model}_force_all"], calibration,
                args.reprojection_threshold_px,
            ),
            "radial_reprojection_diagnostic": radial_reprojection_diagnostic(
                inputs[f"{model}_force_all"], calibration
            ),
        }
        strict_by_frame = {row["frame"]: row for row in strict_frames}
        force_by_frame = {row["frame"]: row for row in force_frames}
        for frame in sorted(expected):
            for mode, source in (("strict", strict_by_frame), ("force_all", force_by_frame)):
                row = source.get(frame, {})
                lengths = row.get("lengths_mm", {})
                qualities = row.get("bone_quality", {})
                ratios = row.get("ratios", {})
                for bone in BONES:
                    flat_rows.append({
                        "frame": frame, "model": model, "mode": mode,
                        "metric": bone, "kind": "bone_length_mm",
                        "value": lengths.get(bone), "quality": qualities.get(bone, "missing"),
                    })
                for ratio in RATIOS:
                    flat_rows.append({
                        "frame": frame, "model": model, "mode": mode,
                        "metric": ratio, "kind": "ratio",
                        "value": ratios.get(ratio), "quality": "derived_from_available_bones",
                    })

    two_d_rows, two_d_stats = two_d_comparison(
        inputs["pmpose_strict"], inputs["sapiens_strict"]
    )
    result["two_d_sapiens_minus_pmpose"] = two_d_stats

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(
        args.output_csv, flat_rows,
        ["frame", "model", "mode", "metric", "kind", "value", "quality"],
    )
    write_csv(
        args.output_2d_csv, two_d_rows,
        [
            "frame", "view", "joint", "pmpose_x", "pmpose_y",
            "sapiens_x", "sapiens_y", "sapiens_minus_pmpose_dx",
            "sapiens_minus_pmpose_dy", "distance_px",
        ],
    )
    print(json.dumps({
        "output_json": str(args.output_json),
        "output_csv": str(args.output_csv),
        "output_2d_csv": str(args.output_2d_csv),
        "missing_frames": missing,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
