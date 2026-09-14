#!/usr/bin/env python3
"""Controlled single-variable experiment on the manual-floor-mask stereo failure.

Frozen reference: ``observe_local_ground_semantic_stereo.py`` on the same 12
manually audited ``floor_eligible`` mask pairs reported ``direct=2/12``.  The
read-only diagnostic
(``diagnose_floor_mask_stereo_correspondence.py``) attributed that result to the
left/right consistency gate: the gate compares the forward disparity with a
reverse field produced by OpenCV SGBM searching the partner *to the left* of the
reference pixel, while in this calibration the right pixel's partner lies to its
*right*.  That field cannot represent the correspondence, its pass rate at the
correct partner position is at or below its own null controls, and the gate
removed 98.1% of the pixels that reached it.

This tool changes exactly ONE thing:

    the search direction of the reverse disparity field read by the left/right
    consistency test

Everything else is frozen and inherited from the baseline run metadata: the same
12 pairs, the same manual masks with ``manually_audited`` identity, the same
calibration, the same runtime size, the same virtual focal length, the same
disparity range, the same matcher settings, the same left/right consistency
limit, the same photometric limit, the same depth limits, the same RANSAC
settings, and the same four decision gates with the same thresholds.  No
threshold is changed, nothing is loosened, and no gate is removed.

To make the single-variable claim checkable, the same process also runs the
frozen direction ("control arm") and aborts if that arm does not reproduce the
frozen baseline output for the frame.  Only the experiment arm writes records.

Boundaries: a repaired consistency test is not ground truth.  This tool does not
produce ground-plane accuracy, foot height, contact, support or gait, and it
cannot compare planes across frames because no accepted relative pose exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


TOOLS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_ROOT.parents[1]
REALTIME_ROOT = PROJECT_ROOT / "realtime_app"
for candidate in (str(REALTIME_ROOT), str(TOOLS_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from pose_app.calibration import StereoCalibration  # noqa: E402
import observe_local_ground_semantic_stereo as strict  # noqa: E402
import diagnose_floor_mask_stereo_correspondence as diagnostic  # noqa: E402


SCHEMA_VERSION = "local_ground_state_v1"
EXPERIMENT_ID = "floor_mask_lr_consistency_direction_control_v1"
UNIQUE_VARIABLE = (
    "search direction of the reverse disparity field read by the left/right consistency test "
    "(partner searched to the right of the right-image pixel, matching this calibration), "
    "with every threshold, mask, frame, matcher setting and gate left frozen"
)

FROZEN_PARAMETER_KEYS = (
    "runtime_width",
    "runtime_height",
    "virtual_focal_px",
    "num_disparities",
    "lr_consistency_px",
    "ransac_distance_mm",
    "ransac_iterations",
    "minimum_candidates",
    "minimum_inliers",
    "minimum_inlier_fraction",
    "minimum_coverage_fraction",
    "maximum_median_residual_mm",
    "maximum_median_reprojection_px",
    "frame_step",
    "max_frames",
)

VERIFIED_FIELDS = ("candidate_points", "ransac_inliers", "inlier_coverage_fraction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--left-dir", type=Path, required=True)
    parser.add_argument("--right-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--semantic-left-dir", type=Path, required=True)
    parser.add_argument("--semantic-right-dir", type=Path, required=True)
    parser.add_argument(
        "--frozen-baseline-dir", type=Path, required=True,
        help="frozen manual-mask baseline directory holding run_metadata.json and local_ground_state.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--no-ascii", action="store_true")
    return parser.parse_args()


def load_frozen(baseline_dir: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    metadata = json.loads((baseline_dir / "run_metadata.json").read_text(encoding="utf-8"))
    parameters = {key: metadata["parameters"][key] for key in FROZEN_PARAMETER_KEYS}
    rows: dict[int, dict[str, Any]] = {}
    with (baseline_dir / "local_ground_state.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[int(row["pair_id"])] = row
    return parameters, rows


def reconstruct(
    left_local: np.ndarray,
    right_local: np.ndarray,
    left_mask_local: np.ndarray,
    right_mask_local: np.ndarray,
    rectification: Any,
    forward: np.ndarray,
    reverse: np.ndarray,
    num_disparities: int,
    consistency_px: float,
) -> dict[str, Any]:
    """The frozen candidate chain with the reverse field supplied by the caller."""
    height, width = left_local.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    translation_x = float(rectification.translation_right[0])
    if translation_x >= 0.0:
        raise RuntimeError("this pipeline assumes the left camera as reference")
    partner_x = np.rint(xx.astype(np.float32) - forward).astype(np.int32)
    clipped = np.clip(partner_x, 0, width - 1)
    at_partner = reverse[yy, clipped]
    depth_scale = abs(float(rectification.virtual_K[0, 0] * translation_x))
    depth = depth_scale / np.where(forward > 0, forward, np.nan)
    left_gray = cv2.cvtColor(left_local, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_local, cv2.COLOR_BGR2GRAY)
    in_bounds = (partner_x >= 0) & (partner_x < width)
    valid = (
        (forward > 1.0)
        & in_bounds
        & (left_mask_local > 0)
        & (right_mask_local[yy, clipped] > 0)
        & (at_partner > 1.0)
        & (np.abs(forward - at_partner) <= consistency_px)
        & (np.abs(left_gray.astype(np.int16) - right_gray[yy, clipped].astype(np.int16)) <= 45)
        & np.isfinite(depth) & (depth > 250.0) & (depth < 8000.0)
    )
    ys, left_x = np.nonzero(valid)
    right_x = partner_x[ys, left_x]
    measured = forward[ys, left_x]
    candidate_depth = depth_scale / measured
    rectified_x = (left_x.astype(np.float64) - rectification.virtual_K[0, 2]) * candidate_depth / rectification.virtual_K[0, 0]
    rectified_y = (ys.astype(np.float64) - rectification.virtual_K[1, 2]) * candidate_depth / rectification.virtual_K[1, 1]
    points = (rectification.rotation_left.T @ np.column_stack((rectified_x, rectified_y, candidate_depth)).T).T
    return {
        "points": points,
        "left_pixels": np.column_stack((left_x, ys)).astype(np.int32),
        "right_pixels": np.column_stack((right_x, ys)).astype(np.int32),
        "measured_disparity": measured,
        "rectified_xyz": np.column_stack((rectified_x, rectified_y, candidate_depth)),
    }


def plane_orientation_checks(
    normal: np.ndarray, rectified_xyz: np.ndarray, inlier_mask: np.ndarray | None,
    baseline_direction: np.ndarray, optical_axis: np.ndarray,
) -> dict[str, Any]:
    """Internal orientation and extent checks of a fitted plane.

    Uses only quantities the pipeline already owns: the plane normal in the left
    camera frame, the rectified local baseline direction, the local optical axis,
    and the inliers' rectified coordinates.  Descriptive self-consistency only:
    not a ground-truth check, and it assumes no walker geometry.
    """
    if inlier_mask is None or not np.any(inlier_mask):
        return {}
    unit_normal = normal / np.linalg.norm(normal)
    rectified = rectified_xyz[inlier_mask]
    return {
        "plane_normal_angle_to_local_optical_axis_deg": float(
            np.degrees(np.arccos(np.clip(abs(float(unit_normal @ optical_axis)), 0.0, 1.0)))
        ),
        "plane_normal_angle_to_rectified_baseline_deg": float(
            np.degrees(np.arccos(np.clip(abs(float(unit_normal @ baseline_direction)), 0.0, 1.0)))
        ),
        "inlier_rectified_depth_mm_p05": float(np.percentile(rectified[:, 2], 5)),
        "inlier_rectified_depth_mm_median": float(np.median(rectified[:, 2])),
        "inlier_rectified_depth_mm_p95": float(np.percentile(rectified[:, 2], 95)),
        "inlier_rectified_depth_span_mm": float(np.ptp(rectified[:, 2])),
        "inlier_rectified_x_span_mm": float(np.ptp(rectified[:, 0])),
        "inlier_rectified_y_span_mm": float(np.ptp(rectified[:, 1])),
        "orientation_check_note": (
            "descriptive internal orientation and extent of the accepted point set in the calibrated left-camera "
            "frame; a plane that contains the rectified baseline direction and is tilted from the local optical axis "
            "is what a horizontal floor with a horizontal baseline would look like, but this is not ground truth"
        ),
    }


def fit_and_score(
    *, candidate: dict[str, Any], rectification: Any, calibration: StereoCalibration,
    parameters: dict[str, Any], decision_args: argparse.Namespace, index: int, shape: tuple[int, int],
) -> dict[str, Any]:
    points = candidate["points"]
    left_pixels = candidate["left_pixels"]
    right_pixels = candidate["right_pixels"]
    fit_points = points if len(points) <= 12000 else points[np.linspace(0, len(points) - 1, 12000, dtype=np.int64)]
    fit = strict.fit_plane_ransac(
        fit_points, float(parameters["ransac_distance_mm"]), int(parameters["ransac_iterations"]), 20260909 + index
    )
    inlier_mask = None if fit is None else np.abs(points @ fit.normal + fit.offset) <= float(parameters["ransac_distance_mm"])
    inliers = points[inlier_mask] if inlier_mask is not None else np.empty((0, 3))
    inlier_left = left_pixels[inlier_mask] if inlier_mask is not None else np.empty((0, 2), dtype=np.int32)
    inlier_right = right_pixels[inlier_mask] if inlier_mask is not None else np.empty((0, 2), dtype=np.int32)
    coverage = strict.local_coverage_fraction(inlier_left, shape) if inlier_mask is not None else 0.0
    left_error, right_error = strict.reprojection_medians(inliers, inlier_left, inlier_right, rectification, calibration)
    fraction = None if len(points) == 0 else float(len(inliers) / len(points))
    residual = None if fit is None else fit.median_distance_mm
    state, reasons = strict.decide_observation(
        semantic_source="external_binary_mask",
        semantic_identity_evidence="manually_audited",
        candidate_count=len(points),
        inlier_count=len(inliers),
        inlier_fraction=fraction,
        coverage_fraction=coverage,
        residual_mm=residual,
        reprojection_left_px=left_error,
        reprojection_right_px=right_error,
        args=decision_args,
    )
    plane = None
    if fit is not None:
        plane = strict.canonical_plane_toward_camera(fit.normal, float(fit.offset))
        if state == "direct" and plane is None:
            state = "unavailable"
            reasons = reasons + ["invalid_signed_plane_candidate"]
    return {
        "fit": fit,
        "inlier_mask": inlier_mask,
        "inliers": inliers,
        "inlier_left": inlier_left,
        "inlier_right": inlier_right,
        "coverage": float(coverage),
        "inlier_fraction": fraction,
        "residual": residual,
        "left_error": left_error,
        "right_error": right_error,
        "state": state,
        "reasons": reasons,
        "plane": plane,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    parameters, baseline_rows = load_frozen(args.frozen_baseline_dir)
    runtime_size = (int(parameters["runtime_width"]), int(parameters["runtime_height"]))
    consistency = float(parameters["lr_consistency_px"])
    num_disparities = int(parameters["num_disparities"])

    pairs = strict.read_pairs(args.left_dir, args.right_dir, int(parameters["frame_step"]), int(parameters["max_frames"]))
    if not pairs:
        raise RuntimeError("no pairs selected")
    if len(pairs) != len(baseline_rows):
        raise RuntimeError(
            f"frozen baseline holds {len(baseline_rows)} rows but {len(pairs)} pairs were selected"
        )
    calibration = StereoCalibration.load(args.calibration).for_runtime_sizes(runtime_size, runtime_size)
    decision_args = argparse.Namespace(**parameters)
    args.output_dir.mkdir(parents=True)

    records: list[dict[str, Any]] = []
    comparison: list[dict[str, Any]] = []
    reproduction_report: list[dict[str, Any]] = []
    for index, (name, left_path, right_path) in enumerate(pairs):
        pair_id = int(Path(name).stem.removeprefix("pair_"))
        baseline_row = baseline_rows[pair_id]
        left_upright = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right_upright = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left_upright is None or right_upright is None:
            raise RuntimeError(f"cannot read pair {name}")
        shape_upright = left_upright.shape[:2]
        left_mask_upright = strict.load_mask(args.semantic_left_dir / name, shape_upright)
        right_mask_upright = strict.load_mask(args.semantic_right_dir / name, shape_upright)
        left_seed = strict.mask_seed_upright(left_mask_upright)
        right_seed = strict.mask_seed_upright(right_mask_upright)
        if left_seed is None or right_seed is None:
            raise RuntimeError(f"empty semantic mask for {name}")
        left_raw, right_raw = strict.inverse_upright(left_upright, right_upright)
        raw_size = (left_raw.shape[1], left_raw.shape[0])
        left_seed_raw = strict.upright_point_to_raw(left_seed, "left", (shape_upright[1], shape_upright[0])) * np.asarray(
            (runtime_size[0] / raw_size[0], runtime_size[1] / raw_size[1])
        )
        right_seed_raw = strict.upright_point_to_raw(right_seed, "right", (shape_upright[1], shape_upright[0])) * np.asarray(
            (runtime_size[0] / raw_size[0], runtime_size[1] / raw_size[1])
        )
        rectification = strict.make_mask_directed_rectification(
            calibration, left_seed_raw, right_seed_raw, runtime_size, float(parameters["virtual_focal_px"])
        )
        left_raw = cv2.resize(left_raw, runtime_size, interpolation=cv2.INTER_AREA)
        right_raw = cv2.resize(right_raw, runtime_size, interpolation=cv2.INTER_AREA)
        left_mask_raw = cv2.resize(strict.rotate_mask_to_raw(left_mask_upright, "left"), runtime_size, interpolation=cv2.INTER_NEAREST)
        right_mask_raw = cv2.resize(strict.rotate_mask_to_raw(right_mask_upright, "right"), runtime_size, interpolation=cv2.INTER_NEAREST)
        left_local = cv2.remap(left_raw, rectification.left_map_x, rectification.left_map_y, cv2.INTER_LINEAR)
        right_local = cv2.remap(right_raw, rectification.right_map_x, rectification.right_map_y, cv2.INTER_LINEAR)
        left_mask_local = cv2.remap(left_mask_raw, rectification.left_map_x, rectification.left_map_y, cv2.INTER_NEAREST)
        right_mask_local = cv2.remap(right_mask_raw, rectification.right_map_x, rectification.right_map_y, cv2.INTER_NEAREST)
        shape = left_local.shape[:2]

        forward = strict.dense_disparity(left_local, right_local, num_disparities)

        # --- control arm: frozen search direction, must reproduce the baseline --
        reverse_frozen = strict.dense_disparity(right_local, left_local, num_disparities)
        control_candidate = reconstruct(
            left_local, right_local, left_mask_local, right_mask_local, rectification,
            forward, reverse_frozen, num_disparities, consistency,
        )
        control_score = fit_and_score(
            candidate=control_candidate, rectification=rectification, calibration=calibration,
            parameters=parameters, decision_args=decision_args, index=index, shape=shape,
        )
        reproduction = {
            "pair_id": pair_id,
            "fields": {},
            "reproduced": True,
        }
        control_values = {
            "candidate_points": int(len(control_candidate["points"])),
            "ransac_inliers": int(len(control_score["inliers"])),
            "inlier_coverage_fraction": float(control_score["coverage"]),
            "median_plane_residual_mm": control_score["residual"],
            "median_fisheye_reprojection_left_px": control_score["left_error"],
            "median_fisheye_reprojection_right_px": control_score["right_error"],
        }
        for key in VERIFIED_FIELDS:
            expected = baseline_row["quality"][key]
            actual = control_values[key]
            same = (expected is None or actual is None) and expected == actual
            if not same:
                same = abs(float(expected) - float(actual)) <= 1e-9 * max(1.0, abs(float(expected)))
            reproduction["fields"][key] = {"frozen": expected, "control_arm": actual, "match": bool(same)}
            if not same:
                reproduction["reproduced"] = False
        reproduction["control_arm_state"] = control_score["state"]
        reproduction["control_arm_reasons"] = control_score["reasons"]
        reproduction["frozen_state"] = baseline_row["observation_state"]
        reproduction["state_matches"] = control_score["state"] == baseline_row["observation_state"]
        reproduction["reasons_match"] = sorted(control_score["reasons"]) == sorted(baseline_row["reason"])
        reproduction_report.append(reproduction)
        if not reproduction["reproduced"] or not reproduction["state_matches"] or not reproduction["reasons_match"]:
            raise RuntimeError(f"control arm failed to reproduce the frozen baseline for {name}: {json.dumps(reproduction)}")

        # --- experiment arm: THE single changed variable ------------------------
        reverse_corrected = diagnostic.dense_disparity_right_direction(right_local, left_local, num_disparities)
        experiment_candidate = reconstruct(
            left_local, right_local, left_mask_local, right_mask_local, rectification,
            forward, reverse_corrected, num_disparities, consistency,
        )
        experiment_score = fit_and_score(
            candidate=experiment_candidate, rectification=rectification, calibration=calibration,
            parameters=parameters, decision_args=decision_args, index=index, shape=shape,
        )

        measured = experiment_candidate["measured_disparity"]
        boundary = {
            "candidate_pixels": int(len(measured)),
            "candidate_fraction_at_range_top": float(np.mean(measured >= num_disparities - 1)) if len(measured) else None,
            "candidate_disparity_median": float(np.median(measured)) if len(measured) else None,
            "candidate_disparity_p95": float(np.percentile(measured, 95)) if len(measured) else None,
        }
        geometry = diagnostic.inlier_geometry(
            experiment_score["inliers"], experiment_score["inlier_left"], left_mask_local, shape
        )
        geometry.update(diagnostic.split_half_plane_stability(experiment_score["inliers"], 3, 20260909 + index))
        orientation_checks: dict[str, Any] = {}
        if experiment_score["plane"] is not None:
            orientation_checks = plane_orientation_checks(
                np.asarray(experiment_score["plane"]["normal_left_camera"], dtype=np.float64),
                experiment_candidate["rectified_xyz"],
                experiment_score["inlier_mask"],
                rectification.rotation_left[0],
                rectification.rotation_left[2],
            )

        direct_plane = experiment_score["plane"] if experiment_score["state"] == "direct" else None
        ground_state: dict[str, Any] = {
            "status": experiment_score["state"],
            "reasons": experiment_score["reasons"],
            "provenance": "semantic_mask_direct_stereo_lr_direction_control" if direct_plane is not None else None,
        }
        if direct_plane is not None:
            ground_state.update({
                "coordinate_frame": "left_camera",
                "length_unit": "millimeter",
                "plane": direct_plane,
            })
        record = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "unique_variable": UNIQUE_VARIABLE,
            "pair_id": pair_id,
            "frame_index": pair_id,
            "frame_id": name,
            "coordinate_frame": "left_camera",
            "left_camera": "cam0",
            "length_unit": "millimeter",
            "observation_state": experiment_score["state"],
            "reason": experiment_score["reasons"],
            "plane": direct_plane,
            "unaccepted_candidate_plane": experiment_score["plane"] if direct_plane is None else None,
            "plane_in_left_camera": None if direct_plane is None else {
                "normal_toward_camera_unit": direct_plane["normal_left_camera"],
                "offset_mm": direct_plane["offset_mm"],
            },
            "ground_state": ground_state,
            "evidence": {
                "semantic_source": "external_binary_mask",
                "semantic_identity_evidence": "manually_audited",
                "left_mask_fraction_upright": float((left_mask_upright > 0).mean()),
                "right_mask_fraction_upright": float((right_mask_upright > 0).mean()),
                "matching": (
                    "mask-directed local fisheye rectification; strict left-right disparity consistency with the "
                    "reverse field searched in the direction that matches this calibration; points transformed to "
                    "left_camera"
                ),
            },
            "quality": {
                "candidate_points": int(len(experiment_candidate["points"])),
                "ransac_inliers": int(len(experiment_score["inliers"])),
                "ransac_inlier_fraction": experiment_score["inlier_fraction"],
                "inlier_coverage_fraction": float(experiment_score["coverage"]),
                "median_plane_residual_mm": experiment_score["residual"],
                "median_fisheye_reprojection_left_px": experiment_score["left_error"],
                "median_fisheye_reprojection_right_px": experiment_score["right_error"],
            },
            "artifact_checks": {
                "reverse_field_valid_fraction_frozen": float((reverse_frozen > 1.0).mean()),
                "reverse_field_valid_fraction_corrected": float((reverse_corrected > 1.0).mean()),
                **boundary,
                "inlier_distinct_rows": geometry.get("inlier_distinct_rows"),
                "inlier_pixel_linearity": geometry.get("inlier_pixel_linearity"),
                "inlier_pca_sigma2_over_sigma1": geometry.get("inlier_pca_sigma2_over_sigma1"),
                "inlier_camera_distance_mm_median": geometry.get("inlier_camera_distance_mm_median"),
                "inlier_hull_area_px2": geometry.get("inlier_hull_area_px2"),
                "split_half_normal_angle_deg_median": geometry.get("split_half_normal_angle_deg_median"),
                **orientation_checks,
                "note": (
                    "internal diagnostics only; a plane can be internally stable and still not be the floor. These "
                    "checks exist so that more 'direct' frames are not mistaken for a better ground observation"
                ),
            },
        }
        records.append(record)
        comparison.append({
            "pair_id": pair_id,
            "baseline_state": baseline_row["observation_state"],
            "baseline_reasons": baseline_row["reason"],
            "baseline_candidates": baseline_row["quality"]["candidate_points"],
            "baseline_inliers": baseline_row["quality"]["ransac_inliers"],
            "baseline_coverage": baseline_row["quality"]["inlier_coverage_fraction"],
            "baseline_residual_mm": baseline_row["quality"]["median_plane_residual_mm"],
            "baseline_reprojection_right_px": baseline_row["quality"]["median_fisheye_reprojection_right_px"],
            "experiment_state": experiment_score["state"],
            "experiment_reasons": experiment_score["reasons"],
            "experiment_candidates": record["quality"]["candidate_points"],
            "experiment_inliers": record["quality"]["ransac_inliers"],
            "experiment_coverage": record["quality"]["inlier_coverage_fraction"],
            "experiment_residual_mm": record["quality"]["median_plane_residual_mm"],
            "experiment_reprojection_right_px": record["quality"]["median_fisheye_reprojection_right_px"],
            "state_changed": experiment_score["state"] != baseline_row["observation_state"],
            "candidate_fraction_at_range_top": boundary["candidate_fraction_at_range_top"],
            "inlier_pixel_linearity": geometry.get("inlier_pixel_linearity"),
        })

        if not args.no_visualization:
            output_path = args.output_dir / "visualizations" / f"{Path(name).stem}_lr_direction_control.png"
            strict.write_visualization(
                output_path,
                left_raw, left_local, right_local,
                experiment_candidate["left_pixels"],
                experiment_score["inlier_mask"],
                rectification,
                (
                    f"{name}: baseline={baseline_row['observation_state']} "
                    f"({baseline_row['quality']['candidate_points']}c/{baseline_row['quality']['ransac_inliers']}i) -> "
                    f"experiment={experiment_score['state']} ({len(experiment_candidate['points'])}c/"
                    f"{len(experiment_score['inliers'])}i); reasons={','.join(experiment_score['reasons']) or 'none'}"
                ),
            )
        if not args.no_ascii:
            ascii_dir = args.output_dir / "ascii"
            ascii_dir.mkdir(parents=True, exist_ok=True)
            (ascii_dir / f"{Path(name).stem}_experiment_coverage.txt").write_text(
                f"{name}: baseline={baseline_row['observation_state']} experiment={experiment_score['state']} "
                f"candidates={len(experiment_candidate['points'])} inliers={len(experiment_score['inliers'])}\n"
                + diagnostic.ascii_raster(
                    experiment_score["inlier_left"], experiment_candidate["left_pixels"],
                    left_mask_local, forward > 1.0,
                )
                + "\n",
                encoding="utf-8",
            )
        print(json.dumps({
            "pair": pair_id,
            "baseline": baseline_row["observation_state"],
            "experiment": experiment_score["state"],
            "baseline_candidates": baseline_row["quality"]["candidate_points"],
            "experiment_candidates": record["quality"]["candidate_points"],
            "experiment_inliers": record["quality"]["ransac_inliers"],
            "control_arm_reproduced": reproduction["reproduced"],
        }, ensure_ascii=False))

    with (args.output_dir / "local_ground_state.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    experiment_direct = sum(record["observation_state"] == "direct" for record in records)
    reason_counts: dict[str, int] = {}
    for record in records:
        for reason in record["reason"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "unique_variable": UNIQUE_VARIABLE,
        "frame_count": len(records),
        "direct_count": experiment_direct,
        "unavailable_count": len(records) - experiment_direct,
        "unavailable_reason_counts": reason_counts,
        "baseline_direct_count": sum(row["observation_state"] == "direct" for row in baseline_rows.values()),
        "baseline_unavailable_reason_counts": {
            reason: sum(reason in row["reason"] for row in baseline_rows.values())
            for reason in sorted({reason for row in baseline_rows.values() for reason in row["reason"]})
        },
        "control_arm_reproduced_frozen_baseline": all(item["reproduced"] and item["state_matches"] and item["reasons_match"] for item in reproduction_report),
        "acceptance_note": (
            "A larger direct count is not an improvement and is not evidence about the floor. The acceptance evidence "
            "is the per-frame table plus the internal artefact checks, and no cross-frame plane comparison is made "
            "because no accepted relative pose exists."
        ),
        "interpretation_boundary": (
            "Internal floor-identity, calibrated stereo-geometry and single-variable control diagnostics only. These "
            "records are not ground truth, real ground-plane accuracy, foot contact, support phase, or gait "
            "measurements."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "control_arm_reproduction.json").write_text(
        json.dumps({"items": reproduction_report}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "comparison_vs_manual_mask_baseline.json").write_text(
        json.dumps({"items": comparison}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "unique_variable": UNIQUE_VARIABLE,
        "unchanged_from_baseline": {
            "pairs": "same 12 pair ids, same frame_step and max_frames read from the frozen baseline metadata",
            "masks": "same manual floor masks with semantic_identity_evidence=manually_audited",
            "calibration": str(args.calibration),
            "matcher": "same CLAHE preprocessing and same SGBM settings except the reverse field's search direction",
            "thresholds": {key: parameters[key] for key in FROZEN_PARAMETER_KEYS},
            "gates": "same four decision gates with the same thresholds; none removed or relaxed",
        },
        "inputs": {
            "left_dir": str(args.left_dir),
            "right_dir": str(args.right_dir),
            "semantic_left_dir": str(args.semantic_left_dir),
            "semantic_right_dir": str(args.semantic_right_dir),
            "frozen_baseline_dir": str(args.frozen_baseline_dir),
        },
        "interpretation_boundary": summary["interpretation_boundary"],
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output_dir),
        "experiment_direct": experiment_direct,
        "baseline_direct": summary["baseline_direct_count"],
        "control_arm_reproduced": summary["control_arm_reproduced_frozen_baseline"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
