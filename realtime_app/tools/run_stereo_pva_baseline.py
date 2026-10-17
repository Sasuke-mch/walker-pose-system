"""Replay the causal stereo pixel-feedback [p,v,a] baseline and plot limb curves.

Requires a first-frame-only SMPL-H fit (one frame, ID 0). Subsequent measurements
are current raw-fisheye pixels. Capture host-return times give dt; they are not
exposure timestamps. Stage/contact and world motion are outside this baseline.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from pose_app.body_observations import load_ordered_predictions  # noqa: E402
from pose_app.replay_pair_timing import load_capture_pair_times  # noqa: E402
from pose_app.fisheye_camera import load_stereo_fisheye  # noqa: E402
from pose_app.stereo_pva_tracker import StereoPVAConfig, StereoPVATracker  # noqa: E402


JOINT_INDICES = tuple(range(5, 17))
JOINT_NAMES = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
               "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
               "right_knee", "left_ankle", "right_ankle")


# Historical name retained for existing callers and flat imports; the single
# implementation lives in pose_app.replay_pair_timing.load_capture_pair_times.
_read_capture_times = load_capture_pair_times


# Historical name retained for existing callers and flat imports; the single
# implementation lives in pose_app.body_observations.load_ordered_predictions.
_load_predictions = load_ordered_predictions


def _load_initial_fit(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as fitted:
        points = fitted["predicted_coco"].astype(np.float64)
        if points.shape != (1, 17, 3) or fitted["pair_id"].tolist() != [0]:
            raise ValueError("initializer must be a frame-0-only fit; full-sequence fits leak future information")
    if not np.isfinite(points[:, JOINT_INDICES]).all():
        raise ValueError("first-frame limb positions are non-finite")
    return points[0, JOINT_INDICES] * 1000.0  # Current fitter's model output is metres.


def _json_array(values: np.ndarray) -> list:
    def replace(value):
        if isinstance(value, list):
            return [replace(item) for item in value]
        return float(value) if np.isfinite(value) else None
    return replace(np.asarray(values).tolist())


def _stats(values: np.ndarray) -> dict:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return dict(count=int(len(finite)), min=float(np.min(finite)) if len(finite) else None,
                median=float(np.median(finite)) if len(finite) else None,
                p95=float(np.percentile(finite, 95)) if len(finite) else None,
                max=float(np.max(finite)) if len(finite) else None)


def _plot_curves(output: Path, times: np.ndarray, states: np.ndarray,
                 statuses: np.ndarray, segments: np.ndarray,
                 coordinate_label: str = "current left-camera coordinates (includes camera motion)") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = (0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11)
    for block, filename, label in ((1, "velocity_components.png", "Velocity (m/s)"),
                                    (2, "acceleration_components.png", "Acceleration (m/s^2)")):
        fig, axes = plt.subplots(4, 3, figsize=(14, 9), sharex=True, layout="constrained")
        values = states[:, :, block * 3:(block + 1) * 3] / 1000.0
        for ax, j in zip(axes.flat, order):
            curve = values[:, j].copy()
            # Reset priors do not represent motion; do not draw through segment changes.
            breaks = np.r_[False, np.diff(segments[:, j]) != 0]
            curve[breaks | (statuses[:, j] == "reinitialized")] = np.nan
            for axis, style in enumerate(("-", "--", ":")):
                ax.plot(times, curve[:, axis], color="black", linestyle=style, linewidth=0.8, label="xyz"[axis])
            predicted = statuses[:, j] == "predicted"
            ax.fill_between(times, 0, 1, where=predicted, transform=ax.get_xaxis_transform(),
                            color="0.7", alpha=0.3, linewidth=0, label="prediction only")
            ax.set_title(JOINT_NAMES[j], fontsize=10)
            ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("Host-return midpoint time (s)")
        for ax in axes[:, 0]:
            ax.set_ylabel(label)
        fig.suptitle(label + " in " + coordinate_label + "\n"
                     "Zero start states are priors; grey spans are predictions; gaps mark resets/unavailable", fontsize=11)
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output: {args.output}")
    left, right = _load_predictions(args.left_predictions, "left"), _load_predictions(args.right_predictions, "right")
    if left.shape != right.shape or left.shape[1:] != (17, 3) or len(left) < 2:
        raise ValueError("left/right predictions must both have shape (N>=2,17,3)")
    times, skew, pair_ids, capture_path = _read_capture_times(args.input_pairs, len(left))
    initial = _load_initial_fit(args.initial_fit)
    cal = load_stereo_fisheye(args.calibration_dir)
    if (cal.image_width, cal.image_height) != (1920, 1080):
        raise ValueError("the existing PMPose inverse-rotation reader expects 1920x1080 raw pixels")
    config = StereoPVAConfig(min_score=args.min_score, max_reprojection_px=args.max_reprojection_px,
                             pixel_sigma_px=args.pixel_sigma_px,
                             jerk_spectral_density_mm2_s5=args.jerk_spectral_density_mm2_s5)
    tracker = StereoPVATracker(cal, joint_count=12, config=config)
    args.output.mkdir(parents=True, exist_ok=True)
    states, statuses, segments, residuals, step_ms = [], [], [], [], []
    reasons = Counter()
    with (args.output / "trajectory.jsonl").open("w", encoding="utf-8") as stream:
        for i in range(len(left)):
            l, r = left[i, JOINT_INDICES], right[i, JOINT_INDICES]
            start = perf_counter()
            result = tracker.step(float(times[i]), l[:, :2], r[:, :2], l[:, 2], r[:, 2], initial if i == 0 else None)
            step_ms.append((perf_counter() - start) * 1000)
            states.append(result["state"])
            statuses.append(result["status"])
            segments.append(result["segment_id"])
            residuals.append(result["reprojection_px"])
            reasons.update(str(x) for x in result["reason"])
            covariance = np.diagonal(result["covariance"], axis1=1, axis2=2)
            record = {"frame_index": i, "capture_pair_id": int(pair_ids[i]), "timestamp_s": float(times[i]),
                      "left_right_host_return_skew_ms": float(skew[i]), "dt_s": result["dt_s"],
                      "joint_indices_coco17": JOINT_INDICES, "joint_names": JOINT_NAMES,
                      "status": result["status"].tolist(), "reason": result["reason"].tolist(),
                      "segment_id": result["segment_id"].tolist(),
                      "position_mm": _json_array(result["state"][:, :3]),
                      "velocity_mm_s": _json_array(result["state"][:, 3:6]),
                      "acceleration_mm_s2": _json_array(result["state"][:, 6:]),
                      "state_covariance_diag_p_v_a": _json_array(covariance),
                      "prediction_age_s": _json_array(result["prediction_age_s"]),
                      "innovation_squared_mahalanobis": _json_array(result["innovation_squared_mahalanobis"]),
                      "reprojection_px": _json_array(result["reprojection_px"]),
                      "stereo_reprojection_max_px": _json_array(result["stereo_reprojection_max_px"]),
                      "ray_angle_deg": _json_array(result["ray_angle_deg"]),
                      "coordinate_frame": result["coordinate_frame"],
                      "derivative_definition": result["derivative_definition"], "world_state_available": False}
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    states, statuses, segments = np.asarray(states), np.asarray(statuses), np.asarray(segments)
    residuals = np.asarray(residuals)
    updated = statuses == "updated"
    per_joint = {}
    for j, name in enumerate(JOINT_NAMES):
        per_joint[name] = {"status_counts": dict(Counter(statuses[:, j])),
                           "segment_count": int(segments[:, j].max() + 1),
                           "updated_reprojection_each_view_px": _stats(residuals[:, j][updated[:, j]]),
                           "updated_speed_m_s": _stats(np.linalg.norm(states[:, j, 3:6], axis=1)[updated[:, j]] / 1000),
                           "updated_acceleration_m_s2": _stats(np.linalg.norm(states[:, j, 6:], axis=1)[updated[:, j]] / 1000)}
    endpoint_distances = {}
    for name, j, k in (("left_upper_arm", 0, 2), ("right_upper_arm", 1, 3),
                       ("left_forearm", 2, 4), ("right_forearm", 3, 5),
                       ("left_thigh", 6, 8), ("right_thigh", 7, 9),
                       ("left_shank", 8, 10), ("right_shank", 9, 11)):
        both_updated = updated[:, j] & updated[:, k]
        distances = np.linalg.norm(states[:, j, :3] - states[:, k, :3], axis=1)[both_updated]
        endpoint_distances[name] = _stats(distances)
    summary = {
        "status": "completed_engineering_baseline", "frames": len(states), "joint_count": 12,
        "joint_indices_coco17": JOINT_INDICES, "status_counts": dict(Counter(statuses.ravel())),
        "reason_counts": dict(reasons), "per_joint": per_joint,
        "updated_reprojection_each_view_px": _stats(residuals[updated]),
        "updated_speed_norm_m_s": _stats(np.linalg.norm(states[:, :, 3:6], axis=2)[updated] / 1000),
        "updated_acceleration_norm_m_s2": _stats(np.linalg.norm(states[:, :, 6:], axis=2)[updated] / 1000),
        "predicted_speed_norm_m_s": _stats(np.linalg.norm(states[:, :, 3:6], axis=2)[statuses == "predicted"] / 1000),
        "predicted_acceleration_norm_m_s2": _stats(np.linalg.norm(states[:, :, 6:], axis=2)[statuses == "predicted"] / 1000),
        "endpoint_distance_both_updated_mm": endpoint_distances,
        "timestamp_kind": "mean_left_right_host_read_return_not_exposure", "timestamp_span_s": float(times[-1]),
        "dt_s": _stats(np.diff(times)), "abs_host_return_skew_ms": _stats(abs(skew)),
        "step_ms_geometry_prediction_pixel_update_only": _stats(np.asarray(step_ms)),
        "configuration": asdict(config), "coordinate_frame": "current_left_camera",
        "derivative_definition": "v=d(p_C(t))/dt,a=d2(p_C(t))/dt2;includes_camera_motion",
        "world_state_available": False, "stage_used": False, "contact_used": False,
        "initialization": "first_frame_only_SMPLH_2D_fit;zero_v_a_priors;no_reuse_of_first_frame_pixels",
        "inputs": {"left": str(args.left_predictions.resolve()), "right": str(args.right_predictions.resolve()),
                   "selection": str((args.input_pairs / "selection_manifest.csv").resolve()),
                   "capture_pairs": str(capture_path.resolve()), "calibration": str(args.calibration_dir.resolve()),
                   "initial_fit": str(args.initial_fit.resolve())},
        "source": "realtime_app/pose_app/stereo_pva_tracker.py;realtime_app/tools/run_stereo_pva_baseline.py",
        "limitations": ["Camera-relative derivatives include rig motion and are not world velocity/acceleration.",
                        "Independent joints do not preserve bone lengths; no heel/contact constraint is present.",
                        "R/Q and the covariance are uncalibrated engineering assumptions, not certified uncertainty.",
                        "Restarted v/a are zero priors, not measured motion; plots break at each restart.",
                        "Host-return timing does not prove exposure time or stereo synchronization.",
                        "Reprojection uses the same observations as fitting; there is no independent 3D truth."],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    _plot_curves(args.output, times, states, statuses, segments)
    print(json.dumps({"output": str(args.output.resolve()), "frames": len(states),
                      "status_counts": summary["status_counts"],
                      "step_ms": summary["step_ms_geometry_prediction_pixel_update_only"]}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-predictions", type=Path, required=True)
    parser.add_argument("--right-predictions", type=Path, required=True)
    parser.add_argument("--input-pairs", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--initial-fit", type=Path, required=True, help="frame-0-only SMPL-H result.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=0.20)
    parser.add_argument("--max-reprojection-px", type=float, default=25.0)
    parser.add_argument("--pixel-sigma-px", type=float, default=4.0)
    parser.add_argument("--jerk-spectral-density-mm2-s5", type=float, default=1e8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
