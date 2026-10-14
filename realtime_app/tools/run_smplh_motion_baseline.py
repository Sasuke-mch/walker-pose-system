"""Track and export the complete SMPL-H model using causal raw-stereo 2D feedback."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from pose_app.fisheye_camera import load_stereo_fisheye  # noqa: E402
from pose_app.smplh_motion_tracker import SMPLHMotionConfig, SMPLHMotionTracker, SMPLHObservationMap  # noqa: E402
from pose_app.smplh_first_frame_labels import apply_first_frame_labels, refine_first_frame_2d  # noqa: E402
from run_stereo_pva_baseline import (  # noqa: E402
    JOINT_INDICES, JOINT_NAMES, _json_array, _load_predictions, _plot_curves,
    _read_capture_times, _stats,
)


def run(args) -> dict:
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"refusing nonempty output: {args.output}")
    left, right = _load_predictions(args.left_predictions, "left"), _load_predictions(args.right_predictions, "right")
    if left.shape != right.shape or len(left) < 2:
        raise ValueError("prediction frame sets differ or have fewer than two frames")
    cal = load_stereo_fisheye(args.calibration_dir)
    if (cal.image_width, cal.image_height) != (1920, 1080):
        raise ValueError("PMPose raw reader requires 1920x1080 calibration")
    times, skew, pair_ids, capture_path = _read_capture_times(args.input_pairs, len(left))
    model = SMPLHObservationMap(args.initial_fit, args.smplh_model, args.regressor, args.vposer_dir,
                                args.mano_left, args.mano_right, args.device)
    args.output.mkdir(parents=True)
    effective_initial_fit = args.initial_fit
    if args.first_frame_labels is not None:
        manual, label_record = apply_first_frame_labels(args.first_frame_labels, args.input_pairs, left, right)
        (args.output/"first_frame_labels.json").write_text(json.dumps(label_record, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        if args.first_frame_refine_steps > 0:
            effective_initial_fit = refine_first_frame_2d(model, cal, np.array([left[0, JOINT_INDICES], right[0, JOINT_INDICES]]),
                                                          manual, args.output/"first_frame_fit", args.first_frame_refine_steps)
    config = SMPLHMotionConfig()
    tracker = SMPLHMotionTracker(model, cal, config)
    parameters = {"inputs": {key: str(getattr(args, key).resolve()) for key in
                             ("left_predictions", "right_predictions", "input_pairs", "calibration_dir",
                              "initial_fit", "smplh_model", "regressor", "vposer_dir", "mano_left", "mano_right")},
                  "capture_pairs": str(capture_path.resolve()), "configuration": asdict(config),
                  "device": args.device, "frames": len(left), "joint_indices_coco17": JOINT_INDICES,
                  "effective_initial_fit": str(effective_initial_fit.resolve()),
                  "first_frame_labels": None if args.first_frame_labels is None else str(args.first_frame_labels.resolve()),
                  "first_frame_refine_steps_this_run": args.first_frame_refine_steps,
                  "fit_object": "complete_SMPLH_model", "observation": "raw_fisheye_2d_only",
                  "regressor_role": "model_vertices_to_2d_observation_skeleton", "world_state_available": False,
                  "fixed_body_rotation_indices_zero_based_excluding_root": model.fixed_body_indices,
                  "source": ["realtime_app/pose_app/smplh_motion_tracker.py", "realtime_app/tools/run_smplh_motion_baseline.py"]}
    (args.output / "run_parameters.json").write_text(json.dumps(parameters, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    records, outputs, elapsed = [], [], []
    all_errors, used_errors, pixel_reasons = [], [], Counter()
    states, joint_states, segments = [], [], []
    with (args.output / "trajectory.jsonl").open("w", encoding="utf-8") as stream:
        for i in range(len(left)):
            l, r = left[i, JOINT_INDICES], right[i, JOINT_INDICES]
            start = perf_counter()
            result = tracker.step(float(times[i]), l, r)
            elapsed.append((perf_counter() - start) * 1000)
            outputs.append(result)
            status = result["joint_status"].tolist()
            states.append(np.c_[result["position_mm"], result["velocity_mm_s"], result["acceleration_mm_s2"]])
            joint_states.append(status)
            segments.append(np.repeat(result["segment_id"], 12))
            valid, _ = tracker._supported_pixels(np.array([l, r]), epipolar=False)
            error = np.full((2, 12), np.nan)
            if result["projected_pixels"] is not None and result["pose_available"]:
                error = np.linalg.norm(result["projected_pixels"] - np.array([l, r])[..., :2], axis=2)
                all_errors.extend(error[valid].tolist())
                used_errors.extend(error[result["used_views"]].tolist())
            pixel_reasons.update(result["view_reason"].ravel().tolist())
            record = {"frame_index": i, "capture_pair_id": int(pair_ids[i]), "timestamp_s": float(times[i]),
                      "left_right_host_return_skew_ms": float(skew[i]), "model_status": result["status"],
                      "pose_available": bool(result["pose_available"]), "segment_id": result["segment_id"],
                      "joint_names": JOINT_NAMES, "status": status, "used_views": result["used_views"].tolist(),
                      "view_reason": result["view_reason"].tolist(), "position_mm": _json_array(result["position_mm"]),
                      "velocity_mm_s": _json_array(result["velocity_mm_s"]), "acceleration_mm_s2": _json_array(result["acceleration_mm_s2"]),
                      "derivative_valid": result["derivative_valid"].tolist(), "position_std_m": _json_array(result["position_std_m"]),
                      "derivative_reason": result["derivative_reason"].tolist(), "update_scale": result["update_scale"],
                      "raw_velocity_mm_s": _json_array(result["raw_velocity_mm_s"]),
                      "raw_acceleration_mm_s2": _json_array(result["raw_acceleration_mm_s2"]),
                      "joint_observation_age_s": _json_array(result["joint_observation_age_s"]), "reprojection_px": _json_array(error),
                      "translation_m": _json_array(result["translation_m"]), "root_rotation": _json_array(result["root_rotation"]),
                      "vposer_latent": _json_array(result["vposer_latent"]), "generalized_velocity": _json_array(result["generalized_velocity"]),
                      "generalized_acceleration": _json_array(result["generalized_acceleration"]), "state_covariance_diag": _json_array(result["covariance_diag"]),
                      "coordinate_frame": "current_left_camera", "derivative_definition": "derivative_of_model_point_camera_coordinate_signal",
                      "world_state_available": False}
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            records.append(record)
            if i % 64 == 0:
                print(f"frame {i}/{len(left)} {result['status']} views={result['used_views'].sum()}", flush=True)
    available = np.array([o["pose_available"] for o in outputs])
    def array(key):
        values = np.asarray([o[key] for o in outputs])
        values[~available] = np.nan
        return values
    body_rotations = array("body_rotations")
    # Convert only available rotations; invalid frames remain NaN in all published model fields.
    body_pose = np.full((len(outputs), 63), np.nan, dtype=np.float32)
    global_orient = np.full((len(outputs), 3), np.nan, dtype=np.float32)
    body_pose[available] = Rotation.from_matrix(body_rotations[available].reshape(-1, 3, 3)).as_rotvec().reshape(-1, 63)
    global_orient[available] = Rotation.from_matrix(array("root_rotation")[available]).as_rotvec()
    with np.load(effective_initial_fit, allow_pickle=False) as init:
        frozen = {key: init[key].copy() for key in ("betas", "left_hand_pose", "right_hand_pose")}
    np.savez_compressed(args.output / "result.npz", pair_id=np.arange(len(outputs)), timestamp_s=times,
                        pose_available=available, vertices=array("vertices_m"), faces=model.faces,
                        predicted_coco=array("coco_m"), smplh_joints=array("joints_m"),
                        transl=array("translation_m"), global_orient=global_orient, body_pose=body_pose,
                        root_rotation=array("root_rotation"), body_rotations=body_rotations,
                        vposer_latent=array("vposer_latent"), generalized_velocity=array("generalized_velocity"),
                        generalized_acceleration=array("generalized_acceleration"),
                        joint_velocity_mm_s=np.array([o["velocity_mm_s"] for o in outputs]),
                        joint_acceleration_mm_s2=np.array([o["acceleration_mm_s2"] for o in outputs]),
                        derivative_valid=np.array([o["derivative_valid"] for o in outputs]),
                        raw_joint_velocity_mm_s=np.array([o["raw_velocity_mm_s"] for o in outputs]),
                        raw_joint_acceleration_mm_s2=np.array([o["raw_acceleration_mm_s2"] for o in outputs]),
                        segment_id=np.array([o["segment_id"] for o in outputs]), **frozen)
    joints = array("joints_m")
    limb_pairs = ((16, 18), (18, 20), (17, 19), (19, 21), (1, 4), (4, 7), (2, 5), (5, 8))
    lengths = np.stack([np.linalg.norm(joints[:, a]-joints[:, b], axis=1) for a, b in limb_pairs], axis=1)
    speeds = np.array([o["velocity_mm_s"] for o in outputs]) / 1000
    accelerations = np.array([o["acceleration_mm_s2"] for o in outputs]) / 1000
    summary = {"status": "completed_engineering_SMPLH_motion_baseline", "frames": len(outputs),
               "output_model": "SMPLH:6890 vertices,13776 faces,full body pose,root,betas,frozen hand poses",
               "observation_term": "2d_stereo_pixels_only;no_triangulation_or_3d_position_term",
               "pose_available_frames": int(available.sum()), "frame_status_counts": dict(Counter(r["model_status"] for r in records)),
               "joint_status_counts": dict(Counter(np.asarray(joint_states).ravel())), "view_reason_counts": dict(pixel_reasons),
               "all_valid_raw_view_reprojection_px": _stats(np.array(all_errors)), "used_view_reprojection_px": _stats(np.array(used_errors)),
               "derivative_available_joint_samples": int(np.array([o["derivative_valid"] for o in outputs]).sum()),
               "derivative_reason_counts": dict(Counter(np.array([o["derivative_reason"] for o in outputs]).ravel())),
               "update_gain_scale": _stats(np.array([o["update_scale"] for o in outputs if o["update_scale"] > 0])),
               "reported_speed_norm_m_s": _stats(np.linalg.norm(speeds, axis=2)),
               "reported_acceleration_norm_m_s2": _stats(np.linalg.norm(accelerations, axis=2)),
               "raw_acceleration_norm_m_s2_before_derivative_mask": _stats(np.linalg.norm(np.array([o["raw_acceleration_mm_s2"] for o in outputs])/1000, axis=2)),
               "internal_kinematic_limb_length_spread_max_mm": float(np.nanmax(np.nanmax(lengths, axis=0)-np.nanmin(lengths, axis=0))*1000),
               "step_ms_model_geometry_state_update_and_derivative_decode": _stats(np.asarray(elapsed)),
               "timestamp_span_s": float(times[-1]), "dt_s": _stats(np.diff(times)), "abs_host_return_skew_ms": _stats(abs(skew)),
               "configuration": asdict(config), "coordinate_frame": "current_left_camera", "world_state_available": False,
               "initialization": "frame0-only model parameters; when labels supplied, refine frame0 in pure 2D and freeze new shape; no stored 3D observations read",
               "limits": ["No independent metric 3D or derivative truth; pose availability and smoothness are not accuracy.",
                          "Shape from one frame is frozen and can be biased; regressed surface keypoint distances need not be perfectly rigid.",
                          "R/Q and covariance are engineering assumptions; ignored model bias is not covered by covariance.",
                          "Unobserved points may be inferred from the model, but derivatives expire after 0.1s without own pixel support.",
                          "All derivatives include camera motion. No world, Stage or heel contact constraint is estimated.",
                          "Host-return timestamps do not prove exposure times or stereo synchronization.",
                          "Combined model and conservative parameter changes; comparison to free-point tracker is not a single-variable ablation."]}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    _plot_curves(args.output, times, np.array(states), np.array(joint_states), np.array(segments))
    print(json.dumps({k: summary[k] for k in ("frames", "pose_available_frames", "frame_status_counts", "all_valid_raw_view_reprojection_px", "reported_acceleration_norm_m_s2")}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("left_predictions", "right_predictions", "input_pairs", "initial_fit", "output"):
        parser.add_argument("--"+key.replace("_", "-"), type=Path, required=True)
    defaults = {"calibration_dir": "realtime_app/calibration/results", "smplh_model": "third_party/WiLoR/mano_data/models/SMPLH_male.pkl",
                "regressor": "models/smpl/J_regressor_coco.npy", "vposer_dir": "models/VPoser02_05/V02_05",
                "mano_left": "third_party/WiLoR/mano_data/models/MANO_LEFT.pkl", "mano_right": "third_party/WiLoR/mano_data/models/MANO_RIGHT.pkl"}
    for key, default in defaults.items():
        parser.add_argument("--"+key.replace("_", "-"), type=Path, default=Path(default))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--first-frame-labels", type=Path, default=None)
    parser.add_argument("--first-frame-refine-steps", type=int, default=500,
                        help="0 reuses an already refined single-frame input model")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
