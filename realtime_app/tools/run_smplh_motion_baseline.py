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
from pose_app.smplh_first_frame_labels import apply_first_frame_labels, apply_first_frame_observations, refine_first_frame_2d  # noqa: E402
from run_stereo_pva_baseline import (  # noqa: E402
    JOINT_INDICES, JOINT_NAMES, _json_array, _load_predictions, _plot_curves,
    _read_capture_times, _stats,
)


def run(args) -> dict:
    frame0_observations = getattr(args, 'first_frame_observations', None)
    if frame0_observations is not None and args.first_frame_labels is not None:
        raise ValueError("choose detector frame0 observations or manual labels")
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
    if args.shape_fit is not None:
        if (args.first_frame_shape_policy != "fixed" or (args.first_frame_labels is None and frame0_observations is None)):
            raise ValueError("a calibrated shape requires fixed-shape frame0 pose refinement with labels")
        # Read only the explicitly supplied shared shape. Future poses and mesh
        # fields are never used as a frame0 pose or as tracking observations.
        with np.load(args.shape_fit, allow_pickle=False) as fit:
            shape = fit["betas"].copy()
        if shape.shape != (1, 10) or not np.isfinite(shape).all():
            raise ValueError("calibrated shape must be one finite shared1x10 beta")
        if args.first_frame_refine_steps == 0 and not np.array_equal(shape, model.betas.detach().cpu().numpy()):
            raise ValueError("reused frame0 pose must already have the supplied calibrated beta")
        model.betas = model.torch.as_tensor(shape, dtype=model.torch.float32, device=model.device)
    args.output.mkdir(parents=True)
    effective_initial_fit = args.initial_fit
    if frame0_observations is not None:
        source_record = apply_first_frame_observations(frame0_observations, args.input_pairs, left, right)
        (args.output/"first_frame_observations.json").write_text(json.dumps(source_record, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        if args.first_frame_refine_steps > 0:
            effective_initial_fit = refine_first_frame_2d(model, cal, np.array([left[0, JOINT_INDICES], right[0, JOINT_INDICES]]),
                                                          np.zeros((2, 12), dtype=bool), args.output/"first_frame_fit",
                                                          args.first_frame_refine_steps, optimize_shape=args.first_frame_shape_policy == "refine",
                                                          lbfgs_steps=getattr(args, 'first_frame_lbfgs_steps', 0),
                                                          latent_prior_sigma=getattr(args, 'first_frame_latent_prior_sigma', 1.5),
                                                          lbfgs_double=getattr(args, 'first_frame_lbfgs_double', False))
    if args.first_frame_labels is not None:
        manual, label_record = apply_first_frame_labels(args.first_frame_labels, args.input_pairs, left, right)
        (args.output/"first_frame_labels.json").write_text(json.dumps(label_record, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        if args.first_frame_refine_steps > 0:
            effective_initial_fit = refine_first_frame_2d(model, cal, np.array([left[0, JOINT_INDICES], right[0, JOINT_INDICES]]),
                                                          manual, args.output/"first_frame_fit", args.first_frame_refine_steps,
                                                          optimize_shape=args.first_frame_shape_policy == "refine",
                                                          lbfgs_steps=getattr(args, 'first_frame_lbfgs_steps', 0),
                                                          latent_prior_sigma=getattr(args, 'first_frame_latent_prior_sigma', 1.5),
                                                          lbfgs_double=getattr(args, 'first_frame_lbfgs_double', False))
    config = SMPLHMotionConfig(max_latent_correction_norm=args.max_latent_correction_norm,
                              latent_prior_sigma=args.latent_prior_sigma)
    optimizer_config = None
    scene_data = None
    if args.world_scene is not None:
        if args.update_method != "ekf":
            raise ValueError("iterated-map currently estimates camera motion states; world contact has its own MAP solver")
        from pose_app.smplh_world_motion_tracker import SMPLHWorldMotionTracker, WorldGroundConfig
        from pose_app.smplh_ground_constraints import load_sole_vertex_indices
        scene = args.world_scene
        source = json.loads((scene/"scene_sources.json").read_text(encoding="utf-8"))
        if any(source.get(key, False) for key in ("old_stage_jsonl_read", "old_dynamic_ground_read", "old_dynamic_walker_pose_read")):
            raise ValueError("old dynamic inputs cannot define this world's transforms")
        for name, key in (("left_predictions", "raw_left_pmpose"), ("right_predictions", "raw_right_pmpose"), ("input_pairs", "input_pair_dir")):
            if Path(source[key]).resolve() != getattr(args, name).resolve():
                raise ValueError("world scene and body observation sources differ")
        with np.load(scene/"scene_transforms.npz", allow_pickle=False) as fitted:
            scene_data = {key: fitted[key].copy() for key in ("rotation_ground_from_left", "translation_ground_from_left_mm", "pair_timestamp_sec")}
        if len(scene_data["rotation_ground_from_left"]) != len(times) or not np.allclose(times, scene_data["pair_timestamp_sec"], atol=1e-7, rtol=0):
            raise ValueError("world scene and real capture timestamps differ")
        scene_data["stage"] = [json.loads(line)["stage"]["operational"] for line in (scene/"stage1_stage2.jsonl").read_text(encoding="utf-8").splitlines()]
        scene_data["pose_status"] = [json.loads(line)["pose_audit"]["status"] for line in (scene/"dynamic_ground_pose.jsonl").read_text(encoding="utf-8").splitlines()]
        tracker = SMPLHWorldMotionTracker(model, cal, config, scene_data["rotation_ground_from_left"][0],
                 scene_data["translation_ground_from_left_mm"][0]/1000, load_sole_vertex_indices(args.surface_sets),
                 WorldGroundConfig(max_heel_displacement_m=args.max_foot_displacement_mm/1000),
                 constrain_ground=args.ground_constraint_policy == "contact")
    else:
        if args.ground_constraint_policy != "none":
            raise ValueError("contact requires an explicit same-run world scene")
        if args.update_method == "iterated-map":
            from pose_app.smplh_iterated_motion_tracker import SMPLHIteratedMotionTracker, IteratedMAPConfig
            optimizer_config = IteratedMAPConfig(maximum_iterations=args.map_iterations)
            tracker = SMPLHIteratedMotionTracker(model, cal, config, optimizer_config)
        else:
            tracker = SMPLHMotionTracker(model, cal, config)
    parameters = {"inputs": {key: str(getattr(args, key).resolve()) for key in
                             ("left_predictions", "right_predictions", "input_pairs", "calibration_dir",
                              "initial_fit", "smplh_model", "regressor", "vposer_dir", "mano_left", "mano_right")},
                  "capture_pairs": str(capture_path.resolve()), "configuration": asdict(config),
                  "update_method": args.update_method,
                  "iterated_map_configuration": None if optimizer_config is None else asdict(optimizer_config),
                  "device": args.device, "frames": len(left), "joint_indices_coco17": JOINT_INDICES,
                  "effective_initial_fit": str(effective_initial_fit.resolve()),
                  "first_frame_labels": None if args.first_frame_labels is None else str(args.first_frame_labels.resolve()),
                  "first_frame_observations": None if frame0_observations is None else str(frame0_observations.resolve()),
                  "first_frame_refine_steps_this_run": args.first_frame_refine_steps,
                  "first_frame_lbfgs_steps_this_run": getattr(args, 'first_frame_lbfgs_steps', 0),
                  "first_frame_latent_prior_sigma": getattr(args, 'first_frame_latent_prior_sigma', 1.5),
                  "first_frame_lbfgs_double": getattr(args, 'first_frame_lbfgs_double', False),
                  "first_frame_shape_policy": args.first_frame_shape_policy,
                  "shape_fit": None if args.shape_fit is None else str(args.shape_fit.resolve()),
                  "shape_information": "precalibrated_shared_shape_from_full_sequence" if args.shape_fit is not None else "frame0_seed_or_refinement",
                  "fit_object": "complete_SMPLH_model", "observation": "raw_fisheye_2d_only",
                  "regressor_role": "model_vertices_to_2d_observation_skeleton", "world_state_available": False,
                  "fixed_body_rotation_indices_zero_based_excluding_root": model.fixed_body_indices,
                  "source": ["realtime_app/pose_app/smplh_motion_tracker.py", "realtime_app/tools/run_smplh_motion_baseline.py"],
                  "world_scene": None if args.world_scene is None else str(args.world_scene.resolve()),
                  "ground_constraint_policy": args.ground_constraint_policy,
                  "world_pose_certified": False,
                  "model_storage_frame": "current_left_camera",
                  "coordinate_frame": "current_left_camera" if scene_data is None else "stage_conditioned_fixed_ground"}
    if scene_data is not None:
        parameters.update(world_state_available=True, scene_role="stage_conditioned_world_estimation",
                          ground_configuration=asdict(tracker.ground_config), surface_sets=str(args.surface_sets.resolve()))
    if optimizer_config is not None:
        parameters["source"].append("realtime_app/pose_app/smplh_iterated_motion_tracker.py")
    (args.output / "run_parameters.json").write_text(json.dumps(parameters, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    records, outputs, elapsed = [], [], []
    all_errors, used_errors, pixel_reasons = [], [], Counter()
    eligible_raw_slots = 0
    states, joint_states, segments = [], [], []
    with (args.output / "trajectory.jsonl").open("w", encoding="utf-8") as stream:
        for i in range(len(left)):
            l, r = left[i, JOINT_INDICES], right[i, JOINT_INDICES]
            start = perf_counter()
            if scene_data is not None:
                tracker.set_scene_frame(scene_data["rotation_ground_from_left"][i],
                  scene_data["translation_ground_from_left_mm"][i]/1000, scene_data["stage"][i], scene_data["pose_status"][i])
            result = tracker.step(float(times[i]), l, r)
            elapsed.append((perf_counter() - start) * 1000)
            if scene_data is not None:
                R, t = tracker.camera_rotation, tracker.camera_translation
                for key in ("vertices_m", "joints_m", "coco_m", "translation_m", "root_rotation"):
                    result["world_"+key] = result[key].copy()
                for key in ("vertices_m", "joints_m", "coco_m"):
                    result[key] = (result[key]-t) @ R
                result["translation_m"] = R.T @ (result["translation_m"]+tracker.pelvis_offset-t)-tracker.pelvis_offset
                result["root_rotation"] = R.T @ result["root_rotation"]
            outputs.append(result)
            status = result["joint_status"].tolist()
            states.append(np.c_[result["position_mm"], result["velocity_mm_s"], result["acceleration_mm_s2"]])
            joint_states.append(status)
            segments.append(np.repeat(result["segment_id"], 12))
            valid, _ = tracker._supported_pixels(np.array([l, r]), epipolar=False)
            eligible_raw_slots += int(valid.sum())
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
                      "prediction_rejection": result["prediction_rejection"],
                      "iterated_update": result.get("iterated_update"),
                      "raw_velocity_mm_s": _json_array(result["raw_velocity_mm_s"]),
                      "raw_acceleration_mm_s2": _json_array(result["raw_acceleration_mm_s2"]),
                      "joint_observation_age_s": _json_array(result["joint_observation_age_s"]), "reprojection_px": _json_array(error),
                      "translation_m": _json_array(result["translation_m"]), "root_rotation": _json_array(result["root_rotation"]),
                      "vposer_latent": _json_array(result["vposer_latent"]), "generalized_velocity": _json_array(result["generalized_velocity"]),
                      "generalized_acceleration": _json_array(result["generalized_acceleration"]), "state_covariance_diag": _json_array(result["covariance_diag"]),
                      "coordinate_frame": parameters["coordinate_frame"],
                      "derivative_definition": "derivative_of_model_point_"+parameters["coordinate_frame"]+"_signal",
                      "world_state_available": scene_data is not None, "world_pose_certified": False}
            record["model_storage_frame"] = "current_left_camera"
            if scene_data is not None:
                record.update(ground_constraint=result["ground_constraint"],
                  state_translation_world_m=_json_array(result["world_translation_m"]),
                  state_root_rotation_world=_json_array(result["world_root_rotation"]),
                  scene_pose_status=scene_data["pose_status"][i], stage=scene_data["stage"][i])
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
        frozen = {key: init[key].copy() for key in ("left_hand_pose", "right_hand_pose")}
    frozen["betas"] = model.betas.detach().cpu().numpy().copy()
    world_arrays = {} if scene_data is None else {"world_"+key: array("world_"+name) for key, name in
         (("vertices", "vertices_m"), ("predicted_coco", "coco_m"), ("transl", "translation_m"), ("root_rotation", "root_rotation"))}
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
                        segment_id=np.array([o["segment_id"] for o in outputs]), **frozen, **world_arrays)
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
               "all_raw_view_slots": len(outputs)*24, "eligible_raw_view_slots": eligible_raw_slots,
               "eligible_raw_view_slots_with_available_model": len(all_errors),
               "eligible_raw_view_fit_within25px_slots": int(np.sum(np.asarray(all_errors) <= 25.)),
               "eligible_raw_view_fit_within25px_rate_including_failures": float(np.sum(np.asarray(all_errors) <= 25.)/eligible_raw_slots) if eligible_raw_slots else None,
               "update_method": args.update_method,
               "iterated_update_status_counts": dict(Counter(r["iterated_update"]["status"] for r in records if r["iterated_update"] is not None)),
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
               "initialization": "frame0-only pose seed; current frame0 pixels refine pose; calibrated shared shape when explicitly supplied, otherwise seed/refined frame0 shape; no stored3D observations or future poses read",
               "shape_information": parameters["shape_information"], "shape_fit": parameters["shape_fit"],
               "limits": ["No independent metric 3D or derivative truth; pose availability and smoothness are not accuracy.",
                          "Shape from one frame is frozen and can be biased; regressed surface keypoint distances need not be perfectly rigid.",
                          "R/Q and covariance are engineering assumptions; ignored model bias is not covered by covariance.",
                          "Unobserved points may be inferred from the model, but derivatives expire after 0.1s without own pixel support.",
                          "All derivatives include camera motion. No world, Stage or heel contact constraint is estimated.",
                          "Host-return timestamps do not prove exposure times or stereo synchronization.",
                          "Combined model and conservative parameter changes; comparison to free-point tracker is not a single-variable ablation."]}
    if scene_data is not None:
        constraint_rows = [r["ground_constraint"] for r in records]
        moving = np.array([c["stage2"] for c in constraint_rows])
        summary.update(status="completed_stage_conditioned_world_engineering_candidate",
            coordinate_frame=parameters["coordinate_frame"], world_state_available=True, world_pose_certified=False,
            scene_role="stage_conditioned_world_estimation;fixed_lag_not_independent_visual_odometry",
            ground_constraint_policy=args.ground_constraint_policy,
            model_storage_frame="current_left_camera",
            observation_term="raw_fisheye_stereo_2d_body_data_plus_explicit_stage_conditioned_ground_priors;no_3d_body_measurements",
            ground_configuration=asdict(tracker.ground_config),
            constraint_status_counts=dict(Counter(c["status"] for c in constraint_rows)),
            constraint_satisfied_frames=sum(c.get("constraint_satisfied") is True for c in constraint_rows),
            stage2_frames=int(moving.sum()), stage2_constraint_satisfied_frames=sum(c.get("constraint_satisfied") is True and c["stage2"] for c in constraint_rows),
            sole_height_mm=_stats(np.array([c.get("sole_min_m", [np.nan]*2) for c in constraint_rows])*1000),
            stage2_sole_max_displacement_mm=_stats(np.array([c.get("sole_max_displacement_m") or [np.nan]*2 for c in constraint_rows])[moving]*1000))
        summary["limits"] = ["Stage camera transforms already use triangulated ankle anchors and stop-support assumptions; contact agreement is circular engineering consistency, not independent world accuracy.",
             "Scene uses fixed-lag output nominally0.5s and is not immediate causal output.",
             "Controlled walking assumption: at least one candidate sole within5mm of ground; both only during all declared Stage2 frames.",
             "Whole54-point sole surface and heel ball use episode anchors after feasible contact; transitions retain anchors until Stage1.",
             "Solver uses0.2mm interior margin; output physical5/10mm bounds allow only1micrometre numerical tolerance; active-set covariance and derivative conditioning are local approximations.",
             "No independent metric position/velocity/acceleration or physical contact truth; rejected/failed output remains recorded."]
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    _plot_curves(args.output, times, np.array(states), np.array(joint_states), np.array(segments),
                 "Stage-conditioned fixed ground (not independently certified)" if scene_data is not None
                 else "current left-camera coordinates (includes camera motion)")
    print(json.dumps({k: summary[k] for k in ("frames", "pose_available_frames", "frame_status_counts", "all_valid_raw_view_reprojection_px", "reported_acceleration_norm_m_s2")}, indent=2))
    return summary


def build_parser():
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
    parser.add_argument("--first-frame-observations", type=Path, default=None,
                        help="Explicit same-image raw stereo detector points for pair0 only; excludes manual labels.")
    parser.add_argument("--first-frame-refine-steps", type=int, default=500,
                        help="0 reuses an already refined single-frame input model")
    parser.add_argument("--first-frame-lbfgs-steps", type=int, default=0,
                        help="Optional first-frame L-BFGS polish on the same objective after Adam.")
    parser.add_argument("--first-frame-lbfgs-double", action='store_true',
                        help="Use double precision and scaled parameters for optional L-BFGS initialization.")
    parser.add_argument("--first-frame-latent-prior-sigma", type=float, default=1.5,
                        help="Whole-body VPoser prior for initialization only; no local anatomical prior.")
    parser.add_argument("--first-frame-shape-policy", choices=("refine", "fixed"), default="refine",
                        help="refine reproduces prior frame0 joint shape fitting; fixed optimizes pose only")
    parser.add_argument("--shape-fit", type=Path, default=None,
                        help="explicit pre-calibrated shared beta only; requires fixed-shape frame0 refinement")
    parser.add_argument("--max-latent-correction-norm", type=float, default=.4,
                        help="single-frame latent correction bound; recorded in run parameters")
    parser.add_argument("--update-method", choices=("ekf", "iterated-map"), default="ekf")
    parser.add_argument("--latent-prior-sigma", type=float, default=1.5,
                        help="whole-body VPoser prior sigma; halving sigma multiplies its weight by four")
    parser.add_argument("--map-iterations", type=int, default=15)
    parser.add_argument("--world-scene", type=Path)
    parser.add_argument("--ground-constraint-policy", choices=("none", "contact"), default="none")
    parser.add_argument("--surface-sets", type=Path, default=Path("research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/contact_surface_sets_smplh_v1/contact_vertex_sets.json"))
    parser.add_argument("--max-foot-displacement-mm", type=float, default=10.)
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
