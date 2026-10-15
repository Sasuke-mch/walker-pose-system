"""Compare pixel trackers on analytic motion of the same complete SMPL-H body.

Known shape and initial pose are shared engineering conditions. Analytic truth
is evaluation-only; no true rate is given to a tracker. Local articulation is
fixed, so these clips do not establish accuracy for articulated human motion.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from pose_app.fisheye_camera import load_stereo_fisheye, fisheye_project_numpy  # noqa: E402
from pose_app.smplh_motion_tracker import SMPLHMotionConfig, SMPLHMotionTracker, SMPLHObservationMap, MotionPose  # noqa: E402
from pose_app.smplh_iterated_motion_tracker import SMPLHIteratedMotionTracker, IteratedMAPConfig  # noqa: E402


VARIANTS = (("ekf_sigma15", "ekf", 1.5), ("map_sigma15", "map", 1.5),
            ("map_sigma075", "map", .75))


def analytic_motion(points, times, kind):
    """Rigid point motion and exact derivatives, independent of tracker stencils."""
    times = np.asarray(times)
    frequency = 2.1 if kind == "turn" else 1.8
    amplitude = np.array([.075, .025, .035])
    angle_frequency, angle_amplitude = 1.4, .12
    if kind == "turn":
        displacement = (1-np.cos(frequency*times))[:, None]*amplitude
        speed = (frequency*np.sin(frequency*times))[:, None]*amplitude
        acceleration = (frequency**2*np.cos(frequency*times))[:, None]*amplitude
        angle = angle_amplitude*(1-np.cos(angle_frequency*times))
        omega = angle_amplitude*angle_frequency*np.sin(angle_frequency*times)
        alpha = angle_amplitude*angle_frequency**2*np.cos(angle_frequency*times)
    else:
        displacement = np.sin(frequency*times)[:, None]*amplitude
        speed = (frequency*np.cos(frequency*times))[:, None]*amplitude
        acceleration = (-frequency**2*np.sin(frequency*times))[:, None]*amplitude
        angle = angle_amplitude*np.sin(angle_frequency*times)
        omega = angle_amplitude*angle_frequency*np.cos(angle_frequency*times)
        alpha = -angle_amplitude*angle_frequency**2*np.sin(angle_frequency*times)
    Z = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]])
    if kind == "camera":
        # Fixed world points, moving camera. Score CAMERA-signal derivatives,
        # which are generally nonzero even though the world body is stationary.
        rotations = Rotation.from_rotvec(np.c_[np.zeros((len(times), 2)), -angle]).as_matrix()
        position = np.einsum("nij,nvj->nvi", rotations, points[None]-displacement[:, None])
        rotated_speed = np.einsum("nij,nj->ni", rotations, speed)[:, None]
        rotated_acceleration = np.einsum("nij,nj->ni", rotations, acceleration)[:, None]
        cross = position @ Z.T
        velocity = -omega[:, None, None]*cross-rotated_speed
        acc = (-alpha[:, None, None]*cross+omega[:, None, None]**2*(cross @ Z.T)
               +2*omega[:, None, None]*(rotated_speed @ Z.T)-rotated_acceleration)
        translation = -np.einsum("nij,nj->ni", rotations, displacement)
    else:
        rotations = Rotation.from_rotvec(np.c_[np.zeros((len(times), 2)), angle]).as_matrix()
        center = points[[11, 12]].mean(axis=0)
        relative = np.einsum("nij,vj->nvi", rotations, points-center)
        position = center+relative+displacement[:, None]
        cross = relative @ Z.T
        velocity = omega[:, None, None]*cross+speed[:, None]
        acc = alpha[:, None, None]*cross+omega[:, None, None]**2*(cross @ Z.T)+acceleration[:, None]
        translation = center-np.einsum("nij,j->ni", rotations, center)+displacement
    return position, velocity, acc, rotations, translation


def norm_rmse(estimate, truth, valid):
    available = valid & np.isfinite(estimate).all(axis=-1)
    return None if not available.any() else float(np.sqrt(np.mean(np.sum((estimate[available]-truth[available])**2, axis=-1))))


def summarize_saved_clips(directory):
    """Report raw and published errors with coverage and pairwise intersections."""
    results = json.loads((directory/"results.json").read_text(encoding="utf-8"))
    clips = {row["clip"]["id"]: row["clip"] for row in results["clips"]}
    aggregate = {}
    for split in ("development", "held_out"):
        paths = [directory/(name+".npz") for name, clip in clips.items() if clip["split"] == split]
        joined = {}
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                keep = data["time"] >= .3
                for key in data.files:
                    if key != "time":
                        joined.setdefault(key, []).append(data[key][keep])
        arrays = {key: np.concatenate(parts) for key, parts in joined.items()}
        denominator = len(arrays["truth_position"])*12
        variants = {}
        for name, _, _ in VARIANTS:
            row = {"denominator_joint_samples_after0_3s": denominator,
                   "published_derivative_samples": int(arrays[name+"_valid"].sum()),
                   "available_pose_joint_samples": int(arrays[name+"_pose_available"].sum())*12}
            for key in ("position", "velocity", "acceleration", "raw_velocity", "raw_acceleration"):
                truth_key = key.removeprefix("raw_")
                row[key+"_available_norm_rmse_m_units"] = norm_rmse(arrays[name+"_"+key], arrays["truth_"+truth_key], np.ones((len(arrays["truth_position"]), 12), bool))
            variants[name] = row
        paired = {}
        for first, second in (("ekf_sigma15", "map_sigma15"), ("map_sigma15", "map_sigma075")):
            mask = arrays[first+"_valid"] & arrays[second+"_valid"]
            paired[first+"_vs_"+second] = {"common_samples": int(mask.sum()),
                "denominator": denominator, "variants": {name: {key: norm_rmse(arrays[name+"_"+key], arrays["truth_"+key], mask)
                for key in ("position", "velocity", "acceleration")} for name in (first, second)}}
        aggregate[split] = {"variants": variants, "paired_published": paired}
    results["aggregate_after_warmup"] = aggregate
    results["coverage_boundary"] = "RMSE conditions on finite estimates;all failures remain explicit in full denominators and coverage;pairwise-common is secondary"
    (directory/"results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return aggregate


def evaluate(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("refusing nonempty known-motion output")
    args.output.mkdir(parents=True, exist_ok=True)
    p = json.loads(args.run_parameters.read_text(encoding="utf-8"))
    inputs = p["inputs"]
    model = SMPLHObservationMap(Path(p["effective_initial_fit"]), Path(inputs["smplh_model"]),
        Path(inputs["regressor"]), Path(inputs["vposer_dir"]), Path(inputs["mano_left"]),
        Path(inputs["mano_right"]), args.device)
    cal = load_stereo_fisheye(inputs["calibration_dir"])
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))["synthetic_protocol"]
    data = model.evaluate(model.initial_pose)
    initial_points = data["coco_m"][0]
    pelvis_rest = data["joints_m"][0, 0]-model.initial_pose.translation_m
    times = np.r_[0., np.cumsum(np.resize(protocol["dt_pattern_s"], protocol["frames_each"]-1))]
    summaries, common = [], {}
    for index, clip in enumerate(protocol["clips"]):
        position, velocity, acceleration, rotations, translation = analytic_motion(initial_points, times, clip["motion"])
        # The analytic rigid action agrees with the actual model's shaped-root
        # convention. This checks source geometry, not derivative truth.
        sample = len(times)//2
        q = MotionPose(rotations[sample] @ (model.initial_pose.translation_m+pelvis_rest)+translation[sample]-pelvis_rest,
                       rotations[sample] @ model.initial_pose.root_rotation, model.initial_pose.latent.copy())
        parity = float(np.max(abs(model.evaluate(q)["coco_m"][0]-position[sample])))
        if parity > 3e-6:
            raise ValueError(f"analytic rigid action and actual SMPL-H disagree: {parity}")
        right = position @ cal.R_cam0_to_cam1.T + cal.T_cam0_to_cam1_mm/1000
        projected = np.stack((fisheye_project_numpy(position[:, 5:17], cal.K0, cal.D0),
                             fisheye_project_numpy(right[:, 5:17], cal.K1, cal.D1)), axis=1)
        rng = np.random.default_rng(20261015+index)
        pixels = np.concatenate((projected+rng.normal(0, clip["noise_px"], projected.shape),
                                 np.ones(projected.shape[:-1]+(1,))), axis=-1)
        if clip["occlusion"]:
            pixels[28:33] = np.nan
        after_warmup = np.broadcast_to((times >= .3)[:, None], (len(times), 12))
        arrays = dict(time=times, pixels=pixels, truth_position=position[:, 5:17],
                      truth_velocity=velocity[:, 5:17], truth_acceleration=acceleration[:, 5:17])
        estimates = {}
        for name, method, sigma in VARIANTS:
            config = SMPLHMotionConfig(max_latent_correction_norm=1.2, latent_prior_sigma=sigma)
            tr = (SMPLHMotionTracker(model, cal, config) if method == "ekf" else
                  SMPLHIteratedMotionTracker(model, cal, config, IteratedMAPConfig(maximum_iterations=args.map_iterations)))
            output, elapsed = [], []
            for i, now in enumerate(times):
                started = perf_counter()
                output.append(tr.step(float(now), pixels[i, 0], pixels[i, 1]))
                elapsed.append((perf_counter()-started)*1000)
            measured = {"position": np.asarray([o["position_mm"] for o in output])/1000,
                        "velocity": np.asarray([o["velocity_mm_s"] for o in output])/1000,
                        "acceleration": np.asarray([o["acceleration_mm_s2"] for o in output])/1000,
                        "raw_velocity": np.asarray([o["raw_velocity_mm_s"] for o in output])/1000,
                        "raw_acceleration": np.asarray([o["raw_acceleration_mm_s2"] for o in output])/1000}
            valid = np.asarray([o["derivative_valid"] for o in output])
            estimates[name] = measured | {"valid": valid}
            for key, values in measured.items():
                arrays[name+"_"+key] = values
            arrays[name+"_valid"] = valid
            arrays[name+"_pose_available"] = np.asarray([o["pose_available"] for o in output])
            row = {"clip": clip, "variant": name, "frames": len(times), "model_parity_max_m": parity,
                   "motion_config": asdict(config), "step_median_ms": float(np.median(elapsed)),
                   "step_p95_ms": float(np.percentile(elapsed, 95)),
                   "pose_available_frames": sum(o["pose_available"] for o in output),
                   "derivative_valid_samples": int(valid.sum()), "all_joint_samples": len(times)*12,
                   "derivative_valid_after_warmup_samples": int((valid & after_warmup).sum()),
                   "after_warmup_joint_samples": int(after_warmup.sum()),
                   "optimizer_unconverged_frames": sum(o.get("iterated_update", {}).get("status") == "feasible_unconverged" for o in output)}
            for key in ("position", "velocity", "acceleration"):
                for period, mask in (("all", np.ones_like(valid)), ("after_warmup", after_warmup)):
                    truth = arrays["truth_"+key]
                    row[key+"_"+period+"_available_norm_rmse_m_units"] = norm_rmse(measured[key], truth, mask)
            summaries.append(row)
            print(clip["id"], name, "pose", row["pose_available_frames"], "derivatives", row["derivative_valid_samples"],
                  "position/v/a", *[row[key+"_after_warmup_available_norm_rmse_m_units"] for key in ("position", "velocity", "acceleration")], flush=True)
        mask = after_warmup & np.logical_and.reduce([estimates[name]["valid"] for name, _, _ in VARIANTS])
        common[clip["id"]] = {"split": clip["split"], "samples": int(mask.sum()), "denominator": int(after_warmup.sum()),
            "variants": {name: {key: norm_rmse(estimates[name][key], arrays["truth_"+key], mask)
                                for key in ("position", "velocity", "acceleration")} for name, _, _ in VARIANTS}}
        np.savez_compressed(args.output/(clip["id"]+".npz"), **arrays)
        (args.output/"results.json").write_text(json.dumps({"clips": summaries, "common_after_warmup": common,
            "scope": "known-model rigid-motion engineering check;no articulated-local-motion or real-world accuracy claim",
            "input_parameters": str(args.run_parameters.resolve()), "protocol": str(args.protocol.resolve()),
            "optimizer_iterations": args.map_iterations}, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    summarize_saved_clips(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-parameters", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--map-iterations", type=int, default=15)
    evaluate(parser.parse_args())
