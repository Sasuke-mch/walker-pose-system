"""Train synthetic PPO and replay all 448 real frames without masking failure."""

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from .body import PhysicsBody
from .data import load_real
from .environment import PixelTrackingEnv, make_synthetic, pixel_controller
from pose_app.lower_limb_kinematics import three_point_derivatives


def serial(value):
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [serial(x) for x in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def save_json(path, value):
    path.write_text(
        json.dumps(serial(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def stats(values):
    x = np.asarray(values, float).ravel()
    x = x[np.isfinite(x)]
    return dict(
        count=len(x),
        mean=float(x.mean()) if len(x) else None,
        median=float(np.median(x)) if len(x) else None,
        p95=float(np.percentile(x, 95)) if len(x) else None,
        max=float(x.max()) if len(x) else None,
    )


class Progress(BaseCallback):
    def __init__(self, output: Path, wall_limit: float):
        super().__init__()
        self.output = output
        self.wall_limit = wall_limit
        self.started = time.perf_counter()
        self.records = []

    def _on_step(self):
        if self.num_timesteps % 4096 == 0:
            elapsed = time.perf_counter() - self.started
            record = dict(
                steps=self.num_timesteps,
                wall_s=elapsed,
                steps_per_s=self.num_timesteps / elapsed,
            )
            self.records.append(record)
            print(json.dumps(record), flush=True)
            save_json(self.output, self.records)
        return time.perf_counter() - self.started < self.wall_limit


def new_body(dataset, heel_mode="surface"):
    return PhysicsBody(
        dataset.initial_fit, dataset.rotation[0], dataset.translation[0], heel_mode
    )


def prepare(root):
    real = load_real()
    body = new_body(real, heel_mode="rigid")
    (root / "humanoid.xml").write_text(body.xml, encoding="utf-8")
    dataset, synthetic = make_synthetic(real, body, root / "synthetic.npz")
    synthetic["train_clips"] = np.arange(12)
    synthetic["test_clips"] = np.arange(12, 16)
    counts = {
        int(c): int(np.sum(synthetic["clip_id"] == c))
        for c in np.unique(synthetic["clip_id"])
    }
    if any(counts.get(c, 0) < 3 for c in range(16)):
        raise ValueError(
            "synthetic clip generation failed; retain output and do not train"
        )
    save_json(
        root / "protocol.json",
        dict(
            objective="Conditional scene raw-fisheye physics/PPO tracking, all real448",
            real_frames="0..447",
            initial_fit=str(real.initial_fit),
            pixel_source="V20260908_people0_1_2_pmpose_c3_chain/people_1/c3_predictions/{left,right}/pmpose/raw_predictions.json",
            scene_source="G20261013_reliability448_view_v1/view/scene/scene_transforms.npz",
            scene_semantics="frozen Stage conditional trajectory, not independent world truth; nominal release delay .5s",
            training="synthetic PD teacher rollouts, clips0..11; held-out complete clips12..15; real448 never rewards training",
            synthetic_clip_counts=counts,
            observation="past<=.5s summary,current pixels,state; no future poses/actions",
            physical_model="22 ball-jointed links, fixed frame0 shape and frozen hands, 63.4kg uncalibrated capsule surrogate; no self collisions",
            observation_map="exact selected SMPL-H surface vertices with pose blends and all52 LBS transforms",
            initial_mapping_max_m=body.initial_mapping_max_m,
            action="63 residual PD-angle targets relative to frame0, .4rad component scale, torques120Nm",
            support="initial model wrists define assumed camera-rigid grips, K900N/m,D45Ns/m, force<=300N each; all replay frames; independent support validation unavailable",
            heel="training/teacher use explicit historical rigid sites; final real replay uses actual SMPL-H heel3387/6787 surface map with position/velocity projection, tolerance1mm; old rigid runs retained. This constraint-only correction uses frozen PPO weights, not a retrained surface-contact policy.",
            controls="same simulated body/actions: zero PD hold, one-step pixel Jacobian control, PPO; each with noheel/hardheel",
            success="complete448 records incl failures; exact initial map, actual dt, no clipping of published derivatives, constrained heel numerical gate",
            accuracy_gate="held-out synthetic and complete real diagnostics only; no independent real manual/3D labels, so real accuracy promotion unavailable",
            stop="numerical/input/map failure; bounded PPO budget; real invalid frames retained, no silent reset/mixing",
        ),
    )
    return real, dataset, synthetic


def load_prepared(root):
    real = load_real()
    with np.load(root / "synthetic.npz", allow_pickle=False) as z:
        synthetic = {
            k: z[k].copy()
            for k in (
                "qpos",
                "qvel",
                "points",
                "wrist_target",
                "clip_id",
                "exact_pixels",
            )
        }
        dataset = replace(
            real,
            pixels=z["pixels"].copy(),
            valid=z["valid"].copy(),
            time=z["time"].copy(),
            rotation=z["rotation"].copy(),
            translation=z["translation"].copy(),
            stage=z["stage"].astype(str),
        )
    synthetic["train_clips"] = np.arange(12)
    synthetic["test_clips"] = np.arange(12, 16)
    return real, dataset, synthetic


def train(root, dataset, synthetic, seed, steps, wall):
    target = root / f"seed{seed}"
    target.mkdir(exist_ok=False)
    env = PixelTrackingEnv(
        dataset,
        new_body(dataset, heel_mode="rigid"),
        synthetic=synthetic,
        training=True,
        horizon=60,
    )
    torch.set_num_threads(1)
    policy = PPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=128,
        n_epochs=5,
        gamma=np.exp(-(1 / 30) / 2),
        gae_lambda=0.95,
        learning_rate=3e-4,
        ent_coef=0.001,
        clip_range=0.2,
        seed=seed,
        device="cpu",
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-1.5
        ),
        verbose=0,
    )
    save_json(
        target / "training_arguments.json",
        dict(
            seed=seed,
            steps=steps,
            wall_limit_s=wall,
            simulator="MuJoCo3.16 CPU",
            policy_device="cpu",
            observation_dim=370,
            action_dim=63,
            n_steps=512,
            batch_size=128,
            n_epochs=5,
            training_clips=synthetic["train_clips"],
            heldout_clips=synthetic["test_clips"],
            versions=dict(torch=torch.__version__),
        ),
    )
    start = time.perf_counter()
    policy.learn(
        total_timesteps=steps, callback=Progress(target / "progress.json", wall)
    )
    policy.save(target / "policy")
    save_json(
        target / "training_result.json",
        dict(actual_steps=policy.num_timesteps, wall_s=time.perf_counter() - start),
    )
    return policy


def evaluate_real(root, dataset, policy, name, hard=False, support=True):
    target = root / name
    target.mkdir(exist_ok=False)
    body = new_body(dataset)
    env = PixelTrackingEnv(dataset, body, hard_heel=hard, support=support)
    observation, _ = env.reset()
    positions = [body.points()]
    states = ["initialized"]
    segments = [0]
    records = []
    velocity = [np.zeros((12, 3))]
    acceleration = [np.zeros((12, 3))]
    cameras = [(positions[0] - dataset.translation[0]) @ dataset.rotation[0]]
    qpos = [body.data.qpos.copy()]
    heels = [body.heel_positions()]
    all_residual = [
        np.linalg.norm(dataset.project(positions[0], 0) - dataset.pixels[0], axis=-1)
    ]
    valid = [dataset.valid[0]]
    step_ms = [0.0]
    reasons = Counter()
    failed = False
    records.append(
        dict(
            frame_index=0,
            status="initialized",
            reason="frame0_only_fit",
            position_world_m=positions[0],
            timestamp_s=dataset.time[0],
            world_pose_certified=False,
        )
    )
    for i in range(1, len(dataset.time)):
        start = time.perf_counter()
        if failed:
            # Never erase a fall by restarting from a future fitted body state.
            info = dict(
                frame_index=i,
                ok=False,
                reason="unavailable_after_failure",
                valid=np.zeros((2, 12), bool),
                reprojection_px=np.full((2, 12), np.nan),
                hard_heel_active=bool(hard and "stage2" in dataset.stage[i]),
            )
            state = "unavailable"
            p = v = a = np.full((12, 3), np.nan)
        else:
            action = (
                pixel_controller(env)
                if policy == "classic"
                else np.zeros(63)
                if policy == "hold"
                else policy.predict(observation, deterministic=True)[0]
            )
            observation, _, terminated, _, info = env.step(action)
            p, v, a = body.point_derivatives()
            state = "tracked" if info["ok"] else "failed"
            failed = bool(terminated)
        elapsed = (time.perf_counter() - start) * 1000
        reasons[info["reason"]] += 1
        positions.append(p)
        velocity.append(v)
        acceleration.append(a)
        states.append(state)
        segments.append(0)
        cameras.append((p - dataset.translation[i]) @ dataset.rotation[i])
        qpos.append(
            body.data.qpos.copy() if not failed else np.full(body.model.nq, np.nan)
        )
        heels.append(
            body.heel_positions() if state == "tracked" else np.full((2, 3), np.nan)
        )
        all_residual.append(info["reprojection_px"])
        valid.append(info["valid"])
        step_ms.append(elapsed)
        records.append(
            dict(
                **info,
                status=state,
                timestamp_s=dataset.time[i],
                capture_pair_id=int(dataset.capture_ids[i]),
                scene_status=dataset.scene_status[i],
                stage=dataset.stage[i],
                position_world_m=p,
                velocity_world_m_s=v,
                acceleration_world_m_s2=a,
                position_current_camera_m=cameras[-1],
                step_ms=elapsed,
            )
        )
    p = np.asarray(positions)
    v = np.asarray(velocity)
    a = np.asarray(acceleration)
    available = np.asarray(states) == "tracked"
    available[0] = True
    numerical_v, numerical_a = three_point_derivatives(
        dataset.time, p, np.repeat(available[:, None], 12, axis=1)
    )
    camera_v, camera_a = three_point_derivatives(
        dataset.time, np.asarray(cameras), np.repeat(available[:, None], 12, axis=1)
    )
    reprojection = np.asarray(all_residual)
    mask = np.asarray(valid) & available[:, None, None]
    summary = dict(
        name=name,
        frames=len(p),
        complete_records=len(records),
        status_counts=dict(Counter(states)),
        tracked_frames=int(np.sum(np.asarray(states) == "tracked")),
        reason_counts=dict(reasons),
        detector_reprojection_px=stats(reprojection[mask]),
        per_joint_reprojection_px=[
            stats(reprojection[:, :, j][mask[:, :, j]]) for j in range(12)
        ],
        velocity_world_m_s=stats(np.linalg.norm(v[available], axis=-1)),
        acceleration_world_m_s2=stats(np.linalg.norm(a[available], axis=-1)),
        numerical_acceleration_world_m_s2=stats(np.linalg.norm(numerical_a, axis=-1)),
        step_ms=stats(np.asarray(step_ms)[np.asarray(states) == "tracked"]),
        hard_heel_active_frames=sum(bool(r.get("hard_heel_active")) for r in records),
        hard_heel_evaluated_frames=sum(
            bool(r.get("hard_heel_active")) and r["status"] == "tracked"
            for r in records
        ),
        heel_error_m=stats(
            [
                r.get("heel_error_m", np.nan)
                for r in records
                if r.get("hard_heel_active") and r["status"] == "tracked"
            ]
        ),
        constraint_impulse_ns=stats(
            [r.get("projection_impulse_norm_ns", np.nan) for r in records]
        ),
        support_force_n=stats([r.get("wrist_force_max_n", np.nan) for r in records]),
        first_failed_frame=next(
            (i for i, s in enumerate(states) if s == "failed"), None
        ),
        interpretation="synthetic-trained conditional-scene experiment; no real held-out manual/3D truth; all failures retained",
    )
    with (target / "trajectory.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(serial(record), ensure_ascii=False, allow_nan=False) + "\n"
            )
    np.savez_compressed(
        target / "trajectory.npz",
        time=dataset.time,
        p_world=p,
        v_world=v,
        a_world=a,
        finite_difference_v=numerical_v,
        finite_difference_a=numerical_a,
        p_camera=cameras,
        v_camera=camera_v,
        a_camera=camera_a,
        status=states,
        residual_px=reprojection,
        valid=mask,
        qpos=qpos,
        heels=heels,
    )
    save_json(target / "summary.json", summary)
    print(json.dumps(serial(summary), ensure_ascii=False), flush=True)
    return summary


def evaluate_synthetic(root, dataset, synthetic, policy, name):
    errors = []
    alive = []
    residual = []
    for clip in synthetic["test_clips"]:
        ids = np.flatnonzero(synthetic["clip_id"] == clip)
        start, end = int(ids[0]), int(ids[-1])
        body = new_body(dataset, heel_mode="rigid")
        env = PixelTrackingEnv(
            dataset, body, synthetic=synthetic, hard_heel=bool(clip % 2 == 0)
        )
        body.reset(synthetic["qpos"][start], synthetic["qvel"][start])
        env.index = start + 1
        env.last_action[:] = 0
        env.age = 0.0
        failed = False
        for i in range(start + 1, end + 1):
            if failed:
                alive.append(False)
                continue
            obs = env.observation()
            action = (
                pixel_controller(env)
                if policy == "classic"
                else np.zeros(63)
                if policy == "hold"
                else policy.predict(obs, deterministic=True)[0]
            )
            _, _, terminated, _, info = env.step(action)
            alive.append(not terminated)
            if not terminated:
                errors.extend(
                    np.linalg.norm(body.points() - synthetic["points"][i], axis=-1)
                )
                residual.extend(info["reprojection_px"][info["valid"]])
            failed = bool(terminated)
    result = dict(
        name=name,
        heldout_clips=synthetic["test_clips"],
        frames=len(alive),
        available_fraction=float(np.mean(alive)),
        position_error_m_conditional_on_available=stats(errors),
        pixel_error_conditional_on_available=stats(residual),
    )
    save_json(root / f"synthetic_{name}.json", result)
    return result


def plot(root, names):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex=True, layout="constrained")
    groups = ((0, 2, 4), (1, 3, 5), (6, 8, 10), (7, 9, 11))
    for n, name in enumerate(names):
        if name == "pva_existing":
            from .data import ROOT

            path = (
                ROOT
                / "research_records/engineering_validation/G20261014_stereo_pva_baseline448_v2/tracking_final/trajectory.jsonl"
            )
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            times = np.asarray([r["timestamp_s"] for r in records])
            positions = np.asarray([r["position_mm"] for r in records], float) / 1000
            segments = np.asarray([r["segment_id"] for r in records])
            valid = np.isfinite(positions).all(axis=-1)
            breaks = np.diff(segments, axis=0) != 0
            valid[1:] &= ~breaks
            valid[:-1] &= ~breaks
            velocity, acceleration = three_point_derivatives(times, positions, valid)
            z = {"time": times, "v_camera": velocity, "a_camera": acceleration}
        else:
            with np.load(root / name / "trajectory.npz", allow_pickle=False) as saved:
                z = {k: saved[k].copy() for k in ("time", "v_camera", "a_camera")}
        if z:
            for row, joints in enumerate(groups):
                for column, key in enumerate(("v_camera", "a_camera")):
                    norm = np.linalg.norm(z[key][:, joints], axis=-1).mean(axis=1)
                    axes[row, column].plot(
                        z["time"],
                        norm,
                        color="black",
                        linestyle=("-", "--", ":", "-.", (0, (5, 1, 1, 1)))[n % 5],
                        label=name,
                        lw=0.8,
                    )
    for row in range(4):
        limb = ("Left arm", "Right arm", "Left leg", "Right leg")[row]
        axes[row, 0].set_title(limb + ": mean of three endpoint speed norms")
        axes[row, 1].set_title(limb + ": mean of three endpoint acceleration norms")
        axes[row, 0].set_ylabel("Camera-coordinate speed (m/s)")
        axes[row, 1].set_ylabel("Camera-coordinate acceleration (m/s2)")
        for ax in axes[row]:
            ax.grid(color=".85")
            ax.legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Host-return midpoint time (s)")
    fig.suptitle(
        "448 conditional-scene replay; all curves use position finite differences, not latent EKF states; gaps are unavailable"
    )
    fig.savefig(root / "camera_motion_comparison.png", dpi=150, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=["prepare", "train", "evaluate", "plot", "analyze", "annotations"],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100352)
    parser.add_argument("--wall", type=float, default=2400)
    parser.add_argument(
        "--distal-joints",
        action="store_true",
        help="Only after user confirms hand=wrist and foot=ankle",
    )
    args = parser.parse_args()
    root = args.output
    torch.set_num_threads(1)
    if args.mode == "prepare":
        root.mkdir(parents=True, exist_ok=False)
        prepare(root)
        return
    if args.mode == "analyze":
        from .analyze import analyze

        analyze(root)
        return
    if args.mode == "annotations":
        from .annotations import evaluate_labels

        evaluate_labels(root, args.distal_joints)
        return
    real, dataset, synthetic = load_prepared(root)
    if args.mode == "train":
        train(root, dataset, synthetic, args.seed, args.steps, args.wall)
    elif args.mode == "evaluate":
        summaries = []
        synthetic_summaries = []
        for label, controller in [("hold", "hold"), ("classic", "classic")]:
            for hard in (False, True):
                name = f"{label}_{'heel' if hard else 'free'}"
                if not (root / name).exists():
                    summaries.append(evaluate_real(root, real, controller, name, hard))
            if not (root / f"synthetic_{label}.json").exists():
                synthetic_summaries.append(
                    evaluate_synthetic(root, dataset, synthetic, controller, label)
                )
        for seed in (0, 1, 2):
            path = root / f"seed{seed}/policy.zip"
            if not path.exists():
                continue
            policy = PPO.load(path, device="cpu")
            for hard in (False, True):
                name = f"ppo{seed}_{'heel' if hard else 'free'}"
                if not (root / name).exists():
                    summaries.append(evaluate_real(root, real, policy, name, hard))
            if not (root / f"synthetic_ppo{seed}.json").exists():
                synthetic_summaries.append(
                    evaluate_synthetic(root, dataset, synthetic, policy, f"ppo{seed}")
                )
        save_json(
            root / "comparison.json",
            dict(
                real=[
                    json.loads(p.read_text(encoding="utf-8"))
                    for p in root.glob("*/summary.json")
                ],
                synthetic=[
                    json.loads(p.read_text(encoding="utf-8"))
                    for p in root.glob("synthetic_*.json")
                ],
            ),
        )
    else:
        plot(
            root,
            [
                name
                for name in (
                    "classic_free",
                    "ppo0_free",
                    "ppo0_heel",
                    "ppo1_heel",
                    "pva_existing",
                )
                if (root / name).exists() or name == "pva_existing"
            ],
        )


if __name__ == "__main__":
    main()
