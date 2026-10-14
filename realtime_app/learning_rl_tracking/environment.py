"""Causal simulator-state estimator with current raw pixel feedback."""

from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .body import PhysicsBody
from .data import TrackingData


class PixelTrackingEnv(gym.Env):
    def __init__(
        self,
        dataset: TrackingData,
        body: PhysicsBody,
        *,
        hard_heel=False,
        synthetic=None,
        training=False,
        support=True,
        horizon=90,
    ):
        self.dataset, self.body = dataset, body
        self.hard_heel, self.support, self.training = hard_heel, support, training
        self.synthetic = synthetic
        self.horizon = horizon
        self.action_space = gym.spaces.Box(-1.0, 1.0, (63,), np.float32)
        self.observation_space = gym.spaces.Box(-20.0, 20.0, (370,), np.float32)
        self.last_action = np.zeros(63)
        self.index = 1
        self.age = 0.0
        self.steps = 0
        self.start = 0
        self.last_info = {}
        # In the conditional real replay, these are assumed camera-rigid grips.
        self.wrist_camera = (
            body.wrist_positions() - dataset.translation[0]
        ) @ dataset.rotation[0]

    def _wrist_target(self, index):
        if not self.support:
            return None
        if self.synthetic is not None:
            return self.synthetic["wrist_target"][index]
        return (
            self.wrist_camera @ self.dataset.rotation[index].T
            + self.dataset.translation[index]
        )

    def observation(self):
        d, i, b = self.dataset, self.index, self.body
        points = b.points()
        predicted = d.project(points, i)
        mask = d.valid[i] & np.isfinite(predicted).all(axis=-1)
        residual = np.where(mask[..., None], (d.pixels[i] - predicted) / 100, 0.0)
        pixel = np.where(
            d.valid[i, ..., None], d.pixels[i] / np.array([1920.0, 1080.0]) - 0.5, 0.0
        )
        first = np.searchsorted(d.time, d.time[i] - 0.5)
        if self.synthetic is not None:
            same_clip = np.flatnonzero(
                self.synthetic["clip_id"] == self.synthetic["clip_id"][i]
            )
            first = max(first, int(same_clip[0]))
        past = d.pixels[first : i + 1] / np.array([1920.0, 1080.0]) - 0.5
        valid_past = d.valid[first : i + 1]
        history = (
            np.sum(np.where(valid_past[..., None], past, 0), axis=0)
            / np.maximum(1, valid_past.sum(axis=0))[..., None]
        )
        joint_angles = Rotation.from_matrix(b.rotations()).as_rotvec().ravel()
        root_r = b.data.xmat[b.body_ids[0]].reshape(3, 3)
        gravity = root_r.T @ np.array([0, 0, -1.0])
        dt = d.time[i] - d.time[i - 1]
        anchored = self.hard_heel and "stage2" in d.stage[i]
        observation = np.r_[
            joint_angles,
            b.data.qvel / 5,
            gravity,
            b.data.qpos[2],
            residual.ravel(),
            pixel.ravel(),
            mask.ravel().astype(float),
            history.ravel(),
            self.last_action,
            dt,
            float(anchored),
            self.age,
        ]
        if observation.shape != self.observation_space.shape:
            raise ValueError(
                f"observation shape {observation.shape} != {self.observation_space.shape}"
            )
        return np.clip(observation, -20, 20).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.training:
            if self.synthetic is None:
                raise ValueError(
                    "training resets require explicit synthetic reference states"
                )
            allowed = self.synthetic.get(
                "train_clips", np.unique(self.synthetic["clip_id"])
            )
            clip = int(self.np_random.choice(allowed))
            indices = np.flatnonzero(self.synthetic["clip_id"] == clip)
            index = int(self.np_random.choice(indices[:-2]))
            self.body.reset(
                self.synthetic["qpos"][index], self.synthetic["qvel"][index]
            )
            self.hard_heel = bool(self.np_random.integers(2))
        else:
            index = 0
            self.body.reset()
        self.start, self.index, self.steps = index, index + 1, 0
        self.last_action[:] = 0
        self.age = 0.0
        self.last_info = {}
        return self.observation(), {}

    def step(self, action):
        action = np.asarray(action, float)
        if action.shape != (63,) or not np.isfinite(action).all():
            raise ValueError("finite 63-dimensional action required")
        action = np.clip(action, -1, 1)
        d, b, i = self.dataset, self.body, self.index
        dt = float(d.time[i] - d.time[i - 1])
        anchored = self.hard_heel and "stage2" in d.stage[i]
        b.enable_heel(anchored)
        # A fixed first-frame PD hold is the shared base, not a future pose target.
        target = (
            b.initial_rotations
            @ Rotation.from_rotvec(action.reshape(21, 3) * 0.4).as_matrix()
        )
        ok, energy, reason = b.advance(target, dt, self._wrist_target(i))
        points = b.points()
        projection = d.project(points, i)
        valid = d.valid[i] & np.isfinite(projection).all(axis=-1)
        residual = np.linalg.norm(projection - d.pixels[i], axis=-1)
        count = int(valid.sum())
        self.age = 0.0 if count else self.age + dt
        if count:
            scaled = residual[valid] / 100
            huber = np.where(scaled < 1, 0.5 * scaled**2, scaled - 0.5).mean()
            tracking_reward = np.exp(-huber)
        else:
            huber, tracking_reward = np.nan, 0.0
        reward = float(
            (dt / (1 / 30))
            * (
                tracking_reward
                - 0.00002 * energy / max(dt, 1e-8)
                - 0.01 * np.mean((action - self.last_action) ** 2)
            )
        )
        if (
            np.linalg.norm(b.data.qvel) > 150
            or b.data.qpos[2] < 0.3
            or b.data.qpos[2] > 2
        ):
            ok, reason = False, "unstable_or_fallen"
        if self.age > 0.1:
            ok, reason = False, "observation_gap"
        self.last_info = {
            "frame_index": i,
            "dt_s": dt,
            "ok": bool(ok),
            "reason": reason,
            "reprojection_px": residual,
            "valid": valid,
            "energy_j": float(energy),
            "reward": reward,
            "observed_slots": count,
            "hard_heel_active": bool(anchored),
            "heel_error_m": b.max_heel_error,
            "projection_delta": b.max_projection_delta,
            "projection_impulse_norm_ns": b.max_projection_impulse,
            "wrist_force_max_n": b.max_wrist_force,
            "qvel_norm": float(np.linalg.norm(b.data.qvel)),
            "world_pose_certified": False,
            "support_assumption": bool(self.support),
        }
        self.last_action = action.copy()
        self.steps += 1
        self.index += 1
        truncated = self.index >= len(d.time) or (
            self.training and self.steps >= self.horizon
        )
        if self.synthetic is not None and self.index < len(d.time):
            truncated |= (
                self.synthetic["clip_id"][self.index] != self.synthetic["clip_id"][i]
            )
        observation = (
            self.observation()
            if not truncated
            else np.zeros(self.observation_space.shape, np.float32)
        )
        if not ok:
            reward -= 1.0
        return observation, reward, bool(not ok), bool(truncated), self.last_info


def make_synthetic(
    real: TrackingData, body: PhysicsBody, output: Path, clips=16, frames=120
):
    """Generate reachable teacher rollouts; splits use complete random-seed clips.

    Teacher PD schedules are only used to create synthetic trajectories. PPO
    sees their noisy 2D projections, never future target angles or teacher actions.
    """
    rng = np.random.default_rng(713)
    qpos, qvel, points, wrists, stages, clip_ids = [], [], [], [], [], []
    for clip in range(clips):
        body.reset()
        grip = body.wrist_positions()
        axes = rng.normal(size=(21, 3))
        axes[[2, 5, 8, 9, 10, 11, 14]] *= 0.2
        amplitude = rng.uniform(0.02, 0.12, (21, 1))
        phase = rng.uniform(0, 2 * np.pi, (21, 1))
        for frame in range(frames):
            t = frame / 30
            # Known world trajectory; cameras vary independently of the character.
            wrist_target = grip + np.array(
                [0.025 * np.sin(0.8 * t + clip), 0.025 * np.sin(0.6 * t), 0]
            )
            body.enable_heel(clip % 2 == 0)
            if frame:
                target = (
                    body.initial_rotations
                    @ Rotation.from_rotvec(
                        axes * amplitude * np.sin(1.2 * t + phase)
                    ).as_matrix()
                )
                ok, _, _ = body.advance(target, 1 / 30, wrist_target)
                if not ok or body.data.qpos[2] < 0.35:
                    # A new complete clip boundary is explicit; no interpolation.
                    break
            qpos.append(body.data.qpos.copy())
            qvel.append(body.data.qvel.copy())
            points.append(body.points())
            wrists.append(wrist_target)
            stages.append(
                "stage2_feet_static_walker_moving"
                if clip % 2 == 0
                else "stage1_walker_static_human_moving"
            )
            clip_ids.append(clip)
    count = len(points)
    time = np.arange(count) / 30
    rotation = np.repeat(real.rotation[:1], count, axis=0)
    translation = np.repeat(real.translation[:1], count, axis=0)
    translation[:, 0] += 0.05 * np.sin(time * 0.6)
    dataset = replace(
        real,
        pixels=np.zeros((count, 2, 12, 2)),
        valid=np.ones((count, 2, 12), bool),
        time=time,
        rotation=rotation,
        translation=translation,
        stage=np.asarray(stages),
        scene_status=np.full(count, "known_synthetic"),
        capture_ids=np.arange(count),
        skew_ms=np.zeros(count),
    )
    exact = np.asarray([dataset.project(p, i) for i, p in enumerate(points)])
    dataset.pixels = exact + rng.normal(0, 4, exact.shape)
    dataset.valid = np.isfinite(exact).all(axis=-1) & (
        rng.random((count, 2, 12)) > 0.05
    )
    synthetic = dict(
        qpos=np.asarray(qpos),
        qvel=np.asarray(qvel),
        points=np.asarray(points),
        wrist_target=np.asarray(wrists),
        clip_id=np.asarray(clip_ids),
        exact_pixels=exact,
    )
    np.savez_compressed(
        output,
        **synthetic,
        pixels=dataset.pixels,
        valid=dataset.valid,
        time=time,
        rotation=rotation,
        translation=translation,
        stage=dataset.stage,
    )
    body.reset()
    return dataset, synthetic


def pixel_controller(env: PixelTrackingEnv):
    """One-step damped visual Jacobian controller with exactly the same PD limits."""
    b, d, i = env.body, env.dataset, env.index
    qpos = b.data.qpos.copy()
    points = b.points()
    projection = d.project(points, i)
    valid = d.valid[i] & np.isfinite(projection).all(axis=-1)
    if valid.sum() < 4:
        return np.zeros(63)
    error = (d.pixels[i] - projection)[valid].ravel()
    jac = np.empty((len(error), 63))
    eps = 0.002
    for column, dof in enumerate(b.dof_indices.ravel()):
        delta = np.zeros(b.model.nv)
        delta[dof] = eps
        b.data.qpos[:] = qpos
        mujoco.mj_integratePos(b.model, b.data.qpos, delta, 1.0)
        mujoco.mj_forward(b.model, b.data)
        projected = d.project(b.points(), i)
        jac[:, column] = ((projected - projection)[valid] / eps).ravel()
    b.data.qpos[:] = qpos
    mujoco.mj_forward(b.model, b.data)
    if not np.isfinite(jac).all():
        return np.zeros(63)
    delta = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(len(error)) * 400, error)
    delta = np.clip(delta, -0.15, 0.15).reshape(21, 3)
    target = b.rotations() @ Rotation.from_rotvec(delta).as_matrix()
    action = (
        Rotation.from_matrix(b.initial_rotations.transpose(0, 2, 1) @ target)
        .as_rotvec()
        .ravel()
        / 0.4
    )
    return np.clip(action, -1, 1)
