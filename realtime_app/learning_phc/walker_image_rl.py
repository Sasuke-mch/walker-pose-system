"""Bounded PPO adaptation and causal full448 replay with a free walker.

Old image-PPO weights initialize the network only. No world wrist support,
teacher action, future reference pose, or per-frame root overwrite is used.
"""
from isaacgym import gymapi  # noqa: F401 -- must precede torch

import argparse
from bisect import bisect_right
import json
from pathlib import Path
import time

import numpy as np
import torch

from .humanoid_walker import HumanoidWalkerScene
from .image_policy import ImageActorCritic, ImageObservation
from .walker_scene import H, clean


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


class Environment:
    """60 Hz decisions, 240 Hz paired grips, 480 Hz PhysX integration."""

    def __init__(self, foundation, packet):
        self.foundation, self.packet = foundation, packet
        # Recorded dynamic camera poses are initialization-only: every policy
        # projection uses the camera rigidly mounted on the CURRENT free walker.
        live = dict(packet)
        live['camera_rotation'] = np.empty_like(packet['camera_rotation'])
        live['camera_translation_m'] = np.empty_like(packet['camera_translation_m'])
        self.observation = ImageObservation(live)
        self.scene = None
        self.resets, self.simulations = 0, 0

    def reset(self):
        if self.scene is not None:
            self.simulations += self.scene.explicit_simulate_calls
            self.scene.close()
        d = self.packet
        stereo = np.eye(4)
        stereo[:3, :3], stereo[:3, 3] = d['R01'], d['T01_m']
        self.scene = HumanoidWalkerScene(self.foundation/'synthetic_fixture_asset_v1',
            self.foundation/'shape_shoe25_diagonal_asset_v1/humanoid.xml',
            d['initial_position_m'], d['initial_rotation'], d['camera_rotation'][0],
            d['camera_translation_m'][0], 400., 4., stereo_transform=stereo,
            project_initial_contact=True, grip_mode='constraint_predict', physics_substeps=2)
        self.tick, self.episode_steps = 0, 0
        self.resets += 1
        limits = self.scene.dof_properties
        self.scale = np.repeat([min(1.2*max(abs(limits['lower'][j:j+3]).max(),
            abs(limits['upper'][j:j+3]).max()), np.pi) or np.pi for j in range(0, 69, 3)], 3)
        # Exactly the original PHC action convention used by the loaded network.
        self.scale[4] = self.scale[16] = 5.
        return self.features(0)

    def index(self):
        return bisect_right(self.packet['timestamp_s'], self.tick*H)-1

    def features(self, index):
        camera = self.scene.camera_mount()
        self.observation.data['camera_rotation'][index] = camera[:3, :3]
        self.observation.data['camera_translation_m'][index] = camera[:3, 3]
        return self.observation.features(index, *self.scene.human_state())

    def advance(self, action, ticks=4):
        if np.shape(action) != (69,) or not np.isfinite(action).all():
            raise ValueError('finite69D policy action required')
        s = self.scene
        s.joint_target = (action*self.scale).astype(np.float32)
        s.gym.set_actor_dof_position_targets(s.env, s.human, s.joint_target)
        force = []
        for _ in range(ticks):
            connections = s.advance_coupled(active=True)
            force.append([g['force_a'] for g in connections])
            self.tick += 1
        self.episode_steps += 1
        return np.asarray(force)

    def quality(self):
        p, r, v, w = self.scene.human_state()
        walker = self.scene.read(False)
        finite = all(np.isfinite(a).all() for a in (p, r, v, w, walker['p'], walker['q'], walker['v'], walker['w']))
        # A declared operational fall proxy, not an independent fall annotation.
        fallen = bool(p[0, 2] < .45 or r[0, 2, 2] < .3)
        return finite, fallen

    def step(self, action):
        forces = self.advance(action)
        index = self.index()
        features = self.features(index)
        p, r, _, _ = self.scene.human_state()
        base, image, _ = self.observation.reward(index, p, r, action)
        finite, fallen = self.quality()
        reward = base+.2*float(np.clip(r[0, 2, 2], 0, 1))-float(fallen)
        done = not finite or fallen or self.episode_steps >= 128 or index >= 255
        if not finite or not np.isfinite(features).all() or not np.isfinite(forces).all():
            raise RuntimeError('nonfinite training state; preserve run and stop')
        return features, reward, done, dict(index=index, image=image, fallen=fallen,
            force_peak_N=float(np.linalg.norm(forces, axis=2).max()))

    def close(self):
        if self.scene is not None:
            self.simulations += self.scene.explicit_simulate_calls
            self.scene.close()
            self.scene = None


def train(foundation, packet, initial, output, updates):
    torch.manual_seed(0)
    model = ImageActorCritic()
    model.load_state_dict(torch.load(initial, map_location='cpu')['model'])
    before = {k: v.clone() for k, v in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    env = Environment(foundation, packet)
    observation = env.reset()
    visited, history = set(), []
    parameters = dict(seed=0, updates=updates, horizon=64, epochs=4, learning_rate=3e-5,
        gamma=.97, gae_lambda=.95, clip=.2, decision_hz=60, grip_hz=240, physics_hz=480,
        action='absolute69D original PHC scale; not reference residual',
        initial_checkpoint=str(initial.resolve()), teacher_runtime=False,
        image_camera='current simulated walker-mounted camera; recorded trajectory never used after reset',
        training_reset='first frame, zero initial velocities; fall or128 decisions; no random pose synthesis',
        training_observations='only arrived frames0..255; actual visited range saved',
        reward='existing confidence-weighted 2D reward +0.2 upright -1 operational fall',
        assumptions='two bilateral point grips stay attached; synthetic dynamics; not calibrated real grasp')
    (output/'parameters.json').write_text(json.dumps(parameters, indent=2), encoding='utf-8')
    try:
        with (output/'training.jsonl').open('w', encoding='utf-8') as log:
            for update in range(updates):
                data = []
                for _ in range(64):
                    x = torch.tensor(observation)
                    with torch.no_grad():
                        normal, value = model.distribution(x)
                        raw = normal.sample()
                        lp = model.log_probability(normal, raw)
                    nxt, reward, done, detail = env.step(raw.tanh().numpy())
                    visited.add(detail['index'])
                    data.append((x, raw, lp, value, reward, done, detail))
                    observation = env.reset() if done else nxt
                with torch.no_grad():
                    _, bootstrap = model.distribution(torch.tensor(observation))
                advantage, gae = [], torch.tensor(0.)
                for _, _, _, value, reward, done, _ in reversed(data):
                    mask = float(not done)
                    delta = reward+.97*bootstrap*mask-value
                    gae = delta+.97*.95*mask*gae
                    advantage.append(gae)
                    bootstrap = value
                advantages = torch.stack(advantage[::-1])
                obs = torch.stack([d[0] for d in data]); raw = torch.stack([d[1] for d in data])
                old_lp = torch.stack([d[2] for d in data]); values = torch.stack([d[3] for d in data])
                returns = advantages+values
                advantages = (advantages-advantages.mean())/(advantages.std()+1e-6)
                for _ in range(4):
                    normal, predicted = model.distribution(obs)
                    new_lp = model.log_probability(normal, raw)
                    ratio = (new_lp-old_lp).exp()
                    surrogate = torch.minimum(ratio*advantages, ratio.clamp(.8, 1.2)*advantages)
                    loss = -surrogate.mean()+.5*(predicted-returns).square().mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite PPO objective')
                    optimizer.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    if float((old_lp-new_lp).mean()) > .04:
                        break
                row = dict(update=update+1, decisions=(update+1)*64,
                    reward_mean=float(np.mean([d[4] for d in data])),
                    image_reward_mean=float(np.mean([d[6]['image'] for d in data])),
                    terminated_episodes=sum(d[5] for d in data), resets=env.resets,
                    visited_min=min(visited), visited_max=max(visited), loss=float(loss.detach()))
                log.write(json.dumps(row)+'\n'); log.flush()
                print(json.dumps(row), flush=True)
                history.append(row)
        delta = float(torch.sqrt(sum((v.detach()-before[k]).square().sum() for k, v in model.named_parameters())))
        actor_delta = float(torch.sqrt(sum((v.detach()-before[k]).square().sum()
            for k, v in model.named_parameters() if k.startswith('actor.'))))
        if not delta > 0 or not actor_delta > 0:
            raise RuntimeError('PPO did not update actor parameters')
        torch.save(dict(model=model.state_dict(), phase='free_walker_image_ppo', parameters=parameters), output/'ppo.pt')
        report = dict(status='bounded_ppo_training_completed', updates=updates, decisions=updates*64,
            parameter_delta_l2=delta, actor_parameter_delta_l2=actor_delta,
            visited_frames=sorted(visited), resets=env.resets, last_update=history[-1])
    finally:
        env.close()
    report['simulate_calls'] = env.simulations
    (output/'training_result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


def replay(foundation, packet, checkpoint, output, reference=None):
    output.mkdir(exist_ok=False)
    if reference is None:
        model = ImageActorCritic()
        saved = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(saved['model']); model.eval()
    else:
        from .policy import FrozenPHC
        model = FrozenPHC(checkpoint, checkpoint.with_name('phc_comp_kp_2.pth'))
        saved = dict(phase='official_frozen_PHC_3D_reference_diagnostic')
    env = Environment(foundation, packet)
    env.reset()
    result = dict(status='in_progress', frames=448, checkpoint=str(checkpoint.resolve()),
        phase=saved['phase'], resets_after_frame0=0, teacher_runtime=False, future_3D_runtime=False,
        input='latest arrived2D, simulated self-state and current simulated camera; first pose only',
        dynamic_recorded_camera_used=False, initialization=env.scene.initialization_contract,
        assumptions='both point grips always active; mass/inertia/friction synthetic; no real release labels',
        scope='full-sequence pipeline feasibility, no independent3D accuracy or heldout generalization claim')
    if reference is not None:
        result.update(input='arrived privileged3D reference position/velocity and simulated self-state',
            controller='official PHC, no image-student network or PPO update',
            teacher_runtime=True, future_3D_runtime=None, reference_derivative_causality_certified=False,
            scope='controller substitution diagnostic, not deployable2D inference; offline reference velocity may use future frames',
            reference_source=str((foundation.parent/'corrections_v1/reference_matched.npz').resolve()),
            composer=str(checkpoint.with_name('phc_comp_kp_2.pth').resolve()))
    env_initial = env.scene.motion_state()
    np.savez_compressed(output/'initial_state.npz', **env_initial)
    records, actions, forces, indices, decision_times, pixel_errors = [], [], [], [], [], []
    next_frame, last_action, force_acc = 0, np.zeros(69), []
    start = time.perf_counter()
    try:
        while next_frame < 448:
            now = env.tick*H
            # A frame is emitted at the first physical tick at/after its arrival.
            # Current2D changes the NEXT interval, never the state before arrival.
            while next_frame < 448 and packet['timestamp_s'][next_frame] <= now+1e-12:
                i = next_frame
                env.features(i)  # sets current camera for evaluation only
                s = env.scene; p, r, v, w = s.human_state(); walk = s.read(False)
                gap = [np.linalg.norm(p[g['index']]+r[g['index']]@g['local_wrist']-
                    walk['p']-walk['R']@g['local_handle']) for g in s.grips]
                finite, fallen = env.quality()
                projected = env.observation.project(env.observation.points(p, r), i)
                error = np.linalg.norm(projected-packet['pixels'][i, :, :, :2], axis=-1)
                pixel_errors.append(error)
                records.append(dict(time=now, human=np.concatenate([p[..., None], r], axis=2),
                    walker=np.r_[walk['p'], walk['q'], walk['v'], walk['w']],
                    camera=s.camera_mount(), right=s.camera_mount('right'), gap=gap,
                    finite=finite, fallen=fallen, velocity=v, angular=w, action=last_action.copy(),
                    force_peak=np.max(np.linalg.norm(force_acc, axis=2), axis=0) if len(force_acc) else np.zeros(2)))
                force_acc = []; next_frame += 1
                if not finite:
                    raise RuntimeError('nonfinite physical state at frame%d; remaining unavailable' % i)
            if next_frame == 448:
                break
            if env.tick % 4 == 0:
                index = env.index()
                if reference is None:
                    feature = env.features(index)
                    with torch.no_grad():
                        last_action = model.action(torch.tensor(feature)).numpy()
                else:
                    from .policy import observation_v7, recovery_reference
                    p, r, v, w = env.scene.human_state()
                    target, speed = recovery_reference(p, v, reference['reference_position_m'][index],
                        reference['reference_velocity_m_s'][index])
                    last_action, _ = model.act(observation_v7(p, r, v, w, target, speed))
                actions.append(last_action.copy()); indices.append(index); decision_times.append(now)
            force = env.advance(last_action, ticks=1)
            forces.append(force[0]); force_acc.append(force[0])
            if env.tick % 960 == 0:
                print(json.dumps(dict(replay_frame=next_frame, t=env.tick*H)), flush=True)
        result['status'] = 'complete448_frozen_free_walker_ppo' if reference is None else 'complete448_free_walker_official_PHC_reference_diagnostic'
    except Exception as error:
        result.update(status='failed_incomplete', reason=str(error))
        raise
    finally:
        n = len(records)
        np.savez_compressed(output/'trajectory.npz', human=np.array([r['human'] for r in records]),
            walker=np.array([r['walker'] for r in records]), cameras=np.array([r['camera'] for r in records]),
            right_cameras=np.array([r['right'] for r in records]), grip_gap_m=np.array([r['gap'] for r in records]),
            timestamp_s=np.array([r['time'] for r in records]), source_timestamp_s=packet['timestamp_s'][:n],
            pair_id=packet['pair_id'][:n], available=np.array([r['finite'] for r in records]),
            fallen=np.array([r['fallen'] for r in records]), action=np.array([r['action'] for r in records]),
            velocity=np.array([r['velocity'] for r in records]), angular=np.array([r['angular'] for r in records]),
            interval_force_peak_N=np.array([r['force_peak'] for r in records]), pixel_error_px=np.array(pixel_errors))
        np.savez_compressed(output/'controls.npz', action=actions, force_N=forces,
            observation_index=indices, decision_time_s=decision_times)
        errors = np.asarray(pixel_errors)
        finite_errors = errors[np.isfinite(errors)]
        result.update(saved_frames=n, finite_frames=sum(r['finite'] for r in records),
            fallen_frames=sum(r['fallen'] for r in records),
            first_fallen_frame=next((i for i, r in enumerate(records) if r['fallen']), None),
            simulate_calls=env.scene.explicit_simulate_calls, policy_calls=len(actions),
            force_peak_N=float(np.linalg.norm(forces, axis=2).max()) if forces else 0.,
            max_grip_gap_m=float(max(max(r['gap']) for r in records)),
            finite_projected_slots=int(np.isfinite(errors).sum()), all_slots_denominator=448*24,
            reprojection_finite_median_px=float(np.median(finite_errors)) if len(finite_errors) else None,
            all_slots_within25_fraction=float(np.sum(errors < 25)/(448*24)),
            tracking_passed=bool(n == 448 and not any(r['fallen'] for r in records)
                and np.sum(errors < 25)/(448*24) >= .5),
            completion_latency_max_s=float(max(r['time']-packet['timestamp_s'][i] for i, r in enumerate(records))),
            runtime_s=time.perf_counter()-start)
        (output/'results.json').write_text(json.dumps(clean(result), indent=2), encoding='utf-8')
        env.close()
    print(json.dumps(clean(result)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('foundation', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--updates', type=int, default=16)
    parser.add_argument('--checkpoint', type=Path, help='frozen replay only, without any optimizer or teacher')
    parser.add_argument('--controller', choices=('image', 'phc-reference'), default='image',
        help='phc-reference is a privileged3D controller diagnostic, never an image-policy result')
    args = parser.parse_args()
    if args.updates <= 0:
        parser.error('positive PPO update count required')
    torch.set_num_threads(2)
    base = args.foundation.parent
    packet = arrays(base/'direct_image_20261016/inference_packet.npz')
    # Enforce input separation: only frame0 camera is retained from Stage.
    packet['camera_rotation'][1:] = np.nan
    packet['camera_translation_m'][1:] = np.nan
    if args.controller == 'phc-reference':
        checkpoint = Path(__file__).resolve().parents[2]/'models/phc/phc_kp_2.pth'
        replay(args.foundation, packet, checkpoint, args.output,
            reference=arrays(base/'corrections_v1/reference_matched.npz'))
        return
    if args.checkpoint is not None:
        replay(args.foundation, packet, args.checkpoint, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    initial = base/'direct_image_20261016/train_seed0_complete/ppo.pt'
    train(args.foundation, packet, initial, args.output, args.updates)
    replay(args.foundation, packet, args.output/'ppo.pt', args.output/'replay448')


if __name__ == '__main__':
    main()
