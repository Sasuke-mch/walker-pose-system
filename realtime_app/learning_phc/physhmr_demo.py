"""Frozen PhysHMR inference in the author's task, on a fixed official clip.

No video estimator, fine tuning, teacher policy, or invented image features.
"""

import argparse
import faulthandler
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

import numpy as np


def plot_run(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with np.load(output / 'trajectory.npz', allow_pickle=False) as data:
        position = data['position']
        reference = data['reference_position']
        termination = data['termination']
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(position[:, 0, 2], 'k-', label='PhysHMR physics')
    axes[0].plot(reference[:, 0, 2], 'k--', label='Input reference')
    axes[0].set_ylabel('Pelvis height (m)')
    axes[0].legend()
    error = np.linalg.norm(position-reference, axis=-1).mean(-1)*1000
    axes[1].plot(error, 'k-')
    if termination.any():
        indices = np.flatnonzero(termination)
        axes[1].plot(indices, error[indices], 'kx', label='Author termination flag')
        axes[1].legend()
    axes[1].set_ylabel('Reference discrepancy (mm)')
    axes[1].set_xlabel('Control step (30 Hz)')
    fig.suptitle('Frozen PhysHMR / official AIST clip / CPU / no resets')
    fig.tight_layout()
    fig.savefig(output / 'curves.png', dpi=160)
    plt.close(fig)


def main():
    faulthandler.enable()
    faulthandler.dump_traceback_later(90, repeat=False)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--motion', type=Path, help='explicit adapted motion NPZ; retain initialized frame0')
    parser.add_argument('--plot', action='store_true', help='plot an existing run, without simulation')
    args = parser.parse_args()
    if args.motion is not None:
        args.motion = args.motion.resolve()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.plot:
        plot_run(output)
        return
    if (output / 'frames.jsonl').exists():
        raise FileExistsError('preserve the previous run; choose another output')
    from isaacgym import gymapi  # noqa: F401 -- must precede torch.
    import torch
    import joblib

    torch.set_num_threads(2)
    # Import the required SMPLSim submodules without initializing its unrelated
    # Gymnasium locomotion environments. Actual geometry code is unchanged.
    smpl_path = root / 'third_party/SMPLSim/smpl_sim'
    spec = importlib.util.spec_from_file_location('smpl_sim', smpl_path / '__init__.py',
                                                submodule_search_locations=[str(smpl_path)])
    smpl_sim = importlib.util.module_from_spec(spec)
    sys.modules['smpl_sim'] = smpl_sim
    repo = root / 'third_party/PhysHMR'
    sys.path[:0] = [str(repo), str(repo / 'phc')]
    os.chdir(repo)
    from rl_games.algos_torch import torch_ext
    torch_ext.load_checkpoint = lambda path: torch.load(path, map_location='cpu')
    from learning.im_amp_players import IMAMPPlayerContinuous

    exported = root / ('research_records/engineering_validation/G20261015_phc_pretrained448_v1/'
                       'official_pretrained_20261016/physhmr_first_sample.npz')
    if args.motion is not None:
        exported = args.motion.resolve()
    if args.motion is not None or (args.sample_index == 0 and exported.exists()):
        # Exact first sample exported from the official file in the preceding
        # stage; avoid decompressing all 3233 clips on the WSL mounted drive.
        with np.load(exported, allow_pickle=False) as data:
            key = str(data['sample_name'].item())
            sample = {k: data[k].copy() for k in data.files if k != 'sample_name'}
        sample['root_trans_offset'] = torch.from_numpy(sample['root_trans_offset'])
        sample['fps'], sample['gender'] = int(sample['fps']), str(sample['gender'].item())
    else:
        dataset = joblib.load(root / 'models/physhmr/AIST_test.pkl')
        key = list(dataset)[args.sample_index]
        sample = dataset[key]
        del dataset
    motion_file = output / ('adapted_sample.pkl' if args.motion else 'official_sample.pkl')
    joblib.dump({key: sample}, motion_file)
    count = len(sample['pose_aa'])
    faulthandler.cancel_dump_traceback_later()

    def bounded_run(player):
        obs = player.env_reset()
        player.get_batch_size(obs['obs'], 1)
        task = player.env.task
        properties = task.gym.get_actor_dof_properties(task.envs[0], task.humanoid_handles[0])
        original_drive = properties['driveMode'].tolist()
        actual = properties
        contract = {'source_key': key, 'frames': count, 'seed': 0,
            'original_drive_mode': original_drive, 'drive_mode': actual['driveMode'].tolist(),
            'stiffness': actual['stiffness'].tolist(), 'damping': actual['damping'].tolist(),
            'control_dt_s': task.dt, 'observation_dim': task.num_obs,
            'optimizer_steps': 0, 'teacher_inference': False,
            'sample_index': args.sample_index, 'position_drive_fix': False,
            'source': 'models/physhmr/AIST_test.pkl', 'reset_after_initial': False}
        if args.motion:
            initial = task._get_state_from_motionlib_cache(task._sampled_motion_ids,
                task._motion_start_times + task._motion_start_times_offset, task._global_offset)
            contract.update(source=str(exported), frame0_is_initialized=True,
                physics_steps=count-1, source_frame_indices=list(range(count)),
                input_height_shift_m=float(initial['visual_feat'][0, 0]),
                scope='adapted walker448; real features; own-model root orientation and initial pose')
        (output / 'runtime.json').write_text(json.dumps(contract, indent=2), encoding='utf-8')
        positions, rotations, references, actions, terminated, heights, residuals = [], [], [], [], [], [], []
        with (output / 'frames.jsonl').open('x', encoding='utf-8') as stream:
            for frame in range(count):
                initialized = bool(args.motion and frame == 0)
                if initialized:
                    action = torch.full((1, 69), float('nan'))
                    info = {'body_pos_gt': initial['rg_pos'].detach().cpu().numpy(),
                            'terminate': torch.zeros(1, dtype=torch.bool)}
                    done = torch.zeros(1, dtype=torch.bool)
                else:
                    action = player.get_action(obs, True)
                    if action.ndim == 1:
                        action = action.unsqueeze(0)
                    obs, reward, done, info = player.env_step(player.env, action)
                    if not isinstance(obs, dict) or 'obs' not in obs:
                        obs = player.obs_to_torch(obs)
                positions.append(task._rigid_body_pos[0].detach().cpu().numpy().copy())
                rotations.append(task._rigid_body_rot[0].detach().cpu().numpy().copy())
                references.append(info['body_pos_gt'][0].copy())
                actions.append(action[0].detach().cpu().numpy().copy())
                terminated.append(bool(info['terminate'][0]))
                heights.append(float(task._rigid_body_pos[0, 0, 2]))
                residuals.append(float(np.linalg.norm(positions[-1]-references[-1], axis=-1).mean()))
                stream.write(json.dumps({'frame': frame, 'position': positions[-1].tolist(),
                    'rotation_xyzw': rotations[-1].tolist(), 'action': None if initialized else actions[-1].tolist(),
                    'initialized': initialized,
                    'termination': terminated[-1], 'done': bool(done[0]),
                    'reference_error_m': residuals[-1]}) + '\n')
                if frame % 20 == 0:
                    stream.flush()
                    print(f'frame={frame} height={heights[-1]:.3f} reference_error={residuals[-1]:.4f}', flush=True)
        positions = np.asarray(positions)
        np.savez_compressed(output / 'trajectory.npz', position=positions, rotation_xyzw=rotations,
            reference_position=references, action=actions, termination=terminated)
        fail = np.flatnonzero(terminated)
        summary = {'frames': count, 'finite_frames': int(np.isfinite(positions).all(axis=(1, 2)).sum()),
            'first_termination_frame': int(fail[0]) if len(fail) else None,
            'termination_frames': len(fail), 'minimum_pelvis_height_m': min(heights),
            'reference_joint_error_mean_m': float(np.mean(residuals)),
            'reference_joint_error_p95_m': float(np.percentile(residuals, 95)),
            'scope': ('adapted walker448, 447 physics steps plus initialized frame0; reference is not independent truth'
                      if args.motion else 'author official clip, local CPU bounded no-reset inference; reference is not independent ground truth'),
            'position_drive_fix': False, 'optimizer_steps': 0}
        summary['engineering_gate_passed'] = bool(summary['finite_frames'] == count and not len(fail)
            and min(heights) >= .3 and summary['reference_joint_error_mean_m'] <= .1)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2), flush=True)

    IMAMPPlayerContinuous.run = bounded_run
    sys.argv = ['phc/run_hydra.py', 'learning=im_big', 'exp_name=BC+RL+kp2d_mixed_rerun',
        'env=env_im', 'robot=smpl_humanoid', f'env.motion_file={motion_file.as_posix()}',
        'env.obs_v=358', 'env.kin_loss=False', 'epoch=67500', 'test=True', 'env.num_envs=1',
        'im_eval=True', 'headless=True', 'sim.pipeline=cpu', 'sim.sim_device=cpu',
        'device=cpu', 'rl_device=cpu', '+device_type=cpu',
        '+learning.params.config.device_name=cpu', 'env.stateInit=Start', 'no_log=True',
        f'hydra.run.dir={output.as_posix()}/hydra']
    runpy.run_path('phc/run_hydra.py', run_name='__main__')


if __name__ == '__main__':
    main()
