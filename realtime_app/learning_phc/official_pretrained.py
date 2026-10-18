"""Bounded, frozen-weight probes of the two authors' released policies.

Network equivalence is an interface check, never a tracking-accuracy claim.
The optional Mimic2DM demo uses the author's environment and observation code.
"""

import argparse
import ast
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import sys
import types

import numpy as np


def definitions(path, names, namespace):
    """Execute selected author definitions, avoiding their training-only imports."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    selected = [node for node in tree.body if getattr(node, 'name', None) in names]
    if len(selected) != len(names):
        raise ValueError(f'author definitions changed: {path}: {names}')
    for node in selected:
        node.decorator_list = []
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def network_probe(root, output):
    import torch
    from torch import nn
    from .policy import FrozenPHC, _mlp

    torch.set_num_threads(2)
    torch.manual_seed(42)
    result = {'scope': 'frozen network equivalence only; no physical tracking score',
              'optimizer_steps': 0, 'seed': 42}
    ph = torch.load(root / 'models/physhmr/Humanoid_00067500.pth',
                    map_location='cpu', weights_only=False)
    state = ph['model']
    actor = nn.Sequential(_mlp(state, 'a2c_network.actor_mlp.'), nn.SiLU())
    mu = nn.Linear(512, 69)
    mu.load_state_dict({k: state['a2c_network.mu.' + k] for k in ('weight', 'bias')})
    actor.eval().requires_grad_(False)
    mu.eval().requires_grad_(False)
    # Bind the actual author eval_actor method to the checkpoint modules.
    path = root / 'third_party/PhysHMR/phc/learning/amp_network_builder.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    outer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AMPBuilder')
    inner = next(n for n in outer.body if isinstance(n, ast.ClassDef) and n.name == 'Network')
    method = next(n for n in inner.body if isinstance(n, ast.FunctionDef) and n.name == 'eval_actor')
    ns = {'torch': torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), ns)
    author = types.SimpleNamespace(actor_cnn=nn.Identity(), actor_mlp=actor,
        mu=mu, mu_act=nn.Identity(), sigma_act=nn.Identity(), has_rnn=False,
        is_discrete=False, is_multi_discrete=False, is_continuous=True,
        space_config={'fixed_sigma': True}, sigma=state['a2c_network.sigma'])
    norm = ph['running_mean_std']
    raw = norm['running_mean'].float() + torch.randn(128, 924) * torch.sqrt(norm['running_var'].float()+1e-5)
    with torch.inference_mode():
        obs = FrozenPHC.normalize(raw, norm['running_mean'], norm['running_var']).float()
        expected = ns['eval_actor'](author, {'obs': obs})[0]
        actual = mu(actor(obs))
    error = float((actual-expected).abs().max())
    assert error <= 1e-6 and torch.isfinite(actual).all()
    result['physhmr'] = {'observation_dim': 924, 'action_dim': 69,
        'batch': 128, 'author_max_abs_error': error, 'finite': True,
        'epoch': ph['epoch'], 'physical_demo': 'not_run_missing_neutral_smpl',
        'walker448': 'not_run_missing_GVHMR_features_and_compatible_camera_preprocessing'}

    path = root / 'third_party/Mimic2DM/mimic2dm/learning'
    spec = importlib.util.spec_from_file_location('author_distribution_probe', path / 'distributions.py')
    distribution = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(distribution)
    ns = definitions(path / 'mview_network_builder.py', {'_norm_initializer', 'MlpPolicyNet'},
        {'torch': torch, 'nn': nn, 'np': np, 'Parameter': nn.Parameter,
         'DiagGaussianDistribution': distribution.DiagGaussianDistribution})
    mc = torch.load(root / 'third_party/Mimic2DM/output/singleview_dribble_ball_control.pth',
                    map_location='cpu', weights_only=False)
    net = ns['MlpPolicyNet'](164, 208, 28, 512, -2.9)
    prefix = 'a2c_network.pi_net.'
    net.load_state_dict({k[len(prefix):]: v for k, v in mc['model'].items() if k.startswith(prefix)})
    net.eval().requires_grad_(False)
    self_obs = torch.randn(128, 164)
    future = torch.randn(128, 208)
    with torch.inference_mode():
        action = net(self_obs, future, mview_obs=future[:, None],
            view_ang=torch.zeros(128, 1, 3), alpha=torch.ones(128, 1),
            enable_rot_emb=False, enable_src_view_emb=True).mode()
        expected = net.mlp(torch.cat([self_obs, net._tracking_head(future)], -1))
    error = float((action-expected).abs().max())
    assert error <= 1e-6 and torch.isfinite(action).all()
    result['mimic2dm'] = {'self_observation_dim': 164, 'tracking_dim': 208,
        'action_dim': 28, 'batch': 128, 'author_max_abs_error': error,
        'finite': True, 'epoch': mc['epoch'],
        'walker448': 'not_run_released_ball_specific_body_and_observation_contract'}
    (output / 'network_probe.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


def mimic_demo(root, output, steps, position_drive=False):
    # Isaac Gym must be imported before torch. CPU has no CUDA tensor deserialization.
    from isaacgym import gymapi  # noqa: F401
    import torch

    torch.set_num_threads(2)
    torch.storage._load_from_bytes = lambda data: torch.load(io.BytesIO(data), map_location='cpu')
    os.chdir(root / 'third_party/Mimic2DM')
    sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / 'mimic2dm')]
    # The author's entry unconditionally initializes Horovod, even with multi_gpu=False.
    # Only rank/init are needed for single-process inference; all other uses still fail.
    hvd = types.ModuleType('horovod.torch')
    hvd.init, hvd.rank, hvd.size, hvd.local_rank = lambda: None, lambda: 0, lambda: 1, lambda: 0
    parent = types.ModuleType('horovod')
    parent.torch = hvd
    sys.modules['horovod'], sys.modules['horovod.torch'] = parent, hvd
    from learning.mview_players import MultiViewPlayerContinuous
    from rl_games.algos_torch import torch_ext
    import yaml

    torch_ext.load_checkpoint = lambda path: torch.load(path, map_location='cpu')

    sample = Path.cwd() / 'sample_data/dribble_ball_control/amass_isaac_ball_control_shot_10_track_24_upright_slim.pkl'
    motion_file = output / 'mimic_single_sample.yaml'
    motion_file.write_text(yaml.safe_dump({'motions': {'avatar': [
        {'file': str(sample), 'weight': 1.0}]}}), encoding='utf-8')
    train_cfg = yaml.safe_load(Path('mimic2dm/data/cfg/train/rlg/singleview_tracking_humanoid.yaml').read_text())
    train_cfg['params']['config']['device_name'] = 'cpu'
    train_file = output / 'mimic_cpu_player.yaml'
    train_file.write_text(yaml.safe_dump(train_cfg), encoding='utf-8')

    def bounded_run(player):
        obs = player.env_reset()
        task = player.env.task
        properties = task.gym.get_actor_dof_properties(task.envs[0], task.humanoid_handles[0])
        original_drive = properties['driveMode'].tolist()
        if position_drive:
            properties['driveMode'] = gymapi.DOF_MODE_POS
            task.gym.set_actor_dof_properties(task.envs[0], task.humanoid_handles[0], properties)
            properties = task.gym.get_actor_dof_properties(task.envs[0], task.humanoid_handles[0])
        contract = {'drive_mode': properties['driveMode'].tolist(),
            'original_drive_mode': original_drive, 'position_drive_fix': position_drive,
            'stiffness': properties['stiffness'].tolist(), 'damping': properties['damping'].tolist(),
            'position_drive_enum': int(gymapi.DOF_MODE_POS), 'control_dt_s': task.dt,
            'num_actions': task.num_actions, 'player_device': str(player.device)}
        (output / 'mimic_runtime_contract.json').write_text(json.dumps(contract, indent=2), encoding='utf-8')
        if steps == 0:
            print(json.dumps(contract, indent=2))
            return
        positions, actions, done_flags, errors = [], [], [], []
        with (output / 'mimic_frames.jsonl').open('x', encoding='utf-8') as stream:
            for frame in range(steps):
                action = player.get_action(obs, True)
                if action.ndim == 1:
                    action = action.unsqueeze(0)
                obs, reward, done, info = player.env_step(player.env, action)
                if 'obs' not in obs:
                    obs = player.obs_to_torch(obs)
                task = player.env.task
                positions.append(task._rigid_body_pos[0].detach().cpu().numpy().copy())
                actions.append(action[0].detach().cpu().numpy().copy())
                done_flags.append(bool(done[0]))
                ids = task._track_joint_ids
                error = torch.linalg.vector_norm(task._real_body_pos_loc2d[:, ids] -
                            task._ref_body_pos_loc2d[:, 0, task._ref_joint_ids], dim=-1)
                errors.append(error[0].detach().cpu().numpy().copy())
                stream.write(json.dumps({'frame': frame, 'position': positions[-1].tolist(),
                    'action': actions[-1].tolist(), 'termination': done_flags[-1],
                    'author_2d_coordinate_error': errors[-1].tolist()}) + '\n')
                if frame % 30 == 0:
                    stream.flush()
                    print(f'frame={frame} done={bool(done[0])} error={float(error.mean()):.3f}', flush=True)
        positions, errors = np.asarray(positions), np.asarray(errors)
        np.savez_compressed(output / 'mimic_demo.npz', position=positions,
                            action=actions, termination=done_flags, pixel_error=errors)
        summary = {'steps': steps, 'reset_count_after_initial': 0,
            'position_drive_fix': position_drive,
            'finite_frames': int(np.isfinite(positions).all(axis=(1, 2)).sum()),
            'termination_frames': int(sum(done_flags)),
            'minimum_pelvis_height_m': float(positions[:, 0, 2].min()),
            'author_2d_coordinate_error_median': float(np.median(errors)),
            'author_2d_coordinate_error_p95': float(np.percentile(errors, 95)),
            'scope': 'author demo environment, frozen weights, CPU, bounded no-reset diagnostic; not walker448'}
        (output / 'mimic_demo.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2))

    MultiViewPlayerContinuous.run = bounded_run
    sys.argv = ['mimic2dm/run.py', '--test', '--task', 'HumanoidAMPTracking2D',
        '--cfg_env', 'mimic2dm/data/cfg/dribble-smpl_tracking2d_single_view_test.yaml',
        '--cfg_train', str(train_file),
        '--motion_file', str(motion_file),
        '--num_envs', '1', '--checkpoint', 'output/singleview_dribble_ball_control.pth',
        '--seed', '42', '--headless', '--sim_device', 'cpu', '--pipeline', 'cpu', '--rl_device', 'cpu']
    runpy.run_path('mimic2dm/run.py', run_name='__main__')


def compare_mimic(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    results = {}
    for directory, label, style in [('', 'Released drive mode (NONE)', '--'),
            ('mimic_position_drive', 'Position drive enabled', '-')]:
        path = output / directory
        with np.load(path / 'mimic_demo.npz', allow_pickle=False) as data:
            height = data['position'][:, 0, 2]
            error = data['pixel_error']  # Legacy key: author's internal 2D coordinates.
            terminated = data['termination']
        first = np.flatnonzero(terminated)
        low = np.flatnonzero(height < .3)
        summary = json.loads((path / 'mimic_demo.json').read_text(encoding='utf-8'))
        summary.update(first_termination_step=int(first[0]) if len(first) else None,
            first_pelvis_below_0_3m_step=int(low[0]) if len(low) else None,
            pelvis_below_0_3m_frames=int(len(low)))
        results[label] = summary
        axes[0].plot(height, style, color='black', lw=1.2, label=label)
        axes[1].plot(error.mean(1), style, color='black', lw=1.2, label=label)
    axes[0].set_ylabel('Pelvis height (m)')
    axes[0].legend()
    axes[1].set_ylabel('Mean 2D residual\n(author coordinate units)')
    axes[1].set_xlabel('Physics control steps (30 Hz)')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('Frozen Mimic2DM / author dribble sample / CPU / no resets')
    fig.tight_layout()
    fig.savefig(output / 'mimic_comparison.png', dpi=160)
    plt.close(fig)
    (output / 'mimic_comparison.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['network', 'mimic-demo', 'compare'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--position-drive', action='store_true',
                        help='explicitly enable the position drives required by pdControl=True')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    result_name = {'network': 'network_probe.json', 'mimic-demo': 'mimic_demo.npz',
                   'compare': 'mimic_comparison.json'}[args.mode]
    if (args.output / result_name).exists():
        raise FileExistsError('preserve the previous probe; choose a new output directory')
    if args.mode == 'network':
        network_probe(root, args.output)
    elif args.mode == 'mimic-demo':
        mimic_demo(root, args.output, args.steps, args.position_drive)
    else:
        compare_mimic(args.output)


if __name__ == '__main__':
    main()
