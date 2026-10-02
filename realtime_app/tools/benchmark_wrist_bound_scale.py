"""Paired corrected-route wrist soft-bound scale ablation on fixed short windows."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from benchmark_constrained_grasp import assess


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-metadata', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=50)
    args = parser.parse_args()
    if args.output_root.exists() or args.steps <= 0:
        raise ValueError('fresh output root and positive steps required')
    metadata = json.loads(args.reference_metadata.read_text(encoding='utf-8'))
    common = []
    for name, path in metadata['inputs'].items():
        if name != 'output_dir':
            common += ['--' + name.replace('_', '-'), str(path)]
    config = dict(metadata['config'])
    config.update(upper_steps=args.steps, body_steps=args.steps,
                  body_polish_steps=args.steps, hand_polish_steps=args.steps)
    for name, value in config.items():
        if name in ('start', 'stop', 'grasp_mesh', 'constrained_update',
                    'projection_trust_radius', 'nonlinear_correction_steps',
                    'wrist_direction_diagnostics', 'wrist_bound_scale_m'):
            continue
        flag = '--' + name.replace('_', '-')
        if isinstance(value, bool):
            if value:
                common.append(flag)
        elif value is not None:
            common += [flag, str(value)]
    common += ['--wrist-direction-diagnostics', '--constrained-update',
               '--projection-trust-radius', '.2', '--nonlinear-correction-steps', '4']
    protocol = {'scope': 'short-window corrected body update, one-factor wrist soft-bound scale',
                'windows': [[270, 296], [60, 101]], 'scales_m': [.001, .005],
                'routes': ['scale1mm', 'scale5mm'], 'steps_per_phase': args.steps,
                'inputs': metadata['inputs'], 'config_except_scale': config,
                'fixed_update': {'constrained': True, 'nonlinear_correction_steps': 4,
                                 'projection_radius': .2},
                'single_variable': 'wrist-bound residual denominator only',
                'final_gates': {'observation_regression_mm': 10, 'wrist_p95_mm': 10,
                                'sole_minimum_z_mm': -3, 'leg_acceleration_not_increased': True,
                                'all_actual_hand_frames_pass': True},
                'relative_gate': 'both wrist P95 and angle P95, body overall and every available group RMS, each sole minimum, and leg acceleration no worse than scale1mm beyond 0.001mm/1e-6 term; at least one strict improvement beyond tolerance',
                'failure_rule': 'retain all failures; no full448 or mainline promotion if any final gate fails'}
    args.output_root.mkdir(parents=True)
    write(args.output_root / 'protocol.json', protocol)
    fit_tool = Path(__file__).with_name('refine_body_with_constructed_grasp.py')
    reports = {}
    for start, stop in protocol['windows']:
        key = f'window{start}_{stop-1}'
        window = args.output_root / key
        window.mkdir()
        reports[key] = {}
        for route, scale in zip(protocol['routes'], protocol['scales_m']):
            out = window / route
            command = [sys.executable, str(fit_tool), *common,
                       '--start', str(start), '--stop', str(stop),
                       '--wrist-bound-scale-m', str(scale), '--output-dir', str(out)]
            write(window / f'{route}_command.json', command)
            print(key, route, 'started', flush=True)
            with (window / f'{route}.log').open('w', encoding='utf-8') as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                write(args.output_root / 'FAILURE.json', {'window': key, 'route': route,
                    'returncode': process.returncode, 'log': str(window / f'{route}.log')})
                raise RuntimeError('fitting failed; evidence retained')
            audit = [sys.executable, str(fit_tool.with_name('audit_constructed_grasp_body.py')),
                     '--result', str(out / 'result.npz'),
                     '--walker-model', metadata['inputs']['walker_model'],
                     '--output', str(out / 'hand_geometry_audit.json')]
            write(window / f'{route}_audit_command.json', audit)
            with (window / f'{route}_audit.log').open('w', encoding='utf-8') as log:
                process = subprocess.run(audit, stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                write(args.output_root / 'FAILURE.json', {'window': key, 'route': route,
                    'returncode': process.returncode, 'log': str(window / f'{route}_audit.log')})
                raise RuntimeError('hand audit failed; evidence retained')
            reports[key][route] = assess(out)
            write(args.output_root / 'partial_results.json', reports)
            print(key, route, 'completed', flush=True)
    write(args.output_root / 'comparison.json', {'protocol': protocol, 'runs': reports,
        'accepted_for_main_fit': False, 'full448_started': False})


if __name__ == '__main__':
    main()
