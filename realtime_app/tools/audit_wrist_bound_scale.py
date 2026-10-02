"""Independent frozen-input and gate review of the wrist-scale ablation."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_constrained_grasp import assess


def same_arrays(a, b):
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root
    protocol = json.loads((root / 'protocol.json').read_text(encoding='utf-8'))
    comparison = json.loads((root / 'comparison.json').read_text(encoding='utf-8'))
    reports = {}
    for start, stop in protocol['windows']:
        window = f'window{start}_{stop-1}'
        records = {}
        for route, expected_scale in zip(protocol['routes'], protocol['scales_m']):
            path = root / window / route
            meta = json.loads((path / 'run_metadata.json').read_text(encoding='utf-8'))
            with np.load(path / 'result.npz') as result, np.load(path / 'observation_guard_reference.npz') as reference:
                frames = result['pair_id'].copy()
                refs = {key: reference[key].copy() for key in reference.files}
                mesh_finite = bool(np.isfinite(result['vertices']).all())
                frame_match = bool(np.array_equal(frames, np.arange(start, stop)) and
                                   np.array_equal(frames, reference['pair_id']))
            with np.load(path / 'stage2_anchors.npz') as anchors, np.load(path / 'frozen_foot_states.npz') as foot:
                anchor_arrays = {key: anchors[key].copy() for key in anchors.files}
                foot_arrays = {key: foot[key].copy() for key in foot.files}
                frame_match = frame_match and np.array_equal(frames, anchors['pair_id']) and np.array_equal(frames, foot['pair_id'])
            checked = assess(path)
            if any(checked[key] != comparison['runs'][window][route][key]
                   for key in ('frames', 'gates', 'initial', 'final', 'hand_passed', 'hand_total')):
                raise ValueError(f'comparison disagrees with saved result: {window}/{route}')
            records[route] = {'inputs': {k: v for k, v in meta['inputs'].items() if k != 'output_dir'},
                              'config': {k: v for k, v in meta['config'].items()
                                         if k not in ('start', 'stop', 'wrist_bound_scale_m')},
                              'scale': meta['config']['wrist_bound_scale_m'],
                              'scale_declared': meta['config']['wrist_bound_scale_m'] == expected_scale,
                              'protocol_config_match': all(meta['config'].get(k) == v
                                  for k, v in protocol['config_except_scale'].items()
                                  if k not in ('start', 'stop', 'grasp_mesh', 'constrained_update',
                                               'projection_trust_radius', 'nonlinear_correction_steps',
                                               'wrist_direction_diagnostics', 'wrist_bound_scale_m')),
                              'protocol_inputs_match': all(meta['inputs'].get(k) == v
                                  for k, v in protocol['inputs'].items() if k != 'output_dir'),
                              'frame_match': frame_match, 'mesh_finite': mesh_finite,
                              'initial': checked['initial'], 'reference': refs,
                              'anchors': anchor_arrays, 'foot_states': foot_arrays,
                              'gates': checked['gates'], 'hand_coverage': checked['hand_coverage'],
                              'final': checked['final']}
        a, b = (records[r] for r in protocol['routes'])
        initial_a = json.loads(json.dumps(a['initial']))
        initial_b = json.loads(json.dumps(b['initial']))
        initial_a['terms'].pop('wrist_bound', None)
        initial_b['terms'].pop('wrist_bound', None)
        frozen = a['inputs'] == b['inputs'] and a['config'] == b['config'] and initial_a == initial_b
        immutable = (same_arrays(a['reference'], b['reference']) and
                     same_arrays(a['anchors'], b['anchors']) and
                     same_arrays(a['foot_states'], b['foot_states']))
        af, bf = a['final'], b['final']
        group_coverage_equal = all((old is None) == (bf['body_groups'].get(name) is None)
                                   for name, old in af['body_groups'].items()) and af['body_groups'].keys() == bf['body_groups'].keys()
        wrist_position = {side: bf['wrist_error_mm'][side]['p95'] - af['wrist_error_mm'][side]['p95'] for side in ('left', 'right')}
        wrist_angle = {side: bf['wrist_angle_deg'][side]['p95'] - af['wrist_angle_deg'][side]['p95'] for side in ('left', 'right')}
        body = bf['body_3d_rms_mm'] - af['body_3d_rms_mm']
        groups = {name: bf['body_groups'][name]['rms_mm'] - old['rms_mm']
                  for name, old in af['body_groups'].items() if old is not None and bf['body_groups'][name] is not None}
        feet = {side: bf['sole_minimum_z_mm'][side] - af['sole_minimum_z_mm'][side] for side in ('left', 'right')}
        leg = bf['terms']['leg_ground_acceleration'] - af['terms']['leg_ground_acceleration']
        nonworsening = (all(v <= .001 for v in wrist_position.values()) and
                        all(v <= .001 for v in wrist_angle.values()) and body <= .001 and
                        all(v <= .001 for v in groups.values()) and all(v >= -.001 for v in feet.values()) and leg <= 1e-6)
        improvement = (any(v < -.001 for v in wrist_position.values()) or any(v < -.001 for v in wrist_angle.values()) or
                       body < -.001 or any(v < -.001 for v in groups.values()) or
                       any(v > .001 for v in feet.values()) or leg < -1e-6)
        update_contract = all(x['config'].get('constrained_update') is True and
                              x['config'].get('nonlinear_correction_steps') == protocol['fixed_update']['nonlinear_correction_steps'] and
                              x['config'].get('projection_trust_radius') == protocol['fixed_update']['projection_radius'] and
                              x['config'].get('wrist_bound_m') == .01 for x in (a, b))
        contract = frozen and immutable and group_coverage_equal and update_contract and all(
            x['scale_declared'] and x['frame_match'] and x['mesh_finite'] and
            x['hand_coverage']['passed'] and x['gates']['observations'] and
            x['protocol_config_match'] and x['protocol_inputs_match'] for x in (a, b))
        reports[window] = {'contract_passed': contract, 'inputs_config_initial_equal_except_scale': frozen,
                           'immutable_budget_equal': immutable, 'body_group_coverage_equal': group_coverage_equal,
                           'declared_update_rules_passed': update_contract,
                           'protocol_config_inputs_match': all(x['protocol_config_match'] and x['protocol_inputs_match'] for x in (a,b)),
                           'scales_m': {r: records[r]['scale'] for r in protocol['routes']},
                           'wrist_p95_difference_mm': wrist_position,
                           'wrist_angle_p95_difference_deg': wrist_angle,
                           'body_rms_difference_mm': body, 'body_group_difference_mm': groups,
                           'sole_minimum_z_difference_mm': feet, 'leg_term_difference': leg,
                           'relative_nonworsening': nonworsening,
                           'relative_strict_improvement': improvement,
                           'relative_gate_passed': contract and nonworsening and improvement,
                           'original_final_gates': {r: records[r]['gates'] for r in protocol['routes']}}
    output = root / 'independent_scale_audit.json'
    if output.exists():
        raise ValueError('refuse overwrite independent audit')
    output.write_text(json.dumps({'windows': reports,
        'accepted_for_main_fit': False,
        'all_windows_relative_gate_passed': all(x['relative_gate_passed'] for x in reports.values()),
        'all_original_final_gates_passed': all(all(all(g.values()) for g in x['original_final_gates'].values()) for x in reports.values())},
        ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
