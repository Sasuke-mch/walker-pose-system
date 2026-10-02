"""Read-only consistency audit of saved grasp targets and Stage2 foot anchors."""

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
_tool_prepare_imports()
import argparse
import json
from pathlib import Path

import numpy as np


def audit_window(run, expected_frames):
    metadata = json.loads((run / 'run_metadata.json').read_text(encoding='utf-8'))
    grasp = json.loads(Path(metadata['inputs']['grasp']).read_text(encoding='utf-8'))
    target_checks = {}
    for side in ('left', 'right'):
        hand = grasp['hands'][side]
        position = np.asarray(hand['wrist_walker_m'], dtype=float).reshape(3)
        rotation = np.asarray(hand['rotation_walker_from_wrist'], dtype=float).reshape(3, 3)
        transform = np.asarray(hand['transform_walker_from_wrist'], dtype=float).reshape(4, 4)
        target_checks[side] = {'translation_difference_m': float(np.linalg.norm(transform[:3, 3]-position)),
                               'rotation_matrix_difference': float(np.max(np.abs(transform[:3, :3]-rotation))),
                               'rotation_orthogonality_error': float(np.max(np.abs(rotation.T@rotation-np.eye(3)))),
                               'rotation_determinant': float(np.linalg.det(rotation))}
    target_contract = grasp['coordinate_frame'] == 'walker_rigid_local' and grasp['length_unit'] == 'm' and all(
        c['translation_difference_m'] < 1e-6 and c['rotation_matrix_difference'] < 1e-6 and
        c['rotation_orthogonality_error'] < 1e-6 and abs(c['rotation_determinant']-1) < 1e-6
        for c in target_checks.values())
    if not target_contract:
        raise ValueError(f'grasp target transform contract failed: {run}')
    anchor_info = json.loads((run / 'stage2_anchor_assumption.json').read_text(encoding='utf-8'))
    reach = json.loads((run / 'feasibility_initial.json').read_text(encoding='utf-8'))
    with np.load(run / 'result.npz') as result, np.load(run / 'stage2_anchors.npz') as anchors, np.load(run / 'frozen_foot_states.npz') as foot:
        ids = result['pair_id'].copy()
        matched = all(np.array_equal(x, expected_frames) for x in (ids, anchors['pair_id'], foot['pair_id']))
        if not matched:
            raise ValueError(f'frame identity mismatch: {run}')
        wrist = result['wrist_walker_m'].astype(float)
        rotations = result['wrist_rotation_walker'].astype(float)
        vertices = result['vertices_ground_m'].astype(float)
        patch_ids, targets, segments = anchors['patch_ids'], anchors['targets_m'], anchors['segment']
        chosen = vertices[np.arange(len(ids))[:, None, None], patch_ids]
        active = segments >= 0
        deviations = np.linalg.norm(chosen - targets, axis=-1)
        adjacent = active[1:] & active[:-1] & (segments[1:] == segments[:-1])
        velocity = np.linalg.norm((chosen[1:] - chosen[:-1]) * 30, axis=-1)
        records = []
        for t, pair_id in enumerate(ids):
            item = {'pair_id': int(pair_id), 'stage2_segment': int(segments[t]),
                    'frame_valid': bool(foot['frame_ok'][t]),
                    'stage2_anchor_residual_max_mm': float(deviations[t].max() * 1000) if active[t] else None,
                    'stage2_patch_velocity_max_m_s': float(velocity[t-1].max()) if t and adjacent[t-1] else None,
                    'feet': {}}
            for side_index, side in enumerate(('left', 'right')):
                target = np.asarray(grasp['hands'][side]['wrist_walker_m'], dtype=float).reshape(3)
                target_rotation = np.asarray(grasp['hands'][side]['rotation_walker_from_wrist'], dtype=float).reshape(3, 3)
                cosine = np.clip((np.trace(rotations[t, side_index] @ target_rotation.T) - 1) / 2, -1, 1)
                item['feet'][side] = {'frozen_support_candidate': bool(foot['support'][t, side_index]),
                    'frozen_state': str(foot['state'][t, side_index]), 'frozen_reason': str(foot['reason'][t, side_index]),
                    'sole_patch_min_z_mm': (float(chosen[t, side_index*2:(side_index+1)*2, :, 2].min() * 1000)
                                               if active[t] else None),
                    'wrist_error_mm': float(np.linalg.norm(wrist[t, side_index] - target) * 1000),
                    'wrist_rotation_error_deg': float(np.rad2deg(np.arccos(cosine)))}
            records.append(item)
        reach_records = reach['records']
        lift = [{'full_sequence_start': int(record['start']), 'full_sequence_stop': int(record['stop']),
                 'left_mm': float(record['rigid_lift_m'][0])*1000,
                 'right_mm': float(record['rigid_lift_m'][1])*1000}
                for record in anchor_info['records'] if record.get('available')
                and record['start'] < int(ids[-1])+1 and record['stop'] > int(ids[0])]
        return {'run': str(run.resolve()), 'pair_ids': ids.tolist(), 'frame_identity_passed': matched,
                'coordinate_target_frame': metadata['coordinate_frame'],
                'target_source': metadata['inputs']['grasp'], 'target_contract_passed': target_contract,
                'target_checks': target_checks,
                'stage2_boundary_source': anchor_info['boundary_source'],
                'stage2_measured_contact': anchor_info['measured_contact'],
                'stage2_active_frames': int(active.sum()), 'stage2_adjacent_pairs': int(adjacent.sum()),
                'overlapping_anchor_rigid_lifts': lift,
                'anchor_residual_max_mm': float(deviations[active].max()*1000) if active.any() else None,
                'stage2_velocity_max_m_s': float(velocity[adjacent].max()) if adjacent.any() else None,
                'two_link_outer_excess_max_mm': max(float(r['outer_reach_excess_m']) for r in reach_records)*1000,
                'two_link_inner_excess_max_mm': max(float(r['inner_reach_excess_m']) for r in reach_records)*1000,
                'model_joint_coco_wrist_offset_max_mm': max(float(r['model_joint_vs_coco_wrist_m']) for r in reach_records)*1000,
                'global_feasibility': 'unknown', 'physical_contact': 'unverified', 'records': records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison-root', type=Path, required=True)
    parser.add_argument('--route', default='corrected')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.comparison_root / 'protocol.json').read_text(encoding='utf-8'))
    reports = {}
    for start, stop in protocol['windows']:
        key = f'window{start}_{stop-1}'
        reports[key] = audit_window(args.comparison_root / key / args.route, np.arange(start, stop))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError('refuse overwrite of prior audit')
    args.output.write_text(json.dumps({'scope': 'saved engineering target/anchor consistency; no new fit',
                                      'windows': reports}, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
