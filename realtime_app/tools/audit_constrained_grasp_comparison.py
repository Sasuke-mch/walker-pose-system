"""Independent saved-array audit of frozen projected-update experiments."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'realtime_app'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    args = parser.parse_args()
    comparison = json.loads((args.output_root/'comparison.json').read_text(encoding='utf-8'))
    from pose_app.smplx_fitting import load_vposer_explicit
    from pose_app.vposer_grasp_body import decode_body_rotations
    vposer,_,_ = load_vposer_explicit(Path(comparison['protocol']['inputs']['vposer_dir']),'cpu')
    reports = {}
    for window,runs in comparison['runs'].items():
        root = args.output_root/window
        metadata = {r:json.loads((root/r/'run_metadata.json').read_text(encoding='utf-8')) for r in ('control','projected')}
        config = {r:{k:v for k,v in d['config'].items() if k!='constrained_update'} for r,d in metadata.items()}
        inputs = {r:{k:v for k,v in d['inputs'].items() if k!='output_dir'} for r,d in metadata.items()}
        frozen = config['control']==config['projected'] and inputs['control']==inputs['projected']
        initial_equal = runs['control']['initial']==runs['projected']['initial']
        saved = {}
        references = {}
        for route in ('control','projected'):
            with np.load(root/route/'result.npz') as z, np.load(root/route/'observation_guard_reference.npz') as ref:
                references[route] = {k:ref[k].copy() for k in ref.files}
                residual = np.linalg.norm(z['predicted_coco']-z['raw_triangulated_points']/1000,axis=-1)
                maximum = float((residual-ref['baseline_error_m'])[ref['valid']].max())
                with torch.no_grad():
                    decoded = decode_body_rotations(vposer,torch.from_numpy(z['vposer_latent']))
                replay_error = float(np.abs(decoded.numpy()-z['body_rotation_matrices']).max())
                saved[route] = dict(max_observation_regression_mm=maximum*1000,
                    cpu_vposer_decode_max_element_error=replay_error,
                    observation_guard_passed=maximum<=.010002,
                    frame_ids_match=bool(np.array_equal(z['pair_id'],ref['pair_id'])),
                    actual_mesh_shape=list(z['vertices'].shape),faces_shape=list(z['faces'].shape),
                    finite_mesh=bool(np.isfinite(z['vertices']).all()),
                    active_decode_passed=replay_error<1e-5)
            diagnostic = json.loads((root/route/'feasibility_initial.json').read_text(encoding='utf-8'))
            records = diagnostic['records']
            saved[route]['initial_reach_diagnostic'] = dict(
                outer_excess_max_mm=max(x['outer_reach_excess_m'] for x in records)*1000,
                inner_excess_max_mm=max(x['inner_reach_excess_m'] for x in records)*1000,
                wrist_joint_coco_offset_max_mm=max(x['model_joint_vs_coco_wrist_m'] for x in records)*1000,
                global_feasibility='unknown')
        reference_equal = all(np.array_equal(references['control'][k],references['projected'][k]) for k in references['control'])
        transactions = json.loads((root/'projected/update_transactions.json').read_text(encoding='utf-8'))['updates']
        body = [x for x in transactions if 'projection' in x]
        rejected = [x for x in body if not x['accepted']]
        worst = max((f['max_linear_violation'] for x in body for f in x['projection']['frames']),default=0.)
        rejection = dict(body_transactions=len(body),rejected=len(rejected),
            positive_objective_direction_derivative=sum(x['projection']['objective_directional_derivative']>0 for x in body),
            trust_clipped_frames=(sum(x['projection']['trust_clipped_frames'] for x in body)
                                  if all('trust_clipped_frames' in x['projection'] for x in body) else None),
            max_linear_projection_violation=worst,
            last_attempt_observation_block=sum(bool(x['attempts'] and x['attempts'][-1]['violating_pair_joint']) for x in rejected),
            last_attempt_leg_block=sum(bool(x['attempts'] and not x['attempts'][-1]['upper_leg_shift_guard_passed']) for x in rejected))
        reports[window] = dict(frozen_inputs_config_equal=frozen,initial_metrics_equal=initial_equal,
                               immutable_guard_arrays_equal=reference_equal,routes=saved,projection_diagnostic=rejection)
    passed = all(r['frozen_inputs_config_equal'] and r['initial_metrics_equal'] and r['immutable_guard_arrays_equal'] and
        all(v['observation_guard_passed'] and v['frame_ids_match'] and v['finite_mesh'] and v['active_decode_passed'] for v in r['routes'].values()) for r in reports.values())
    report = dict(engineering_contracts_passed=passed,windows=reports,
                  accepted_for_main_fit=False,scope='numerical contracts, not physical fit quality')
    (args.output_root/'independent_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
