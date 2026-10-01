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
    routes = comparison['protocol'].get('routes',['control','projected'])
    for window,runs in comparison['runs'].items():
        root = args.output_root/window
        metadata = {r:json.loads((root/r/'run_metadata.json').read_text(encoding='utf-8')) for r in routes}
        config = {r:{k:v for k,v in d['config'].items() if k not in ('constrained_update','nonlinear_correction_steps')} for r,d in metadata.items()}
        inputs = {r:{k:v for k,v in d['inputs'].items() if k!='output_dir'} for r,d in metadata.items()}
        frozen = all(config['control']==config[r] and inputs['control']==inputs[r] for r in routes)
        initial_equal = all(runs['control']['initial']==runs[r]['initial'] for r in routes)
        saved = {}
        references = {}
        for route in routes:
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
        reference_equal = all(np.array_equal(references['control'][k],references[r][k]) for k in references['control'] for r in routes)
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
        if 'corrected' in routes:
            transactions = json.loads((root/'corrected/update_transactions.json').read_text(encoding='utf-8'))['updates']
            body = [x for x in transactions if 'projection' in x]
            accepted = [x for x in body if x['accepted']]
            used = [(x,next(a for a in x['attempts'] if a['accepted'])) for x in accepted]
            last_rejected = [x['attempts'][-1] for x in body if not x['accepted'] and x['attempts']]
            reports[window]['recovery_diagnostic'] = dict(body_transactions=len(body),accepted=len(accepted),
                linear_projection_not_certified=sum(x.get('reason')=='linear_projection_not_certified' for x in body),
                accepted_after_correction=sum(bool(a.get('corrections')) for x,a in used),
                accepted_wrist_squared_distance_descent=sum(
                    sum(x['before_wrist_squared_error_sum_m2'])-sum(a['wrist_squared_error_sum_m2'])>1e-6 for x,a in used),
                recovery_evaluations=sum(len(a.get('corrections',[])) for x in body for a in x['attempts']),
                accepted_with_failed_exact_guard=sum(bool(a['violating_pair_joint']) or not a['upper_leg_shift_guard_passed'] or not a['pca_ok'] for x,a in used),
                final_rejected_observation_block=sum(bool(a['violating_pair_joint']) for a in last_rejected),
                final_rejected_leg_block=sum(not a['upper_leg_shift_guard_passed'] for a in last_rejected),
                final_rejected_guard_feasible_but_not_accepted=sum(a['observation_excess_m']<=1e-6 and
                    a['upper_leg_shift_guard_passed'] and a['pca_ok'] for a in last_rejected))
            reports[window]['wrist_direction_probes'] = {}
            for route in routes:
                probes = {}
                for path in sorted((root/route).glob('wrist_direction_*.json')):
                    probe = json.loads(path.read_text(encoding='utf-8'))
                    probes[path.stem] = dict(status=probe['status'],
                        frames_with_both_linear_derivatives_negative=sum(
                            all(v<0 for v in f['linear_wrist_squared_distance_derivatives']) for f in probe.get('frames',[])),
                        witness=next((a for a in probe.get('attempts',[]) if a['witness']),None),
                        global_feasibility='unknown')
                reports[window]['wrist_direction_probes'][route] = probes
    passed = all(r['frozen_inputs_config_equal'] and r['initial_metrics_equal'] and r['immutable_guard_arrays_equal'] and
        r.get('recovery_diagnostic',{}).get('accepted_with_failed_exact_guard',0)==0 and
        all(v['observation_guard_passed'] and v['frame_ids_match'] and v['finite_mesh'] and v['active_decode_passed'] for v in r['routes'].values()) for r in reports.values())
    report = dict(engineering_contracts_passed=passed,windows=reports,
                  accepted_for_main_fit=False,scope='numerical contracts, not physical fit quality')
    (args.output_root/'independent_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
