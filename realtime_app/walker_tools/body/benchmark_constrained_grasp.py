"""Single-factor projected-update comparison on two preregistered windows."""

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
_tool_prepare_imports()
import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
import numpy as np


def hand_audit_coverage(records, frames):
    """All-frame/all-hand means exactly one record for each expected identity."""
    frames = [int(v) for v in frames]
    expected = {(p,s) for p in frames for s in ('left','right')}
    actual = Counter()
    malformed = 0
    for record in records:
        try:
            pair = int(record['pair_id'])
            side = record['hand']
            if side not in ('left','right'):
                raise ValueError('unknown hand')
            actual[pair,side] += 1
        except (KeyError,TypeError,ValueError):
            malformed += 1
    missing = sorted(expected-set(actual))
    unexpected = sorted(set(actual)-expected)
    duplicates = sorted(k for k,v in actual.items() if v!=1)
    return dict(passed=bool(expected) and len(set(frames))==len(frames) and not
                (missing or unexpected or duplicates or malformed),
                expected_records=2*len(frames),actual_records=len(records),
                missing=missing,unexpected=unexpected,duplicates=duplicates,malformed=malformed)


def assess(out):
    summary = json.loads((out/'fit_summary.json').read_text(encoding='utf-8'))
    initial = summary['checkpoints']['initial']
    final = list(summary['checkpoints'].values())[-1]
    with np.load(out/'result.npz') as result, np.load(out/'observation_guard_reference.npz') as reference:
        residual = np.linalg.norm(result['predicted_coco']-result['raw_triangulated_points']/1000,axis=-1)
        regression = float((residual-reference['baseline_error_m'])[reference['valid']].max()*1000)
        frames = result['pair_id'].tolist()
    audit = json.loads((out/'hand_geometry_audit.json').read_text(encoding='utf-8'))
    coverage = hand_audit_coverage(audit['records'],frames)
    updates = json.loads((out/'update_transactions.json').read_text(encoding='utf-8'))['updates']
    gates = dict(observations=regression<=10.002,
                 wrists=all(v['p95']<=10 for v in final['wrist_error_mm'].values()),
                 feet=all(v>=-3 for v in final['sole_minimum_z_mm'].values()),
                 leg_temporal=final['terms']['leg_ground_acceleration']<=initial['terms']['leg_ground_acceleration'],
                 hands=coverage['passed'] and all(v['passed_geometry_proxy'] for v in audit['records']))
    return dict(initial=initial,final=final,frames=frames,max_observation_regression_mm=regression,
                updates={stage:dict(accepted=sum(x['accepted'] for x in updates if x['stage']==stage),
                                   total=sum(x['stage']==stage for x in updates)) for stage in sorted({x['stage'] for x in updates})},
                hand_passed=sum(x['passed_geometry_proxy'] for x in audit['records']),hand_total=len(audit['records']),hand_coverage=coverage,
                gates=gates,passed=all(gates.values()),accepted_for_main_fit=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-metadata',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=50)
    parser.add_argument('--include-corrected',action='store_true',help='add one-factor nonlinear recovery route and common read-only wrist probes')
    args = parser.parse_args()
    if args.output_root.exists() or args.steps<=0:
        raise ValueError('new output root and positive steps required')
    metadata = json.loads(args.reference_metadata.read_text(encoding='utf-8'))
    args.output_root.mkdir(parents=True)
    def write(path,value):
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    common = []
    for name,path in metadata['inputs'].items():
        if name!='output_dir':
            common += ['--'+name.replace('_','-'),str(path)]
    config = dict(metadata['config'])
    config.update(upper_steps=args.steps,body_steps=args.steps,body_polish_steps=args.steps,hand_polish_steps=args.steps)
    for name,value in config.items():
        if name in ('start','stop','grasp_mesh','constrained_update','projection_trust_radius','nonlinear_correction_steps','wrist_direction_diagnostics'):
            continue
        flag = '--'+name.replace('_','-')
        if isinstance(value,bool):
            if value:
                common.append(flag)
        elif value is not None:
            common += [flag,str(value)]
    if args.include_corrected:
        common += ['--wrist-direction-diagnostics']
    protocol = dict(single_variable='body update rule: guarded Adam vs normalized frame-local halfspace projection + exact guard',
        windows=[[270,296],[60,101]],steps_per_phase=args.steps,inputs=metadata['inputs'],config=config,
        scene_scope='frozen existing balanced_full448 scene; algorithm ablation, not new end-to-end replay',
        projection=dict(scales=dict(latent=1,root_radians=1,translation_m=.05),per_frame_trust_radius=.2),
        gates=dict(observation_regression_mm=10,wrist_p95_mm=10,sole_minimum_z_mm=-3,
                   leg_acceleration_not_increased=True,all_actual_hand_frames_pass=True),
        diagnostic_progress='both wrist P95 at least 10% below same-window control; exact observation gate retained',
        stopping='Any main gate fails: no full448 upgrade. No weights, thresholds or contact targets changed.')
    if args.include_corrected:
        protocol.update(routes=['control','projected','corrected'],nonlinear_correction_steps=4,
            recovery_scope='candidate Jacobian projection, max 4 corrections per alpha, original radius and exact gates',
            correction_single_variable='corrected vs projected changes only nonlinear feasibility recovery',
            additional_diagnostic_gate='both wrists P95 >=10% better than Adam; body RMS and every available body-group RMS, each sole minimum, leg acceleration no worse than Adam',
            sustained_progress_gate='each of 3 body stages has wrist squared-distance reduction >1e-6 m2 in at least 3 distinct 10-step blocks; observational gate required')
    write(args.output_root/'protocol.json',protocol)
    reports = {}
    tool = (Path(__file__).resolve().parents[2] / "tools" / 'refine_body_with_constructed_grasp.py')
    for start,stop in protocol['windows']:
        key = f'window{start}_{stop-1}'
        root = args.output_root/key
        root.mkdir()
        reports[key] = {}
        for route in protocol.get('routes',('control','projected')):
            out = root/route
            command = [sys.executable,str(tool),*common,'--start',str(start),'--stop',str(stop),'--output-dir',str(out)]
            if route in ('projected','corrected'):
                command += ['--constrained-update','--projection-trust-radius','.2']
            if route=='corrected':
                command += ['--nonlinear-correction-steps','4']
            write(root/(route+'_command.json'),command)
            print(key,route,'started',flush=True)
            with (root/(route+'.log')).open('w',encoding='utf-8') as log:
                process = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            if process.returncode:
                write(args.output_root/'FAILURE.json',dict(window=key,route=route,returncode=process.returncode,log=str(root/(route+'.log'))))
                raise RuntimeError('fitting failed; evidence retained')
            audit = [sys.executable,str(tool.with_name('audit_constructed_grasp_body.py')),'--result',str(out/'result.npz'),
                     '--walker-model',metadata['inputs']['walker_model'],'--output',str(out/'hand_geometry_audit.json')]
            write(root/(route+'_audit_command.json'),audit)
            with (root/(route+'_audit.log')).open('w',encoding='utf-8') as log:
                subprocess.run(audit,stdout=log,stderr=subprocess.STDOUT,check=True)
            reports[key][route] = assess(out)
            write(args.output_root/'partial_results.json',reports)
            print(key,route,'completed',flush=True)
        control,projected = reports[key]['control'],reports[key]['projected']
        reports[key]['diagnostic_progress'] = projected['gates']['observations'] and all(
            projected['final']['wrist_error_mm'][side]['p95'] <= .9*control['final']['wrist_error_mm'][side]['p95'] for side in ('left','right'))
        if args.include_corrected:
            corrected = reports[key]['corrected']
            a,b = corrected['final'],control['final']
            nonworsening = dict(body=a['body_3d_rms_mm']<=b['body_3d_rms_mm']+.001,
                body_groups=all(a['body_groups'][g] is not None and a['body_groups'][g]['rms_mm']<=v['rms_mm']+.001
                                for g,v in b['body_groups'].items() if v is not None),
                feet=all(a['sole_minimum_z_mm'][s]>=b['sole_minimum_z_mm'][s]-.001 for s in ('left','right')),
                leg_temporal=a['terms']['leg_ground_acceleration']<=b['terms']['leg_ground_acceleration']+1e-6)
            transactions=json.loads((root/'corrected/update_transactions.json').read_text(encoding='utf-8'))['updates']
            blocks={}
            for stage in ('upper_only','upper_and_torso','body_wrist_polish'):
                values=[]
                for start_step in range(0,args.steps,10):
                    progress=0.
                    for t in transactions:
                        if t['stage']==stage and start_step<=t['step']<start_step+10 and t['accepted']:
                            attempt=next(x for x in t['attempts'] if x['accepted'])
                            progress+=sum(t['before_wrist_squared_error_sum_m2'])-sum(attempt['wrist_squared_error_sum_m2'])
                    values.append(progress)
                blocks[stage]=values
            sustained=all(sum(v>1e-6 for v in values)>=3 for values in blocks.values())
            reports[key]['corrected_diagnostics']=dict(nonworsening_vs_adam=nonworsening,
                wrist_progress=corrected['gates']['observations'] and all(a['wrist_error_mm'][s]['p95']<=.9*b['wrist_error_mm'][s]['p95'] for s in ('left','right')),
                wrist_progress_blocks_m2=blocks,sustained_progress=sustained)
            reports[key]['corrected_diagnostics']['passed']=all(nonworsening.values()) and sustained and reports[key]['corrected_diagnostics']['wrist_progress']
    report = dict(protocol=protocol,runs=reports,accepted_for_main_fit=False,full448_started=False,
                  all_windows_passed=all(r['corrected' if args.include_corrected else 'projected']['passed'] for r in reports.values()))
    write(args.output_root/'comparison.json',report)
    print(json.dumps({k:dict(progress=r['diagnostic_progress'],gates=r['projected']['gates'],
                            corrected=r.get('corrected_diagnostics')) for k,r in reports.items()},indent=2),flush=True)


if __name__=='__main__':
    main()
