"""Reproduce a frozen short-window comparison before any full-sequence run."""

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
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference-metadata',type=Path,required=True)
    ap.add_argument('--output-root',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=50)
    args = ap.parse_args()
    if args.output_root.exists() or args.steps <= 0:
        raise ValueError('require new output and positive steps')
    args.output_root.mkdir(parents=True)
    metadata = json.loads(args.reference_metadata.read_text(encoding='utf-8'))
    common = []
    for name,path in metadata['inputs'].items():
        if name not in ('output_dir',):
            common += ['--'+name.replace('_','-'),path]
    common += ['--start','60','--stop','101','--full-body','--surface-refine',
        '--lr','.003','--wrist-weight','20','--orientation-weight','.1',
        '--upper-steps',str(args.steps),'--body-steps',str(args.steps),
        '--body-polish-steps',str(args.steps),'--hand-polish-steps',str(args.steps),
        '--collision-refresh','25']
    protocol = dict(start=60,stop=101,steps_per_phase=args.steps,
        comparison='whole strategy package vs existing route; not a single-factor ablation',
        gates=dict(max_observation_regression_mm=10,wrist_p95_mm=10,sole_minimum_z_mm=-3,
                   leg_acceleration_not_increased=True,all_hand_geometry_frames_pass=True),
        stopping_rule='Do not expand to full448 if any gate fails; retain every rejected update and failed frame')
    (args.output_root/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    reports = {}
    for name,flags in [('control',[]),('balanced',['--balanced-stages','--stage2-static-assumption'])]:
        out = args.output_root/name
        command = [sys.executable,str((Path(__file__).resolve().parents[2] / "tools" / 'refine_body_with_constructed_grasp.py')),*common,*flags,'--output-dir',str(out)]
        (args.output_root/(name+'_command.json')).write_text(json.dumps(command,ensure_ascii=False,indent=2),encoding='utf-8')
        with (args.output_root/(name+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        with (args.output_root/(name+'_audit.log')).open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,str((Path(__file__).resolve().parents[2] / "tools" / 'audit_constructed_grasp_body.py')),
                '--result',str(out/'result.npz'),'--walker-model',metadata['inputs']['walker_model'],
                '--output',str(out/'hand_geometry_audit.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
        summary = json.loads((out/'fit_summary.json').read_text(encoding='utf-8'))
        final = list(summary['checkpoints'].values())[-1]
        z = np.load(out/'result.npz'); guard = np.load(out/'observation_guard_reference.npz')
        error = np.linalg.norm(z['predicted_coco']-z['raw_triangulated_points']/1000,axis=-1)
        valid = guard['valid']; regression = error-guard['baseline_error_m']
        reports[name] = dict(initial=summary['checkpoints']['initial'],final=final,
            max_observation_regression_mm=float(regression[valid].max()*1000),
            hand_audit=json.loads((out/'hand_geometry_audit.json').read_text(encoding='utf-8')))
        if flags:
            transactions = json.loads((out/'update_transactions.json').read_text(encoding='utf-8'))['updates']
            reports[name]['updates'] = {phase:dict(accepted=sum(r['accepted'] for r in transactions if r['stage']==phase),
                rejected=sum(not r['accepted'] for r in transactions if r['stage']==phase)) for phase in {r['stage'] for r in transactions}}
    balanced = reports['balanced']; final = balanced['final']
    gates = dict(observations=balanced['max_observation_regression_mm']<=10.002,
        wrists=all(r['p95']<=10 for r in final['wrist_error_mm'].values()),
        feet=all(v>=-3 for v in final['sole_minimum_z_mm'].values()),
        leg_temporal=final['terms']['leg_ground_acceleration']<=balanced['initial']['terms']['leg_ground_acceleration'])
    # Audit schema is deliberately retained verbatim; evaluate all frame records.
    audit = balanced['hand_audit']
    gates['hands'] = bool(audit['records']) and all(r['passed_geometry_proxy'] for r in audit['records'])
    report = dict(protocol=protocol,runs=reports,gates=gates,passed=all(gates.values()),
        accepted_for_main_fit=False,full448_started=False)
    (args.output_root/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(gates=gates,passed=report['passed']),indent=2))


if __name__=='__main__':
    main()
