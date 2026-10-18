"""Freeze validation selection, then report full448 and temporal heldout scores."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from .image_policy import ImageObservation


def arrays(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def score(error, valid, first, last):
    selected=error[first:last+1]
    mask=valid[first:last+1]
    finite=mask&np.isfinite(selected)
    values=selected[finite]
    return dict(first=first,last=last,denominator=int(mask.sum()),finite_slots=int(finite.sum()),
        median_px=float(np.median(values)) if len(values) else None,
        p95_px=float(np.percentile(values,95)) if len(values) else None,
        all_slots_within25=float(np.sum(finite&(selected<=25))/max(1,mask.sum())))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--training-run',default='direct_image_20261016/train_seed0_complete')
    parser.add_argument('--output-name',default='evaluation',help='new evaluation directory; retain prior failures')
    args=parser.parse_args()
    root=args.root.resolve()
    training=root/args.training_run
    destination=training/args.output_name
    destination.mkdir(exist_ok=False)
    packet=arrays(root/'direct_image_20261016/inference_packet.npz')
    observation=ImageObservation(packet)
    observed=packet['pixels']
    valid=(np.isfinite(observed).all(-1)&(observed[...,2]>=.2)&(observed[...,2]<=1)
        &(observed[...,0]>=0)&(observed[...,0]<1920)&(observed[...,1]>=0)&(observed[...,1]<1080))
    validation={}
    for name in ('bc','ppo'):
        saved=arrays(training/('replay_'+name)/'rollout.npz')
        error=np.full((448,2,12),np.nan)
        # No test pixels/outputs are scored before the checkpoint selection.
        for i in range(256,320):
            if saved['available'][i]:
                points=observation.points(saved['position_m'][i],saved['rotation_world'][i])
                error[i]=np.linalg.norm(observation.project(points,i)-observed[i,:,:,:2],axis=-1)
                # Match the existing evaluator: a joint must be in front of
                # BOTH cameras. Policy rewards intentionally score each view,
                # but validation selection and final reporting share one gate.
                left=(points-packet['camera_translation_m'][i])@packet['camera_rotation'][i]
                right=left@packet['R01'].T+packet['T01_m']
                error[i,:,~((left[:,2]>0)&(right[:,2]>0))]=np.nan
        metrics=score(error,valid,256,319)
        metrics['fallen_frames']=int(np.sum(saved['position_m'][256:320,0,2]<.3))
        validation[name]=metrics
    # Fixed lexicographic stability then coverage/error rule. A failed candidate
    # can be named for reporting but is never promoted as a successful model.
    selected=min(validation,key=lambda n:(validation[n]['fallen_frames'],
        -validation[n]['all_slots_within25'],validation[n]['median_px'] or float('inf')))
    selection=dict(selected_checkpoint=selected,rule='validation-only: falls, all-slot25 fraction, median',
        validation=validation,test_inspected_for_selection=False,training_tuned_after_test=False,
        independent_test=False,interpretation='same video temporal holdout; upstream beta/scene conditional')
    (destination/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    source=json.loads((root/'reference448.json').read_text(encoding='utf-8'))['source']
    command=[sys.executable,'-m','learning_phc.analyze','--root',str(root),'--source',source,
        '--reference',str(root/'corrections_v1/reference_matched.npz'),
        '--runs','corrections_v1/matched_body',args.training_run+'/replay_bc',args.training_run+'/replay_ppo',
        '--output',str(destination)]
    (destination/'analysis_command.json').write_text(json.dumps(command,indent=2),encoding='utf-8')
    with (destination/'analysis.log').open('w',encoding='utf-8') as stream:
        subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    report=json.loads((destination/'analysis.json').read_text())
    result=dict(selection=selection,methods={},promotion=False)
    for name in ('corrections_v1/matched_body',args.training_run+'/replay_bc',args.training_run+'/replay_ppo'):
        run=arrays(root/name/'decoded_smplh.npz')
        state=arrays(root/name/'rollout.npz')
        summary=json.loads((root/name/'summary.json').read_text())
        metrics=dict(full=report['methods'][name],
            splits={key:score(run['reprojection_error_px'],valid,*bounds)
                    for key,bounds in (('train',(0,255)),('validation',(256,319)),('test',(320,447)))},
            finite_frames=summary['finite_frames'],fallen_frames=summary['fallen_finite_frames'],
            first_fallen_frame=next((int(i) for i in np.flatnonzero(state['position_m'][:,0,2]<.3)),None),
            heel10mm=report['stage2_both_heels_within10mm'][name],
            heel_mm=report['stage2_heel_episode_displacement_mm'][name])
        for key,bounds in (('train',(0,255)),('validation',(256,319)),('test',(320,447))):
            metrics['splits'][key]['fallen_frames']=int(np.sum(state['position_m'][bounds[0]:bounds[1]+1,0,2]<.3))
        result['methods'][name]=metrics
        if name.endswith('/replay_bc') or name.endswith('/replay_ppo'):
            phase=name.rsplit('_',1)[1]
            for key in ('median_px','p95_px','all_slots_within25'):
                np.testing.assert_allclose(validation[phase][key],metrics['splits']['validation'][key],rtol=0,atol=2e-3)
    chosen=args.training_run+'/replay_'+selected
    metrics=result['methods'][chosen]
    raw=metrics['full']['raw_reprojection_px']
    result['promotion']=bool(metrics['finite_frames']==448 and metrics['fallen_frames']==0
        and raw['median']<=60.5543424659*.95 and raw['p95']<=159.3022911524*.95
        and metrics['full']['full_sequence_fraction_within25']>=.1376488095)
    result['decision']='continue research, no promotion; diagnose distillation/long-horizon stability before scaling' if not result['promotion'] else 'engineering gate passed; independent validation still needed'
    (destination/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(selected=selected,promotion=result['promotion'],
        scores={n:dict(median=v['full']['raw_reprojection_px']['median'],p95=v['full']['raw_reprojection_px']['p95'],
                       falls=v['fallen_frames'],heldout=v['splits']['test']) for n,v in result['methods'].items()})),flush=True)


if __name__=='__main__':
    main()
