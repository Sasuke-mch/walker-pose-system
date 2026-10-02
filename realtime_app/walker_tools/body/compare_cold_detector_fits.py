"""Audit identical cold-fit settings and compare full448 engineering outputs."""

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
_tool_prepare_imports()
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'realtime_app'))
from pose_app.independent_wrist_reference import load_reference


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def stats(values):
    return dict(median=float(np.median(values)), p95=float(np.percentile(values,95)),
                max=float(np.max(values)), rms=float(np.sqrt(np.mean(values**2))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run, output = a.run.resolve(), a.output.resolve()
    if output.exists(): raise FileExistsError(output)
    names = dict(sapiens2='smplh_cold_start_v2_joint300',pmpose='pmpose_cold_start_v1_joint300')
    commands = {k:read(run/v/'fit_command.json') for k,v in names.items()}
    ca, cb = commands.values()
    if len(ca)!=len(cb): raise ValueError('command length differs')
    changes=[dict(argument=ca[i-1],sapiens2=x,pmpose=y) for i,(x,y) in enumerate(zip(ca,cb)) if x!=y]
    if {c['argument'] for c in changes}!={'--left-raw','--right-raw','--output-dir'}:
        raise ValueError('fit controls differ')
    results = {k:np.load(run/v/'fit/result.npz',allow_pickle=False) for k,v in names.items()}
    summaries = {k:read(run/v/'fit/fit_summary.json') for k,v in names.items()}
    if summaries['sapiens2']['stage_schedule']!=summaries['pmpose']['stage_schedule']:
        raise ValueError('stage schedules differ')
    metas = {k:read(run/v/'fit/run_metadata.json') for k,v in names.items()}
    target, _ = load_reference(metas['sapiens2']['inputs']['wrist_reference'],
                              metas['sapiens2']['inputs']['calibration_dir'],True)
    if metas['sapiens2']['inputs']['wrist_reference']!=metas['pmpose']['inputs']['wrist_reference']:
        raise ValueError('wrist inputs differ')
    errors, metrics = {}, {}
    sole_sets=read(ROOT/'research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/contact_surface_sets_smplh_v1/contact_vertex_sets.json')['sets']
    scenes={k:np.load(run/k/'scene/scene_transforms.npz',allow_pickle=False) for k in names}
    for k,v in names.items():
        z=results[k]
        init=np.load(run/v/'fit/cold_initialization.npz',allow_pickle=False)
        if not np.array_equal(z['pair_id'],np.arange(448)) or np.any(init['betas']) or np.any(init['vposer_latent']):
            raise ValueError('not full448 cold initialization')
        errors[k]=np.linalg.norm(z['smplh_joints'][:,[20,21]]-target,axis=-1)*1000
        own=np.linalg.norm(z['predicted_coco']-z['raw_triangulated_points']/1000,axis=-1)*1000
        mano=read(run/v/'fit/wilor_mano_parameter_audit.json')
        metrics[k]=dict(wrist_reference_mm={s:stats(errors[k][:,j]) for j,s in enumerate(('left','right'))},
            own_body_observation_rms_mm=float(np.sqrt(np.mean(own[z['body_accepted']]**2))),
            weighted_body_rms_mm=summaries[k]['history'][-1]['body_m']*1000,
            betas=z['betas'].tolist(),body_observation_count=int(z['body_accepted'].sum()),
            native_mano_coverage=mano['pca_reconstruction'],
            hand_geometry=read(run/v/'fit/grasp_geometry_audit.json')['summary'])
        metrics[k]['sole_minimum_z_mm_by_transform']={}
        for frame_source,scene in scenes.items():
            grounded=np.einsum('nij,nvj->nvi',scene['rotation_ground_from_left'],z['vertices'])+scene['translation_ground_from_left_mm'][:,None]/1000
            metrics[k]['sole_minimum_z_mm_by_transform'][frame_source]={s:float(grounded[:,np.unique(sum(sole_sets[s+'_sole_surface_candidate'].values(),[])),2].min()*1000) for s in ('left','right')}
        # Cross-observation residuals are sensitivity diagnostics, not accuracy.
        metrics[k]['cross_observation_rms_mm']={}
        for reference,rz in results.items():
            mask=results['sapiens2']['body_accepted'] & results['pmpose']['body_accepted']
            d=np.linalg.norm(z['predicted_coco']-rz['raw_triangulated_points']/1000,axis=-1)*1000
            metrics[k]['cross_observation_rms_mm'][reference]=float(np.sqrt(np.mean(d[mask]**2)))
    output.mkdir(parents=True)
    # Record association differences explicitly instead of assuming same file
    # means same accepted hand hypotheses.
    associations={k:read(run/v/'fit/wilor_mano_parameter_audit.json') for k,v in names.items()}
    selected={}
    for camera in ('left_camera','right_camera'):
        def accepted(k):
            return {(r['frame_index'],r['hand'],r['record_index']) for r in associations[k][camera] if r.get('accepted')}
        aa,bb=accepted('sapiens2'),accepted('pmpose')
        selected[camera]=dict(common=len(aa&bb),sapiens2_only=len(aa-bb),pmpose_only=len(bb-aa),
            sapiens2_only_records=sorted(aa-bb),pmpose_only_records=sorted(bb-aa))
    report=dict(control_audit=dict(passed=True,command_differences=changes,stage_schedule_equal=True,
        beta_and_latent_zero_for_both=True,frames=448), metrics=metrics,association_differences=selected,
        paired_wrist_comparison={s:dict(sapiens2_lower_frames=int((errors['sapiens2'][:,j]<errors['pmpose'][:,j]).sum()),
            pmpose_lower_frames=int((errors['pmpose'][:,j]<errors['sapiens2'][:,j]).sum()),
            pmpose_minus_sapiens2_mm=stats(errors['pmpose'][:,j]-errors['sapiens2'][:,j])) for j,s in enumerate(('left','right'))},
        limits='Same algorithm and settings; detector changes initialization, weights, native association and fresh scene. Own observation RMS is not a shared truth. Manual wrist reference failed original geometry gates. No detector accuracy or physical contact conclusion.',
        accepted_for_main_fit=False)
    (output/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    with (output/'paired_wrist_errors.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['pair_id','sapiens_left_mm','sapiens_right_mm','pmpose_left_mm','pmpose_right_mm'])
        for i in range(448):w.writerow([i,*errors['sapiens2'][i],*errors['pmpose'][i]])
    page='''<!doctype html><meta charset="utf-8"><title>SMPL-H 对照</title>
<style>body{margin:0;font:14px sans-serif;background:white;color:black}header{padding:10px}main{display:flex;height:calc(100vh - 110px)}iframe{width:50%;border:1px solid black}input{width:65%}</style>
<header><b>SMPL-H 对照 · 448 帧</b><p>左：Sapiens2　右：PMPose</p><button id="play">播放</button> <input id="frame" aria-label="同步帧" type="range" min="0" max="447" value="0"> <span id="number">0 / 447</span><div id="metrics"></div></header>
<main><iframe id="sap" title="Sapiens2" src="../smplh_cold_start_v2_joint300/body_viewer.html"></iframe><iframe id="pm" title="PMPose" src="../pmpose_cold_start_v1_joint300/body_viewer.html"></iframe></main>
<script>let timer=null;const slider=document.getElementById('frame');function sync(){const info=[];for(const id of ['sap','pm']){const d=document.getElementById(id).contentDocument,s=d&&d.getElementById('slider');if(s){s.value=slider.value;s.dispatchEvent(new Event('input',{bubbles:true}));info.push((id==='sap'?'Sapiens2':'PMPose')+' 腕 '+d.getElementById('wrist').textContent+'；脚底 '+d.getElementById('fr').textContent);}}document.getElementById('number').textContent=slider.value+' / 447';document.getElementById('metrics').textContent=info.join(' | ');}slider.oninput=sync;for(const id of ['sap','pm'])document.getElementById(id).onload=()=>{const d=document.getElementById(id).contentDocument,style=d.createElement('style');style.textContent='.side,.bottom,.top p{display:none}.main{grid-template-columns:1fr}.top h1{font-size:13px}.top{padding:6px;gap:4px}.badge{font-size:10px}.app{grid-template-rows:auto 1fr}';d.head.append(style);sync();};document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.getElementById('play').textContent='播放';}else{timer=setInterval(()=>{slider.value=(Number(slider.value)+1)%448;sync();},1000/30);document.getElementById('play').textContent='暂停';}};</script>'''
    (output/'index.html').write_text(page,encoding='utf8')
    (output/'compare.html').write_text('<!doctype html><meta charset="utf-8"><title>SMPL-H 对照</title><meta http-equiv="refresh" content="0;url=index.html"><a href="index.html">打开对照</a>',encoding='utf8')
    print(json.dumps(metrics,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
