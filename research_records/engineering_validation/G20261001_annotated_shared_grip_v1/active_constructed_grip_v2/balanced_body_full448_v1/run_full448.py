import json, subprocess, sys
from pathlib import Path
import numpy as np

root=Path.cwd(); run=Path(__file__).resolve().parent; base=run.parent
metadata=json.loads((base/'vposer_body_full448_v1/fit/run_metadata.json').read_text(encoding='utf8'))
static=json.loads((base/'vposer_body_full448_v1/scene/scene_sources.json').read_text(encoding='utf8'))
def execute(name,command):
    (run/(name+'_command.json')).write_text(json.dumps(command,ensure_ascii=False,indent=2),encoding='utf8')
    print('starting',name,flush=True)
    with (run/(name+'.log')).open('w',encoding='utf8') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('completed',name,flush=True)

(run/'protocol.json').write_text(json.dumps(dict(frames=448,steps_per_phase=50,
    source='original immutable VPoser fit, constructed bilateral grasp',
    authorization='user explicitly requests full sequence despite rejected short window; diagnostic run only',
    gates=dict(observation_regression_mm=10,wrist_p95_mm=10,sole_minimum_z_mm=-3,leg_acceleration_not_increased=True,all_hand_geometry_frames_pass=True),
    accepted_for_main_fit=False,no_smoothing=True,no_old_dynamic_inputs=True),indent=2),encoding='utf8')
scene=run/'scene'
command=[sys.executable,str(root/'research_records/engineering_validation/G20260924_smpl_vposer_shared_beta_v1/pipeline/scene/replay_current_run.py'),
    '--left-video',static['raw_left_video'],'--right-video',static['raw_right_video'],
    '--input-pair-dir',str(root/'research_records/engineering_validation/V20260908_people0_1_2_pmpose_c3_chain/people_1/input_448pairs'),
    '--left-pmpose',metadata['inputs']['left_raw'],'--right-pmpose',metadata['inputs']['right_raw'],
    '--calibration',static['calibration'],'--ground-reference',static['static_ground_reference'],
    '--walker-model',static['static_walker_model'],'--output',str(scene)]
execute('scene_replay',command)
source=np.load(metadata['inputs']['source_result']); transforms=np.load(scene/'scene_transforms.npz')
sets=json.loads(Path(metadata['inputs']['contact_vertex_sets']).read_text(encoding='utf8'))['sets']
vg=np.einsum('nij,nvj->nvi',transforms['rotation_ground_from_left'],source['vertices'])+transforms['translation_ground_from_left_mm'][:,None]/1000
height=np.stack([vg[:,np.unique(sum(sets[s+'_sole_surface_candidate'].values(),[])),2].min(1) for s in ('left','right')],1)
np.savez_compressed(run/'current_foot_candidates.npz',foot_contact_weight=(1/(1+np.exp((np.abs(height)-.03)/.015))).astype(np.float32),
    baseline_surface_min_z=height,pair_id=np.arange(448),provenance='immutable reference surface transformed by this run fresh scene; engineering candidate only')
inputs=dict(metadata['inputs']); inputs.update(walker_model=str(scene/'static_walker_model.json'),scene_transforms=str(scene/'scene_transforms.npz'),
    contact_labels=str(run/'current_foot_candidates.npz'),output_dir=str(run/'fit'))
command=[sys.executable,str(root/'realtime_app/tools/refine_body_with_constructed_grasp.py')]
for key,path in inputs.items(): command += ['--'+key.replace('_','-'),path]
command += ['--full-body','--surface-refine','--balanced-stages','--stage2-static-assumption',
    '--start','0','--stop','448','--upper-steps','50','--body-steps','50','--body-polish-steps','50','--hand-polish-steps','50',
    '--lr','.003','--wrist-weight','20','--orientation-weight','.1','--collision-refresh','25']
execute('fit',command)
execute('hand_geometry_audit',[sys.executable,str(root/'realtime_app/tools/audit_constructed_grasp_body.py'),
    '--result',str(run/'fit/result.npz'),'--walker-model',str(scene/'static_walker_model.json'),'--output',str(run/'fit/grasp_geometry_audit.json')])
execute('viewer_build',[sys.executable,str(root/'realtime_app/tools/build_grasp_body_canvas_viewer.py'),
    '--result',str(run/'fit/result.npz'),'--scene',str(scene),'--surface-sets',inputs['contact_vertex_sets'],
    '--geometry-audit',str(run/'fit/grasp_geometry_audit.json'),'--output',str(run/'body_viewer.html')])
execute('viewer_build_final',[sys.executable,str(root/'realtime_app/tools/build_grasp_body_canvas_viewer.py'),
    '--result',str(run/'fit/result.npz'),'--scene',str(scene),'--surface-sets',inputs['contact_vertex_sets'],
    '--geometry-audit',str(run/'fit/grasp_geometry_audit.json'),'--output',str(run/'body_viewer_v2.html')])
execute('assessment',[sys.executable,str(run/'assess_full448.py')])
