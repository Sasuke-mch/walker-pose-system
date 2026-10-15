import json,sys,pathlib,time
import numpy as np
root=pathlib.Path('research_records/engineering_validation/G20261014_smplh_motion448_v1/slope_motion_20261015')
sys.path.insert(0,str(root.resolve())); import prefix_derivative_feedback as x
from pose_app.smplh_world_motion_tracker import SMPLHWorldMotionTracker,WorldGroundConfig
from pose_app.smplh_ground_constraints import foot_geometry
p=json.loads((root/'run_contact80_feasible/run_parameters.json').read_text(encoding='utf-8')); inp=p['inputs']; source=root/'run_contact80_feasible'
l=x._load_predictions(pathlib.Path(inp['left_predictions']),'left');r=x._load_predictions(pathlib.Path(inp['right_predictions']),'right')
times,*_=x._read_capture_times(pathlib.Path(inp['input_pairs']),len(l)); x.apply_first_frame_observations(pathlib.Path(p['first_frame_observations']),pathlib.Path(inp['input_pairs']),l,r)
m=x.SMPLHObservationMap(*[pathlib.Path(inp[k]) for k in ['initial_fit','smplh_model','regressor','vposer_dir','mano_left','mano_right']],p['device']);c=x.load_stereo_fisheye(pathlib.Path(inp['calibration_dir'])); soles=x.load_sole_vertex_indices(x.ROOT/'research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/contact_surface_sets_smplh_v1/contact_vertex_sets.json')
s=np.load(root/'scene/scene_transforms.npz'); R=s['rotation_ground_from_left'];t=s['translation_ground_from_left_mm']/1000
stages=[json.loads(v)['stage']['operational'] for v in (root/'scene/stage1_stage2.jsonl').read_text().splitlines()]
tr=SMPLHWorldMotionTracker(m,c,x.SMPLHMotionConfig(**p['configuration']),R[0],t[0],soles,WorldGroundConfig(**p['ground_configuration']))
a=[]; report={'single_variable':'Remove active hard inequality rows from Gaussian posterior;identical pixels/wholeVPoser/PVA/5mm/10mm/80 budget','frames':[0,61],'baseline':'run_contact80_feasible','rows':a,'status':'running'}
start=time.perf_counter()
for i in range(62):
 tr.set_scene_frame(R[i],t[i],stages[i],'same_fresh_scene');out=tr.step(float(times[i]),l[i,5:17],r[i,5:17]); g=foot_geometry(out['vertices_m'],soles);err=np.linalg.norm(out['projected_pixels']-np.array([l[i,5:17,:2],r[i,5:17,:2]]),axis=-1)
 a.append({'frame':i,'error':err.tolist(),'sole_mm':(g['sole_min_m']*1000).tolist(),'mean_ankle_z_mm':float(out['coco_m'][15:17,2].mean()*1000),'pelvis_z_mm':float(out['joints_m'][0,2]*1000),'constraint':out['ground_constraint'],'q_std_mm':(np.sqrt(np.diag(tr.covariance)[:3])*1000).tolist()})
 if i%10==0 or i==61: print(i,'error',np.median(err),'sole',g['sole_min_m']*1000,flush=True)
 (root/'contact_covariance_prefix.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
errs=np.array([v['error'] for v in a]); report.update(status='completed',seconds=time.perf_counter()-start,raw_median_px=float(np.median(errs)),raw_p95_px=float(np.percentile(errs,95)),all_contact=sum(v['constraint']['constraint_satisfied'] for v in a),stage2=sum(v['constraint']['stage2'] for v in a),stage2_satisfied=sum(v['constraint']['stage2'] and v['constraint']['constraint_satisfied'] for v in a),mean_ankle_35_61_rise_mm=a[61]['mean_ankle_z_mm']-a[35]['mean_ankle_z_mm'])
(root/'contact_covariance_prefix.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print({k:v for k,v in report.items() if k!='rows'},flush=True)
