"""Same-frame leave-pair counterfactual using the actual recorded predicted P."""
from pathlib import Path
from dataclasses import replace
import argparse
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation

REPO=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(REPO/'realtime_app'),str(REPO/'realtime_app/tools')]
from pose_app.fisheye_camera import load_stereo_fisheye
from pose_app.smplh_motion_tracker import SMPLHObservationMap, SMPLHMotionConfig, MotionPose
from pose_app.smplh_iterated_motion_tracker import SMPLHIteratedMotionTracker, IteratedMAPConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', choices=('leave-pair', 'event-budget'), default='leave-pair')
    args = parser.parse_args()
    out=Path(__file__).resolve().parent
    run=out.parent/'first_frame_sapiens_20261015/replay_selected_init/map_sigma075'
    p=json.loads((run/'run_parameters.json').read_text());inputs=p['inputs']
    model=SMPLHObservationMap(Path(p['effective_initial_fit']),
        *[Path(inputs[k]) for k in ('smplh_model','regressor','vposer_dir','mano_left','mano_right')],device='cuda')
    config=SMPLHMotionConfig(**p['configuration'])
    opt=IteratedMAPConfig(**p['iterated_map_configuration'])
    cal=load_stereo_fisheye(Path(inputs['calibration_dir']))
    with np.load(out/'motion_chain_snapshots.npz') as z:snap={k:z[k].copy() for k in z.files}
    with np.load(out.parent/'first_frame_sapiens_20261015/visualization/scene/scene_transforms.npz') as z:
        rotations=z['rotation_ground_from_left'].copy();translations=z['translation_ground_from_left_mm'].copy()/1000
    policies = ({'all_budget80': ()} if args.probe == 'event-budget' else
                {'all': (), 'leave_hips': (6, 7), 'leave_shoulders': (0, 1),
                 'leave_knees': (8, 9), 'leave_right_knee': (9,),
                 'leave_left_knee': (8,), 'leave_ankles': (10, 11)})
    report={'source_run':str(run.resolve()),'snapshot':str((out/'motion_chain_snapshots.npz').resolve()),
        'policy':'Diagnostic only. Saved MAP15 leave-pair probes plus one MAP80 all-observation probe per event frame; beta,whole VPoser,full temporal P prior unchanged; no production change',
        'limits':['Not a production observation gate or proposal to remove high-error joints',
                  'All24 raw residuals and original objective retained for every mask;mask objective decrease is not accuracy',
                  'One-frame counterfactual does not quantify causal whole-sequence changes',
                  'No independent height/3D truth;triangulation never used in optimization'], 'frames':[]}
    previous=out/'leave_pair_fixed_prior_diagnostic.json'
    if previous.exists():
        report['frames']=json.loads(previous.read_text())['frames']
    for frame in ((37, 59) if args.probe == 'event-budget' else (35, 61, 37, 59)):
        key=str(frame)+'_';pixels=snap[key+'pixels'];original_valid=snap[key+'valid']
        prior=MotionPose(snap[key+'prior_t'],snap[key+'prior_R'],snap[key+'prior_z'])
        P=snap[key+'P'];precision=np.linalg.inv(P[:38,:38]);R,t=rotations[frame],translations[frame]
        frame_row=next((r for r in report['frames'] if r['frame_index']==frame),None)
        if frame_row is None:
            frame_row={'frame_index':frame,'policies':{}};report['frames'].append(frame_row)
        for policy,removed in policies.items():
            tracker=SMPLHIteratedMotionTracker(model,cal,config,
                replace(opt,maximum_iterations=80) if policy=='all_budget80' else opt)
            tracker.pose=prior.copy();tracker.covariance=P.copy()
            tracker.last_accepted_pose=prior.copy()
            mask=original_valid.copy();mask[:,removed]=False
            def describe(pose):
                data=model.evaluate(pose)
                residual=tracker._pixels(data['coco_m'])[0]-pixels[...,:2]
                norm=np.linalg.norm(residual,axis=-1)
                costs=np.where(norm<=config.huber_px,.5*norm**2,config.huber_px*(norm-.5*config.huber_px))/config.pixel_sigma_px**2
                delta=np.r_[pose.translation_m-prior.translation_m,
                    Rotation.from_matrix(prior.root_rotation.T@pose.root_rotation).as_rotvec(),pose.latent-prior.latent]
                temporal=.5*delta@precision@delta
                vposer=.5*(pose.latent@pose.latent)/config.latent_prior_sigma**2
                return {'all24_error_median_p95_px':[float(np.median(norm)),float(np.percentile(norm,95))],
                    'all24_error_px':norm.tolist(),'original_used_data_cost':float((costs*original_valid).sum()),
                    'retained_used_data_cost':float((costs*mask).sum()),'temporal_cost':float(temporal),'vposer_cost':float(vposer),
                    'original_objective':float((costs*original_valid).sum()+temporal+vposer),
                    'retained_objective':float((costs*mask).sum()+temporal+vposer),
                    'pair_data_costs':{name:float((costs[:,idx]*original_valid[:,idx]).sum()) for name,idx in
                        {'shoulders':(0,1),'elbows':(2,3),'wrists':(4,5),'hips':(6,7),'knees':(8,9),'ankles':(10,11)}.items()},
                    'ankle_height_mm':((data['coco_m'][0,15:17]@R.T+t)[:,2]*1000).tolist(),
                    'native_pelvis_height_mm':float((data['joints_m'][0,0]@R.T+t)[2]*1000)}
            before=describe(prior)
            accepted,used,status=tracker._update(pixels,mask,np.full((2,12),'available',object))
            after=describe(tracker.pose)
            row={'accepted':bool(accepted),'status':status,'used_mask':used.tolist(),'before':before,'after':after,
                'maximum_iterations':tracker.optimizer_config.maximum_iterations,'optimizer':tracker.optimizer_diagnostic}
            if policy=='all':
                row['reproduction_max_q_abs_difference']=float(max(np.max(abs(tracker.pose.translation_m-snap[key+'post_t'])),
                    np.max(abs(tracker.pose.root_rotation-snap[key+'post_R'])),np.max(abs(tracker.pose.latent-snap[key+'post_z']))))
                assert row['reproduction_max_q_abs_difference']==0
            frame_row['policies'][policy]=row
            (out/'leave_pair_fixed_prior_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
            print(frame,policy,'height',np.round(after['ankle_height_mm'],3),'origobj',round(after['original_objective'],4),
                'fullraw',np.round(after['all24_error_median_p95_px'],3),'repro',row.get('reproduction_max_q_abs_difference'),flush=True)

if __name__=='__main__':main()
