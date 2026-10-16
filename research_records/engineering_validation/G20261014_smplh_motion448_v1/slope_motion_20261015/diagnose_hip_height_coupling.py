"""Read-only local gradient diagnostic; does not change tracking observations."""
from pathlib import Path
import json
import sys
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO/'realtime_app'), str(REPO/'realtime_app/tools')]
from pose_app.fisheye_camera import load_stereo_fisheye
from pose_app.smplh_motion_tracker import SMPLHObservationMap, SMPLHMotionTracker, SMPLHMotionConfig, MotionPose
from run_stereo_pva_baseline import _load_predictions, JOINT_NAMES


def main():
    root = Path(__file__).resolve().parent.parent
    run = root/'first_frame_sapiens_20261015/replay_selected_init/map_sigma075'
    params = json.loads((run/'run_parameters.json').read_text())
    inputs = params['inputs']
    model = SMPLHObservationMap(Path(params['effective_initial_fit']),
        *[Path(inputs[k]) for k in ('smplh_model','regressor','vposer_dir','mano_left','mano_right')], device='cuda')
    config = SMPLHMotionConfig(**params['configuration'])
    tracker = SMPLHMotionTracker(model, load_stereo_fisheye(Path(inputs['calibration_dir'])), config)
    px = np.array([_load_predictions(Path(inputs[f'{side}_predictions']), side) for side in ('left','right')])
    with np.load(run/'result.npz') as z:
        saved = {k:z[k].copy() for k in ('transl','root_rotation','vposer_latent','betas')}
    with np.load(root/'first_frame_sapiens_20261015/visualization/scene/scene_transforms.npz') as z:
        rotations = z['rotation_ground_from_left'].copy()
        translations = z['translation_ground_from_left_mm'].copy()/1000
    trajectory = [json.loads(s) for s in (run/'trajectory.jsonl').read_text().splitlines()]
    groups = {'shoulders':(0,1),'elbows':(2,3),'wrists':(4,5),'hips':(6,7),'knees':(8,9),'ankles':(10,11)}
    report = {'source_run':str(run.resolve()), 'fixed_beta':saved['betas'].tolist(),
        'method':'Fixed saved q; raw fisheye Huber residual/Jacobian; single unit-norm whole-latent steepest descent probe of length0.002 for each observation pair; root/translation/beta unchanged',
        'limits':['Local directional sensitivity does not identify causal temporal posterior coupling',
                  'Probe omits temporal P prior and VPoser penalty; production path unchanged',
                  'Geometry and heights are model estimates in operational Stage ground, not independent truth',
                  'Excluded observations remain in raw residual statistics but not used data gradient'], 'frames':[]}

    def huber(errors):
        norm = np.linalg.norm(errors,axis=-1)
        return np.where(norm<=config.huber_px,.5*norm**2,config.huber_px*(norm-.5*config.huber_px))/config.pixel_sigma_px**2

    for frame in (35,48,61):
        pose = MotionPose(saved['transl'][frame],saved['root_rotation'][frame],saved['vposer_latent'][frame])
        tracker.pose = pose
        data = model.evaluate(pose)
        projected,J,_ = tracker._linearize()
        observed = px[:,frame,5:17,:2]
        residual = projected-observed
        used = np.array(trajectory[frame]['used_views'],bool)
        norm = np.linalg.norm(residual,axis=-1)
        robust = np.minimum(1.,config.huber_px/np.maximum(norm,1e-12))
        costs = huber(residual)
        R,t = rotations[frame],translations[frame]
        heights = (data['coco_m'][0]@R.T+t)[15:17,2]
        # Finite differences for height, using the same perturbation as the pixel Jacobian.
        n,eps = tracker.n,1e-3
        offsets = np.r_[np.eye(n)*eps,-np.eye(n)*eps]
        batch = model.evaluate_batch(pose.translation_m+offsets[:,:3],
            pose.root_rotation@Rotation.from_rotvec(offsets[:,3:6]).as_matrix(),pose.latent+offsets[:,6:])['coco_m']
        h = (batch@R.T+t)[:,15:17,2].mean(axis=1)
        h_jac = (h[:n]-h[n:])/(2*eps)
        row={'frame_index':frame,'ankle_height_mm':(heights*1000).tolist(),
             'used_mask':used.tolist(),'residual_model_minus_detector_raw_xy_px':residual.tolist(),
             'per_joint':[], 'pair_gradient_probes':{}}
        cal=tracker.cal
        normalized=[cv2.fisheye.undistortPoints(observed[view].reshape(1,-1,2), K, D).reshape(-1,2)
            for view,K,D in ((0,cal.K0,cal.D0),(1,cal.K1,cal.D1))]
        homogeneous=cv2.triangulatePoints(np.c_[np.eye(3),np.zeros(3)],
            np.c_[cal.R_cam0_to_cam1,cal.T_cam0_to_cam1_mm/1000],*([p.T for p in normalized]))
        triangulated=(homogeneous[:3]/homogeneous[3]).T
        full_points=np.zeros((1,17,3));full_points[0,5:17]=triangulated
        tri_projected=tracker._pixels(full_points)
        row['unfiltered_triangulation_diagnostic']={
            'not_used_in_fit':True,'positive_left_right_camera_depth':bool(tri_projected is not None),
            'hip_width_mm':float(np.linalg.norm(triangulated[6]-triangulated[7])*1000),
            'model_hip_width_mm':float(np.linalg.norm(data['coco_m'][0,11]-data['coco_m'][0,12])*1000),
            'hip_midpoint_model_minus_triangulation_ground_mm':
                (((data['coco_m'][0,11:13].mean(axis=0)-triangulated[6:8].mean(axis=0))@R.T)*1000).tolist(),
            'tri_raw_reprojection_error_px':None if tri_projected is None else np.linalg.norm(tri_projected[0]-observed,axis=-1).tolist()}
        for joint,name in enumerate(JOINT_NAMES):
            row['per_joint'].append({'name':name,'left_right_error_px':norm[:,joint].tolist(),
                'raw_huber_cost':float(costs[:,joint].sum()),'used_huber_cost':float((costs[:,joint]*used[:,joint]).sum())})
        total=float((costs*used).sum())
        row['used_data_cost']=total
        for name,joints in groups.items():
            mask=np.zeros_like(used);mask[:,joints]=used[:,joints]
            weighted=residual*robust[...,None]*mask[...,None]/config.pixel_sigma_px**2
            gradient=J.T@weighted.ravel()
            g=gradient[6:]
            direction=-g/max(np.linalg.norm(g),1e-30)
            probe=pose.copy();probe.latent+=.002*direction
            probe_data=model.evaluate(probe)
            probe_pixels=tracker._pixels(probe_data['coco_m'])[0]
            probe_heights=(probe_data['coco_m'][0]@R.T+t)[15:17,2]
            new_costs=huber(probe_pixels-observed)
            row['pair_gradient_probes'][name]={
                'used_cost':float((costs*mask).sum()),'used_cost_fraction':float((costs*mask).sum()/total),
                'latent_gradient_norm':float(np.linalg.norm(g)),
                'translation_gradient_norm_per_m':float(np.linalg.norm(gradient[:3])),
                'root_gradient_norm_per_rad':float(np.linalg.norm(gradient[3:6])),
                'predicted_mean_ankle_height_change_mm':float(h_jac[6:]@(.002*direction)*1000),
                'actual_left_right_ankle_height_change_mm':((probe_heights-heights)*1000).tolist(),
                'pair_used_cost_change':float(((new_costs-costs)*mask).sum()),
                'all_used_cost_change':float(((new_costs-costs)*used).sum()),
                'vposer_penalty_change':float((probe.latent@probe.latent-pose.latent@pose.latent)/(2*config.latent_prior_sigma**2))}
        report['frames'].append(row)
    out=Path(__file__).with_name('hip_height_coupling_diagnostic.json')
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    for row in report['frames']:
        print('frame',row['frame_index'],'height',row['ankle_height_mm'],'data cost',row['used_data_cost'])
        for name,v in row['pair_gradient_probes'].items():
            print(name, 'cost_share',round(v['used_cost_fraction'],3),'ankle_delta',np.round(v['actual_left_right_ankle_height_change_mm'],4),'own_cost_delta',round(v['pair_used_cost_change'],4),'all_delta',round(v['all_used_cost_change'],4))

if __name__=='__main__':
    main()
