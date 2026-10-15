"""Frozen-output diagnosis; pose substitutions are NOT feasible controller runs."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .analyze import load_model, stats
from .contract import SMPL_INDICES
from .policy import observation_v7, recovery_reference
from pose_app.fisheye_camera import load_stereo_fisheye, fisheye_project_numpy
from tools.run_stereo_pva_baseline import _load_predictions

ROOT = Path(__file__).resolve().parents[2]


def archive(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def diagnose(root):
    info = json.loads((root/'reference448.json').read_text(encoding='utf-8'))
    source_path = Path(info['source'])
    source = archive(source_path)
    reference = archive(root/'reference448.npz')
    parameters = json.loads(source_path.with_name('run_parameters.json').read_text(encoding='utf-8'))
    previous = json.loads((root/'analysis.json').read_text(encoding='utf-8'))
    obs = np.stack([_load_predictions(Path(parameters['inputs'][s+'_predictions']),s)
                    for s in ('left','right')],1)[:,:,5:17]
    cal = load_stereo_fisheye(Path(parameters['inputs']['calibration_dir']))
    r,t = reference['camera_rotation'],reference['camera_translation_m']
    world_source = np.einsum('nij,nvj->nvi',r,source['predicted_coco'])+t[:,None]
    valid = (np.isfinite(obs).all(-1) & (obs[...,2]>=.2) & (obs[...,2]<=1)
             & (obs[...,0]>=0) & (obs[...,0]<1920) & (obs[...,1]>=0) & (obs[...,1]<1080))
    denominator = int(valid.sum())
    if denominator!=previous['denominator']:
        raise ValueError('diagnostic denominator differs from the original experiment')

    def score(world):
        camera = np.einsum('nji,nvj->nvi',r,world-t[:,None])[:,5:17]
        right = camera@cal.R_cam0_to_cam1.T+cal.T_cam0_to_cam1_mm/1000
        pixels = np.stack([fisheye_project_numpy(camera,cal.K0,cal.D0),
                           fisheye_project_numpy(right,cal.K1,cal.D1)],1)
        error = np.linalg.norm(pixels-obs[...,:2],axis=-1)
        positive = (camera[...,2]>0)&(right[...,2]>0)
        error[~(valid&positive[:,None])] = np.nan
        return dict(raw_reprojection_px=stats(error),
            all_slots_within25_fraction=float(np.sum(np.isfinite(error)&(error<=25))/denominator))

    result = dict(scope='saved448 outputs only; no retraining, simulation correction or promotion',
        source=str(source_path),denominator=denominator,
        counterfactual='substitute one saved parameter group with source; not an executable physical trajectory; nonlinear effects are not additive',
        source_score=score(world_source),methods={})
    np.testing.assert_allclose(result['source_score']['raw_reprojection_px']['median'],
                               previous['methods']['source_smplh']['raw_reprojection_px']['median'],atol=1e-6,rtol=0)
    model = load_model()
    beta = torch.tensor(source['betas'],dtype=torch.float32)
    hands = [torch.tensor(Rotation.from_rotvec(source[key][0].reshape(15,3)).as_matrix()[None],dtype=torch.float32)
             for key in ('left_hand_pose','right_hand_pose')]
    rest = model.v_template+torch.einsum('bl,vcl->bvc',beta,model.shapedirs)[0]
    pelvis_offset = (model.J_regressor[0]@rest).detach().numpy()
    regressor = np.load(ROOT/'models/smpl/J_regressor_coco.npy')
    native_reference = reference['reference_position_m']
    source_joints = np.einsum('nij,nvj->nvi',r,source['smplh_joints'][:,:22])+t[:,None]
    native_as_smpl = native_reference[:,np.argsort(SMPL_INDICES)[:22]]
    result['same_rotations_neutral_retarget_joint_difference_mm'] = stats(np.linalg.norm(native_as_smpl-source_joints,axis=-1)*1000)
    result['retarget_joint_median_mm_smpl0to21'] = np.median(np.linalg.norm(native_as_smpl-source_joints,axis=-1)*1000,axis=0).tolist()
    time_s = reference['timestamp_s']
    dt = np.diff(time_s)
    result['host_capture_dt_s'] = stats(dt)
    result['dt_below20ms_count'] = int((dt<.02).sum())
    result['dt_above45ms_count'] = int((dt>.045).sum())
    wrist_camera = (native_reference[0,[17,22]]-t[0])@r[0]
    support_target = np.einsum('nij,kj->nki',r,wrist_camera)+t[:,None]
    result['reference_wrist_vs_assumed_fixed_camera_target_mm'] = stats(np.linalg.norm(native_reference[:,[17,22]]-support_target,axis=-1)*1000)
    target_v = np.diff(support_target,axis=0)/dt[:,None,None]
    result['omitted_moving_target_damping_term_norm_n'] = stats(np.linalg.norm(45*target_v,axis=-1))

    for name in ('replay448_physx_recovery','replay448_physx_support'):
        simulated = archive(root/name/'rollout.npz')
        decoded = archive(root/name/'decoded_smplh.npz')
        delta = simulated['position_m']-native_reference
        root_error = np.linalg.norm(delta[:,0],axis=-1)
        relative_error = np.linalg.norm(delta-delta[:,0,None],axis=-1)
        # Recovery observes pre-step p_{i-1} against current reference_i.
        recovery_distance = np.linalg.norm(simulated['position_m'][:-1,0]-native_reference[1:,0],axis=-1)
        entry = dict(root_error_mm=stats(root_error*1000),
            root_relative_mean_body_error_mm=stats(relative_error.mean(1)*1000),
            far_recovery_observation_count=int((recovery_distance>.25).sum()),
            action_component_clipped_fraction=float(np.mean(abs(simulated['action'][1:])>=.999)),
            actual_score=score(decoded['world_predicted_coco']),counterfactuals={})
        np.testing.assert_allclose(entry['actual_score']['raw_reprojection_px']['median'],
            previous['methods'][name]['raw_reprojection_px']['median'],atol=1e-6,rtol=0)
        for fixed in ('root_position','root_rotation','body_rotation'):
            root_rotation = source['world_root_rotation'] if fixed=='root_rotation' else decoded['world_root_rotation']
            body_rotation = source['body_rotations'] if fixed=='body_rotation' else decoded['body_rotations']
            pelvis = native_reference[:,0] if fixed=='root_position' else simulated['position_m'][:,0]
            coco = np.full((448,17,3),np.nan)
            for start in range(0,448,32):
                end = min(start+32,448)
                with torch.inference_mode():
                    output = model(betas=beta.expand(end-start,-1),
                        global_orient=torch.tensor(root_rotation[start:end],dtype=torch.float32),
                        body_pose=torch.tensor(body_rotation[start:end],dtype=torch.float32),
                        transl=torch.tensor(pelvis[start:end]-pelvis_offset,dtype=torch.float32),
                        left_hand_pose=hands[0].expand(end-start,-1,-1,-1),
                        right_hand_pose=hands[1].expand(end-start,-1,-1,-1))
                np.testing.assert_allclose(output.joints[:,0].numpy(),pelvis[start:end],atol=2e-6,rtol=0)
                coco[start:end] = np.einsum('jv,nvc->njc',regressor,output.vertices.numpy())
            entry['counterfactuals']['restore_'+fixed] = score(coco)
        result['methods'][name] = entry
        print(name,json.dumps(entry,ensure_ascii=False),flush=True)
    result['interpretation'] = 'detector consistency and saved geometry; no independent3D truth; wrist force and timing causal effects need separate simulation ablation'
    output = root/'diagnosis.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


def observation_diagnosis(root):
    """Reconstruct actual pre-step observations; no additional policy rollout."""
    reference = archive(root/'reference448.npz')
    observations = {}
    for name in ('replay448_physx_recovery','replay448_physx_support'):
        simulated = archive(root/name/'rollout.npz')
        rows = []
        for i in range(1,448):
            p = simulated['position_m'][i-1]
            v = simulated['velocity_m_s'][i-1]
            target, velocity = recovery_reference(p,v,reference['reference_position_m'][i],
                                                  reference['reference_velocity_m_s'][i])
            rows.append(observation_v7(p,simulated['rotation_world'][i-1],v,
                simulated['angular_velocity_rad_s'][i-1],target,velocity))
        observations[name] = np.asarray(rows)
    result = {name:{} for name in observations}
    for label,filename in (('primitive','phc_kp_2.pth'),('composer','phc_comp_kp_2.pth')):
        checkpoint = torch.load(ROOT/'models/phc'/filename,map_location='cpu',weights_only=False)
        norm = checkpoint['running_mean_std']
        mean,var = norm['running_mean'].float().numpy(),norm['running_var'].float().numpy()
        for name,obs in observations.items():
            clipped = abs((obs-mean)/np.sqrt(var+1e-5))>5
            result[name][label] = dict(component_fraction=float(clipped.mean()),
                self_component_fraction=float(clipped[:,:358].mean()),
                task_component_fraction=float(clipped[:,358:].mean()),
                frames_with_any_clipping=int(clipped.any(1).sum()))
        del checkpoint
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--observations-only',action='store_true',help='append normalized-input diagnosis to existing results')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.observations_only:
        path = args.root/'diagnosis.json'
        result = json.loads(path.read_text(encoding='utf-8'))
        result['normalized_observation_clipping'] = observation_diagnosis(args.root)
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(result['normalized_observation_clipping']),flush=True)
    else:
        if (args.root/'diagnosis.json').exists():
            raise FileExistsError(args.root/'diagnosis.json')
        diagnose(args.root.resolve())
