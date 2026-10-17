"""Decode native PHC states back to the SAME frozen SMPL-H and raw fisheye pixels."""

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .contract import BASE_ROTATION, SMPL_INDICES
from .reference import BODY_PARENTS
from pose_app.fisheye_camera import load_stereo_fisheye, fisheye_project_numpy
from pose_app.smpl_coco_observation import _install_legacy_smpl_pickle_compatibility
from pose_app.body_observations import load_ordered_predictions as _load_predictions
from tools.plot_body_joint_motion_curves import three_point_derivatives

ROOT = Path(__file__).resolve().parents[2]


def stats(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return dict(count=len(values), median=float(np.median(values)), mean=float(values.mean()),
                p95=float(np.percentile(values, 95)), max=float(values.max())) if len(values) else dict(count=0)


def smpl_rotations(physics_rotation):
    global_smpl = physics_rotation[:, np.argsort(SMPL_INDICES)] @ BASE_ROTATION
    root = global_smpl[:, 0]
    local = np.stack([global_smpl[:, BODY_PARENTS[j]].transpose(0, 2, 1) @ global_smpl[:, j]
                      for j in range(1, 22)], axis=1)
    return root, local


def load_model():
    import smplx
    from smplx.utils import Struct
    _install_legacy_smpl_pickle_compatibility()
    path = ROOT / 'third_party/WiLoR/mano_data/models/SMPLH_male.pkl'
    with path.open('rb') as stream:
        data = pickle.load(stream, encoding='latin1')
    for side, name in (('l', 'LEFT'), ('r', 'RIGHT')):
        with path.with_name(f'MANO_{name}.pkl').open('rb') as stream:
            mano = pickle.load(stream, encoding='latin1')
        for key in ('hands_components', 'hands_mean'):
            data[key+side] = np.asarray(mano[key], np.float32)
    model = smplx.SMPLHLayer(str(path), data_struct=Struct(**data), gender='male',
                            use_pca=False, flat_hand_mean=True).eval()
    model.requires_grad_(False)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='separate analysis directory; keep old comparisons unchanged')
    parser.add_argument('--runs', nargs='+', default=['replay448_physx_recovery', 'replay448_physx_support'])
    args = parser.parse_args()
    destination = args.root if args.output is None else args.output
    destination.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    model = load_model()
    with np.load(args.source, allow_pickle=False) as z:
        beta = torch.tensor(z['betas'].copy(), dtype=torch.float32)
        hands = [torch.tensor(Rotation.from_rotvec(z[k][0].reshape(15, 3)).as_matrix()[None], dtype=torch.float32)
                 for k in ('left_hand_pose', 'right_hand_pose')]
        source_coco_camera = z['predicted_coco'].copy()
        source_available = z['pose_available'].copy()
        source_world_vertices = z['world_vertices'].copy()
    with np.load(args.reference, allow_pickle=False) as z:
        reference = {k: z[k] for k in ('timestamp_s', 'camera_rotation', 'camera_translation_m', 'stage', 'segment_id',
                                       'reference_rotation', 'reference_position_m')}
    rest = model.v_template + torch.einsum('bl,vcl->bvc', beta, model.shapedirs)[0]
    pelvis_offset = (model.J_regressor[0] @ rest).detach().numpy()
    regressor = np.load(ROOT / 'models/smpl/J_regressor_coco.npy')
    people = ROOT / 'research_records/engineering_validation/V20260908_people0_1_2_pmpose_c3_chain/people_1'
    observations = np.stack([_load_predictions(people/f'c3_predictions/{side}/pmpose/raw_predictions.json', side)
                             for side in ('left', 'right')], axis=1)[:, :, 5:17]
    valid = (np.isfinite(observations).all(-1) & (observations[..., 2] >= .2) & (observations[..., 2] <= 1)
             & (observations[..., 0] >= 0) & (observations[..., 0] < 1920)
             & (observations[..., 1] >= 0) & (observations[..., 1] < 1080))
    cal = load_stereo_fisheye(ROOT / 'realtime_app/calibration/results')
    def error_for(camera):
        right = camera @ cal.R_cam0_to_cam1.T + cal.T_cam0_to_cam1_mm/1000
        with np.errstate(invalid='ignore', divide='ignore'):
            pixels = np.stack([fisheye_project_numpy(camera, cal.K0, cal.D0),
                               fisheye_project_numpy(right, cal.K1, cal.D1)], axis=1)
        error = np.linalg.norm(pixels - observations[..., :2], axis=-1)
        positive = (camera[..., 2] > 0) & (right[..., 2] > 0)
        error[~(valid & positive[:, None])] = np.nan
        return error
    surface_path = ROOT / 'research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/contact_surface_sets_smplh_v1/contact_vertex_sets.json'
    surfaces = json.loads(surface_path.read_text(encoding='utf-8'))['sets']
    sole = [sorted(set(sum(surfaces[f'{side}_sole_surface_candidate'].values(), []))) for side in ('left', 'right')]
    base_camera = source_coco_camera[:, 5:17]
    base_camera[~source_available] = np.nan
    base_world = np.einsum('nij,nkj->nki', reference['camera_rotation'], base_camera) + reference['camera_translation_m'][:, None]
    all_points, all_errors = {'source_smplh': base_world}, {'source_smplh': error_for(base_camera)}
    all_sole = {'source_smplh': np.stack([source_world_vertices[:, indices, 2].min(axis=1)
                                         for indices in sole], axis=1)}
    result = dict(denominator=int(valid.sum()), source=str(args.source.resolve()),
        reference=str(args.reference.resolve()), interpretation='same-detector2D consistency; no independent3D truth',
        contact_interpretation='SMPL-H candidate sole/heel geometry; no physical force or contact validation', methods={})
    # Check that retargeting's coordinate round trip preserves the original
    # SMPL-H mesh before attributing any pixel change to the policy.
    reference_root, reference_body = smpl_rotations(reference['reference_rotation'])
    roundtrip_max = 0.
    for first in range(0, 448, 32):
        indices = np.flatnonzero(source_available[first:first+32])+first
        if not len(indices):
            continue
        with torch.inference_mode():
            recovered = model(betas=beta.expand(len(indices), -1),
                global_orient=torch.tensor(reference_root[indices], dtype=torch.float32),
                body_pose=torch.tensor(reference_body[indices], dtype=torch.float32),
                transl=torch.tensor(reference['reference_position_m'][indices,0]-pelvis_offset, dtype=torch.float32),
                left_hand_pose=hands[0].expand(len(indices), -1, -1, -1),
                right_hand_pose=hands[1].expand(len(indices), -1, -1, -1)).vertices.numpy()
        roundtrip_max = max(roundtrip_max, float(np.max(np.abs(recovered-source_world_vertices[indices]))))
    if roundtrip_max > 8e-6:
        raise ValueError(f'SMPL-H reference roundtrip changed the original mesh: {roundtrip_max}m')
    result['reference_mesh_roundtrip_max_coordinate_m'] = roundtrip_max
    for run in args.runs:
        directory = args.root / run
        with np.load(directory/'rollout.npz', allow_pickle=False) as z:
            available = z['available'].copy()
            rotation = z['rotation_world'].copy()
            pelvis = z['position_m'][:, 0].copy()
            ids, time_s = z['pair_id'].copy(), z['timestamp_s'].copy()
        if ids.tolist() != list(range(448)):
            raise ValueError('controller frame identity mismatch')
        np.testing.assert_array_equal(time_s, reference['timestamp_s'])
        vertices = np.full((448, 6890, 3), np.nan, dtype=np.float32)
        root, body = np.full((448, 3, 3), np.nan), np.full((448, 21, 3, 3), np.nan)
        root[available], body[available] = smpl_rotations(rotation[available])
        translation = pelvis - pelvis_offset
        for first in range(0, 448, 32):
            indices = np.flatnonzero(available[first:first+32])+first
            if not len(indices):
                continue
            with torch.inference_mode():
                output = model(betas=beta.expand(len(indices), -1),
                    global_orient=torch.tensor(root[indices], dtype=torch.float32),
                    body_pose=torch.tensor(body[indices], dtype=torch.float32),
                    transl=torch.tensor(translation[indices], dtype=torch.float32),
                    left_hand_pose=hands[0].expand(len(indices), -1, -1, -1),
                    right_hand_pose=hands[1].expand(len(indices), -1, -1, -1))
            np.testing.assert_allclose(output.joints[:, 0].numpy(), pelvis[indices], atol=2e-6, rtol=0)
            vertices[indices] = output.vertices.numpy()
        coco = np.einsum('jv,nvc->njc', regressor, vertices)
        camera = np.einsum('nji,nkj->nki', reference['camera_rotation'],
                           coco-reference['camera_translation_m'][:, None])
        all_errors[run] = error_for(camera[:, 5:17])
        all_points[run] = coco[:, 5:17]
        height = np.stack([vertices[:, indices, 2].min(axis=1) for indices in sole], axis=1)
        all_sole[run] = height
        heels = vertices[:, [3387, 6787]]
        anchor = None
        heel_displacement = np.full((448, 2), np.nan)
        for i, stage in enumerate(reference['stage']):
            if 'stage1' in stage:
                anchor = None
            if 'stage2' in stage:
                if anchor is None:
                    anchor = heels[i].copy()
                heel_displacement[i] = np.linalg.norm(heels[i]-anchor, axis=-1)
        np.savez_compressed(directory/'decoded_smplh.npz', pair_id=ids, timestamp_s=time_s,
            world_vertices=vertices, faces=model.faces, world_predicted_coco=coco,
            world_root_rotation=root, body_rotations=body, world_transl=translation,
            betas=beta.numpy(), pose_available=available, sole_min_height_m=height,
            heel_position_world_m=heels, stage2_heel_episode_displacement_m=heel_displacement,
            reprojection_error_px=all_errors[run])
        result.setdefault('stage2_heel_episode_displacement_mm', {})[run] = stats(heel_displacement*1000)
        stage2=np.array(['stage2' in stage for stage in reference['stage']])
        pass_heel=np.isfinite(heel_displacement).all(1)&(heel_displacement<=.01).all(1)
        result.setdefault('stage2_both_heels_within10mm',{})[run]=dict(total_frames=int(stage2.sum()),
            passed_frames=int((stage2&pass_heel).sum()),failed_frames=int((stage2&~pass_heel).sum()))
    derivatives = {}
    for name, error in all_errors.items():
        within = valid & np.isfinite(error) & (error <= 25)
        method = dict(raw_reprojection_px=stats(error),
            full_sequence_fraction_within25=float(within.sum()/valid.sum()),
            per_joint_median_px=[stats(error[:, :, j]) for j in range(12)])
        if name in all_sole:
            method['candidate_sole_height_mm'] = stats(all_sole[name]*1000)
            method['candidate_sole_below_minus5mm_foot_frames'] = int(np.sum(all_sole[name] < -.005))
        finite = np.isfinite(all_points[name]).all(-1)
        # Analysis only: centered three-sample derivatives use the next sample.
        # They are never fed to the policy and are not causal instantaneous a.
        velocity, acceleration = three_point_derivatives(reference['timestamp_s'], all_points[name], finite)
        breaks = np.flatnonzero(np.diff(reference['segment_id']) != 0)+1
        for boundary in breaks:
            velocity[max(0,boundary-1):boundary+1] = np.nan
            acceleration[max(0,boundary-1):boundary+1] = np.nan
        derivatives[name] = (velocity, acceleration)
        method['sampled_speed_m_s'] = stats(np.linalg.norm(velocity, axis=-1))
        method['sampled_acceleration_m_s2'] = stats(np.linalg.norm(acceleration, axis=-1))
        result['methods'][name] = method
    result['derivatives'] = 'offline irregular-time centered3point; NaN endpoints/segment boundaries;not simulation instantaneous dynamics or causal deployment'
    result['decision'] = 'pretrained integration completed; not promoted: support needed for stability, worse raw2D fit, heel-fixed requirement unmet'
    (destination/'analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    colors = {name: f'C{i}' for i, name in enumerate(all_errors)}
    for name, error in all_errors.items():
        median = np.full(448, np.nan)
        some = np.isfinite(error).any(axis=(1,2))
        median[some] = np.nanmedian(error[some], axis=(1,2))
        axes[0].plot(reference['timestamp_s'], median, label=name, color=colors[name])
    axes[0].axhline(25, color='black', linestyle='--', linewidth=.7)
    axes[0].set_ylabel('Raw reprojection median (px)')
    axes[0].legend(fontsize=8)
    for name, height in all_sole.items():
        axes[1].plot(reference['timestamp_s'], height.min(axis=1)*1000, label=name, color=colors[name])
    axes[1].axhline(0, color='black', linewidth=.7)
    axes[1].set_ylabel('SMPL-H candidate sole min (mm)')
    axes[1].set_xlabel('Capture time (s)')
    axes[1].legend(fontsize=8)
    figure.suptitle('Frozen PHC+ complete448: detector consistency and decoded surface geometry')
    figure.tight_layout()
    figure.savefig(destination/'comparison.png', dpi=150)
    plt.close(figure)
    figure, axes = plt.subplots(4, 2, figsize=(13, 10), sharex=True)
    for row, (label, indices) in enumerate((('Left arm',[0,2,4]), ('Right arm',[1,3,5]),
                                           ('Left leg',[6,8,10]), ('Right leg',[7,9,11]))):
        for name, (velocity, acceleration) in derivatives.items():
            for col, values in enumerate((velocity, acceleration)):
                axes[row,col].plot(reference['timestamp_s'], np.linalg.norm(values[:,indices],axis=-1).mean(axis=1), label=name, color=colors[name])
        axes[row,0].set_ylabel(label+' speed (m/s)')
        axes[row,1].set_ylabel(label+' accel (m/s2)')
    axes[0,0].legend(fontsize=7)
    axes[-1,0].set_xlabel('Capture time (s)')
    axes[-1,1].set_xlabel('Capture time (s)')
    figure.suptitle('Raw sampled derivatives, offline3point: no smoothing; gaps retained')
    figure.tight_layout()
    figure.savefig(destination/'motion_curves.png', dpi=150)
    plt.close(figure)
    print(json.dumps({name:{k:v for k,v in value.items() if k != 'per_joint_median_px'}
                      for name,value in result['methods'].items()}), flush=True)


if __name__ == '__main__':
    main()
