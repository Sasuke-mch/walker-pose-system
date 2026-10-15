"""Same-frame contact counterfactual; never replaces a saved trajectory."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO / 'realtime_app'), str(REPO / 'realtime_app/tools')]

from pose_app.fisheye_camera import load_stereo_fisheye
from pose_app.smplh_ground_constraints import foot_geometry, load_sole_vertex_indices
from pose_app.smplh_motion_tracker import MotionPose, SMPLHMotionConfig, SMPLHObservationMap
from pose_app.smplh_world_motion_tracker import SMPLHWorldMotionTracker, WorldGroundConfig
from run_stereo_pva_baseline import _load_predictions


def main():
    root = Path(__file__).resolve().parent.parent
    base = root / 'first_frame_sapiens_20261015'
    source_run = base / 'replay_selected_init/map_sigma075'
    parameters = json.loads((source_run / 'run_parameters.json').read_text(encoding='utf-8'))
    inputs = parameters['inputs']
    model = SMPLHObservationMap(Path(parameters['effective_initial_fit']),
        *[Path(inputs[k]) for k in ('smplh_model', 'regressor', 'vposer_dir', 'mano_left', 'mano_right')], device='cuda')
    config = SMPLHMotionConfig(**parameters['configuration'])
    cal = load_stereo_fisheye(Path(inputs['calibration_dir']))
    left = _load_predictions(Path(inputs['left_predictions']), 'left')
    right = _load_predictions(Path(inputs['right_predictions']), 'right')
    scene = base / 'visualization/scene'
    with np.load(scene / 'scene_transforms.npz', allow_pickle=False) as data:
        rotations = data['rotation_ground_from_left'].copy()
        translations = data['translation_ground_from_left_mm'].copy()/1000
    stages = [json.loads(line)['stage']['operational']
              for line in (scene / 'stage1_stage2.jsonl').read_text(encoding='utf-8').splitlines()]
    with np.load(source_run / 'result.npz', allow_pickle=False) as data:
        saved = {key: data[key].copy() for key in ('global_orient', 'transl', 'vposer_latent', 'betas')}
    assert np.array_equal(saved['betas'], model.betas.detach().cpu().numpy())
    surface_sets = REPO / ('research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/'
                           'contact_surface_sets_smplh_v1/contact_vertex_sets.json')
    soles = load_sole_vertex_indices(surface_sets)
    ground_config = WorldGroundConfig(maximum_iterations=80, bootstrap_iterations=80)
    rows = []
    result_path = Path(__file__).with_name('same_frame_contact_diagnostic.json')
    report = {
        'purpose': 'Compare the same SLSQP fixed-prior MAP with/without current Stage1 ground assumptions',
        'script': str(Path(__file__).resolve()), 'source_run': str(source_run.resolve()),
        'source_scene': str(scene.resolve()), 'frame_order': [48, 61, 35, 188],
        'observation_source': 'same-frame original PMPose raw-fisheye pixels;no frame0 observations are reused',
        'fixed_shape': saved['betas'].tolist(), 'beta_exact_equal_to_source': True,
        'prior': 'Initial114x114 diagonal covariance of SMPLHWorldMotionTracker;not the saved recursive temporal covariance',
        'prior_scales_q_translation_root_latent': [.1, .35, .5],
        'v_a_start': 'zero;same independent diagnostic prior for each frame/policy/iteration budget',
        'whole_vposer_prior_sigma': config.latent_prior_sigma,
        'policies': {
            'ground_disabled': 'Both condition functions constant+1;stage2 flag only bypasses hard-coded support-side checks;stationary derivative conditioning disabled;no heel or sole constraint',
            'stage1_contact': 'Existing Stage1 one-side support union plus both-foot nonpenetration;same physical5mm bounds and existing0.2mm interior margin',
        },
        'iteration_policy': '80 first;only finite improving but optimizer-unsuccessful cases rerun200 from identical original q/P,not from80 posterior',
        'limits': ['No independent world/ground/contact/body truth',
                   'New diagonal diagnostic prior is not an ablation of the real temporal update',
                   'SLSQP success reports its termination rule,not a global optimum or anatomically correct pose',
                   'Contact assumptions are engineering conditions;failed estimates retained',
                   'No current or future triangulated body point enters the objective'],
        'frames': rows,
    }

    def write():
        result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

    for frame in report['frame_order']:
        assert stages[frame] == 'stage1_walker_static_human_moving'
        R, t = rotations[frame], translations[frame]
        pixels = np.array([left[frame, 5:17], right[frame, 5:17]])
        frame_row = {'frame_index': frame, 'stage': stages[frame], 'policies': {}}
        rows.append(frame_row)
        for policy in report['policies']:
            attempts = []
            frame_row['policies'][policy] = attempts
            for budget in (80, 200):
                tracker = SMPLHWorldMotionTracker(model, cal, config, R, t, soles,
                    replace(ground_config, maximum_iterations=budget, bootstrap_iterations=budget), constrain_ground=True)
                camera_root = Rotation.from_rotvec(saved['global_orient'][frame]).as_matrix()
                tracker.pose = MotionPose(R @ (saved['transl'][frame]+tracker.pelvis_offset)+t-tracker.pelvis_offset,
                                         R @ camera_root, saved['vposer_latent'][frame].copy())
                tracker.last_accepted_pose = tracker.pose.copy()
                tracker.covariance = tracker.initial_covariance.copy()
                tracker.velocity[:] = tracker.acceleration[:] = 0
                tracker.set_scene_frame(R, t, stages[frame], 'same_saved_stage1_transform')
                if policy == 'ground_disabled':
                    tracker.stage2 = True
                    tracker._conditions = lambda geometry: np.ones(geometry['sole_min_m'].shape[:-1]+(1,))
                    tracker._solver_conditions = tracker._conditions
                    tracker._condition_stationary_derivatives = lambda: None
                prior = tracker.pose.copy()
                precision = np.linalg.inv(tracker.covariance[:tracker.n, :tracker.n])
                valid, reasons = tracker._supported_pixels(pixels)

                def describe():
                    data = model.evaluate(tracker.pose, mesh=True)
                    projected = tracker._pixels(data['coco_m'])[0]
                    error = np.linalg.norm(projected-pixels[..., :2], axis=2)
                    norms = error[valid]
                    data_cost = np.sum(np.where(norms <= config.huber_px, .5*norms**2,
                                      config.huber_px*(norms-.5*config.huber_px)))/config.pixel_sigma_px**2
                    delta = np.r_[tracker.pose.translation_m-prior.translation_m,
                        Rotation.from_matrix(prior.root_rotation.T @ tracker.pose.root_rotation).as_rotvec(),
                        tracker.pose.latent-prior.latent]
                    prior_cost = .5*delta @ precision @ delta
                    latent_cost = .5*np.sum(tracker.pose.latent**2)/config.latent_prior_sigma**2
                    geometry = foot_geometry(data['vertices_m'][0], soles)
                    return {'raw_all24_median_px': float(np.median(error)),
                        'raw_all24_p95_px': float(np.percentile(error, 95)), 'raw_all24_max_px': float(error.max()),
                        'raw_all24_error_px': error.tolist(), 'used_views_median_px': float(np.median(norms)),
                        'objective': float(data_cost+prior_cost+latent_cost),
                        'objective_terms': {'stereo2d': float(data_cost), 'fixed_Pqq_pose': float(prior_cost), 'whole_vposer': float(latent_cost)},
                        'sole_min_mm_left_right': (geometry['sole_min_m']*1000).tolist(),
                        'ankle_height_mm_left_right': (data['coco_m'][0, 15:17, 2]*1000).tolist(),
                        'native_pelvis_height_mm': float(data['joints_m'][0, 0, 2]*1000),
                        'latent_norm': float(np.linalg.norm(tracker.pose.latent)),
                        'world_translation_m': tracker.pose.translation_m.tolist(),
                        'world_root_matrix': tracker.pose.root_rotation.tolist(),
                        'latent': tracker.pose.latent.tolist()}

                before = describe()
                started = perf_counter()
                accepted, used, status = tracker._constrained_update(pixels, valid, reasons, initialize=False)
                elapsed = perf_counter()-started
                after = describe()
                diagnostic = tracker.contact_diagnostic
                attempt = {'iterations_budget': budget, 'used_mask': used.tolist(),
                    'original_valid_mask': valid.tolist(), 'update_accepted': bool(accepted),
                    'update_status': status, 'seconds': elapsed, 'before': before, 'after': after,
                    'optimizer_diagnostic': diagnostic}
                attempts.append(attempt)
                write()
                print(json.dumps({'frame': frame, 'policy': policy, 'budget': budget,
                    'accepted': bool(accepted), 'status': status, 'iterations': diagnostic.get('iterations'),
                    'objective_before_after': [before['objective'], after['objective']],
                    'raw_median_before_after': [before['raw_all24_median_px'], after['raw_all24_median_px']],
                    'sole_mm_before_after': [before['sole_min_mm_left_right'], after['sole_min_mm_left_right']],
                    'seconds': elapsed}), flush=True)
                improving = diagnostic.get('objective', np.inf) < before['objective']-.01
                if diagnostic.get('optimizer_success', False) or not improving:
                    break
    print('saved', result_path, flush=True)


if __name__ == '__main__':
    main()
