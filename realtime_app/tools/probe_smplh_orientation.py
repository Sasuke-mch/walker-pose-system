"""Bounded orientation diagnosis and pelvis-local hip observation experiment.

This tool never penalizes rotation amplitude. The optional prefix replay changes
only the hip observation operator; native SMPL-H geometry stays unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

APP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(APP), str(APP / 'tools')]
from pose_app.fisheye_camera import fisheye_project_numpy, load_stereo_fisheye
from pose_app.smplh_ground_constraints import load_sole_vertex_indices
from pose_app.smplh_iterated_motion_tracker import (
    IteratedMAPConfig,
    SMPLHIteratedMotionTracker,
)
from pose_app.smplh_motion_tracker import SMPLHMotionConfig, SMPLHObservationMap
from pose_app.body_observations import load_ordered_predictions as _load_predictions
from pose_app.replay_pair_timing import load_capture_pair_times as _read_capture_times

ROOT = APP.parent / 'research_records/engineering_validation/G20261014_smplh_motion448_v1'
REPAIR = ROOT / 'hip_pose_repair_20261016'
OUTPUT = REPAIR / 'orientation_20261016'


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def project(coco, cal):
    other = coco @ cal.R_cam0_to_cam1.T + cal.T_cam0_to_cam1_mm / 1000
    return np.stack((fisheye_project_numpy(coco, cal.K0, cal.D0),
                     fisheye_project_numpy(other, cal.K1, cal.D1)), axis=-3)


def hip_local(coco, joints, roots):
    return np.einsum('nji,nkj->nki', roots, coco[:, 11:13] - joints[:, :1])


def local_hips(joints, roots, offsets):
    return joints[:, :1] + np.einsum('nij,kj->nki', roots, offsets)


class PelvisHipMap(SMPLHObservationMap):
    """Experimental observation landmarks rigidly attached to native pelvis."""

    hip_offsets = None

    def evaluate_batch(self, translations, roots, latents, mesh=False):
        data = super().evaluate_batch(translations, roots, latents, mesh)
        if self.hip_offsets is not None:
            data['coco_m'][:, 11:13] = local_hips(data['joints_m'], roots, self.hip_offsets)
        return data


def load():
    protocol = json.loads((REPAIR / 'hip_operator_feedback/protocol.json').read_text())
    source = Path(protocol['source_run'])
    p = json.loads((source / 'run_parameters.json').read_text())
    inp = p['inputs']
    model = PelvisHipMap(Path(p['effective_initial_fit']),
        *[Path(inp[k]) for k in ('smplh_model', 'regressor', 'vposer_dir', 'mano_left', 'mano_right')], device='cuda')
    model.torch.set_num_threads(2)
    cal = load_stereo_fisheye(inp['calibration_dir'])
    tracker = SMPLHIteratedMotionTracker(model, cal, SMPLHMotionConfig(**protocol['configuration']),
                                       IteratedMAPConfig(**protocol['iterated_map_configuration']))
    obs = np.stack([_load_predictions(Path(inp[f'{s}_predictions']), s) for s in ('left', 'right')], axis=1)
    obs[0] = np.asarray(json.loads(Path(p['first_frame_observations']).read_text())['keypoints'])
    times = _read_capture_times(Path(inp['input_pairs']), len(obs))[0]
    return protocol, model, tracker, obs[:, :, 5:17], times


def calibrate(model, tracker, obs, reference):
    # Poses1..10 precede adaptation and are identical in the saved A run.
    ids = np.arange(1, 11)
    roots = reference['root_rotation'][ids]
    data = model.evaluate_batch(reference['transl'][ids], roots, reference['vposer_latent'][ids])
    co, joints = data['coco_m'], data['joints_m']
    base = hip_local(co, joints, roots).mean(axis=0)
    width = np.einsum('nji,nkj->nki', roots, co[:, 11:13] - co[:, [12, 11]]).mean(axis=0)
    thigh = np.einsum('nji,nkj->nki', roots, co[:, 13:15] - co[:, 11:13]).mean(axis=0)
    mask = np.stack([tracker._supported_pixels(obs[i])[0] for i in ids])[:, :, 6:8]

    def residual(weights):
        hips = local_hips(joints, roots, base + weights[0] * width + weights[1] * thigh)
        return ((project(hips, tracker.cal) - obs[ids, :, 6:8, :2])[mask] / 8).ravel()

    fit = least_squares(residual, [.15, .15], bounds=([0., 0.], [.5, .4]), max_nfev=30)
    offsets = base + fit.x[0] * width + fit.x[1] * thigh
    return offsets, {'weights': fit.x.tolist(), 'offsets_pelvis_m': offsets.tolist(),
        'base_pelvis_m': base.tolist(), 'width_basis_m': width.tolist(), 'thigh_basis_m': thigh.tolist(),
        'nfev': fit.nfev, 'cost': fit.cost, 'optimality': fit.optimality, 'success': bool(fit.success),
        'at_bounds': bool(np.any(np.isclose(fit.x, [0., 0.], atol=1e-5)) or
                          np.any(np.isclose(fit.x, [.5, .4], atol=1e-5))),
        'calibration_frames': [1, 10], 'used_hip_slots': int(mask.sum())}


def costs(coco, obs, mask, cal):
    norms = np.linalg.norm(project(coco[:, 5:17], cal) - obs[..., :2], axis=-1)
    cost = np.where(norms <= 25, .5 * norms**2, 25 * (norms - 12.5)) / 64
    return {name: float(cost[0][:, indices][mask[:, indices]].sum()) for name, indices in
            [('hip', [6, 7]), ('knee', [8, 9]), ('ankle', [10, 11]), ('upper', list(range(6)))]}


def diagnose():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    proto, model, tracker, obs, _ = load()
    reference = np.load(REPAIR / 'hip_operator_feedback/result.npz')
    _, calibration = calibrate(model, tracker, obs, reference)
    protocol = {'source_run': proto['source_run'], 'reference': str(REPAIR / 'hip_operator_feedback'),
        'inputs': json.loads((Path(proto['source_run']) / 'run_parameters.json').read_text())['inputs'],
        'command': [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        'code': 'realtime_app/tools/probe_smplh_orientation.py',
        'configuration': proto['configuration'], 'iterated_map_configuration': proto['iterated_map_configuration'],
        'single_change': 'From frame11 replace only COCO hips by J0 + Rroot * fixed pelvis-local offsets. Two shared coefficients calibrated once on the same saved pre-adaptation poses1..10; fixed local basis is their mean original hip/width/thigh geometry.',
        'unchanged': 'beta, initial pose, native skeleton, hand pose, all other COCO points, raw pixels,dt,gates,15-iteration budget,P/q/v/a process and VPoser prior; no contact/smoothing/rotation penalty',
        'diagnosis': 'Saved poses35,47,61; root ground yaw and local hip x-axis rotation/bone-axis twist, ankle twist +/-10deg; compare dynamic map and rigid hip map matched at the central pose to isolate observation derivatives. Static probes are not adopted trajectories.',
        'prefix_gate': {'frames': [0, 61], 'primary': [35, 61], 'hip_and_nonhip_median_p95_max_ratio': 1.05,
            'mean_ankle_rise_max_mm': 30, 'sole_min_degradation_max_mm': 5,
            'no_more_penetrating_foot_frames_below_minus5mm': True, 'no_new_missing_or_restarts': True,
            'unconverged_count_must_not_increase': True},
        'orientation_acceptance': 'A smaller orientation range alone NEVER passes. Direction accuracy requires unused image foot/hip direction evidence; passing numeric prefix gate only permits that next stage, not promotion.',
        'stop': 'One calibration, one optional prefix; no parameter search, no448 until pixel/height/solver gates and independent direction review pass.',
        'limits': 'Same-source engineering experiment. Frozen beta used original whole sequence; calibration holdout is not independent subject/shape/3D truth. Saved operational ground is evaluation-only, not tracker input.',
        'hip_calibration': calibration}
    save(OUTPUT / 'protocol.json', protocol)
    alpha = proto['hip_calibration_result']['alpha']
    fraction = proto['hip_calibration_result']['thigh_fraction']
    original = model.regressor.clone()
    model.regressor[11:13] = original[11:13] + alpha * (original[11:13] - original[[12, 11]]) + fraction * (original[13:15] - original[11:13])
    scene = np.load(REPAIR / 'view/scene/scene_transforms.npz')
    ground_R = scene['rotation_ground_from_left']
    motion = hip_local(reference['predicted_coco'][11:], reference['smplh_joints'][11:], reference['root_rotation'][11:])
    report = {'calibration': calibration, 'adapted_hip_local_coordinate_range_mm': (np.ptp(motion, axis=0) * 1000).tolist(),
              'probes': []}
    torch = model.torch
    with torch.inference_mode():
        shaped = model.model.v_template + torch.einsum('vci,bi->bvc', model.model.shapedirs, model.betas)[0]
        rest = (model.model.J_regressor @ shaped).cpu().numpy()
    for frame in (35, 47, 61):
        root = reference['root_rotation'][frame]
        base_body = reference['body_rotations'][frame]
        # Twist is about the rest child-bone axis, expressed in joint-local axes.
        axes = {'root_yaw': (None, ground_R[frame].T @ np.array([0., 0., 1.])),
                'left_hip_local_x': (0, np.array([1., 0., 0.])),
                'right_hip_local_x': (1, np.array([1., 0., 0.])),
                'left_hip_twist': (0, rest[4] - rest[1]),
                'right_hip_twist': (1, rest[5] - rest[2]),
                'left_ankle_twist': (6, rest[10] - rest[7]),
                'right_ankle_twist': (7, rest[11] - rest[8])}
        mask = tracker._supported_pixels(obs[frame])[0]
        central_offsets = hip_local(reference['predicted_coco'][frame:frame+1],
                                    reference['smplh_joints'][frame:frame+1], root[None])[0]
        for name, (joint, axis) in axes.items():
            angles = np.array([-10., -5., 0., 5., 10.])
            axis = axis / np.linalg.norm(axis)
            rotations = Rotation.from_rotvec(np.deg2rad(angles)[:, None] * axis).as_matrix()
            roots = np.broadcast_to(root, (5, 3, 3)).copy()
            bodies = np.broadcast_to(base_body, (5, 21, 3, 3)).copy()
            if joint is None:
                roots = rotations @ root
            else:
                bodies[:, joint] = base_body[joint] @ rotations
            def tensor(a):
                return torch.as_tensor(a, dtype=torch.float32, device=model.device)
            with torch.inference_mode():
                out = model.model(betas=model.betas.expand(5, -1),
                    transl=tensor(np.broadcast_to(reference['transl'][frame], (5, 3)).copy()),
                    global_orient=tensor(roots), body_pose=tensor(bodies),
                    left_hand_pose=model.hand_rotations[0].expand(5, -1, -1, -1),
                    right_hand_pose=model.hand_rotations[1].expand(5, -1, -1, -1))
                co = torch.einsum('jv,bvc->bjc', model.regressor, out.vertices).cpu().numpy().astype(float)
                joints = out.joints[:, :52].cpu().numpy().astype(float)
            rigid = co.copy()
            rigid[:, 11:13] = local_hips(joints, roots, central_offsets)
            dynamic_cost, rigid_cost = [], []
            for i in range(5):
                dynamic_cost.append(costs(co[i:i+1], obs[frame], mask, tracker.cal))
                rigid_cost.append(costs(rigid[i:i+1], obs[frame], mask, tracker.cal))
            sensitivity = np.linalg.norm((project(co[3, 11:13], tracker.cal) - project(co[1, 11:13], tracker.cal)) / 10, axis=-1)
            report['probes'].append({'frame': frame, 'perturbation': name, 'angle_deg': angles.tolist(),
                'dynamic_cost': dynamic_cost, 'matched_rigid_cost': rigid_cost,
                'dynamic_hip_pixel_sensitivity_px_per_deg': sensitivity.tolist(),
                'used_hip_slots': mask[:, 6:8].tolist()})
    save(OUTPUT / 'diagnosis.json', report)
    print(json.dumps({'calibration': calibration, 'hip_local_ranges_mm': report['adapted_hip_local_coordinate_range_mm'],
        'probes': [{k: r[k] for k in ('frame', 'perturbation', 'dynamic_hip_pixel_sensitivity_px_per_deg')} for r in report['probes']]}, indent=2), flush=True)


def metrics(data, sole, obs, R, t, rows):
    co = data['predicted_coco']
    err = np.linalg.norm(project(co[:, 5:17], load_stereo_fisheye(
        json.loads((OUTPUT / 'protocol.json').read_text())['inputs']['calibration_dir'])) - obs[..., :2], axis=-1)
    valid = np.isfinite(obs).all(axis=-1) & (obs[..., 2] >= .2) & (obs[..., 2] <= 1)
    valid &= (obs[..., 0] >= 0) & (obs[..., 0] < 1920) & (obs[..., 1] >= 0) & (obs[..., 1] < 1080)
    valid &= data['pose_available'][:, None, None]
    out = {'available': int(data['pose_available'].sum()), 'segments': len(np.unique(data['segment_id'])),
           'optimizer_statuses': dict(Counter(r['optimizer']['status'] for r in rows if r['optimizer'].get('status') != 'not_run'))}
    for start, end, label in [(35, 62, 'primary'), (11, 62, 'heldout')]:
        out[label] = {}
        for name, indices in [('hip', [6, 7]), ('nonhip', [0, 1, 2, 3, 4, 5, 8, 9, 10, 11])]:
            values = err[start:end, :, indices]
            mask = valid[start:end, :, indices]
            values = values[mask]
            out[label][name] = {'finite_slots': len(values), 'median_px': float(np.median(values)), 'p95_px': float(np.percentile(values, 95))}
    world = np.einsum('nij,nkj->nki', R, co) + t[:, None]
    out['mean_ankle_rise35_61_mm'] = float((world[61, 15:17, 2] - world[35, 15:17, 2]).mean() * 1000)
    out['sole_min35_61_mm'] = float(sole[35:62].min() * 1000)
    out['penetrating_foot_frames35_61'] = int((sole[35:62] < -.005).sum())
    native = np.einsum('nij,nkj->nki', R, data['smplh_joints'])
    dirs = np.stack((native[:, 2] - native[:, 1], native[:, 10] - native[:, 7], native[:, 11] - native[:, 8]), axis=1)
    angles = np.rad2deg(np.unwrap(np.arctan2(dirs[..., 1], dirs[..., 0]), axis=0))
    out['direction_range35_61_deg_pelvis_left_right'] = np.ptp(angles[35:62], axis=0).tolist()
    return out


def replay():
    protocol = json.loads((OUTPUT / 'protocol.json').read_text())
    calibration = protocol['hip_calibration']
    if not calibration['success'] or calibration['at_bounds']:
        raise ValueError('Calibration failed or hit bounds; do not replay')
    dest = OUTPUT / 'pelvis_local_prefix'
    dest.mkdir(exist_ok=False)
    _, model, tracker, obs, times = load()
    reference = np.load(REPAIR / 'hip_operator_feedback/result.npz')
    scene = np.load(REPAIR / 'view/scene/scene_transforms.npz')
    R = scene['rotation_ground_from_left'][:62]
    t = scene['translation_ground_from_left_mm'][:62] / 1000
    sole_indices = load_sole_vertex_indices(APP.parent / 'research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/contact_surface_sets_smplh_v1/contact_vertex_sets.json')
    arrays = {k: [] for k in ['predicted_coco', 'smplh_joints', 'transl', 'root_rotation', 'vposer_latent', 'body_rotations',
        'pose_available', 'segment_id', 'velocity', 'acceleration', 'covariance', 'used_views', 'sole_min_m']}
    rows = []
    started = perf_counter()
    save(dest / 'protocol.json', dict(protocol, command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]))
    with (dest / 'trajectory.jsonl').open('w', encoding='utf-8') as stream:
        for i in range(62):
            if i == 11:
                model.hip_offsets = np.asarray(calibration['offsets_pelvis_m'])
                tracker.last_point_jacobian = None
            result = tracker.step(float(times[i]), *obs[i])
            for key, value in [('predicted_coco', result['coco_m']), ('smplh_joints', result['joints_m']),
                ('transl', result['translation_m']), ('root_rotation', result['root_rotation']),
                ('vposer_latent', result['vposer_latent']), ('body_rotations', result['body_rotations']),
                ('pose_available', result['pose_available']), ('segment_id', result['segment_id']),
                ('velocity', tracker.velocity.copy()), ('acceleration', tracker.acceleration.copy()),
                ('covariance', tracker.covariance.copy()), ('used_views', result['used_views'])]:
                arrays[key].append(value)
            mesh = model.evaluate(tracker.pose, mesh=True)['vertices_m'][0]
            z = (mesh @ R[i].T + t[i])[:, 2]
            arrays['sole_min_m'].append([z[ids].min() for ids in sole_indices])
            row = {'frame': i, 'status': result['status'], 'optimizer': result['iterated_update'],
                   'used': result['used_views'].tolist(), 'reason': result['view_reason'].tolist(),
                   'derivative_count': int(result['derivative_valid'].sum())}
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            rows.append(row)
            if i % 10 == 0:
                print('frame', i, row['status'], row['optimizer']['status'], flush=True)
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    arrays.update(betas=model.betas.cpu().numpy(), timestamp_s=times[:62], hip_offsets_pelvis_m=model.hip_offsets)
    np.savez_compressed(dest / 'result.npz', **arrays)
    # Rebuild only reference geometry for evaluation; no reference fit is rerun.
    ref_sole = []
    for i in range(62):
        mesh = model.evaluate_batch(reference['transl'][i:i+1], reference['root_rotation'][i:i+1], reference['vposer_latent'][i:i+1], True)['vertices_m'][0]
        z = (mesh @ R[i].T + t[i])[:, 2]
        ref_sole.append([z[ids].min() for ids in sole_indices])
    ref_rows = [json.loads(line) for line in (REPAIR / 'hip_operator_feedback/trajectory.jsonl').read_text().splitlines()]
    a = metrics(reference, np.asarray(ref_sole), obs[:62], R, t, ref_rows)
    b = metrics(arrays, arrays['sole_min_m'], obs[:62], R, t, rows)
    gate = {f'{window}_{part}_{stat}': b[window][part][stat] <= 1.05 * a[window][part][stat]
        for window in ('primary', 'heldout') for part in ('hip', 'nonhip') for stat in ('median_px', 'p95_px')}
    gate.update(all62_available=b['available'] == 62, no_new_segments=b['segments'] == a['segments'],
        no_more_unconverged=b['optimizer_statuses'].get('feasible_unconverged', 0) <= a['optimizer_statuses'].get('feasible_unconverged', 0),
        ankle_rise_below30mm=b['mean_ankle_rise35_61_mm'] < 30,
        sole_min_within5mm=b['sole_min35_61_mm'] >= a['sole_min35_61_mm'] - 5,
        no_more_penetrating_foot_frames=b['penetrating_foot_frames35_61'] <= a['penetrating_foot_frames35_61'])
    report = {'runtime_s': perf_counter() - started, 'reference': a, 'candidate': b, 'gates': gate,
              'numeric_prefix_passed': all(gate.values()), 'orientation_accuracy_validated': False,
              'decision': 'Proceed to unused image direction evidence only if numeric prefix passes; otherwise reject this mapping and stop its expansion.'}
    save(dest / 'summary.json', report)
    print(json.dumps(report, indent=2), flush=True)


def inspect_saved():
    """Decompose saved hip movement and expose unoverlaid source foot crops."""
    import cv2

    reference = np.load(REPAIR / 'hip_operator_feedback/result.npz')
    protocol = json.loads((REPAIR / 'hip_operator_feedback/protocol.json').read_text())
    alpha = protocol['hip_calibration_result']['alpha']
    fraction = protocol['hip_calibration_result']['thigh_fraction']
    co, roots = reference['predicted_coco'][11:62], reference['root_rotation'][11:62]
    # Invert the saved affine mapping exactly; no body forward/fit is rerun.
    matrix = np.array([[1 + alpha - fraction, -alpha], [-alpha, 1 + alpha - fraction]])
    hips = np.einsum('ij,njk->nik', np.linalg.inv(matrix), co[:, 11:13] - fraction * co[:, 13:15])
    def local(points):
        return np.einsum('nji,nkj->nki', roots, points)
    parts = {'base': local(hips - reference['smplh_joints'][11:62, :1]),
             'widening': local(alpha * (hips - hips[:, [1, 0]])),
             'knee_term': local(fraction * (co[:, 13:15] - hips))}
    decomposition = {k: {'xyz_range_mm': (np.ptp(v, axis=0) * 1000).tolist(),
        'max_pair_distance_mm': [float(np.linalg.norm(v[:, s, None] - v[None, :, s], axis=-1).max() * 1000) for s in range(2)]}
        for k, v in parts.items()}
    save(OUTPUT / 'hip_motion_decomposition.json', decomposition)
    p = json.loads((Path(protocol['source_run']) / 'run_parameters.json').read_text())
    pairs = Path(p['inputs']['input_pairs'])
    rows, sources = [], []
    for frame in [0, 35, 47, 61]:
        tiles = []
        for side, folder in [('left', 'left_ccw90'), ('right', 'right_cw90')]:
            path = pairs / folder / f'pair_{frame:04d}.png'
            image = cv2.imread(str(path))
            if image is None or image.shape[:2] != (1920, 1080):
                raise ValueError(f'Expected1080x1920 upright source: {path}')
            crop = cv2.resize(image[1350:1920], (648, 342))
            cv2.putText(crop, f'frame{frame} {side}: upright x[0,1080) y[1350,1920)', (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(crop)
            sources.append({'frame': frame, 'side': side, 'upright_image': str(path),
                            'crop_xywh': [0, 1350, 1080, 570], 'output_scale': .6})
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(OUTPUT / 'foot_visibility.jpg'), np.vstack(rows))
    save(OUTPUT / 'foot_image_sources.json', {'images': sources,
        'purpose': 'Visibility review before adding foot landmarks. Original colors; no model overlays or inferred hidden heel points. Not contact or3D truth.'})
    # Relabel coordinate-axis perturbations without recomputing saved probes.
    diagnosis = json.loads((OUTPUT / 'diagnosis.json').read_text())
    for probe in diagnosis['probes']:
        probe['perturbation'] = probe['perturbation'].replace('_hip_swing', '_hip_local_x')
    save(OUTPUT / 'diagnosis.json', diagnosis)
    result_path = OUTPUT / 'pelvis_local_prefix/result.npz'
    if result_path.exists():
        result = np.load(result_path)
        # The first11 frames must precede the operator switch in both runs.
        prefix_difference = float(np.max(abs(result['predicted_coco'][:11] - reference['predicted_coco'][:11])))
        summary_path = OUTPUT / 'pelvis_local_prefix/summary.json'
        summary = json.loads(summary_path.read_text())
        summary['pre_switch_coco_max_difference_m'] = prefix_difference
        summary['numeric_prefix_passed'] = all(summary['gates'].values())
        save(summary_path, summary)
        print('pre-switch COCO maximum difference m:', prefix_difference)
    print(json.dumps(decomposition, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--replay', action='store_true')
    modes.add_argument('--inspect-saved', action='store_true')
    args = parser.parse_args()
    if args.inspect_saved:
        inspect_saved()
    elif args.replay:
        replay()
    else:
        diagnose()
