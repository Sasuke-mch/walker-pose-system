"""Replay the single high-priority autograd/contact SMPL-H route, 448 frames.

Uses the existing direction MAP, fixed hip mapping and q/v/a process. No
finite-difference/control route is run. Physical floor Z=0 is unchanged; model
sole contact is referenced to Z=25 mm for footwear.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

APP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(APP), str(APP / 'tools')]
from pose_app.foot_ground_reference import SHOE_HEIGHT_M
from probe_smplh_foot_observations import (
    EXTENSION_HELDOUT,
    HELDOUT,
    OUT,
    ROOT,
    direction_error,
    load,
    project,
    save,
)
from probe_smplh_soft_contact import (
    BALL,
    SOLE,
    SURFACE,
    SURFACE_PATH,
    ContactPointMap,
    contact_residuals,
    support_labels,
)
from smplh_contact_autograd import AutogradContactTracker
from walker_tools.visualization._scene_export import finite, prepare_scene

DEST = OUT / 'autograd_contact_shoe25_full448_20261017'


def statistics(values):
    values = np.asarray(values)
    valid = values[np.isfinite(values)]
    return {'total_slots': int(values.size), 'finite_slots': int(valid.size),
            'median': float(np.median(valid)) if valid.size else None,
            'p95': float(np.percentile(valid, 95)) if valid.size else None,
            'max': float(valid.max()) if valid.size else None}


def finite_min_mm(values):
    valid = np.asarray(values)[np.isfinite(values)]
    return float(valid.min()*1000) if valid.size else None


def analyze(arrays, rows, body, toes, calibration):
    errors = np.linalg.norm(project(arrays['predicted_coco'][:, 5:17], calibration)-body[..., :2], axis=-1)
    report = {}
    for name, selection in [('all448', slice(None)), ('primary35_61', slice(35, 62))]:
        report[name] = {part: statistics(errors[selection][:, :, indices]) for part, indices in
            [('body', list(range(12))), ('hip', [6, 7]), ('nonhip', [0, 1, 2, 3, 4, 5, 8, 9, 10, 11])]}
    clearance = arrays['sole_min_m']-SHOE_HEIGHT_M
    labels = arrays['support_ball']
    report.update(available_frames=int(arrays['pose_available'].sum()), frames=len(rows),
        solver_status=dict(Counter(row['optimizer']['status'] for row in rows)),
        derivative_valid_slots=int(sum(row['derivative_count'] for row in rows)),
        derivative_total_slots=12*len(rows), segments=len(np.unique(arrays['segment_id'])),
        sole_height_mm=statistics(arrays['sole_min_m']*1000), sole_clearance_mm=statistics(clearance*1000),
        sole_min_height_mm=finite_min_mm(arrays['sole_min_m']),
        below_shoe_tolerance_slots=int((clearance < -.005).sum()),
        below_shoe_tolerance_frames=int((clearance < -.005).any(axis=1).sum()),
        nonfinite_foot_slots=int((~np.isfinite(clearance)).sum()),
        nonfinite_foot_frames=int((~np.isfinite(clearance)).any(axis=1).sum()),
        below_physical_floor_slots=int((arrays['sole_min_m'] < 0).sum()),
        below_physical_floor_frames=int((arrays['sole_min_m'] < 0).any(axis=1).sum()),
        support_abs_error_mm=statistics(abs(arrays['support_ball_height_m'][labels]-SHOE_HEIGHT_M)*1000),
        support_within10mm_slots=int((abs(arrays['support_ball_height_m'][labels]-SHOE_HEIGHT_M) <= .010).sum()))
    heldout = HELDOUT+EXTENSION_HELDOUT
    report['withheld_direction_deg'] = statistics(direction_error(project(arrays['toe_m'][heldout], calibration), toes[heldout, ..., :2]))
    report['engineering_checks'] = {'all448_available': report['available_frames'] == 448,
        'no_restarts': report['segments'] == 1,
        'sole_within_5mm_of_shoe_lower_bound': report['below_shoe_tolerance_slots'] == 0 and report['nonfinite_foot_slots'] == 0,
        'all_labeled_support_within10mm': report['support_within10mm_slots'] == int(labels.sum()),
        'all_updated_frames_converged': all(row['optimizer']['status'] == 'converged' for row in rows[1:])}
    report['all_engineering_checks_passed'] = all(report['engineering_checks'].values())
    report['limits'] = 'Same-source engineering diagnostics only. Soft penalties do not guarantee geometric feasibility. Ground is Stage/ankle anchored;26 offline forefoot support hypotheses, not independent contact truth. Research priority is not an accuracy certification.'
    return report


def visualize(dest):
    from build_contact_experiment_viewers import (
        add_observations,
        model_extras,
        write_page,
    )
    from build_foot_observation_viewers import export
    target = dest / 'view'
    payload, metadata = export(dest.name, '自动微分＋接触 · 鞋高25mm · 448帧', '', dest/'scene_input', target=target, write_page=False)
    result = np.load(dest/'result.npz')
    rows = [json.loads(line) for line in (dest/'trajectory.jsonl').read_text(encoding='utf-8').splitlines()]
    calibration = add_observations(payload, metadata, target)
    details = [f"求解：{r['optimizer']['status']}；防下穿：启用；贴地标签：{r['support_ball']}" for r in rows]
    model_extras(payload['models'][0], str(dest/'result.npz'), list(range(448)), dest/'scene_input', calibration, details)
    payload.update(soleReferenceMm=SHOE_HEIGHT_M*1000, singleRoute=True,
        rejection='软接触工程路线；具体未满足项见帧信息与summary.json')
    entry = payload['models'][0]
    entry['soleReferenceMm'] = SHOE_HEIGHT_M*1000
    entry['soleClearance'] = finite((result['sole_min_m']-SHOE_HEIGHT_M)*1000)
    metadata.update(foot_ground_reference_m=SHOE_HEIGHT_M, route_id='smplh_causal_autograd_contact', priority='high',
        contact_labels=26, full_sequence_anti_penetration=True)
    write_page(payload, metadata, target, False)


def run(dest):
    dest = dest.resolve()
    if dest.parent != OUT.resolve():
        raise ValueError('Route output must be a new directory under the foot-direction experiment')
    dest.mkdir(exist_ok=False)
    save(dest/'protocol.json', {'route_id': 'smplh_causal_autograd_contact', 'priority': 'high',
        'command': [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        'frame_range': [0, 447], 'jacobian': 'autograd', 'contact_enabled': True,
        'state_coordinate_frame': 'current_left_camera',
        'causality_scope': 'Recursive body state; scene has fixed lag and support labels are offline, not an immediate end-to-end causal contact system.',
        'foot_ground_reference_m': SHOE_HEIGHT_M, 'physical_floor_z_m': 0.,
        'contact': {'lower_bound_m': SHOE_HEIGHT_M-.005, 'tolerance_m': .005,
            'penetration_scale_m': .005, 'height_target_m': SHOE_HEIGHT_M, 'height_scale_m': .010,
            'anti_penetration_frames': [1, 447], 'frame0': 'saved initialization; evaluated, not silently shifted',
            'support_intervals_inclusive': {'right_ball': [35, 47], 'left_ball': [49, 61]},
            'unlabeled_frames': 'no attachment label inferred; anti-penetration remains enabled',
            'surface_sets': str(SURFACE_PATH), 'slip': 'no new slip term'},
        'method': '38D translation/root-SO3/VPoser q/v/a +114D covariance; real dt; raw stereo body pixels; transverse forefoot direction; fixed virtual hip map after10;15-iteration MAP; one pixel/pose information update, contact curvature excluded',
        'derivative_scope': 'Automatic MAP pixel/direction/contact Jacobians. Existing numerical point/time derivatives for reporting remain unchanged.',
        'heldout_toe_frames': HELDOUT+EXTENSION_HELDOUT,
        'success': 'Report availability, all raw pixel slots, sole>=20mm, labeled ball within25+/-10mm, convergence and unavailable derivatives on all448. A failed check stays failed; no prerequisite prefix gate blocks the user-authorized full replay.',
        'stop': 'One complete448 run of this route; no controls, weight sweep, smoothing or post-solve foot translation.',
        'limits': '25mm is user-specified shoe thickness, not estimated calibration. Offline reviewed labels only. Research priority does not certify accuracy/contact.'})
    prepare_scene(ROOT/'skeleton_tracking_20261016', dest/'scene_input')
    proto, model, tracker, body, toes, times = load(448, ContactPointMap, AutogradContactTracker)
    tracker.observation_mode = 'direction'
    scene = np.load(dest/'scene_input/scene/scene_transforms.npz')
    rotation, translation = scene['rotation_ground_from_left'], scene['translation_ground_from_left_mm']/1000
    if len(rotation) != 448 or not np.allclose(times, scene['pair_timestamp_sec'], rtol=0, atol=1e-7):
        raise ValueError('Fresh scene and body frame/timestamp alignment differ')
    protocol = json.loads((dest/'protocol.json').read_text(encoding='utf-8'))
    protocol.update(configuration=proto['configuration'], iterated_map_configuration=proto['iterated_map_configuration'],
        source_body_run=proto['source_run'], hip_calibration_result=proto['hip_calibration_result'],
        fresh_scene=str(dest/'scene_input/scene/scene_transforms.npz'))
    save(dest/'protocol.json', protocol)
    # Check the duplicated NumPy/Torch objective at the real initialized pose:
    # a missed footwear offset would otherwise optimize a different objective.
    import torch
    tracker.contact_rotation, tracker.contact_translation = rotation[0], translation[0]
    data = model.evaluate(tracker.pose, mesh=True)
    selected = np.ones(48, bool)
    expected = np.r_[tracker._pixels(data['coco_m']).ravel(),
        contact_residuals(data['contact_surface_m'], rotation[0], translation[0], [0, 0]).ravel()]
    delta = torch.zeros(tracker.n, dtype=torch.float64, device=model.device)
    actual = tracker._torch_objective_values(tracker.pose, delta, selected).detach().cpu().numpy()
    objective_difference = float(np.max(abs(actual-expected)))
    if not np.isfinite(actual).all() or objective_difference > 1e-8:
        raise ValueError(f'Autograd and NumPy shoe-height objectives differ: {objective_difference}')
    protocol['real_pose_objective_max_difference'] = objective_difference
    save(dest/'protocol.json', protocol)
    print('real-pose NumPy/autograd objective agreement', objective_difference, flush=True)
    labels, original = support_labels(448), model.regressor.clone()
    archive = {key: [] for key in ['predicted_coco', 'smplh_joints', 'transl', 'root_rotation', 'vposer_latent',
        'body_rotations', 'pose_available', 'segment_id', 'toe_m', 'velocity', 'acceleration', 'covariance',
        'sole_min_m', 'support_ball_height_m', 'derivative_valid']}
    rows, started = [], perf_counter()
    with (dest/'trajectory.jsonl').open('w', encoding='utf-8') as stream:
        for frame in range(448):
            if frame == 11:
                a, b = proto['hip_calibration_result']['alpha'], proto['hip_calibration_result']['thigh_fraction']
                model.regressor[11:13] = original[11:13]+a*(original[11:13]-original[[12, 11]])+b*(original[13:15]-original[11:13])
                tracker.last_point_jacobian = None
            tracker.contact_rotation, tracker.contact_translation, tracker.support = rotation[frame], translation[frame], labels[frame]
            tracker.extra_pixels = None if frame in HELDOUT+EXTENSION_HELDOUT or frame == 0 else toes[frame]
            result = tracker.step(float(times[frame]), *body[frame])
            surface = result['vertices_m'][SURFACE]
            heights = (surface@rotation[frame].T+translation[frame])[:, 2]
            sole = np.array([heights[:len(SOLE[0])].min(), heights[len(SOLE[0]):].min()])
            ball = np.array([heights[indices].min() for indices in BALL])
            values = {'predicted_coco': result['coco_m'], 'smplh_joints': result['joints_m'], 'transl': result['translation_m'],
                'root_rotation': result['root_rotation'], 'vposer_latent': result['vposer_latent'], 'body_rotations': result['body_rotations'],
                'pose_available': result['pose_available'], 'segment_id': result['segment_id'], 'toe_m': result['toe_m'],
                'velocity': tracker.velocity.copy(), 'acceleration': tracker.acceleration.copy(), 'covariance': tracker.covariance.copy(),
                'sole_min_m': sole, 'support_ball_height_m': ball, 'derivative_valid': result['derivative_valid']}
            for key, value in values.items():
                archive[key].append(value)
            residual = contact_residuals(surface[None], rotation[frame], translation[frame], labels[frame])[0]
            row = {'frame': frame, 'status': result['status'], 'optimizer': result['iterated_update'],
                'contact_active': frame > 0, 'support_ball': labels[frame].tolist(), 'sole_min_mm': finite(sole*1000),
                'sole_clearance_mm': finite((sole-SHOE_HEIGHT_M)*1000),
                'contact_diagnostic_computed': True,
                'contact_used_in_optimizer': bool(frame > 0 and result['iterated_update'].get('status') not in
                    ('not_run', 'insufficient_joint_support')),
                'contact_final_cost': finite([.5*np.sum(residual[:len(SURFACE)]**2), .5*np.sum(residual[len(SURFACE):]**2)]),
                'body_used': result['used_views'].tolist(), 'body_reason': result['view_reason'].tolist(),
                'toe_used': getattr(tracker, 'foot_used', np.zeros((2, 4), bool)).tolist(),
                'toe_reason': getattr(tracker, 'foot_reason', np.full((2, 4), 'initial_not_reused', dtype=object)).tolist(),
                'derivative_valid': result['derivative_valid'].tolist(),
                'derivative_reason': result['derivative_reason'].tolist(),
                'joint_status': result['joint_status'].tolist(),
                'position_std_m': finite(result['position_std_m']),
                'joint_observation_age_s': finite(result['joint_observation_age_s']),
                'prediction_rejection': result['prediction_rejection'],
                'derivative_count': int(result['derivative_valid'].sum())}
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); stream.flush()
            if frame % 10 == 0 or frame == 447:
                print(f"frame {frame}/447 {row['optimizer']['status']} sole_mm={row['sole_min_mm']}", flush=True)
    arrays = {key: np.asarray(value) for key, value in archive.items()}
    arrays.update(betas=model.betas.cpu().numpy(), timestamp_s=times, regressor=model.regressor.cpu().numpy(),
        support_ball=labels, foot_ground_reference_m=SHOE_HEIGHT_M)
    np.savez_compressed(dest/'result.npz', **arrays)
    report = analyze(arrays, rows, body, toes, tracker.cal)
    report['runtime_s'] = perf_counter()-started
    save(dest/'summary.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    visualize(dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEST)
    parser.add_argument('--visualize-only', action='store_true')
    args = parser.parse_args()
    visualize(args.output.resolve()) if args.visualize_only else run(args.output)
