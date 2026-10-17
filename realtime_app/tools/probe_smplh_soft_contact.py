"""Bounded soft-contact experiments and a matched derivative-only control.

Contact is an engineering regularizer in the current MAP, never a pixel,
independent measurement, output correction, or contact truth label.
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
from probe_smplh_foot_observations import (
    HELDOUT,
    OUT,
    REPAIR,
    FootPixelTracker,
    FootPointMap,
    direction_error,
    load,
    project,
    save,
)
from pose_app.foot_ground_reference import SHOE_HEIGHT_M

SURFACE_PATH = APP.parent / 'research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/contact_surface_sets_smplh_v1/contact_vertex_sets.json'
SETS = json.loads(SURFACE_PATH.read_text())['sets']
SOLE = [np.unique(np.concatenate(list(SETS[f'{side}_sole_surface_candidate'].values()))) for side in ('left', 'right')]
SURFACE = np.concatenate(SOLE)
BALL = [np.flatnonzero(np.isin(SURFACE, SETS[f'{side}_sole_surface_candidate']['ball'])) for side in ('left', 'right')]
DEST = OUT / 'soft_contact_prefix'


def support_labels(count=62):
    """Frozen BEFORE solving, from source images, not fitted foot heights.

    The subject's right forefoot is stationary through47 while the left steps;
    the left is stationary after landing,49..61, while the right steps. Exclude
    the left landing47..48 and right lift48 onward. This is a support hypothesis
    from visible slipper/floor contact, not independently measured bare skin.
    """
    mask = np.zeros((count, 2), bool)
    mask[35:48, 1] = True
    mask[49:62, 0] = True
    return mask


def ground_surface(points, rotation, translation):
    return points @ rotation.T + translation


def contact_residuals(surface_m, rotation, translation, support):
    """Batch110 residuals:108 normalized unilateral +2 robust support heights."""
    heights = ground_surface(surface_m, rotation, translation)[..., 2]
    penetration = np.maximum(0., (SHOE_HEIGHT_M-.005)-heights) / (.005 * np.sqrt(len(SURFACE)))
    support_height = np.stack([heights[:, ids].min(axis=1) for ids in BALL], axis=1)
    u = (support_height - SHOE_HEIGHT_M) / .010
    # Stable signed square-root encoding: .5*r*r = sqrt(1+u*u)-1.
    attachment = u * np.sqrt(2. / (np.sqrt(1. + u*u) + 1.)) * support
    return np.concatenate((penetration, attachment), axis=1)


class ContactPointMap(FootPointMap):
    def evaluate_batch(self, translations, roots, latents, mesh=False):
        data = super().evaluate_batch(translations, roots, latents, mesh=True)
        data['contact_surface_m'] = data['vertices_m'][:, SURFACE].copy()
        if not mesh:
            del data['vertices_m']
        return data


class ContactTracker(FootPixelTracker):
    contact_rotation = None
    contact_translation = None
    support = np.zeros(2, bool)

    def _engineering_residuals(self, data):
        if self.contact_rotation is None:
            return np.empty((len(data['coco_m']), 0))
        return contact_residuals(data['contact_surface_m'], self.contact_rotation,
                                 self.contact_translation, self.support)


def summarize(arrays, vertices, rows, body, toes, cal, rotation, translation, labels):
    heights = np.einsum('nij,nkj->nki', rotation, vertices[:, SURFACE])[:, :, 2] + translation[:, None, 2]
    ball = np.stack([heights[:, ids].min(axis=1) for ids in BALL], axis=1)
    sole = np.stack([heights[:, :len(SOLE[0])].min(axis=1), heights[:, len(SOLE[0]):].min(axis=1)], axis=1)
    body_error = np.linalg.norm(project(arrays['predicted_coco'][:, 5:17], cal) - body[..., :2], axis=-1)
    result = {}
    for window, ids in [('primary35_61', np.arange(35, 62)), ('context11_61', np.arange(11, 62))]:
        result[window] = {}
        for part, indices in [('hip', [6, 7]), ('nonhip', [0, 1, 2, 3, 4, 5, 8, 9, 10, 11])]:
            values = body_error[ids][:, :, indices]
            result[window][part] = {'slots': int(values.size), 'median_px': float(np.median(values)), 'p95_px': float(np.percentile(values, 95))}
    support_error = abs(ball[labels]-SHOE_HEIGHT_M) * 1000
    angle = direction_error(project(arrays['toe_m'][HELDOUT], cal), toes[HELDOUT, ..., :2])
    selected_rows = rows[35:62]
    result.update(support_ball_abs_height_mm={'slots': int(support_error.size), 'median': float(np.median(support_error)), 'p95': float(np.percentile(support_error, 95)), 'max': float(support_error.max())},
        sole_min35_61_mm=float(1000 * sole[35:].min()), penetrating_foot_frames35_61=int((sole[35:] < SHOE_HEIGHT_M-.005).sum()),
        withheld_direction_median_deg=float(np.median(angle)), withheld_direction_max_deg=float(angle.max()),
        available=int(arrays['pose_available'].sum()), segments=len(np.unique(arrays['segment_id'])),
        primary_solver_status=dict(Counter(row['optimizer']['status'] for row in selected_rows)),
        primary_unconverged=sum(row['optimizer']['status'] != 'converged' for row in selected_rows),
        primary_max_posterior_updates=max(row['optimizer']['posterior_updates'] for row in selected_rows))
    return result, sole, ball


def run(jacobian='finite', contact=True):
    tracker_class = ContactTracker
    if jacobian == 'autograd':
        from smplh_contact_autograd import AutogradContactTracker
        tracker_class = AutogradContactTracker
    elif not contact:
        raise ValueError('The contact-free numerical control is only defined for autograd')
    dest = DEST if jacobian == 'finite' else OUT / ('soft_contact_autograd_prefix' if contact else 'toe_direction_autograd_prefix')
    reference_dir = OUT / ('toe_direction_autograd_prefix' if jacobian == 'autograd' and contact else 'toe_direction_prefix')
    if not (reference_dir / 'result.npz').exists():
        raise ValueError('Run the matching contact-free control first')
    proto, model, tr, body, toes, times = load(62, ContactPointMap, tracker_class)
    tr.observation_mode = 'direction'
    labels = support_labels()
    scene_path = REPAIR / 'view/scene/scene_transforms.npz'
    scene = np.load(scene_path)
    rotation = scene['rotation_ground_from_left'][:62]
    translation = scene['translation_ground_from_left_mm'][:62] / 1000
    if not (np.allclose(rotation[35:], rotation[35], atol=1e-12, rtol=0) and
            np.allclose(translation[35:], translation[35], atol=1e-12, rtol=0)):
        raise ValueError('The preregistered contact window requires one fixed ground frame')
    dest.mkdir(exist_ok=False)
    protocol = {'code': str(Path(__file__).relative_to(APP.parent)), 'command': [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        'baseline': str(reference_dir), 'input_frames': [0, 61], 'contact_frames': [35, 61] if contact else [],
        'source_body_run': proto['source_run'], 'surface_sets': str(SURFACE_PATH), 'fixed_ground': str(scene_path),
        'configuration': proto['configuration'], 'iterated_map_configuration': proto['iterated_map_configuration'],
        'single_change': ('Add normalized unilateral sole and support-ball height regularization to the matching direction MAP from35; all earlier frames unchanged.' if contact else
            'Change only the pixel/direction MAP Jacobian from central finite differences to automatic differentiation, including the one final pixel/pose information update.'),
        'jacobian': jacobian, 'contact_enabled': contact,
        'jacobian_code': 'realtime_app/tools/smplh_contact_autograd.py' if jacobian == 'autograd' else 'realtime_app/pose_app/smplh_iterated_motion_tracker.py',
        'numerical_control': 'Same float32 assets, forward objective, parameters,15 iteration budget, pixel gates, q/v/a/P process and covariance policy. Derivative publication still uses the original point finite differences.',
        'contact': {'tolerance_m': .005, 'penetration_scale_m': .005, 'height_scale_m': .010, 'height_target_m': SHOE_HEIGHT_M,
            'penetration_normalization': 'sqrt(108), independent of vertex count', 'support_height': 'true minimum of18 fixed ball vertices perfoot',
            'support_penalty': 'pseudo-Huber sqrt(1+((h-.025)/.010)^2)-1', 'slip': 'not introduced in this stage'},
        'support_labels': labels.tolist(), 'support_intervals_inclusive': {'right_ball': [35, 47], 'left_ball': [49, 61]},
        'label_evidence': 'Source stereo crops contact_source_review.jpg at35,39,43,47,51,55,59,61; right stationary during left swing, left stationary during right swing. Landing/lift uncertainty excluded. These labels assume the visible slipper front approximates model ball support; heels not labeled.',
        'information_policy': 'Engineering-regularized point estimate. One final original pixel/pose local information update; engineering curvature omitted. Changed pose can indirectly change pixel Jacobian/P. Not a full statistically consistent contact posterior.',
        'success': {'support_abs_median_and_p95_max_ratio': .5, 'body_hip_nonhip_median_p95_max_ratio': 1.05,
            'no_new_missing_restarts_or_penetrating_foot_frames': True, 'sole_height_lower_bound_m': SHOE_HEIGHT_M-.005,
            'no_more_unconverged': True, 'direction_median_max_deg': 'max(3, baseline*1.1)', 'direction_max_max_deg': 'baseline+2'},
        'stop': 'One62-frame experiment. No weight search, no smoothing, no448 on failure. A pass permits further contact-label validation, not automatic promotion.',
        'limits': 'Stage/ankle-anchored reference ground and assistant support hypotheses are not independent contact/3D truth. Bare-model surface target25mm is the user-specified shoe thickness. Source image review is offline; no causal contact classifier is claimed.'}
    if not contact:
        del protocol['success']['support_abs_median_and_p95_max_ratio']
        protocol['contact']['enabled'] = False
        protocol['information_policy'] = 'Original single pixel/pose local information update, using automatic derivatives; no engineering residual is active.'
        protocol['stop'] = 'One62-frame solver-only control. Same body/direction/availability/ground and non-increasing unconverged gates; no contact improvement is required or claimed.'
    save(dest / 'protocol.json', protocol)
    archive = {k: [] for k in ['predicted_coco', 'smplh_joints', 'transl', 'root_rotation', 'vposer_latent', 'body_rotations',
        'pose_available', 'segment_id', 'toe_m', 'velocity', 'acceleration', 'covariance']}
    rows, vertices, started = [], [], perf_counter()
    original = model.regressor.clone()
    with (dest / 'trajectory.jsonl').open('w', encoding='utf-8') as stream:
        for i in range(62):
            if i == 11:
                a, b = proto['hip_calibration_result']['alpha'], proto['hip_calibration_result']['thigh_fraction']
                model.regressor[11:13] = original[11:13] + a*(original[11:13]-original[[12, 11]]) + b*(original[13:15]-original[11:13])
                tr.last_point_jacobian = None
            tr.contact_rotation = rotation[i] if contact and i >= 35 else None
            tr.contact_translation, tr.support = translation[i], labels[i]
            tr.extra_pixels = None if i in HELDOUT or i == 0 else toes[i]
            result = tr.step(float(times[i]), *body[i])
            vertices.append(result['vertices_m'])
            for key, value in [('predicted_coco', result['coco_m']), ('smplh_joints', result['joints_m']),
                ('transl', result['translation_m']), ('root_rotation', result['root_rotation']), ('vposer_latent', result['vposer_latent']),
                ('body_rotations', result['body_rotations']), ('pose_available', result['pose_available']), ('segment_id', result['segment_id']),
                ('toe_m', result['toe_m']), ('velocity', tr.velocity.copy()), ('acceleration', tr.acceleration.copy()), ('covariance', tr.covariance.copy())]:
                archive[key].append(value)
            residual = contact_residuals(result['vertices_m'][None, SURFACE], rotation[i], translation[i], labels[i])[0] if contact and i >= 35 else np.empty(0)
            row = {'frame': i, 'status': result['status'], 'optimizer': result['iterated_update'], 'support_ball': labels[i].tolist(),
                'contact_active': contact and i >= 35, 'contact_final_cost': {'penetration': float(.5*sum(residual[:108]**2)), 'attachment': float(.5*sum(residual[108:]**2))},
                'body_used': result['used_views'].tolist(), 'body_reason': result['view_reason'].tolist(),
                'toe_used': getattr(tr, 'foot_used', np.zeros((2, 4), bool)).tolist(),
                'toe_reason': getattr(tr, 'foot_reason', np.full((2, 4), 'initial_not_reused', dtype=object)).tolist(),
                'derivative_count': int(result['derivative_valid'].sum())}
            stream.write(json.dumps(row) + '\n'); stream.flush()
            rows.append(row)
            if i % 10 == 0:
                print('frame', i, 'support', labels[i], 'optimizer', row['optimizer']['status'], flush=True)
    arrays = {k: np.asarray(v) for k, v in archive.items()}
    arrays.update(betas=model.betas.cpu().numpy(), timestamp_s=times, regressor=model.regressor.cpu().numpy(), support_ball=labels)
    reference = np.load(reference_dir / 'result.npz')
    baseline_rows = [json.loads(line) for line in (reference_dir / 'trajectory.jsonl').read_text().splitlines()]
    baseline_vertices = np.concatenate([model.evaluate_batch(reference['transl'][s:s+16], reference['root_rotation'][s:s+16],
        reference['vposer_latent'][s:s+16], True)['vertices_m'] for s in range(0, 62, 16)])
    a, base_sole, base_ball = summarize(reference, baseline_vertices, baseline_rows, body, toes, tr.cal, rotation, translation, labels)
    b, new_sole, new_ball = summarize(arrays, np.asarray(vertices), rows, body, toes, tr.cal, rotation, translation, labels)
    arrays.update(sole_min_m=new_sole, support_ball_height_m=new_ball)
    np.savez_compressed(dest / 'result.npz', **arrays)
    gates = {f'{window}_{part}_{stat}': b[window][part][stat] <= a[window][part][stat]*1.05
        for window in ['primary35_61', 'context11_61'] for part in ['hip', 'nonhip'] for stat in ['median_px', 'p95_px']}
    if contact:
        gates.update({f'support_{stat}_halved': b['support_ball_abs_height_mm'][stat] <= .5*a['support_ball_abs_height_mm'][stat] for stat in ['median', 'p95']})
    gates.update(no_missing=b['available'] == 62, no_restarts=b['segments'] == a['segments'],
        no_penetrating_foot_frames=b['penetrating_foot_frames35_61'] <= a['penetrating_foot_frames35_61'],
        sole_above_shoe_lower_bound=b['sole_min35_61_mm'] >= (SHOE_HEIGHT_M-.005)*1000, no_more_unconverged=b['primary_unconverged'] <= a['primary_unconverged'],
        direction_median_retained=b['withheld_direction_median_deg'] <= max(3., a['withheld_direction_median_deg']*1.1),
        direction_max_retained=b['withheld_direction_max_deg'] <= a['withheld_direction_max_deg']+2.)
    max_prefix_delta = max(float(np.max(abs(arrays[k][:35]-reference[k][:35]))) for k in ['transl', 'root_rotation', 'vposer_latent', 'covariance'])
    if contact:
        gates['inactive_prefix_unchanged'] = max_prefix_delta == 0.
    report = {'variants': {'direction_baseline': a, 'soft_contact' if contact else 'autograd_direction': b},
        'jacobian': jacobian, 'contact_enabled': contact, 'gates': gates, 'passed': all(gates.values()),
        'prefix0_34_max_difference': max_prefix_delta, 'runtime_s': perf_counter()-started,
        'decision': 'Continue label validation only if all gates pass; otherwise keep direction baseline and stop this contact candidate without weight search.',
        'limits': protocol['limits']}
    save(dest / 'comparison.json', report)
    np.savez_compressed(dest / 'contact_comparison.npz', baseline_sole_m=base_sole, candidate_sole_m=new_sole,
        baseline_ball_m=base_ball, candidate_ball_m=new_ball, support_ball=labels, timestamps_s=times)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jacobian', choices=['finite', 'autograd'], default='finite')
    parser.add_argument('--without-contact', action='store_true')
    args = parser.parse_args()
    run(args.jacobian, not args.without_contact)
