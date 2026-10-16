"""Test archived Sapiens2 toe pixels in the existing SMPL-H iterated tracker.

Only the observation set changes. Hidden heels are excluded. Saved image-read
references and three withheld detector frames never enter the fit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

APP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(APP), str(APP / 'tools')]
from pose_app.fisheye_camera import load_stereo_fisheye
from pose_app.rotation import model_to_raw_point
from pose_app.smplh_ground_constraints import load_sole_vertex_indices
from pose_app.smplh_iterated_motion_tracker import (
    IteratedMAPConfig,
    SMPLHIteratedMotionTracker,
)
from pose_app.smplh_motion_tracker import SMPLHMotionConfig, SMPLHObservationMap
from probe_smplh_orientation import REPAIR, ROOT, metrics, project, save
from run_stereo_pva_baseline import _load_predictions, _read_capture_times

OUT = REPAIR / 'foot_direction_20261016'
DETECTOR = ROOT.parent / 'G20261002_sapiens2_body_mainline448_v2'
TOE_VERTICES = [3216, 3226, 6617, 6624]
TOE_INDICES = [15, 16, 18, 19]
HELDOUT = [35, 47, 61]
EXTENSION_HELDOUT = [267, 335, 412]


class FootPointMap(SMPLHObservationMap):
    """Append four real surface landmarks; never alter any native articulation."""

    def evaluate_batch(self, translations, roots, latents, mesh=False):
        # Reuse the exact forward implementation, including frozen feet/hands.
        data = super().evaluate_batch(translations, roots, latents, mesh=True)
        data['coco_m'] = np.concatenate((data['coco_m'], data['vertices_m'][:, TOE_VERTICES]), axis=1)
        if not mesh:
            del data['vertices_m']
        return data


class FootPixelTracker(SMPLHIteratedMotionTracker):
    """Augment the same current-frame solve, with one final posterior update.

    Body publication remains12 joints. The inherited finite-difference model
    generates all16 pixel Jacobians during solving and the same12 body point
    Jacobians for derivative publication. No second pose correction/filter.
    """

    extra_pixels = None
    solve_with_feet = False
    observation_mode = 'points'

    def _pixels(self, points):
        if not self.solve_with_feet:
            return super()._pixels(points)
        points = points[:, 5:21]
        other = points @ self.cal.R_cam0_to_cam1.T + self.cal.T_cam0_to_cam1_mm / 1000
        if not np.isfinite(points).all() or min(points[..., 2].min(), other[..., 2].min()) <= 1e-4:
            return None
        projected = project(points, self.cal)
        if self.observation_mode == 'direction':
            toe = projected[:, :, 12:].reshape(-1, 2, 2, 2, 2)
            vector = toe[..., 1, :] - toe[..., 0, :]
            length = np.linalg.norm(vector, axis=-1)
            if np.any(length < 1e-6):
                return None
            measured = self.extra_pixels[..., :2].reshape(2, 2, 2, 2)
            observed_vector = measured[..., 1, :] - measured[..., 0, :]
            observed_length = np.linalg.norm(observed_vector, axis=-1)
            observed_length = np.where(np.isfinite(observed_length), observed_length, 0.)
            encoded = np.zeros_like(toe)
            encoded[..., 0, :] = vector / length[..., None] * observed_length[None, ..., None]
            projected[:, :, 12:] = encoded.reshape(-1, 2, 4, 2)
        return projected

    def _update(self, pixels, valid, reasons):
        self.foot_used = np.zeros((2, 4), bool)
        self.foot_reason = np.full((2, 4), 'withheld_or_absent', dtype=object)
        if self.extra_pixels is None or np.any(valid, axis=0).sum() < self.config.min_observed_joints:
            return super()._update(pixels, valid, reasons)
        # Apply the exact existing raw-pixel/score/epipolar gate to extra slots.
        padded = np.full((2, 12, 3), np.nan)
        padded[:, :4] = self.extra_pixels
        supported, why = self._supported_pixels(padded)
        foot_valid, foot_reason = supported[:, :4], why[:, :4]
        self.foot_reason = foot_reason.copy()
        extra = self.extra_pixels
        if self.observation_mode == 'direction':
            raw = extra[..., :2].reshape(2, 2, 2, 2)
            vector = raw[..., 1, :] - raw[..., 0, :]
            eligible = foot_valid.reshape(2, 2, 2).all(axis=-1) & (np.linalg.norm(vector, axis=-1) >= 16.)
            # One2D unit-direction residual per foot/view; the two original
            # landmarks are never counted as two independent angle sensors.
            extra = np.zeros_like(self.extra_pixels)
            extra[:, [0, 2], :2] = vector
            foot_valid = np.zeros_like(foot_valid)
            foot_valid[:, [0, 2]] = eligible
            foot_reason = np.full((2, 4), 'paired_direction_not_independent', dtype=object)
            foot_reason[:, [0, 2]] = np.where(eligible, 'available_direction', 'invalid_paired_toe_direction')
            self.foot_reason = foot_reason.copy()
        if not foot_valid.any():
            return super()._update(pixels, valid, reasons)
        combined = np.concatenate((pixels, extra), axis=1)
        combined_valid = np.concatenate((valid, foot_valid), axis=1)
        combined_reasons = np.concatenate((reasons, foot_reason), axis=1)
        self.solve_with_feet = True
        try:
            accepted, used, status = super()._update(combined, combined_valid, combined_reasons)
        finally:
            self.solve_with_feet = False
        reasons[:] = combined_reasons[:, :12]
        self.foot_used = used[:, 12:].copy()
        self.foot_reason = combined_reasons[:, 12:].copy()
        return accepted, used[:, :12], status

    def _output(self, now, status, used, reasons, observed_pixels=None):
        result = super()._output(now, status, used, reasons, observed_pixels)
        result['toe_m'] = result['coco_m'][17:].copy()
        result['coco_m'] = result['coco_m'][:17]
        return result


def load(count, model_class=FootPointMap, tracker_class=FootPixelTracker):
    proto = json.loads((REPAIR / 'hip_operator_feedback/protocol.json').read_text())
    params = json.loads((Path(proto['source_run']) / 'run_parameters.json').read_text())
    inp = params['inputs']
    model = model_class(Path(params['effective_initial_fit']),
        *[Path(inp[k]) for k in ('smplh_model', 'regressor', 'vposer_dir', 'mano_left', 'mano_right')], device='cuda')
    model.torch.set_num_threads(2)
    tracker = tracker_class(model, load_stereo_fisheye(inp['calibration_dir']),
        SMPLHMotionConfig(**proto['configuration']), IteratedMAPConfig(**proto['iterated_map_configuration']))
    body = np.stack([_load_predictions(Path(inp[f'{s}_predictions']), s) for s in ('left', 'right')], axis=1)[:count, :, 5:17]
    body[0] = np.asarray(json.loads(Path(params['first_frame_observations']).read_text())['keypoints'])[:, 5:17]
    toes = np.full((count, 2, 4, 3), np.nan)
    for view, (side, rotation) in enumerate([('left', 'ccw90'), ('right', 'cw90')]):
        with (DETECTOR / f'{side}_sapiens_raw.jsonl').open(encoding='utf-8') as stream:
            for frame, line in enumerate(stream):
                if frame >= count:
                    break
                row = json.loads(line)
                if row['pair_id'] != frame or row['file_name'] != f'pair_{frame:04d}.png':
                    raise ValueError('Archived toe/body image identities differ')
                if not row['instances']:
                    continue
                person = row['instances'][0]  # Same frozen prompt/instance as Sapiens body source.
                for point, idx in enumerate(TOE_INDICES):
                    x, y = person['keypoints308'][idx]
                    toes[frame, view, point] = (*model_to_raw_point(x, y, 1920, 1080, rotation), person['keypoint_scores'][idx])
    times = _read_capture_times(Path(inp['input_pairs']), 448)[0][:count]
    return proto, model, tracker, body, toes, times


def direction_error(points, observed):
    """Per-view transverse big-to-small toe direction, with no heel invention."""
    a = points.reshape(-1, 2, 2, 2)
    b = observed.reshape(-1, 2, 2, 2)
    da, db = a[:, :, 1] - a[:, :, 0], b[:, :, 1] - b[:, :, 0]
    angle = np.arctan2(da[..., 1], da[..., 0]) - np.arctan2(db[..., 1], db[..., 0])
    return np.abs(np.rad2deg(np.arctan2(np.sin(angle), np.cos(angle))))


def compare(arrays, model, tracker, body, toes, R, t, sole, rows, mode):
    count = len(body)
    full = count == 448
    selected = HELDOUT + EXTENSION_HELDOUT if full else HELDOUT
    name = 'hip_operator_feedback_full448' if full else 'hip_operator_feedback'
    reference = np.load(REPAIR / name / 'result.npz')
    # Bounded batches avoid448 simultaneous full6890-vertex model evaluations.
    batches = [model.evaluate_batch(reference['transl'][i:i+32], reference['root_rotation'][i:i+32],
        reference['vposer_latent'][i:i+32], True) for i in range(0, count, 32)]
    data = {k: np.concatenate([v[k] for v in batches]) for k in ['coco_m', 'vertices_m']}
    base_toes = data['coco_m'][:, 17:]
    base_z = np.einsum('nij,nkj->nki', R, data['vertices_m'])[:, :, 2] + t[:, None, 2]
    base_sole = np.stack([base_z[:, ids].min(axis=1) for ids in sole], axis=1)
    baseline_rows = [json.loads(line) for line in (REPAIR / name / 'trajectory.jsonl').read_text().splitlines()]
    reports = {'baseline': metrics(reference, base_sole, body, R, t, baseline_rows),
               'toe_pixels': metrics(arrays, arrays['sole_min_m'], body, R, t, rows)}
    raw_refs = json.loads((OUT / 'heldout_reference.json').read_text())
    manual, ref_mask = np.full((3, 2, 4, 2), np.nan), np.zeros((3, 2, 4), bool)
    for j, frame in enumerate(HELDOUT):
        for v, (side, rotation) in enumerate([('left', 'ccw90'), ('right', 'cw90')]):
            for k, point in enumerate(raw_refs['references'][str(frame)][side]):
                if point is not None:
                    manual[j, v, k] = model_to_raw_point(*point, 1920, 1080, rotation)
                    ref_mask[j, v, k] = raw_refs['usable_mask'][side][k]
    for name, pred in [('baseline', base_toes), ('toe_pixels', arrays['toe_m'])]:
        pixels = project(pred, tracker.cal)
        error = np.linalg.norm(pixels[selected] - toes[selected, ..., :2], axis=-1)
        image_error = np.linalg.norm(pixels[HELDOUT] - manual, axis=-1)[ref_mask]
        reports[name]['heldout_detector_toe'] = {'slots': int(error.size), 'median_px': float(np.median(error)), 'p95_px': float(np.percentile(error, 95))}
        angles = direction_error(pixels[selected], toes[selected, ..., :2])
        reports[name]['heldout_transverse_detector_direction'] = {'foot_view_slots': int(angles.size), 'median_deg': float(np.median(angles)), 'max_deg': float(angles.max())}
        reports[name]['heldout_manual_visible_big_toe'] = {'slots': int(image_error.size), 'median_px': float(np.median(image_error)), 'p95_px': float(np.percentile(image_error, 95))}
        reports[name]['heldout_toe_errors_px'] = error.tolist()
        reports[name]['heldout_direction_errors_deg'] = angles.tolist()
        if full:
            extra_error = np.linalg.norm(pixels[EXTENSION_HELDOUT] - toes[EXTENSION_HELDOUT, ..., :2], axis=-1)
            extra_angle = direction_error(pixels[EXTENSION_HELDOUT], toes[EXTENSION_HELDOUT, ..., :2])
            reports[name]['extension_withheld_toes'] = {'median_px': float(np.median(extra_error)), 'p95_px': float(np.percentile(extra_error, 95)),
                'direction_median_deg': float(np.median(extra_angle)), 'direction_max_deg': float(extra_angle.max())}
    a, b = reports['baseline'], reports['toe_pixels']
    gates = {f'{window}_{part}_{stat}': b[window][part][stat] <= 1.05 * a[window][part][stat]
        for window in ['primary', 'heldout'] for part in ['hip', 'nonhip'] for stat in ['median_px', 'p95_px']}
    gates.update(manual_toe_median_improves25pct=b['heldout_manual_visible_big_toe']['median_px'] <= .75 * a['heldout_manual_visible_big_toe']['median_px'],
        detector_direction_improves30pct=b['heldout_transverse_detector_direction']['median_deg'] <= .7 * a['heldout_transverse_detector_direction']['median_deg'],
        no_missing=b['available'] == count, no_restarts=b['segments'] == a['segments'],
        sole_min_within5mm=b['sole_min35_61_mm'] >= a['sole_min35_61_mm'] - 5,
        no_more_penetrating_foot_frames=b['penetrating_foot_frames35_61'] <= a['penetrating_foot_frames35_61'])
    if full:
        for variant, data_body in [('baseline', reference), ('toe_pixels', arrays)]:
            norms = np.linalg.norm(project(data_body['predicted_coco'][:, 5:17], tracker.cal) - body[..., :2], axis=-1)
            reports[variant]['all448_raw_body'] = {}
            for group, ids in [('hip', [6, 7]), ('nonhip', [0, 1, 2, 3, 4, 5, 8, 9, 10, 11])]:
                values = norms[:, :, ids]
                reports[variant]['all448_raw_body'][group] = {'slots': int(values.size), 'median_px': float(np.median(values)), 'p95_px': float(np.percentile(values, 95))}
            reports[variant]['all448_sole_below_minus5mm'] = int((base_sole < -.005).sum()) if variant == 'baseline' else int((arrays['sole_min_m'] < -.005).sum())
            reports[variant]['all448_sole_min_mm'] = float(base_sole.min() * 1000) if variant == 'baseline' else float(arrays['sole_min_m'].min() * 1000)
        for group in ['hip', 'nonhip']:
            for stat in ['median_px', 'p95_px']:
                gates[f'all448_{group}_{stat}'] = b['all448_raw_body'][group][stat] <= 1.05 * a['all448_raw_body'][group][stat]
        gates.update(extension_toe_median_improves25pct=b['extension_withheld_toes']['median_px'] <= .75 * a['extension_withheld_toes']['median_px'],
            extension_direction_improves30pct=b['extension_withheld_toes']['direction_median_deg'] <= .7 * a['extension_withheld_toes']['direction_median_deg'],
            full_sole_min_within5mm=b['all448_sole_min_mm'] >= a['all448_sole_min_mm'] - 5,
            full_no_more_penetrating_foot_frames=b['all448_sole_below_minus5mm'] <= a['all448_sole_below_minus5mm'])
    suffix = ('_direction' if mode == 'direction' else '') + ('_full448' if full else '')
    np.savez_compressed(OUT / f'comparison_points{suffix}.npz', baseline_toe_m=base_toes, candidate_toe_m=arrays['toe_m'],
                        detector_raw=toes, manual_raw=manual, manual_usable=ref_mask, heldout_frames=selected)
    report = {'variants': reports, 'gates': gates, 'passed': all(gates.values()),
        'decision': 'Proceed only if all gates pass. A failed body/ground gate rejects this whole-latent foot candidate; do not tune weights or claim independent3D direction.',
        'limits': 'Detector-heldout is same-source2D consistency, not independent truth. Manual6 visible big-toe references are assistant image readings with8px uncertainty; no heel or contact truth.'}
    save(OUT / f'comparison{suffix}.json', report)
    print(json.dumps(report, indent=2), flush=True)


def inspect_saved(count, mode):
    """Check saved pixel placement and withheld usage without rerunning tracking."""
    from collections import Counter

    import cv2

    full = count == 448
    suffix = ('_direction' if mode == 'direction' else '') + ('_full448' if full else '')
    points = np.load(OUT / f'comparison_points{suffix}.npz')
    proto = json.loads((REPAIR / 'hip_operator_feedback/protocol.json').read_text())
    params = json.loads((Path(proto['source_run']) / 'run_parameters.json').read_text())
    cal = load_stereo_fisheye(params['inputs']['calibration_dir'])
    projections = {name: project(points[key], cal) for name, key in
                   [('baseline', 'baseline_toe_m'), ('candidate', 'candidate_toe_m')]}
    pairs = Path(params['inputs']['input_pairs'])
    panels = []
    frames = points['heldout_frames'].tolist()
    for frame in frames:
        tiles = []
        for view, (side, folder, rotation) in enumerate([('left', 'left_ccw90', 'ccw90'), ('right', 'right_cw90', 'cw90')]):
            image = cv2.imread(str(pairs / folder / f'pair_{frame:04d}.png'))
            for name, color in [('baseline', (70, 70, 255)), ('candidate', (70, 255, 70))]:
                raw = projections[name][frame, view]
                # Inverse of model_to_raw is an orthogonal90deg image rotation.
                xy = np.column_stack((raw[:, 1], 1919 - raw[:, 0])) if side == 'left' else np.column_stack((1079 - raw[:, 1], raw[:, 0]))
                for foot in [0, 1]:
                    pair = xy[2*foot:2*foot+2]
                    if np.isfinite(pair).all():
                        cv2.line(image, tuple(np.rint(pair[0]).astype(int)), tuple(np.rint(pair[1]).astype(int)), color, 2)
                for k, pt in enumerate(xy):
                    if np.isfinite(pt).all():
                        cv2.circle(image, tuple(np.rint(pt).astype(int)), 5, color, 2)
                        cv2.putText(image, str(k), tuple(np.rint(pt + [5, -5]).astype(int)), 0, .45, color, 1)
            raw = points['detector_raw'][frame, view, :, :2]
            xy = np.column_stack((raw[:, 1], 1919 - raw[:, 0])) if rotation == 'ccw90' else np.column_stack((1079 - raw[:, 1], raw[:, 0]))
            for pt in xy:
                if np.isfinite(pt).all():
                    cv2.drawMarker(image, tuple(np.rint(pt).astype(int)), (0, 255, 255), cv2.MARKER_CROSS, 13, 2)
            cv2.putText(image, f'{frame} {side}: red=A green=B yellow=withheld', (8, 1430), 0, .65, (255, 255, 255), 2)
            tiles.append(cv2.resize(image[1400:], (648, 312)))
        panels.append(np.hstack(tiles))
    cv2.imwrite(str(OUT / f'heldout_projection{suffix}.jpg'), np.vstack(panels))
    stem = 'toe_direction' if mode == 'direction' else 'toe_pixels'
    dest = OUT / (stem + ('_full448' if full else '_prefix'))
    rows = [json.loads(line) for line in (dest / 'trajectory.jsonl').read_text().splitlines()]
    reasons = Counter(reason for row in rows for view in row['toe_reason'] for reason in view)
    used = np.asarray([row['toe_used'] for row in rows])
    error = np.linalg.norm(points['detector_raw'][HELDOUT, ..., :2] - points['manual_raw'], axis=-1)[points['manual_usable']]
    report = {'frames': count, 'withheld_toe_frames': frames, 'withheld_toe_used_slots': int(used[frames].sum()),
        'toe_used_slots': int(used.sum()), 'toe_reason_counts': dict(reasons),
        'source_raw_body': str(Path(proto['source_run']) / 'run_parameters.json'),
        'detector_vs_assistant6_toe_median_px': float(np.median(error)), 'detector_vs_assistant6_toe_p95_px': float(np.percentile(error, 95)),
        'limits': 'Assistant references are operational image readings; image-toe direction is not independent3D truth. No heel inferred.'}
    if full:
        prefix = np.load(OUT / (stem + '_prefix/result.npz'))
        result = np.load(dest / 'result.npz')
        report['full_prefix_coco_max_difference_m'] = float(np.max(abs(prefix['predicted_coco'] - result['predicted_coco'][:62])))
        report['beta_unchanged'] = bool(np.array_equal(prefix['betas'], result['betas']))
    save(OUT / f'saved_review{suffix}.json', report)
    print(json.dumps(report, indent=2))


def run(count, mode):
    full = count == 448
    prefix_comparison = 'comparison_direction.json' if mode == 'direction' else 'comparison.json'
    if full and not json.loads((OUT / prefix_comparison).read_text())['passed']:
        raise ValueError('Prefix gate failed; do not run448')
    proto, model, tracker, body, toes, times = load(count)
    tracker.observation_mode = mode
    stem = 'toe_direction' if mode == 'direction' else 'toe_pixels'
    dest = OUT / (stem + ('_full448' if full else '_prefix'))
    dest.mkdir(exist_ok=False)
    protocol = json.loads((OUT / 'protocol.json').read_text())
    protocol.update(command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        code='realtime_app/tools/probe_smplh_foot_observations.py', source_body_run=proto['source_run'],
        configuration=proto['configuration'], iterated_map_configuration=proto['iterated_map_configuration'])
    if mode == 'direction':
        protocol.update(single_change='Relative to the point-toe candidate, replace only the foot residual with observed-length times(unit predicted big-to-small toe vector minus unit observed vector). One vector residual perfoot/view. Do not fit toe midpoint or projected width.',
            rationale='Point-toe full448 improved direction but moved ankles: medians10.82/10.86 to16.98/16.42px; failed nonhip aggregate+9.32%. Test whether absolute toe location/width conflict causes this tradeoff; no weight sweep.',
            direction_pair_minimum_length_px=16, observation_mode=mode,
            stop='One direction-only prefix. Reject on any original prefix gate; no adjusting weights/thresholds. Full448 only after passing. All earlier six detector withheld frames are now development diagnostics, not fresh independent holdout.')
    if full:
        protocol.update(input_frames=[0, 447], withheld_toe_frames=HELDOUT + EXTENSION_HELDOUT,
            full_gate='Hip/nonhip raw median/P95 each <= original*1.05; all448 available/no restart; sole min within5mm and no additional foot-frame below-5mm; new withheld267/335/412 toe median improves25% and transverse direction30%. One pass, no sweep.',
            limits=protocol['limits'] + ' The six manual references and first three withheld frames were used in prefix screening; only267/335/412 are untouched extension frames. Detector direction is not physical truth.')
    save(dest / 'protocol.json', protocol)
    scene = np.load(REPAIR / 'view/scene/scene_transforms.npz')
    R, t = scene['rotation_ground_from_left'][:count], scene['translation_ground_from_left_mm'][:count] / 1000
    sole = load_sole_vertex_indices(APP.parent / 'research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/contact_surface_sets_smplh_v1/contact_vertex_sets.json')
    archive = {k: [] for k in ['predicted_coco', 'smplh_joints', 'transl', 'root_rotation', 'vposer_latent', 'body_rotations',
        'pose_available', 'segment_id', 'toe_m', 'sole_min_m', 'velocity', 'acceleration', 'covariance']}
    original = model.regressor.clone()
    rows, started = [], perf_counter()
    with (dest / 'trajectory.jsonl').open('w', encoding='utf-8') as stream:
        for i in range(count):
            if i == 11:
                a = proto['hip_calibration_result']['alpha']
                b = proto['hip_calibration_result']['thigh_fraction']
                model.regressor[11:13] = original[11:13] + a * (original[11:13] - original[[12, 11]]) + b * (original[13:15] - original[11:13])
                tracker.last_point_jacobian = None
            withheld = i in HELDOUT or (full and i in EXTENSION_HELDOUT)
            tracker.extra_pixels = None if withheld or i == 0 else toes[i]
            result = tracker.step(float(times[i]), *body[i])
            world = result['vertices_m'] @ R[i].T + t[i]
            for key, value in [('predicted_coco', result['coco_m']), ('smplh_joints', result['joints_m']),
                ('transl', result['translation_m']), ('root_rotation', result['root_rotation']),
                ('vposer_latent', result['vposer_latent']), ('body_rotations', result['body_rotations']),
                ('pose_available', result['pose_available']), ('segment_id', result['segment_id']), ('toe_m', result['toe_m']),
                ('sole_min_m', [world[ids, 2].min() for ids in sole]), ('velocity', tracker.velocity.copy()),
                ('acceleration', tracker.acceleration.copy()), ('covariance', tracker.covariance.copy())]:
                archive[key].append(value)
            row = {'frame': i, 'status': result['status'], 'optimizer': result['iterated_update'],
                'body_used': result['used_views'].tolist(), 'body_reason': result['view_reason'].tolist(),
                'toe_used': getattr(tracker, 'foot_used', np.zeros((2, 4), bool)).tolist(),
                'toe_reason': getattr(tracker, 'foot_reason', np.full((2, 4), 'initial_not_reused', dtype=object)).tolist(),
                'foot_withheld': withheld, 'derivative_count': int(result['derivative_valid'].sum())}
            stream.write(json.dumps(row) + '\n'); stream.flush()
            rows.append(row)
            if i % 10 == 0:
                print('frame', i, 'feet used', np.sum(row['toe_used']), 'optimizer', row['optimizer']['status'], flush=True)
    arrays = {k: np.asarray(v) for k, v in archive.items()}
    arrays.update(betas=model.betas.cpu().numpy(), timestamp_s=times, regressor=model.regressor.cpu().numpy())
    np.savez_compressed(dest / 'result.npz', **arrays)
    save(dest / 'runtime.json', {'runtime_s': perf_counter() - started})
    compare(arrays, model, tracker, body, toes, R, t, sole, rows, mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, choices=[62, 448], default=62)
    parser.add_argument('--inspect-saved', action='store_true')
    parser.add_argument('--observation-mode', choices=['points', 'direction'], default='points')
    args = parser.parse_args()
    inspect_saved(args.frames, args.observation_mode) if args.inspect_saved else run(args.frames, args.observation_mode)
