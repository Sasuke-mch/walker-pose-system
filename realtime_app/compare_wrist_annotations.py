"""Compare existing detector wrists with corrected sparse manual pixels.

No fitting, nearest-to-label selection, interpolation or acceptance promotion.
WiLoR is a model-projected wrist with existing body-based association.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from check_mainline_regression import fit_arguments
from pose_app.body_observations import raw_side
from pose_app.fisheye_camera import load_stereo_fisheye
from pose_app.smplh_hand_observation import read_view
from walker_tools.hands.solve_annotated_wrist_targets import upright_to_raw, wrist_points, solve_point


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def stats(values):
    x = np.asarray(values, float)
    if not len(x):
        return {'count': 0, 'mean': None, 'rms': None, 'median': None, 'p95': None, 'max': None}
    if not np.isfinite(x).all():
        raise ValueError('nonfinite metric')
    return {'count': len(x), 'mean': float(x.mean()), 'rms': float(np.sqrt(np.mean(x*x))),
            'median': float(np.median(x)), 'p95': float(np.percentile(x, 95)), 'max': float(x.max())}


def compare_point(prediction, label, reason='unavailable'):
    if label['state'] != 'present':
        return {'state': 'manual_unavailable', 'reason': label.get('reason')}
    p = np.asarray(prediction, float)
    if p.shape != (2,) or not np.isfinite(p).all():
        return {'state': 'prediction_unavailable', 'reason': reason}
    raw = np.asarray(label['raw_px'])
    residual = p - raw
    return {'state': 'compared', 'prediction_raw_px': p.tolist(), 'manual_raw_px': raw.tolist(),
            'delta_raw_px': residual.tolist(), 'error_px': float(np.linalg.norm(residual)),
            'in_raw_bounds': bool(0 <= p[0] < 1920 and 0 <= p[1] < 1080)}


def summarize(records, methods, common=False):
    rows = records
    if common:
        rows = [r for r in rows if all(r['methods'][m]['state'] == 'compared' for m in methods)]
    out = {}
    for method in methods:
        valid = [r for r in rows if r['methods'][method]['state'] == 'compared']
        out[method] = {'all': stats([r['methods'][method]['error_px'] for r in valid]),
                       'eligible_manual_points': sum(r['manual']['state'] == 'present' for r in rows)}
        for camera in ('left', 'right'):
            for hand in ('left', 'right'):
                subset = [r for r in valid if r['camera'] == camera and r['hand'] == hand]
                out[method][camera + '_camera_' + hand + '_hand'] = stats([r['methods'][method]['error_px'] for r in subset])
        out[method]['out_of_bounds_compared'] = sum(not r['methods'][method]['in_raw_bounds'] for r in valid)
    return out


def unique_wrists(path, n):
    """Detector-independent sensitivity: only one same-side candidate allowed."""
    points = np.full((n, 2, 2), np.nan)
    reasons = {}
    seen = set()
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        f = row['frame_index']
        if f in seen or not 0 <= f < n or Path(row['image']).stem != f'pair_{f:04d}':
            raise ValueError('WiLoR frame/image mismatch')
        seen.add(f)
        for h, side in enumerate(('left', 'right')):
            candidates = [c for c in row['records'] if c['side'] == side]
            reasons[f, side] = 'no_candidate' if not candidates else 'multiple_candidates_no_label_selection'
            if len(candidates) == 1:
                c = candidates[0]
                if c['pixel_frame'] != 'raw_fisheye':
                    raise ValueError('WiLoR coordinate mismatch')
                p = np.asarray(c['keypoints_2d_raw_fisheye'], float)
                if p.shape != (21, 2):
                    raise ValueError('WiLoR shape mismatch')
                points[f, h] = p[0]
                reasons[f, side] = 'unique_same_side_candidate'
    if seen != set(range(n)):
        raise ValueError('WiLoR incomplete frame range')
    return points, reasons


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--annotations', type=Path, required=True)
    ap.add_argument('--route', type=Path, default=Path('realtime_app/configs/current_mainline.json'))
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError('refuse existing output')
    route = json.loads(args.route.read_text(encoding='utf-8'))
    manifest = json.loads((args.annotations / 'manifest.json').read_text(encoding='utf-8'))
    commands = {m: Path(route[key]['result']) / 'fit_command.json'
                for m, key in [('PMPose', 'matched_control'), ('Sapiens2', 'body_candidate')]}
    fit_args = {m: fit_arguments(path) for m, path in commands.items()}
    for key in ('calibration_dir', 'wilor_left', 'wilor_right'):
        if Path(getattr(fit_args['PMPose'], key)).resolve() != Path(getattr(fit_args['Sapiens2'], key)).resolve():
            raise ValueError('different ' + key)
    cal = load_stereo_fisheye(fit_args['PMPose'].calibration_dir)
    if (cal.image_width, cal.image_height) != (1920, 1080):
        raise ValueError('unexpected calibration size')
    bodies = {}
    for m, options in fit_args.items():
        bodies[m] = {}
        for camera in ('left', 'right'):
            rows = raw_side(getattr(options, camera + '_raw'), camera)
            if sorted(rows) != list(range(route['frames'])):
                raise ValueError('body frame range mismatch')
            bodies[m][camera] = np.stack([rows[i] for i in range(route['frames'])])
    protocol = {'schema': 'sparse_manual_wrist_comparison_v1', 'annotations': str(args.annotations.resolve()),
        'frames': manifest['frames'], 'sequence_frames': route['frames'], 'inputs': {m: {k: str(getattr(a, k)) for k in ('left_raw', 'right_raw', 'wilor_left', 'wilor_right', 'calibration_dir')}
                                               for m, a in fit_args.items()},
        'primary_methods': ['PMPose', 'Sapiens2', 'WiLoR'],
        'wilor_primary': 'existing read_view using PMPose wrist association; no manual selection',
        'sensitivity': 'same WiLoR pixels with Sapiens2 association, plus unique-side candidates without body association',
        'coordinates': 'raw1920x1080 fisheye; anatomical COCO9/10 versus MANO projected joint0; Euclidean pixels',
        'criteria': 'all available manual points and common-support comparison, retain every unavailable/rejected state',
        'three_dimensional': 'per-pair solve_point with unchanged geometry gate; rejected manual geometry is diagnostic only',
        'limits': '10 labelled frames in one sequence; visible manual wrist, not anatomical3D truth; no fitted SMPLH or shared wrist target used'}
    args.output.mkdir(parents=True)
    write(args.output / 'protocol.json', protocol)
    predictions, reasons = {}, {}
    for camera in ('left', 'right'):
        path = getattr(fit_args['PMPose'], 'wilor_' + camera)
        for body_name, name in [('PMPose', 'WiLoR'), ('Sapiens2', 'WiLoR_Sapiens2_association')]:
            p, _, audit = read_view(path, bodies[body_name][camera], (1920, 1080), camera)
            predictions[name, camera] = p[:, :, 0]
            reasons[name, camera] = {}
            for row in audit:
                k = row['frame_index'], row['hand']
                if row['selected'] or k not in reasons[name, camera]:
                    reasons[name, camera][k] = row['reason']
            write(args.output / f'{name}_{camera}_association.json', audit)
        predictions['WiLoR_unique_candidate', camera], reasons['WiLoR_unique_candidate', camera] = unique_wrists(path, route['frames'])
    names = ['PMPose', 'Sapiens2', 'WiLoR', 'WiLoR_Sapiens2_association', 'WiLoR_unique_candidate']
    records, seen = [], set()
    for e in manifest['images']:
        f, camera = e['frame_index'], e['camera']
        if (f, camera) in seen or f not in manifest['frames']:
            raise ValueError('manual frame duplicate/mismatch')
        seen.add((f, camera))
        image, source = cv2.imread(e['image']), cv2.imread(e['source'])
        if image is None or source is None or image.shape[:2] != (1920, 1080) or not np.array_equal(image, source):
            raise ValueError('annotation image/source mismatch')
        labels = wrist_points(Path(e['annotation']))
        for h, hand in enumerate(('left', 'right')):
            label = labels[hand]
            if label['state'] == 'present':
                label['raw_px'] = upright_to_raw(label['upright_px'], camera).tolist()
            methods = {}
            for name in names:
                p = bodies[name][camera][f, 9+h, :2] if name in bodies else predictions[name, camera][f, h]
                reason = 'nonfinite_detector_point' if name in bodies else reasons[name, camera].get((f, hand), 'unavailable')
                methods[name] = compare_point(p, label, reason)
                if name in bodies:
                    methods[name]['detector_confidence'] = float(bodies[name][camera][f, 9+h, 2])
            records.append({'frame_index': f, 'camera': camera, 'hand': hand, 'manual': label, 'methods': methods})
    if seen != {(f, c) for f in manifest['frames'] for c in ('left', 'right')}:
        raise ValueError('manual manifest incomplete')
    by_key = {(r['frame_index'], r['camera'], r['hand']): r for r in records}
    stereo = []
    for f in manifest['frames']:
        for hand in ('left', 'right'):
            pair = [by_key[f, c, hand] for c in ('left', 'right')]
            row = {'frame_index': f, 'hand': hand, 'methods': {}}
            if not all(r['manual']['state'] == 'present' for r in pair):
                row['manual'] = {'state': 'unavailable', 'reason': 'missing_manual_stereo_point'}
                row['methods'] = {name: {'state': 'unavailable', 'reason': 'missing_manual_stereo_point'} for name in names}
            else:
                row['manual'] = solve_point(*[r['manual']['raw_px'] for r in pair], cal)
                for name in names:
                    if not all(r['methods'][name]['state'] == 'compared' for r in pair):
                        row['methods'][name] = {'state': 'unavailable', 'reason': 'missing_prediction_stereo_point'}
                        continue
                    solved = solve_point(*[r['methods'][name]['prediction_raw_px'] for r in pair], cal)
                    if 'wrist_left_camera_m' in solved and 'wrist_left_camera_m' in row['manual']:
                        solved['manual_distance_mm'] = float(np.linalg.norm(np.asarray(solved['wrist_left_camera_m'])-row['manual']['wrist_left_camera_m'])*1000)
                    row['methods'][name] = solved
            stereo.append(row)
    summary = {'all_available': summarize(records, names), 'common_primary': summarize(records, names[:3], True),
        'common_all_association_variants': summarize(records, names, True),
        'manual_points_present': sum(r['manual']['state'] == 'present' for r in records),
        'manual_points_unavailable': sum(r['manual']['state'] != 'present' for r in records),
        'manual_stereo_accepted': sum(r['manual']['state'] == 'accepted_manual_reference_geometry' for r in stereo),
        'stereo': {}}
    for name in names:
        available = [r for r in stereo if 'manual_distance_mm' in r['methods'].get(name, {})]
        gated = [r for r in available if r['manual']['state'] == r['methods'][name]['state'] == 'accepted_manual_reference_geometry']
        summary['stereo'][name] = {'all_finite_diagnostic_mm': stats([r['methods'][name]['manual_distance_mm'] for r in available]),
                                 'both_geometry_accepted_mm': stats([r['methods'][name]['manual_distance_mm'] for r in gated])}
    primary_stereo = [r for r in stereo if all('manual_distance_mm' in r['methods'][m] for m in names[:3])]
    summary['stereo_common_primary_diagnostic_mm'] = {
        m: stats([r['methods'][m]['manual_distance_mm'] for r in primary_stereo]) for m in names[:3]}
    summary['stereo_common_primary_geometry_accepted_pairs'] = sum(
        r['manual']['state'] == 'accepted_manual_reference_geometry'
        and all(r['methods'][m]['state'] == 'accepted_manual_reference_geometry' for m in names[:3])
        for r in primary_stereo)
    write(args.output / 'points.json', records)
    write(args.output / 'stereo_diagnostics.json', stereo)
    write(args.output / 'summary.json', summary)
    with (args.output / 'points.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['frame', 'camera', 'hand', 'method', 'state', 'reason', 'error_px', 'delta_x_px', 'delta_y_px', 'in_bounds'])
        writer.writeheader()
        for r in records:
            for name, v in r['methods'].items():
                delta = v.get('delta_raw_px', [None, None])
                writer.writerow({'frame': r['frame_index'], 'camera': r['camera'], 'hand': r['hand'], 'method': name,
                    'state': v['state'], 'reason': v.get('reason'), 'error_px': v.get('error_px'), 'delta_x_px': delta[0], 'delta_y_px': delta[1], 'in_bounds': v.get('in_raw_bounds')})
    print(json.dumps({'manual_points': summary['manual_points_present'],
                     'common_primary': {m: v['all'] for m, v in summary['common_primary'].items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
