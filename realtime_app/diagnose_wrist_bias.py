"""Read-only wrist bias diagnosis from existing predictions and sparse labels.

No model inference, point correction or changes to geometry acceptance.
Directions are evaluated with detector-elbow and manual-wrist axes separately;
both use estimated elbows and are not independent anatomical axes.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from pose_app.body_observations import raw_side
from pose_app.fisheye_camera import load_stereo_fisheye, fisheye_project_numpy

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'research_records/engineering_validation/G20261003_single_model_wrist_processing_v2'
LABELS = ROOT/'research_records/engineering_validation/G20261003_wrist_detector_manual_v2'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summary(rows):
    return {'n': len(rows), 'toward_elbow': sum(x['along_px'] < 0 for x in rows),
            **{k: float(np.mean([x[k] for x in rows])) if rows else None
               for k in ('error_px', 'along_px', 'absolute_across_px', 'extension_fraction')}}


def correlation(a, b):
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def stratified(rows, predictor):
    x = np.array([r[predictor] for r in rows])
    y = np.array([r['error_px'] for r in rows])
    dx, dy = x.copy(), y.copy()
    groups = {}
    for key in sorted({(r['camera'], r['hand']) for r in rows}):
        ids = [i for i, r in enumerate(rows) if (r['camera'], r['hand']) == key]
        dx[ids] -= x[ids].mean()
        dy[ids] -= y[ids].mean()
        groups['_'.join(key)] = {'n': len(ids), 'pearson': correlation(x[ids], y[ids])}
    return {'pooled': correlation(x, y), 'within_camera_hand_centered': correlation(dx, dy), 'groups': groups,
            'boundary': 'Repeated frames of one subject; correlation is descriptive and not causal.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    labels = read(LABELS/'points.json')
    protocol = read(LABELS/'protocol.json')
    cal = load_stereo_fisheye(ROOT/'realtime_app/calibration/results')
    sparse = [json.loads(s) for s in (BASE/'sparse.jsonl').read_text().splitlines()]
    lookup = {(r['frame'], r['camera'], r['variant']): r for r in sparse}
    bodies = {m: {c: raw_side(Path(paths[c+'_raw']), c) for c in ('left', 'right')}
              for m, paths in protocol['inputs'].items()}
    records = []
    for label in labels:
        if label['manual']['state'] != 'present':
            continue
        f, c, hand = label['frame_index'], label['camera'], label['hand']
        elbow, wrist = (7, 9) if hand == 'left' else (8, 10)
        manual = np.array(label['manual']['raw_px'])
        base = lookup[f, c, 'full_flip']
        e = np.array(base['raw'][elbow])
        axis = manual-e
        axis /= np.linalg.norm(axis)
        original_direction = np.array(base['raw'][wrist])-e
        original_direction /= np.linalg.norm(original_direction)
        predictions = {v: (np.array(lookup[f, c, v]['raw'][wrist]),
                           np.array(lookup[f, c, v]['raw'][elbow]))
                       for v in sorted({r['variant'] for r in sparse})}
        for m in ('PMPose', 'WiLoR'):
            item = label['methods'][m]
            if item['state'] == 'compared':
                predictions[m] = (np.array(item['prediction_raw_px']),
                                  bodies['PMPose'][c][f][elbow, :2] if m == 'PMPose' else e)
        for method, (p, own_elbow) in predictions.items():
            delta = p-manual
            own_axis = manual-own_elbow
            own_axis /= np.linalg.norm(own_axis)
            K = cal.K0 if c == 'left' else cal.K1
            radial = manual-K[:2, 2]
            records.append({'frame': f, 'camera': c, 'hand': hand, 'method': method,
                'prediction_raw_px': p.tolist(), 'manual_raw_px': manual.tolist(),
                'error_px': float(np.linalg.norm(delta)), 'along_px': float(delta@axis),
                'along_original_detector_axis_px': float(delta@original_direction),
                'along_own_elbow_axis_px': float(delta@own_axis),
                'absolute_across_px': float(abs(delta@np.array([-axis[1], axis[0]]))),
                'extension_fraction': float(-(delta@axis)/np.linalg.norm(manual-e)),
                'radial_px': float(np.linalg.norm(radial)),
                'signed_radial_error_px': float(delta@radial/np.linalg.norm(radial)),
                'input_forearm_px': base['input_forearm_length_px'][0 if hand == 'left' else 1]})
    methods = {}
    for method in sorted({r['method'] for r in records}):
        rows = [r for r in records if r['method'] == method]
        methods[method] = {'all': summary(rows),
            'own_elbow_axis_toward_count': sum(r['along_own_elbow_axis_px'] < 0 for r in rows),
            'groups': {c+'_'+h: summary([r for r in rows if r['camera'] == c and r['hand'] == h])
                       for c in ('left', 'right') for h in ('left', 'right')},
            'radial_correlation': stratified(rows, 'radial_px'),
            'input_size_correlation': stratified(rows, 'input_forearm_px')}
    route = read(ROOT/'realtime_app/configs/current_mainline.json')
    native = {}
    for key in ('body_candidate', 'matched_control'):
        path = ROOT/route[key]['result']/'fit/result.npz'
        with np.load(path) as a:
            coco = a['predicted_coco'][:, [9, 10]]
            joints = a['smplh_joints'][:, [20, 21]]
        delta = np.linalg.norm(coco-joints, axis=-1)*1000
        item = {'source': str(path), 'offset_mm_mean': delta.mean(0).tolist(),
                'offset_mm_p95': np.percentile(delta, 95, axis=0).tolist(),
                'offset_mm_max': delta.max(0).tolist()}
        for c in ('left', 'right'):
            K, D = (cal.K0, cal.D0) if c == 'left' else (cal.K1, cal.D1)
            def project(x):
                if c == 'right':
                    x = x@cal.R_cam0_to_cam1.T+cal.T_cam0_to_cam1_mm/1000
                return fisheye_project_numpy(x.reshape(-1, 3)*1000, K, D).reshape(-1, 2, 2)
            px = np.linalg.norm(project(coco)-project(joints), axis=-1)
            item[c+'_projection_offset_px_mean'] = px.mean(0).tolist()
        native[key] = item
    # Argmax comparison isolates subpixel decoding without rerunning the model.
    decoder = {}
    for variant in sorted({r['variant'] for r in sparse}):
        shift = [float(x) for row in sparse if row['variant'] == variant
                 for x in np.linalg.norm(np.asarray(row['raw'])[9:11]
                                         -np.asarray(row['argmax_raw'])[9:11], axis=1)]
        decoder[variant] = {'n': len(shift), 'mean_udp_argmax_shift_px': float(np.mean(shift)),
                            'max_udp_argmax_shift_px': float(np.max(shift))}
    stereo = read(LABELS/'stereo_diagnostics.json')
    manual_geometry = [{'frame': row['frame_index'], 'hand': row['hand'],
                        'state': row['manual']['state'],
                        'reprojection_error_px': row['manual']['reprojection_error_px']}
                       for row in stereo if 'wrist_left_camera_m' in row['manual']]
    result = {'protocol': {'inputs': [str(BASE), str(LABELS)],
        'methods': 'existing pose outputs only; no multi-model fusion',
        'manual_axis': 'Sapiens2 elbow to manual wrist; own-elbow sensitivity retained',
        'regressor_role': 'Direct detector-manual pixel residual contains neither SMPLH nor regressor nor triangulation',
        'limits': '10 repeated frames, one subject, 28 manual wrists; elbows not independently annotated'},
        'methods': methods, 'native_vs_coco': native, 'decoder': decoder,
        'manual_geometry': manual_geometry, 'records': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'methods': {m:v['all'] for m,v in methods.items()}, 'native_vs_coco': native}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
