"""Replay one Sapiens2 model: ROI, perspective, decoding, stereo and causality."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from pose_app.fisheye_camera import load_stereo_fisheye
from pose_app.wrist_processing import (CausalPointFilter, epipolar_select,
    heatmap_candidates, perspective_view, raw_to_upright, upright_to_raw, upper_box, refine_forearm)
from walker_tools.hands.solve_annotated_wrist_targets import solve_point
from walker_tools.stereo.evaluate_offline_stereo_predictions import SAPIENS_TO_COCO17

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'research_records/engineering_validation/G20261002_sapiens2_body_mainline448_v2'
LABELS = ROOT/'research_records/engineering_validation/G20261003_wrist_detector_manual_v2'
ANNOTATIONS = ROOT/'research_records/annotation_handoffs/H20261001_people1_wrist_joint_10pairs_v5_corrected'
VARIANTS = ('full_flip', 'upper_flip', 'perspective_flip', 'full_single', 'upper_single', 'perspective_single')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def stats(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {'n': 0, 'mean': None, 'median': None, 'rms': None, 'p95': None, 'max': None}
    return {'n': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)), 'rms': float(np.sqrt(np.mean(a*a))), 'p95': float(np.percentile(a, 95)), 'max': float(a.max())}


def sources():
    det, cached = {}, {}
    for c in ('left', 'right'):
        det[c] = read(SOURCE/f'{c}_detections.json')['images']
        cached[c] = np.array([x['keypoints'][0] for x in read(SOURCE/f'{c}_sapiens_coco17_model_input.json')['images']])
    return det, cached


def prepare(output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    frames = read(ANNOTATIONS/'manifest.json')['frames']
    windows = [[max(0, f-7), f] for f in frames]
    write(output/'protocol.json', {'source': str(SOURCE), 'manual_evaluation': str(LABELS),
        'model': 'Sapiens2-0.4B only', 'input_size': [768, 1024], 'dtype': 'unchanged float32',
        'frames': frames, 'continuous_windows': windows, 'variants': list(VARIANTS),
        'sparse_roi': 'same detector cached previous frame, never current/manual points; frame0 full bootstrap',
        'continuous_roi': 'own previous inference only; each window initializes full, no cached pose',
        'upper_box': 'combined COCO5:11; min confidence .25; padding x .25 ytop .30 ybottom .45; min span200x250',
        'perspective': 'previous6 upper joints as support,1.35span,.95 coverage,768x1024; padded-box background is not a geometric support; inverse mapping to raw',
        'heatmap': 'original UDP DARK versus integer argmax; local stereo candidates max12 raw px, response>=.5',
        'epipolar_cost': '-log(response_product)+.3*sampson_px_squared/25+.05*movement_squared/64',
        'causal_filter': 'min_cutoff2Hz,beta.015,derivative1Hz; missing reset; actual file timestamps',
        'geometry': 'original solve_point, 10px and1deg; retain bounds, low-confidence and rejected rows',
        'timing': 'CUDA synchronized; 3warmups; includes view construction, preprocessing, forward, CPU transfer, decode and inverse mapping; excludes PNG read, model load, cached person detector, capture/display',
        'budget': 'six variants on20 annotated images; three single-pass variants on10 causal windows; full448 cached single-model filter replay',
        'success': 'compare all28 manual2D slots, same8 stereo references, missing and geometry; report lower-limb coverage cost; no automatic mainline promotion',
        'stop': 'coordinate/frame mismatch, nonfinite active inputs or resource failure; retain failure files',
        'code': ['realtime_app/experiment_wrist_processing.py', 'realtime_app/pose_app/wrist_processing.py']})


class Engine:
    def __init__(self):
        import torch
        self.torch = torch
        pose = ROOT/'third_party/sapiens2/repo/sapiens/pose'
        sys.path[:0] = [str(pose/'tools/vis'), str(pose.parents[1])]
        self.cwd = Path.cwd()
        os.chdir(pose)
        self.runner = importlib.import_module('run_sapiens2_from_yolo26_bboxes')
        self.model = self.runner.prepare_model('configs/keypoints308/shutterstock_goliath_3po/sapiens2_0.4b_keypoints308_shutterstock_goliath_3po-1024x768.py', str(ROOT/'models/sapiens2/pose-0.4b/sapiens2_0.4b_pose.safetensors'), 'cuda:0')
        self.cal = load_stereo_fisheye(ROOT/'realtime_app/calibration/results')

    def infer(self, upright, entry, variant, previous):
        self.torch.cuda.synchronize()
        start = time.perf_counter()
        camera = entry['camera']
        box = None if previous is None else upper_box(previous)
        bootstrap = box is None or variant.startswith('full')
        image = upright
        view = None
        if bootstrap:
            boxes, scores = self.runner.extract_person_boxes(entry)
            if len(boxes) != 1:
                raise ValueError('expected one frozen person prompt')
            box = boxes[0]
            score = float(scores[0])
        else:
            score = 1.
        if not bootstrap and variant.startswith('perspective'):
            K, D = (self.cal.K0, self.cal.D0) if camera == 'left' else (self.cal.K1, self.cal.D1)
            view = perspective_view(box, K, D, camera, support=previous[5:11, :2])
            raw = cv2.rotate(upright, cv2.ROTATE_90_CLOCKWISE if camera == 'left' else cv2.ROTATE_90_COUNTERCLOCKWISE)
            image = view.remap(raw)
            box = np.array([0., 0., image.shape[1]-1, image.shape[0]-1])
        view_ms = (time.perf_counter()-start)*1000
        inputs, sample = self.runner.preprocess_person(self.model, image, box, score)
        prediction = self.runner.forward_chunk(self.model, inputs, variant.endswith('flip'))
        self.torch.cuda.synchronize()
        pred = prediction[0].detach().cpu().numpy()
        xy, confidence = self.runner.decode_prediction(self.model, pred, sample)
        meta = {k: np.asarray(sample['meta'][k]).reshape(2).tolist() for k in ('input_size', 'bbox_scale', 'bbox_center')}
        ids = list(SAPIENS_TO_COCO17)
        to_raw = view.to_raw if view is not None else lambda p: upright_to_raw(p, camera)
        raw = to_raw(xy[ids])
        h, w = pred.shape[-2:]
        maxima = pred.reshape(len(pred), -1).argmax(-1)
        grid = np.column_stack((maxima % w, maxima//w))
        argmax_image = grid/[w-1, h-1]*np.asarray(meta['bbox_scale'])+np.asarray(meta['bbox_center'])-np.asarray(meta['bbox_scale'])/2
        argmax = to_raw(argmax_image[ids])
        upright_points = raw_to_upright(raw, camera)
        points = np.column_stack((upright_points, confidence[ids]))
        candidates = []
        for joint in (9, 10):
            pts, responses = heatmap_candidates(pred[ids[joint]], meta, to_raw, raw[joint])
            candidates.append({'raw': pts.tolist(), 'response': responses.tolist()})
        self.torch.cuda.synchronize()
        elapsed = (time.perf_counter()-start)*1000
        # Verify the inverse mapping without rounding to integer image pixels.
        error = float(np.max(np.abs(view.from_raw(raw)-xy[ids]))) if view is not None else float(np.max(np.abs(upright_to_raw(upright_points, camera)-raw)))
        if error > 1e-6:
            raise ValueError('inverse coordinate roundtrip differs')
        # Effective scale in detector input; this is not a visual wrist-width measurement.
        input_xy = (xy[ids]-np.asarray(meta['bbox_center'])+np.asarray(meta['bbox_scale'])/2)/np.asarray(meta['bbox_scale'])*np.asarray(meta['input_size'])
        arms = [float(np.linalg.norm(input_xy[e]-input_xy[r])) for e, r in ((7, 9), (8, 10))]
        return {'raw': raw.tolist(), 'upright': points.tolist(), 'argmax_raw': argmax.tolist(),
            'confidence': confidence[ids].tolist(), 'candidates': candidates,
            'elapsed_ms': elapsed, 'view_ms': view_ms, 'bootstrap_full': bootstrap,
            'box': box.tolist(), 'input_forearm_length_px': arms, 'roundtrip_max_px': error}


def run(output):
    protocol = read(output/'protocol.json')
    det, cached = sources()
    engine = Engine()
    write(output/'hardware.json', {'torch': engine.torch.__version__, 'gpu': engine.torch.cuda.get_device_name(), 'threads': engine.torch.get_num_threads(), 'tf32': engine.torch.backends.cuda.matmul.allow_tf32})
    entry = dict(det['left'][0], camera='left')
    image = cv2.imread(entry['image_path'])
    for _ in range(3):
        engine.infer(image, entry, 'full_single', None)
    path = output/'sparse.jsonl'
    if path.exists():
        raise FileExistsError(path)
    with path.open('w', encoding='utf-8') as out:
        for f in protocol['frames']:
            for camera in ('left', 'right'):
                entry = dict(det[camera][f], camera=camera)
                image = cv2.imread(entry['image_path'])
                if image is None:
                    raise FileNotFoundError(entry['image_path'])
                for variant in VARIANTS:
                    row = dict(frame=f, camera=camera, variant=variant, **engine.infer(image, entry, variant, cached[camera][f-1] if f else None))
                    out.write(json.dumps(row, allow_nan=False)+'\n')
                    out.flush()
            print(f'sparse {f} done', flush=True)
    path = output/'continuous.jsonl'
    with path.open('w', encoding='utf-8') as out:
        for variant in ('full_single', 'upper_single', 'perspective_single'):
            for start, stop in protocol['continuous_windows']:
                previous = {'left': None, 'right': None}
                for f in range(start, stop+1):
                    for camera in ('left', 'right'):
                        entry = dict(det[camera][f], camera=camera)
                        image = cv2.imread(entry['image_path'])
                        if image is None:
                            raise FileNotFoundError(entry['image_path'])
                        row = dict(frame=f, window_start=start, camera=camera, variant=variant, **engine.infer(image, entry, variant, previous[camera]))
                        previous[camera] = np.asarray(row['upright'])
                        out.write(json.dumps(row, allow_nan=False)+'\n')
                        out.flush()
                print(f'continuous {variant} {start}:{stop} done', flush=True)
    os.chdir(engine.cwd)


def load_lines(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]


def evaluate(rows, labels, cal, points_key='raw', confidence_gate=False):
    errors, missing, stereo, elbows = [], [], [], []
    for label in labels:
        key = label['frame_index'], label['camera']
        row = rows.get(key)
        if label['manual']['state'] != 'present':
            continue
        j = 9 if label['hand'] == 'left' else 10
        if row is None:
            missing.append(dict(frame=key[0], camera=key[1], hand=label['hand'], reason='missing_frame'))
            continue
        point = np.asarray(row[points_key][j], float)
        available = np.isfinite(point).all() and 0 <= point[0] < 1920 and 0 <= point[1] < 1080
        if confidence_gate:
            available &= row['confidence'][j] >= .25
        if not available:
            missing.append(dict(frame=key[0], camera=key[1], hand=label['hand'], reason='low_confidence_nonfinite_or_bounds'))
            continue
        errors.append(dict(frame=key[0], camera=key[1], hand=label['hand'], error_px=float(np.linalg.norm(point-label['manual']['raw_px']))))
    label_map = {(r['frame_index'], r['camera'], r['hand']): r for r in labels}
    for f in sorted({k[0] for k in rows}):
        for side, j in (('left', 9), ('right', 10)):
            pair = [rows.get((f, c)) for c in ('left', 'right')]
            if any(x is None for x in pair):
                continue
            points = [np.asarray(x[points_key][j], float) for x in pair]
            valid = all(np.isfinite(p).all() and 0 <= p[0] < 1920 and 0 <= p[1] < 1080 for p in points)
            if confidence_gate:
                valid &= all(x['confidence'][j] >= .25 for x in pair)
            solved = solve_point(*points, cal) if valid else {'state': 'unavailable', 'reasons': ['confidence_or_bounds_or_missing']}
            record = dict(frame=f, hand=side, model=solved)
            manual = [label_map.get((f, c, side)) for c in ('left', 'right')]
            if all(x is not None and x['manual']['state'] == 'present' for x in manual):
                ref = solve_point(*[x['manual']['raw_px'] for x in manual], cal)
                record['manual'] = ref
                if 'wrist_left_camera_m' in ref and 'wrist_left_camera_m' in solved:
                    record['distance_mm'] = float(np.linalg.norm(np.asarray(ref['wrist_left_camera_m'])-solved['wrist_left_camera_m'])*1000)
            stereo.append(record)
        for j in (7, 8):
            pair = [rows.get((f, c)) for c in ('left', 'right')]
            if any(x is None for x in pair):
                continue
            p = [np.asarray(x[points_key][j], float) for x in pair]
            if all(np.isfinite(x).all() and 0 <= x[0] < 1920 and 0 <= x[1] < 1080 for x in p):
                elbows.append(solve_point(*p, cal))
    joint_pass = [x for x in stereo if 'manual' in x and x['manual']['state'] == x['model']['state'] == 'accepted_manual_reference_geometry']
    return {'manual2d_px': stats([x['error_px'] for x in errors]), 'manual2d_missing': len(missing),
        'stereo_diagnostic_mm': stats([x['distance_mm'] for x in stereo if 'distance_mm' in x]),
        'stereo_both_gates_mm': stats([x['distance_mm'] for x in joint_pass]),
        'wrist_geometry_passed': sum(x['model']['state'] == 'accepted_manual_reference_geometry' for x in stereo), 'wrist_geometry_total': len(stereo),
        'elbow_geometry_passed': sum(x['state'] == 'accepted_manual_reference_geometry' for x in elbows), 'elbow_geometry_total': len(elbows),
        'errors': errors, 'missing': missing, 'stereo': stereo}


def analyze(output):
    write(output/'postprocessing_protocol.json', {'heatmap': 'UDP versus integer peak; local image-supported stereo candidates, same fixed weights as initial protocol',
        'filter': 'three fixed profiles: cutoff2/beta.015,cutoff5/beta.015,cutoff2/beta.05; full448 observations; no future points or missing fill',
        'forearm': 'same baseline448; require confident in-bounds accepted elbow AND wrist; 5accepted-pair warmup;median last30 raw lengths; observation3px,length15mm;max20iterations,rawshift12px;retain geometry gates',
        'manual_sensitivity': 'seed0; coordinate Gaussian sigma1,2,5px;30draws per8 references;not measured annotation uncertainty',
        'status': 'diagnostic followup configured before postprocessing results; no independent test set or tuning claim'})
    labels = read(LABELS/'points.json')
    cal = load_stereo_fisheye(ROOT/'realtime_app/calibration/results')
    sparse, continuous = load_lines(output/'sparse.jsonl'), load_lines(output/'continuous.jsonl')
    results = {'sparse': {}, 'continuous': {}, 'filter': {}, 'diagnosis': {}, 'forearm': {}}
    for collection, name in ((sparse, 'sparse'), (continuous, 'continuous')):
        for variant in sorted({x['variant'] for x in collection}):
            selected = [x for x in collection if x['variant'] == variant]
            mapping = {(x['frame'], x['camera']): x for x in selected}
            metrics = evaluate(mapping, labels, cal)
            metrics['confidence_gated'] = evaluate(mapping, labels, cal, confidence_gate=True)
            metrics['timing_ms'] = stats([x['elapsed_ms'] for x in selected])
            metrics['view_ms'] = stats([x['view_ms'] for x in selected])
            metrics['stereo_pair_processing_ms'] = stats([mapping[f, 'left']['elapsed_ms']+mapping[f, 'right']['elapsed_ms'] for f in sorted({x['frame'] for x in selected})])
            metrics['bootstrap_full'] = sum(x['bootstrap_full'] for x in selected)
            metrics['input_forearm_px'] = stats([p for x in selected for p in x['input_forearm_length_px']])
            metrics['roundtrip_max_px'] = max(x['roundtrip_max_px'] for x in selected)
            results[name][variant] = metrics
            if name == 'sparse':
                results['sparse'][variant+'_argmax'] = evaluate(mapping, labels, cal, 'argmax_raw')
                enhanced = {k: dict(v, raw=np.asarray(v['raw']).copy()) for k, v in mapping.items()}
                updates, costs = [], []
                for f in sorted({x['frame'] for x in selected}):
                    start = time.perf_counter()
                    for h, j in enumerate((9, 10)):
                        a, b = [mapping[f, c]['candidates'][h] for c in ('left', 'right')]
                        l, r, audit = epipolar_select((np.array(a['raw']), np.array(a['response'])), (np.array(b['raw']), np.array(b['response'])), cal)
                        enhanced[f, 'left']['raw'][j] = l
                        enhanced[f, 'right']['raw'][j] = r
                        updates.append(dict(frame=f, hand=h, **audit))
                    costs.append((time.perf_counter()-start)*1000)
                results['sparse'][variant+'_epipolar'] = dict(evaluate(enhanced, labels, cal), processing_ms=stats(costs), shifts=updates)
    det, cached = sources()
    sequence = ROOT/'research_records/engineering_validation/V20260908_people0_1_2_pmpose_c3_chain/people_1/pmpose_strict_stereo/offline_stereo_results.jsonl'
    timestamps = [x['pair_timestamp_sec'] for x in load_lines(sequence)]
    if len(timestamps) != 448 or np.any(np.diff(timestamps) <= 0):
        raise ValueError('invalid sequence timestamps')
    for min_cutoff, beta in ((2., .015), (5., .015), (2., .05)):
        mapping, shifts, overhead, accelerations, raw_acc, range_ratio, lag_frames = {}, [], [], [], [], [], []
        for c in ('left', 'right'):
            filters = [CausalPointFilter(min_cutoff, beta) for _ in range(4)]
            series, original = [], []
            for f in range(448):
                raw = upright_to_raw(cached[c][f, :, :2], c)
                orig = raw.copy()
                state = []
                start = time.perf_counter()
                for k, j in enumerate((7, 8, 9, 10)):
                    valid = cached[c][f, j, 2] >= .25 and np.isfinite(raw[j]).all() and 0 <= raw[j, 0] < 1920 and 0 <= raw[j, 1] < 1080
                    p, s = filters[k].update(raw[j], timestamps[f], valid)
                    if np.isfinite(p).all():
                        shifts.append(float(np.linalg.norm(p-raw[j])))
                    raw[j] = p
                    state.append(s)
                overhead.append((time.perf_counter()-start)*1000)
                # JSON output retains missing states as null, not old predictions.
                mapping[f, c] = {'raw': raw.tolist(), 'confidence': cached[c][f, :, 2].tolist(), 'filter_state': state}
                series.append(raw[[7, 8, 9, 10]])
                original.append(orig[[7, 8, 9, 10]])
            for series_input, destination in ((series, accelerations), (original, raw_acc)):
                velocity = np.diff(np.asarray(series_input), axis=0)/np.diff(timestamps)[:, None, None]
                dt = (np.diff(timestamps)[1:]+np.diff(timestamps)[:-1])/2
                acc = np.linalg.norm(np.diff(velocity, axis=0)/dt[:, None, None], axis=-1)
                destination.extend(acc[np.isfinite(acc)].tolist())
            s, o = np.asarray(series), np.asarray(original)
            ratio = np.linalg.norm(np.nanmax(s, axis=0)-np.nanmin(s, axis=0), axis=-1)/np.linalg.norm(np.nanmax(o, axis=0)-np.nanmin(o, axis=0), axis=-1)
            range_ratio.extend(ratio.tolist())
            dv, ov = np.diff(s, axis=0), np.diff(o, axis=0)
            correlations = []
            for lag in range(6):
                a, b = (dv, ov) if lag == 0 else (dv[lag:], ov[:-lag])
                mask = np.isfinite(a) & np.isfinite(b)
                correlations.append(float(np.corrcoef(a[mask], b[mask])[0, 1]))
            lag_frames.append(int(np.argmax(correlations)))
        results['filter'][f'cutoff{min_cutoff}_beta{beta}'] = dict(evaluate(mapping, labels, cal), processing_ms=stats(overhead), observation_shift_px=stats(shifts), acceleration_px_s2=stats(accelerations), baseline_acceleration_px_s2=stats(raw_acc), coordinate_range_ratio=stats(range_ratio), velocity_correlation_lag_frames=lag_frames)
    # A causal length prior can be initialized only from accepted elbow/wrist pairs.
    mapping = {(f, c): {'raw': upright_to_raw(cached[c][f, :, :2], c).tolist(), 'confidence': cached[c][f, :, 2].tolist()} for f in range(448) for c in ('left', 'right')}
    histories = {'left': [], 'right': []}
    audits, overhead = [], []
    for f in range(448):
        start = time.perf_counter()
        for side, joints in (('left', (7, 9)), ('right', (8, 10))):
            observed = np.array([[mapping[f, c]['raw'][j] for j in joints] for c in ('left', 'right')])
            confidence_ok = all(mapping[f, c]['confidence'][j] >= .25 for c in ('left', 'right') for j in joints)
            bounds_ok = np.isfinite(observed).all() and np.all((observed[..., 0] >= 0) & (observed[..., 0] < 1920) & (observed[..., 1] >= 0) & (observed[..., 1] < 1080))
            solved = [solve_point(*observed[:, k], cal) for k in range(2)] if confidence_ok and bounds_ok else []
            if len(solved) != 2 or any(s['state'] != 'accepted_manual_reference_geometry' for s in solved):
                audits.append({'frame': f, 'hand': side, 'state': 'not_updated_geometry_or_observation'})
                continue
            initial = np.array([s['wrist_left_camera_m'] for s in solved])
            if len(histories[side]) >= 5:
                refined, audit = refine_forearm(initial, observed, np.median(histories[side][-30:]), cal)
                # Candidate must still satisfy the same geometry gate and raw bounds.
                bounds = np.all((refined[..., 0] >= 0) & (refined[..., 0] < 1920) & (refined[..., 1] >= 0) & (refined[..., 1] < 1080))
                gated = [solve_point(*refined[:, k], cal)['state'] == 'accepted_manual_reference_geometry' for k in range(2)] if bounds else [False]
                applied = audit['accepted_update'] and all(gated)
                if applied:
                    for ci, c in enumerate(('left', 'right')):
                        for k, j in enumerate(joints):
                            mapping[f, c]['raw'][j] = refined[ci, k].tolist()
                audits.append(dict(frame=f, hand=side, state='updated' if applied else 'candidate_rejected', **audit))
            else:
                audits.append({'frame': f, 'hand': side, 'state': 'length_prior_warmup'})
            histories[side].append(float(np.linalg.norm(initial[1]-initial[0])))
        overhead.append((time.perf_counter()-start)*1000)
    results['forearm'] = dict(evaluate(mapping, labels, cal), processing_ms=stats(overhead), audit=audits, accepted_length_samples={k: len(v) for k, v in histories.items()}, updated_pairs=sum(x['state'] == 'updated' for x in audits))
    baseline = read(LABELS/'summary.json')['all_available']['Sapiens2']['all']
    fresh = results['sparse']['full_flip']['errors']
    existing = {(x['frame_index'], x['camera'], x['hand']): x['methods']['Sapiens2']['prediction_raw_px'] for x in labels if x['manual']['state'] == 'present'}
    by_key = {(x['frame'], x['camera']): x for x in sparse if x['variant'] == 'full_flip'}
    results['diagnosis']['fresh_vs_cached_max_px'] = max(float(np.linalg.norm(np.asarray(by_key[f, c]['raw'][9 if h == 'left' else 10])-p)) for (f, c, h), p in existing.items())
    results['diagnosis']['cached_baseline2d'] = baseline
    results['diagnosis']['fresh_baseline2d'] = stats([x['error_px'] for x in fresh])
    results['diagnosis']['forearm_not_wrist_width'] = True
    results['diagnosis']['no_elbow_manual_labels'] = True
    rng = np.random.default_rng(0)
    reference_stereo = read(LABELS/'stereo_diagnostics.json')
    manual_map = {(r['frame_index'], r['camera'], r['hand']): r for r in labels}
    sensitivity = {}
    for sigma in (1., 2., 5.):
        displacement, passed = [], 0
        for row in reference_stereo:
            if row['manual']['state'] == 'unavailable':
                continue
            pixels = np.array([manual_map[row['frame_index'], c, row['hand']]['manual']['raw_px'] for c in ('left', 'right')])
            ref = np.array(row['manual']['wrist_left_camera_m'])
            for _ in range(30):
                solved = solve_point(*(pixels+rng.normal(0, sigma, (2, 2))), cal)
                if 'wrist_left_camera_m' in solved:
                    displacement.append(float(np.linalg.norm(np.array(solved['wrist_left_camera_m'])-ref)*1000))
                    passed += solved['state'] == 'accepted_manual_reference_geometry'
        sensitivity[str(sigma)] = dict(stats(displacement), perturbed_geometry_passed=passed, noise='independent Gaussian per coordinate; hypothetical, not measured annotator noise')
    results['diagnosis']['manual_sensitivity_mm'] = sensitivity
    write(output/'results.json', results)
    print(json.dumps({k: {m: v['manual2d_px']['mean'] for m, v in group.items()} for k, group in results.items() if k in ('sparse', 'continuous', 'filter')}, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=('prepare', 'run', 'analyze'))
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    output = args.output.resolve()
    {'prepare': prepare, 'run': run, 'analyze': analyze}[args.stage](output)


if __name__ == '__main__':
    main()
