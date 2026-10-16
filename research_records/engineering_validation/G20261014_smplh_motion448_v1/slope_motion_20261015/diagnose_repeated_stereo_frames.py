"""One bounded control: do repeated decoded stereo images drive ankle rise?"""
import csv
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import prefix_derivative_feedback as shared  # noqa: E402


def main():
    source = OUT.parent / 'first_frame_sapiens_20261015/replay_selected_init/map_sigma075'
    params = json.loads((source / 'run_parameters.json').read_text(encoding='utf-8'))
    inputs = params['inputs']
    pairs = Path(inputs['input_pairs'])
    with (pairs / 'selection_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        selection = list(csv.DictReader(stream))[:62]
    repeated = np.zeros((62, 2), dtype=bool)
    for view, folder in enumerate(('left_ccw90', 'right_cw90')):
        previous = None
        for i, row in enumerate(selection):
            image = cv2.imread(str(pairs / folder / row['file_name']), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f'missing image: {folder}/{row["file_name"]}')
            repeated[i, view] = previous is not None and np.array_equal(image, previous)
            previous = image
    skip = repeated.all(axis=1)
    model = shared.SMPLHObservationMap(*[Path(inputs[key]) for key in
        ('initial_fit', 'smplh_model', 'regressor', 'vposer_dir', 'mano_left', 'mano_right')], params['device'])
    cal = shared.load_stereo_fisheye(Path(inputs['calibration_dir']))
    config = shared.SMPLHMotionConfig(**params['configuration'])
    opt = shared.IteratedMAPConfig(**params['iterated_map_configuration'])
    left = shared._load_predictions(Path(inputs['left_predictions']), 'left')
    right = shared._load_predictions(Path(inputs['right_predictions']), 'right')
    shared.apply_first_frame_observations(Path(params['first_frame_observations']), pairs, left, right)
    times, *_ = shared._read_capture_times(pairs, len(left))
    with np.load(OUT / 'scene/scene_transforms.npz') as scene:
        rotations = scene['rotation_ground_from_left'][:62]
        translations = scene['translation_ground_from_left_mm'][:62] / 1000
    with np.load(source / 'result.npz') as archive:
        baseline = archive['predicted_coco'][:62].copy()
    trace = json.loads((OUT / 'motion_chain_trace.json').read_text(encoding='utf-8'))
    assert trace['saved_baseline_max_coco_difference_m'] == 0

    class SkipRepeatedPair(shared.SMPLHIteratedMotionTracker):
        def _update(self, pixels, valid, reasons):
            if skip[self.frame_index]:
                self.optimizer_diagnostic = {'status': 'repeated_decoded_stereo_pair_no_new_update',
                                             'posterior_updates': 0}
                self.derivative_estimate_converged = False
                reasons[:] = 'repeated_decoded_stereo_pair_no_new_update'
                return False, np.zeros_like(valid), 'repeated_decoded_stereo_pair_no_new_update'
            return super()._update(pixels, valid, reasons)

    tracker = SkipRepeatedPair(model, cal, config, opt)
    rows, baseline_error, control_error, baseline_height, control_height = [], [], [], [], []
    start = time.perf_counter()
    for i in range(62):
        tracker.frame_index = i
        pixels = np.array([left[i, 5:17], right[i, 5:17]])
        result = tracker.step(float(times[i]), *pixels)
        raw_ok, _ = tracker._supported_pixels(pixels, epipolar=False)
        old_pixels = tracker._pixels(baseline[i:i+1])[0]
        errors = [np.linalg.norm(projected - pixels[..., :2], axis=-1)
                  for projected in (old_pixels, result['projected_pixels'])]
        for error, destination in zip(errors, (baseline_error, control_error)):
            destination.extend(error[raw_ok].tolist())
        heights = [float(((points[15:17] @ rotations[i].T + translations[i])[:, 2]).mean() * 1000)
                   for points in (baseline[i], result['coco_m'])]
        baseline_height.append(heights[0])
        control_height.append(heights[1])
        rows.append({'frame': i, 'repeated_decoded_views': repeated[i].tolist(),
                     'skipped_update': bool(skip[i]), 'status': result['status'],
                     'baseline_mean_ankle_height_mm': heights[0], 'control_mean_ankle_height_mm': heights[1],
                     'control_raw_error_px': errors[1].tolist(),
                     'optimizer': {k: v for k, v in result['iterated_update'].items() if k != 'history'}})
        if i % 15 == 0 or i == 61:
            print(i, 'skip', skip[i], 'ankle height', heights, flush=True)
    report = {
        'question': 'Does reuse of an exactly repeated stereo image pair materially cause the Stage1 ankle rise?',
        'source_run': str(source), 'parameters_source': str(source / 'run_parameters.json'),
        'frames': [0, 61], 'decoded_image_comparison': 'np.array_equal;no image hash or keypoint proxy',
        'single_variable': 'Only suppress _update when BOTH decoded input views exactly equal their preceding selected image;retain all62 rows,original host dt,prediction,raw residuals,beta,VPoser,gates,solver budget;single-side repeats unchanged',
        'limits': ['Diagnostic only;no production input gate adopted',
                   'Repeated content does not identify hardware exposure time or the upstream duplication stage',
                   'This control isolates repeated updates,not host-time errors or one-side pairing errors',
                   'All-MAP-unconverged baseline already adds no Gaussian posterior information,so mean reuse and covariance reuse must be distinguished',
                   'No contact,height,3D observation,interpolation or derivative accuracy claim'],
        'predeclared_stop': 'One62-prefix control;no448 or additional parameter sweep. Causal repair eligibility requires all62 finite,>=20percent smaller35..61 height rise,and median/P95 raw residual each no>5percent worse.',
        'repeated_views_count': repeated.sum(axis=0).tolist(), 'repeated_both_count': int(skip.sum()),
        'repeated_either_count': int(repeated.any(axis=1).sum()),
        'elapsed_seconds': time.perf_counter() - start,
        'baseline': {'raw_error': shared.stats(baseline_error),
                     'mean_ankle_height_35_61_mm': [baseline_height[35], baseline_height[61]],
                     'mean_ankle_rise_35_61_mm': baseline_height[61] - baseline_height[35]},
        'control': {'raw_error': shared.stats(control_error),
                    'mean_ankle_height_35_61_mm': [control_height[35], control_height[61]],
                    'mean_ankle_rise_35_61_mm': control_height[61] - control_height[35]},
        'rows': rows}
    a, b = report['baseline'], report['control']
    checks = {'all62_finite': bool(np.isfinite(control_height).all() and len(control_error) == len(baseline_error)),
              'height_rise_reduction20percent': b['mean_ankle_rise_35_61_mm'] <= .8 * a['mean_ankle_rise_35_61_mm'],
              'raw_median_no5percent_regression': b['raw_error']['median'] <= 1.05 * a['raw_error']['median'],
              'raw_p95_no5percent_regression': b['raw_error']['p95'] <= 1.05 * a['raw_error']['p95']}
    report['checks'] = checks
    report['full_replay_eligible'] = all(checks.values())
    report['interpretation'] = ('Skipping repeated updates does not eliminate the rise. Endpoint differences include '
        'a changed frame35 start;do not call the27.37mm rise difference extra final-frame floating. '
        'Original host-time errors and hardware duplication causes remain unresolved.')
    (OUT / 'repeated_stereo_prefix_control.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print({k: v for k, v in report.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    main()
