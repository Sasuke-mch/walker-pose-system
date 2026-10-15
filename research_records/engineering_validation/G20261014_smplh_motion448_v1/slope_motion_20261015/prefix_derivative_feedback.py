"""Bounded diagnostic: retain predicted v/a after an unconverged pose update."""
import json
import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'realtime_app'))
sys.path.insert(0, str(ROOT / 'realtime_app/tools'))
from pose_app.fisheye_camera import load_stereo_fisheye
from pose_app.smplh_first_frame_labels import apply_first_frame_observations
from pose_app.smplh_ground_constraints import load_sole_vertex_indices
from pose_app.smplh_iterated_motion_tracker import IteratedMAPConfig, SMPLHIteratedMotionTracker
from pose_app.smplh_motion_tracker import SMPLHMotionConfig, SMPLHObservationMap
from run_stereo_pva_baseline import JOINT_INDICES, _load_predictions, _read_capture_times


class RetainPredictedDerivatives(SMPLHIteratedMotionTracker):
    def _update(self, pixels, valid, reasons):
        velocity, acceleration = self.velocity.copy(), self.acceleration.copy()
        result = super()._update(pixels, valid, reasons)
        if result[0] and not self.derivative_estimate_converged:
            self.velocity[:] = velocity
            self.acceleration[:] = acceleration
        return result


class ResetDetectorSwitchPrediction(SMPLHIteratedMotionTracker):
    def _predict(self, dt):
        if not hasattr(self, '_detector_switch_reset_done'):
            self._detector_switch_reset_done = True
            self.velocity[:] = 0
            self.acceleration[:] = 0
            self.covariance[:self.n, self.n:] = 0
            self.covariance[self.n:, :self.n] = 0
            self.covariance[self.n:, self.n:] = self.initial_covariance[self.n:, self.n:]
            return
        return super()._predict(dt)


def stats(v):
    a = np.asarray(v)
    a = a[np.isfinite(a)]
    return {'n': len(a), 'median': float(np.median(a)), 'p95': float(np.percentile(a, 95)), 'max': float(a.max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only-detector-switch-reset', action='store_true')
    args = parser.parse_args()
    out = Path(__file__).resolve().parent
    source = out.parent / 'first_frame_sapiens_20261015'
    params = json.loads((source / 'replay_selected_init/map_sigma075/run_parameters.json').read_text(encoding='utf-8'))
    inp = params['inputs']
    left = _load_predictions(Path(inp['left_predictions']), 'left')
    right = _load_predictions(Path(inp['right_predictions']), 'right')
    times, _, _, _ = _read_capture_times(Path(inp['input_pairs']), len(left))
    apply_first_frame_observations(Path(params['first_frame_observations']), Path(inp['input_pairs']), left, right)
    cal = load_stereo_fisheye(Path(inp['calibration_dir']))
    model = SMPLHObservationMap(*(Path(inp[k]) for k in ['initial_fit', 'smplh_model', 'regressor', 'vposer_dir', 'mano_left', 'mano_right']), params['device'])
    conf = SMPLHMotionConfig(**params['configuration'])
    opt = IteratedMAPConfig(**params['iterated_map_configuration'])
    soles = load_sole_vertex_indices(ROOT / 'research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/contact_surface_sets_smplh_v1/contact_vertex_sets.json')
    scene = np.load(source / 'visualization/scene/scene_transforms.npz', allow_pickle=False)
    rotations = scene['rotation_ground_from_left'][:62]
    translations = scene['translation_ground_from_left_mm'][:62]
    report_path = out / 'prefix_derivative_feedback.json'
    results = json.loads(report_path.read_text(encoding='utf-8'))['results'] if args.only_detector_switch_reset and report_path.exists() else {}
    controls = [('reset_va_cross_covariance_at_detector_switch_frame1_only', ResetDetectorSwitchPrediction)] if args.only_detector_switch_reset else [('baseline', SMPLHIteratedMotionTracker), ('retain_predicted_va_if_unconverged', RetainPredictedDerivatives)]
    for name, cls in controls:
        tracker = cls(model, cal, conf, opt)
        states, records = [], []
        started = perf_counter()
        for i in range(62):
            l, r = left[i, JOINT_INDICES], right[i, JOINT_INDICES]
            res = tracker.step(float(times[i]), l, r)
            pixels = np.array([l, r])
            valid, _ = tracker._supported_pixels(pixels, epipolar=False)
            error = np.linalg.norm(res['projected_pixels'] - pixels[..., :2], axis=-1)
            error[~valid] = np.nan
            coco_g = res['coco_m'] * 1000 @ rotations[i].T + translations[i]
            pelvis_g = res['joints_m'][0] * 1000 @ rotations[i].T + translations[i]
            verts_g = res['vertices_m'] * 1000 @ rotations[i].T + translations[i]
            sole_z = np.array([verts_g[ix, 2].min() for ix in soles])
            states.append({'coco': res['coco_m'], 'coco_g': coco_g, 'pelvis_g': pelvis_g,
                           'sole_z': sole_z, 'velocity': res['generalized_velocity'],
                           'acceleration': res['generalized_acceleration'], 'error': error})
            records.append({'frame': i, 'status': res['status'], 'iterated_update': res['iterated_update'],
                            'derivative_estimate_converged': bool(tracker.derivative_estimate_converged)})
            if i % 16 == 0 or i == 61:
                print(f'{name} {i}/61 {res["iterated_update"].get("status")}', flush=True)
        arrays = {k: np.array([x[k] for x in states]) for k in states[0]}
        np.savez_compressed(out / f'prefix_{name}.npz', **arrays, timestamps=times[:62], raw_pixels=np.stack([left[:62, JOINT_INDICES], right[:62, JOINT_INDICES]], axis=1))
        (out / f'prefix_{name}_solver.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        run = {'raw_error_all_62': stats(arrays['error']), 'raw_error_stage1_35_61': stats(arrays['error'][35:62]),
               'solver_status_counts': {s: sum(r['iterated_update'].get('status') == s for r in records) for s in {r['iterated_update'].get('status') for r in records}},
               'seconds': perf_counter() - started}
        for key, value in [('native_pelvis', arrays['pelvis_g']), ('coco_hip_midpoint', arrays['coco_g'][:, 11:13].mean(axis=1)), ('left_ankle', arrays['coco_g'][:, 15]), ('right_ankle', arrays['coco_g'][:, 16])]:
            run[key + '_stage1_endpoint_delta_xyz_mm'] = (np.median(value[57:62], axis=0) - np.median(value[35:40], axis=0)).tolist()
            run[key + '_stage1_z_range_mm'] = [float(value[35:62, 2].min()), float(value[35:62, 2].max())]
        run['sole_stage1_35_61_abs_height_mm'] = stats(abs(arrays['sole_z'][35:62]))
        run['sole_stage1_endpoint_delta_mm'] = (np.median(arrays['sole_z'][57:62], axis=0) - np.median(arrays['sole_z'][35:40], axis=0)).tolist()
        run['final_generalized_velocity_norm'] = float(np.linalg.norm(arrays['velocity'][-1]))
        run['final_generalized_acceleration_norm'] = float(np.linalg.norm(arrays['acceleration'][-1]))
        if name == 'baseline':
            old = np.load(source / 'replay_selected_init/map_sigma075/result.npz', allow_pickle=False)
            run['saved_baseline_coco_max_abs_difference_m'] = float(np.max(abs(arrays['coco'] - old['predicted_coco'][:62])))
            run['saved_baseline_velocity_max_abs_difference'] = float(np.max(abs(arrays['velocity'] - old['generalized_velocity'][:62])))
            run['saved_baseline_acceleration_max_abs_difference'] = float(np.max(abs(arrays['acceleration'] - old['generalized_acceleration'][:62])))
        results[name] = run
        print(json.dumps({name: run}), flush=True)
    report = {'question': 'Does an unconverged accepted pose correction feed v/a and amplify subsequent motion?',
              'frames': [0, 61], 'parameters_source': str(source / 'replay_selected_init/map_sigma075/run_parameters.json'),
              'single_variable': 'Retain-v/a control: after accepted unconverged _update restore pre-update predicted velocity/acceleration only, keep pose/covariance. Detector-switch control: replace first prediction only by zero-v/a and initial derivative covariance with q-derivative cross covariances zero, preserve q and Pqq; subsequent predictions unchanged.',
              'observations': 'frame0 Sapiens, frames1..61 unchanged PMPose; frozen calibrated beta; actual capture times',
              'ground_role': 'constant existing Stage1 display transform; not estimator input or independent contact truth', 'results': results}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
