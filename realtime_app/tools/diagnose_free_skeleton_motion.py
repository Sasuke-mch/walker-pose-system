"""Compare SMPL-H motion against causal Cartesian skeleton tracking without SMPL assets."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

APP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(APP), str(APP / 'tools')]
from pose_app.fisheye_camera import load_stereo_fisheye  # noqa: E402
from pose_app.stereo_pva_tracker import StereoPVATracker, StereoPVAConfig  # noqa: E402
from run_stereo_pva_baseline import _load_predictions, _read_capture_times, JOINT_NAMES  # noqa: E402


def stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {'count': int(values.size), **{key: float(value) for key, value in
        zip(('min', 'median', 'p95', 'max'), np.percentile(values, (0, 50, 95, 100)))}} if values.size else {'count': 0}


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(safe_json(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def segment_metrics(world, segment, status, segment_ids):
    a, b = segment
    ankle = world[a:b+1, 10:12, 2]
    joint_ok = np.isfinite(ankle)
    all_ok = bool(joint_ok.all())
    edge = min(5, len(ankle))
    result = {'frames': [a, b], 'frame_count': b-a+1,
              'available_ankle_slots': int(joint_ok.sum()), 'total_ankle_slots': int(ankle.size),
              'height_range_each_ankle_mm': [stats(ankle[:, j]) for j in (0, 1)],
              'ankle_status_counts': dict(Counter(status[a:b+1, 10:12].ravel().tolist()))}
    uninterrupted = all_ok and all(len(np.unique(segment_ids[a:b+1, j])) == 1 for j in (10, 11))
    result['both_ankles_available_without_restart'] = uninterrupted
    if all_ok:
        delta = ankle[-1]-ankle[0]
        robust = np.median(ankle[-edge:], axis=0)-np.median(ankle[:edge], axis=0)
        result.update(endpoint_height_change_each_ankle_mm=delta,
                      endpoint_mean_ankle_height_change_mm=float(delta.mean()),
                      median5_height_change_each_ankle_mm=robust,
                      median5_mean_ankle_height_change_mm=float(robust.mean()),
                      both_ankles_rise20mm=bool(uninterrupted and np.all(delta >= 20) and np.all(robust >= 20)),
                      persistent_rise_30mm=bool(uninterrupted and delta.mean() >= 30 and robust.mean() >= 30))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--analyze-only', action='store_true', help='Read saved tracking arrays; never repeat tracking')
    args = parser.parse_args()
    if not args.analyze_only and args.output.exists() and any(args.output.iterdir()):
        raise ValueError('output is nonempty; preserve the preceding run')
    params = json.loads((args.source_run / 'run_parameters.json').read_text(encoding='utf-8'))
    inputs = params['inputs']
    left = _load_predictions(Path(inputs['left_predictions']), 'left')
    right = _load_predictions(Path(inputs['right_predictions']), 'right')
    count = len(left)
    times, skew, pair_ids, capture = _read_capture_times(Path(inputs['input_pairs']), count)
    first = json.loads(Path(params['first_frame_observations']).read_text(encoding='utf-8'))
    if first['pair_id'] != 0 or first['coordinate_space'] != 'raw_fisheye_pixels':
        raise ValueError('frame0 observations must be raw-fisheye pair0')
    for side, folder in (('left', 'left_ccw90'), ('right', 'right_cw90')):
        if not Path(first['images'][side]).samefile(Path(inputs['input_pairs']) / folder / 'pair_0000.png'):
            raise ValueError('frame0 detector source image differs')
    frame0 = np.asarray(first['keypoints'], dtype=float)
    if frame0.shape != (2, 17, 3) or not np.isfinite(frame0).all():
        raise ValueError('invalid frame0 detector points')
    left[0], right[0] = frame0
    observations = np.stack((left[:, 5:17], right[:, 5:17]), axis=1)
    if left.shape != right.shape or count != 448:
        raise ValueError('expected the same ordered448 observation rows')
    cal = load_stereo_fisheye(Path(inputs['calibration_dir']))
    stage_rows = [json.loads(line) for line in (args.scene / 'stage1_stage2.jsonl').read_text(encoding='utf-8').splitlines()]
    stages = [row['stage']['operational'] for row in stage_rows]
    with np.load(args.scene / 'scene_transforms.npz', allow_pickle=False) as scene:
        rotations = scene['rotation_ground_from_left'].copy()
        translations = scene['translation_ground_from_left_mm'].copy()
        scene_times = scene['pair_timestamp_sec'].copy()
    if len(stages) != count or not np.allclose(scene_times, times, atol=1e-7, rtol=0):
        raise ValueError('scene/reference frame-time alignment differs')
    stage1 = []
    for i, stage in enumerate(stages):
        if stage == 'stage1_walker_static_human_moving':
            if i == 0 or stages[i-1] != stage:
                start = i
            if i == count-1 or stages[i+1] != stage:
                stage1.append((start, i))
    transform_audit = [{'frames': [a, b],
                        'max_rotation_entry_change': float(abs(rotations[a:b+1]-rotations[a]).max()),
                        'max_translation_change_mm': float(abs(translations[a:b+1]-translations[a]).max())}
                       for a, b in stage1]
    if any(row['max_rotation_entry_change'] > 1e-10 or row['max_translation_change_mm'] > 1e-8
           for row in transform_audit):
        raise ValueError('static episodes contain a changing reference transform')
    configs = {'pva_frozen_sigma4': StereoPVAConfig(),
               'pva_matched_pixel_sigma8': replace(StereoPVAConfig(), pixel_sigma_px=8.)}
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {'question': 'Does causal stereo skeleton tracking without SMPL-H reproduce the diagnosed ankle rise and hip image residual?',
        'source_run_for_paths_and_later_comparison': str(args.source_run.resolve()),
        'tracking_inputs': {k: inputs[k] for k in ('left_predictions', 'right_predictions', 'input_pairs', 'calibration_dir')},
        'first_frame_observations': params['first_frame_observations'], 'capture_times': str(capture),
        'frames': [0, count-1], 'joint_names': JOINT_NAMES,
        'method': 'Existing independent9-state Cartesian[p,v,a] pixel-feedback EKF per joint;triangulation only initializes/restarts and gates,not a3D observation update',
        'initialization': 'No initial_points_mm supplied: first valid stereo pair per joint initializes3D,zero v/a and frozen P0. No SMPL-H pose,shape,mesh or pretrained model read by tracker.',
        'configurations_frozen_before_run': {k: asdict(v) for k, v in configs.items()},
        'control_variable': 'The second pass changes only pixel_sigma_px4->8,matching the SMPL-H pixel scale;no parameter search',
        'primary_window': [35, 61],
        'decision': 'Both ankle tracks must be finite and uninterrupted. A persistent rise is defined diagnostically as mean endpoint AND median5-endpoint rise>=30mm. If both PVA passes avoid that rise while SMPL-H shows it,prioritize the coupled SMPL-H route;do not claim SMPL-H asset is intrinsically wrong.',
        'stop': 'One full448 pass per predeclared configuration;no retries or parameter sweeps. Missing/rejected/predicted/restarted points remain in all output rows. If ankles missing/restarted in primary window,conclusion is inconclusive.',
        'reference': {'scene': str(args.scene.resolve()), 'role': 'Existing same-source Stage/R/t reused ONLY posthoc for numerical comparison;never fed into tracker. No fresh3D scene/viewer is produced.',
                      'static_episode_transform_check': transform_audit,
                      'world_truth': False},
        'limitations': ['Route comparison,not single-variable SMPL-H ablation:Cartesian state,initialization,process/gates differ.',
                       'Free endpoints have no fixed-bone/anatomical coupling and cannot establish a valid rigid skeleton.',
                       'Same biased2D/fisheye calibration and host-time can affect both;neither comparison establishes actual3D/physical accuracy.',
                       'COCO ankle height is not sole/heel height. Only static-camera episodes support motion attribution;full moving-stage world values are reference-conditioned.',
                       'Do not rotate camera-relative v/a into purported world derivatives;archive those as camera-relative only.',
                       'SMPL-H results are loaded for comparison only AFTER independent tracking finishes.']}
    if not args.analyze_only:
        write_json(args.output / 'protocol.json', protocol)
    tracked = {}
    for name, config in configs.items():
        if args.analyze_only:
            with np.load(args.output / f'{name}.npz', allow_pickle=False) as archive:
                data = {k: archive[k].copy() for k in archive.files if k not in ('timestamp_s', 'pair_id')}
            previous = json.loads((args.output / 'summary.json').read_text(encoding='utf-8'))
            data['runtime_seconds'] = previous['configurations'][name]['runtime_seconds']
            data['segments'] = [segment_metrics(data['world_reference_points_mm'], segment, data['status'], data['segment_id']) for segment in stage1]
            tracked[name] = data
            continue
        tracker = StereoPVATracker(cal, 12, config)
        states, status, reasons, segments, reprojection, ages, variances = [], [], [], [], [], [], []
        started = perf_counter()
        for i, now in enumerate(times):
            pixels = observations[i]
            result = tracker.step(float(now), pixels[0, :, :2], pixels[1, :, :2], pixels[0, :, 2], pixels[1, :, 2])
            states.append(result['state']); status.append(result['status']); reasons.append(result['reason'])
            segments.append(result['segment_id']); reprojection.append(result['reprojection_px'])
            ages.append(result['prediction_age_s']); variances.append(np.diagonal(result['covariance'], axis1=1, axis2=2))
        data = {'state_mm': np.array(states), 'status': np.array(status, dtype=str), 'reason': np.array(reasons, dtype=str),
                'segment_id': np.array(segments), 'reprojection_px': np.array(reprojection), 'prediction_age_s': np.array(ages),
                'covariance_diagonal': np.array(variances), 'runtime_seconds': perf_counter()-started}
        data['camera_points_mm'] = data['state_mm'][..., :3]
        data['world_reference_points_mm'] = np.einsum('tij,tkj->tki', rotations, data['camera_points_mm']) + translations[:, None]
        data['segments'] = [segment_metrics(data['world_reference_points_mm'], segment, data['status'], data['segment_id']) for segment in stage1]
        tracked[name] = data
        np.savez_compressed(args.output / f'{name}.npz', timestamp_s=times, pair_id=pair_ids,
                            **{k: v for k, v in data.items() if isinstance(v, np.ndarray)})
        print(name, 'completed', count, 'status', dict(Counter(data['status'].ravel().tolist())), flush=True)
    # Tracking above consumed no SMPL-H result fields. Comparison begins here.
    with np.load(args.source_run / 'result.npz', allow_pickle=False) as fit:
        if not np.allclose(fit['timestamp_s'], times, atol=1e-7, rtol=0):
            raise ValueError('SMPL-H comparison frame times differ')
        smpl_camera = fit['predicted_coco'][:, 5:17].astype(float) * 1000
    smpl_world = np.einsum('tij,tkj->tki', rotations, smpl_camera) + translations[:, None]
    smpl_errors = np.full((count, 12, 2), np.nan)
    diagnostic_projector = StereoPVATracker(cal, 12)
    for i in range(count):
        for j in range(12):
            projection = diagnostic_projector._project(smpl_camera[i, j])
            if projection is not None:
                smpl_errors[i, j] = np.linalg.norm(projection.reshape(2, 2)-observations[i, :, j, :2], axis=1)
    raw = observations[..., :2]
    raw_ok = np.isfinite(observations).all(axis=-1) & (observations[..., 2] >= .2) & (observations[..., 2] <= 1)
    raw_ok &= ((raw[..., 0] >= 0) & (raw[..., 0] < cal.image_width)
               & (raw[..., 1] >= 0) & (raw[..., 1] < cal.image_height))
    raw_ok = raw_ok.transpose(0, 2, 1)
    def errors_summary(errors):
        return {'all_raw_slots': stats(np.where(raw_ok, errors, np.nan)),
                'hip_raw_slots': stats(np.where(raw_ok[:, 6:8], errors[:, 6:8], np.nan)),
                'hip_slots_total': count*2*2, 'all_slots_total': count*12*2}
    summary = {'frames_retained': count, 'primary_window': [35, 61], 'configurations': {},
               'smplh_comparison': {'raw_reprojection_px': errors_summary(smpl_errors),
                   'events': [{'frame': i, 'ankle_height_change_from_previous_mm': smpl_world[i, 10:12, 2]-smpl_world[i-1, 10:12, 2]}
                              for i in (37, 59)],
                   'static_segments': [segment_metrics(smpl_world, segment, np.full((count, 12), 'model', dtype='<U5'),
                                                       np.zeros((count, 12), int)) for segment in stage1]},
               'limitations': protocol['limitations']}
    bones = {'hip_width': (6, 7), 'left_thigh': (6, 8), 'right_thigh': (7, 9),
             'left_shank': (8, 10), 'right_shank': (9, 11)}
    for name, data in tracked.items():
        row = {'configuration': asdict(configs[name]), 'runtime_seconds': data['runtime_seconds'],
               'status_counts': dict(Counter(data['status'].ravel().tolist())),
               'reason_counts': dict(Counter(data['reason'].ravel().tolist())),
               'raw_reprojection_px': errors_summary(data['reprojection_px']),
               'static_segments': data['segments'],
               'bone_lengths_mm': {bone: stats(np.linalg.norm(data['camera_points_mm'][:, a]-data['camera_points_mm'][:, b], axis=1))
                                   for bone, (a, b) in bones.items()},
               'events': [{'frame': i, 'ankle_height_change_from_previous_mm': data['world_reference_points_mm'][i, 10:12, 2]-data['world_reference_points_mm'][i-1, 10:12, 2],
                           'ankle_status': data['status'][i, 10:12], 'hip_raw_residual_px': data['reprojection_px'][i, 6:8]}
                          for i in (37, 59)],
               'no_smplh_tracker_input': True}
        common = raw_ok & np.isfinite(data['reprojection_px']) & np.isfinite(smpl_errors)
        row['same_finite_slots_comparison'] = {
            'total_all_slots': count*12*2, 'total_hip_slots': count*2*2,
            'pva_all_px': stats(data['reprojection_px'][common]), 'smplh_all_px': stats(smpl_errors[common]),
            'pva_hips_px': stats(data['reprojection_px'][:, 6:8][common[:, 6:8]]),
            'smplh_hips_px': stats(smpl_errors[:, 6:8][common[:, 6:8]])}
        row['per_joint_status_counts'] = {joint: dict(Counter(data['status'][:, j].tolist())) for j, joint in enumerate(JOINT_NAMES)}
        summary['configurations'][name] = row
    write_json(args.output / 'summary.json', summary)
    plot(args.output / 'comparison.png', times, tracked, smpl_world, smpl_errors)
    print(json.dumps(safe_json({name: {'primary': next(r for r in row['static_segments'] if r['frames'] == [35, 61]),
                                                   'hips': row['raw_reprojection_px']['hip_raw_slots']}
                                for name, row in summary['configurations'].items()})), flush=True)


def plot(path, times, tracked, smpl_world, smpl_errors):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    labels = {'pva_frozen_sigma4': ('PVA sigma4', '-'), 'pva_matched_pixel_sigma8': ('PVA sigma8', '--')}
    for ax, j, name in zip(axes[0], (10, 11), ('Left ankle', 'Right ankle')):
        for key, (label, style) in labels.items():
            ax.plot(times[35:62], tracked[key]['world_reference_points_mm'][35:62, j, 2], style, label=label, linewidth=1.5)
        ax.plot(times[35:62], smpl_world[35:62, j, 2], ':', label='SMPL-H MAP', color='black', linewidth=2)
        ax.set_title(f'{name}: fixed-camera frames35..61'); ax.set_ylabel('COCO ankle height (mm)')
        ax.set_xlabel('Host-return time (s)'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    for ax, joint, name in zip(axes[1], (6, 7), ('Left hip', 'Right hip')):
        for key, (label, style) in labels.items():
            # Both camera residuals retained; report their maximum for each hip/frame.
            values = np.max(tracked[key]['reprojection_px'][:, joint], axis=1)
            ax.plot(times, values, style, label=label, linewidth=.8)
        ax.plot(times, np.max(smpl_errors[:, joint], axis=1), ':', color='black', label='SMPL-H MAP', linewidth=1)
        ax.set_title(f'{name}: max of two raw-image residuals'); ax.set_ylabel('Reprojection residual (px)')
        ax.set_xlabel('Host-return time (s)'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle('Tracked skeleton comparison: no contact/ground constraints; heights are reference-conditioned')
    fig.savefig(path, dpi=170, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
