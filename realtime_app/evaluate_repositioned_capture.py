"""Read-only diagnostics for the paired September 27 cold-start experiment.

Detector disagreement and fit residuals are internal consistency, not accuracy.
Host-return times are used only for temporal diagnostics, never optimization.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from check_mainline_regression import fit_arguments
from experiment_repositioned_capture import read, write
from pose_app.mainline_regression import load_archive, observation_snapshot
from pose_app.smplh_ablation_metrics import stats, leg_diagnostics
from pose_app.fisheye_camera import fisheye_project_numpy


def host_time_diagnostics(joints, times):
    times = np.asarray(times, dtype=np.float64)
    if times.shape != (len(joints),) or not np.isfinite(times).all():
        raise ValueError('invalid host timestamp array')
    dt = np.diff(times)
    if len(dt) < 2 or np.any(dt <= 0):
        raise ValueError('host timestamps must increase')
    relative = joints[:, [4, 5, 7, 8]] - joints[:, [1, 2]].mean(1)[:, None]
    velocity = np.diff(relative, axis=0) / dt[:, None, None]
    acceleration = np.diff(velocity, axis=0) / ((dt[1:] + dt[:-1]) * .5)[:, None, None]
    return {'host_interval_s': stats(dt),
            'leg_relative_speed_m_s': stats(np.linalg.norm(velocity, axis=-1)),
            'leg_relative_acceleration_m_s2': stats(np.linalg.norm(acceleration, axis=-1)),
            'time_basis': 'mean of paired host-return times; not exposure timestamps'}


def image_diagnostics(out):
    """Same-frame overlays; no point relocation or reference-model promotion."""
    predictions = {}
    direction = {}
    for side in ('left', 'right'):
        for method in ('pmpose', 'sapiens'):
            name = f'{side}_pmpose_raw.json' if method == 'pmpose' else f'{side}_sapiens_coco17_model_input.json'
            predictions[side, method] = np.array([r['keypoints'][0][:17] for r in read(out/name)['images']])
        a, b = predictions[side, 'pmpose'], predictions[side, 'sapiens']
        for group, (j, k) in (('shoulders', (5, 6)), ('hips', (11, 12))):
            dot = ((a[:, j, :2]-a[:, k, :2])*(b[:, j, :2]-b[:, k, :2])).sum(-1)
            indices = np.flatnonzero(dot < 0)
            direction[side+'_'+group] = {'opposite_direction_frames': int(len(indices)),
                'original_pair_ids': (1157+indices).tolist(),
                'boundary': 'direction disagreement relative to PMPose, not annotated swap-error rate'}
    write(out/'side_direction_diagnostics.json', direction)
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(12, 11))
    edges = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
             (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
    for row, i in enumerate((20, 41)):
        for camera, side in enumerate(('left', 'right')):
            folder = 'left_ccw90' if side == 'left' else 'right_cw90'
            frame = cv2.imread(str(out/'input'/folder/f'pair_{i:04d}.png'))
            if frame is None:
                raise FileNotFoundError(f'{side}/{i}')
            for column, method in enumerate(('pmpose', 'sapiens')):
                points = predictions[side, method][i]
                ax = axes[row, camera*2+column]
                ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                color = 'lime' if method == 'pmpose' else 'magenta'
                ax.scatter(points[:, 0], points[:, 1], s=7, c=color)
                for j, k in edges:
                    ax.plot(points[[j, k], 0], points[[j, k], 1], lw=.8, c=color)
                for j in (5, 6, 7, 8, 9, 10):
                    ax.text(points[j, 0]+8, points[j, 1], str(j), color='white', fontsize=7)
                ax.set_title(f'{side} {method} pair {1157+i}', fontsize=9)
                ax.set_xlim(0, 1080)
                ax.set_ylim(1920, 0)
                ax.axis('off')
    fig.tight_layout()
    fig.savefig(out/'shoulder_diagnostic.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    image_diagnostics(out)
    frames = read(out/'frames.json')
    times = np.array([(int(r['left_host_return_timestamp_ns']) + int(r['right_host_return_timestamp_ns'])) / 2e9
                      for r in frames])
    times -= times[0]
    report = {'boundary': 'engineering consistency only; no human wrist or independent 3D truth',
              'manual_wrist_reference': 'disabled in both fits', 'methods': {}}
    report['capture_timing'] = {
        'abs_host_skew_ms': stats(np.array([float(r['abs_host_delta_ms']) for r in frames])),
        'time_basis': 'host_return_timestamp_ns, not exposure or hardware sync',
        'strict_replay_time_basis': 'unchanged offline evaluator uses synthetic pair_index/30 timestamps',
        'fitting_sync_weight': 'unchanged body triangulator qsync=1; no new temporal compensation'}
    observed, results = {}, {}
    for method in ('pmpose', 'sapiens2'):
        directory = out/method
        audit = read(directory/'implementation_audit.json')
        observed[method] = obs = observation_snapshot(fit_arguments(directory/'fit_command.json'))
        if audit.get('status') == 'fit_failed':
            report['methods'][method] = {'status': 'fit_failed', 'failure': audit['failure'],
                'initialization': audit['initialization'], 'frames': len(obs['pair_id']),
                'accepted_body_points': int(obs['accepted'].sum()),
                'body_rejection_reasons': dict(Counter(obs['reason'][~obs['accepted']].tolist())),
                'strict_stereo': read(directory/'strict_stereo/offline_stereo_summary.json'),
                'fitted_metrics': 'unavailable; do not substitute older results'}
            continue
        assert all(c['passed'] for c in audit['controls'].values())
        assert all(c['passed'] for c in audit['parameter_replay'].values())
        results[method] = z = load_archive(directory/'fit/result.npz')
        mask = z['body_accepted']
        residual = np.linalg.norm(z['predicted_coco'] - obs['triangulated_mm']/1000, axis=-1)*1000
        quality = z['body_quality'][mask]
        assert quality.sum() > 0
        temporal, _ = leg_diagnostics(z['smplh_joints'], 30)
        summary = read(directory/'fit/fit_summary.json')
        metrics = {'status': 'fit_completed_and_audited', 'frames': len(z['pair_id']), 'betas': z['betas'].tolist(),
                   'accepted_body_points': int(mask.sum()),
                   'body_observation_mm': stats(residual[mask]),
                   'quality_weighted_body_rms_mm': float(np.sqrt((quality*residual[mask]**2).sum()/quality.sum())),
                   'body_joint_observation_mm': {str(j): stats(residual[:, j][mask[:, j]]) for j in range(17)},
                   'body_rejection_reasons': dict(Counter(obs['reason'][~mask].tolist())),
                   'nominal30fps_temporal': temporal,
                   'host_time_temporal': host_time_diagnostics(z['smplh_joints'], times),
                   'initialization': summary['initialization'],
                   'hand_geometry': read(directory/'fit/wilor_hand_geometry_summary.json'),
                   'native_mano_input_audit': read(directory/'fit/wilor_mano_parameter_audit.json'),
                   'strict_stereo': read(directory/'strict_stereo/offline_stereo_summary.json'),
                   'body_frozen_during_hand_stages': audit['body_frozen_during_hand_stages']}
        available = z['root_init_source'] == 'shoulder_hip_rigid_basis'
        metrics['initialization_support'] = {
            'root_source_counts': dict(Counter(z['root_init_source'].tolist())),
            'translation_source_counts': dict(Counter(z['translation_init_source'].tolist())),
            'basis_unavailable_original_pair_ids': [int(frames[i]['pair_id']) for i in np.flatnonzero(~available)],
            'body_residual_mm_by_initial_basis': {
                label: stats(residual[mask & select[:, None]])
                for label, select in (('available', available), ('unavailable', ~available))}}
        metrics['native_mano_target_consistency'] = {}
        for side in ('left', 'right'):
            pose = z[side+'_hand_pose'].reshape(-1, 15, 3)
            rotation = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(len(pose), 15, 3, 3)
            targets = z['mano_'+side+'_target_rotations']
            weights = z['mano_'+side+'_view_weights']
            cosine = np.clip(((rotation[:, None]*targets).sum(axis=(-1, -2))-1)*.5, -1, 1)
            metrics['native_mano_target_consistency'][side] = {
                'supported_view_frames': int((weights > 0).sum()),
                'rotation_residual_deg': stats(np.rad2deg(np.arccos(cosine))[weights > 0]),
                'interpretation': 'agreement with native WiLoR fitted target, not independent hand accuracy'}
        timing = directory/'fit.execution.json'
        metrics['fit_process_timing'] = read(timing) if timing.exists() else {'status': 'not_recorded'}
        metrics['pose2d_inference_timing_ms'] = {}
        for side in ('left', 'right'):
            if method == 'pmpose':
                with (out/f'{side}_pmpose_timing.csv').open(encoding='utf-8-sig') as handle:
                    elapsed = [float(row['elapsed_ms']) for row in csv.DictReader(handle)]
            else:
                # Native runner records one synchronized call for all prompted persons.
                elapsed = [json.loads(line)['elapsed_ms'] for line in (out/f'{side}_sapiens_raw.jsonl').read_text().splitlines()]
            assert len(elapsed) == len(frames)
            metrics['pose2d_inference_timing_ms'][side] = stats(elapsed)
        metrics['timing_boundary'] = 'Original runner timings; different runtimes; excludes capture, detector, WiLoR and fitting'
        for camera, side in enumerate(('left', 'right')):
            points = z['predicted_coco']
            if camera:
                points = points @ obs['R_cam0_to_cam1'].T + obs['T_cam0_to_cam1_mm']/1000
            pixels = fisheye_project_numpy(points.reshape(-1, 3), obs[f'K{camera}'], obs[f'D{camera}']).reshape(-1, 17, 2)
            px_error = np.linalg.norm(pixels-obs[side+'_raw_pixels'][:, :, :2], axis=-1)
            usable = mask & np.isfinite(px_error)
            metrics[side+'_body_reprojection_px'] = stats(px_error[usable])
            metrics[side+'_invalid_projection_accepted'] = int((mask & ~usable).sum())
        report['methods'][method] = metrics
        np.savez_compressed(directory/'evaluation_arrays.npz', pair_id=z['pair_id'],
                            accepted=mask, body_residual_mm=residual, host_seconds=times,
                            observation_reasons=obs['reason'])
    common = observed['pmpose']['accepted'] & observed['sapiens2']['accepted']
    tri_delta = np.linalg.norm(observed['pmpose']['triangulated_mm'] - observed['sapiens2']['triangulated_mm'], axis=-1)
    if len(results) != 2:
        report['between_models'] = {'common_accepted_points': int(common.sum()),
            'triangulation_disagreement_mm': stats(tri_delta[common]),
            'fitted_comparison': 'unavailable: one fit failed original initialization gate'}
        write(out/'comparison_metrics.json', report)
        print(out/'comparison_metrics.json')
        return
    fit_delta = np.linalg.norm(results['pmpose']['predicted_coco'] - results['sapiens2']['predicted_coco'], axis=-1)*1000
    report['between_models'] = {'common_accepted_points': int(common.sum()),
        'triangulation_disagreement_mm': stats(tri_delta[common]),
        'fitted_coco_disagreement_mm': stats(fit_delta),
        'wrist_elbow_triangulation_disagreement_mm': {str(j): stats(tri_delta[:, j][common[:, j]]) for j in (7, 8, 9, 10)},
        'common_mask_own_observation_residual_mm': {m: stats(
            np.linalg.norm(results[m]['predicted_coco']-observed[m]['triangulated_mm']/1000, axis=-1)[common]*1000)
            for m in results},
        'interpretation': 'Common frame/joint support; each fit has its own model observation, no accuracy ranking'}
    report['between_models']['native_mano_association_equal'] = {
        side: {field: bool(np.array_equal(results['pmpose']['mano_'+side+'_'+field],
                                         results['sapiens2']['mano_'+side+'_'+field], equal_nan=True))
               for field in ('target_rotations', 'view_weights')}
        for side in ('left', 'right')}
    report['between_models']['hand_input_note'] = (
        'Same newly inferred raw WiLoR records; unchanged association may select different hypotheses '
        'because it uses the respective body wrist observations.')
    write(out/'comparison_metrics.json', report)
    print(out/'comparison_metrics.json')


if __name__ == '__main__':
    main()
