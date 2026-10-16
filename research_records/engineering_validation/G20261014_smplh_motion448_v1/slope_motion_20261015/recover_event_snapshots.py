"""Recover exact pre-update states for identified jumps from a nearby trace."""
import json
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import prefix_derivative_feedback as shared  # noqa: E402
from pose_app.smplh_motion_tracker import MotionPose  # noqa: E402


def main():
    source = OUT.parent / 'first_frame_sapiens_20261015/replay_selected_init/map_sigma075'
    p = json.loads((source/'run_parameters.json').read_text(encoding='utf-8'))
    inp = p['inputs']
    model = shared.SMPLHObservationMap(*[Path(inp[k]) for k in
        ('initial_fit', 'smplh_model', 'regressor', 'vposer_dir', 'mano_left', 'mano_right')], p['device'])
    tr = shared.SMPLHIteratedMotionTracker(model, shared.load_stereo_fisheye(Path(inp['calibration_dir'])),
        shared.SMPLHMotionConfig(**p['configuration']), shared.IteratedMAPConfig(**p['iterated_map_configuration']))
    left = shared._load_predictions(Path(inp['left_predictions']), 'left')
    right = shared._load_predictions(Path(inp['right_predictions']), 'right')
    saved = np.load(source/'result.npz', allow_pickle=False)
    snapshots = dict(np.load(OUT/'motion_chain_snapshots.npz', allow_pickle=False))
    tr.covariance = snapshots['35_P'].copy()
    tr.pose = MotionPose(snapshots['35_prior_t'], snapshots['35_prior_R'], snapshots['35_prior_z'])
    pixels = snapshots['35_pixels']
    valid, reasons = tr._supported_pixels(pixels)
    tr._update(pixels, valid, reasons)
    # Repeating the exact35 update restores the full post-P; initialize v/a
    # from saved35 because the snapshot stores the pre-update pose/P only.
    tr.velocity = saved['generalized_velocity'][35].copy()
    tr.acceleration = saved['generalized_acceleration'][35].copy()
    checks = []
    for i in range(36, 60):
        tr._predict(float(saved['timestamp_s'][i]-saved['timestamp_s'][i-1]))
        pixels = np.array([left[i, 5:17], right[i, 5:17]])
        valid, reasons = tr._supported_pixels(pixels)
        prior, P = tr.pose.copy(), tr.covariance.copy()
        tr._update(pixels, valid, reasons)
        diffs = [np.max(abs(tr.pose.translation_m-saved['transl'][i])),
                 np.max(abs(tr.pose.root_rotation-saved['root_rotation'][i])),
                 np.max(abs(tr.pose.latent-saved['vposer_latent'][i]))]
        assert max(diffs) == 0, (i, diffs)
        if i in (37, 59):
            snapshots.update({f'{i}_{k}': v for k, v in
                {'prior_t': prior.translation_m, 'prior_R': prior.root_rotation, 'prior_z': prior.latent,
                 'P': P, 'pixels': pixels, 'valid': valid, 'post_t': tr.pose.translation_m,
                 'post_R': tr.pose.root_rotation, 'post_z': tr.pose.latent}.items()})
            print('recovered', i, 'post exact', diffs, flush=True)
        checks.append({'frame': i, 'maximum_pose_difference': float(max(diffs))})
    np.savez_compressed(OUT/'motion_chain_snapshots.npz', **snapshots)
    (OUT/'event_snapshot_recovery.json').write_text(json.dumps({
        'method': 'restore actual35 prior P;repeat identical35 update;initialize saved35 v/a;repeat actual36..59 predict/update',
        'purpose': 'pre-update full-P for two traced upward-jump events,without resetting diagnostic covariance',
        'event_frames': [37, 59], 'checks': checks}, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
