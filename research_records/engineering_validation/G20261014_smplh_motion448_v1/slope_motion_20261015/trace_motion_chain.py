"""Trace exact saved camera-MAP predictions and updates; no production changes."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import prefix_derivative_feedback as shared  # noqa: E402


def main():
    source = OUT.parent / 'first_frame_sapiens_20261015/replay_selected_init/map_sigma075'
    p = json.loads((source / 'run_parameters.json').read_text(encoding='utf-8'))
    inp = p['inputs']
    model = shared.SMPLHObservationMap(*[Path(inp[k]) for k in
        ('initial_fit', 'smplh_model', 'regressor', 'vposer_dir', 'mano_left', 'mano_right')], p['device'])
    cal = shared.load_stereo_fisheye(Path(inp['calibration_dir']))
    left = shared._load_predictions(Path(inp['left_predictions']), 'left')
    right = shared._load_predictions(Path(inp['right_predictions']), 'right')
    shared.apply_first_frame_observations(Path(p['first_frame_observations']), Path(inp['input_pairs']), left, right)
    times, *_ = shared._read_capture_times(Path(inp['input_pairs']), len(left))
    scene = np.load(OUT / 'scene/scene_transforms.npz', allow_pickle=False)
    rotations, translations = scene['rotation_ground_from_left'], scene['translation_ground_from_left_mm']/1000
    soles = shared.load_sole_vertex_indices(shared.ROOT / 'research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/contact_surface_sets_smplh_v1/contact_vertex_sets.json')
    snapshots, rows = {}, []

    class Traced(shared.SMPLHIteratedMotionTracker):
        def describe(self, pose):
            data = model.evaluate(pose, mesh=True)
            world = data['coco_m'][0] @ rotations[self.frame].T + translations[self.frame]
            vertices = data['vertices_m'][0] @ rotations[self.frame].T + translations[self.frame]
            return {'mean_ankle_z_mm': float(world[15:17, 2].mean()*1000),
                    'ankle_z_mm': (world[15:17, 2]*1000).tolist(),
                    'sole_z_mm': [float(vertices[ix, 2].min()*1000) for ix in soles]}

        def _predict(self, dt):
            self.before_predict = self.describe(self.pose)
            super()._predict(dt)

        def _update(self, pixels, valid, reasons):
            prior, P = self.pose.copy(), self.covariance.copy()
            predicted = self.describe(prior)
            result = super()._update(pixels, valid, reasons)
            posterior = self.describe(self.pose)
            if self.frame in (35, 48, 61):
                snapshots.update({f'{self.frame}_{k}': value for k, value in
                    {'prior_t': prior.translation_m, 'prior_R': prior.root_rotation, 'prior_z': prior.latent,
                     'P': P, 'pixels': pixels, 'valid': valid,
                     'post_t': self.pose.translation_m, 'post_R': self.pose.root_rotation,
                     'post_z': self.pose.latent}.items()})
            px0 = self._pixels(model.evaluate(prior)['coco_m'])[0]
            px1 = self._pixels(model.evaluate(self.pose)['coco_m'])[0]
            sigma, huber = self.config.pixel_sigma_px, self.config.huber_px

            def pixel_terms(projected):
                norm = np.linalg.norm(projected-pixels[..., :2], axis=-1)
                return np.where(valid, np.where(norm <= huber, .5*norm**2, huber*(norm-.5*huber))/sigma**2, 0)

            delta = np.r_[self.pose.translation_m-prior.translation_m,
                          Rotation.from_matrix(prior.root_rotation.T @ self.pose.root_rotation).as_rotvec(),
                          self.pose.latent-prior.latent]
            row = {'frame': self.frame, 'before_prediction': self.before_predict,
                   'prediction': predicted, 'posterior': posterior,
                   'prediction_height_increment_mm': predicted['mean_ankle_z_mm']-self.before_predict['mean_ankle_z_mm'],
                   'update_height_increment_mm': posterior['mean_ankle_z_mm']-predicted['mean_ankle_z_mm'],
                   'pixel_cost_before_by_view_joint': pixel_terms(px0).tolist(),
                   'pixel_cost_after_by_view_joint': pixel_terms(px1).tolist(),
                   'vposer_cost_before': float(.5*prior.latent@prior.latent/self.config.latent_prior_sigma**2),
                   'vposer_cost_after': float(.5*self.pose.latent@self.pose.latent/self.config.latent_prior_sigma**2),
                   'state_cost_after': float(.5*delta @ np.linalg.solve(P[:self.n, :self.n], delta)),
                   'valid': valid.tolist(), 'optimizer': {k:v for k,v in self.optimizer_diagnostic.items() if k != 'history'}}
            rows.append(row)
            return result

    tr = Traced(model, cal, shared.SMPLHMotionConfig(**p['configuration']), shared.IteratedMAPConfig(**p['iterated_map_configuration']))
    output_points = []
    for i in range(62):
        tr.frame = i
        result = tr.step(float(times[i]), left[i, 5:17], right[i, 5:17])
        output_points.append(result['coco_m'])
        if i % 15 == 0 or i == 61:
            print('traced', i, flush=True)
    old = np.load(source / 'result.npz', allow_pickle=False)
    diff = float(np.max(abs(np.array(output_points)-old['predicted_coco'][:62])))
    assert diff == 0, diff
    np.savez_compressed(OUT / 'motion_chain_snapshots.npz', **snapshots)
    chosen = [r for r in rows if 36 <= r['frame'] <= 61]
    report = {'source': str(source), 'frames': [0, 61], 'saved_baseline_max_coco_difference_m': diff,
              'question': 'At which operation does the Stage1 rise arise, and which objective terms accept it?',
              'observation_policy': 'same Sapiens0/PMPose1..61;frozen beta;whole VPoser sigma0.75;no new loss/ground input',
              'stage1_35_61': {'prediction_height_sum_mm': sum(r['prediction_height_increment_mm'] for r in chosen),
                               'update_height_sum_mm': sum(r['update_height_increment_mm'] for r in chosen)},
              'snapshot_frames': [35, 48, 61], 'rows': rows,
              'limits': 'Trace and objective attribution only;not physical truth or proof of a unique cause.'}
    (OUT / 'motion_chain_trace.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(report['stage1_35_61'], flush=True)


if __name__ == '__main__':
    main()
