"""Three independent matched-prior contact feasibility diagnostics.

These are single-frame optimizations, never a replacement trajectory. Each
prediction uses only saved previous-frame q/v/a/P and the original predictor.
"""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.optimize import minimize, nnls
from scipy.spatial.transform import Rotation

APP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(APP), str(APP / 'tools')]
from pose_app.smplh_motion_tracker import MotionPose, retract
from pose_app.foot_ground_reference import SHOE_HEIGHT_M
from probe_smplh_foot_observations import (
    HELDOUT,
    TOE_VERTICES,
    direction_error,
    load,
    project,
    save,
)
from probe_smplh_soft_contact import (
    BALL,
    OUT,
    REPAIR,
    SOLE,
    SURFACE,
    ContactPointMap,
    ContactTracker,
    contact_residuals,
    support_labels,
)
from smplh_contact_autograd import AutogradContactTracker, so3_exp_torch

FRAMES = [40, 55, 61]
DEST = OUT / 'contact_matched_prior_feasibility'


def restore_prediction(tracker, reference, frame, times):
    previous = frame-1
    tracker.pose = MotionPose(reference['transl'][previous].copy(), reference['root_rotation'][previous].copy(),
                             reference['vposer_latent'][previous].copy())
    tracker.velocity = reference['velocity'][previous].copy()
    tracker.acceleration = reference['acceleration'][previous].copy()
    tracker.covariance = reference['covariance'][previous].copy()
    tracker._predict(float(times[frame]-times[previous]))
    return tracker.pose.copy(), tracker.covariance.copy()


def observation_spec(tracker, body, toes, frame):
    valid, reasons = tracker._supported_pixels(body)
    use_feet = False
    pixels = body.copy()
    if frame not in HELDOUT and np.any(valid, axis=0).sum() >= tracker.config.min_observed_joints:
        padded = np.full((2, 12, 3), np.nan)
        padded[:, :4] = toes
        supported, _ = tracker._supported_pixels(padded)
        raw = toes[..., :2].reshape(2, 2, 2, 2)
        vector = raw[..., 1, :]-raw[..., 0, :]
        eligible = supported[:, :4].reshape(2, 2, 2).all(axis=-1) & (np.linalg.norm(vector, axis=-1) >= 16.)
        if eligible.any():
            extra, foot_valid = np.zeros_like(toes), np.zeros((2, 4), bool)
            extra[:, [0, 2], :2], foot_valid[:, [0, 2]] = vector, eligible
            pixels = np.concatenate((pixels, extra), axis=1)
            valid = np.concatenate((valid, foot_valid), axis=1)
            use_feet = True
    return pixels, valid, reasons, use_feet


class MatchedProblem:
    def __init__(self, tracker, prior, covariance, body, toes, frame, rotation, translation):
        self.tracker, self.prior = tracker, prior
        self.rotation, self.translation = rotation, translation
        self.body, self.toes, self.frame = body, toes, frame
        self.support = support_labels()[frame]
        self.L = np.linalg.cholesky(covariance[:tracker.n, :tracker.n])
        self.body_valid = tracker._supported_pixels(body)[0]
        pixels, valid, _, use_feet = observation_spec(tracker, body, toes, frame)
        self.selected = np.repeat(valid.ravel(), 2)
        self.target = pixels[..., :2].reshape(-1)[self.selected]
        tracker.observation_mode = 'direction'
        tracker.solve_with_feet = use_feet
        tracker.extra_pixels = None if frame in HELDOUT else toes
        tracker.contact_rotation = None
        self.cache = {}

    def _torch_heights(self, delta):
        model = self.tracker.model
        def tensor(value):
            return torch.as_tensor(value, dtype=delta.dtype, device=delta.device)
        latent = (tensor(self.prior.latent)+delta[6:]).float()[None]
        body = model.vposer.decode(latent)['pose_body_matrot'].reshape(1, 21, 3, 3).clone()
        body[:, model.fixed_body_indices] = model.fixed_body[list(model.fixed_body_indices)]
        result = model.model(betas=model.betas,
            transl=(tensor(self.prior.translation_m)+delta[:3]).float()[None],
            global_orient=(tensor(self.prior.root_rotation)@so3_exp_torch(delta[3:6])).float()[None],
            body_pose=body, left_hand_pose=model.hand_rotations[0], right_hand_pose=model.hand_rotations[1])
        surface = result.vertices[0, SURFACE].to(delta.dtype)
        return (surface@tensor(self.rotation).T+tensor(self.translation))[:, 2]

    def evaluate(self, y, jacobian=False):
        if not self.cache or not np.array_equal(y, self.cache['y']):
            delta = self.L@y
            pose = retract(self.prior, delta)
            data = self.tracker.model.evaluate(pose, mesh=True)
            projected = self.tracker._pixels(data['coco_m'])
            if projected is None:
                raise ValueError('SLSQP entered the original invalid projection domain; no candidate accepted')
            values = projected.reshape(-1)[self.selected]
            residual = (values-self.target).reshape(-1, 2)
            norms = np.linalg.norm(residual, axis=-1)
            cfg = self.tracker.config
            pixel = float(np.where(norms <= cfg.huber_px, .5*norms**2,
                cfg.huber_px*(norms-.5*cfg.huber_px)).sum()/cfg.pixel_sigma_px**2)
            prediction, latent = float(.5*y@y), float(.5*(pose.latent@pose.latent)/cfg.latent_prior_sigma**2)
            heights = (data['vertices_m'][0, SURFACE].astype(float)@self.rotation.T+self.translation)[:, 2]
            ball = np.array([heights[ids].min() for ids in BALL])
            # All constraints are >=0. Latent bounds preserve the original domain.
            constraints = np.r_[heights-SHOE_HEIGHT_M+.005, SHOE_HEIGHT_M+.010-ball[self.support], cfg.max_abs_latent-pose.latent,
                                cfg.max_abs_latent+pose.latent]
            self.cache = {'y': y.copy(), 'delta': delta, 'pose': pose, 'data': data, 'pixel_values': values,
                'residual': residual, 'pixel': pixel, 'prediction': prediction, 'latent': latent,
                'objective': pixel+prediction+latent, 'heights': heights, 'ball': ball, 'constraints': constraints}
        if jacobian and 'gradient' not in self.cache:
            value = torch.as_tensor(self.cache['delta'], dtype=torch.float64, device=self.tracker.model.device).requires_grad_(True)
            def function(delta):
                pixels = self.tracker._torch_objective_values(self.prior, delta, self.selected)
                return torch.cat((pixels, self._torch_heights(delta)))
            derivative = torch.autograd.functional.jacobian(function, value, vectorize=True).detach().cpu().numpy()@self.L
            pixel_count = len(self.target)
            J, height_J = derivative[:pixel_count], derivative[pixel_count:]
            cfg = self.tracker.config
            norms = np.linalg.norm(self.cache['residual'], axis=-1)
            robust = np.repeat(np.minimum(1., cfg.huber_px/np.maximum(norms, 1e-12)), 2)/cfg.pixel_sigma_px**2
            gradient = J.T@(robust*self.cache['residual'].ravel())+y+self.L[6:].T@self.cache['pose'].latent/cfg.latent_prior_sigma**2
            ball_J = np.array([height_J[ids[np.argmin(self.cache['heights'][ids])]] for ids in BALL])
            self.cache.update(gradient=gradient, constraint_jacobian=np.r_[height_J, -ball_J[self.support], -self.L[6:], self.L[6:]])
        return self.cache

    def objective(self, y):
        return self.evaluate(y)['objective']

    def gradient(self, y):
        return self.evaluate(y, True)['gradient']

    def constraints(self, y):
        return self.evaluate(y)['constraints']

    def constraint_jacobian(self, y):
        return self.evaluate(y, True)['constraint_jacobian']

    def report(self, pose):
        delta = np.r_[pose.translation_m-self.prior.translation_m,
            Rotation.from_matrix(self.prior.root_rotation.T@pose.root_rotation).as_rotvec(), pose.latent-self.prior.latent]
        y = np.linalg.solve(self.L, delta)
        value = self.evaluate(y, True)
        data, heights, ball = value['data'], value['heights'], value['ball']
        errors = np.linalg.norm(project(data['coco_m'][0, 5:17], self.tracker.cal)-self.body[..., :2], axis=-1)
        directions = direction_error(project(data['vertices_m'][0, TOE_VERTICES], self.tracker.cal), self.toes[..., :2])
        engineering = contact_residuals(data['vertices_m'][:, SURFACE], self.rotation, self.translation, self.support)[0]
        fit_points = data['coco_m'][0, 5:21 if self.tracker.solve_with_feet else 17]
        other_points = fit_points@self.tracker.cal.R_cam0_to_cam1.T+self.tracker.cal.T_cam0_to_cam1_mm/1000
        minimum_depth = float(min(fit_points[:, 2].min(), other_points[:, 2].min()))
        active = value['constraints'] <= 1e-5
        lagrangian = None
        if active.any():
            multipliers, _ = nnls(value['constraint_jacobian'][active].T, value['gradient'])
            lagrangian = float(np.linalg.norm(value['gradient']-value['constraint_jacobian'][active].T@multipliers, ord=np.inf))
        report = {'costs': {'pixel': value['pixel'], 'prediction': value['prediction'], 'latent': value['latent'],
            'original_map': value['objective'], 'soft_contact': float(.5*engineering@engineering)},
            'available': bool(np.isfinite(data['vertices_m']).all()), 'support_ball_mm': (ball[self.support]*1000).tolist(),
            'raw_body_errors_px_view_joint': errors.tolist(), 'body_fit_valid_view_joint': self.body_valid.tolist(),
            'minimum_fitted_camera_depth_m': minimum_depth, 'original_projection_domain_valid': minimum_depth > 1e-4,
            'sole_min_mm': [float(heights[:len(SOLE[0])].min()*1000), float(heights[len(SOLE[0]):].min()*1000)],
            'direction_errors_deg': directions.tolist(), 'direction_median_deg': float(np.median(directions)),
            'direction_max_deg': float(directions.max()), 'direction_was_withheld': self.frame in HELDOUT,
            'original_gradient_inf': float(np.linalg.norm(value['gradient'], ord=np.inf)),
            'estimated_active_set_kkt_inf': lagrangian,
            'constraint_min_margin_m': float(value['constraints'][:len(SURFACE)+self.support.sum()].min()),
            'constraint_violation_max_m': float(max(0., -value['constraints'][:len(SURFACE)+self.support.sum()].min())),
            'latent_domain_violation': float(max(0., np.max(abs(pose.latent))-self.tracker.config.max_abs_latent)),
            'active_constraints_at1e_5': int(active.sum()), 'delta': delta.tolist(), 'whitened_delta': y.tolist()}
        if hasattr(self, 'reference_pose'):
            changes = np.rad2deg(Rotation.from_matrix(self.reference_body.transpose(0, 2, 1)@data['body_rotations'][0]).magnitude())
            report.update(root_change_from_saved_deg=float(np.rad2deg(Rotation.from_matrix(
                self.reference_pose.root_rotation.T@pose.root_rotation).magnitude())),
                root_translation_change_from_saved_mm=float(1000*np.linalg.norm(pose.translation_m-self.reference_pose.translation_m)),
                maximum_body_joint_change_deg=float(changes.max()), body_joint_changes_from_saved_deg=changes.tolist())
        for name, ids in [('hip', [6, 7]), ('nonhip', [0, 1, 2, 3, 4, 5, 8, 9, 10, 11])]:
            selected = errors[:, ids][self.body_valid[:, ids]]
            report[name] = {'slots': int(selected.size), 'median_px': float(np.median(selected)), 'p95_px': float(np.percentile(selected, 95))}
        return report, data


def gates(candidate, baseline, solver_success):
    result = {f'{part}_{stat}': candidate[part][stat] <= 1.05*baseline[part][stat]
        for part in ['hip', 'nonhip'] for stat in ['median_px', 'p95_px']}
    result.update(available=candidate['available'], solver_success=bool(solver_success),
        original_projection_domain=candidate['original_projection_domain_valid'],
        contact_feasible=candidate['constraint_violation_max_m'] <= 1e-6,
        latent_domain=candidate['latent_domain_violation'] <= 1e-8,
        direction_median=candidate['direction_median_deg'] <= max(3., baseline['direction_median_deg']*1.1),
        direction_max=candidate['direction_max_deg'] <= baseline['direction_max_deg']+2.)
    return result


def run():
    proto, model, initial_tracker, body, toes, times = load(62, ContactPointMap, AutogradContactTracker)
    original = model.regressor.clone()
    a, b = proto['hip_calibration_result']['alpha'], proto['hip_calibration_result']['thigh_fraction']
    model.regressor[11:13] = original[11:13]+a*(original[11:13]-original[[12, 11]])+b*(original[13:15]-original[11:13])
    reference = np.load(OUT / 'toe_direction_prefix/result.npz')
    scene = np.load(REPAIR / 'view/scene/scene_transforms.npz')
    rotations, translations = scene['rotation_ground_from_left'][:62], scene['translation_ground_from_left_mm'][:62]/1000
    DEST.mkdir(exist_ok=False)
    protocol = {'code': str(Path(__file__).relative_to(APP.parent)), 'command': [sys.executable, str(Path(__file__).resolve())],
        'frames': FRAMES, 'reference': str(OUT / 'toe_direction_prefix/result.npz'), 'ground': str(REPAIR / 'view/scene/scene_transforms.npz'),
        'configuration': proto['configuration'], 'iterated_map_configuration': proto['iterated_map_configuration'],
        'prediction': 'For each frame independently restore ONLY previous saved q/v/a/P; call original _predict(actual dt). No current posterior as prior, no cross-frame candidate propagation.',
        'controls': 'Original15-step FD soft-contact MAP versus autograd soft-contact MAP from exactly the same prediction and P.',
        'constrained': {'solver': 'SLSQP', 'maximum_iterations': 60, 'ftol': 1e-6,
            'objective': 'original pixel radial Huber/sigma8 + latent prior + y@y/2; NO soft-contact penalty',
            'constraints': '108 sole heights >=20mm; marked support ball minimum <=35mm (25mm shoe reference); abs(latent)<=6',
            'derivatives': 'torch autograd in float64 retraction/projection with original float32 VPoser/SMPLH; same fixed prediction chart; true minimum active vertex'},
        'support': support_labels()[FRAMES].tolist(), 'withheld_direction_frames': HELDOUT,
        'evaluation': 'Same eligible original raw-body slots across all three; original-map and soft-contact costs reported separately. Baseline is saved direction candidate at same frame, not an independent truth.',
        'gate': 'Every frame solver certified, contact violation<=1e-6m, abs(latent)<=6, all hip/nonhip median/P95<=1.05 saved baseline, direction median<=max(3,baseline*1.1), max<=baseline+2deg; all candidates and failures retained.',
        'stop': 'One3-frame matched-prior study, no weight/budget/offset sweep. Only an all-frame constrained pass supports considering62-frame validation.',
        'limits': 'Conditional reference ground and image support hypotheses; no physical contact truth or slip constraint. SLSQP is local and its failure does not prove global infeasibility. This study produces no temporal derivative/publishable posterior.'}
    save(DEST / 'protocol.json', protocol)
    summary, started = {}, perf_counter()
    archive = {'betas': model.betas.cpu().numpy(), 'left_hand_rotations': model.hand_rotations[0].cpu().numpy(),
        'right_hand_rotations': model.hand_rotations[1].cpu().numpy(), 'regressor': model.regressor.cpu().numpy(),
        'frames': np.array(FRAMES), 'timestamp_s': times[FRAMES]}
    for frame in FRAMES:
        rotation, translation = rotations[frame], translations[frame]
        prior, covariance = restore_prediction(initial_tracker, reference, frame, times)
        evaluator = AutogradContactTracker(model, initial_tracker.cal, initial_tracker.config, initial_tracker.optimizer_config)
        problem = MatchedProblem(evaluator, prior, covariance, body[frame], toes[frame], frame, rotation, translation)
        saved = MotionPose(reference['transl'][frame], reference['root_rotation'][frame], reference['vposer_latent'][frame])
        problem.reference_pose, problem.reference_body = saved, reference['body_rotations'][frame]
        baseline, _ = problem.report(saved)
        prediction, _ = problem.report(prior)
        frame_report = {'saved_direction_baseline': baseline, 'prediction': prediction,
            'prediction_dt_s': float(times[frame]-times[frame-1]),
            'prior_q_covariance_eigenvalues': np.linalg.eigvalsh(covariance[:38, :38]).tolist()}
        archive[f'{frame}_prior_covariance'] = covariance
        archive[f'{frame}_prior_transl'] = prior.translation_m
        archive[f'{frame}_prior_root_rotation'] = prior.root_rotation
        archive[f'{frame}_prior_vposer_latent'] = prior.latent
        for name, tracker_class in [('finite_soft_contact', ContactTracker), ('autograd_soft_contact', AutogradContactTracker)]:
            tracker = tracker_class(model, initial_tracker.cal, initial_tracker.config, initial_tracker.optimizer_config)
            restored, P = restore_prediction(tracker, reference, frame, times)
            if not (np.array_equal(P, covariance) and np.array_equal(restored.latent, prior.latent)
                    and np.array_equal(restored.root_rotation, prior.root_rotation) and np.array_equal(restored.translation_m, prior.translation_m)):
                raise ValueError('Controls do not have identical predicted q/P')
            tracker.observation_mode = 'direction'
            tracker.extra_pixels = None if frame in HELDOUT else toes[frame]
            tracker.contact_rotation, tracker.contact_translation, tracker.support = rotation, translation, support_labels()[frame]
            valid, reasons = tracker._supported_pixels(body[frame])
            tick = perf_counter()
            accepted, _, status = tracker._update(body[frame], valid, reasons)
            result, data = problem.report(tracker.pose)
            result.update(solver=tracker.optimizer_diagnostic, accepted=bool(accepted), status=status, runtime_s=perf_counter()-tick)
            result['gates'] = gates(result, baseline, tracker.optimizer_diagnostic.get('optimizer_success', False))
            result['passed'] = all(result['gates'].values())
            frame_report[name] = result
            for key, value in [('transl', tracker.pose.translation_m), ('root_rotation', tracker.pose.root_rotation),
                    ('vposer_latent', tracker.pose.latent), ('body_rotations', data['body_rotations'][0]), ('coco_m', data['coco_m'][0]),
                    ('covariance', tracker.covariance), ('velocity', tracker.velocity), ('acceleration', tracker.acceleration)]:
                archive[f'{frame}_{name}_{key}'] = value
            print(frame, name, result['status'], 'violation mm', result['constraint_violation_max_m']*1000, flush=True)
        tick = perf_counter()
        fit = minimize(problem.objective, np.zeros(38), jac=problem.gradient, method='SLSQP',
            constraints=[{'type': 'ineq', 'fun': problem.constraints, 'jac': problem.constraint_jacobian}],
            options={'maxiter': 60, 'ftol': 1e-6, 'disp': False})
        candidate = retract(prior, problem.L@fit.x)
        result, data = problem.report(candidate)
        result.update(solver={'success': bool(fit.success), 'status': int(fit.status), 'message': str(fit.message),
            'iterations': int(fit.nit), 'nfev': int(fit.nfev), 'njev': int(fit.njev)}, runtime_s=perf_counter()-tick)
        result['gates'] = gates(result, baseline, fit.success)
        result['passed'] = all(result['gates'].values())
        frame_report['constrained_original_map'] = result
        for key, value in [('transl', candidate.translation_m), ('root_rotation', candidate.root_rotation),
                ('vposer_latent', candidate.latent), ('body_rotations', data['body_rotations'][0]), ('coco_m', data['coco_m'][0]),
                ('whitened_delta', fit.x), ('delta', problem.L@fit.x)]:
            archive[f'{frame}_constrained_original_map_{key}'] = value
        summary[str(frame)] = frame_report
        save(DEST / 'summary.json', summary)
        np.savez_compressed(DEST / 'candidates.npz', **archive)
        print(frame, 'SLSQP', fit.success, fit.message, 'violation mm', result['constraint_violation_max_m']*1000,
              'pixel gates', result['gates'], flush=True)
    summary['runtime_s'] = perf_counter()-started
    summary['constrained_all_frames_pass'] = all(summary[str(frame)]['constrained_original_map']['passed'] for frame in FRAMES)
    summary['decision'] = 'Consider62-frame constrained validation' if summary['constrained_all_frames_pass'] else 'Stop this constrained candidate; do not run62 or sweep weights/budgets'
    save(DEST / 'summary.json', summary)
    print(summary['decision'], 'runtime_s', summary['runtime_s'], flush=True)


if __name__ == '__main__':
    run()
