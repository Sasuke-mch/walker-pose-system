"""Static lower-body chart comparison after a rejected contact prefix.

This is a diagnostic, not a new tracker. Both charts freeze the same root,
translation and15 other body joints, and omit both temporal and latent priors.
All previously withheld toe frames remain withheld. No weight/offset search.
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

APP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(APP), str(APP / 'tools')]
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
    contact_residuals,
    support_labels,
)
from probe_smplh_soft_contact import DEST as CONTACT

LOWER = np.array([0, 1, 3, 4, 6, 7])
FROZEN = np.setdiff1d(np.arange(21), LOWER)
FRAMES = [35, 40, 47, 55, 61]
DEST = OUT / 'contact_expressibility_complete'


def local_bodies(base, increments):
    """Right-retract only six joints; inputs/outputs are complete rotations."""
    increments = np.asarray(increments).reshape(-1, 6, 3)
    rotations = Rotation.from_rotvec(increments.reshape(-1, 3)).as_matrix().reshape(-1, 6, 3, 3)
    bodies = np.broadcast_to(base, (len(increments), 21, 3, 3)).copy()
    bodies[:, LOWER] = base[LOWER] @ rotations
    return bodies


def radial_huber_residuals(residual, threshold=25., sigma=8.):
    """Encode vector-norm Huber energy once, leaving contact energy separate."""
    norm = np.linalg.norm(residual, axis=-1)
    factor = np.ones_like(norm)
    outside = norm > threshold
    factor[outside] = np.sqrt((2*threshold*norm[outside]-threshold**2) / norm[outside]**2)
    return residual * factor[..., None] / sigma


class StaticProblem:
    def __init__(self, model, tracker, reference, frame, body, toes, rotation, translation):
        self.model, self.tracker, self.reference, self.frame = model, tracker, reference, frame
        self.base = reference['body_rotations'][frame]
        self.body, self.toes = body, toes
        self.rotation, self.translation = rotation, translation
        self.valid = tracker._supported_pixels(body)[0]
        padded = np.full((2, 12, 3), np.nan)
        padded[:, :4] = toes
        toe_valid = tracker._supported_pixels(padded)[0][:, :4].reshape(2, 2, 2).all(axis=-1)
        raw = toes[..., :2].reshape(2, 2, 2, 2)
        self.toe_vector = raw[..., 1, :] - raw[..., 0, :]
        self.toe_length = np.linalg.norm(self.toe_vector, axis=-1)
        self.direction_valid = toe_valid & (self.toe_length >= 16.) & (frame not in HELDOUT)
        self.support = support_labels()[frame]

    def bodies(self, values, chart):
        if chart == 'so3':
            return local_bodies(self.base, values)
        torch = self.model.torch
        with torch.inference_mode():
            z = torch.as_tensor(values + self.reference['vposer_latent'][self.frame], dtype=torch.float32, device=self.model.device)
            bodies = self.model.vposer.decode(z)['pose_body_matrot'].reshape(-1, 21, 3, 3).cpu().numpy().copy()
        bodies[:, FROZEN] = self.base[FROZEN]
        return bodies

    def forward(self, values, chart):
        bodies = self.bodies(values, chart)
        model, torch = self.model, self.model.torch
        count = len(bodies)
        def tensor(value):
            return torch.as_tensor(value, dtype=torch.float32, device=model.device)
        with torch.inference_mode():
            result = model.model(betas=model.betas.expand(count, -1),
                transl=tensor(np.broadcast_to(self.reference['transl'][self.frame], (count, 3)).copy()),
                global_orient=tensor(np.broadcast_to(self.reference['root_rotation'][self.frame], (count, 3, 3)).copy()),
                body_pose=tensor(bodies), left_hand_pose=model.hand_rotations[0].expand(count, -1, -1, -1),
                right_hand_pose=model.hand_rotations[1].expand(count, -1, -1, -1))
            coco = torch.einsum('jv,bvc->bjc', model.regressor, result.vertices).cpu().numpy().astype(float)
            vertices = result.vertices.cpu().numpy()
        return {'coco_m': coco, 'vertices_m': vertices, 'body_rotations': bodies,
                'body_pixels': project(coco[:, 5:17], self.tracker.cal),
                'toe_pixels': project(vertices[:, TOE_VERTICES], self.tracker.cal)}

    def components(self, data):
        body = radial_huber_residuals(data['body_pixels'] - self.body[None, ..., :2])[:, self.valid].reshape(len(data['coco_m']), -1)
        toe = data['toe_pixels'].reshape(-1, 2, 2, 2, 2)
        vector = toe[..., 1, :] - toe[..., 0, :]
        unit = vector / np.linalg.norm(vector, axis=-1)[..., None]
        direction = radial_huber_residuals(unit*self.toe_length[None, ..., None]-self.toe_vector[None])
        direction = direction[:, self.direction_valid].reshape(len(body), -1)
        contact = contact_residuals(data['vertices_m'][:, SURFACE], self.rotation, self.translation, self.support)
        return {'body': body, 'direction': direction, 'contact': contact}

    def residuals(self, values, chart, contact):
        terms = self.components(self.forward(np.atleast_2d(values), chart))
        keys = ['body', 'direction', 'contact'] if contact else ['body', 'direction']
        return np.concatenate([terms[key] for key in keys], axis=1)

    def objective_residual(self, values, chart, contact):
        return self.residuals(values, chart, contact)[0]

    def jacobian(self, values, chart, contact):
        eps, size = 1e-3, len(values)
        offsets = np.r_[values[None]+eps*np.eye(size), values[None]-eps*np.eye(size)]
        residual = self.residuals(offsets, chart, contact)
        return ((residual[:size]-residual[size:])/(2*eps)).T

    def report(self, values, chart):
        data = self.forward(np.atleast_2d(values), chart)
        parts = self.components(data)
        world = data['vertices_m'][0, SURFACE] @ self.rotation.T + self.translation
        heights = world[:, 2]
        sole = [float(heights[s:s+len(ids)].min()*1000) for s, ids in [(0, SOLE[0]), (len(SOLE[0]), SOLE[1])]]
        ball = [float(heights[ids].min()*1000) for ids in BALL]
        rotations = self.base.transpose(0, 2, 1) @ data['body_rotations'][0]
        angles = np.rad2deg(Rotation.from_matrix(rotations[LOWER]).magnitude())
        errors = np.linalg.norm(data['body_pixels'][0]-self.body[..., :2], axis=-1)
        report = {'costs': {key: float(.5*np.sum(value**2)) for key, value in parts.items()},
            'sole_min_mm': sole, 'ball_min_mm': ball,
            'support_ball_abs_mm': float(np.max(abs(np.array(ball)[self.support]))),
            'support_sole_min_mm': float(np.max(np.array(sole)[self.support])),
            'direction_errors_deg_view_foot': direction_error(data['toe_pixels'], self.toes[..., :2]).tolist(),
            'local_change_deg_hipL_hipR_kneeL_kneeR_ankleL_ankleR': angles.tolist(),
            'frozen_body_max_rotation_matrix_difference': float(abs(data['body_rotations'][0, FROZEN]-self.base[FROZEN]).max())}
        for name, ids in [('hip', [6, 7]), ('nonhip', [0, 1, 2, 3, 4, 5, 8, 9, 10, 11])]:
            selected = errors[:, ids][self.valid[:, ids]]
            report[name] = {'median_px': float(np.median(selected)), 'p95_px': float(np.percentile(selected, 95))}
        return report, data


def local_chart_spectrum(problem):
    """Test whether decoded lower rotations already have18 local directions."""
    size, eps = 32, 1e-3
    values = np.r_[eps*np.eye(size), -eps*np.eye(size)]
    bodies = problem.bodies(values, 'latent')
    relative = problem.base[LOWER].transpose(0, 2, 1) @ bodies[:, LOWER]
    coords = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(2*size, -1)
    jacobian = ((coords[:size]-coords[size:])/(2*eps)).T
    singular = np.linalg.svd(jacobian, compute_uv=False)
    return {'singular_values': singular.tolist(), 'rank_at_relative_tolerance1e_4': int((singular > singular[0]*1e-4).sum()),
            'condition_number': float(singular[0]/singular[-1])}


def reference_geometry(model, rotation, translation, labels):
    result = {}
    for name, path in [('direction', OUT / 'toe_direction_prefix'), ('previous_contact', CONTACT)]:
        arrays = np.load(path / 'result.npz')
        vertices = np.concatenate([model.evaluate_batch(arrays['transl'][s:s+16], arrays['root_rotation'][s:s+16], arrays['vposer_latent'][s:s+16], True)['vertices_m'] for s in range(0, 62, 16)])
        heights = (np.einsum('nij,nkj->nki', rotation, vertices[:, SURFACE]) + translation[:, None])[..., 2]
        result[name] = {}
        for foot, side in enumerate(['left', 'right']):
            ids = np.flatnonzero(labels[:, foot])
            start = foot*len(SOLE[0])
            sole = heights[ids, start:start+len(SOLE[foot])].min(axis=1)
            ball = heights[ids][:, BALL[foot]].min(axis=1)
            result[name][side] = {'support_sole_median_mm': float(np.median(sole)*1000),
                'ball_minus_sole_median_mm': float(np.median(ball-sole)*1000)}
    return result


def run():
    proto, model, tracker, body, toes, _ = load(62)
    original = model.regressor.clone()
    a, b = proto['hip_calibration_result']['alpha'], proto['hip_calibration_result']['thigh_fraction']
    model.regressor[11:13] = original[11:13]+a*(original[11:13]-original[[12, 11]])+b*(original[13:15]-original[11:13])
    reference = np.load(OUT / 'toe_direction_prefix/result.npz')
    scene = np.load(REPAIR / 'view/scene/scene_transforms.npz')
    rotation, translation = scene['rotation_ground_from_left'][:62], scene['translation_ground_from_left_mm'][:62]/1000
    DEST.mkdir(exist_ok=False)
    protocol = {'code': str(Path(__file__).relative_to(APP.parent)), 'command': [sys.executable, str(Path(__file__).resolve())],
        'reference': str(OUT / 'toe_direction_prefix/result.npz'), 'contact_protocol': str(CONTACT / 'protocol.json'), 'output_directory': str(DEST),
        'frames': FRAMES, 'lower_body_indices': LOWER.tolist(), 'frozen': 'root,translation,other15 body rotations,beta,hands,regressor,ground,raw pixels,gates,support labels,contact scales',
        'factorial_controls': 'lower chart latent32 vs local SO3_18; each data-only and data+contact from identical saved pose',
        'static_objective': 'same radial25px Huber/sigma8 body and eligible toe-direction; original contact residuals unchanged; no temporal or VPoser prior in EITHER chart',
        'toe_withheld': HELDOUT, 'fit_direction_only_frames': [40, 55],
        'solver': {'method': 'scipy least_squares trf', 'maximum_evaluations': 60, 'finite_difference_step': .001, 'gtol': .05, 'ftol': 1e-4, 'xtol': 1e-4, 'x_scale': 'jac', 'bounds': 'none, static mathematical diagnostic'},
        'candidate_gate': 'At every selected frame contact halves support-ball gap, raw hip/nonhip median/P95 <=1.05 saved reference, sole >=-5mm, direction median<=max(3deg,baseline*1.1) and max<=baseline+2deg, gradient optimality<=.05; no claiming feasible from solver success alone.',
        'stop': 'One static factorial study, no parameter grid, no trajectory substitution. A failed or nonphysical chart is not promoted; a local success alone cannot establish temporal tracking.',
        'limits': 'Removing priors makes this a permissive geometry/solver diagnostic. SO3 knees can exploit implausible axes; all joint changes retained. Rank/one optimizer cannot prove global VPoser feasibility. Ground and contact hypotheses remain non-independent.'}
    save(DEST / 'protocol.json', protocol)
    started, report, archive = perf_counter(), {}, {}
    report['reference_geometry'] = reference_geometry(model, rotation, translation, support_labels())
    for frame in FRAMES:
        problem = StaticProblem(model, tracker, reference, frame, body[frame], toes[frame], rotation[frame], translation[frame])
        base, data = problem.report(np.zeros(18), 'so3')
        difference = float(abs(data['coco_m'][0]-reference['predicted_coco'][frame]).max())
        if difference > 2e-6:
            raise ValueError(f'Static forward does not reproduce saved pose: {difference}')
        results = {'baseline': base, 'baseline_coco_max_difference_m': difference,
                   'latent_lower_chart_spectrum': local_chart_spectrum(problem), 'direction_fit_slots': int(problem.direction_valid.sum())}
        for chart, size in [('latent', 32), ('so3', 18)]:
            for contact in [False, True]:
                name = chart + ('_contact' if contact else '_data_only')
                fit = least_squares(partial(problem.objective_residual, chart=chart, contact=contact), np.zeros(size),
                    jac=partial(problem.jacobian, chart=chart, contact=contact), method='trf', x_scale='jac', max_nfev=60,
                    gtol=.05, ftol=1e-4, xtol=1e-4)
                values, data = problem.report(fit.x, chart)
                direction = np.array(values['direction_errors_deg_view_foot'])
                base_direction = np.array(base['direction_errors_deg_view_foot'])
                gates = {f'{part}_{stat}': values[part][stat] <= 1.05*base[part][stat] for part in ['hip', 'nonhip'] for stat in ['median_px', 'p95_px']}
                gates.update(support_halved=values['support_ball_abs_mm'] <= .5*base['support_ball_abs_mm'],
                    no_penetration=min(values['sole_min_mm']) >= -5., direction_median_retained=float(np.median(direction)) <= max(3., float(np.median(base_direction))*1.1),
                    direction_max_retained=float(direction.max()) <= float(base_direction.max())+2., gradient_converged=bool(fit.optimality <= .05))
                values.update(solver={'success': bool(fit.success), 'message': fit.message, 'nfev': fit.nfev,
                    'njev': fit.njev, 'optimality': float(fit.optimality), 'objective': float(fit.cost)}, gates=gates,
                    passed=bool(contact and all(gates.values())))
                results[name] = values
                archive[f'{frame}_{name}_body_rotations'] = data['body_rotations'][0]
                archive[f'{frame}_{name}_coco_m'] = data['coco_m'][0]
                archive[f'{frame}_{name}_parameters'] = fit.x
                print(frame, name, 'support mm', round(values['support_ball_abs_mm'], 2), 'max direction', round(float(direction.max()), 2), 'optimality', round(float(fit.optimality), 3), flush=True)
        report[str(frame)] = results
        np.savez_compressed(DEST / 'static_poses.npz', **archive)
        save(DEST / 'summary.json', report)
    report['runtime_s'] = perf_counter()-started
    report['all_frames_pass'] = {chart: all(report[str(f)][chart+'_contact']['passed'] for f in FRAMES) for chart in ['latent', 'so3']}
    save(DEST / 'summary.json', report)
    print('all frames pass', report['all_frames_pass'], 'runtime', report['runtime_s'], flush=True)


if __name__ == '__main__':
    run()
