"""Frozen residual evaluation: coverage AND width, no fitting or promotion."""
import math

import torch

from .noise_calibration import ResidualStatistics

LEVELS = (.5, .9, .95)


def frozen_population_gate(statistics, heldout, mode):
    if not isinstance(statistics, ResidualStatistics):
        raise TypeError('typed frozen residual statistics required')
    heldout.validate()
    split = statistics.population['split']
    if mode == 'fixture':
        if split != 'fixture' or heldout.split != 'fixture':
            raise ValueError('fixture evaluation cannot upgrade real or synthetic split roles')
    elif mode in ('validation', 'locked_test'):
        expected = 'validation' if mode == 'validation' else 'test'
        if split != 'train' or heldout.split != expected:
            raise ValueError('explicit frozen train/validation or locked_test roles required')
    else:
        raise ValueError('explicit evaluation role required')
    for field in ('subjects', 'original_actions', 'sequences'):
        if set(statistics.population[field]) & set(getattr(heldout, field)):
            raise ValueError(f'frozen calibration population leakage:{field}')
    if statistics.camera_ids != heldout.camera_ids or (
            statistics.population['pixel_domain'], statistics.population['joint_operator']) != (
                heldout.pixel_domain, heldout.joint_operator):
        raise ValueError('frozen noise rig/operator mismatch; explicit transfer protocol required')


def reference_metrics(errors, second_moment, floor):
    second_moment = second_moment.to(errors)
    eigenvalues, vectors = torch.linalg.eigh(second_moment)
    if not torch.isfinite(eigenvalues).all() or eigenvalues.min() < -1e-8:
        raise ValueError('invalid frozen residual second moment')
    covariance = (vectors*eigenvalues.clamp_min(floor)[None]) @ vectors.T
    precision = torch.linalg.inv(covariance)
    scores = torch.einsum('ni,ij,nj->n', errors, precision, errors)
    logdet = torch.linalg.slogdet(covariance).logabsdet
    area_factor = math.pi*float(torch.linalg.det(covariance).sqrt())
    quantiles = errors.new_tensor([.5, .9, .95, .99])
    return {
        'coverage': {str(p): float((scores <= -2*math.log1p(-p)).double().mean()) for p in LEVELS},
        'ellipse_area_px2': {str(p): area_factor*(-2*math.log1p(-p)) for p in LEVELS},
        'mean_bias_px': errors.mean(0).tolist(),
        'radial_quantile_px': torch.quantile(errors.norm(dim=-1), quantiles).tolist(),
        'quadratic_score_p99': float(torch.quantile(scores, .99)),
        'gaussian_reference_log_score_mean': float((scores+logdet+2*math.log(2*math.pi)).mean()/2),
        'reference_boundary': 'nominal zero-centered Gaussian ellipse/log score using a proxy second moment; not FM likelihood',
    }, scores


def evaluate_frozen_residuals(statistics, heldout, *, mode):
    frozen_population_gate(statistics, heldout, mode)
    errors, valid = heldout.residuals()
    errors = errors.detach().double()
    scores = errors.new_full(valid.shape, torch.nan)
    cells = []
    for view in range(2):
        for joint in range(17):
            mask = valid[:, view, joint]
            values = errors[:, view, joint][mask]
            row = {'view': view, 'joint': joint,
                   'admitted_count': int(heldout.admitted[:, view, joint].sum()),
                   'reference_usable_count': int(mask.sum()),
                   'reference_missing_count': int((heldout.admitted[:, view, joint] & ~heldout.reference_valid[:, view, joint]).sum()),
                   'training_count': int(statistics.counts[view, joint]),
                   'available': False, 'reasons': [], 'trained_proxy': None, 'isotropic10px_control': None}
            if len(values) == 0:
                row['reasons'].append('no_independent_reference_in_heldout_cell')
            else:
                row['isotropic10px_control'] = reference_metrics(values, errors.new_tensor([[100., 0.], [0., 100.]]), .25)[0]
                if not statistics.eligible[view, joint]:
                    row['reasons'].append(statistics.reasons[view][joint])
                else:
                    try:
                        metrics, quadratic = reference_metrics(values, statistics.second_moment[view, joint], statistics.floor_px2)
                        metrics['bias_change_from_training_px'] = (values.mean(0)-statistics.bias[view, joint]).tolist()
                        row['trained_proxy'], row['available'] = metrics, True
                        scores[:, view, joint][mask] = quadratic
                    except (ValueError, RuntimeError) as error:
                        row['reasons'].append(f'invalid_frozen_noise:{error}')
            cells.append(row)
    return {'mode': mode, 'frozen_source': statistics.source, 'heldout_detector_source': heldout.detector_source,
            'heldout_reference_source': heldout.reference_source, 'camera_ids': heldout.camera_ids,
            'nominal_levels': LEVELS, 'cells': cells, 'quadratic_scores': scores,
            'reference_valid': valid.detach().clone(), 'original_reasons': heldout.reasons,
            'frame_metadata': tuple(zip(heldout.subjects, heldout.original_actions, heldout.sequences,
                                        heldout.frames, heldout.timestamps)),
            'calibrated': False, 'statistics_updated': False,
            'boundary': 'independent-reference residual diagnostic; coverage without width/bias/tails or dependence audit is not calibration'}
