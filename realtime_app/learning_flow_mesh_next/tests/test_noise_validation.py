import math
from dataclasses import replace

import pytest
import torch
from scipy.stats import chi2

from ..noise_calibration import fit_reference_statistics
from ..noise_validation import LEVELS, evaluate_frozen_residuals
from .test_noise_calibration import reference_fixture, with_reference_mask


def gaussian_fixture(seed, *, population, n=2048, sigma=2.):
    data = reference_fixture(n)
    rng = torch.Generator().manual_seed(seed)
    error = torch.randn(data.detected.shape, generator=rng, dtype=torch.float64)*sigma
    return replace(data, detected=data.reference+error, subjects=(f'{population}_person',)*n,
                   original_actions=(f'{population}_action',)*n, sequences=(f'{population}_sequence',)*n,
                   detector_source=f'{population}_analytic_detector', reference_source=f'{population}_analytic_reference')


def test_frozen_coverage_width_shift_and_independent_quantiles():
    torch.set_num_threads(1)
    train, val = gaussian_fixture(21, population='train'), gaussian_fixture(23, population='holdout')
    statistics = fit_reference_statistics(train)
    before = [value.clone() for value in (statistics.bias, statistics.covariance, statistics.second_moment,
                                         statistics.counts, statistics.eligible)]
    report = evaluate_frozen_residuals(statistics, val, mode='fixture')
    cell = report['cells'][0]
    p, control = cell['trained_proxy'], cell['isotropic10px_control']
    assert torch.isfinite(report['quadratic_scores']).all()
    assert float((report['quadratic_scores'][:, 0, 0] <= -2*math.log(.1)).double().mean()) == p['coverage']['0.9']
    assert .87 < p['coverage']['0.9'] < .93
    assert control['coverage']['0.9'] == 1
    assert control['ellipse_area_px2']['0.9'] > p['ellipse_area_px2']['0.9']*20
    for level in LEVELS:
        assert abs(-2*math.log1p(-level)-float(chi2.ppf(level, 2))) < 1e-12
    shifted = replace(val, detected=val.detected+torch.tensor([20., 0.]))
    bad = evaluate_frozen_residuals(statistics, shifted, mode='fixture')['cells'][0]['trained_proxy']
    assert bad['coverage']['0.9'] < .01 and bad['mean_bias_px'][0] > 19
    wider = replace(val, detected=val.reference+(val.detected-val.reference)*2)
    degraded = evaluate_frozen_residuals(statistics, wider, mode='fixture')['cells'][0]['trained_proxy']
    assert degraded['coverage']['0.9'] < .5
    assert not report['calibrated'] and not report['statistics_updated']
    assert all(torch.equal(a, b) for a, b in zip(before, (statistics.bias, statistics.covariance,
               statistics.second_moment, statistics.counts, statistics.eligible)))


def test_original_unknown_and_insufficient_training_cells_retained():
    train, val = gaussian_fixture(21, population='train', n=64), gaussian_fixture(23, population='holdout', n=64)
    mask = train.reference_valid.clone()
    mask[:, 0, 9] = False
    statistics = fit_reference_statistics(with_reference_mask(train, mask))
    vmask = val.reference_valid.clone()
    vmask[:, 1, 4] = False
    pixels = val.reference.clone()
    pixels[:, 1, 4] = torch.nan
    report = evaluate_frozen_residuals(statistics, replace(with_reference_mask(val, vmask), reference=pixels), mode='fixture')
    unavailable_train = report['cells'][9]
    unavailable_ref = report['cells'][21]
    assert not unavailable_train['available'] and unavailable_train['isotropic10px_control'] is not None
    assert unavailable_train['reference_usable_count'] == 64
    assert not unavailable_ref['available'] and unavailable_ref['reference_missing_count'] == 64
    assert torch.isnan(report['quadratic_scores'][:, 0, 9]).all()
    assert torch.isnan(report['quadratic_scores'][:, 1, 4]).all()
    assert report['original_reasons'] == with_reference_mask(val, vmask).reasons


def test_population_roles_rig_and_test_lock_gate():
    train, val = gaussian_fixture(21, population='train', n=64), gaussian_fixture(23, population='holdout', n=64)
    statistics = fit_reference_statistics(train)
    with pytest.raises(ValueError, match='population leakage'):
        evaluate_frozen_residuals(statistics, train, mode='fixture')
    with pytest.raises(ValueError, match='rig/operator mismatch'):
        evaluate_frozen_residuals(statistics, replace(val, camera_ids=('new_L', 'new_R')), mode='fixture')
    with pytest.raises(ValueError, match='roles'):
        evaluate_frozen_residuals(statistics, val, mode='locked_test')
    # Declarative role fixtures exercise gate behavior, not independent human data.
    real_train = replace(train, reference_kind='independent_manual_raw_2d', split='train')
    real_val = replace(val, reference_kind='independent_manual_raw_2d', split='validation')
    real_statistics = fit_reference_statistics(real_train)
    evaluate_frozen_residuals(real_statistics, real_val, mode='validation')
    test = replace(real_val, split='test')
    with pytest.raises(ValueError, match='roles'):
        evaluate_frozen_residuals(real_statistics, test, mode='validation')
    assert evaluate_frozen_residuals(real_statistics, test, mode='locked_test')['mode'] == 'locked_test'
