from dataclasses import replace

import pytest
import torch

from ..noise_calibration import (
    ReferencePixels,
    fit_reference_statistics,
    require_disjoint,
    temporal_residual_pairs,
)
from .test_firewall import observation


def reference_fixture(n=64):
    ref = torch.full((n, 2, 17, 2), 200., dtype=torch.float64)
    error = torch.zeros(n, 2, 17, 2, dtype=torch.float64)
    # Exact known anisotropic, biased residuals; no stochastic accuracy claim.
    error[..., 0] = (torch.arange(n) % 2 * 2-1)[:, None, None] + 4
    error[..., 1] = (torch.arange(n) % 4 // 2 * 2-1)[:, None, None] * 2
    mask = torch.ones(n, 2, 17, dtype=torch.bool)
    return ReferencePixels(ref+error, ref, mask, mask.clone(), ('subject',)*n,
                           ('raw_action',)*n, ('sequence',)*n, tuple(range(n)),
                           tuple(i*.1 for i in range(n)), ('PnP_LEFT_fixture', 'PnP_RIGHT_fixture'),
                           tuple((('',)*17, ('',)*17) for _ in range(n)), 'fixture',
                           'analytic_detector', 'analytic_reference', 'analytic_known_raw_2d',
                           'constructed exact fixture pixels', 'zero_known_fixture_only',
                           timestamp_kind='analytic_fixture_clock')


def with_reference_mask(data, mask):
    reasons = tuple(tuple(tuple('reference_missing' if not mask[i, v, j] else data.reasons[i][v][j]
                                for j in range(17)) for v in range(2)) for i in range(len(data.frames)))
    return replace(data, reference_valid=mask, reasons=reasons)


def test_bias_covariance_second_moment_and_no_calibrated_promotion():
    data = reference_fixture()
    stats = fit_reference_statistics(data)
    error = (data.detected-data.reference)[:, 0, 0]
    assert torch.allclose(stats.covariance[0, 0], torch.cov(error.T), atol=1e-12)
    assert stats.bias[0, 0, 0] == 4
    assert stats.second_moment[0, 0, 0, 0] == 17
    assert stats.covariance[0, 0, 0, 0] < 1.1
    noise = stats.proxy_noise()
    assert not noise.calibrated and noise.camera_ids == data.camera_ids
    assert torch.allclose(torch.linalg.inv(noise.precision[0, 0]), stats.second_moment[0, 0], atol=1e-12)
    with pytest.raises(ValueError, match='rig mismatch'):
        noise.validate_for(replace(observation(), camera_ids=('other_L', 'other_R')))


def test_self_reference_and_heldout_fit_rejected():
    data = reference_fixture()
    with pytest.raises(ValueError, match='independent reference'):
        fit_reference_statistics(replace(data, reference_kind='current_body_fit_reprojection'))
    with pytest.raises(ValueError, match='relabeled'):
        fit_reference_statistics(replace(data, split='train'))
    real = replace(data, reference_kind='independent_manual_raw_2d', split='validation')
    with pytest.raises(ValueError, match='fit only'):
        fit_reference_statistics(real)
    with pytest.raises(ValueError, match='fit only'):
        fit_reference_statistics(replace(real, split='test'))


def test_group_firewall_and_missing_cells_not_filled():
    data = reference_fixture()
    train = replace(data, reference_kind='independent_manual_raw_2d', split='train')
    val = replace(train, split='validation')
    with pytest.raises(ValueError, match='group leakage'):
        require_disjoint(train, val)
    n = len(data.frames)
    val = replace(val, subjects=('val_subject',)*n, original_actions=('val_action',)*n, sequences=('val_sequence',)*n)
    require_disjoint(train, val)
    valid, refs = data.reference_valid.clone(), data.reference.clone()
    valid[:, 0, 9] = False
    refs[:, 0, 9] = torch.nan
    unknown = replace(with_reference_mask(data, valid), reference=refs)
    stats = fit_reference_statistics(unknown)
    assert stats.counts[0, 9] == 0 and stats.reasons[0][9] == 'insufficient_reference_count:0'
    assert torch.isnan(unknown.reference[:, 0, 9]).all()
    with pytest.raises(ValueError, match='incomplete reference'):
        stats.proxy_noise()


def test_temporal_pairs_no_crossing_missing_frames_or_sequences():
    data = reference_fixture(8)
    report = temporal_residual_pairs(data)
    assert report['eligible_frame_pairs'] == 7
    assert report['pair_correlation_xy'][0, 0, 0] == -1
    missing_mask = data.reference_valid.clone()
    missing_mask[3] = False
    report = temporal_residual_pairs(with_reference_mask(data, missing_mask))
    assert report['eligible_frame_pairs'] == 5  # no bridging 2->4
    second = replace(data, sequences=('one',)*4+('two',)*4)
    assert temporal_residual_pairs(second)['eligible_frame_pairs'] == 6
    gap = replace(data, frames=(0, 1, 2, 4, 5, 6, 7, 8))
    assert temporal_residual_pairs(gap)['eligible_frame_pairs'] == 6
    clocks = replace(data, timestamps=(0., .1, .2, .3, 1., 1.1, 1.2, 1.3))
    assert temporal_residual_pairs(clocks)['eligible_frame_pairs'] == 6


def test_sequence_bias_is_not_temporal_random_correlation():
    data = reference_fixture(8)
    detected = data.reference.clone()
    detected[:4] += 5
    detected[4:] -= 5
    grouped = replace(data, detected=detected, sequences=('one',)*4+('two',)*4)
    assert torch.isnan(temporal_residual_pairs(grouped)['pair_correlation_xy']).all()


def test_correlated_errors_not_reclassified_as_independent_or_calibrated():
    data = reference_fixture(256)
    rng = torch.Generator().manual_seed(41)
    innovation = torch.randn(256, 2, generator=rng, dtype=torch.float64)
    errors = torch.zeros_like(innovation)
    for i in range(1, len(errors)):
        errors[i] = .95*errors[i-1] + innovation[i]
    correlated = replace(data, detected=data.reference+errors[:, None, None])
    report = temporal_residual_pairs(correlated)
    assert report['pair_counts'][0, 0] == 255
    assert report['pair_correlation_xy'][0, 0].min() > .85
    assert not fit_reference_statistics(correlated).proxy_noise().calibrated
    assert 'effective_sample_count' not in report
