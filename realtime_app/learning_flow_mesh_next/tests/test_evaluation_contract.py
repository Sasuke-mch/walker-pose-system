from dataclasses import replace

import pytest
import torch

from ..acceptance import PairedEvidence, preregistered_gates
from ..evaluate import paired_bootstrap
from ..evaluation_contract import DistributionScale, standardized_distributions


def scales(multiplier=1., split='engineering_fixture'):
    x = torch.tensor([-1., 1.], dtype=torch.float64)[:, None, None].expand(2, 17, 3)*multiplier
    b = torch.tensor([-1., 1.], dtype=torch.float64)[:, None].expand(2, 10)
    return DistributionScale.fit(x, b, source='known_reference_fixture', source_kind='analytic_fixture', split=split, basis='fixture_same_basis')


def score(multiplier=1., *, unknown_beta=False, invalid=False):
    x = torch.tensor([-1., 1.], dtype=torch.float64)[:, None, None].expand(2, 17, 3).clone()*multiplier
    b = torch.zeros(2, 10, dtype=torch.float64)
    if invalid:
        x[1] = torch.nan
    mask = torch.tensor([True, not invalid])
    return standardized_distributions(x, b, torch.ones(17, 3, dtype=torch.float64)*multiplier,
        torch.zeros(10, dtype=torch.float64), mask, mask, scales(multiplier), basis='fixture_same_basis',
        beta_reference_available=not unknown_beta)


def records(proposed=1.):
    return tuple({'subject': s, 'sequence': 'sequence', 'block': block, 'training_seed': seed,
                  'baseline': 2., 'proposed': proposed} for s in ('a', 'b') for block in (0, 1) for seed in (7, 17, 27))


def evidence(metric='standardized_joint_energy', proposed=1.):
    return PairedEvidence(records(proposed), metric, 'registered_fixture_stratum', 'engineering_fixture',
                          'analytic_fixture', 'known_fixture_records', 'G0', 'G1', 'same_K32_declared_fixture_budget')


def test_unit_invariance_separate_scores_and_original_aliases():
    a, b = score(), score(1000.)
    assert a['joints']['raw']['energy_score'] == pytest.approx(51**.5/2)
    assert a['joints']['raw'] == b['joints']['raw']
    assert a['beta']['raw']['energy_score'] == 0
    joints = torch.tensor([-1., 1.])[:, None, None].expand(2, 17, 3).clone()
    beta = torch.zeros(2, 10)
    stats = DistributionScale.fit(joints, beta, source='fixture', source_kind='analytic_fixture', split='engineering_fixture', basis='same')
    saved = stats.joint_mean.clone()
    joints += 10
    assert torch.equal(stats.joint_mean, saved)


def test_missing_and_illegal_pools_never_silently_repaired():
    report = score(unknown_beta=True, invalid=True)
    assert report['joints']['raw']['energy_score'] is None
    assert report['joints']['numerically_legal']['count'] == 1
    assert report['original_candidate_count'] == 2 and report['illegal_count'] == 1
    assert report['beta']['status'] == 'reference_unavailable'
    with pytest.raises(ValueError, match='basis'):
        standardized_distributions(torch.zeros(2, 17, 3), torch.zeros(2, 10), torch.zeros(17, 3), torch.zeros(10),
            torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool), scales(), basis='other')


@pytest.mark.parametrize('split', ['validation', 'test', 'train'])
def test_fixture_cannot_be_training_or_heldout_statistics(split):
    with pytest.raises(ValueError, match='training-only'):
        scales(split=split)


def test_real_reference_statistics_train_only_and_unknown_rejected():
    good = replace(scales(), split='train', source_kind='fitted_reference')
    good.validate()
    with pytest.raises(ValueError, match='training-only'):
        replace(good, split='test').validate()
    with pytest.raises(ValueError, match='unknown reference'):
        DistributionScale.fit(torch.full((2, 17, 3), torch.nan), torch.zeros(2, 10),
            source='unknown', source_kind='fitted_reference', split='train', basis='same')


def test_three_seed_gate_rejects_matching_one_seed():
    one_seed = tuple(r for r in records() if r['training_seed'] == 7)
    assert paired_bootstrap(one_seed, iterations=20)['training_seeds'] == [7]
    with pytest.raises(ValueError, match='preregistered training seeds'):
        replace(evidence(), records=one_seed).summarize(iterations=20)
    with pytest.raises(ValueError, match='at least two subjects'):
        replace(evidence(), records=tuple(r for r in records() if r['subject'] == 'a')).summarize(iterations=20)


def test_bool_cannot_certify_ci_and_fixture_cannot_certify_efficacy():
    empty = preregistered_gates({'energy_paired_ci_supports_improvement': True,
                                'aggregation_paired_ci_supports_improvement': True,
                                'ambiguity_energy_relative_improvement': .5})
    assert empty['energy_paired_ci_supports_improvement'] == 'not_evaluated'
    assert empty['ambiguity_energy_relative_improvement'] == 'not_evaluated'
    good = preregistered_gates({}, paired_evidence=(evidence(), evidence('selected_error')))
    assert good['energy_paired_ci_supports_improvement'] == 'pass'
    assert good['aggregation_paired_ci_supports_improvement'] == 'pass'
    assert good['mechanism_efficacy_claim'] == 'blocked_analytic_fixture'
    assert good['accuracy_claim'] == 'blocked'
    bad = preregistered_gates({}, paired_evidence=(evidence('selected_error', proposed=3.),))
    assert bad['aggregation_paired_ci_supports_improvement'] == 'fail'
    assert bad['aggregation_relative_improvement'] == 'fail'


def test_ci_crossing_zero_zero_baseline_and_no_cherry_picked_seeds():
    mixed = tuple({**r, 'proposed': 1. if r['subject'] == 'a' else 3.} for r in records())
    report = replace(evidence(), records=mixed).summarize(iterations=200)
    assert report['ci95'][0] < 0 < report['ci95'][1]
    assert not report['ci_supports_improvement']
    with pytest.raises(ValueError, match='positive baseline'):
        replace(evidence(), records=tuple({**r, 'baseline': 0.} for r in records())).summarize(iterations=20)
    with pytest.raises(ValueError, match='seed coverage'):
        replace(evidence(), records=records()[:-1]).summarize(iterations=20)
