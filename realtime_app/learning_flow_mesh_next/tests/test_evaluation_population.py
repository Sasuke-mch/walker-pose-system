import copy
from dataclasses import replace

import pytest
import torch

from ..acceptance import preregistered_gates
from ..evaluation_population import fit_registered_statistics, registry_records
from .test_evaluation_contract import evidence
from .test_firewall import manifest


def bound():
    m = manifest()
    m['beta_basis'] = 'fixture_same_basis'
    rows = []
    for key, split in (('train', 'train'), ('a', 'engineering_replay'), ('b', 'engineering_replay')):
        rows.append({'record_id': key, 'subject': key, 'original_action': key, 'sequence': key,
                     'rig': 'rig_fixture', 'source_path': f'known_fixture:{key}', 'split': split,
                     'frames': [0, 1], 'blocks': [{'id': n, 'frames': [n], 'stratum': 'preregistered_ambiguity'} for n in (0, 1)]})
    m['records'] = rows
    ev = evidence()
    registry = {'manifest': m, 'source_kind': 'analytic_fixture', 'comparisons': {
        ev.metric: {k: getattr(ev, k) for k in ('stratum', 'baseline_method', 'proposed_method', 'budget_scope', 'source', 'split')}}}
    population = [{'record_id': 'train', 'frame': n} for n in (0, 1)]
    x = torch.tensor([-1., 1.])[:, None, None].expand(2, 17, 3)
    b = torch.zeros(2, 10)
    stats = fit_registered_statistics(x, b, registry=registry, rows=population, source='registered_fixture_stats')
    registered = []
    for row in ev.records:
        source = next(r for r in rows if r['record_id'] == row['subject'])
        registered.append({**row, **{k: source[k] for k in ('original_action', 'sequence', 'rig', 'source_path')},
                           'record_id': source['record_id'], 'frames': [row['block']], 'statistics_source': stats.source})
    return replace(ev, registry=registry, records=tuple(registered), statistics=stats)


def test_registered_fixture_population_and_statistics_alias_safety():
    ev = bound()
    report = ev.summarize(iterations=20)
    assert report['population']['original_blocks'] == 4
    assert report['population']['status'] == 'registered_metadata_verified'
    assert report['source_kind'] == 'analytic_fixture'
    rows = [{'record_id': 'train', 'frame': n} for n in (0, 1)]
    stats = fit_registered_statistics(torch.zeros(2, 17, 3), torch.zeros(2, 10),
        registry=ev.registry, rows=rows, source='fixture')
    rows[0]['frame'] = 100
    assert stats.population[0]['frame'] == 0
    saved = stats.to_dict()
    saved['population'][0]['frame'] = 200
    assert stats.population[0]['frame'] == 0


def test_registered_strong_geometry_cannot_satisfy_ambiguity_gates():
    ev = bound()
    registry = copy.deepcopy(ev.registry)
    registry['comparisons'][ev.metric]['stratum'] = 'strong_geometry_only'
    for record in registry['manifest']['records']:
        for block in record['blocks']:
            block['stratum'] = 'strong_geometry_only'
    ev = replace(ev, stratum='strong_geometry_only', registry=registry)

    result = preregistered_gates({}, paired_evidence=(ev,))

    # Original rows, seeds and registration still agree; they answer another question.
    report = result['paired_evidence'][ev.metric]
    assert report['population']['status'] == 'registered_metadata_verified'
    assert report['relative_improvement'] == .5
    assert result['ambiguity_energy_relative_improvement'] == 'not_evaluated'
    assert result['energy_paired_ci_supports_improvement'] == 'not_evaluated'
    assert result['mechanism_efficacy_claim'] == 'blocked_analytic_fixture'


def test_nonfixture_unregistered_evidence_rejected():
    with pytest.raises(ValueError, match='registered population required'):
        replace(evidence(), source_kind='fitted_reference', split='validation').summarize(iterations=20)


@pytest.mark.parametrize('split, status', [('validation', 'not_evaluated'), ('test', 'pass')])
def test_declared_heldout_split_must_match_acceptance_contract(split, status):
    # Artificial rows exercise declarations only; this is not real held-out data.
    ev = bound()
    registry = copy.deepcopy(ev.registry)
    registry['source_kind'] = 'fitted_reference'
    registry['manifest']['license_status'] = 'locally_authorized'
    registry['comparisons'][ev.metric]['split'] = split
    for row in registry['manifest']['records']:
        if row['split'] == 'engineering_replay':
            row['split'] = split
    stats = replace(ev.statistics, source_kind='fitted_reference', split='train')
    ev = replace(ev, registry=registry, statistics=stats, source_kind='fitted_reference', split=split)
    result = preregistered_gates({}, paired_evidence=(ev,))
    assert result['ambiguity_energy_relative_improvement'] == status
    assert result['energy_paired_ci_supports_improvement'] == status
    assert result['mechanism_efficacy_claim'] == 'not_certified_by_numeric_gates'


@pytest.mark.parametrize('field', ['subject', 'original_action', 'sequence', 'rig', 'source_path', 'frames'])
def test_original_population_mismatch_not_accepted(field):
    ev = bound()
    rows = list(ev.records)
    rows[0] = {**rows[0], field: [9] if field == 'frames' else 'other'}
    with pytest.raises(ValueError, match='population mismatch|frames differ'):
        replace(ev, records=tuple(rows)).summarize(iterations=20)


@pytest.mark.parametrize('field', ['stratum', 'baseline_method', 'proposed_method', 'budget_scope', 'source'])
def test_comparison_registration_not_a_free_text_label(field):
    ev = bound()
    with pytest.raises(ValueError, match='differs from registration'):
        replace(ev, **{field: 'other'}).summarize(iterations=20)


def test_missing_whole_block_and_duplicate_cannot_change_population():
    ev = bound()
    with pytest.raises(ValueError, match='all registered'):
        replace(ev, records=tuple(r for r in ev.records if not (r['subject'] == 'a' and r['block'] == 0))).summarize(iterations=20)
    with pytest.raises(ValueError, match='duplicate paired block'):
        replace(ev, records=(*ev.records, ev.records[0])).summarize(iterations=20)


def test_cross_split_leakage_and_frame_partition_failures():
    ev = bound()
    registry = copy.deepcopy(ev.registry)
    registry['manifest']['records'][1]['subject'] = 'train'
    with pytest.raises(ValueError, match='leakage'):
        registry_records(registry)
    registry = copy.deepcopy(ev.registry)
    registry['manifest']['records'][1]['blocks'][1]['frames'] = [0]
    with pytest.raises(ValueError, match='partition'):
        registry_records(registry)


def test_heldout_statistics_and_wrong_origin_or_basis_rejected():
    ev = bound()
    wrong = replace(ev.statistics, population=[{'record_id': 'a', 'frame': 0}])
    with pytest.raises(ValueError, match='original training rows'):
        replace(ev, statistics=wrong).summarize(iterations=20)
    with pytest.raises(ValueError, match='basis/evidence level'):
        replace(ev, statistics=replace(ev.statistics, basis='other')).summarize(iterations=20)
    rows = [{**r, 'statistics_source': 'other'} for r in ev.records]
    with pytest.raises(ValueError, match='bound training statistics'):
        replace(ev, records=tuple(rows)).summarize(iterations=20)
    with pytest.raises(ValueError, match='values must align'):
        fit_registered_statistics(torch.zeros(2, 17, 3), torch.zeros(2, 10),
            registry=ev.registry, rows=[{'record_id': 'train', 'frame': 0}], source='bad')
    with pytest.raises(ValueError, match='original training rows'):
        fit_registered_statistics(torch.zeros(2, 17, 3), torch.zeros(2, 10),
            registry=ev.registry, rows=[{'record_id': 'train', 'frame': False},
                                      {'record_id': 'train', 'frame': 1}], source='bad')
