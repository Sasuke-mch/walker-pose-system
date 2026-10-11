import copy
import json
from dataclasses import replace

import pytest
import torch

from ..evaluate import candidate_metrics
from ..evaluation_blocks import FrameScores, export_paired_blocks
from ..evaluation_contract import standardized_distributions
from .test_evaluation_population import bound


def original_fixture(metric='standardized_joint_energy'):
    ev = bound()
    registry = copy.deepcopy(ev.registry)
    if metric == 'selected_error':
        comparison = registry['comparisons'].pop(ev.metric)
        registry['comparisons'][metric] = comparison
    scores = []
    mask = torch.ones(2, dtype=torch.bool)
    target = torch.ones(17, 3)
    beta = torch.zeros(2, 10)
    for row in ev.records:
        baseline = torch.tensor([-1., 1.])[:, None, None].expand(2, 17, 3)
        proposed = torch.ones(2, 17, 3)
        if metric == 'standardized_joint_energy':
            def report(points):
                return standardized_distributions(points, beta, target, torch.zeros(10), mask, mask,
                                                  ev.statistics, basis=ev.statistics.basis)
        else:
            def report(points):
                return candidate_metrics(points, target, mask, mask, selected_index=0)
        scores.append(FrameScores(row['record_id'], row['frames'][0], row['training_seed'], report(baseline), report(proposed)))
    return registry, scores, ev.statistics


@pytest.mark.parametrize('metric', ['standardized_joint_energy', 'selected_error'])
def test_actual_evaluator_reports_export_and_input_order_invariance(metric):
    registry, scores, stats = original_fixture(metric)
    a = export_paired_blocks(registry, metric, scores, statistics=stats)
    b = export_paired_blocks(registry, metric, list(reversed(scores)), statistics=stats)
    assert a.report['status'] == 'ready_for_numeric_pairing'
    assert a.report['blocks'] == b.report['blocks']
    assert a.evidence.summarize(iterations=20)['relative_improvement'] == 1
    assert len(a.report['blocks']) == 12
    assert json.loads(json.dumps(a.report, allow_nan=False)) == a.report
    scores[0].baseline['modified'] = True
    assert 'modified' not in a.report['original_frame_reports'][0]['baseline']


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'wrong_frame', 'wrong_seed', 'bool_frame', 'nan_frame', 'unknown_metric', 'nan_metric', 'statistics_source'])
def test_failures_preserved_and_no_partial_evidence(fault):
    registry, scores, stats = original_fixture()
    if fault == 'missing':
        scores = scores[1:]
    elif fault == 'duplicate':
        scores.append(scores[0])
    elif fault in ('wrong_frame', 'bool_frame', 'nan_frame', 'wrong_seed'):
        scores[0] = replace(scores[0], **({'training_seed': 99} if fault == 'wrong_seed' else
                                         {'frame': False if fault == 'bool_frame' else float('nan') if fault == 'nan_frame' else 999}))
    else:
        report = copy.deepcopy(scores[0].baseline)
        if fault == 'statistics_source':
            report['statistics_source'] = 'other'
        else:
            report['joints']['raw']['energy_score'] = float('nan') if fault == 'nan_metric' else None
            # A legal-pool score may exist; it must not substitute for raw ES.
            report['joints']['numerically_legal']['energy_score'] = 0.
        scores[0] = replace(scores[0], baseline=report)
    output = export_paired_blocks(registry, 'standardized_joint_energy', scores, statistics=stats)
    assert output.evidence is None
    assert output.report['status'] == 'blocked_failed_or_missing_frames'
    assert len(output.report['blocks']) == 12
    assert output.report['original_report_count'] == len(scores)
    assert output.report['failed_blocks'] or output.report['input_errors']
    json.dumps(output.report, allow_nan=False)


def test_multi_frame_block_mean_never_drops_missing_endpoint():
    registry, scores, stats = original_fixture()
    # Merge two original one-frame blocks for each evaluation subject.
    for record in registry['manifest']['records'][1:]:
        record['blocks'] = [{'id': 0, 'frames': [0, 1], 'stratum': record['blocks'][0]['stratum']}]
    # Use known frame scores, keeping origins/statistics and all seeds.
    for i, score in enumerate(scores):
        report = copy.deepcopy(score.baseline)
        report['joints']['raw']['energy_score'] = 2. if score.frame == 0 else 4.
        scores[i] = replace(score, baseline=report)
    good = export_paired_blocks(registry, 'standardized_joint_energy', scores, statistics=stats)
    assert all(row['baseline'] == 3. for row in good.report['blocks'])
    scores = [r for r in scores if not (r.record_id == 'a' and r.frame == 1 and r.training_seed == 7)]
    bad = export_paired_blocks(registry, 'standardized_joint_energy', scores, statistics=stats)
    assert bad.evidence is None
    assert bad.report['blocks'][0]['baseline'] is None
    assert bad.report['blocks'][0]['frames'] == [0, 1]
    assert bad.report['failed_blocks'][0]['reasons'][0]['frame'] == 1
