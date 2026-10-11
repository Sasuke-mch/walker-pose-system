"""Pre-test validation locks and evidence-aware preregistered decision gates."""
from collections import defaultdict
from dataclasses import asdict, dataclass

import torch

from .evaluate import paired_bootstrap
from .information_risk import RiskPolicy


@dataclass(frozen=True)
class PairedEvidence:
    records: tuple
    metric: str
    stratum: str
    split: str
    source_kind: str
    source: str
    baseline_method: str
    proposed_method: str
    budget_scope: str

    def summarize(self, *, iterations=2000):
        if self.metric not in ('standardized_joint_energy', 'selected_error') or not all(
                (self.stratum, self.source, self.baseline_method, self.proposed_method, self.budget_scope)):
            raise ValueError('paired metric, stratum, source, methods and budget scope required')
        if (self.source_kind, self.split) not in (('analytic_fixture', 'engineering_fixture'),
                ('fitted_reference', 'validation'), ('fitted_reference', 'test'),
                ('independent_reference', 'validation'), ('independent_reference', 'test')):
            raise ValueError('declared held-out or explicit fixture evidence required')
        if self.baseline_method == self.proposed_method:
            raise ValueError('distinct matched methods required')
        report = paired_bootstrap(self.records, iterations=iterations, required_seeds=(7, 17, 27))
        if report['subjects'] < 2:
            raise ValueError('at least two subjects for a subject bootstrap; not a power guarantee')
        grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for record in self.records:
            if record['baseline'] < 0 or record['proposed'] < 0:
                raise ValueError('nonnegative score/error required')
            grouped[record['subject']][record['sequence']][record['block']].append(record['baseline'])
        subject_means = []
        for sequences in grouped.values():
            sequence_means = []
            for blocks in sequences.values():
                sequence_means.append(sum(sum(v)/len(v) for v in blocks.values())/len(blocks))
            subject_means.append(sum(sequence_means)/len(sequence_means))
        baseline = sum(subject_means)/len(subject_means)
        if baseline <= 0:
            raise ValueError('positive baseline required for relative improvement')
        return {**report, 'relative_improvement': -report['proposed_minus_baseline']/baseline,
                'ci_supports_improvement': report['ci95'][1] < 0,
                'metric': self.metric, 'stratum': self.stratum, 'source': self.source,
                'source_kind': self.source_kind, 'split': self.split,
                'baseline_method': self.baseline_method, 'proposed_method': self.proposed_method,
                'budget_scope': self.budget_scope,
                'scope_limit': 'declared evidence only; source independence, test lock and budgets require external audit'}


def lock_policy(validation_risks, *, split, lambda0, lambda_g, iqr_fraction, source):
    if split != "validation" or iqr_fraction not in (0, .01, .05) or lambda_g not in (0, .1, 1, 10):
        raise ValueError("policy selection must use the preregistered validation grid")
    values = torch.as_tensor(validation_risks, dtype=torch.float64)
    if not len(values) or not torch.isfinite(values).all() or not source:
        raise ValueError("finite validation risks and calibration/validation source required")
    iqr = torch.quantile(values, .75)-torch.quantile(values, .25)
    policy = RiskPolicy(lambda0=lambda0, lambda_g=lambda_g, delta=float(iqr)*iqr_fraction,
                        source=source, ambiguity_calibrated=False).validate()
    return {"policy": asdict(policy), "validation_risk_iqr": float(iqr), "iqr_fraction": iqr_fraction,
            "locked_before_test": True, "test_used": False}


def preregistered_gates(metrics, *, paired_evidence=()):
    metrics = dict(metrics)
    paired = {}
    for evidence in paired_evidence:
        if not isinstance(evidence, PairedEvidence) or evidence.metric in paired:
            raise ValueError('typed distinct paired-evidence metrics required')
        paired[evidence.metric] = evidence.summarize()
    for metric, improvement, ci_key in (
            ('standardized_joint_energy', 'ambiguity_energy_relative_improvement', 'energy_paired_ci_supports_improvement'),
            ('selected_error', 'aggregation_relative_improvement', 'aggregation_paired_ci_supports_improvement')):
        report = paired.get(metric)
        metrics[improvement] = report['relative_improvement'] if report else None
        metrics[ci_key] = report['ci_supports_improvement'] if report else None
    requirements = {"ambiguity_energy_relative_improvement": (lambda x: x >= .05),
                    "energy_paired_ci_supports_improvement": bool,
                    "strong_geometry_relative_degradation": (lambda x: x <= .02),
                    "aggregation_relative_improvement": (lambda x: x >= .05),
                    "aggregation_paired_ci_supports_improvement": bool,
                    "retention_difference_absolute": (lambda x: x <= .01),
                    "illegal_rate_increase": (lambda x: x <= .005),
                    "coverage90": (lambda x: .87 <= x <= .93),
                    "full_pipeline_p95_ms": (lambda x: x <= 33.3)}
    result = {}
    for key, check in requirements.items():
        value = metrics.get(key)
        result[key] = "not_evaluated" if value is None else ("pass" if check(value) else "fail")
    if result["retention_difference_absolute"] == "fail":
        result["aggregation_comparison"] = "coverage_matching_required_before_claim"
    result["accuracy_claim"] = "blocked" if not metrics.get("independent_3d_truth") else "requires_all_relevant_gates"
    result["shape_claim"] = "blocked" if not metrics.get("independent_scan_or_bounded_dimensions") else "requires_shape_metrics"
    result["mechanism_revisions_remaining"] = max(0, 2-int(metrics.get("justified_performance_revisions", 0)))
    result["direction"] = "retain_FM; no mechanism efficacy claimed without preregistered evidence"
    result['paired_evidence'] = paired
    result['mechanism_efficacy_claim'] = 'not_certified_by_numeric_gates'
    if any(r['source_kind'] == 'analytic_fixture' for r in paired.values()):
        result['mechanism_efficacy_claim'] = 'blocked_analytic_fixture'
    return result
