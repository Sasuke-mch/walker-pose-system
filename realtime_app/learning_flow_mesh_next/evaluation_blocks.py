"""Original frame reports to registered blocks, with no failure deletion."""
import copy
import math
from dataclasses import dataclass

from .acceptance import PairedEvidence
from .evaluation_population import registry_records, validate_paired_population


@dataclass(frozen=True)
class FrameScores:
    record_id: str
    frame: int
    training_seed: int
    baseline: dict
    proposed: dict


@dataclass
class BlockExport:
    report: dict
    evidence: PairedEvidence | None


def archive(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'unavailable_numeric': repr(value)}
    if isinstance(value, dict):
        return {key: archive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [archive(item) for item in value]
    return copy.deepcopy(value)


def frame_value(report, metric, statistics):
    if metric == 'standardized_joint_energy':
        if statistics is None or report.get('statistics_source') != statistics.source:
            raise ValueError('frame_statistics_source_mismatch')
        value = report.get('joints', {}).get('raw', {}).get('energy_score')
    elif metric == 'selected_error':
        value = report.get('absolute_mpjpe_m_selected')
    else:
        raise ValueError('unregistered_metric')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('frame_metric_unavailable_or_nonfinite')
    return float(value)


def export_paired_blocks(registry, metric, frame_scores, *, statistics=None):
    indexed = registry_records(registry)
    comparison = registry.get('comparisons', {}).get(metric)
    if comparison is None:
        raise ValueError('registered comparison required')
    split = 'engineering_replay' if comparison['split'] == 'engineering_fixture' else comparison['split']
    blocks = [(key, record, block) for key, record in indexed.items() if record['split'] == split
              for block in record['blocks'] if block['stratum'] == comparison['stratum']]
    expected = {(key, frame, seed) for key, _, block in blocks for frame in block['frames'] for seed in (7, 17, 27)}
    by_frame, errors = {}, []
    original = []
    for row in frame_scores:
        if not isinstance(row, FrameScores):
            raise TypeError('typed original frame reports required')
        original.append(archive(vars(row)))
        key = (row.record_id, row.frame, row.training_seed)
        if type(row.record_id) is not str or type(row.frame) is not int or type(row.training_seed) is not int or key not in expected:
            errors.append({'key': archive(list(key)), 'reason': 'unregistered_original_frame_or_seed'})
        elif key in by_frame:
            errors.append({'key': list(key), 'reason': 'duplicate_original_frame'})
        else:
            by_frame[key] = row
    if not blocks:
        errors.append({'reason': 'no_registered_stratum_blocks'})
    rows, failures = [], []
    for key, record, block in blocks:
        for seed in (7, 17, 27):
            values = {'baseline': [], 'proposed': []}
            reasons = []
            for frame in block['frames']:
                item = by_frame.get((key, frame, seed))
                if item is None:
                    reasons.append({'frame': frame, 'reason': 'missing_original_frame'})
                    continue
                for method, method_values in values.items():
                    try:
                        method_values.append(frame_value(getattr(item, method), metric, statistics))
                    except (ValueError, AttributeError, TypeError) as error:
                        reasons.append({'frame': frame, 'method': method, 'reason': str(error)})
            row = {name: record[name] for name in ('subject', 'original_action', 'sequence', 'rig', 'source_path')}
            row.update(record_id=key, block=block['id'], frames=list(block['frames']), training_seed=seed,
                       statistics_source=statistics.source if statistics is not None else None)
            for method, method_values in values.items():
                row[method] = sum(method_values)/len(method_values) if not reasons and method_values else None
            row['failure_reasons'] = reasons
            rows.append(row)
            if reasons:
                failures.append({'record_id': key, 'block': block['id'], 'training_seed': seed, 'reasons': reasons})
    report = {'status': 'blocked_failed_or_missing_frames' if errors or failures else 'ready_for_numeric_pairing',
              'original_frame_reports': original, 'original_report_count': len(original), 'expected_frame_report_count': len(expected),
              'blocks': rows, 'failed_blocks': failures, 'input_errors': errors,
              'scope': 'registered per-frame report means; no imputation/deletion or actual efficacy certificate'}
    evidence = None
    if not errors and not failures:
        evidence = PairedEvidence(tuple(copy.deepcopy(rows)), metric, comparison['stratum'], comparison['split'],
            registry['source_kind'], comparison['source'], comparison['baseline_method'], comparison['proposed_method'],
            comparison['budget_scope'], copy.deepcopy(registry), statistics)
        validate_paired_population(evidence)
    return BlockExport(report, evidence)
