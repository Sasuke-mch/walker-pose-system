"""Registered original population checks; metadata is not independent truth."""
import copy
from dataclasses import replace

from .data_firewall import validate_manifest
from .evaluation_contract import DistributionScale


def registry_records(registry):
    manifest = registry.get('manifest', {})
    validate_manifest(manifest)
    kind = registry.get('source_kind')
    if kind not in ('analytic_fixture', 'fitted_reference', 'independent_reference'):
        raise ValueError('explicit registry evidence level required')
    if kind != 'analytic_fixture' and manifest['license_status'] != 'locally_authorized':
        raise ValueError('registered real data requires declared local authorization')
    indexed = {}
    for record in manifest['records']:
        key = record.get('record_id')
        frames = record.get('frames', [])
        if not key or key in indexed or not frames or any(type(f) is not int or f < 0 for f in frames) or frames != sorted(set(frames)):
            raise ValueError('unique record ID and ordered original integer frames required')
        covered = []
        ids = set()
        for block in record.get('blocks', []):
            block_frames = block.get('frames', [])
            if block.get('id') is None or block.get('id') in ids or not block.get('stratum') or not block_frames:
                raise ValueError('unique block IDs, original frames and registered stratum required')
            ids.add(block.get('id'))
            start = len(covered)
            if block_frames != frames[start:start+len(block_frames)]:
                raise ValueError('nonoverlapping ordered original block partition required')
            covered.extend(block_frames)
        if covered != frames:
            raise ValueError('blocks must retain every original registered frame')
        indexed[key] = record
    if not indexed:
        raise ValueError('no registered population records')
    return indexed


def training_rows(registry, rows):
    indexed = registry_records(registry)
    seen = set()
    for row in rows:
        key = (row.get('record_id'), row.get('frame'))
        record = indexed.get(key[0])
        if record is None or record['split'] != 'train' or type(key[1]) is not int or key[1] not in record['frames'] or key in seen:
            raise ValueError('statistics require unique registered original training rows')
        seen.add(key)
    if not seen:
        raise ValueError('statistics population required')
    return indexed


def fit_registered_statistics(joints, beta, *, registry, rows, source):
    training_rows(registry, rows)
    if len(rows) != len(joints):
        raise ValueError('statistics values must align with original population rows')
    kind = registry['source_kind']
    stats = DistributionScale.fit(joints, beta, source=source, source_kind=kind,
        split='engineering_fixture' if kind == 'analytic_fixture' else 'train',
        basis=registry['manifest']['beta_basis'])
    return replace(stats, population=copy.deepcopy(list(rows)))


def validate_paired_population(evidence):
    if not evidence.registry:
        if evidence.source_kind == 'analytic_fixture':
            return {'status': 'explicit_unbound_fixture'}
        raise ValueError('registered population required for nonfixture evidence')
    registry = evidence.registry
    indexed = registry_records(registry)
    if evidence.source_kind != registry['source_kind']:
        raise ValueError('evidence level differs from registered source')
    comparison = registry.get('comparisons', {}).get(evidence.metric)
    declared = {'stratum': evidence.stratum, 'baseline_method': evidence.baseline_method,
                'proposed_method': evidence.proposed_method, 'budget_scope': evidence.budget_scope,
                'source': evidence.source, 'split': evidence.split}
    if comparison != declared:
        raise ValueError('comparison stratum/method/budget/source/split differs from registration')
    split = 'engineering_replay' if evidence.split == 'engineering_fixture' else evidence.split
    expected = {}
    for key, record in indexed.items():
        if record['split'] == split:
            for block in record['blocks']:
                if block['stratum'] == evidence.stratum:
                    expected[(key, block['id'])] = (record, block)
    seen = set()
    for row in evidence.records:
        key = (row.get('record_id'), row.get('block'))
        unique = (*key, row.get('training_seed'))
        if key not in expected or unique in seen:
            raise ValueError('unregistered or duplicate paired block')
        record, block = expected[key]
        for field in ('subject', 'original_action', 'sequence', 'rig', 'source_path'):
            if row.get(field) != record[field]:
                raise ValueError(f'paired original population mismatch:{field}')
        if not isinstance(row.get('frames'), list) or any(type(f) is not int for f in row['frames']) or row['frames'] != block['frames']:
            raise ValueError('paired block frames differ from registration')
        seen.add(unique)
    required = {(key, block, seed) for key, block in expected for seed in (7, 17, 27)}
    if not required or seen != required:
        raise ValueError('all registered stratum blocks and three seeds required')
    if evidence.metric == 'standardized_joint_energy':
        stats = evidence.statistics
        if not isinstance(stats, DistributionScale):
            raise ValueError('typed original training statistics required for energy evidence')
        stats.validate()
        training_rows(registry, stats.population)
        if stats.basis != registry['manifest']['beta_basis'] or stats.source_kind != evidence.source_kind:
            raise ValueError('statistics basis/evidence level differs from registration')
        if any(row.get('statistics_source') != stats.source for row in evidence.records):
            raise ValueError('energy records do not reference bound training statistics')
    return {'status': 'registered_metadata_verified', 'original_blocks': len(expected),
            'training_seeds': [7, 17, 27], 'claim_limit': 'declared original population only; file content, compute, independence and external locks not certified'}
