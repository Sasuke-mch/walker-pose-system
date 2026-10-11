"""Train-only independent-reference residual statistics, not calibrated truth.

Manual label uncertainty remains part of residual dispersion. This module never
promotes sample covariance, a fixture, or a count threshold to calibrated=True.
"""
from dataclasses import dataclass
from itertools import pairwise

import torch

from .geometry_observation import NoiseModel


@dataclass(frozen=True)
class ReferencePixels:
    detected: torch.Tensor
    reference: torch.Tensor
    admitted: torch.Tensor
    reference_valid: torch.Tensor
    subjects: tuple
    original_actions: tuple
    sequences: tuple
    frames: tuple
    timestamps: tuple
    camera_ids: tuple
    reasons: tuple
    split: str
    detector_source: str
    reference_source: str
    reference_kind: str
    reference_evidence: str
    reference_uncertainty: str
    timestamp_kind: str = 'undeclared'
    pixel_domain: str = 'raw_fisheye'
    joint_operator: str = 'COCO17_manual_pixel'
    roles: tuple = ('LEFT', 'RIGHT')

    def validate(self):
        n = len(self.frames)
        if self.detected.shape != (n, 2, 17, 2) or self.reference.shape != self.detected.shape:
            raise ValueError('explicit N/V2/COCO17/xy paired pixels required')
        if n == 0 or self.split not in ('train', 'validation', 'test', 'fixture'):
            raise ValueError('nonempty explicit calibration split required')
        for mask in (self.admitted, self.reference_valid):
            if mask.shape != (n, 2, 17) or mask.dtype != torch.bool:
                raise ValueError('original admitted and independent reference masks required')
        if self.pixel_domain != 'raw_fisheye' or self.joint_operator != 'COCO17_manual_pixel':
            raise ValueError('raw-fisheye COCO reference definition required')
        if self.timestamp_kind not in ('exposure_time', 'host_read_return', 'analytic_fixture_clock'):
            raise ValueError('explicit physical/fixture clock provenance in seconds required')
        if len(self.camera_ids) != 2 or len(set(self.camera_ids)) != 2 or self.roles != ('LEFT', 'RIGHT'):
            raise ValueError('distinct registered PnP cameras in LEFT/RIGHT order required')
        if not all(isinstance(v, str) and v for v in self.camera_ids):
            raise ValueError('nonempty camera registry identifiers required')
        if self.reference_kind not in ('independent_manual_raw_2d', 'analytic_known_raw_2d'):
            raise ValueError('independent reference required; fitted self-reprojection is forbidden')
        if self.reference_kind == 'analytic_known_raw_2d' and self.split != 'fixture':
            raise ValueError('analytic reference cannot be relabeled as real calibration')
        if (not all((self.detector_source, self.reference_source, self.reference_evidence,
                     self.reference_uncertainty)) or self.detector_source == self.reference_source):
            raise ValueError('distinct sources, independent-reference evidence and uncertainty required')
        for values in (self.subjects, self.original_actions, self.sequences, self.timestamps, self.reasons):
            if len(values) != n:
                raise ValueError('keep metadata for every original frame')
        if not all(v for values in (self.subjects, self.original_actions, self.sequences) for v in values):
            raise ValueError('original subject/action/sequence IDs required')
        for i in range(n):
            if (len(self.reasons[i]) != 2 or any(len(row) != 17 for row in self.reasons[i])
                    or int(self.frames[i]) != self.frames[i] or self.frames[i] < 0
                    or not torch.isfinite(torch.tensor(self.timestamps[i], dtype=torch.float64))):
                raise ValueError('all frame clocks, IDs and per-joint reasons required')
            for view in range(2):
                for joint in range(17):
                    if not (self.admitted[i, view, joint] and self.reference_valid[i, view, joint]) and not self.reasons[i][view][joint]:
                        raise ValueError('every rejected/reference-missing slot needs an explicit reason')
        if not torch.isfinite(self.detected[self.admitted]).all():
            raise ValueError('nonfinite admitted detector measurement')
        if not torch.isfinite(self.reference[self.reference_valid]).all():
            raise ValueError('nonfinite valid reference')
        groups = {}
        for i, key in enumerate(zip(self.subjects, self.original_actions, self.sequences)):
            groups.setdefault(key, []).append(i)
        for indices in groups.values():
            indices.sort(key=lambda i: self.frames[i])
            if any(self.frames[b] <= self.frames[a] or self.timestamps[b] <= self.timestamps[a]
                   for a, b in pairwise(indices)):
                raise ValueError('unique frames and increasing physical clocks within sequences required')
        return self

    def residuals(self):
        self.validate()
        valid = self.admitted & self.reference_valid
        # Unknown NaNs remain unknown. They are never zero-filled or selected.
        return self.detected-self.reference, valid


def require_disjoint(train, heldout):
    train.validate()
    heldout.validate()
    if train.split != 'train' or heldout.split not in ('validation', 'test'):
        raise ValueError('explicit real train/heldout split required')
    for name in ('subjects', 'original_actions', 'sequences'):
        if set(getattr(train, name)) & set(getattr(heldout, name)):
            raise ValueError(f'calibration group leakage:{name}')
    if (train.camera_ids, train.pixel_domain, train.joint_operator) != (
            heldout.camera_ids, heldout.pixel_domain, heldout.joint_operator):
        raise ValueError('calibration rig/operator transfer requires a separate protocol')


@dataclass(frozen=True)
class ResidualStatistics:
    bias: torch.Tensor
    covariance: torch.Tensor
    second_moment: torch.Tensor
    counts: torch.Tensor
    eligible: torch.Tensor
    reasons: tuple
    camera_ids: tuple
    source: str
    reference_uncertainty: str
    floor_px2: float
    population: dict

    def proxy_noise(self):
        if not self.eligible.all():
            raise ValueError('incomplete reference cells; no silent isotropic fallback')
        eigenvalues, vectors = torch.linalg.eigh(self.second_moment)
        # Second moment includes systematic bias; it is NOT zero-mean covariance.
        bounded = (vectors*eigenvalues.clamp_min(self.floor_px2)[..., None, :]) @ vectors.transpose(-1, -2)
        return NoiseModel(torch.linalg.inv(bounded),
                          f'proxy reference-residual second moment; eigenfloor={self.floor_px2}px2; '
                          f'reference uncertainty={self.reference_uncertainty}; {self.source}',
                          False, self.camera_ids).validate()


def fit_reference_statistics(data, *, minimum_count=30, floor_px2=.25):
    data.validate()
    if data.split not in ('train', 'fixture'):
        raise ValueError('calibration statistics fit only real train or explicit fixture')
    if minimum_count < 2 or floor_px2 <= 0 or not torch.isfinite(torch.tensor(floor_px2)):
        raise ValueError('explicit count threshold>=2 and positive finite numerical floor required')
    residual, valid = data.residuals()
    residual = residual.detach().double()
    bias = residual.new_full((2, 17, 2), torch.nan)
    covariance = residual.new_full((2, 17, 2, 2), torch.nan)
    second = covariance.clone()
    counts = valid.sum(0)
    eligible = counts >= minimum_count
    reasons = []
    for view in range(2):
        row = []
        for joint in range(17):
            values = residual[:, view, joint][valid[:, view, joint]]
            n = len(values)
            if n:
                mean = values.mean(0)
                bias[view, joint] = mean
                second[view, joint] = values.T @ values/n
                if n > 1:
                    centered = values-mean
                    covariance[view, joint] = centered.T @ centered/(n-1)
            row.append('' if eligible[view, joint] else f'insufficient_reference_count:{n}')
        reasons.append(tuple(row))
    return ResidualStatistics(bias, covariance, second, counts, eligible, tuple(reasons), data.camera_ids,
                              f'{data.split}:{data.reference_kind}:{data.detector_source}:{data.reference_source}',
                              data.reference_uncertainty, floor_px2,
                              {'split': data.split, 'reference_kind': data.reference_kind,
                               'subjects': tuple(sorted(set(data.subjects))),
                               'original_actions': tuple(sorted(set(data.original_actions))),
                               'sequences': tuple(sorted(set(data.sequences))),
                               'pixel_domain': data.pixel_domain, 'joint_operator': data.joint_operator})


def temporal_residual_pairs(data, *, max_gap_s=.5):
    """Within-sequence adjacent-pair Pearson diagnostic, not uniform-grid ACF.

    NIST's usual ACF assumes equal spacing. Here actual dt and missing endpoints
    remain explicit; no AR(1) effective sample count or independence certificate.
    """
    if max_gap_s <= 0 or not torch.isfinite(torch.tensor(max_gap_s)):
        raise ValueError('finite positive physical gap bound required')
    residual, valid = data.residuals()
    residual = residual.double()
    groups = {}
    for i, key in enumerate(zip(data.subjects, data.original_actions, data.sequences)):
        groups.setdefault(key, []).append(i)
    products = residual.new_zeros(2, 17, 2)
    left_squares, right_squares = products.clone(), products.clone()
    counts = torch.zeros(2, 17, dtype=torch.long, device=residual.device)
    dts = []
    for indices in groups.values():
        indices.sort(key=lambda i: data.frames[i])
        group_valid = valid[indices]
        # Sequence centering is offline diagnostics, never a causal correction.
        mean = torch.where(group_valid[..., None], residual[indices], 0).sum(0)
        mean /= group_valid.sum(0).clamp_min(1)[..., None]
        for a, b in pairwise(indices):
            dt = data.timestamps[b]-data.timestamps[a]
            if data.frames[b]-data.frames[a] != 1 or dt > max_gap_s:
                continue
            pair = valid[a] & valid[b]
            if not pair.any():
                continue
            dts.append(float(dt))
            left = torch.where(pair[..., None], residual[a]-mean, 0)
            right = torch.where(pair[..., None], residual[b]-mean, 0)
            products += left*right
            left_squares += left.square()
            right_squares += right.square()
            counts += pair
    denominator = (left_squares*right_squares).sqrt()
    correlation = torch.where((counts >= 2)[..., None] & (denominator > 0),
                              products/denominator.clamp_min(1e-30), torch.nan)
    return {'pair_counts': counts, 'pair_correlation_xy': correlation,
            'eligible_frame_pairs': len(dts), 'dt_range_s': [min(dts), max(dts)] if dts else None,
            'boundary': 'sequence-centered pair diagnostic; irregular dt retained, not NIST ACF or independent sample count'}
