"""Offline dimensionless distribution scores; labels never enter deployment."""
import copy
from dataclasses import dataclass, field

import torch

from .evaluate import distribution


@dataclass(frozen=True)
class DistributionScale:
    joint_mean: torch.Tensor
    joint_scale: torch.Tensor
    beta_mean: torch.Tensor
    beta_scale: torch.Tensor
    source: str
    source_kind: str
    split: str
    basis: str
    operator: str = 'COCO17_SURFACE_REGRESSOR_6890'
    floor: float = 1e-4
    population: list = field(default_factory=list)

    def to_dict(self):
        self.validate()
        return {key: value.detach().cpu().tolist() if torch.is_tensor(value) else copy.deepcopy(value)
                for key, value in vars(self).items()}

    def validate(self):
        for mean, scale, shape in ((self.joint_mean, self.joint_scale, (17, 3)),
                                   (self.beta_mean, self.beta_scale, (10,))):
            if mean.shape != shape or scale.shape != shape or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError('finite positive same-operator evaluation statistics required')
        if not self.source or not self.basis or self.operator != 'COCO17_SURFACE_REGRESSOR_6890':
            raise ValueError('explicit source, beta basis and surface operator required')
        if (self.source_kind, self.split) not in (('analytic_fixture', 'engineering_fixture'),
                                                ('fitted_reference', 'train'), ('independent_reference', 'train')):
            raise ValueError('training-only or explicitly named fixture statistics required')
        return self

    @classmethod
    def fit(cls, joints, beta, *, source, source_kind, split, basis, floor=1e-4):
        if joints.ndim != 3 or joints.shape[1:] != (17, 3) or beta.shape != (len(joints), 10) or len(joints) < 2:
            raise ValueError('complete N,17,3 own-T surface references and N,10 beta required')
        if not torch.isfinite(joints).all() or not torch.isfinite(beta).all():
            raise ValueError('unknown reference slots cannot silently enter statistics')
        if not 0 < floor < float('inf'):
            raise ValueError('explicit finite positive numerical floor required')
        return cls(joints.mean(0).detach().clone(), joints.std(0, unbiased=False).clamp_min(floor).detach().clone(),
                   beta.mean(0).detach().clone(), beta.std(0, unbiased=False).clamp_min(floor).detach().clone(),
                   source, source_kind, split, basis, floor=floor).validate()


def standardized_distributions(joints, beta, target_joints, target_beta, numerical, actions,
                               statistics, *, basis, joint_reference_available=True, beta_reference_available=True):
    """One original frame, K whole candidates; separate joint and shape ES."""
    statistics.validate()
    if basis != statistics.basis:
        raise ValueError('shape reference and candidates must share audited beta basis')
    k = len(joints)
    if joints.shape != (k, 17, 3) or beta.shape != (k, 10):
        raise ValueError('K,17,3 own-T surface points and K,10 beta required')
    if numerical.shape != (k,) or actions.shape != (k,) or numerical.dtype != torch.bool or actions.dtype != torch.bool or (actions & ~numerical).any():
        raise ValueError('explicit aligned numerical/action masks required')
    for value in (joints[numerical], beta[numerical]):
        if not torch.isfinite(value).all():
            raise ValueError('declared legal candidates must be finite')
    output = {'statistics_source': statistics.source, 'statistics_split': statistics.split,
              'source_kind': statistics.source_kind, 'basis': basis,
              'scope': 'separate dimensionless own-T COCO17 and beta10; no joint-plus-beta composite',
              'legal_retention': float(numerical.double().mean()) if k else None,
              'action_retention': float(actions.double().mean()) if k else None,
              'illegal_count': int((~numerical).sum()), 'original_candidate_count': k}
    for name, samples, target, mean, scale, available in (
        ('joints', joints, target_joints, statistics.joint_mean, statistics.joint_scale, joint_reference_available),
        ('beta', beta, target_beta, statistics.beta_mean, statistics.beta_scale, beta_reference_available)):
        if not available:
            output[name] = {'status': 'reference_unavailable', 'energy_score': None}
            continue
        if target.shape != mean.shape or not torch.isfinite(target).all():
            raise ValueError('available reference block must be complete and finite')
        standardized = ((samples-mean.to(samples))/scale.to(samples)).flatten(1)
        reference = ((target-mean.to(target))/scale.to(target)).flatten()
        output[name] = {pool: distribution(standardized[mask], reference) for pool, mask in
                        (('raw', torch.ones(k, dtype=torch.bool, device=joints.device)),
                         ('numerically_legal', numerical), ('action_filtered', actions))}
        output[name]['width_units'] = 'training_standard_deviations'
    return output
