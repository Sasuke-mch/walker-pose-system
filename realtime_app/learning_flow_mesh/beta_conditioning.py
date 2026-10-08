"""H8 explicit beta-conditioned velocity input; base generator remains unchanged."""
from dataclasses import dataclass, replace

import torch
from torch import nn

from .shape_features import ShapeFeatureMeshCandidate, ShapeFeatureState


class BetaConditionedFlow(nn.Module):
    """Read-only wrapper around original transport; adds 10*width parameters.

    Unknown beta gives zero residual while the published beta stays unavailable.
    Shape readiness is distinct from current observation availability: a known
    shared beta can condition internal missing-frame diagnostics, never publish
    them. This is feature conditioning, not a hierarchical SO(3) flow.
    """

    def __init__(self, base, *, enabled=True):
        super().__init__()
        if type(enabled) is not bool:
            raise ValueError("conditioning enabled must be boolean")
        self.base = base
        self.projection = nn.Linear(10, base.cfg.width, bias=False)
        self.enabled = enabled

    @property
    def cfg(self):
        return self.base.cfg

    @property
    def encoder(self):
        return self.base.encoder

    @property
    def parents(self):
        return self.base.parents

    def beta_residual(self, condition, betas, ready):
        if condition.ndim != 4 or condition.shape[-2:] != (52, self.cfg.width):
            raise ValueError("condition must be B,T,52,width")
        if betas.shape != (*condition.shape[:2], 10):
            raise ValueError("beta must be explicit B,T,10 shared-shape readout")
        if ready.shape != condition.shape[:2] or ready.dtype != torch.bool:
            raise ValueError("shape readiness must be boolean B,T")
        if any(value.device != condition.device for value in (betas, ready, self.projection.weight)):
            raise ValueError("beta/readiness/projection device must match conditions")
        if condition.dtype != self.projection.weight.dtype or betas.dtype != condition.dtype:
            raise ValueError("beta/projection dtype must match conditions")
        if not torch.isfinite(betas[ready]).all():
            raise ValueError("declared ready beta must be finite")
        clean = torch.where(ready[..., None], betas, 0)
        # Bias-free mapping keeps absent/zero beta exactly neutral, without fill.
        return self.projection(clean)[:, :, None]

    def transport(self, encoded, **kwargs):
        if not self.enabled:
            return self.base.transport(encoded, **kwargs)
        delta = self.beta_residual(encoded["condition"], encoded["betas"], encoded["shape_available"])
        return self.base.transport(dict(encoded, condition=encoded["condition"] + delta), **kwargs)


@dataclass(frozen=True)
class BetaConditionState(ShapeFeatureState):
    beta_conditioning_enabled: bool


class BetaConditionedMeshCandidate(ShapeFeatureMeshCandidate):
    def __init__(self, model, conditioner, **kwargs):
        if not isinstance(model, BetaConditionedFlow):
            raise TypeError("explicit BetaConditionedFlow wrapper required")
        super().__init__(model, conditioner, **kwargs)

    def step(self, batch, frame_ids, sequence_ids, admitted, state=None):
        if state is not None and (
                not isinstance(state, BetaConditionState)
                or state.beta_conditioning_enabled != self.model.enabled):
            raise ValueError("beta conditioning policy state changed; reset explicitly")
        output = super().step(batch, frame_ids, sequence_ids, admitted, state)
        s = output.state
        return replace(output, state=BetaConditionState(
            s.shape, s.hidden, s.initial_noise, s.noise_seed, s.shape_feature_source,
            s.missing_policy, self.model.enabled))
