"""Measurement-only causal memory, explicit elapsed time and observation ages."""
import torch
from torch import nn

from .schema import Observation, TemporalState


class StateEncoder(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        # xy2 + score1 + origin3 + direction3 + masks3 + age1 + seen1 + dt1
        self.local = nn.Sequential(nn.Linear(15, width), nn.GELU(), nn.Linear(width, width))
        self.joint_ids = nn.Parameter(torch.randn(17, width) * 0.02)
        self.view_roles = nn.Parameter(torch.randn(2, width) * 0.02)
        self.gru = nn.GRUCell(width, width)
        self.time_context = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))
        self.width = width

    def forward(self, obs: Observation, state: TemporalState | None = None):
        if not isinstance(obs, Observation):
            raise TypeError("inference accepts only typed Observation, never a training batch")
        obs.validate()
        if state is None:
            hidden = obs.xy.new_zeros(self.width)
            age = obs.xy.new_zeros(2, 17)
            seen = torch.zeros_like(obs.present)
            dt = 0.0
        else:
            if (state.identity, state.sequence, state.camera_ids) != (obs.identity, obs.sequence, obs.camera_ids):
                raise ValueError("incompatible stream; explicit reset required")
            if obs.timestamp <= state.last_timestamp or obs.frame <= state.last_frame:
                raise ValueError("physical timestamps and frame IDs must strictly increase")
            if state.hidden.shape != (self.width,) or state.age.shape != (2, 17) or state.seen.shape != (2, 17):
                raise ValueError("incompatible temporal-state dimensions")
            if not torch.isfinite(state.hidden).all() or not torch.isfinite(state.age).all() or (state.age < 0).any():
                raise ValueError("corrupt serialized temporal state")
            dt = obs.timestamp - state.last_timestamp
            hidden, age, seen = state.hidden, state.age + dt, state.seen
        valid = obs.admitted()
        geom = valid & obs.geometry
        age = torch.where(valid, 0, age)
        seen = seen | valid
        # Unknown ages are time since sequence start, accompanied by explicit seen=False.
        ages = torch.log1p(age).unsqueeze(-1)
        features = torch.cat((torch.where(valid[..., None], obs.xy / 1000.0, 0),
                              torch.where(valid, obs.confidence, 0)[..., None],
                              torch.where(geom[..., None], obs.origins, 0),
                              torch.where(geom[..., None], obs.directions, 0),
                              torch.stack((obs.present, obs.quality, geom), -1).to(obs.xy),
                              ages, seen[..., None].to(obs.xy),
                              obs.xy.new_full((2, 17, 1), dt)), -1)
        if valid.any():
            tokens = self.local(features) + self.joint_ids[None] + self.view_roles[:, None]
            # Missing/nonadmitted slots never update the GRU, even via pooling.
            measurement = tokens[valid].mean(0)
            hidden = self.gru(measurement, hidden)
        timing = obs.xy.new_tensor([dt, float(age.mean()), float(valid.float().mean()),
                                   float(seen.float().mean())])
        context = hidden + self.time_context(torch.log1p(timing))
        output = TemporalState(obs.identity, obs.sequence, obs.camera_ids, hidden, age,
                               obs.timestamp, obs.frame, seen)
        return context, output, bool(valid.any())
