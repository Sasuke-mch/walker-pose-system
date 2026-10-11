"""Shared joint R/T/beta velocity graph for CFM, regression and Euler sampling."""
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .data_firewall import Normalization
from .schema import Topology
from .state_encoder import StateEncoder


@dataclass(frozen=True)
class ModelConfig:
    nodes: int = 52
    width: int = 128
    heads: int = 4
    blocks: int = 4

    def __post_init__(self):
        Topology(self.nodes)
        if self.width != 128 or self.heads != 4 or self.blocks != 4:
            raise ValueError("first preregistered core is width128/heads4/blocks4")

    def to_dict(self):
        return asdict(self)


class JointFlow(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        config = config or ModelConfig()
        self.config = config
        self.topology = Topology(config.nodes)
        n, w = config.nodes, config.width
        self.encoder = StateEncoder(w)
        self.pose_in, self.position_in, self.shape_in = nn.Linear(6, w), nn.Linear(3, w), nn.Linear(10, w)
        self.token_ids = nn.Parameter(torch.randn(n + 2, w) * 0.02)
        self.flow_time = nn.Sequential(nn.Linear(3, w), nn.GELU(), nn.Linear(w, w))
        self.info_pose = nn.Linear(9, w, bias=False)
        self.info_position = nn.Linear(9, w, bias=False)
        self.info_shape = nn.Linear(100, w, bias=False)
        self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(
            w, config.heads, dim_feedforward=w * 2, dropout=0, activation="gelu",
            batch_first=True, norm_first=True) for _ in range(config.blocks))
        self.pose_out, self.position_out, self.shape_out = nn.Linear(w, 6), nn.Linear(w, 3), nn.Linear(w, 10)

    def velocity(self, state, flow_time, context, information=None):
        batch, d = state.shape
        n = self.config.nodes
        if d != self.topology.dimensions or context.shape != (batch, self.config.width):
            raise ValueError("joint velocity state/context mismatch")
        if flow_time.shape != (batch,) or not torch.isfinite(flow_time).all() or ((flow_time < 0) | (flow_time > 1)).any():
            raise ValueError("one shared CFM time per joint sample")
        pose, position, shape = state[:, :6*n].reshape(batch, n, 6), state[:, 6*n:6*n+3], state[:, -10:]
        tokens = torch.cat((self.pose_in(pose), self.position_in(position)[:, None], self.shape_in(shape)[:, None]), 1)
        time = torch.stack((flow_time, (2*torch.pi*flow_time).sin(), (2*torch.pi*flow_time).cos()), -1)
        tokens = tokens + self.token_ids + context[:, None] + self.flow_time(time)[:, None]
        if information is not None:
            m = self.topology.tangent_dimensions
            if information.shape != (m, m) or not torch.isfinite(information).all():
                raise ValueError("invalid frozen anchor information")
            info = information.detach().to(state)
            # Signed log compression; still a proxy unless noise was calibrated.
            info = info.sign() * torch.log1p(info.abs())
            blocks = torch.stack([info[3*i:3*i+3, 3*i:3*i+3].flatten() for i in range(n)])
            residual = torch.cat((self.info_pose(blocks),
                                  self.info_position(info[3*n:3*n+3, 3*n:3*n+3].flatten())[None],
                                  self.info_shape(info[-10:, -10:].flatten())[None]), 0)
            tokens = tokens + residual[None]
        for block in self.blocks:
            tokens = block(tokens)
        return torch.cat((self.pose_out(tokens[:, :n]).flatten(1),
                          self.position_out(tokens[:, n]), self.shape_out(tokens[:, n+1])), -1)

    def regression(self, context):
        # Same registered core and exact parameter count; only the objective changes.
        return self.velocity(context.new_zeros(len(context), self.topology.dimensions),
                             context.new_ones(len(context)), context)

    def sample(self, context, initial, *, steps=8, information=None):
        if not isinstance(steps, int) or steps < 1 or not torch.isfinite(initial).all():
            raise ValueError("positive Euler steps and finite initial noise required")
        state = initial
        expanded = context.expand(len(initial), -1)
        for step in range(steps):
            state = state + self.velocity(state, state.new_full((len(state),), step / steps),
                                          expanded, information) / steps
        return state


def cfm_loss(model, context, labels, normalization: Normalization, *, generator=None, information=None):
    if labels.topology != model.topology or normalization.topology != model.topology:
        raise ValueError("training topology mismatch")
    target, _ = labels.packed(require_complete=True)
    normalization.validate()
    x = normalization.normalize(target)
    epsilon = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    time = torch.rand(len(x), generator=generator, device=x.device, dtype=x.dtype)
    z = (1-time[:, None])*epsilon + time[:, None]*x
    velocity = model.velocity(z, time, context, information)
    squared = (velocity - (x-epsilon)).square()
    n = model.config.nodes * 6
    parts = {"pose": squared[:, :n].mean(), "translation": squared[:, n:n+3].mean(),
             "shape": squared[:, -10:].mean()}
    return sum(parts.values()) / 3, parts


def regression_loss(model, context, labels, normalization):
    target, _ = labels.packed(require_complete=True)
    if labels.topology != model.topology or normalization.topology != model.topology:
        raise ValueError("regression topology mismatch")
    error = (model.regression(context) - normalization.normalize(target)).square()
    n = model.config.nodes*6
    return (error[:, :n].mean() + error[:, n:n+3].mean() + error[:, -10:].mean())/3
