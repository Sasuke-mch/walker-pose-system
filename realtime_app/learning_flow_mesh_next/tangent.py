"""Shortest SO(3) distances and anchor-transported standardized tangent vectors."""
from dataclasses import dataclass

import torch


def angles(relative):
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1)
    return torch.acos(cosine)


def shortest_log(relative, *, pi_margin=1e-3):
    angle = angles(relative)
    if (angle > torch.pi-pi_margin).any():
        raise ValueError("near_pi_rotation_chart")
    vector = torch.stack((relative[..., 2, 1]-relative[..., 1, 2],
                          relative[..., 0, 2]-relative[..., 2, 0],
                          relative[..., 1, 0]-relative[..., 0, 1]), -1)/2
    factor = torch.where(angle < 1e-4, 1 + angle.square()/6,
                         angle / torch.sin(angle).clamp_min(1e-8))
    return vector * factor[..., None]


def skew(vector):
    zero = torch.zeros_like(vector[..., 0])
    x, y, z = vector.unbind(-1)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*vector.shape[:-1], 3, 3)


@dataclass(frozen=True)
class TangentScale:
    values: torch.Tensor
    source: str

    def validate(self, topology):
        if self.values.shape != (topology.tangent_dimensions,) or not self.source:
            raise ValueError("tangent scales must declare radians/metres/beta basis and source")
        if not torch.isfinite(self.values).all() or (self.values <= 0).any():
            raise ValueError("positive finite tangent scales required")
        # General anisotropic SO3 metric must transport its axes as well.
        # First version uses isotropic scale within each local rotation block.
        pose = self.values[:topology.nodes*3].reshape(-1, 3)
        if not torch.allclose(pose, pose[:, :1].expand_as(pose)):
            raise ValueError("first version requires isotropic scale per rotation block")
        return self

    @classmethod
    def fixture(cls, topology):
        return cls(torch.ones(topology.tangent_dimensions, dtype=torch.float64), "analytic_fixture_only")


def complete_distances(store, scale):
    scale.validate(store.topology)
    n = store.topology.nodes
    relative = store.rotations[:, None].transpose(-1, -2) @ store.rotations[None]
    pose = (angles(relative).square() / scale.values[:3*n:3].to(store.raw).square()).sum(-1)
    displacement = torch.cat((store.translation[:, None]-store.translation[None],
                              store.beta[:, None]-store.beta[None]), -1)
    return pose + (displacement / scale.values[3*n:].to(store.raw)).square().sum(-1)


def pairwise_tangent(store, anchor, scale):
    """delta_kl uses shortest pair log transported by Ad(R_anchor^T R_k).

    Common chart log differences would not equal shortest pair rotation differences.
    """
    scale.validate(store.topology)
    relative = store.rotations[:, None].transpose(-1, -2) @ store.rotations[None]
    logs = shortest_log(relative)
    anchor_to_k = store.rotations[anchor].transpose(-1, -2) @ store.rotations
    # Also reject candidate-to-anchor chart ambiguity, independently of pair ambiguity.
    shortest_log(anchor_to_k)
    pose = torch.einsum("knij,klnj->klni", anchor_to_k, logs).flatten(2)
    position = store.translation[None] - store.translation[:, None]
    shape = store.beta[None] - store.beta[:, None]
    return torch.cat((pose, position, shape), -1) / scale.values.to(store.raw)
