"""Observation-slot identity must bind to content before invariant pooling."""
from dataclasses import replace

import pytest
import torch

from ..state_encoder import StateEncoder
from .test_firewall import observation


def paired_observations(kind):
    base = observation()
    xy = torch.arange(68, dtype=torch.float64).reshape(2, 17, 2)*9
    base = replace(base, xy=xy, confidence=base.confidence.double(),
                   origins=base.origins.double(), directions=base.directions.double())
    order = torch.arange(17)
    order[0], order[16] = 16, 0
    swapped = xy[:, order] if kind == "joint" else xy.flip(0)
    return base, replace(base, xy=swapped)


@pytest.mark.parametrize("seed", [7, 19, 31])
@pytest.mark.parametrize("kind", ["joint", "view"])
def test_observation_identity_survives_pooling(seed, kind):
    torch.manual_seed(seed)
    model = StateEncoder().double().eval()
    first, swapped = paired_observations(kind)
    with torch.no_grad():
        a = model(first)[0]
        b = model(swapped)[0]
    assert (a-b).abs().max() > 1e-8


def test_old_additive_identity_has_permutation_collision():
    torch.manual_seed(7)
    model = StateEncoder().double()
    features = torch.randn(2, 17, 15, dtype=torch.float64)
    identity = model.joint_ids[None] + model.view_roles[:, None]
    original = (model.local(features)+identity).mean((0, 1))
    for shifted in (features.flip(1), features.flip(0)):
        collided = (model.local(shifted)+identity).mean((0, 1))
        assert torch.allclose(original, collided, atol=1e-14, rtol=0)
