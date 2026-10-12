"""Observation-slot identity must bind to content before invariant pooling."""
from dataclasses import replace

import pytest
import torch

from ..schema import TemporalState
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


@pytest.mark.parametrize("seed", [7, 17, 27])
@pytest.mark.parametrize("kind", ["incoming_age", "prior_seen"])
def test_admitted_measurement_preserves_incoming_freshness(seed, kind):
    """Matched carry states isolate a discarded feature, not task accuracy."""
    torch.manual_seed(seed)
    model = StateEncoder().double().eval()
    obs = observation(1, .2)
    obs = replace(obs, xy=obs.xy.double(), confidence=obs.confidence.double(),
                  origins=obs.origins.double(), directions=obs.directions.double())
    first = TemporalState(obs.identity, obs.sequence, obs.camera_ids,
                          torch.zeros(128, dtype=torch.float64), torch.zeros(2, 17, dtype=torch.float64),
                          0., 0, torch.ones(2, 17, dtype=torch.bool))
    second = replace(first, age=first.age.clone(), seen=first.seen.clone())
    if kind == "incoming_age":
        second.age[0, 0] = .8
    else:
        second.seen[0, 0] = False
    incoming = second.to_dict()
    with torch.no_grad():
        a, state_a, _ = model(obs, first)
        b, state_b, _ = model(obs, second)
    # Freshness now legitimately changes the GRU hidden. Reset age/seen and
    # stream metadata must still agree because both just observed every slot.
    assert {key: value for key, value in state_a.to_dict().items() if key != "hidden"} == {
        key: value for key, value in state_b.to_dict().items() if key != "hidden"}
    assert second.to_dict() == incoming  # No mutation of carried evidence.
    assert (a-b).abs().max() > 1e-8


def test_elapsed_age_and_first_seen_features_follow_observed_missing_reappearance():
    torch.manual_seed(7)
    model = StateEncoder().double().eval()
    recorded = []
    hook = model.local[0].register_forward_pre_hook(lambda _, args: recorded.append(args[0].detach().clone()))
    obs = observation()
    obs = replace(obs, xy=obs.xy.double(), confidence=obs.confidence.double(),
                  origins=obs.origins.double(), directions=obs.directions.double())
    with torch.no_grad():
        _, state, _ = model(obs)
        missing = replace(obs, frame=1, timestamp=.2, present=torch.zeros_like(obs.present),
                          xy=torch.full_like(obs.xy, torch.nan))
        _, absent, _ = model(missing, state)
        _, returned, _ = model(replace(obs, frame=2, timestamp=.9), absent)
    hook.remove()
    assert len(recorded) == 2  # No local-MLP call for all-missing observations.
    assert recorded[0][..., 12:14].count_nonzero() == 0  # Age0 with prior seen=False.
    assert torch.allclose(recorded[1][..., 12], torch.log1p(torch.full((2, 17), .9, dtype=torch.float64)))
    assert recorded[1][..., 13].eq(1).all()
    assert torch.equal(state.hidden, absent.hidden)
    assert absent.age.eq(.2).all() and returned.age.eq(0).all() and returned.seen.all()
