from dataclasses import replace

import torch

from ..aggregate import decide
from ..geometry_information import Information
from ..information_risk import (
    DecisionState,
    RiskPolicy,
    decide_information,
    information_metric,
)
from ..tangent import TangentScale, pairwise_tangent
from .test_aggregate import store_fixture
from .test_firewall import observation


def information(store, matrix=None):
    scale = TangentScale.fixture(store.topology)
    return Information(torch.zeros(169, 169, dtype=torch.float64) if matrix is None else matrix,
                       None, "external_anchor", store.rotations[0], scale.values,
                       "analytic_fixture_only", False, [])


def test_information_bounded_metric_and_pairwise_antisymmetry():
    store = store_fixture()
    info = information(store, torch.diag(torch.arange(169, dtype=torch.float64)))
    B = information_metric(info, RiskPolicy())
    expected = 1/(torch.arange(169, dtype=torch.float64)+1)
    assert torch.allclose(B.diag(), expected)
    assert torch.linalg.eigvalsh(B).min() > 0 and B.diag().max() == 1
    delta = pairwise_tangent(store, store.rotations[0], TangentScale.fixture(store.topology))
    assert torch.allclose(delta, -delta.transpose(0, 1))


def test_risk_permutation_and_near_pi_fallback():
    store = store_fixture()
    scale = TangentScale.fixture(store.topology)
    decision, _ = decide_information(store, information(store), scale, RiskPolicy(), observation=observation())
    order = torch.tensor([4, 3, 0, 2, 1])
    shuffled, _ = decide_information(store.permuted(order), information(store), scale, RiskPolicy(), observation=observation())
    assert shuffled.selected_id == decision.selected_id
    store.rotations[1, 0] = torch.diag(torch.tensor([1., -1., -1.], dtype=torch.float64))
    fallback, _ = decide_information(store, information(store), scale, RiskPolicy(), observation=observation())
    assert fallback.diagnostic["fallback"] and "near_pi" in fallback.reasons[0]


def test_time_never_beats_current_subset_and_gap_disables():
    store = store_fixture()
    scale = TangentScale.fixture(store.topology)
    info = information(store)
    previous = DecisionState("person", "sequence", 0., store.rotations[0], store.translation[0], store.beta[0], "old")
    spatial, _ = decide_information(store, info, scale, RiskPolicy(), observation=observation(1, .1))
    no_tie, _ = decide_information(store, info, scale, RiskPolicy(delta=0), method="A7", observation=observation(1, .1), previous=previous)
    assert no_tie.selected_id == spatial.selected_id
    tie, _ = decide_information(store, info, scale, RiskPolicy(delta=2), method="A7", observation=observation(1, .1), previous=previous)
    assert tie.selected_id == "id0"
    assert tie.costs[tie.selected_index] <= tie.costs.min()+2
    long_gap, _ = decide_information(store, info, scale, RiskPolicy(delta=2), method="A7", observation=observation(2, .6), previous=previous)
    assert long_gap.selected_id == spatial.selected_id
    assert "gap_over_0.5s" in long_gap.reasons[0]
    missing = replace(observation(2, .2), present=torch.zeros(2, 17, dtype=torch.bool))
    unavailable, retained = decide_information(store, info, scale, RiskPolicy(), method="A7", observation=missing, previous=previous)
    assert unavailable.selected_id is None and retained is previous


def test_first_candidate_control_retains_first_failure():
    store = store_fixture()
    store.actions[0] = False
    assert decide(store, "A0", TangentScale.fixture(store.topology)).selected_id is None


def test_invalid_information_and_coordinate_mismatch_fallback():
    store = store_fixture()
    scale = TangentScale.fixture(store.topology)
    for info in (information(store, torch.full((169, 169), torch.nan, dtype=torch.float64)),
                 replace(information(store), scale=scale.values*2),
                 replace(information(store), noise_source="proxy-only", provenance={"frame": 999})):
        decision, _ = decide_information(store, info, scale, RiskPolicy(), observation=observation())
        assert decision.diagnostic["fallback"] and decision.ambiguity
