from dataclasses import replace

import pytest
import torch

from ..candidate_store import infer
from ..data_firewall import Normalization
from ..information_encoding import fixed_probes, information_features
from ..joint_flow import JointFlow, ModelConfig, cfm_loss
from .test_firewall import labels, observation


def features(matrix, nodes, mode):
    return information_features(matrix, nodes, mode=mode,
                                pose_probes=fixed_probes(len(matrix), 3),
                                shape_probes=fixed_probes(len(matrix), 10))


def collision_matrices(nodes):
    m = 3*nodes+13
    independent = torch.eye(m, dtype=torch.float64)
    coupled = independent.clone()
    coupled[0, -10] = coupled[-10, 0] = 1
    opposite = independent.clone()
    opposite[0, -10] = opposite[-10, 0] = -1
    return independent, coupled, opposite


@pytest.mark.parametrize('nodes', [22, 52])
def test_known_diagonal_collision_and_signed_coupling_distinction(nodes):
    matrices = collision_matrices(nodes)
    old = [features(M, nodes, 'diagonal_blocks') for M in matrices]
    assert all(torch.equal(a, b) for pair in old[1:] for a, b in zip(old[0], pair))
    new = [features(M, nodes, 'cross_block_sketch') for M in matrices]
    assert torch.equal(new[0][0], old[0][0]) and torch.equal(new[0][1], old[0][1])
    assert all(not torch.equal(new[0][1], pair[1]) for pair in new[1:])
    assert not torch.equal(new[1][0], new[2][0])
    assert torch.linalg.eigvalsh(matrices[1]).min().abs() < 1e-12


def blind_coupling(nodes):
    m = 3*nodes+13
    # u perpendicular to all ten shape probes over pose coordinates, v
    # perpendicular to all three pose probes over shape coordinates.
    U, _, _ = torch.linalg.svd(fixed_probes(m, 10)[:3*nodes], full_matrices=True)
    V, _, _ = torch.linalg.svd(fixed_probes(m, 3)[-10:], full_matrices=True)
    u, v = U[:, -1], V[:, -1]
    C = torch.zeros(m, m, dtype=torch.float64)
    C[:3*nodes, -10:] = .5*u[:, None]*v[None]
    C[-10:, :3*nodes] = C[:3*nodes, -10:].T
    return torch.eye(m, dtype=torch.float64), C


def test_preserve_noninjective_psd_counterexample_instead_of_claiming_full_information():
    identity, C = blind_coupling(52)
    assert C.abs().max() > .01
    assert torch.linalg.eigvalsh(identity+C).min() > .49
    a, b = features(identity, 52, 'cross_block_sketch'), features(identity+C, 52, 'cross_block_sketch')
    assert all(torch.allclose(x, y, atol=1e-14) for x, y in zip(a, b))


def test_legacy_feature_values_and_nontrivial_block_diagonal_control():
    torch.manual_seed(9)
    A = torch.randn(169, 169, dtype=torch.float64)
    matrix = A.T @ A
    compressed = matrix.sign()*torch.log1p(matrix.abs())
    expected = torch.stack([compressed[3*i:3*i+3, 3*i:3*i+3].flatten() for i in range(53)])
    actual, shape = features(matrix, 52, 'diagonal_blocks')
    assert torch.equal(actual, expected) and torch.equal(shape, compressed[-10:, -10:].flatten())
    blocks = [matrix[3*i:3*i+3, 3*i:3*i+3] for i in range(53)] + [matrix[-10:, -10:]]
    diagonal_matrix = torch.block_diag(*blocks)
    old, new = features(diagonal_matrix, 52, 'diagonal_blocks'), features(diagonal_matrix, 52, 'cross_block_sketch')
    assert all(torch.equal(a, b) for a, b in zip(old, new))


def test_candidate_explicit_history_missing_frame_and_provenance():
    torch.manual_seed(7)
    model = JointFlow(ModelConfig(information_encoding='cross_block_sketch')).eval()
    normal = Normalization.fixture(model.topology)
    matrix = collision_matrices(52)[1].float()
    with torch.no_grad():
        store, memory = infer(model, normal, observation(), k=2, steps=1, information=matrix)
        missing = replace(observation(1, .2), present=torch.zeros(2, 17, dtype=torch.bool))
        unavailable, new_memory = infer(model, normal, missing, memory, k=2, steps=1, information=matrix)
    assert store.provenance['information_encoding'] == 'cross_block_sketch'
    assert torch.equal(new_memory.hidden, memory.hidden) and not unavailable.actions.any()


@pytest.mark.parametrize('nodes, parameters', [(22, 707603), (52, 711443)])
def test_same_capacity_default_identity_and_shared_loss_sample_graph(nodes, parameters):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    old = JointFlow(ModelConfig(nodes=nodes)).double().eval()
    torch.manual_seed(7)
    new = JointFlow(ModelConfig(nodes=nodes, information_encoding='cross_block_sketch')).double().eval()
    assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters()) == parameters
    assert old.state_dict().keys() == new.state_dict().keys()
    assert all(torch.equal(v, new.state_dict()[key]) for key, v in old.state_dict().items())
    context, state = torch.zeros(2, 128, dtype=torch.float64), torch.zeros(2, 6*nodes+13, dtype=torch.float64)
    t = torch.full((2,), .5, dtype=torch.float64)
    independent, coupled, _ = collision_matrices(nodes)
    assert torch.equal(old.velocity(state, t, context), new.velocity(state, t, context))
    assert torch.equal(old.velocity(state, t, context, independent), new.velocity(state, t, context, independent))
    assert not torch.equal(old.velocity(state, t, context, coupled), new.velocity(state, t, context, coupled))
    grad_info = coupled.clone().requires_grad_()
    value = new.velocity(state, t, context, grad_info)
    value.square().mean().backward()
    assert grad_info.grad is None and new.info_shape.weight.grad.abs().sum() > 0
    # Loss fixture labels/normalization are explicitly float32; audit the same
    # graph in that dtype rather than mixing them with the float64 matrix audit.
    new = new.float()
    target = labels(nodes)
    loss, _ = cfm_loss(new, context.float(), target, Normalization.fixture(new.topology), information=coupled.float())
    assert torch.isfinite(loss)
    assert torch.isfinite(new.sample(context[:1].float(), state.float(), steps=2, information=coupled.float())).all()
