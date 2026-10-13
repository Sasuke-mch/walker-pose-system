from dataclasses import replace

import pytest
import torch

from ..candidate_store import infer
from ..data_firewall import Normalization
from ..information_encoding import fixed_probes, information_features
from ..joint_flow import JointFlow, ModelConfig, cfm_loss
from ..schema import TemporalState
from ..state_encoder import StateEncoder
from .test_firewall import labels, observation


def features(matrix, nodes, mode):
    return information_features(matrix, nodes, mode=mode,
                                pose_probes=fixed_probes(len(matrix), 3),
                                shape_probes=fixed_probes(len(matrix), 10))


@pytest.mark.parametrize('nodes', [22, 52])
@pytest.mark.parametrize('mode', ['diagonal_blocks', 'cross_block_sketch'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_batched_blocks_match_original_loop_without_mutating_input(nodes, mode, dtype, device):
    dimensions = 3*nodes+13
    generator = torch.Generator().manual_seed(72)
    # Noncontiguous and nonsymmetric to expose row/column or flatten-order errors.
    matrix = torch.randn(dimensions, dimensions, generator=generator, dtype=dtype).to(device).T
    matrix.requires_grad_()
    before = matrix.detach().clone()
    slices = [slice(3*i, 3*i+3) for i in range(nodes+1)] + [slice(-10, None)]
    info = matrix.detach()
    def compress(value):
        return value.sign()*torch.log1p(value.abs())
    diagonal = [compress(info[part, part]).flatten() for part in slices]
    if mode == 'cross_block_sketch':
        cross = info.clone()
        for part in slices:
            cross[part, part] = 0
        projected = cross @ fixed_probes(dimensions, 3).to(info)
        shape_projected = cross[-10:] @ fixed_probes(dimensions, 10).to(info)
        expected = (torch.stack([old+compress(projected[part]).flatten()
                                for old, part in zip(diagonal[:-1], slices[:-1])]),
                    diagonal[-1]+compress(shape_projected).flatten())
    else:
        expected = torch.stack(diagonal[:-1]), diagonal[-1]
    actual = features(matrix, nodes, mode)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert torch.equal(matrix, before)
    assert not any(value.requires_grad for value in actual)


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


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), -float("inf")])
def test_corrupt_carry_clock_rejected_before_neural_forward(timestamp, restored):
    model = StateEncoder().eval()
    with torch.no_grad():
        _, state, _ = model(observation())
    state = replace(state, last_timestamp=timestamp)
    if restored:
        state = TemporalState.from_dict(state.to_dict())
    calls = []
    hook = model.local[0].register_forward_pre_hook(lambda *_: calls.append("measurement"))
    try:
        with pytest.raises(ValueError, match="finite.*timestamp"):
            model(observation(1, .2), state)
        assert calls == []
    finally:
        hook.remove()


@pytest.mark.parametrize("field, value", [
    ("last_frame", -1), ("last_frame", .5), ("last_frame", True),
    ("hidden", torch.full((128,), torch.nan)),
    ("age", torch.full((2, 17), -1.)),
    ("seen", torch.ones(2, 17)),
])
def test_corrupt_carry_fields_share_encoder_validation(field, value):
    model = StateEncoder().eval()
    with torch.no_grad():
        _, state, _ = model(observation())
    with pytest.raises(ValueError):
        model(observation(2, .2), replace(state, **{field: value}))


@pytest.mark.parametrize("nodes", [22, 52])
@pytest.mark.parametrize("mode", ["diagonal_blocks", "cross_block_sketch"])
def test_sampler_reuses_only_call_local_information_with_equal_values_and_gradients(nodes, mode):
    torch.set_num_threads(1)
    torch.manual_seed(31)
    model = JointFlow(ModelConfig(nodes=nodes, information_encoding=mode)).double().eval()
    initial = torch.randn(2, model.topology.dimensions, dtype=torch.float64, requires_grad=True)
    context = torch.randn(1, 128, dtype=torch.float64, requires_grad=True)
    info = collision_matrices(nodes)[1].requires_grad_()
    steps, calls = 8, []
    targets = (initial, context, model.info_pose.weight, model.info_position.weight,
               model.info_shape.weight, model.blocks[0].linear1.weight, model.pose_out.weight, info)
    def original_euler(information):
        state = initial
        for step in range(steps):
            state = state + model.velocity(state, state.new_full((len(state),), step/steps),
                                            context.expand(len(state), -1), information)/steps
        return state
    hook = model.info_shape.register_forward_hook(lambda *_: calls.append(1))
    try:
        expected = original_euler(info)
        assert len(calls) == steps
        expected_grad = torch.autograd.grad(expected.square().mean(), targets, allow_unused=True)
        calls.clear()
        actual = model.sample(context, initial, steps=steps, information=info)
        assert len(calls) == 1 and torch.equal(actual, expected)
        actual_grad = torch.autograd.grad(actual.square().mean(), targets, allow_unused=True)
        assert actual_grad[-1] is expected_grad[-1] is None
        for a, b in zip(actual_grad[:-1], expected_grad[:-1]):
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)
        with torch.no_grad():
            changed = model.sample(context, initial, steps=steps, information=info*2)
            assert not torch.equal(changed, actual) and torch.equal(changed, original_euler(info*2))
    finally:
        hook.remove()
