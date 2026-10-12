import json
from dataclasses import replace

import pytest
import torch

from realtime_app.learning_flow_mesh.rotations import (
    six_to_matrix,
    validate_rotation_matrices,
)

from ..candidate_store import CandidateStore, infer, nested_noise
from ..data_firewall import Normalization
from ..joint_flow import JointFlow, ModelConfig, cfm_loss
from ..schema import TemporalState, Topology
from ..so3_flow import SO3JointFlow, geodesic_path, haar_rotations
from ..tangent import skew
from ..train import objective
from .test_firewall import labels, observation


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    return JointFlow().eval()


@pytest.mark.parametrize("geometry", ["ambient", "SO3"])
def test_causality_chunk_and_serialized_state(model, geometry):
    if geometry == "SO3":
        model = SO3JointFlow(Normalization.fixture(model.topology)).eval()
    normal = Normalization.fixture(model.topology)
    frames = [observation(i, i*.1) for i in range(5)]
    def replay(sequence, serialize=False):
        state, values = None, []
        for obs in sequence:
            with torch.no_grad():
                store, state = infer(model, normal, obs, state, k=3, steps=2)
            values.append(store.raw)
            if serialize:
                state = TemporalState.from_dict(json.loads(json.dumps(state.to_dict())))
        return values, state
    whole, end = replay(frames)
    chunk, restored = replay(frames, True)
    assert all(torch.equal(a, b) for a, b in zip(whole, chunk))
    assert torch.equal(end.hidden, restored.hidden)
    altered = frames[:3] + [replace(f, xy=f.xy*100) for f in frames[3:]]
    assert all(torch.equal(a, b) for a, b in zip(whole[:3], replay(altered)[0][:3]))


@pytest.mark.parametrize("geometry", ["ambient", "SO3"])
def test_missing_does_not_update_memory_or_publish(model, geometry):
    if geometry == "SO3":
        model = SO3JointFlow(Normalization.fixture(model.topology)).eval()
    normal = Normalization.fixture(model.topology)
    _, state = infer(model, normal, observation(), k=1, steps=1)
    obs = observation(1, .2)
    missing = replace(obs, present=torch.zeros_like(obs.present), xy=torch.full_like(obs.xy, torch.nan))
    store, new = infer(model, normal, missing, state, k=3, steps=1)
    assert torch.equal(new.hidden, state.hidden)
    assert torch.allclose(new.age, state.age+.2)
    assert not store.actions.any()
    assert all("no_current_observation" in r for r in store.reasons)
    with pytest.raises(ValueError, match="explicit reset"):
        infer(model, normal, replace(obs, identity="other"), state, k=1)


def test_nested_independent_frame_noise_and_cache(model, tmp_path):
    a, seeds = nested_noise(observation(), 8, model.topology)
    b, more = nested_noise(observation(), 32, model.topology)
    assert torch.equal(a, b[:8]) and seeds == more[:8]
    c, _ = nested_noise(observation(1, .1), 8, model.topology)
    assert not torch.equal(a, c)
    store, _ = infer(model, Normalization.fixture(model.topology), observation(), k=3, steps=2)
    path = tmp_path / "cache.npz"
    store.save(path)
    cached = CandidateStore.load(path)
    assert cached.ids == store.ids and torch.equal(cached.raw, store.raw)
    assert cached.provenance["rejection_reasons"] == [list(r) for r in observation().rejection_reasons]
    with pytest.raises(FileExistsError):
        store.save(path)
    assert store.resources["network_calls"] == 2
    assert store.resources["candidate_state_evaluations"] == 6


def test_joint_loss_shared_graph_and_gradient(model):
    context = torch.zeros(2, 128)
    normal = Normalization.fixture(model.topology)
    loss, parts = cfm_loss(model, context, labels(), normal,
                           generator=torch.Generator().manual_seed(17))
    assert torch.isfinite(loss) and set(parts) == {"pose", "translation", "shape"}
    loss.backward()  # One numerical gradient check, no optimizer or training.
    for head in (model.pose_out, model.position_out, model.shape_out):
        assert head.weight.grad.abs().sum() > 0
    for block in model.blocks:
        assert block.linear1.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="incomplete"):
        cfm_loss(model, context, replace(labels(), beta_mask=torch.zeros(2, dtype=torch.bool)), normal)
    assert model.regression(context).shape == (2, 325)
    assert JointFlow(ModelConfig(nodes=22)).topology == Topology(22)


def test_g1_detaches_information_and_reuses_graph(model):
    info = torch.eye(169, requires_grad=True)
    context = torch.zeros(2, 128)
    velocity = model.velocity(torch.zeros(2, 325), torch.ones(2)*.5, context, info)
    velocity.sum().backward()
    assert info.grad is None and model.info_shape.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("nodes", [22, 52])
def test_so3_fixed_path_endpoint_and_time_derivative(nodes):
    axis = torch.tensor([-1., 2., -3.], dtype=torch.float64)
    axis = axis/axis.norm()
    source = torch.matrix_exp(skew(torch.tensor([0., 0., .4], dtype=torch.float64))).expand(1, nodes, 3, 3)
    delta = axis*(torch.pi-.0005)
    target = source@torch.matrix_exp(skew(delta))
    start, _ = geodesic_path(source, target, torch.zeros(1, dtype=torch.float64))
    end, _ = geodesic_path(source, target, torch.ones(1, dtype=torch.float64))
    assert torch.allclose(start, source, atol=1e-10, rtol=0)
    assert torch.allclose(end, target, atol=1e-10, rtol=0)
    time = torch.tensor([.4], dtype=torch.float64)
    point, derivative = torch.autograd.functional.jvp(lambda t: geodesic_path(source, target, t)[0], time, torch.ones_like(time))
    assert torch.allclose(derivative, point@skew(delta), atol=1e-10, rtol=0)
    with pytest.raises(ValueError, match="frozen endpoints"):
        geodesic_path(source.clone().requires_grad_(), target, time)


@pytest.mark.parametrize("nodes,parameters", [(22, 707216), (52, 711056)])
def test_so3_group_step_and_nonuniform_output_normalization(nodes, parameters):
    normal = Normalization(torch.full((6*nodes+13,), .2), torch.linspace(.5, 2, 6*nodes+13),
                           Topology(nodes), "analytic_fixture_nonuniform")
    model = SO3JointFlow(normal, ModelConfig(nodes=nodes)).eval()
    assert sum(p.numel() for p in model.parameters()) == parameters
    initial = torch.ones(2, 6*nodes+13)
    q = torch.tensor([torch.cos(torch.tensor(.2)), 0., 0., torch.sin(torch.tensor(.2))])
    initial[:, :6*nodes].reshape(2, nodes, 6)[..., :4] = q
    omega = torch.tensor([.3, .1, -.2])
    auxiliary_velocity = torch.linspace(-.2, .5, 13)
    fixed = torch.cat((omega.repeat(nodes), auxiliary_velocity))
    def constant(state, flow_time, context, information=None):
        return fixed.expand(len(state), -1)
    model.velocity = constant
    physical = normal.denormalize(model.sample(torch.zeros(128), initial, steps=4))
    rotation, valid = six_to_matrix(physical[:, :6*nodes].reshape(2, nodes, 6))
    base = torch.matrix_exp(skew(torch.tensor([0., 0., .4])))
    expected = base@torch.matrix_exp(skew(omega))
    assert valid.all() and torch.allclose(rotation, expected.expand_as(rotation), atol=1e-5, rtol=0)
    expected_auxiliary = normal.mean[-13:]+normal.scale[-13:]*(initial[:, -13:]+auxiliary_velocity)
    assert torch.allclose(physical[:, -13:], expected_auxiliary, atol=1e-6, rtol=0)


@pytest.mark.parametrize("nodes", [22, 52])
def test_so3_objectives_and_sampler_have_active_gradients(nodes):
    normal = Normalization.fixture(Topology(nodes))
    model = SO3JointFlow(normal, ModelConfig(nodes=nodes)).eval()
    context = torch.zeros(2, 128)
    for mode in ("CFM", "regression"):
        model.zero_grad(set_to_none=True)
        kwargs = {"generator": torch.Generator().manual_seed(17)} if mode == "CFM" else {}
        loss, _ = objective(mode, model, context, labels(nodes), normal, **kwargs)
        assert torch.isfinite(loss)
        loss.backward()
        for head in (model.pose_out, model.position_out, model.shape_out):
            assert torch.isfinite(head.weight.grad).all() and head.weight.grad.abs().sum() > 0
        assert model.blocks[0].linear1.weight.grad.abs().sum() > 0
        assert model.info_pose.weight.grad is None
    model.zero_grad(set_to_none=True)
    noise = torch.randn(2, 6*nodes+13, generator=torch.Generator().manual_seed(19))
    sampled = model.sample(context, noise, steps=2)
    (sampled[:, 1].sum()+sampled[:, -13:].square().sum()).backward()
    assert torch.isfinite(model.pose_out.weight.grad).all() and model.pose_out.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="incomplete"):
        objective("CFM", model, context, replace(labels(nodes), beta_mask=torch.zeros(2, dtype=torch.bool)), normal)


def test_so3_candidate_interface_rejects_wrong_codec_and_information():
    normal = Normalization.fixture(Topology())
    model = SO3JointFlow(normal).eval()
    store, _ = infer(model, normal, observation(), k=3, steps=2)
    validate_rotation_matrices(store.rotations)
    assert store.actions.all() and store.resources["network_calls"] == 2
    assert store.resources["solver"] == "LieEuler" and store.raw.shape == (3, 325)
    reg, _ = infer(model, normal, observation(), k=1, regression=True)
    assert reg.actions.all() and reg.resources["network_calls"] == 1
    with pytest.raises(ValueError, match="normalization mismatch"):
        infer(model, replace(normal, mean=normal.mean+1), observation(), k=1)
    with pytest.raises(ValueError, match="Lambda"):
        infer(model, normal, observation(), k=1, information=torch.eye(169))
    with pytest.raises(ValueError, match="invalid SO3 base"):
        haar_rotations(torch.zeros(1, 52, 4))
