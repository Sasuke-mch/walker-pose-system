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
from ..so3_flow import SO3JointFlow, geodesic_path, haar_rotations, rotation_exp
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


@pytest.mark.parametrize("nodes,parameters", [(22, 692112), (52, 695952)])
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
        assert not hasattr(model, "info_pose")
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


@pytest.mark.parametrize("device,dtype,tolerance", [("cpu", torch.float64, 1e-10), ("cuda", torch.float32, 3e-5)])
def test_rotation_exp_closed_form_values_and_gradients(device, dtype, tolerance):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA kernel value check requires CUDA")
    axis = torch.tensor([-1., 2., -3.], device=device, dtype=dtype)
    axis = axis/axis.norm()
    angles = axis.new_tensor([0, 1e-8, .0099, .0101, .4, torch.pi-.0005, torch.pi, 2*torch.pi, 10*torch.pi])
    vectors = angles[:, None]*axis
    actual = rotation_exp(vectors)
    expected = torch.matrix_exp(skew(vectors))
    assert torch.allclose(actual, expected, atol=tolerance, rtol=0)
    validate_rotation_matrices(actual, atol=tolerance)
    if device == "cpu":
        for vector in (vectors[0], vectors[4], vectors[5]):
            gradient = torch.autograd.functional.jacobian(rotation_exp, vector)
            reference = torch.autograd.functional.jacobian(lambda w: torch.matrix_exp(skew(w)), vector)
            assert torch.isfinite(gradient).all()
            assert torch.allclose(gradient, reference, atol=tolerance, rtol=0)
    assert not torch.isfinite(rotation_exp(axis.new_full((3,), torch.nan))).all()


def test_so3_complete_sampler_matches_matrix_exp_kernel():
    normal = Normalization.fixture(Topology())
    model = SO3JointFlow(normal).eval()
    reference = SO3JointFlow(normal, rotation_kernel="matrix_exp").eval()
    reference.load_state_dict(model.state_dict())
    gaussian = torch.randn(3, 325, generator=torch.Generator().manual_seed(19))
    context = torch.zeros(128)
    with torch.no_grad():
        assert torch.allclose(model.sample(context, gaussian, steps=8),
                              reference.sample(context, gaussian, steps=8), atol=3e-5, rtol=0)
        assert torch.allclose(model.regression(context[None]), reference.regression(context[None]), atol=3e-5, rtol=0)


@pytest.mark.parametrize("nodes", [22, 52])
def test_so3_inactive_information_removal_preserves_active_computation(nodes):
    """Storage-only simplification must not masquerade as changed capacity/quality."""
    torch.set_num_threads(1)
    normal = Normalization.fixture(Topology(nodes))
    torch.manual_seed(7)
    compact = SO3JointFlow(normal, ModelConfig(nodes=nodes)).double().eval()
    torch.manual_seed(7)
    original = SO3JointFlow(normal, ModelConfig(nodes=nodes), retain_inactive_information=True).double().eval()
    removed = {f"{name}.weight" for name in ("info_pose", "info_position", "info_shape")}
    assert set(original.state_dict())-set(compact.state_dict()) == removed
    assert sum(original.state_dict()[key].numel() for key in removed) == 15104
    assert sum(p.numel() for p in original.parameters()) == (711056 if nodes == 52 else 707216)
    assert sum(p.numel() for p in compact.parameters()) == (695952 if nodes == 52 else 692112)
    assert all(torch.equal(value, original.state_dict()[name]) for name, value in compact.state_dict().items())
    assert not dict(compact.named_buffers())
    assert sum(value.numel()*value.element_size() for value in original.buffers()) == 13*normal.topology.tangent_dimensions*8
    with pytest.raises(RuntimeError, match="Unexpected key"):
        compact.load_state_dict(original.state_dict())  # Never silently migrate old state dictionaries.
    obs = observation()
    obs = replace(obs, xy=obs.xy.double(), confidence=obs.confidence.double(),
                  origins=obs.origins.double(), directions=obs.directions.double())
    state_pair = [None, None]
    for frame, missing in ((0, False), (1, True), (2, False)):
        current = replace(obs, frame=frame, timestamp=frame*.2)
        if missing:
            current = replace(current, present=torch.zeros_like(current.present), xy=torch.full_like(current.xy, torch.nan))
        with torch.no_grad():
            a, state_pair[0] = infer(compact, normal, current, state_pair[0], k=32, steps=8)
            b, state_pair[1] = infer(original, normal, current, state_pair[1], k=32, steps=8)
        for field in ("raw", "rotations", "translation", "beta", "numerical", "actions"):
            assert torch.equal(getattr(a, field), getattr(b, field))
        assert a.ids == b.ids and a.seeds == b.seeds and a.reasons == b.reasons
        assert a.resources["network_calls"] == b.resources["network_calls"] == 8
        assert a.resources["candidate_state_evaluations"] == b.resources["candidate_state_evaluations"] == 256
        assert state_pair[0].to_dict() == state_pair[1].to_dict()
        assert bool(a.actions.any()) == (not missing)
    context = torch.randn(2, 128, dtype=torch.float64, generator=torch.Generator().manual_seed(17))
    reference = labels(nodes)
    reference = replace(reference, rotations=reference.rotations.double(),
                        translation=reference.translation.double(), beta=reference.beta.double())
    for mode in ("CFM", "regression"):
        losses = []
        for model in (compact, original):
            model.zero_grad(set_to_none=True)
            kwargs = {"generator": torch.Generator().manual_seed(19)} if mode == "CFM" else {}
            loss, _ = objective(mode, model, context, reference, normal, **kwargs)
            loss.backward()
            losses.append(loss.detach())
        assert torch.equal(*losses)
        original_parameters = dict(original.named_parameters())
        for name, parameter in compact.named_parameters():
            expected = original_parameters[name].grad
            assert (parameter.grad is None and expected is None) or torch.equal(parameter.grad, expected)
        assert all(original_parameters[name].grad is None for name in removed)
    for model in (compact, original):
        with pytest.raises(ValueError, match="Lambda"):
            model.regression(context, torch.eye(normal.topology.tangent_dimensions, dtype=torch.float64))
