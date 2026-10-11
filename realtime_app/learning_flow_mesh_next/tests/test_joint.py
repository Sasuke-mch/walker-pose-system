import json
from dataclasses import replace

import pytest
import torch

from ..candidate_store import CandidateStore, infer, nested_noise
from ..data_firewall import Normalization
from ..joint_flow import JointFlow, ModelConfig, cfm_loss
from ..schema import TemporalState, Topology
from .test_firewall import labels, observation


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    return JointFlow().eval()


def test_causality_chunk_and_serialized_state(model):
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


def test_missing_does_not_update_memory_or_publish(model):
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
