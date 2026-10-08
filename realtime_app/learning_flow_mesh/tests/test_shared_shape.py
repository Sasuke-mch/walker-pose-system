"""Causal calibration, missing evidence, independent streams and real mesh checks."""
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from realtime_app.learning_flow_mesh.body import BodyDecoder
from realtime_app.learning_flow_mesh.model import ConditionEncoder, ModelConfig
from realtime_app.learning_flow_mesh.shared_shape import (
    PrefixSharedShape,
    encode_with_global_features,
)


@pytest.fixture
def candidate():
    torch.manual_seed(31)
    return PrefixSharedShape(nn.Linear(64, 10), calibration_frames=8)


def inputs():
    torch.manual_seed(32)
    features = torch.randn(2, 13, 64)
    mask = torch.ones(2, 13, dtype=torch.bool)
    mask[0, 1] = False
    mask[1, :3] = False
    frame_ids = torch.arange(13).expand(2, -1).clone()
    return features, mask, frame_ids, ("sequence_a", "sequence_b")


def test_chunks_and_stream_match_batch_and_do_not_modify_old_state(candidate):
    x, mask, ids, names = inputs()
    batch = candidate(x, mask, ids, names)
    state = None
    chunks, ready = [], []
    for start, end in [(0, 2), (2, 7), (7, 11), (11, 13)]:
        before = (state.feature_sum.clone(), state.betas.clone()) if state is not None else None
        result = candidate(x[:, start:end], mask[:, start:end], ids[:, start:end], names, state)
        if state is not None:
            torch.testing.assert_close(state.feature_sum, before[0])
            torch.testing.assert_close(state.betas, before[1], equal_nan=True)
        state = result.state
        chunks.append(result.betas)
        ready.append(result.ready)
    assert torch.equal(batch.ready, torch.cat(ready, dim=1))
    torch.testing.assert_close(batch.betas, torch.cat(chunks, dim=1), rtol=0, atol=0, equal_nan=True)
    # Separate rows freeze at different times and never share a beta.
    assert batch.ready[0].nonzero()[0].item() == 8
    assert batch.ready[1].nonzero()[0].item() == 10
    assert not torch.equal(batch.state.betas[0], batch.state.betas[1])


def test_future_and_masked_values_do_not_change_published_shapes(candidate):
    x, mask, ids, names = inputs()
    base = candidate(x, mask, ids, names)
    changed = x.clone()
    changed[:, 11:] += 10000
    changed[~mask] = torch.nan
    altered = candidate(changed, mask, ids, names)
    torch.testing.assert_close(base.betas, altered.betas, rtol=0, atol=0, equal_nan=True)
    assert torch.isnan(base.betas[~base.ready]).all()
    for row in range(2):
        available = base.betas[row, base.ready[row]]
        assert torch.equal(available, available[:1].expand_as(available))
    # Arbitrary future before another row freezes cannot affect already released row.
    changed[:, 9:] *= -20
    shifted = candidate(changed, mask, ids, names)
    torch.testing.assert_close(base.betas[0], shifted.betas[0], rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(base.betas[:, :9], shifted.betas[:, :9], rtol=0, atol=0, equal_nan=True)


def test_all_missing_remains_unavailable_and_admitted_nan_fails(candidate):
    x, mask, ids, names = inputs()
    empty = candidate(torch.full_like(x, torch.nan), torch.zeros_like(mask), ids, names)
    assert not empty.ready.any() and torch.isnan(empty.betas).all() and empty.state.count.sum() == 0
    x[0, 0] = torch.nan
    with pytest.raises(ValueError, match="admitted features"):
        candidate(x, mask, ids, names)


def test_cross_sequence_and_duplicate_frames_cannot_reuse_state(candidate):
    x, mask, ids, names = inputs()
    first = candidate(x[:, :8], mask[:, :8], ids[:, :8], names)
    with pytest.raises(ValueError, match="another sequence"):
        candidate(x[:, 8:], mask[:, 8:], ids[:, 8:], ("new", "sequence_b"), first.state)
    with pytest.raises(ValueError, match="duplicate or reordered"):
        candidate(x[:, 7:], mask[:, 7:], ids[:, 7:], names, first.state)
    with pytest.raises(ValueError, match="strictly increasing"):
        candidate(x, mask, ids.flip(1), names)
    with pytest.raises(ValueError, match="boolean"):
        candidate(x, mask.float(), ids, names)


def test_single_valid_frame_control_reuses_exact_same_head(candidate):
    x, mask, ids, names = inputs()
    mask.fill_(True)
    control = PrefixSharedShape(candidate.head, 1)
    assert control.head is candidate.head
    single = control(x, mask, ids, names)
    expected = candidate.head(x[:, 0])
    torch.testing.assert_close(single.betas, expected[:, None].expand(-1, 13, -1), atol=0, rtol=0)


def test_real_mesh_and_shape_gradient_contract(candidate):
    config = json.loads(Path("realtime_app/learning_flow_mesh/config.json").read_text(encoding="utf-8"))
    body = BodyDecoder(**config["assets"])
    x, mask, ids, names = inputs()
    x.requires_grad_(True)
    result = candidate(x, mask, ids, names)
    # Decode only available frames, preserving the unavailable prefix in the readout.
    start = result.ready[0].nonzero()[0].item()
    rotations = torch.eye(3).expand(1, 13-start, 52, 3, 3).clone()
    translation = torch.zeros(1, 13-start, 3)
    decoded = body(rotations, result.betas[:1, start], translation)
    assert decoded["vertices"].shape == (1, 13-start, 6890, 3)
    assert torch.isfinite(decoded["vertices"]).all()
    decoded["vertices"].square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad[0, :start+1].abs().max() > 0
    assert x.grad[0, start+1:].abs().max() == 0 and x.grad[~mask].abs().max() == 0
    assert candidate.head.weight.grad is not None and torch.isfinite(candidate.head.weight.grad).all()


@pytest.mark.parametrize("history", [True, False])
def test_original_encoder_hook_preserves_readout_and_is_removed_on_failure(history):
    encoder = ConditionEncoder(ModelConfig(use_history=history)).eval()
    xy = torch.zeros(1, 2, 2, 22, 2)
    present = torch.ones(1, 2, 2, 22, dtype=torch.bool)
    rays = torch.zeros(1, 2, 2, 22, 3)
    rays[..., 2] = 1
    batch = {"xy": xy, "confidence": torch.full_like(present, 0.7, dtype=torch.float32),
             "observation_mask": present, "view_mask": torch.ones(1, 2, 2, dtype=torch.bool),
             "geometry_mask": present, "ray_directions": rays, "ray_origins": torch.zeros_like(rays)}
    layer = encoder.history if history else encoder.norm
    baseline = encoder(batch)
    original_hook_count = len(layer._forward_hooks)
    observed, global_features = encode_with_global_features(encoder, batch)
    assert len(layer._forward_hooks) == original_hook_count
    torch.testing.assert_close(encoder.shape_head(global_features[:, 0]), baseline["betas"], rtol=0, atol=0)
    assert torch.equal(observed["condition"], baseline["condition"])
    batch["xy"][0, 0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        encode_with_global_features(encoder, batch)
    assert len(layer._forward_hooks) == original_hook_count
