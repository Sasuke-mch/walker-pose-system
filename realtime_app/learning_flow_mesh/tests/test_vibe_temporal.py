"""Original VIBE state reset and failure cleanup; no body surrogate."""
import pytest
import torch

from realtime_app.learning_flow_mesh.reproduction.check_vibe_temporal import (
    carried_forward,
    original_encoder,
)


def test_original_native_reset_and_explicit_carry_equivalence():
    torch.manual_seed(113)
    model, _options, _lines = original_encoder()
    features = torch.randn(1, 6, 2048)
    with torch.no_grad():
        whole = model(features)
        first, state = carried_forward(model, features[:, :3])
        last, _state = carried_forward(model, features[:, 3:], state)
        reset = model(features[:, 3:])
    torch.testing.assert_close(torch.cat((first, last), 1), whole, atol=2e-5, rtol=0)
    assert not torch.equal(reset, whole[:, 3:])
    assert not model.gru._forward_hooks and not model.gru._forward_pre_hooks


def test_invalid_hidden_preserves_original_error_and_hook_cleanup():
    model, _options, _lines = original_encoder()
    with pytest.raises(RuntimeError, match="hidden"):
        carried_forward(model, torch.zeros(1, 3, 2048), torch.zeros(1, 1, 3))
    assert not model.gru._forward_hooks and not model.gru._forward_pre_hooks
