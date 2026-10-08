"""H5: hand-local condition residual, independent of shape/history/translation.

Untrained project adaptation, not the original HMR2 decoder. Native hand
hypotheses remain input conditions; generated rotations are never overwritten.
"""
from dataclasses import replace

import numpy as np
import torch
from torch import nn

from .candidate import CausalMeshCandidate
from .hand_conditions import (
    HAND_NODES,
    SIDE_ORDER,
    HandConvention,
    NativeHandTokens,
    native_hand_tokens,
)


def saved_native_hands(arrays, body_frame_ids, view_available, parents, *,
                       device="cpu", dtype=torch.float32):
    """Rebuild checked H4 tokens from saved native sources, never fitted hands."""
    ids = np.asarray(arrays["native_frame_ids"])
    if ids.ndim != 2 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("saved H4 IDs must be an integer B,T array")
    rotations = np.asarray(arrays["source_native_rotations"])
    confidence = np.asarray(arrays["source_box_confidence"])
    if ids.shape[0] != 1 or rotations.ndim != 6 or rotations.shape[1] != 2:
        raise ValueError("saved H4 stream must be one sequence with cam0/cam1 views")
    return native_hand_tokens(
        torch.tensor(rotations, dtype=dtype, device=device)[None],
        torch.tensor(confidence, dtype=dtype, device=device)[None],
        source_frame_ids=torch.tensor(ids.astype(np.int64), device=device),
        target_frame_ids=body_frame_ids, view_available=view_available,
        convention=HandConvention.SMPLH_PARENT_LOCAL_NATIVE, side_order=SIDE_ORDER, parents=parents)


def slice_native_hands(hands, start, end):
    return replace(hands, **{name: getattr(hands, name)[:, start:end] for name in
                            ("tokens", "present", "local_rotations", "source_present",
                             "admitted", "frame_ids")})


def disabled_native_hands(hands):
    """Explicit controlled input ablation; preserve original source availability."""
    return replace(hands, tokens=torch.full_like(hands.tokens, torch.nan),
                   local_rotations=torch.full_like(hands.local_rotations, torch.nan),
                   present=torch.zeros_like(hands.present), admitted=torch.zeros_like(hands.admitted))


class NativeHandConditioner(nn.Module):
    """Same-hand queries attend both camera hypotheses, with exact zero if absent."""

    camera_order = ("cam0", "cam1")

    def __init__(self, width=64, heads=4):
        super().__init__()
        if type(width) is not int or type(heads) is not int or min(width, heads) < 1 or width % heads:
            raise ValueError("positive width divisible by heads required")
        self.width = width
        self.projection = nn.Sequential(nn.Linear(7, width), nn.GELU(), nn.Linear(width, width))
        self.view_identity = nn.Parameter(torch.randn(2, width) * .02)
        self.node_identity = nn.Parameter(torch.randn(2, 15, width) * .02)
        self.memory_norm = nn.LayerNorm(width)
        self.query_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0, batch_first=True)
        self.register_buffer("native_nodes", torch.tensor(HAND_NODES, dtype=torch.long))

    def _validate(self, condition, hands, view_available):
        if not isinstance(hands, NativeHandTokens):
            raise TypeError("explicit NativeHandTokens required, not a guessed tensor layout")
        if condition.ndim != 4 or condition.shape[-2:] != (52, self.width):
            raise ValueError("condition must be B,T,52,width")
        b, t = condition.shape[:2]
        if min(b, t) < 1 or condition.dtype != self.view_identity.dtype or condition.device != self.view_identity.device:
            raise ValueError("nonempty conditions must match conditioner dtype/device")
        expected = (b, t, 2, 2, 15, 7)
        if hands.tokens.shape != expected or hands.tokens.device != condition.device or hands.tokens.dtype != condition.dtype:
            raise ValueError("native tokens must match B,T,cam0/cam1,left/right,15,7 and dtype/device")
        for value, shape in ((hands.present, expected[:-1]), (hands.admitted, expected[:4]),
                             (hands.source_present, expected[:4])):
            if value.dtype != torch.bool or value.shape != shape or value.device != condition.device:
                raise ValueError("native availability must be explicit boolean with correct shape/device")
        if (hands.admitted & ~hands.source_present).any() or not torch.equal(
                hands.present, hands.admitted[..., None].expand_as(hands.present)):
            raise ValueError("native source/admission/joint presence disagree")
        if hands.node_ids.dtype != torch.long or not torch.equal(hands.node_ids, self.native_nodes):
            raise ValueError("native anatomical node IDs differ")
        if view_available is not None and (
                view_available.shape != (b, t, 2) or view_available.dtype != torch.bool
                or view_available.device != condition.device
                or (hands.admitted & ~view_available[..., None]).any()):
            raise ValueError("native admission contradicts selected body view")
        if not torch.isfinite(hands.tokens[hands.present]).all():
            raise ValueError("present native conditions must be finite")
        score = hands.tokens[..., 6][hands.present]
        if ((score <= 0) | (score > 1)).any():
            raise ValueError("present native detector box score must be in (0,1]")

    def forward(self, condition, hands, *, view_available=None):
        self._validate(condition, hands, view_available)
        b, t = condition.shape[:2]
        # Rows are (batch, time, anatomical hand), keys are (camera, native joint).
        present = hands.present.permute(0, 1, 3, 2, 4).reshape(b * t * 2, 30)
        active = present.any(-1)
        residual = condition.new_zeros(b * t * 2, 15, self.width)
        if active.any():
            values = hands.tokens.permute(0, 1, 3, 2, 4, 5).reshape(b * t * 2, 30, 7)
            values = torch.where(present[..., None], values, 0)[active]
            identity = (self.node_identity[:, None] + self.view_identity[None, :, None])
            identity = identity.reshape(1, 2, 30, self.width).expand(b * t, -1, -1, -1)
            identity = identity.reshape(b * t * 2, 30, self.width)[active]
            memory = self.memory_norm(self.projection(values) + identity)
            query = condition[:, :, 22:].reshape(b * t * 2, 15, self.width)[active]
            if not torch.isfinite(query).all():
                raise ValueError("present hand query must be finite")
            attended, _ = self.attention(self.query_norm(query), memory, memory,
                                         key_padding_mask=~present[active], need_weights=False)
            residual[active] = attended
        # No global update, no hard hand-pose substitution, no null-token fallback.
        return torch.cat((condition.new_zeros(b, t, 22, self.width),
                          residual.reshape(b, t, 30, self.width)), dim=2)


class NativeHandMeshCandidate(CausalMeshCandidate):
    def __init__(self, model, conditioner, **kwargs):
        super().__init__(model, **kwargs)
        if not isinstance(conditioner, NativeHandConditioner) or conditioner.width != model.cfg.width:
            raise ValueError("native conditioner must match existing model width")
        self.conditioner = conditioner

    def _encode(self, batch, frame_ids, hidden):
        hands = batch.get("native_hands")
        if not isinstance(hands, NativeHandTokens):
            raise TypeError("explicit native_hands required, even for a missing-hand control")
        if (hands.frame_ids.dtype != torch.long or hands.frame_ids.device != frame_ids.device
                or not torch.equal(hands.frame_ids, frame_ids)):
            raise ValueError("native/body step frame IDs differ")
        encoded, features = super()._encode(batch, frame_ids, hidden)
        delta = self.conditioner(encoded["condition"], hands, view_available=batch["view_mask"])
        # H1 features, GRU hidden, shape/T heads and frame availability stay original.
        return dict(encoded, condition=encoded["condition"] + delta), features
