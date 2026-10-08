"""Isolated H1 candidate: causal prefix calibration, then one frozen shape.

Features come from the unchanged condition encoder. Admissibility is an explicit
caller contract, not a learned quality gate. No beta exists before calibration.
"""
from dataclasses import dataclass

import torch
from torch import nn


def encode_with_global_features(encoder, batch, hidden=None):
    """Read the exact original encoder intermediate, without copying its forward.

    This eager verification adapter uses a temporary read-only hook and removes
    it even on failure. Compiled/exported or concurrent calls need another
    explicit interface and are not deployment claims made by this candidate.
    """
    captures = []
    layer = encoder.history if encoder.cfg.use_history else encoder.norm

    def capture(_module, _args, output):
        if encoder.cfg.use_history:
            captures.append(output[0])
        else:
            b, t = batch["xy"].shape[:2]
            captures.append(output.reshape(b, t, 52, encoder.cfg.width).mean(dim=2))

    hook = layer.register_forward_hook(capture)
    try:
        encoded = encoder(batch, hidden=hidden)
    finally:
        hook.remove()
    if len(captures) != 1:
        raise ValueError("expected exactly one original global-feature calculation")
    return encoded, captures[0]


@dataclass(frozen=True)
class ShapeState:
    sequence_ids: tuple[str, ...]
    feature_sum: torch.Tensor
    count: torch.Tensor
    betas: torch.Tensor
    ready: torch.Tensor
    last_frame: torch.Tensor
    calibration_frames: int


@dataclass(frozen=True)
class ShapeReadout:
    betas: torch.Tensor  # B,T,10, unavailable entries are NaN
    ready: torch.Tensor  # B,T, boolean
    state: ShapeState


class PrefixSharedShape(nn.Module):
    """Reuse an existing linear beta head; add no learned parameter.

The first K admitted frames are averaged equally and consumed only once. Prefix
outputs are not retroactively repaired. State must be carried between chunks of
the same sequence and explicitly reset for a new sequence.
"""

    def __init__(self, shape_head: nn.Linear, calibration_frames: int = 8):
        super().__init__()
        if not isinstance(shape_head, nn.Linear) or shape_head.out_features != 10:
            raise ValueError("need the existing 10-beta linear head")
        if type(calibration_frames) is not int or calibration_frames < 1:
            raise ValueError("calibration_frames must be a positive integer")
        self.head = shape_head
        self.calibration_frames = calibration_frames

    def forward(self, features, admitted, frame_ids, sequence_ids, state=None):
        if features.ndim != 3 or features.shape[-1] != self.head.in_features:
            raise ValueError("features must be B,T,head_width")
        b, t, width = features.shape
        if b < 1 or t < 1 or admitted.shape != (b, t) or admitted.dtype != torch.bool:
            raise ValueError("need a nonempty batch and explicit boolean B,T admission")
        if frame_ids.shape != (b, t) or frame_ids.dtype != torch.long:
            raise ValueError("frame IDs must be int64 B,T")
        if frame_ids.device != features.device or admitted.device != features.device:
            raise ValueError("features, IDs and admission must use the same device")
        if features.dtype != self.head.weight.dtype or features.device != self.head.weight.device:
            raise ValueError("features must match existing head dtype/device")
        ids = tuple(sequence_ids)
        if len(ids) != b or len(set(ids)) != b or any(not isinstance(s, str) or not s for s in ids):
            raise ValueError("need a unique nonempty sequence ID for each batch row")
        if not torch.isfinite(features[admitted]).all():
            raise ValueError("admitted features must be finite")
        if (frame_ids < 0).any() or (frame_ids[:, 1:] <= frame_ids[:, :-1]).any():
            raise ValueError("frame IDs must be nonnegative and strictly increasing")
        if state is None:
            state = ShapeState(ids, features.new_zeros(b, width),
                               torch.zeros(b, device=features.device, dtype=torch.long),
                               features.new_full((b, 10), torch.nan),
                               torch.zeros(b, device=features.device, dtype=torch.bool),
                               torch.full((b,), -1, device=features.device, dtype=torch.long),
                               self.calibration_frames)
        if state.sequence_ids != ids or state.calibration_frames != self.calibration_frames:
            raise ValueError("state belongs to another sequence or calibration setting; reset explicitly")
        if state.feature_sum.shape != (b, width) or state.betas.shape != (b, 10):
            raise ValueError("state batch/feature dimensions changed")
        if state.feature_sum.device != features.device or state.feature_sum.dtype != features.dtype:
            raise ValueError("state device/dtype changed")
        if (frame_ids[:, 0] <= state.last_frame).any():
            raise ValueError("duplicate or reordered frame IDs across chunks")
        sums, counts, beta, ready = state.feature_sum, state.count, state.betas, state.ready
        outputs, availability = [], []
        for frame in range(t):
            consume = admitted[:, frame] & ~ready
            # Multiplication by zero does not remove NaN; select before accumulating.
            sums = sums + torch.where(consume[:, None], features[:, frame], 0)
            counts = counts + consume.long()
            newly_ready = ~ready & (counts >= self.calibration_frames)
            if newly_ready.any():
                # Evaluate only completed rows; unknown rows must not acquire default beta.
                predicted = self.head(sums[newly_ready] / counts[newly_ready, None])
                if not torch.isfinite(predicted).all():
                    raise ValueError("shape head produced nonfinite beta")
                beta = beta.clone()
                beta[newly_ready] = predicted
            ready = ready | newly_ready
            outputs.append(beta)
            availability.append(ready)
        next_state = ShapeState(ids, sums, counts, beta, ready, frame_ids[:, -1].clone(),
                                self.calibration_frames)
        return ShapeReadout(torch.stack(outputs, dim=1), torch.stack(availability, dim=1), next_state)
