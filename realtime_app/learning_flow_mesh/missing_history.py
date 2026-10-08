"""H6 eager missing-history intervention; no new parameters or time model."""
import torch

from .hand_candidate import NativeHandMeshCandidate


def freeze_missing_transitions(gru, pooled, observed, hidden=None):
    """Reuse the original GRU; missing rows carry state and output last hidden.

    Frame duration is deliberately absent. This is an observation-clock control,
    not a physical-time motion model. The scoped caller already ran the original
    batched GRU; direct forward calls here bypass its interception hooks.
    """
    if observed.dtype != torch.bool or observed.shape != pooled.shape[:2]:
        raise ValueError("explicit boolean B,T recurrent observation mask required")
    if observed.device != pooled.device or not gru.batch_first or gru.bidirectional:
        raise ValueError("need same-device batch-first unidirectional original GRU")
    if gru.training or gru.dropout != 0:
        raise ValueError("eager intervention requires eval GRU without dropout")
    if hidden is None:
        hidden = pooled.new_zeros(gru.num_layers, pooled.shape[0], gru.hidden_size)
    outputs = []
    for index in range(pooled.shape[1]):
        current = observed[:, index]
        if current.any():
            _, proposed = gru.forward(pooled[:, index:index + 1], hidden)
            hidden = torch.where(current[None, :, None], proposed, hidden)
        outputs.append(hidden[-1])
    return torch.stack(outputs, 1), hidden


class MissingHistoryMeshCandidate(NativeHandMeshCandidate):
    """Intervene only in H5's recurrent transition, with scoped hook cleanup.

    Shared-module concurrent/exported use is unsupported. The original full GRU
    runs before its outputs are replaced, so this harness is not optimized or a
    latency improvement. Shape pooling and native-hand conditioning stay H5.
    """

    def __init__(self, model, conditioner, *, missing_policy="freeze", **kwargs):
        super().__init__(model, conditioner, **kwargs)
        if missing_policy not in ("freeze", "update_all"):
            raise ValueError("explicit freeze or update_all policy required")
        if not model.cfg.use_history:
            raise ValueError("H6 requires the original enabled recurrent encoder")
        self.missing_policy = missing_policy

    def _encode(self, batch, frame_ids, hidden):
        if self.missing_policy == "update_all":
            return super()._encode(batch, frame_ids, hidden)
        observed = (batch["observation_mask"] & batch["view_mask"][..., None]).any(dim=(-1, -2))
        images = batch.get("image_mask", torch.zeros_like(batch["view_mask"]))
        observed = observed | (images & batch["view_mask"]).any(-1)

        def intercept(module, args, _output):
            pooled, carried = args
            return freeze_missing_transitions(module, pooled, observed, carried)

        hook = self.model.encoder.history.register_forward_hook(intercept)
        try:
            encoded, features = super()._encode(batch, frame_ids, hidden)
        finally:
            hook.remove()
        if not torch.equal(encoded["frame_observed"], observed):
            raise ValueError("original availability differs from declared intervention mask")
        return encoded, features
