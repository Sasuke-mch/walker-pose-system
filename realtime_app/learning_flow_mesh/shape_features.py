"""H7 isolates shape readout from recurrent summaries, preserving pose history."""
from dataclasses import dataclass, replace

from .candidate import CandidateState
from .missing_history import MissingHistoryMeshCandidate


@dataclass(frozen=True)
class ShapeFeatureState(CandidateState):
    shape_feature_source: str
    missing_policy: str


class ShapeFeatureMeshCandidate(MissingHistoryMeshCandidate):
    """Use the original GRU input for shape only, or exact H6 history control.

    This read-only eager pre-hook never changes GRU arguments or outputs. Local
    means frame-local normalized query pooling, not statistically independent
    information. Shared-module concurrency and export remain unsupported.
    """

    def __init__(self, model, conditioner, *, shape_feature_source="local", **kwargs):
        super().__init__(model, conditioner, **kwargs)
        if shape_feature_source not in ("local", "history"):
            raise ValueError("explicit local or history shape feature source required")
        self.shape_feature_source = shape_feature_source

    def _encode(self, batch, frame_ids, hidden):
        if self.shape_feature_source == "history":
            return super()._encode(batch, frame_ids, hidden)
        captured = []

        def capture(_module, args):
            captured.append(args[0])

        hook = self.model.encoder.history.register_forward_pre_hook(capture)
        try:
            encoded, _history = super()._encode(batch, frame_ids, hidden)
        finally:
            hook.remove()
        if len(captured) != 1:
            raise ValueError("expected exactly one original pooled GRU input")
        return encoded, captured[0]

    def step(self, batch, frame_ids, sequence_ids, admitted, state=None):
        if state is not None and (
                not isinstance(state, ShapeFeatureState)
                or state.shape_feature_source != self.shape_feature_source
                or state.missing_policy != self.missing_policy):
            raise ValueError("shape feature/missing policy state changed; reset explicitly")
        output = super().step(batch, frame_ids, sequence_ids, admitted, state)
        carried = output.state
        return replace(output, state=ShapeFeatureState(
            carried.shape, carried.hidden, carried.initial_noise, carried.noise_seed,
            self.shape_feature_source, self.missing_policy))
