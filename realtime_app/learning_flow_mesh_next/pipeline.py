"""Two explicit generators, frozen G0 anchor/J and counted complete work."""
import time
from dataclasses import dataclass

import torch

from .candidate_store import infer
from .geometry_information import local_information


@dataclass
class TwoStageState:
    g0: object = None
    g1: object = None

    def detached(self):
        return TwoStageState(self.g0.detached() if self.g0 is not None else None,
                             self.g1.detached() if self.g1 is not None else None)

    def to_dict(self):
        return {key: getattr(self, key).to_dict() if getattr(self, key) is not None else None
                for key in ("g0", "g1")}

    @classmethod
    def from_dict(cls, value, **kwargs):
        from .schema import TemporalState
        return cls(*(TemporalState.from_dict(value[key], **kwargs) if value[key] is not None else None
                     for key in ("g0", "g1")))


class TwoStageInference:
    def __init__(self, g0, g1, normalization, scale, surface, cameras, noise):
        if g0 is g1 or g0.topology != g1.topology:
            raise ValueError("distinct matched-topology G0/G1 models required")
        self.g0 = g0.eval().requires_grad_(False)
        self.g1 = g1
        self.normalization, self.scale, self.surface = normalization, scale, surface
        self.cameras, self.noise = cameras, noise

    def __call__(self, observation, state=None, *, k=32, steps=8, anchor_k=32, jacobian_step=1e-4):
        state = state or TwoStageState()
        if observation.xy.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            g0, state0 = infer(self.g0, self.normalization, observation, state.g0,
                               k=anchor_k, steps=steps, stream="G0")
            information = local_information(g0, observation, self.cameras, self.noise,
                                            self.scale, self.surface, step=jacobian_step)
        g1, state1 = infer(self.g1, self.normalization, observation, state.g1,
                           k=k, steps=steps, stream="G1", information=information.matrix)
        if observation.xy.is_cuda:
            torch.cuda.synchronize()
        report = {"generators": 2, "network_calls": g0.resources["network_calls"] + g1.resources["network_calls"],
                  "candidate_state_evaluations": g0.resources["candidate_state_evaluations"] + g1.resources["candidate_state_evaluations"],
                  "G0_ms": g0.resources["total_ms"], "J_ms": information.resources.get("total_ms", 0),
                  "G1_ms": g1.resources["total_ms"], "total_ms": (time.perf_counter()-start)*1000,
                  "J_decoder_calls": information.resources.get("decoder_batch_calls", 0),
                  "J_decoder_state_evaluations": information.resources.get("decoder_state_evaluations", 0),
                  "information_fallback_reasons": information.reasons,
                  "matched_total_sampling_G0_control_K": anchor_k+k,
                  "matched_walltime_control": "must measure separately; sample count alone is insufficient"}
        g1.provenance["information"] = {"anchor_id": information.anchor_id, "noise_source": information.noise_source,
                                        "calibrated": information.calibrated, "reasons": information.reasons}
        return g1, g0, information, TwoStageState(state0, state1), report
