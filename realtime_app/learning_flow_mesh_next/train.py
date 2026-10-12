"""Preparation and losses only. No optimizer run or checkpoint loader is exposed."""
import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from .data_firewall import Labels, validate_manifest
from .joint_flow import JointFlow, ModelConfig, cfm_loss, regression_loss
from .schema import Observation, TemporalState


def configuration():
    return {"authorization": "training_disabled", "optimizer": "AdamW", "lr": 3e-4,
            "weight_decay": 0.01, "gradient_clip": 1, "warmup_steps": 1000,
            "schedule": "cosine", "effective_batch": 16, "unroll": 32,
            "smoke_steps": 2000, "main_steps": 50000, "validation_every": 1000,
            "seeds": [7, 17, 27], "objective": "block_equal_weight_linear_CFM",
            "checkpoint_selection": "validation_energy_score_with_point_and_legality_guards",
            "normalization": "train_only_required", "no_endpoint_mse": True,
            "test_locked": True}


def prepare(manifest, nodes=52):
    gates = validate_manifest(manifest)
    model = JointFlow(ModelConfig(nodes=nodes))
    return {"config": configuration(), "model": model.config.to_dict(), "data_gates": gates,
            "parameters": sum(p.numel() for p in model.parameters()),
            "regression_parameters": sum(p.numel() for p in model.parameters()),
            "G0_G1_core_capacity_matched": True, "training_executed": False}


def objective(mode, model, context, labels, normalization, **kwargs):
    if mode == "CFM":
        if hasattr(model, "flow_loss"):
            return model.flow_loss(context, labels, normalization, **kwargs)
        return cfm_loss(model, context, labels, normalization, **kwargs)
    if mode == "regression":
        if hasattr(model, "validate_normalization"):
            model.validate_normalization(normalization)
        return regression_loss(model, context, labels, normalization, **kwargs), {}
    raise ValueError("unregistered objective")


@dataclass(frozen=True)
class LabelFrame:
    """Explicit target-row binding; a declaration, not certified ground truth."""
    identity: str
    sequence: str
    frame: int
    timestamp: float
    camera_ids: tuple


@dataclass(frozen=True)
class SequenceChunk:
    observations: tuple[Observation, ...]
    labels: Labels
    label_frames: tuple[LabelFrame, ...]

    def validate(self, topology, normalization, state=None):
        count = len(self.observations)
        if not 1 <= count <= configuration()["unroll"]:
            raise ValueError("one to32 original frames required per causal chunk")
        if not isinstance(self.labels, Labels) or any(not isinstance(obs, Observation) for obs in self.observations):
            raise TypeError("typed observations and Labels required")
        if state is not None and not isinstance(state, TemporalState):
            raise TypeError("typed causal carry required")
        if self.labels.topology != topology or normalization.topology != topology:
            raise ValueError("sequence target/normalization topology mismatch")
        _, mask = self.labels.packed(require_complete=False)
        if len(mask) != count or len(self.label_frames) != count:
            raise ValueError("all original observation and target rows require aligned bindings")
        normalization.validate()
        if self.labels.split == "train" and self.labels.source_kind != "analytic_fixture":
            normalization.require_training_statistics()
        elif (self.labels.split, self.labels.source_kind) == ("engineering_fixture", "analytic_fixture"):
            if (normalization.split, normalization.source_kind) != ("engineering_fixture", "analytic_fixture"):
                raise ValueError("analytic sequence requires explicit fixture statistics")
        else:
            raise ValueError("sequence objective accepts train or explicit analytic engineering fixtures only")
        first = self.observations[0]
        previous_frame, previous_time = (-1, -float("inf")) if state is None else (state.last_frame, state.last_timestamp)
        if state is not None and (state.identity, state.sequence, state.camera_ids) != (
                first.identity, first.sequence, first.camera_ids):
            raise ValueError("incompatible carry; explicit stream reset required")
        for obs, target in zip(self.observations, self.label_frames):
            if not isinstance(obs, Observation) or not isinstance(target, LabelFrame):
                raise TypeError("typed original observations and label-frame bindings required")
            obs.validate()
            identity = (obs.identity, obs.sequence, obs.camera_ids)
            if identity != (first.identity, first.sequence, first.camera_ids):
                raise ValueError("one identity/sequence/rig per causal chunk required")
            if obs.frame <= previous_frame or obs.timestamp <= previous_time:
                raise ValueError("original frames and timestamps must strictly increase")
            if (target.identity, target.sequence, target.camera_ids, target.frame, target.timestamp) != (
                    obs.identity, obs.sequence, obs.camera_ids, obs.frame, obs.timestamp):
                raise ValueError("target observation frame binding mismatch")
            previous_frame, previous_time = obs.frame, obs.timestamp
        return self


@dataclass
class SequenceObjective:
    loss: torch.Tensor | None
    frame_losses: torch.Tensor
    contexts: torch.Tensor
    supervised: torch.Tensor
    reasons: list
    label_frames: tuple[LabelFrame, ...]
    state: TemporalState
    resources: dict


def sequence_objective(mode, model, chunk, normalization, state=None, *, generator=None):
    """Causal F0 target path with TBPTT carry; no optimizer or checkpoint I/O.

    Missing rows are retained and still advance observation state. Chunk means
    must be reduced with supervised_count, never with total frames or equally.
    The returned carry is detached; contexts/loss retain within-chunk gradients.
    """
    if mode not in ("CFM", "regression"):
        raise ValueError("unregistered objective")
    if not isinstance(chunk, SequenceChunk):
        raise TypeError("typed SequenceChunk required")
    chunk.validate(model.topology, normalization, state)
    if hasattr(model, "validate_normalization"):
        model.validate_normalization(normalization)
    state = None if state is None else state.detached()
    contexts, losses, supported, reasons = [], [], [], []
    computed_count, nonfinite = 0, False
    for index, obs in enumerate(chunk.observations):
        context, state, observed = model.encoder(obs, state)
        contexts.append(context)
        rejected = []
        target = replace(chunk.labels, **{name: getattr(chunk.labels, name)[index:index+1].detach()
                                         for name in ("rotations", "translation", "beta", "pose_mask",
                                                      "translation_mask", "beta_mask")})
        if not observed:
            rejected.append("no_current_observation")
        if not target.pose_mask.all():
            rejected.append("incomplete_pose_label")
        if not target.translation_mask.all():
            rejected.append("translation_label_unavailable")
        if not target.beta_mask.all():
            rejected.append("beta_label_unavailable")
        usable = not rejected
        loss = context.new_full((), torch.nan)
        if usable:
            options = {"generator": generator} if mode == "CFM" else {}
            loss, _ = objective(mode, model, context[None], target, normalization, **options)
            computed_count += 1
            if not torch.isfinite(loss):
                rejected.append("nonfinite_sequence_objective")
                nonfinite, usable = True, False
                loss = context.new_full((), torch.nan)
        losses.append(loss)
        supported.append(usable)
        reasons.append({"unavailable": rejected, "observation_rejection_reasons": obs.rejection_reasons})
    frame_losses = torch.stack(losses)
    supervised = torch.tensor(supported, dtype=torch.bool, device=frame_losses.device)
    aggregate = frame_losses[supervised].mean() if supervised.any() and not nonfinite else None
    return SequenceObjective(aggregate, frame_losses, torch.stack(contexts), supervised, reasons,
                             chunk.label_frames, state.detached(),
                             {"original_frames": len(losses), "supervised_count": int(supervised.sum()),
                              "skipped_count": int((~supervised).sum()), "encoder_frame_calls": len(losses),
                              "velocity_batch_calls": computed_count, "target_state_evaluations": computed_count,
                              "nonfinite_objective": nonfinite, "optimizer_steps": 0,
                              "checkpoint_loaded": False, "scope": "in-memory causal F0 objective; no dataset authenticity certification"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--nodes", type=int, choices=(22, 52), default=52)
    args = parser.parse_args()
    torch.manual_seed(7)
    print(json.dumps(prepare(json.loads(args.manifest.read_text(encoding="utf-8")), args.nodes), indent=2))


if __name__ == "__main__":
    main()
