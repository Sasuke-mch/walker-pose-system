"""Preparation and losses only. No optimizer run or checkpoint loader is exposed."""
import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from .data_firewall import Labels, validate_manifest
from .geometry_information import Information, local_information
from .geometry_observation import NoiseModel, check_rig
from .input_binding import require_information_binding
from .joint_flow import JointFlow, ModelConfig, cfm_loss, regression_loss
from .schema import Observation, TemporalState, Topology
from .tangent import TangentScale


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


def _reduce_objectives(losses, counts=None):
    """Preserve differentiable means without overflowing a finite loss dtype."""
    if not len(losses):
        return None, False
    values = losses.to(torch.float64)
    weights = torch.ones_like(values) if counts is None else values.new_tensor(counts)
    aggregate = (values*(weights/weights.sum())).sum().to(losses.dtype)
    nonfinite = not bool(torch.isfinite(aggregate))
    return None if nonfinite else aggregate, nonfinite


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
        if state is not None:
            state.validate()
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


@dataclass(frozen=True)
class SequenceInformation:
    """Original-row acquisition evidence; declarations do not certify history or assets."""
    frames: tuple[Information, ...]
    anchor_resources: tuple[dict, ...]
    cameras: tuple
    noise: NoiseModel
    scale: TangentScale
    state: TemporalState

    def validate(self, observations, topology):
        if not isinstance(self.noise, NoiseModel) or not isinstance(self.scale, TangentScale):
            raise TypeError("typed acquisition noise and tangent scale required")
        if len(self.frames) != len(observations) or len(self.anchor_resources) != len(observations):
            raise ValueError("information requires every original observation row")
        self.scale.validate(topology)
        for obs, info, resources in zip(observations, self.frames, self.anchor_resources):
            if not isinstance(info, Information):
                raise TypeError("typed original Information required")
            expected = {"identity": obs.identity, "sequence": obs.sequence, "frame": obs.frame,
                        "timestamp": obs.timestamp, "camera_ids": list(obs.camera_ids),
                        "operator": obs.operator, "source": obs.source}
            if info.provenance != expected:
                raise ValueError("sequence information observation provenance mismatch")
            require_information_binding(info, obs, self.cameras, self.noise)
            if info.diagnostics.get("candidate_current_observation_binding") != "verified":
                raise ValueError("unverified sequence candidate observation binding")
            if (info.scale.shape != self.scale.values.shape
                    or not torch.equal(info.scale.to(self.scale.values), self.scale.values)):
                raise ValueError("sequence information coordinate scale mismatch")
            if info.noise_source != self.noise.source or info.calibrated != self.noise.calibrated:
                raise ValueError("sequence information noise declaration mismatch")
            for value in (info.matrix, info.jacobian, info.scale, info.anchor_rotation):
                if value is not None and value.requires_grad:
                    raise ValueError("sequence information must be frozen")
            if info.matrix is None:
                if not info.reasons:
                    raise ValueError("unavailable information requires original reasons")
            elif (info.matrix.shape != (topology.tangent_dimensions,)*2
                  or not torch.isfinite(info.matrix).all() or not torch.allclose(info.matrix, info.matrix.T)
                  or info.anchor_id is None or info.anchor_rotation is None
                  or info.anchor_rotation.shape != (topology.nodes, 3, 3)):
                raise ValueError("invalid or unverified sequence information")
            for record, keys in ((resources, ("network_calls", "candidate_state_evaluations")),
                                 (info.resources, ("decoder_batch_calls", "decoder_state_evaluations"))):
                if any(type(record.get(key)) is not int or record[key] < 0 for key in keys):
                    raise ValueError("explicit original acquisition counts required")
        return self

    def counts(self):
        return {"G0_network_calls": sum(r["network_calls"] for r in self.anchor_resources),
                "G0_candidate_state_evaluations": sum(r["candidate_state_evaluations"] for r in self.anchor_resources),
                "J_decoder_batch_calls": sum(info.resources["decoder_batch_calls"] for info in self.frames),
                "J_decoder_state_evaluations": sum(info.resources["decoder_state_evaluations"] for info in self.frames),
                "available_information_frames": sum(info.matrix is not None for info in self.frames),
                "unavailable_information_frames": sum(info.matrix is None for info in self.frames),
                "computed_in_this_objective": False}


def acquire_sequence_information(g0, observations, normalization, scale, surface, cameras, noise, state=None,
                                 *, anchor_k=32, steps=8, jacobian_step=1e-4, diagnostics=False):
    """Acquire frozen information from observations only, never from target labels."""
    from .candidate_store import infer

    if state is not None and not isinstance(state, TemporalState):
        raise TypeError("typed acquisition carry required")
    if state is not None:
        state.validate(g0.encoder.width)
    if g0.training or any(parameter.requires_grad for parameter in g0.parameters()):
        raise ValueError("explicit frozen eval G0 required")
    if not 1 <= len(observations) <= configuration()["unroll"]:
        raise ValueError("one to32 original acquisition observations required")
    normalization.validate()
    if normalization.topology != g0.topology:
        raise ValueError("acquisition normalization topology mismatch")
    scale.validate(g0.topology)
    previous = state
    for obs in observations:
        if not isinstance(obs, Observation):
            raise TypeError("typed acquisition observation required")
        check_rig(obs, cameras)
        noise.validate_for(obs)
        first = observations[0]
        if (obs.identity, obs.sequence, obs.camera_ids) != (first.identity, first.sequence, first.camera_ids):
            raise ValueError("one acquisition observation stream required")
        if previous is not None:
            previous_frame = previous.last_frame if isinstance(previous, TemporalState) else previous.frame
            previous_time = previous.last_timestamp if isinstance(previous, TemporalState) else previous.timestamp
            if ((previous.identity, previous.sequence, previous.camera_ids) != (obs.identity, obs.sequence, obs.camera_ids)
                    or obs.frame <= previous_frame or obs.timestamp <= previous_time):
                raise ValueError("incompatible or nonincreasing acquisition stream; explicit reset required")
        previous = obs
    state = None if state is None else state.detached()
    frames, resources = [], []
    with torch.no_grad():
        for obs in observations:
            anchor, state = infer(g0, normalization, obs, state, k=anchor_k, steps=steps, stream="G0")
            frames.append(local_information(anchor, obs, cameras, noise, scale, surface,
                                            step=jacobian_step, diagnostics=diagnostics))
            resources.append(anchor.resources)
    result = SequenceInformation(tuple(frames), tuple(resources), tuple(cameras), noise, scale, state.detached())
    return result.validate(observations, g0.topology)


def sequence_objective(mode, model, chunk, normalization, state=None, *, generator=None, information=None):
    """Causal F0/G1 target path with TBPTT carry; no optimizer or checkpoint I/O.

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
    if information is not None:
        if not isinstance(information, SequenceInformation):
            raise TypeError("typed SequenceInformation required")
        if hasattr(model, "rotation_representation"):
            raise ValueError("SO3 F0 does not define sequence G1 information conditioning")
        information.validate(chunk.observations, model.topology)
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
            if information is not None:
                matrix = information.frames[index].matrix
                options["information"] = None if matrix is None else matrix.detach()
            loss, _ = objective(mode, model, context[None], target, normalization, **options)
            computed_count += 1
            if not torch.isfinite(loss):
                rejected.append("nonfinite_sequence_objective")
                nonfinite, usable = True, False
                loss = context.new_full((), torch.nan)
        losses.append(loss)
        supported.append(usable)
        row = {"unavailable": rejected, "observation_rejection_reasons": obs.rejection_reasons}
        if information is not None:
            info = information.frames[index]
            row.update(information_unavailable=list(info.reasons), information_anchor_id=info.anchor_id,
                       information_noise_source=info.noise_source, information_calibrated=info.calibrated)
        reasons.append(row)
    frame_losses = torch.stack(losses)
    supervised = torch.tensor(supported, dtype=torch.bool, device=frame_losses.device)
    aggregate, nonfinite_aggregation = (None, False) if nonfinite else _reduce_objectives(frame_losses[supervised])
    nonfinite = nonfinite or nonfinite_aggregation
    resources = {"original_frames": len(losses), "supervised_count": int(supervised.sum()),
                 "skipped_count": int((~supervised).sum()), "encoder_frame_calls": len(losses),
                 "velocity_batch_calls": computed_count, "target_state_evaluations": computed_count,
                 "nonfinite_objective": nonfinite, "nonfinite_aggregation": nonfinite_aggregation,
                 "optimizer_steps": 0,
                 "checkpoint_loaded": False, "scope": "in-memory causal F0 objective; no dataset authenticity certification"}
    if information is not None:
        resources.update(information_acquisition=information.counts(),
                         scope="causal G1 objective with precomputed frozen G0/J; no history/asset authenticity certification")
    return SequenceObjective(aggregate, frame_losses, torch.stack(contexts), supervised, reasons,
                             chunk.label_frames, state.detached(),
                             resources)


def load_registered_chunks(manifest_path, record_index, normalization, *, asset_contract,
                           device="cpu", dtype=torch.float32):
    """Read one registered train NPZ into consecutive original-row chunks.

    Metadata agreement is required but does not establish authentic labels,
    physical accuracy or permission to train. No optimizer/checkpoint operation.
    """
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    gates = validate_manifest(manifest)  # Check all registered group splits first.
    if not gates["training_data_ready"]:
        raise ValueError("registered_training_data_unavailable")
    if not isinstance(record_index, int) or isinstance(record_index, bool) or not 0 <= record_index < len(manifest["records"]):
        raise ValueError("explicit valid registered record index required")
    record = manifest["records"][record_index]
    if record["split"] != "train":
        raise ValueError("only registered train sequences enter this reader")
    extras = ("identity", "camera_ids", "source_kind", "nodes")
    if any(key not in record for key in extras) or not record.get("identity"):
        raise ValueError("record requires identity/PnP/source_kind/nodes")
    if record["source_kind"] not in ("fitted_reference", "independent_reference"):
        raise ValueError("registered train labels require declared reference source")
    if (not isinstance(record["camera_ids"], list) or len(record["camera_ids"]) != 2
            or any(not isinstance(value, str) or not value for value in record["camera_ids"])
            or len(set(record["camera_ids"])) != 2):
        raise ValueError("two distinct explicitly ordered PnP camera IDs required")
    if not isinstance(record["nodes"], int) or isinstance(record["nodes"], bool):
        raise TypeError("explicit integer target topology required")
    topology = Topology(record["nodes"])
    if normalization.topology != topology:
        raise ValueError("registered sequence/statistics topology mismatch")
    normalization.require_training_statistics()
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("float32/64 sequence tensors required")
    body_fields = ("family", "gender", "beta_basis", "beta_dimensions", "rotation_convention", "hand_mean", "dmpl")
    for key in body_fields:
        if key not in asset_contract or asset_contract[key] != manifest[key]:
            raise ValueError(f"registered target asset contract mismatch:{key}")
    source = Path(record["source_path"])
    source = (source if source.is_absolute() else manifest_path.parent/source).resolve()
    observation_fields = ("xy", "confidence", "present", "quality", "geometry", "views", "origins", "directions")
    label_fields = ("rotations", "translation", "beta", "pose_mask", "translation_mask", "beta_mask")
    required = (*observation_fields, *label_fields, "frame_ids", "timestamps", "label_frame_ids", "label_timestamps",
                "rejection_reasons_json", "metadata_json")
    with np.load(source, allow_pickle=False) as archive:
        if missing := set(required)-set(archive.files):
            raise ValueError(f"registered sequence arrays missing:{sorted(missing)}")
        if archive["metadata_json"].shape != () or archive["metadata_json"].dtype.kind not in ("U", "S"):
            raise ValueError("scalar text archive metadata required")
        metadata = json.loads(archive["metadata_json"].item())
        if metadata.get("format") != "flow_next_sequence_v1":
            raise ValueError("unregistered sequence archive format")
        for key in (*body_fields, "label_scope", "noise_source", "truth_status"):
            if metadata.get(key) != manifest[key]:
                raise ValueError(f"archive manifest contract mismatch:{key}")
        for key in ("subject", "original_action", "sequence", "rig", "split", *extras):
            if metadata.get(key) != record[key]:
                raise ValueError(f"archive record binding mismatch:{key}")
        if (metadata.get("camera_roles"), metadata.get("pixel_domain"), metadata.get("units"),
                metadata.get("operator"), metadata.get("coordinate_frame")) != (
                ["LEFT", "RIGHT"], "raw_fisheye", "metres", "COCO17_SURFACE_REGRESSOR_6890", "cam0_LEFT"):
            raise ValueError("explicit raw-fish LEFT/RIGHT surface/cam0/metres contract required")
        if not isinstance(metadata.get("timestamp_kind"), str) or not metadata["timestamp_kind"]:
            raise ValueError("original physical clock provenance required")
        arrays = {name: archive[name].copy() for name in required if name != "metadata_json"}
        view_times = archive["view_timestamps"].copy() if "view_timestamps" in archive else None
    frames = arrays["frame_ids"]
    if frames.ndim != 1 or not len(frames) or not np.issubdtype(frames.dtype, np.integer):
        raise ValueError("nonempty original integer frame IDs required")
    count = len(frames)
    dimensions = {"xy": (2, 17, 2), "confidence": (2, 17), "present": (2, 17), "quality": (2, 17),
                  "geometry": (2, 17), "views": (2,), "origins": (2, 17, 3), "directions": (2, 17, 3),
                  "rotations": (topology.nodes, 3, 3), "translation": (3,), "beta": (10,),
                  "pose_mask": (topology.nodes,), "translation_mask": (), "beta_mask": (),
                  "timestamps": (), "label_frame_ids": (), "label_timestamps": (), "rejection_reasons_json": ()}
    for key, shape in dimensions.items():
        if arrays[key].shape != (count, *shape):
            raise ValueError(f"registered sequence array shape mismatch:{key}")
    for key in ("present", "quality", "geometry", "views", "pose_mask", "translation_mask", "beta_mask"):
        if arrays[key].dtype != np.bool_:
            raise ValueError(f"explicit Boolean source mask required:{key}")
    for key in ("timestamps", "label_timestamps", "xy", "confidence", "origins", "directions", "rotations", "translation", "beta"):
        if arrays[key].dtype.kind != "f":
            raise ValueError(f"explicit floating source array required:{key}")
    if not np.issubdtype(arrays["label_frame_ids"].dtype, np.integer):
        raise ValueError("integer label frame IDs required")
    if not np.array_equal(frames, arrays["label_frame_ids"]) or not np.array_equal(arrays["timestamps"], arrays["label_timestamps"]):
        raise ValueError("original observation/label frame or clock mapping mismatch; no reorder/fill")
    times = arrays["timestamps"]
    if (frames < 0).any() or not np.isfinite(times).all() or not (frames[1:] > frames[:-1]).all() or not (times[1:] > times[:-1]).all():
        raise ValueError("original frames and physical timestamps must strictly increase")
    if view_times is not None and (view_times.shape != (count, 2) or view_times.dtype.kind != "f" or not np.isfinite(view_times).all()):
        raise ValueError("optional two-view source clocks must be finite and aligned")

    def tensor(key, index=None):
        value = arrays[key] if index is None else arrays[key][index]
        return torch.as_tensor(value, device=device, dtype=torch.bool if arrays[key].dtype == np.bool_ else dtype)

    camera_ids = tuple(record["camera_ids"])
    observations, bindings = [], []
    for index, frame in enumerate(frames):
        raw_reason = arrays["rejection_reasons_json"][index]
        if arrays["rejection_reasons_json"].dtype.kind not in ("U", "S"):
            raise ValueError("original rejection reasons must be JSON text")
        reason = json.loads(raw_reason.item() if hasattr(raw_reason, "item") else raw_reason)
        if (not isinstance(reason, list) or len(reason) != 2 or any(not isinstance(row, list) or len(row) != 17
                or any(not isinstance(item, str) for item in row) for row in reason)):
            raise ValueError("retain2x17 original string rejection reasons")
        obs = Observation(**{name: tensor(name, index) for name in observation_fields},
                          rejection_reasons=tuple(tuple(row) for row in reason), camera_ids=camera_ids,
                          identity=record["identity"], sequence=record["sequence"], frame=int(frame),
                          timestamp=float(times[index]), source=str(source), timestamp_kind=metadata["timestamp_kind"],
                          view_timestamps=tuple(float(v) for v in view_times[index]) if view_times is not None else None).validate()
        observations.append(obs)
        bindings.append(LabelFrame(obs.identity, obs.sequence, int(arrays["label_frame_ids"][index]),
                                   float(arrays["label_timestamps"][index]), camera_ids))
    labels = Labels(**{name: tensor(name) for name in label_fields}, topology=topology,
                    source_kind=record["source_kind"], split="train")
    labels.packed(require_complete=False)
    chunks = []
    for start in range(0, count, configuration()["unroll"]):
        stop = min(count, start+configuration()["unroll"])
        current = replace(labels, **{name: getattr(labels, name)[start:stop] for name in label_fields})
        chunks.append(SequenceChunk(tuple(observations[start:stop]), current, tuple(bindings[start:stop])))
    return tuple(chunks), {"manifest_path": str(manifest_path), "source_path": str(source), "record_index": record_index,
                           "original_frames": count, "original_frame_ids": frames.tolist(), "chunks": len(chunks),
                           "subject": record["subject"], "original_action": record["original_action"],
                           "sequence": record["sequence"], "rig": record["rig"], "camera_ids": list(camera_ids),
                           "source_kind": record["source_kind"], "split": "train", "timestamp_kind": metadata["timestamp_kind"],
                           "asset_contract": {key: metadata[key] for key in body_fields},
                           "declared_training_data_ready": gates["training_data_ready"], "authenticity_verified": False,
                           "training_executed": False, "checkpoint_loaded": False,
                           "boundary": "registered declaration/file consistency; not independent label, asset-basis or physical clock authenticity certification"}


@dataclass
class SequenceBatchObjective:
    loss: torch.Tensor | None
    chunks: tuple[SequenceObjective, ...]
    states: dict
    resources: dict


def sequence_batch_objective(mode, model, chunks, normalization, states=None, *, generator=None, information=None):
    """Bounded original-window targets; no optimizer, checkpoint or stream shuffle.

    Carries are per identity/sequence/PnP stream, even when windows interleave.
    Means are weighted by supervised original frames, never by window count.
    """
    if mode not in ("CFM", "regression"):
        raise ValueError("unregistered batch objective")
    if not 1 <= len(chunks) <= configuration()["effective_batch"]:
        raise ValueError("one to16 original causal chunks required per prepared batch")
    states = {} if states is None else dict(states)
    for key, state in states.items():
        if not isinstance(state, TemporalState):
            raise TypeError("typed per-stream causal carries required")
        if key != (state.identity, state.sequence, tuple(state.camera_ids)):
            raise ValueError("batch carry key differs from its declared stream")
        state.validate(model.encoder.width)
    if information is not None:
        if len(information) != len(chunks) or any(not isinstance(item, SequenceInformation) for item in information):
            raise ValueError("one typed information bundle per original causal chunk required")
        if hasattr(model, "rotation_representation"):
            raise ValueError("SO3 F0 does not define batch G1 information conditioning")
    if hasattr(model, "validate_normalization"):
        model.validate_normalization(normalization)
    previous, keys = dict(states), []
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, SequenceChunk):
            raise TypeError("typed original SequenceChunks required")
        chunk.validate(model.topology, normalization)
        first = chunk.observations[0]
        key = (first.identity, first.sequence, tuple(first.camera_ids))
        if key in previous:
            last = previous[key]
            frame = last.last_frame if isinstance(last, TemporalState) else last.frame
            timestamp = last.last_timestamp if isinstance(last, TemporalState) else last.timestamp
            if first.frame <= frame or first.timestamp <= timestamp:
                raise ValueError("original batch windows must increase within each stream")
        if information is not None:
            information[index].validate(chunk.observations, model.topology)
        previous[key] = chunk.observations[-1]
        keys.append(key)
    # No forward occurs before the entire batch's original-row contracts pass.
    states = {key: state.detached() for key, state in states.items()}
    results = []
    for index, (key, chunk) in enumerate(zip(keys, chunks)):
        result = sequence_objective(mode, model, chunk, normalization, states.get(key), generator=generator,
                                    information=None if information is None else information[index])
        states[key] = result.state
        results.append(result)
    nonfinite = any(result.resources["nonfinite_objective"] for result in results)
    aggregate, nonfinite_aggregation = None, False
    supported = [result for result in results if result.loss is not None]
    if supported and not nonfinite:
        aggregate, nonfinite_aggregation = _reduce_objectives(
            torch.stack([result.loss for result in supported]),
            [result.resources["supervised_count"] for result in supported])
    nonfinite = nonfinite or nonfinite_aggregation
    summed = ("original_frames", "supervised_count", "skipped_count", "encoder_frame_calls",
              "velocity_batch_calls", "target_state_evaluations")
    resources = {key: sum(result.resources[key] for result in results) for key in summed}
    resources.update(streams=len(set(keys)), chunks=len(chunks), nonfinite_objective=nonfinite,
                     nonfinite_aggregation=nonfinite_aggregation,
                     aggregation="supervised_original_frame_weighted", optimizer_steps=0, checkpoint_loaded=False,
                     scope="bounded causal batch targets; not a trained runner or source authenticity certificate")
    if information is not None:
        records = [result.resources["information_acquisition"] for result in results]
        resources["information_acquisition"] = {key: sum(record[key] for record in records)
                                                 for key in records[0] if key != "computed_in_this_objective"}
        resources["information_acquisition"]["computed_in_this_objective"] = False
    return SequenceBatchObjective(aggregate, tuple(results), states, resources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--nodes", type=int, choices=(22, 52), default=52)
    args = parser.parse_args()
    torch.manual_seed(7)
    print(json.dumps(prepare(json.loads(args.manifest.read_text(encoding="utf-8")), args.nodes), indent=2))


if __name__ == "__main__":
    main()
