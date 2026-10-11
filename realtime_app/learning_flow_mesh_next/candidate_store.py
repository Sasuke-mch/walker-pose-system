"""Lossless fixed candidate caches; no resampling, interpolation or beta stitching."""
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from realtime_app.learning_flow_mesh.rotations import six_to_matrix

from .schema import Observation, Topology


def nested_noise(obs, k, topology, *, stream="G0", device="cpu", dtype=torch.float32):
    if k < 1:
        raise ValueError("positive candidate budget required")
    samples, seeds = [], []
    for i in range(k):
        key = json.dumps([obs.identity, obs.sequence, obs.frame, stream, i], separators=(",", ":"))
        seed = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little") % (2**63-1)
        rng = torch.Generator(device=device).manual_seed(seed)
        samples.append(torch.randn(topology.dimensions, generator=rng, device=device, dtype=dtype))
        seeds.append(seed)
    return torch.stack(samples), seeds


@dataclass
class CandidateStore:
    raw: torch.Tensor
    rotations: torch.Tensor
    translation: torch.Tensor
    beta: torch.Tensor
    numerical: torch.Tensor
    actions: torch.Tensor
    reasons: list
    ids: list
    seeds: list
    topology: Topology
    provenance: dict
    resources: dict

    def select(self, index):
        if not self.actions[index]:
            raise ValueError("candidate is not an allowed action")
        return {"id": self.ids[index], "rotations": self.rotations[index],
                "translation": self.translation[index], "beta": self.beta[index]}

    def permuted(self, order):
        indices = order.tolist()
        return CandidateStore(*(getattr(self, k)[order] for k in
                                ("raw", "rotations", "translation", "beta", "numerical", "actions")),
                              [self.reasons[i] for i in indices], [self.ids[i] for i in indices],
                              [self.seeds[i] for i in indices], self.topology, self.provenance, self.resources)

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError("candidate caches are immutable; use a new run path")
        meta = {"reasons": self.reasons, "ids": self.ids, "seeds": self.seeds,
                "nodes": self.topology.nodes, "provenance": self.provenance,
                "resources": self.resources, "weights": "raw_equal_weight_samples"}
        arrays = {key: getattr(self, key).detach().cpu().numpy() for key in
                  ("raw", "rotations", "translation", "beta", "numerical", "actions")}
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive create guards concurrent writers as well as existing caches.
        with path.open("xb") as handle:
            np.savez_compressed(handle, **arrays, metadata=json.dumps(meta))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"].item()))
            values = [torch.from_numpy(archive[key].copy()) for key in
                      ("raw", "rotations", "translation", "beta", "numerical", "actions")]
        return cls(*values, meta["reasons"], meta["ids"], meta["seeds"], Topology(meta["nodes"]),
                   meta["provenance"], meta["resources"])


def from_parameters(raw, topology, *, ids, seeds, current_observed, provenance, resources):
    n = topology.nodes
    if raw.ndim != 2 or raw.shape[1] != topology.dimensions:
        raise ValueError("joint candidate dimensions mismatch")
    k = len(raw)
    if len(ids) != k or len(set(ids)) != k or len(seeds) != k:
        raise ValueError("unique candidate IDs and all seeds required")
    rotations, valid = six_to_matrix(raw[:, :n*6].reshape(k, n, 6))
    translation, beta = raw[:, n*6:n*6+3], raw[:, -10:]
    numerical = valid.all(-1) & torch.isfinite(translation).all(-1) & torch.isfinite(beta).all(-1)
    reasons = []
    for i in range(k):
        rejected = []
        if not valid[i].all():
            rejected.append("invalid_rotation")
        if not torch.isfinite(raw[i]).all():
            rejected.append("nonfinite_joint_state")
        if not current_observed:
            rejected.append("no_current_observation")
        reasons.append(rejected)
    return CandidateStore(raw, rotations, translation, beta, numerical,
                          numerical & current_observed, reasons, ids, seeds, topology, provenance, resources)


def infer(model, normalization, observation: Observation, state=None, *, k=32, steps=8,
          information=None, stream="G0", regression=False):
    if normalization.topology != model.topology:
        raise ValueError("inference normalization topology mismatch")
    start = time.perf_counter()
    if observation.xy.is_cuda:
        torch.cuda.synchronize()
        start = time.perf_counter()
    context, state_out, observed = model.encoder(observation, state)
    encoded_at = time.perf_counter()
    noise, seeds = nested_noise(observation, k, model.topology, stream=stream,
                               device=context.device, dtype=context.dtype)
    if regression:
        if k != 1 or information is not None:
            raise ValueError("regression is a single deterministic control")
        normalized = model.regression(context[None])
        calls = 1
    else:
        normalized = model.sample(context, noise, steps=steps, information=information)
        calls = steps
    parameters = normalization.denormalize(normalized)
    if parameters.is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    resources = {"network_calls": calls, "candidate_state_evaluations": calls*k,
                 "encoder_ms": (encoded_at-start)*1000, "total_ms": elapsed*1000,
                 "solver": "regression" if regression else "Euler", "k": k,
                 "steps": calls, "tta": False}
    provenance = {"identity": observation.identity, "sequence": observation.sequence,
                  "frame": observation.frame, "timestamp": observation.timestamp,
                  "camera_ids": list(observation.camera_ids), "source": observation.source,
                  "rejection_reasons": observation.rejection_reasons,
                  "source_masks": {key: getattr(observation, key).cpu().tolist() for key in
                                   ("present", "quality", "geometry", "views")},
                  "normalization_source": normalization.source, "stream": stream}
    candidates = from_parameters(parameters, model.topology,
                                 ids=[f"{stream}:{observation.frame}:{i}" for i in range(k)],
                                 seeds=seeds, current_observed=observed, provenance=provenance, resources=resources)
    return candidates, state_out
