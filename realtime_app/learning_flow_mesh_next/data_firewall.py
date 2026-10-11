"""Model-family, split and label gates; labels never enter Observation."""
from dataclasses import dataclass

import torch

from realtime_app.learning_flow_mesh.rotations import (
    matrix_to_six,
    validate_rotation_matrices,
)

from .schema import Topology


def validate_manifest(manifest):
    required = ("source_path", "license_status", "family", "gender", "beta_basis",
                "beta_dimensions", "rotation_convention", "hand_mean", "dmpl",
                "label_scope", "noise_source", "records", "truth_status")
    if missing := set(required) - manifest.keys():
        raise ValueError(f"missing manifest keys: {sorted(missing)}")
    if manifest["family"] != "SMPL-H" or manifest["beta_dimensions"] != 10:
        raise ValueError("no SMPL-X beta copying or unverified model conversion")
    if manifest["gender"] not in ("male", "female", "neutral"):
        raise ValueError("undeclared gender")
    if manifest["rotation_convention"] != "local_matrix_packed_columns6d":
        raise ValueError("rotation convention mismatch")
    if manifest["dmpl"] != "excluded_explicitly":
        raise ValueError("DMPL cannot silently become shape")
    if not manifest["beta_basis"] or manifest["hand_mean"] not in ("flat", "asset_mean"):
        raise ValueError("beta basis and hand mean must be explicit")
    if manifest["license_status"] not in ("locally_authorized", "missing"):
        raise ValueError("unverified license status")
    records = manifest["records"]
    owners = {k: {} for k in ("subject", "original_action", "sequence")}
    for record in records:
        split = record["split"]
        if split not in ("train", "noise_calibration", "validation", "test", "engineering_replay"):
            raise ValueError("undeclared split")
        for key, seen in owners.items():
            identity = record[key]
            if not identity or (identity in seen and seen[identity] != split):
                raise ValueError(f"pre-augmentation leakage in {key}")
            seen[identity] = split
        if not record["rig"] or not record["source_path"]:
            raise ValueError("rig and original source required")
    return {"training_data_ready": bool(records) and manifest["license_status"] == "locally_authorized"
            and any(r["split"] == "train" for r in records),
            "truth_status": manifest["truth_status"], "records": len(records)}


@dataclass(frozen=True)
class Labels:
    rotations: torch.Tensor
    translation: torch.Tensor
    beta: torch.Tensor
    pose_mask: torch.Tensor
    translation_mask: torch.Tensor
    beta_mask: torch.Tensor
    topology: Topology
    source_kind: str
    split: str

    def packed(self, *, require_complete=True):
        n = self.topology.nodes
        batch = self.rotations.shape[0]
        if self.rotations.shape != (batch, n, 3, 3):
            raise ValueError("explicit topology mismatch")
        if self.translation.shape != (batch, 3) or self.beta.shape != (batch, 10):
            raise ValueError("joint labels must contain T3 and beta10")
        for mask, shape in ((self.pose_mask, (batch, n)), (self.translation_mask, (batch,)),
                            (self.beta_mask, (batch,))):
            if mask.dtype != torch.bool or mask.shape != shape:
                raise ValueError("label masks must be explicit booleans")
        if self.source_kind not in ("fitted_reference", "independent_reference", "analytic_fixture"):
            raise ValueError("label credibility must be declared")
        if self.split not in ("train", "validation", "test", "engineering_fixture"):
            raise ValueError("invalid label split")
        if self.pose_mask.any():
            validate_rotation_matrices(self.rotations[self.pose_mask])
        for value, mask in ((self.translation, self.translation_mask), (self.beta, self.beta_mask)):
            if not torch.isfinite(value[mask]).all():
                raise ValueError("nonfinite trusted labels")
        mask = torch.cat((self.pose_mask.repeat_interleave(6, -1),
                          self.translation_mask[:, None].expand(-1, 3),
                          self.beta_mask[:, None].expand(-1, 10)), -1)
        if require_complete and not mask.all():
            raise ValueError("incomplete labels cannot certify the full joint distribution")
        values = torch.cat((matrix_to_six(self.rotations).flatten(1), self.translation, self.beta), -1)
        # Unknown labels remain NaN; no unit hand or zero shape truth fabricated.
        return torch.where(mask, values, torch.nan), mask


@dataclass(frozen=True)
class Normalization:
    mean: torch.Tensor
    scale: torch.Tensor
    topology: Topology
    source: str

    def validate(self):
        shape = (self.topology.dimensions,)
        if self.mean.shape != shape or self.scale.shape != shape:
            raise ValueError("normalization dimension mismatch")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("finite positive normalization required")
        return self

    @classmethod
    def fit(cls, labels, *, source_path):
        if labels.split != "train" or labels.source_kind == "analytic_fixture":
            raise ValueError("representation statistics require declared training data")
        values, _ = labels.packed()
        return cls(values.mean(0), values.std(0, unbiased=False).clamp_min(1e-4),
                   labels.topology, str(source_path)).validate()

    @classmethod
    def fixture(cls, topology):
        return cls(torch.zeros(topology.dimensions), torch.ones(topology.dimensions),
                   topology, "analytic_fixture_only_not_training_statistics")

    def normalize(self, value):
        return (value - self.mean.to(value)) / self.scale.to(value)

    def denormalize(self, value):
        return value * self.scale.to(value) + self.mean.to(value)

    def to_dict(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "nodes": self.topology.nodes, "source": self.source}

    @classmethod
    def from_dict(cls, value):
        return cls(torch.tensor(value["mean"]), torch.tensor(value["scale"]),
                   Topology(value["nodes"]), value["source"]).validate()
