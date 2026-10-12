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
    split: str = "legacy_unverified"
    source_kind: str = "legacy_unverified"

    def validate(self):
        shape = (self.topology.dimensions,)
        if self.mean.shape != shape or self.scale.shape != shape:
            raise ValueError("normalization dimension mismatch")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("finite positive normalization required")
        if self.mean.dtype != self.scale.dtype or self.mean.dtype not in (torch.float32, torch.float64):
            raise ValueError("matching float32/float64 statistics dtype required")
        if self.mean.requires_grad or self.scale.requires_grad:
            raise ValueError("normalization statistics must be frozen")
        origins = {"train": ("fitted_reference", "independent_reference"),
                   "engineering_fixture": ("analytic_fixture",),
                   "legacy_unverified": ("legacy_unverified",)}
        if not self.source or self.source_kind not in origins.get(self.split, ()):
            raise ValueError("normalization provenance requires train, explicit fixture or legacy-unverified origin")
        return self

    def require_training_statistics(self):
        """Check declared origin only; authentic data/splits need loader evidence."""
        self.validate()
        if self.split != "train":
            raise ValueError("declared training statistics required; fixture/legacy origin is insufficient")
        return self

    @classmethod
    def fit(cls, labels, *, source_path):
        if labels.split != "train" or labels.source_kind == "analytic_fixture":
            raise ValueError("representation statistics require declared training data")
        values, _ = labels.packed()
        return cls(values.mean(0).detach(), values.std(0, unbiased=False).clamp_min(1e-4).detach(),
                   labels.topology, str(source_path), split="train", source_kind=labels.source_kind).validate()

    @classmethod
    def fixture(cls, topology):
        return cls(torch.zeros(topology.dimensions), torch.ones(topology.dimensions),
                   topology, "analytic_fixture_only_not_training_statistics",
                   split="engineering_fixture", source_kind="analytic_fixture")

    def normalize(self, value):
        return (value - self.mean.to(value)) / self.scale.to(value)

    def denormalize(self, value):
        return value * self.scale.to(value) + self.mean.to(value)

    def to_dict(self):
        self.validate()
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "nodes": self.topology.nodes, "source": self.source,
                "split": self.split, "source_kind": self.source_kind,
                "dtype": str(self.mean.dtype).removeprefix("torch.")}

    @classmethod
    def from_dict(cls, value):
        dtypes = {"float32": torch.float32, "float64": torch.float64}
        if value.get("dtype", "float32") not in dtypes:
            raise ValueError("unregistered normalization statistics dtype")
        dtype = dtypes[value.get("dtype", "float32")]
        return cls(torch.tensor(value["mean"], dtype=dtype), torch.tensor(value["scale"], dtype=dtype),
                   Topology(value["nodes"]), value["source"],
                   split=value.get("split", "legacy_unverified"),
                   source_kind=value.get("source_kind", "legacy_unverified")).validate()
