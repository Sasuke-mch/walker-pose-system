"""Typed deployment inputs, with no labels or arbitrary extra fields."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Topology:
    nodes: int = 52

    def __post_init__(self):
        if self.nodes not in (22, 52):
            raise ValueError("only SMPLH52 or explicitly partial BODY22 supported")

    @property
    def name(self):
        return "SMPLH52" if self.nodes == 52 else "BODY22_partial_no_hand_claim"

    @property
    def dimensions(self):
        return self.nodes * 6 + 13

    @property
    def tangent_dimensions(self):
        return self.nodes * 3 + 13


@dataclass(frozen=True)
class Observation:
    # Single physical frame, V=2 and COCO17. All source masks remain distinct.
    xy: torch.Tensor
    confidence: torch.Tensor
    present: torch.Tensor
    quality: torch.Tensor
    geometry: torch.Tensor
    views: torch.Tensor
    origins: torch.Tensor
    directions: torch.Tensor
    rejection_reasons: tuple
    camera_ids: tuple
    identity: str
    sequence: str
    frame: int
    timestamp: float
    source: str
    pixel_domain: str = "raw_fisheye"
    units: str = "metres"
    operator: str = "COCO17_SURFACE_REGRESSOR_6890"
    timestamp_kind: str = "analytic_fixture_clock"
    view_timestamps: tuple | None = None

    def admitted(self):
        return self.present & self.quality & self.views[:, None]

    def validate(self):
        if self.xy.shape != (2, 17, 2):
            raise ValueError("expected two raw fisheye COCO17 views")
        for name in ("present", "quality", "geometry"):
            value = getattr(self, name)
            if value.shape != (2, 17) or value.dtype != torch.bool:
                raise ValueError(f"invalid source mask {name}")
        if self.views.shape != (2,) or self.views.dtype != torch.bool:
            raise ValueError("invalid view mask")
        if self.confidence.shape != (2, 17):
            raise ValueError("invalid confidence shape")
        if self.origins.shape != (2, 17, 3) or self.directions.shape != (2, 17, 3):
            raise ValueError("invalid ray shape")
        if len(self.camera_ids) != 2 or len(set(self.camera_ids)) != 2:
            raise ValueError("distinct PnP instance IDs required in LEFT, RIGHT order")
        if len(self.rejection_reasons) != 2 or any(len(r) != 17 for r in self.rejection_reasons):
            raise ValueError("retain all per-view per-joint rejection reasons")
        if (self.pixel_domain, self.units, self.operator) != (
            "raw_fisheye", "metres", "COCO17_SURFACE_REGRESSOR_6890"
        ):
            raise ValueError("pixel domain, units or surface operator mismatch")
        if not self.identity or not self.sequence or not self.source or self.frame < 0:
            raise ValueError("provenance and nonnegative absolute frame ID required")
        if not torch.isfinite(torch.tensor(self.timestamp)):
            raise ValueError("finite physical timestamp required")
        if not self.timestamp_kind:
            raise ValueError("physical clock provenance required")
        if self.view_timestamps is not None and (len(self.view_timestamps) != 2 or not
                all(torch.isfinite(torch.tensor(v)) for v in self.view_timestamps)):
            raise ValueError("two finite source-view timestamps required")
        # Invalid missing slots may contain NaN; never sanitize present slots.
        for value in (self.xy[self.present], self.confidence[self.present]):
            if not torch.isfinite(value).all():
                raise ValueError("nonfinite present observation")
        scores = self.confidence[self.present]
        if ((scores < 0) | (scores > 1)).any():
            raise ValueError("confidence outside [0,1]")
        rays = self.geometry & self.admitted()
        if not torch.isfinite(self.origins[rays]).all() or not torch.isfinite(self.directions[rays]).all():
            raise ValueError("nonfinite admitted ray")
        if rays.any() and (self.directions[rays].norm(dim=-1) - 1).abs().max() > 1e-3:
            raise ValueError("admitted ray directions must be unit vectors")
        return self


@dataclass
class TemporalState:
    identity: str
    sequence: str
    camera_ids: tuple
    hidden: torch.Tensor
    age: torch.Tensor
    last_timestamp: float
    last_frame: int
    seen: torch.Tensor

    def detached(self):
        # TBPTT retains values; this is not a reset.
        return TemporalState(self.identity, self.sequence, self.camera_ids,
                             self.hidden.detach(), self.age.detach(),
                             self.last_timestamp, self.last_frame, self.seen.detach())

    def to_dict(self):
        return {"identity": self.identity, "sequence": self.sequence,
                "camera_ids": list(self.camera_ids), "hidden": self.hidden.detach().cpu().tolist(),
                "age": self.age.detach().cpu().tolist(), "last_timestamp": self.last_timestamp,
                "last_frame": self.last_frame, "seen": self.seen.cpu().tolist()}

    @classmethod
    def from_dict(cls, value, *, device="cpu", dtype=torch.float32):
        return cls(value["identity"], value["sequence"], tuple(value["camera_ids"]),
                   torch.tensor(value["hidden"], device=device, dtype=dtype),
                   torch.tensor(value["age"], device=device, dtype=dtype),
                   value["last_timestamp"], value["last_frame"],
                   torch.tensor(value["seen"], device=device, dtype=torch.bool))
