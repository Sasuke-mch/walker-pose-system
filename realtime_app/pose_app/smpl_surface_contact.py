"""Differentiable SMPL surface-to-entity contact distances.

All contact geometry here uses SMPL surface vertices transformed into the
current ground frame. COCO ankle/wrist joints must never enter the surface
contact loss; they remain restricted to observation losses and diagnostics.
"""
from __future__ import annotations

import math
from typing import Any


def _torch() -> Any:
    import torch

    return torch


def signed_ground_distances(vertices_ground_m: "Any") -> "Any":
    """Return z_G for every selected surface vertex, shape [B,K]."""
    return vertices_ground_m[..., 2]


def point_segment_distance(points: "Any", a: "Any", b: "Any") -> "Any":
    """Return Euclidean distance to capsule centerline, shape [B,K]."""
    torch = _torch()
    ab = b - a
    den = (ab * ab).sum(-1, keepdim=True).clamp_min(1e-8)
    u = ((points - a) * ab).sum(-1, keepdim=True) / den
    u = u.clamp(0.0, 1.0)
    return torch.linalg.vector_norm(points - (a + u * ab), dim=-1)


def capsule_surface_residual(points: "Any", a: "Any", b: "Any", radius_m: float) -> "Any":
    """Return distance-to-capsule-surface residual, shape [B,K]."""
    if not radius_m > 0:
        raise ValueError(f"capsule radius must be positive, got {radius_m}")
    return point_segment_distance(points, a, b) - radius_m


def robust_scalar(x: "Any", delta: float) -> "Any":
    """Elementwise pseudo-Huber; never reduce the K dimension."""
    torch = _torch()
    return delta * delta * (torch.sqrt(1.0 + (x / delta) ** 2) - 1.0)


def softmin(values: "Any", temperature: float) -> "Any":
    """Differentiable soft minimum over surface vertices, shape [B]."""
    torch = _torch()
    if not temperature > 0:
        raise ValueError(f"softmin temperature must be positive, got {temperature}")
    return -temperature * torch.logsumexp(-values / temperature, dim=-1)


def centered_softmin(values: "Any", temperature: float) -> "Any":
    """Soft minimum corrected for the log(K) offset of K candidates."""
    torch = _torch()
    if values.shape[-1] <= 0:
        raise ValueError("centered_softmin requires a non-empty candidate dimension")
    if not temperature > 0:
        raise ValueError(f"softmin temperature must be positive, got {temperature}")
    k = values.shape[-1]
    return -temperature * (
        torch.logsumexp(-values / temperature, dim=-1) - math.log(float(k))
    )


def foot_surface_loss(sole_z: "Any", temperature: float = 0.005,
                      delta: float = 0.010,
                      penetration_weight: float = 1.0) -> "Any":
    """Sole loss without forcing every surface vertex to z=0.

    ``sole_z`` is z_G of one foot surface set, shape [B,K]. Uses the
    log(K)-centered soft minimum and a relu penetration penalty.
    """
    torch = _torch()
    foot_base = robust_scalar(centered_softmin(sole_z, temperature), delta)
    foot_penetration = torch.relu(-sole_z).square().mean(dim=-1)
    return foot_base + penetration_weight * foot_penetration


def hand_surface_loss(palm_points: "Any", a: "Any", b: "Any",
                      radius_m: float, delta: float = 0.015,
                      temperature: float = 0.005,
                      penetration_weight: float = 1.0) -> "Any":
    """Wrap-the-handle surface loss for one hand set, shape [B].

    Uses the log(K)-centered soft minimum of capsule-surface residuals and an
    explicit relu penetration penalty.
    """
    torch = _torch()
    residuals = capsule_surface_residual(palm_points, a, b, radius_m)
    hand_base = robust_scalar(centered_softmin(residuals, temperature), delta)
    hand_penetration = torch.relu(-residuals).square().mean(dim=-1)
    return hand_base + penetration_weight * hand_penetration


def surface_coverage(distances_m: "Any", thresholds_m: tuple = (0.015, 0.030, 0.050)) -> dict:
    """Fraction of surface vertices within each capsule-surface distance."""
    out = {}
    for threshold in thresholds_m:
        out[f"within_{int(round(threshold * 1000))}mm"] = float(
            (distances_m < threshold).float().mean().detach().cpu()
        )
    return out


__all__ = [
    "signed_ground_distances",
    "point_segment_distance",
    "capsule_surface_residual",
    "robust_scalar",
    "softmin",
    "centered_softmin",
    "foot_surface_loss",
    "hand_surface_loss",
    "surface_coverage",
]
