"""Body-local COCO-to-SMPL-X joint observation offsets.

The completed shape/offset gate (G20260918_smplx_shape_offset_gate_v1) showed
that COCO annotation points differ from SMPL-X anatomical joint centres by a
repeatable 25-43 mm body-local offset for shoulder/elbow/wrist/hip joints.
The offset is an *observation model*: it predicts the COCO annotation point
from an SMPL-X joint before comparing with observations. It never modifies the
SMPL-X mesh or skeleton.

Convention (matches analyze_smplx_shape_stability.py):
    offset_local = basis^T (SMPL_joint - COCO_point)
    pred_coco    = SMPL_joint - basis @ offset_local

Only joints flagged `stable_offset_candidate` are applied; other joints keep
the identity observation model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


BODY_JOINTS = tuple(range(5, 17))


def body_basis(points: np.ndarray) -> np.ndarray:
    """Left/up/forward orthonormal body basis from COCO shoulder/hip points."""
    left = points[5] - points[6]
    up = 0.5 * (points[5] + points[6]) - 0.5 * (points[11] + points[12])
    left = left / max(float(np.linalg.norm(left)), 1e-9)
    up = up - left * float(left @ up)
    up = up / max(float(np.linalg.norm(up)), 1e-9)
    forward = np.cross(left, up)
    forward = forward / max(float(np.linalg.norm(forward)), 1e-9)
    return np.column_stack((left, up, forward))


class JointOffsetModel:
    """Applies body-local offsets to SMPL-X joints to predict COCO points."""

    def __init__(self, offsets_mm: dict[int, np.ndarray]) -> None:
        self.offsets_mm = {int(index): np.asarray(value, dtype=np.float64) for index, value in offsets_mm.items()}
        self.joint_indices = sorted(self.offsets_mm)

    @classmethod
    def from_audit_json(cls, audit_path: str | Path) -> "JointOffsetModel":
        audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
        offsets: dict[int, np.ndarray] = {}
        for joint_key, record in audit["joint_offset_audit"].items():
            if record.get("stable_offset_candidate"):
                offsets[int(joint_key)] = np.asarray(record["body_local_median_offset_mm"], dtype=np.float64)
        return cls(offsets)

    @property
    def applied_joint_indices(self) -> list[int]:
        return list(self.joint_indices)

    def predict_coco_numpy(self, smplx_joints_mm: np.ndarray) -> np.ndarray:
        """Predict COCO annotation points (world mm) from SMPL-X joints (world mm).

        The body basis comes from the predicted joints. Using target COCO
        points here would leak a 3-D target into a nominally 2-D-only fit.
        """
        smplx = np.asarray(smplx_joints_mm, dtype=np.float64)
        batched = smplx.ndim == 3
        if smplx.ndim == 2:
            smplx = smplx[None]
        if smplx.shape[1] < 17:
            raise ValueError("expected at least 17 joints")
        predicted = smplx.copy()
        for frame in range(smplx.shape[0]):
            basis = body_basis(smplx[frame])
            for joint in self.joint_indices:
                predicted[frame, joint] = smplx[frame, joint] - basis @ self.offsets_mm[joint]
        return predicted[0] if not batched else predicted

    def predict_coco_torch(self, smplx_joints_world_m: "Any") -> "Any":
        """Torch variant using the predicted torso basis, with no target leakage."""
        import torch

        dtype = smplx_joints_world_m.dtype
        predicted = smplx_joints_world_m.clone()
        left = smplx_joints_world_m[:, 5] - smplx_joints_world_m[:, 6]
        left = left / torch.clamp(torch.linalg.vector_norm(left, dim=1, keepdim=True), min=1e-9)
        up = (
            0.5 * (smplx_joints_world_m[:, 5] + smplx_joints_world_m[:, 6])
            - 0.5 * (smplx_joints_world_m[:, 11] + smplx_joints_world_m[:, 12])
        )
        up = up - left * torch.sum(left * up, dim=1, keepdim=True)
        up = up / torch.clamp(torch.linalg.vector_norm(up, dim=1, keepdim=True), min=1e-9)
        forward = torch.linalg.cross(left, up, dim=1)
        forward = forward / torch.clamp(
            torch.linalg.vector_norm(forward, dim=1, keepdim=True), min=1e-9
        )
        bases = torch.stack((left, up, forward), dim=2)
        for joint in self.joint_indices:
            offset = torch.as_tensor(
                self.offsets_mm[joint] / 1000.0, dtype=dtype, device=smplx_joints_world_m.device
            )
            predicted[:, joint] = smplx_joints_world_m[:, joint] - torch.einsum("nij,j->ni", bases, offset)
        return predicted
