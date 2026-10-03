"""Rigid shoulder/hip initialization, before the first optimization stage."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from ..smpl_coco_observation import regress_coco17_torch


@dataclass(frozen=True)
class RigidInitialization:
    root_init_np: np.ndarray
    root_init_source: np.ndarray
    init_source: list[str]
    body_bones: tuple[tuple[int, int], ...]
    ref_bone_lengths: np.ndarray
    root_anchor_target: torch.Tensor
    init_body_rms_mm: float


def make_body_basis(points: np.ndarray) -> np.ndarray | None:
    p = np.asarray(points, dtype=np.float64)
    needed = (5, 6, 11, 12)
    if not np.isfinite(p[list(needed)]).all():
        return None
    left = p[5] - p[6]
    up = 0.5 * (p[5] + p[6]) - 0.5 * (p[11] + p[12])
    nl = np.linalg.norm(left)
    if nl < 1e-8:
        return None
    left = left / nl
    up = up - left * float(left @ up)
    nu = np.linalg.norm(up)
    if nu < 1e-8:
        return None
    up = up / nu
    forward = np.cross(left, up)
    nf = np.linalg.norm(forward)
    if nf < 1e-8:
        return None
    forward = forward / nf
    basis = np.column_stack((left, up, forward))
    if np.linalg.det(basis) < 0:
        basis[:, 2] *= -1.0
    return basis


def initialize_rigid_body(
    *,
    model,
    reg,
    decode_body_pose,
    n,
    latent_dim,
    device,
    tri_m,
    body_mask,
    root,
    transl,
):
    """Initialize root/translation in place and return the unchanged gate diagnostics."""
    # Decode the VPoser mean and use it for the template basis.  A zero axis
    # angle is not treated as an observation-based body orientation.
    with torch.no_grad():
        template_latent = torch.zeros(n, latent_dim, device=device)
        template_body_pose = decode_body_pose(template_latent)
        template = model(
            betas=torch.zeros(n, 10, device=device),
            global_orient=torch.zeros(n, 3, device=device),
            body_pose=template_body_pose,
            left_hand_pose=torch.zeros(n, 45, device=device),
            right_hand_pose=torch.zeros(n, 45, device=device),
            transl=torch.zeros(n, 3, device=device),
            return_verts=True,
        )
        template_coco = (
            regress_coco17_torch(template.vertices, reg).detach().cpu().numpy()
        )
        template_joints = template.joints.detach().cpu().numpy()
    model_basis = make_body_basis(template_coco[0])
    if model_basis is None:
        raise RuntimeError(
            "cannot construct a valid SMPL-H template shoulder/hip basis"
        )
    root_init_np = np.zeros((n, 3), dtype=np.float32)
    root_init_source = np.full(n, "basis_unavailable", dtype="U32")
    from scipy.spatial.transform import Rotation

    for i in range(n):
        observed_basis = (
            make_body_basis(tri_m[i]) if body_mask[i, [5, 6, 11, 12]].all() else None
        )
        if observed_basis is None:
            root_init_np[i] = np.zeros(3, dtype=np.float32)
            continue
        global_rotation = observed_basis @ model_basis.T
        if (
            not np.isfinite(global_rotation).all()
            or np.linalg.det(global_rotation) <= 0
        ):
            continue
        root_init_np[i] = (
            Rotation.from_matrix(global_rotation).as_rotvec().astype(np.float32)
        )
        root_init_source[i] = "shoulder_hip_rigid_basis"

    # Translation is computed from the same rotated template used for root
    # initialization.  This prevents a zero-rotation translation from being
    # paired with a later nonzero global orientation.
    with torch.no_grad():
        root_init_t = torch.tensor(root_init_np, dtype=torch.float32, device=device)
        rotated_template = model(
            betas=torch.zeros(n, 10, device=device),
            global_orient=root_init_t,
            body_pose=template_body_pose,
            left_hand_pose=torch.zeros(n, 45, device=device),
            right_hand_pose=torch.zeros(n, 45, device=device),
            transl=torch.zeros(n, 3, device=device),
            return_verts=True,
        )
        rotated_template_coco = regress_coco17_torch(rotated_template.vertices, reg)
        rotated_template_np = rotated_template_coco.detach().cpu().numpy()
    init_source = []
    t0 = np.zeros((n, 3), np.float32)
    for i in range(n):
        hl = (
            tri_m[i, 11]
            if np.isfinite(tri_m[i, 11]).all() and body_mask[i, 11]
            else None
        )
        hr = (
            tri_m[i, 12]
            if np.isfinite(tri_m[i, 12]).all() and body_mask[i, 12]
            else None
        )
        if hl is not None and hr is not None:
            t0[i] = (hl + hr) / 2 - (
                rotated_template_np[i, 11] + rotated_template_np[i, 12]
            ) / 2
            init_source.append("rotated_template_hips_midpoint")
        elif hl is not None:
            t0[i] = hl - rotated_template_np[i, 11]
            init_source.append("rotated_template_left_hip")
        elif hr is not None:
            t0[i] = hr - rotated_template_np[i, 12]
            init_source.append("rotated_template_right_hip")
        else:
            t0[i] = np.zeros(3, np.float32)
            init_source.append("translation_init_unavailable")
    with torch.no_grad():
        root[:] = root_init_t
        transl[:] = torch.tensor(t0, device=device)

    # SMPL-H supplies the reference lengths.  The triangulated skeleton is
    # not used as a bone-length truth source.
    body_bones = (
        (16, 17),
        (1, 2),
        (16, 18),
        (18, 20),
        (17, 19),
        (19, 21),
        (1, 4),
        (4, 7),
        (2, 5),
        (5, 8),
    )
    ref_bone_lengths = np.asarray(
        [
            np.linalg.norm(template_joints[0, a] - template_joints[0, b])
            for a, b in body_bones
        ],
        dtype=np.float32,
    )
    root_anchor_target = root_init_t.detach().clone()
    init_coco_after_translation = rotated_template_np + t0[:, None, :]
    init_res = init_coco_after_translation - tri_m
    init_valid = body_mask & np.isfinite(init_res).all(axis=-1)
    init_body_rms_mm = (
        float(np.sqrt(np.mean(np.sum(init_res[init_valid] ** 2, axis=-1))) * 1000.0)
        if init_valid.any()
        else float("nan")
    )
    return RigidInitialization(
        root_init_np,
        root_init_source,
        init_source,
        body_bones,
        ref_bone_lengths,
        root_anchor_target,
        init_body_rms_mm,
    )
