"""Reusable, opt-in body refinement primitives; no changes to legacy fitting.

Targets are designed walker-local poses, not observed physical truth. Fingers
and beta stay fixed. Body joint updates are composed on SO(3), not added to
axis angles. Surface contact always consumes actual model vertices.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from smplx.lbs import batch_rodrigues

from . import smpl_surface_contact as surface

UPPER_JOINTS = (13, 14, 16, 17, 18, 19, 20, 21)
TORSO_JOINTS = (3, 6, 9, 12)


def screened_triangle_pairs(vertices, faces):
    """Vectorized equivalent of the existing noncoplanar triangle screen.

    Shared vertices, coplanar overlap and tangencies retain the old exclusion.
    Vectorize candidate edge tests, not the model geometry or frame sequence.
    """
    tri = np.asarray(vertices)[faces]
    lo, hi = tri.min(1), tri.max(1)
    candidate = np.triu(np.ones((len(faces), len(faces)), bool), 1)
    for axis in range(3):
        candidate &= (lo[:, axis, None] <= hi[None, :, axis]) & (hi[:, axis, None] >= lo[None, :, axis])
    i, j = np.nonzero(candidate)
    shared = (faces[i, :, None] == faces[j, None, :]).any((1, 2))
    i, j = i[~shared], j[~shared]
    found = np.zeros(len(i), bool)
    for source, target in ((tri[i], tri[j]), (tri[j], tri[i])):
        e1, e2 = target[:, 1] - target[:, 0], target[:, 2] - target[:, 0]
        for k in range(3):
            origin, direction = source[:, k], source[:, (k + 1) % 3] - source[:, k]
            h = np.cross(direction, e2)
            det = np.einsum("ij,ij->i", e1, h)
            valid = np.abs(det) >= 1e-12
            inv = np.divide(1., det, out=np.zeros_like(det), where=valid)
            s = origin - target[:, 0]
            u = inv * np.einsum("ij,ij->i", s, h)
            q = np.cross(s, e1)
            v = inv * np.einsum("ij,ij->i", direction, q)
            t = inv * np.einsum("ij,ij->i", e2, q)
            found |= valid & (u > 1e-7) & (v > 1e-7) & (u + v < 1 - 1e-7) & (t > 1e-7) & (t < 1 - 1e-7)
    return np.stack([i[found], j[found]], axis=1).tolist()


def triangle_separation_loss(vertices, faces, pairs):
    """Separate screened intersecting triangle pairs, using actual surfaces.

    Pairs are a frozen broad-phase selection, refreshed by the caller. This
    differentiable plane penalty is a surrogate, requiring a final exact screen.
    """
    if pairs.numel() == 0:
        return vertices.sum() * 0
    tri = vertices[:, faces]
    if pairs.shape[1] == 3:
        # Frame-specific intersections: a pair detected in one pose must not
        # constrain the same infinite planes in other, nonintersecting poses.
        first = tri[pairs[:, 0], pairs[:, 1]][None]
        second = tri[pairs[:, 0], pairs[:, 2]][None]
    else:
        first, second = tri[:, pairs[:, 0]], tri[:, pairs[:, 1]]
    def separation(source, target):
        normal = torch.nn.functional.normalize(torch.cross(
            target[..., 1, :] - target[..., 0, :],
            target[..., 2, :] - target[..., 0, :], dim=-1), dim=-1, eps=1e-10)
        signed = ((source - target[..., 0:1, :]) * normal[..., None, :]).sum(-1)
        positive = (torch.relu(.0007 - signed) / .002).square().mean(-1)
        negative = (torch.relu(.0007 + signed) / .002).square().mean(-1)
        return torch.minimum(positive, negative)
    values = separation(first, second) + separation(second, first)
    return values.mean() + values.amax(1).mean()


def load_grasp(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data.get("schema") != "constructed_bilateral_grasp_v1"
            or data.get("coordinate_frame") != "walker_rigid_local"
            or data.get("length_unit") != "m"
            or data.get("angle_unit") != "radian"
            or not data.get("accepted_for_grasp_initialization")
            or data.get("observed_pose") is not False):
        raise ValueError("unsupported constructed grasp contract")
    for side in ("left", "right"):
        h = data["hands"][side]
        for key, shape in (("hand_pca", (1, 12)), ("hand_pose_axis_angle", (1, 45)),
                           ("wrist_walker_m", (3,)),
                           ("rotation_walker_from_wrist", (3, 3))):
            a = np.asarray(h[key], dtype=float)
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError(f"invalid {side} {key}")
        validate_rotation(np.asarray(h["rotation_walker_from_wrist"]))
    return data


def validate_rotation(r: np.ndarray) -> None:
    if not np.isfinite(r).all() or not np.allclose(
            np.swapaxes(r, -1, -2) @ r, np.eye(3), atol=1e-5
    ) or not np.allclose(np.linalg.det(r), 1.0, atol=1e-5):
        raise ValueError("invalid proper rotation")


def compose_body(base_rotations, corrections, joint_ids):
    """Return [N,21,3,3]; untouched joints are bit-identical to base."""
    if corrections.shape != (base_rotations.shape[0], len(joint_ids), 3):
        raise ValueError("joint correction shape mismatch")
    result = base_rotations.clone()
    ids = torch.as_tensor(joint_ids, device=result.device) - 1
    delta = batch_rodrigues(corrections.reshape(-1, 3)).reshape(
        corrections.shape[0], len(joint_ids), 3, 3)
    result[:, ids] = base_rotations[:, ids] @ delta
    return result


def wrist_rotations(root_rotations, body_rotations, parents):
    """Actual FK rotations including local wrist rotation (joints 20/21)."""
    chain = [root_rotations]
    for j in range(1, 22):
        p = int(parents[j])
        if not 0 <= p < j:
            raise ValueError("invalid body parent tree")
        chain.append(chain[p] @ body_rotations[:, j - 1])
    return torch.stack((chain[20], chain[21]), dim=1)


def align_wrist_rotations(root_rotations, body_rotations, parents, target_camera_rotations):
    """Differentiably solve each local wrist rotation from its parent FK.

    A firm-grasp mode: forearms may move; palms keep walker-relative orientation.
    Wrist centers still follow the normal body chain and position constraints.
    """
    chain, local = [root_rotations], []
    for j in range(1, 22):
        parent = int(parents[j])
        if not 0 <= parent < j:
            raise ValueError("invalid body parent tree")
        rotation = body_rotations[:, j - 1]
        if j in (20, 21):
            rotation = chain[parent].transpose(-1, -2) @ target_camera_rotations[j - 20]
        local.append(rotation)
        chain.append(chain[parent] @ rotation)
    return torch.stack(local, dim=1)


def masked_mean(values, mask):
    """Zero unavailable contribution without fabricated observations."""
    if values.shape != mask.shape:
        raise ValueError("masked loss shape mismatch")
    return torch.where(mask, values, torch.zeros_like(values)).sum() / mask.sum().clamp_min(1)


def foot_terms(vertices_ground, sole_indices, weights, support, frame_ok,
               dt_s=1 / 30):
    """C1 surface + full-frame nonpenetration; C2 support/support velocity.

    Same fixed vertex IDs across time. Unknown support never enables C2.
    Original coefficients (1, 1, .001) are applied by the caller.
    """
    contact, penetration, tangent = [], [], []
    active_pairs = []
    for s, indices in enumerate(sole_indices):
        pts = vertices_ground[:, indices]
        w = weights[:, s] * frame_ok.to(weights.dtype)
        contact.append((surface.foot_surface_loss(pts[..., 2]) * w).sum()
                       / w.sum().clamp_min(1e-6))
        penetration.append(masked_mean(
            surface.nonpenetration_loss(pts[..., 2], margin_m=.003), frame_ok))
        active = (support[:-1, s] & support[1:, s]
                  & frame_ok[:-1] & frame_ok[1:])
        mask = active[:, None].expand(-1, len(indices))
        tangent.append(surface.tangential_velocity_loss(
            pts[1:], pts[:-1], dt_s, [0., 0., 1.], mask))
        active_pairs.append(int(active.sum().detach().cpu()))
    return {"foot_contact": torch.stack(contact).mean(),
            "foot_nonpenetration": torch.stack(penetration).mean(),
            "foot_tangential": torch.stack(tangent).mean()}, active_pairs


def frozen_surface_states(vertices_ground, sole_indices, frame_ok, dt_s=1 / 30):
    """Freeze engineering support candidates from same-source baseline only.

    A penetrating baseline is unknown, not evidence of support. All decisions
    and reasons survive export. Never relabel during optimization to lower C2.
    """
    from .contact_state import ContactObservation, ContactState, ContactStateMachine
    n = len(vertices_ground)
    support = np.zeros((n, 2), dtype=bool)
    reasons = np.full((n, 2), "unavailable_previous_frame", dtype="U64")
    states = np.full((n, 2), "unknown", dtype="U32")
    for s, idx in enumerate(sole_indices):
        machine = ContactStateMachine()
        pts = vertices_ground[:, idx]
        for t in range(n):
            finite = bool(np.isfinite(pts[t]).all())
            valid = bool(t > 0 and frame_ok[t] and frame_ok[t - 1] and finite)
            velocity = (pts[t] - pts[t - 1]) / dt_s if t else np.zeros_like(pts[t])
            min_z = float(pts[t, :, 2].min()) if finite else float("nan")
            continuous = bool(valid and np.linalg.norm(pts[t].mean(0) - pts[t - 1].mean(0)) <= .060)
            decision = machine.update(ContactObservation(
                t * dt_s, abs(min_z), float(abs(velocity[:, 2].mean())),
                float(np.linalg.norm(velocity[:, :2], axis=1).mean()),
                float(valid and min_z >= -.003), continuous, finite))
            states[t, s] = decision.state.value
            reasons[t, s] = "baseline_penetration_exceeds_3mm" if min_z < -.003 else decision.reason
            support[t, s] = decision.state == ContactState.STICKING_CANDIDATE
    return support, states, reasons
