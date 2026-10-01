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
