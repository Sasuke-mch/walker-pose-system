"""Validated surface-contact inputs and their differentiable objectives."""

from __future__ import annotations

from dataclasses import dataclass
import json
import numpy as np
import torch
from .. import smpl_surface_contact as surface_contact
from ..constructed_grasp_refinement import validate_rotation


@dataclass(frozen=True)
class SurfaceTargets:
    hand_weight: torch.Tensor | None
    foot_weight: torch.Tensor | None
    handle: torch.Tensor | None
    rotation: torch.Tensor | None
    translation: torch.Tensor | None
    palm_indices: dict[str, torch.Tensor] | None
    sole_indices: dict[str, torch.Tensor] | None
    global_hand_offset: torch.Tensor | None
    global_hand_surface: dict[str, torch.Tensor] | None
    camera_rotation: torch.Tensor | None
    camera_translation: torch.Tensor | None


def _vertex_indices(values, name: str, device):
    indices = np.asarray(values)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(f"{name} must contain a non-empty vertex index vector")
    if indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= 6890):
        raise ValueError(f"{name} must contain integer indices in SMPL-H range 0..6889")
    return torch.tensor(indices, dtype=torch.long, device=device)


def _finite_array(values, shape, name: str):
    array = np.asarray(values)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def _archive_arrays(path, names):
    # Materialize only the consumed fields, then close the NPZ file handle.
    with np.load(path, allow_pickle=True) as archive:
        return {name: archive[name] for name in names if name in archive.files}


def load_surface_targets(args, n: int, device) -> SurfaceTargets:
    """Read the existing contact protocol; keep validation order and units."""
    contact_enabled = args.contact_labels is not None
    contact_hand_weight = contact_foot_weight = contact_handle = contact_R = (
        contact_T
    ) = palm_idx = sole_idx = None
    global_hand_offset = None
    global_hand_surface = global_camera_R = global_camera_t = None
    if contact_enabled:
        labels = _archive_arrays(
            args.contact_labels,
            ("handle_ends_ground_m", "hand_contact_weight", "foot_contact_weight"),
        )
        scene = _archive_arrays(
            args.scene_transforms,
            ("rotation_ground_from_left", "translation_ground_from_left_mm"),
        )
        if int(labels["handle_ends_ground_m"].shape[0]) != n:
            raise ValueError("contact labels frame count does not match raw input")
        if tuple(labels["hand_contact_weight"].shape) != (n, 2):
            raise ValueError("hand_contact_weight must have shape (frames,2)")
        if tuple(labels["foot_contact_weight"].shape) != (n, 2):
            raise ValueError("foot_contact_weight must have shape (frames,2)")
        if tuple(labels["handle_ends_ground_m"].shape) != (n, 2, 2, 3):
            raise ValueError("handle_ends_ground_m must have shape (frames,2,2,3)")
        if tuple(scene["rotation_ground_from_left"].shape) != (n, 3, 3):
            raise ValueError("rotation_ground_from_left must have shape (frames,3,3)")
        if tuple(scene["translation_ground_from_left_mm"].shape) != (n, 3):
            raise ValueError(
                "translation_ground_from_left_mm must have shape (frames,3)"
            )
        if not np.isfinite(labels["hand_contact_weight"]).all() or np.any(
            labels["hand_contact_weight"] < 0
        ):
            raise ValueError("hand_contact_weight must be finite and non-negative")
        if not np.isfinite(labels["foot_contact_weight"]).all() or np.any(
            labels["foot_contact_weight"] < 0
        ):
            raise ValueError("foot_contact_weight must be finite and non-negative")
        if not np.isfinite(labels["handle_ends_ground_m"]).all():
            raise ValueError("handle_ends_ground_m contains non-finite values")
        _finite_array(
            scene["rotation_ground_from_left"], (n, 3, 3), "rotation_ground_from_left"
        )
        _finite_array(
            scene["translation_ground_from_left_mm"],
            (n, 3),
            "translation_ground_from_left_mm",
        )
        validate_rotation(scene["rotation_ground_from_left"])
        contact_handle = torch.tensor(
            labels["handle_ends_ground_m"], dtype=torch.float32, device=device
        )
        contact_hand_weight = torch.tensor(
            labels["hand_contact_weight"], dtype=torch.float32, device=device
        )
        contact_foot_weight = torch.tensor(
            labels["foot_contact_weight"], dtype=torch.float32, device=device
        )
        contact_R = torch.tensor(
            scene["rotation_ground_from_left"], dtype=torch.float32, device=device
        )
        contact_T = torch.tensor(
            scene["translation_ground_from_left_mm"] / 1000.0,
            dtype=torch.float32,
            device=device,
        )
        sets = json.loads(args.contact_vertex_sets.read_text(encoding="utf-8"))["sets"]
        topology = json.loads(args.walker_topology.read_text(encoding="utf-8"))
        handles = topology.get("handle_segments", {})
        if set(handles) != {"left", "right"} or any(
            len(v) != 2 for v in handles.values()
        ):
            raise ValueError(
                "walker topology must expose exactly two endpoints for left/right handles"
            )
        palm_idx = {
            s: _vertex_indices(
                sum(sets[f"{s}_palm_surface_candidate"].values(), []),
                f"{s} palm vertices",
                device,
            )
            for s in ("left", "right")
        }
        sole_idx = {
            s: _vertex_indices(
                sum(sets[f"{s}_sole_surface_candidate"].values(), []),
                f"{s} sole vertices",
                device,
            )
            for s in ("left", "right")
        }
        if (
            float(
                np.asarray(labels["hand_contact_weight"]).sum()
                + np.asarray(labels["foot_contact_weight"]).sum()
            )
            <= 0.0
        ):
            raise ValueError("contact mode requested but all contact weights are zero")
        if args.global_hand_handle_pose is not None:
            prior = json.loads(args.global_hand_handle_pose.read_text(encoding="utf-8"))
            if (
                prior.get("assumption")
                != "hand_static_relative_to_walker_for_entire_video"
            ):
                raise ValueError("global hand-handle prior assumption mismatch")
            if (
                prior.get("status") != "engineering_candidate"
                or prior.get("geometry_source") != "camera_rigid_mount_assumption"
            ):
                raise ValueError(
                    "global prior requires validated camera-rigid solver provenance"
                )
            global_camera_R = torch.tensor(
                _finite_array(
                    topology["rotation_left_camera_from_walker"],
                    (3, 3),
                    "rotation_left_camera_from_walker",
                ),
                dtype=torch.float32,
                device=device,
            )
            global_camera_t = (
                torch.tensor(
                    _finite_array(
                        topology["translation_left_camera_from_walker_mm"],
                        (3,),
                        "translation_left_camera_from_walker_mm",
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                / 1000
            )
            validate_rotation(np.asarray(topology["rotation_left_camera_from_walker"]))
            global_hand_surface = {}
            for s in ("left", "right"):
                indices = prior["hands"][s]["vertex_indices"]
                palm_idx[s] = _vertex_indices(
                    indices, f"{s} global palm vertices", device
                )
                global_hand_surface[s] = torch.tensor(
                    _finite_array(
                        prior["hands"][s]["shared_surface_walker_m"],
                        (len(indices), 3),
                        f"{s} shared_surface_walker_m",
                    ),
                    dtype=torch.float32,
                    device=device,
                )
            global_hand_offset = torch.tensor(
                _finite_array(
                    [
                        prior["hands"][s]["palm_offset_handle_m"]
                        for s in ("left", "right")
                    ],
                    (2, 3),
                    "palm_offset_handle_m",
                ),
                dtype=torch.float32,
                device=device,
            )
    return SurfaceTargets(
        contact_hand_weight,
        contact_foot_weight,
        contact_handle,
        contact_R,
        contact_T,
        palm_idx,
        sole_idx,
        global_hand_offset,
        global_hand_surface,
        global_camera_R,
        global_camera_t,
    )


def surface_contact_losses(
    *, left_in_cam0, contact_enabled, stage_index, vertices, contact
):
    """Surface hand, foot and optional walker-relative hand objectives."""
    contact_hand_loss = left_in_cam0.sum() * 0.0
    contact_foot_loss = left_in_cam0.sum() * 0.0
    global_hand_loss = left_in_cam0.sum() * 0.0
    if contact_enabled and stage_index >= 4:
        vg = (
            torch.einsum("nij,nvj->nvi", contact.rotation, vertices)
            + contact.translation[:, None, :]
        )
        hand_terms = []
        foot_terms = []
        for side, j in (("left", 0), ("right", 1)):
            pts = vg[:, contact.palm_indices[side], :]
            a, b = contact.handle[:, j, 0, :], contact.handle[:, j, 1, :]
            hand_terms.append(
                surface_contact.hand_surface_loss(
                    pts, a[:, None, :], b[:, None, :], 0.016
                )
                * contact.hand_weight[:, j]
            )
        for side, j in (("left", 0), ("right", 1)):
            sole_z = vg[:, contact.sole_indices[side], 2]
            foot_terms.append(
                surface_contact.foot_surface_loss(sole_z) * contact.foot_weight[:, j]
            )
        hand_den = contact.hand_weight.sum().clamp_min(1e-6)
        foot_den = contact.foot_weight.sum().clamp_min(1e-6)
        contact_hand_loss = torch.stack(hand_terms, dim=1).sum() / hand_den
        contact_foot_loss = torch.stack(foot_terms, dim=1).sum() / foot_den
        if contact.global_hand_offset is not None:
            terms = []
            for side in ("left", "right"):
                pc = vertices[:, contact.palm_indices[side], :]
                pw = torch.einsum(
                    "ij,nvj->nvi",
                    contact.camera_rotation.T,
                    pc - contact.camera_translation,
                )
                err = (
                    torch.linalg.vector_norm(
                        pw - contact.global_hand_surface[side], dim=-1
                    )
                    / 0.03
                )
                terms.append(torch.where(err <= 1, 0.5 * err.pow(2), err - 0.5).mean())
            global_hand_loss = torch.stack(terms).mean()
    return contact_hand_loss, contact_foot_loss, global_hand_loss
