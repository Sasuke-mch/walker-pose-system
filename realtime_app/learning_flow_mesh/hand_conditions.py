"""Typed native MANO conditions, not hand positions or independent truth.

Only the existing validated reader's parent-local anatomical rotations enter
here. Camera/global orientation and MANO shape are not transferred. The native
joint order differs from the 21-point OpenPose visualization order.
"""
from dataclasses import dataclass
from enum import Enum

import torch

from .interfaces import SixDLayout, encode_rotation
from .rotations import validate_rotation_matrices

SIDE_ORDER = ("left", "right")
FINGER_ORDER = ("index", "middle", "pinky", "ring", "thumb")
HAND_NODES = (tuple(range(22, 37)), tuple(range(37, 52)))


class HandConvention(Enum):
    SMPLH_PARENT_LOCAL_NATIVE = "native_MANO_anatomical_parent_local_after_existing_mirror"


@dataclass(frozen=True)
class NativeHandTokens:
    tokens: torch.Tensor  # B,T,V,2,15,7; 6D packed columns + detector BOX score
    present: torch.Tensor  # B,T,V,2,15; unavailable token values are NaN
    local_rotations: torch.Tensor  # B,T,V,2,15,3,3; unavailable is NaN
    source_present: torch.Tensor  # B,T,V,2; original weight>0, before H2 view scope
    admitted: torch.Tensor  # B,T,V,2; source present AND explicit H2 view availability
    frame_ids: torch.Tensor
    node_ids: torch.Tensor  # 2,15 in the actual 52-node SMPL-H tree


def validate_hand_tree(parents):
    """Exact 52-node finger identity, not inferred from OpenPose point names."""
    p = torch.as_tensor(parents, dtype=torch.long)
    if p.shape != (52,) or p[0] != -1:
        raise ValueError("need actual 52-node SMPL-H tree")
    relative = p.new_tensor([0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14])
    for start, wrist in ((22, 20), (37, 21)):
        expected = torch.where(relative == 0, wrist, relative + start - 1)
        if not torch.equal(p[start:start + 15], expected):
            raise ValueError("SMPL-H native finger order/parents mismatch")


def native_hand_tokens(rotations, box_confidence, *, source_frame_ids, target_frame_ids,
                       view_available, convention: HandConvention, side_order, parents):
    """Keep both camera hypotheses with explicit temporal/anatomical identity.

    Input rotations already passed existing convert_record (left M R M). No
    mirror, projection, axis-angle/PCA averaging, imputation or new quality
    threshold is applied here. Zero-weight identity placeholders are unavailable.
    """
    if convention is not HandConvention.SMPLH_PARENT_LOCAL_NATIVE or tuple(side_order) != SIDE_ORDER:
        raise ValueError("explicit anatomical parent-local convention and left/right order required")
    if rotations.ndim != 7 or rotations.shape[-4:] != (2, 15, 3, 3):
        raise ValueError("native rotations must be B,T,V,2,15,3,3")
    b, t, views = rotations.shape[:3]
    if min(b, t, views) < 1 or not rotations.is_floating_point():
        raise ValueError("nonempty floating native rotations required")
    if box_confidence.shape != (b, t, views, 2) or not box_confidence.is_floating_point():
        raise ValueError("box confidence must be B,T,V,2")
    if (box_confidence.dtype != rotations.dtype or box_confidence.device != rotations.device
            or not torch.isfinite(box_confidence).all()
            or ((box_confidence < 0) | (box_confidence > 1)).any()):
        raise ValueError("box confidence dtype/device/range must match native rotations")
    if view_available.shape != (b, t, views) or view_available.dtype != torch.bool:
        raise ValueError("explicit boolean B,T,V selected-view availability required")
    for ids in (source_frame_ids, target_frame_ids):
        if ids.shape != (b, t) or ids.dtype != torch.long or ids.device != rotations.device:
            raise ValueError("source and target IDs must be int64 B,T on the same device")
        if (ids < 0).any() or (ids[:, 1:] <= ids[:, :-1]).any():
            raise ValueError("frame IDs must be nonnegative and increasing")
    if not torch.equal(source_frame_ids, target_frame_ids):
        raise ValueError("native/body frame IDs differ; no silent temporal remapping")
    if view_available.device != rotations.device:
        raise ValueError("view availability must match native device")
    validate_hand_tree(parents)
    source_present = box_confidence > 0  # unchanged reader marks rejection weight0
    if source_present.any():
        # Validate source selections even if H2 later masks their view unavailable.
        validate_rotation_matrices(rotations[source_present])
    admitted = source_present & view_available[..., None]
    present = admitted[..., None].expand(b, t, views, 2, 15)
    six = rotations.new_full((b, t, views, 2, 15, 6), torch.nan)
    if admitted.any():
        six[admitted] = encode_rotation(rotations[admitted], SixDLayout.PACKED_COLUMNS)
    confidence = box_confidence[..., None, None].expand(b, t, views, 2, 15, 1)
    tokens = torch.cat((six, torch.where(present[..., None], confidence, torch.nan)), dim=-1)
    local = torch.where(present[..., None, None], rotations, torch.nan)
    nodes = torch.tensor(HAND_NODES, device=rotations.device, dtype=torch.long)
    return NativeHandTokens(tokens, present, local, source_present, admitted,
                            source_frame_ids.clone(), nodes)
