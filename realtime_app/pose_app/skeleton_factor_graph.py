"""Fixed-bone-length kinematic skeleton factor graph (real-time mainline candidate).

A lightweight alternative to SMPL-X: a tree-structured COCO body skeleton with
a virtual pelvis root, SHARED bone lengths across all frames, per-frame root
pose and per-joint rotations, plus factors:

  1. L/R raw-fisheye 2D reprojection (primary observation, GM-robust, quality
     weighted, metric-normalised via local depth linearisation),
  2. second-order temporal smoothness on the 3D joints,
  3. contact-phase foot staticness (soft) and ground non-penetration,
  4. hand-to-handle proximity from the frozen offline interaction records,
  5. weak bone-length prior toward the initial estimates.

Bone lengths are optimised jointly but the skeleton topology keeps them
constant across the window (this is the "fixed bone length" property; it is
not a full anatomical rig and has no mesh).

Topology (star from the virtual pelvis root; limb joints each own a world
rotation that orients the bone away from the world down direction):
    root -> hip_L(11), hip_R(12)            offset +/-hip_width/2 along body left
    root -> sh_L(5),  sh_R(6)               offset spine*up +/- shoulder_width/2
    sh_L -> el_L(7)  -> wr_L(9)             upper_arm, forearm
    sh_R -> el_R(8)  -> wr_R(10)
    hip_L -> kn_L(13) -> ank_L(15)          thigh, shank
    hip_R -> kn_R(14) -> ank_R(16)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


BODY_JOINTS = tuple(range(5, 17))  # COCO body joints 5..16
DOWN_LOCAL = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
UP_LOCAL = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
LEFT_LOCAL = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)


@dataclass(frozen=True)
class BoneSpec:
    name: str
    parent: int  # COCO joint index or -1 for the virtual root
    child: int  # COCO joint index
    rest_local: np.ndarray  # unit direction in the root local frame
    is_limb: bool  # True if the bone direction is controlled by a joint rotation


def build_skeleton() -> dict[int, list[tuple[str, float]]]:
    """Return a mapping joint -> list of (bone_name, signed length factor).

    The factor is applied to the shared bone length vector; negative factors
    flip the direction (e.g. right-side limbs share the same length).
    """
    return {
        -1: [],  # virtual root
        11: [("hip_width", -0.5)],
        12: [("hip_width", +0.5)],
        5: [("spine", 1.0), ("shoulder_width", -0.5)],
        6: [("spine", 1.0), ("shoulder_width", +0.5)],
        7: [("upper_arm_left", -1.0)],
        8: [("upper_arm_right", -1.0)],
        9: [("forearm_left", -1.0)],
        10: [("forearm_right", -1.0)],
        13: [("thigh_left", -1.0)],
        14: [("thigh_right", -1.0)],
        15: [("shank_left", -1.0)],
        16: [("shank_right", -1.0)],
    }


DEFAULT_BONE_NAMES = [
    "spine", "shoulder_width", "hip_width",
    "upper_arm_left", "upper_arm_right", "forearm_left", "forearm_right",
    "thigh_left", "thigh_right", "shank_left", "shank_right",
]

# (parent, child) with the controlling joint rotation name.
LIMB_BONES = [
    (7, "elbow_L"), (8, "elbow_R"),
    (9, "wrist_L"), (10, "wrist_R"),
    (13, "knee_L"), (14, "knee_R"),
    (15, "ankle_L"), (16, "ankle_R"),
]


def rodrigues_torch(axis_angle: "torch.Tensor") -> "torch.Tensor":
    """Batch axis-angle (...,3) -> rotation matrices (...,3,3)."""
    if torch is None:
        raise RuntimeError("torch not available")
    angle = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    zero = angle < 1e-8
    axis = torch.where(zero, torch.as_tensor([1.0, 0.0, 0.0], device=axis_angle.device, dtype=axis_angle.dtype), axis_angle / torch.clamp(angle, min=1e-8))
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    theta = angle[..., 0]
    c = torch.cos(theta)
    s = torch.sin(theta)
    one_c = 1.0 - c
    col0 = torch.stack([c + x * x * one_c, y * x * one_c + z * s, z * x * one_c - y * s], dim=-1)
    col1 = torch.stack([x * y * one_c - z * s, c + y * y * one_c, z * y * one_c + x * s], dim=-1)
    col2 = torch.stack([x * z * one_c + y * s, y * z * one_c - x * s, c + z * z * one_c], dim=-1)
    return torch.stack([col0, col1, col2], dim=-1)


# Position indices inside the 12-vector output of FixedBoneSkeleton.forward
# (columns map to COCO 5..16: 0=Lsh 1=Rsh 2=Lel 3=Rel 4=Lwr 5=Rwr
#  6=Lhi 7=Rhi 8=Lkn 9=Rkn 10=Lank 11=Rank).
POS_LSH, POS_RSH = 0, 1
POS_LEL, POS_REL = 2, 3
POS_LWR, POS_RWR = 4, 5
POS_LHI, POS_RHI = 6, 7
POS_LKN, POS_RKN = 8, 9
POS_LANK, POS_RANK = 10, 11

# Rotation parameter indices for the hinge joints (elbow/knee): soft angle
# range and no-obvious-reverse-hyperextension priors apply to these.
HINGE_PARAM_INDICES = (0, 1, 4, 5)  # elbow_L, elbow_R, knee_L, knee_R

# (joint_position, parent_position, child_position) for each hinge joint.
HINGE_TRIPLES = (
    (POS_LEL, POS_LSH, POS_LWR),   # elbow_L
    (POS_REL, POS_RSH, POS_RWR),   # elbow_R
    (POS_LKN, POS_LHI, POS_LANK),  # knee_L
    (POS_RKN, POS_RHI, POS_RANK),  # knee_R
)


def hinge_soft_prior(joints: "torch.Tensor", max_hinge_rotation_rad: float = 2.9,
                     min_segment_angle_deg: float = 40.0) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Soft angle-range / no-obvious-reverse-hyperextension priors for elbows and knees.

    Both terms are soft (quadratic beyond a threshold) and never override a
    valid 2D observation. The anatomical hinge angle at joint J is the angle
    between u(P->J) and u(C->J) (both segments pointing into the joint);
    straight limb = 180 deg. The priors are:

      - fold-back term:  penalise dot(u(P->J), u(C->J)) > cos(min_segment_angle)
        i.e. the child segment folding back onto the parent segment (anatomical
        hinge angle below the minimum), the visible "obvious reverse
        hyperextension / folded limb" artifact.
      - rotation cap:    penalise ||rot[hinge_param]|| > max_hinge_rotation_rad
        (limb folded more than the cap from its rest direction).

    Args:
        joints: (N, 12, 3) world metres.
        max_hinge_rotation_rad: soft cap on the hinge rotation magnitude.
        min_segment_angle_deg: minimum anatomical hinge angle allowed at the
            elbow/knee.
    Returns:
        (loss_scalar, per_frame_penalty) where per_frame_penalty is (N,) with
        the per-frame fold-back penalty magnitude (0 where not triggered).
    """
    if torch is None:
        raise RuntimeError("torch not available")
    eps = 1e-9
    joint_positions = torch.as_tensor([t[0] for t in HINGE_TRIPLES], dtype=torch.long, device=joints.device)
    parent_positions = torch.as_tensor([t[1] for t in HINGE_TRIPLES], dtype=torch.long, device=joints.device)
    child_positions = torch.as_tensor([t[2] for t in HINGE_TRIPLES], dtype=torch.long, device=joints.device)
    parent_to_joint = joints[:, joint_positions] - joints[:, parent_positions]  # (N,4,3) P -> J
    child_to_joint = joints[:, joint_positions] - joints[:, child_positions]    # (N,4,3) C -> J
    u_pj = parent_to_joint / torch.clamp(torch.linalg.vector_norm(parent_to_joint, dim=2, keepdim=True), min=eps)
    u_cj = child_to_joint / torch.clamp(torch.linalg.vector_norm(child_to_joint, dim=2, keepdim=True), min=eps)
    dot = torch.sum(u_pj * u_cj, dim=2)
    cos_min = float(np.cos(np.deg2rad(min_segment_angle_deg)))
    frame_penalty = torch.clamp(dot - cos_min, min=0.0).square().sum(dim=1)
    return torch.mean(frame_penalty), frame_penalty


class FixedBoneSkeleton:
    """Differentiable fixed-bone skeleton forward kinematics (torch)."""

    def __init__(self, bone_lengths_mm: np.ndarray, joint_rotation_init: np.ndarray | None = None,
                 root_rot_init: np.ndarray | None = None, root_pos_init_mm: np.ndarray | None = None,
                 device: str = "cpu") -> None:
        if torch is None:
            raise RuntimeError("torch not available")
        self.device = torch.device(device)
        self.bone_lengths = torch.nn.Parameter(
            torch.as_tensor(np.asarray(bone_lengths_mm, dtype=np.float32) / 1000.0, device=self.device)
        )
        if joint_rotation_init is not None:
            self.joint_rotation_init = torch.as_tensor(
                np.asarray(joint_rotation_init, dtype=np.float32), device=self.device
            )
        else:
            self.joint_rotation_init = None
        if root_rot_init is not None:
            self.root_rot_init = torch.as_tensor(np.asarray(root_rot_init, dtype=np.float32), device=self.device)
        else:
            self.root_rot_init = None
        if root_pos_init_mm is not None:
            self.root_pos_init = torch.as_tensor(
                np.asarray(root_pos_init_mm, dtype=np.float32) / 1000.0, device=self.device
            )
        else:
            self.root_pos_init = None
        self.rotations: Optional["torch.nn.Parameter"] = None
        self.root_pos: Optional["torch.nn.Parameter"] = None
        self.root_rot: Optional["torch.nn.Parameter"] = None

    def initialise(self, frame_count: int, requires_grad_rotations: bool = True,
                   requires_grad_root: bool = True, requires_grad_bones: bool = True,
                   rotations_offset: "Any" | None = None) -> None:
        if self.joint_rotation_init is None:
            raise RuntimeError("joint_rotation_init required")
        rotations = self.joint_rotation_init[:frame_count].clone()
        if rotations_offset is not None:
            rotations = rotations + rotations_offset.to(self.device)
        self.rotations = torch.nn.Parameter(rotations, requires_grad=requires_grad_rotations)
        if self.root_pos_init is not None:
            root_pos = self.root_pos_init[:frame_count].clone()
        else:
            root_pos = torch.zeros((frame_count, 3), device=self.device)
        self.root_pos = torch.nn.Parameter(root_pos, requires_grad=requires_grad_root)
        if self.root_rot_init is None:
            self.root_rot = torch.nn.Parameter(
                torch.zeros((frame_count, 3), device=self.device), requires_grad=requires_grad_root
            )
        else:
            self.root_rot = torch.nn.Parameter(
                self.root_rot_init[:frame_count].clone(), requires_grad=requires_grad_root
            )
        self.bone_lengths.requires_grad_(requires_grad_bones)

    def forward(self) -> "torch.Tensor":
        """Return (N, 12, 3) world-frame body joints for COCO 5..16."""
        if torch is None:
            raise RuntimeError("torch not available")
        rotations = self.rotations
        root_pos = self.root_pos
        root_rot = self.root_rot
        bone = self.bone_lengths
        dtype = rotations.dtype
        device = self.device

        R_root = rodrigues_torch(root_rot)  # (N,3,3)
        up = torch.as_tensor(UP_LOCAL, dtype=dtype, device=device)
        left = torch.as_tensor(LEFT_LOCAL, dtype=dtype, device=device)
        down = torch.as_tensor(DOWN_LOCAL, dtype=dtype, device=device)

        # Torso (star from the root): hip/shoulder offsets in the root local frame.
        # bone[2] = hip_width, bone[0] = spine, bone[1] = shoulder_width.
        # Subject's +left axis points toward joint 11 (left hip) / joint 5 (left shoulder).
        joints = {}
        joints[11] = root_pos + torch.einsum("nij,j->ni", R_root, left * bone[2] * 0.5)
        joints[12] = root_pos + torch.einsum("nij,j->ni", R_root, left * bone[2] * -0.5)
        spine_vec = torch.einsum("nij,j->ni", R_root, up * bone[0])
        joints[5] = root_pos + spine_vec + torch.einsum("nij,j->ni", R_root, left * bone[1] * 0.5)
        joints[6] = root_pos + spine_vec + torch.einsum("nij,j->ni", R_root, left * bone[1] * -0.5)

        # Limbs: child = parent + R_joint @ (bone_len * down). The joint rotation is
        # a world-frame rotation applied to the world-down rest direction.
        # rotation param index -> (parent COCO, child COCO, bone index)
        limb_specs = [
            (0, 5, 7, 3),   # elbow_L
            (1, 6, 8, 4),   # elbow_R
            (2, 7, 9, 5),   # wrist_L
            (3, 8, 10, 6),  # wrist_R
            (4, 11, 13, 7),  # knee_L
            (5, 12, 14, 8),  # knee_R
            (6, 13, 15, 9),  # ankle_L
            (7, 14, 16, 10),  # ankle_R
        ]
        for param_index, parent_joint, child_joint, bone_index in limb_specs:
            Rj = rodrigues_torch(rotations[:, param_index])
            joints[child_joint] = joints[parent_joint] + torch.einsum("nij,j->ni", Rj, down * bone[bone_index])

        order = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        return torch.stack([joints[index] for index in order], dim=1)


def compute_init_from_points(points_body_mm: np.ndarray) -> dict[str, np.ndarray]:
    """Initialise root pose and joint rotations from per-frame body joints (N,12,3)."""
    # points_body_mm columns map to COCO 5..16 positions 0..11:
    # 0=5(Lsh) 1=6(Rsh) 2=7(Lel) 3=8(Rel) 4=9(Lwr) 5=10(Rwr)
    # 6=11(Lhi) 7=12(Rhi) 8=13(Lkn) 9=14(Rkn) 10=15(Lank) 11=16(Rank)
    root = 0.5 * (points_body_mm[:, 6] + points_body_mm[:, 7])  # mid-hip
    mid_shoulder = 0.5 * (points_body_mm[:, 0] + points_body_mm[:, 1])
    root_pos = root
    up_vec = mid_shoulder - root
    left_vec = points_body_mm[:, 0] - points_body_mm[:, 1]
    forward = np.cross(left_vec, up_vec)
    frame_count = points_body_mm.shape[0]
    root_rot = np.zeros((frame_count, 3), dtype=np.float64)
    for frame in range(frame_count):
        left = left_vec[frame]
        norm = np.linalg.norm(left)
        left = left / norm if norm > 1e-9 else np.asarray([1.0, 0.0, 0.0])
        up = up_vec[frame]
        up = up - left * float(left @ up)
        up = up / np.linalg.norm(up) if np.linalg.norm(up) > 1e-9 else np.asarray([0.0, 1.0, 0.0])
        fwd = np.cross(left, up)
        basis = np.column_stack((left, up, fwd))
        # Log of the rotation matrix R = basis (maps local -> world): axis-angle.
        cos_angle = 0.5 * (np.trace(basis) - 1.0)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        if abs(angle) < 1e-8:
            root_rot[frame] = np.zeros(3)
        else:
            skew = basis - basis.T  # = 2 sin(angle) [axis]x
            axis = np.asarray([skew[2, 1], skew[0, 2], skew[1, 0]])
            norm = np.linalg.norm(axis)
            if norm < 1e-9 or abs(np.sin(angle)) < 1e-9:
                # angle near pi: pick a stable rotation axis from the diagonal.
                axis = np.asarray([basis[0, 0] + 1.0, basis[1, 1] + 1.0, basis[2, 2] + 1.0]) + 1e-9
                axis = axis / np.linalg.norm(axis)
            else:
                axis = axis / (2.0 * np.sin(angle))
            root_rot[frame] = axis * angle

    joint_rotation_init = np.zeros((frame_count, 8, 3), dtype=np.float64)
    parent_to_param = {7: 0, 8: 1, 9: 2, 10: 3, 13: 4, 14: 5, 15: 6, 16: 7}
    bone_dir = {
        7: (0, 2), 8: (1, 3), 9: (2, 4), 10: (3, 5),
        13: (6, 8), 14: (7, 9), 15: (8, 10), 16: (9, 11),
    }
    for parent_joint, (parent_pos, child_pos) in bone_dir.items():
        index = parent_to_param[parent_joint]
        direction = points_body_mm[:, child_pos] - points_body_mm[:, parent_pos]
        for frame in range(frame_count):
            vec = direction[frame]
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                continue
            unit = vec / norm
            down = np.asarray([0.0, -1.0, 0.0])
            cos_angle = float(np.clip(np.dot(unit, down), -1.0, 1.0))
            angle = np.arccos(cos_angle)
            if angle < 1e-6:
                joint_rotation_init[frame, index] = np.zeros(3)
            else:
                axis = np.cross(down, unit)
                norm = np.linalg.norm(axis)
                if norm < 1e-9:
                    axis = np.asarray([1.0, 0.0, 0.0])
                else:
                    axis = axis / norm
                joint_rotation_init[frame, index] = axis * angle
    return {"root_pos_mm": root_pos, "root_rot": root_rot, "joint_rotation_init": joint_rotation_init}


def estimate_bone_lengths_mm(points_body_mm: np.ndarray) -> np.ndarray:
    """Median bone lengths (mm) in DEFAULT_BONE_NAMES order from N x 12 body joints."""
    lengths = np.zeros(len(DEFAULT_BONE_NAMES), dtype=np.float64)
    # spine
    lengths[0] = np.median(np.linalg.norm(
        0.5 * (points_body_mm[:, 0] + points_body_mm[:, 1]) - 0.5 * (points_body_mm[:, 6] + points_body_mm[:, 7]), axis=1))
    # shoulder width
    lengths[1] = np.median(np.linalg.norm(points_body_mm[:, 0] - points_body_mm[:, 1], axis=1))
    # hip width
    lengths[2] = np.median(np.linalg.norm(points_body_mm[:, 6] - points_body_mm[:, 7], axis=1))
    limb_pairs = {
        "upper_arm_left": (0, 2), "upper_arm_right": (1, 3),
        "forearm_left": (2, 4), "forearm_right": (3, 5),
        "thigh_left": (6, 8), "thigh_right": (7, 9),
        "shank_left": (8, 10), "shank_right": (9, 11),
    }
    for index, name in enumerate(DEFAULT_BONE_NAMES[3:], start=3):
        parent_pos, child_pos = limb_pairs[name]
        lengths[index] = np.median(np.linalg.norm(points_body_mm[:, child_pos] - points_body_mm[:, parent_pos], axis=1))
    return lengths
