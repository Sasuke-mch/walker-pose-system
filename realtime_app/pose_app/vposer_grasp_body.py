"""Body-prior graph for walker grasp fitting. No free joint overrides."""
import numpy as np
import torch
from smplx.lbs import batch_rodrigues


def decode_body_rotations(decoder, latent):
    if latent.ndim != 2 or latent.shape[1] != 32 or not torch.isfinite(latent).all():
        raise ValueError("VPoser requires finite [N,32] latent")
    if decoder.training or any(p.requires_grad for p in decoder.parameters()):
        raise ValueError("VPoser weights must be frozen and decoder in eval mode")
    pose = decoder.decode(latent)["pose_body"]
    if pose.numel() != len(latent)*63 or not torch.isfinite(pose).all():
        raise ValueError("VPoser must decode all 21 body joints")
    return batch_rodrigues(pose.reshape(-1,3)).reshape(-1,21,3,3)


def rotation_anchor_loss(current, reference, scale_rad=np.deg2rad(15)):
    """Smooth chordal SO3 anchor to one immutable reference, normalized."""
    if current.shape != reference.shape:
        raise ValueError("rotation anchor shape mismatch")
    return (current-reference).square().sum((-1,-2)).mean()/(2*scale_rad**2)


def rotation_temporal_loss(rotations, frame_valid, scale_rad=np.deg2rad(10)):
    """Joint angular increments; same-frame availability gates both endpoints.

    Scale is radians per sampled frame interval, not an acceleration in SI.
    """
    mask = frame_valid[1:] & frame_valid[:-1]
    values = (rotations[1:]-rotations[:-1]).square().sum((-1,-2)).mean(-1)/(2*scale_rad**2)
    return torch.where(mask, values, torch.zeros_like(values)).sum()/mask.sum().clamp_min(1)
