"""Differentiable SMPL-H objective terms with the original units and masks.

Stage-level total-loss assembly remains explicit in pipeline.py.
"""

import torch


def body_reprojection_loss(
    *,
    args,
    stage_index,
    body_loss,
    proj_cam0,
    proj_cam1,
    raw_left_2d,
    raw_right_2d,
    raw_left_conf,
    raw_right_conf,
    raw_left_valid,
    raw_right_valid,
):
    """Dual raw-fisheye Huber loss and diagnostic pixel residual."""
    body_2d_loss = body_loss.detach() * 0.0
    body_2d_px = body_loss.detach() * 0.0
    if args.body_reprojection_weight > 0 and stage_index in (0, 1, 2):
        e0 = (
            torch.linalg.vector_norm(proj_cam0 - raw_left_2d, dim=-1)
            / args.body_reprojection_scale_px
        )
        e1 = (
            torch.linalg.vector_norm(proj_cam1 - raw_right_2d, dim=-1)
            / args.body_reprojection_scale_px
        )
        delta = 1.0
        h0 = torch.where(e0 <= delta, 0.5 * e0.pow(2), delta * (e0 - 0.5 * delta))
        h1 = torch.where(e1 <= delta, 0.5 * e1.pow(2), delta * (e1 - 0.5 * delta))
        d0 = raw_left_conf * raw_left_valid
        d1 = raw_right_conf * raw_right_valid
        body_2d_loss = (
            (h0 * d0).sum() / d0.sum().clamp_min(1e-6)
            + (h1 * d1).sum() / d1.sum().clamp_min(1e-6)
        ) * 0.5
        body_2d_px = (
            (torch.linalg.vector_norm(proj_cam0 - raw_left_2d, dim=-1) * d0).sum()
            / d0.sum().clamp_min(1e-6)
            + (torch.linalg.vector_norm(proj_cam1 - raw_right_2d, dim=-1) * d1).sum()
            / d1.sum().clamp_min(1e-6)
        ) * 0.5
    return body_2d_loss, body_2d_px


def hand_temporal_loss(
    *,
    args,
    n,
    lhand,
    rhand,
    mask_ll,
    mask_rl,
    mask_lr,
    mask_rr,
    mano_enabled,
    mano_weights,
):
    """Second differences only across three supported hand frames."""
    hand_temporal = (lhand.sum() + rhand.sum()) * 0.0
    if n >= 3 and not args.shared_hand_pose:
        l_valid_frame = mask_ll.any(dim=1) | mask_rl.any(dim=1)
        r_valid_frame = mask_lr.any(dim=1) | mask_rr.any(dim=1)
        if mano_enabled and args.hand_2d_weight == 0:
            l_valid_frame = mano_weights["left"].sum(dim=1) > 0
            r_valid_frame = mano_weights["right"].sum(dim=1) > 0
        elif mano_enabled:
            l_valid_frame = l_valid_frame | (mano_weights["left"].sum(dim=1) > 0)
            r_valid_frame = r_valid_frame | (mano_weights["right"].sum(dim=1) > 0)
        l_edge = l_valid_frame[2:] & l_valid_frame[1:-1] & l_valid_frame[:-2]
        r_edge = r_valid_frame[2:] & r_valid_frame[1:-1] & r_valid_frame[:-2]
        l_acc = (lhand[2:] - 2 * lhand[1:-1] + lhand[:-2]).pow(2).mean(dim=1)
        r_acc = (rhand[2:] - 2 * rhand[1:-1] + rhand[:-2]).pow(2).mean(dim=1)
        l_term = l_acc[l_edge].mean() if l_edge.any() else l_acc.mean() * 0.0
        r_term = r_acc[r_edge].mean() if r_edge.any() else r_acc.mean() * 0.0
        hand_temporal = 0.5 * (l_term + r_term)
    return hand_temporal


def body_temporal_loss(*, args, n, stage_index, body_loss, coco, mask_body):
    """Pelvis-relative Huber acceleration on fully supported frame triplets."""
    body_temporal = body_loss.detach() * 0.0
    if args.body_temporal_weight > 0 and n >= 3 and stage_index in (0, 2):
        pelvis = 0.5 * (coco[:, 11] + coco[:, 12])
        rel = coco - pelvis[:, None, :]
        acc = rel[2:] - 2.0 * rel[1:-1] + rel[:-2]
        body_valid_frame = mask_body.all(dim=1)
        edge = body_valid_frame[2:] & body_valid_frame[1:-1] & body_valid_frame[:-2]
        acc_norm = torch.linalg.vector_norm(acc, dim=-1)
        # Huber in metres: preserve genuine motion while suppressing
        # one-frame detector spikes. Delta is fixed and recorded.
        delta = 0.03
        robust = torch.where(
            acc_norm <= delta, 0.5 * acc_norm.pow(2), delta * (acc_norm - 0.5 * delta)
        )
        body_temporal = robust[edge].mean() if edge.any() else robust.mean() * 0.0
    return body_temporal


def weighted_hand_residual(res, w, mask):
    return (res * w).sum() / w.sum().clamp_min(1e-6) if mask.any() else res.sum() * 0.0
