"""Differentiable SMPL-H objective terms with the original units and masks.

Stage-level total-loss assembly remains explicit in pipeline.py.
"""

import numpy as np
import torch


def raw_pixel_support(raw, image_size):
    """Per-view support independent of triangulation; retain reasons for all slots."""
    raw = np.asarray(raw)
    finite = np.isfinite(raw).all(axis=-1)
    bounds = ((raw[..., 0] >= 0) & (raw[..., 0] < image_size[0])
              & (raw[..., 1] >= 0) & (raw[..., 1] < image_size[1]))
    positive = raw[..., 2] > 0
    valid = finite & bounds & positive
    reasons = np.full(valid.shape, "accepted", dtype="U32")
    reasons[~positive] = "nonpositive_confidence"
    reasons[~bounds] = "out_of_raw_image_bounds"
    reasons[~finite] = "nonfinite_source"
    return valid, reasons


def enabled_body_observation(body_loss, weight, stage_index):
    """Disabled 3D term has no autograd edge, including NaN diagnostics."""
    if weight == 0 or stage_index >= 3:
        return body_loss.detach().new_zeros(())
    return weight * body_loss


def supported_camera_points(points, valid):
    """Keep absent points outside singular projection graphs, without filling observations."""
    placeholder = points.new_tensor([0.0, 0.0, 1.0])
    return torch.where(valid[..., None], points, placeholder)


def observation_weights(quality, terms, accepted, mode, *, reference_reprojection_px=25.0,
                        reference_ray_gap_mm=80.0, reference_confidence_floor=0.6):
    """Change soft weighting only; rejected observations remain rejected."""
    if mode == "quality":
        return quality
    if mode == "uniform":
        return np.ones_like(quality)
    if mode == "reference":
        if (not np.isfinite(reference_reprojection_px) or reference_reprojection_px <= 0
                or not np.isfinite(reference_ray_gap_mm) or reference_ray_gap_mm <= 0
                or not np.isfinite(reference_confidence_floor) or not 0 <= reference_confidence_floor <= 1):
            raise ValueError("invalid reference weighting scales/floor")
        confidence, ray, reprojection, sync, depth = (np.asarray(t, dtype=float) for t in terms)
        stacked = np.stack([confidence, ray, reprojection, sync, depth])
        if not np.isfinite(stacked).all() or np.any((stacked < 0) | (stacked > 1)):
            raise ValueError("quality terms must be finite and in [0,1]")
        # Recover source geometry diagnostics from the unchanged legacy terms.
        # Do not use model residuals to decide whether an observation is valid.
        gap_mm = -30.0 * np.log(np.maximum(ray, 1e-300))
        error_px = -8.0 * np.log(np.maximum(reprojection, 1e-300))
        severity = np.maximum(gap_mm / reference_ray_gap_mm, error_px / reference_reprojection_px)
        tail = 1.0 / (1.0 + np.maximum(severity - 1.0, 0.0) ** 4)
        confidence_reference = reference_confidence_floor + (1-reference_confidence_floor) * confidence
        weights = confidence_reference * np.maximum(tail, 1e-4) * sync * depth
        weights[~accepted] = 0.0
        return weights
    excluded = {"no_confidence": 0, "no_ray": 1, "no_reprojection": 2}[mode]
    weights = np.clip(np.prod(np.stack([term for index, term in enumerate(terms) if index != excluded]), axis=0), 0, 1)
    weights[~accepted] = 0
    return weights


def hand_observation_term(*, weight, stage_index, left_loss, right_loss, body_loss):
    """Leave undefined, unused pixel projections outside the fitting graph."""
    if stage_index < 3 or weight == 0:
        return body_loss.detach().new_zeros(())
    return weight * (left_loss + right_loss)


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
        if getattr(args, "body_observation_mode", "legacy") == "reprojection_only":
            losses, pixels = [], []
            for projection, target, confidence, valid in (
                (proj_cam0, raw_left_2d, raw_left_conf, raw_left_valid),
                (proj_cam1, raw_right_2d, raw_right_conf, raw_right_valid),
            ):
                if not valid.any():
                    continue
                # Select before arithmetic: absent NaN projections must not
                # poison either values or gradients via NaN * zero.
                residual = torch.linalg.vector_norm(projection[valid] - target[valid], dim=-1)
                if not torch.isfinite(residual).all():
                    raise RuntimeError("nonfinite supported body reprojection")
                weights = confidence[valid]
                scaled = residual / args.body_reprojection_scale_px
                huber = torch.where(scaled <= 1, 0.5 * scaled.square(), scaled - 0.5)
                multiplier = getattr(args, "body_ankle_reprojection_multiplier", 1.0)
                if multiplier != 1:
                    if valid.shape[-1] != 17:
                        raise ValueError("ankle multiplier requires COCO17 order")
                    joint_weights = torch.ones_like(confidence)
                    joint_weights[..., 15:17] = multiplier
                    huber = huber * joint_weights[valid]
                # Keep the original confidence denominator: this changes only
                # ankle influence, without reducing other joints or prior scale.
                losses.append((huber * weights).sum() / weights.sum())
                pixels.append((residual * weights).sum() / weights.sum())
            if not losses:
                raise RuntimeError("no supported raw body pixels")
            return torch.stack(losses).mean(), torch.stack(pixels).mean()
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
