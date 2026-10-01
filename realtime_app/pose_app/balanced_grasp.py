"""Opt-in staged fitting guards. Anchors are assumptions, never measurements."""
import numpy as np
import torch
import copy

GROUPS = {"left_arm": (5, 7, 9), "right_arm": (6, 8, 10),
          "torso": (5, 6, 11, 12), "left_leg": (11, 13, 15),
          "right_leg": (12, 14, 16), "head": (0, 1, 2, 3, 4)}
LEG_ROTATIONS = (0, 1, 3, 4, 6, 7, 9, 10)


def grouped_observation_loss(predicted, target, quality, valid, scale=.05):
    """Normalize each group independently; unavailable groups contribute zero."""
    residual = torch.where(valid[..., None], predicted-target, torch.zeros_like(predicted))
    squared = residual.square().sum(-1)
    terms = {}
    for name, ids in GROUPS.items():
        weights = torch.where(valid[:,ids], quality[:,ids], 0.)
        terms[name] = (squared[:, ids] * weights).sum() / weights.sum().clamp_min(1e-8) / scale**2
    available = [terms[name] for name, ids in GROUPS.items() if bool(torch.where(valid[:,ids], quality[:,ids], 0.).sum() > 0)]
    return torch.stack(available).mean() if available else residual.sum()*0, terms


def observation_guard(predicted, target, valid, baseline_error, allowance=.01):
    """Per-frame/per-joint cap: improvements elsewhere cannot hide regression."""
    error = torch.linalg.vector_norm(predicted-target, dim=-1)
    excess = torch.where(valid, error-baseline_error-allowance, torch.zeros_like(error))
    finite = bool(torch.isfinite(predicted).all())
    return finite and bool((excess <= 1e-6).all()), float(excess.detach().max())


def leg_temporal_terms(body, ground_coco, frame_valid, dt=1/30):
    pair = frame_valid[:-1] & frame_valid[1:]
    triple = pair[:-1] & pair[1:]
    rotation_delta = torch.where(pair[:,None,None,None], body[1:,LEG_ROTATIONS]-body[:-1,LEG_ROTATIONS], 0.)
    position_delta = ground_coco[2:,11:]-2*ground_coco[1:-1,11:]+ground_coco[:-2,11:]
    position_delta = torch.where(triple[:,None,None], position_delta, 0.)
    rotation = rotation_delta.square().sum((-1,-2)).mean(-1)
    position = (position_delta/dt**2).square().sum(-1).mean(-1)
    return ((rotation*pair).sum()/pair.sum().clamp_min(1)/(2*np.deg2rad(10)**2),
            (position*triple).sum()/triple.sum().clamp_min(1)/10.**2)


def build_stage2_anchors(vertices, stages, frame_valid, sole_sets):
    """Freeze heel/ball patches per full-sequence segment before slicing.

    XY uses segment median placement, but geometry comes from a representative
    actual reference foot. A rigid lift puts its minimum Z on the plane. Each
    selected vertex is anchored, so rotation about a patch centroid is visible.
    This is a designed anchor, separate from the old support state machine.
    """
    n = len(vertices)
    patch_ids = np.zeros((n, 4, 3), dtype=np.int64)
    targets = np.zeros((n, 4, 3, 3))
    segment_ids = np.full(n, -1, dtype=np.int64)
    active = np.asarray(stages) == "stage2_feet_static_walker_moving"
    records = []
    for segment, start in enumerate(np.flatnonzero(active & ~np.r_[False, active[:-1]])):
        stop = start+1
        while stop < n and active[stop]:
            stop += 1
        usable = np.flatnonzero(frame_valid[start:stop])+start
        if len(usable) < 3:
            records.append(dict(start=int(start), stop=int(stop), available=False, reason="fewer_than_3_valid_frames"))
            continue
        rows, anchors, lifts, representative_frames = [], [], [], []
        for regions in sole_sets:
            sole = np.unique(sum(regions.values(), []))
            # One actual reference frame preserves the complete foot geometry;
            # coordinate-wise medians of vertices need not form a rigid foot.
            centers = vertices[usable][:,sole].mean(1)
            center = np.median(centers,axis=0)
            chosen_frame = int(usable[np.argmin(np.linalg.norm(centers-center,axis=1))])
            shift = center-vertices[chosen_frame,sole].mean(0)
            shift[2] = -vertices[chosen_frame,sole,2].min()
            lift = float(shift[2])
            lifts.append(lift)
            representative_frames.append(chosen_frame)
            for region in ("heel", "ball"):
                ids = np.unique(np.asarray(regions[region], dtype=int))
                if len(ids) < 3:
                    raise ValueError("heel/ball patch requires at least three vertices")
                order = np.argsort(np.median(vertices[usable][:, ids, 2],axis=0),kind="stable")[:3]
                chosen = ids[order]
                anchor = vertices[chosen_frame,chosen] + shift
                rows.append(chosen); anchors.append(anchor)
        patch_ids[start:stop] = rows; targets[start:stop] = anchors
        segment_ids[usable] = segment
        records.append(dict(start=int(start),stop=int(stop),available=True,
                            rigid_lift_m=lifts, representative_frames=representative_frames,
                            patch_ids=np.asarray(rows).tolist(), targets_m=np.asarray(anchors).tolist()))
    return patch_ids, targets, segment_ids, records


def stage2_terms(vertices, patch_ids, targets, segments, dt=1/30):
    points = vertices[torch.arange(len(vertices),device=vertices.device)[:,None,None], patch_ids]
    if targets.shape != points.shape:
        raise ValueError("Stage2 requires individual vertex targets, not patch centroids")
    active = segments >= 0
    pair = active[1:] & active[:-1] & (segments[1:]==segments[:-1])
    delta = torch.where(active[:,None,None,None],points-targets,0.)
    motion = torch.where(pair[:,None,None,None],points[1:]-points[:-1],0.)
    distance = (delta/.01).square().sum(-1).mean((-1,-2))
    velocity = (motion/dt/.05).square().sum(-1).mean((-1,-2))
    return (distance*active).sum()/active.sum().clamp_min(1), (velocity*pair).sum()/pair.sum().clamp_min(1)


def guarded_adam_step(optimizer, params, evaluate, original_loss, trials=8):
    """Backtrack a proposed Adam step; rejected updates restore moments too.

    evaluate returns (scalar objective, guard_ok, diagnostics). The objective
    and collision pairs must stay frozen within one transaction.
    """
    old = [p.detach().clone() for p in params]
    moments = copy.deepcopy(optimizer.state_dict())
    attempts = []
    committed = False
    try:
        optimizer.step()
        proposed = [p.detach().clone() for p in params]
        with torch.no_grad():
            for i in range(trials):
                alpha = 2.**(-i)
                for p, a, b in zip(params, old, proposed):
                    p.copy_(a + alpha*(b-a))
                loss, ok, diagnostic = evaluate()
                accepted = bool(ok and torch.isfinite(loss) and float(loss) <= float(original_loss) + 1e-6)
                attempts.append(dict(alpha=alpha, loss=float(loss), accepted=accepted, **diagnostic))
                if accepted:
                    committed = True
                    return dict(accepted=True, alpha=alpha, attempts=attempts)
        return dict(accepted=False, alpha=0., attempts=attempts)
    finally:
        if not committed:
            with torch.no_grad():
                for p, value in zip(params, old):
                    p.copy_(value)
            optimizer.load_state_dict(moments)
