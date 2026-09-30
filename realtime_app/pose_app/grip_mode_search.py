"""Independent search for plausible fixed grip modes.

The search does not read a previously solved hand pose.  It uses only hand
surface observations in the walker frame and the walker handle geometry.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

from pose_app.global_hand_handle_pose import _frame_from_points, _handle_frame, _robust_center


def _axis_angle(x: np.ndarray, angle: float) -> np.ndarray:
    return Rotation.from_rotvec(np.asarray(x, float) * angle).as_matrix()


def search_grip_modes(palm_walker_m: np.ndarray, handle_ends_walker_m: np.ndarray,
                      radius_m: float = .016) -> dict:
    """Rank handle-axis grip hypotheses using all frames jointly.

    Modes differ by palm roll around the handle axis and by which side of the
    handle the palm normal faces. Scores are diagnostics, not contact truth.
    """
    p = np.asarray(palm_walker_m, float); h = np.asarray(handle_ends_walker_m, float)
    if p.ndim != 3 or h.shape != (p.shape[0], 2, 3):
        raise ValueError("palm=(N,V,3), handles=(N,2,3) required")
    centers, frames, gaps = [], [], []
    for pts, ends in zip(p, h):
        hf, pf = _handle_frame(*ends), _frame_from_points(pts)
        if hf is None or pf is None:
            centers.append(None); frames.append(None); gaps.append(None); continue
        hc, Rh = hf; pc, Rp = pf
        centers.append(Rh.T @ (pc - hc)); frames.append(Rh.T @ Rp)
        axis = ends[1] - ends[0]; q = pts - ends[0]
        u = np.clip(np.sum(q * axis, axis=1) / max(float(axis @ axis), 1e-9), 0, 1)
        nearest = ends[0] + u[:, None] * axis
        gaps.append(np.linalg.norm(pts - nearest, axis=1) - radius_m)
    valid = np.array([x is not None for x in centers])
    if valid.sum() < 3: return {"status": "unavailable", "reason": "fewer_than_three_valid_frames"}
    C = np.asarray([x for x in centers if x is not None]); F = np.asarray([x for x in frames if x is not None])
    offset, keep = _robust_center(C)
    modes = []
    for side in (-1, 1):
        for roll_deg in (0, 90, 180, 270):
            Rtarget = _axis_angle(np.array([1., 0., 0.]), np.deg2rad(roll_deg))
            Rtarget[:, 2] *= side
            if np.linalg.det(Rtarget) < 0: Rtarget[:, -1] *= -1
            # Orientation residual is the angle from each observed palm frame
            # to the candidate fixed hand frame.
            angles = []
            for f in F:
                angles.append(Rotation.from_matrix(Rtarget.T @ f).magnitude())
            all_gaps = np.concatenate([g for g in gaps if g is not None])
            contact = np.mean((all_gaps >= -0.003) & (all_gaps <= .03))
            penetration = np.mean(np.maximum(-all_gaps, 0))
            score = float(np.median(angles) + np.linalg.norm(C[keep] - offset).mean() / .05
                          + 2.0 * penetration - .25 * contact)
            modes.append({"mode": f"normal_{'out' if side > 0 else 'in'}_roll_{roll_deg}",
                          "handle_offset_m": offset.tolist(),
                          "orientation_residual_median_rad": float(np.median(angles)),
                          "offset_residual_median_m": float(np.median(np.linalg.norm(C[keep]-offset,axis=1))),
                          "contact_band_fraction": float(contact),
                          "penetration_mean_m": float(penetration),
                          "score": score})
    modes.sort(key=lambda x: x["score"])
    return {"status": "engineering_candidate", "valid_frames": int(valid.sum()),
            "rejected_frames": np.flatnonzero(~valid).astype(int).tolist(),
            "radius_m": radius_m, "modes": modes}


def refine_grip_modes(palm_walker_m, handle_ends_walker_m,
                      radius_m=.016, starts=24, bootstrap=0, seed=20260930):
    """Fit a rigid real hand surface with capsule nonpenetration constraints.

    This estimates placement of an existing articulated hand, not new finger
    articulation. Per-vertex temporal medians are observation targets only;
    the output is a rigid transformation of one actual frame's surface.
    """
    from scipy.optimize import minimize
    p = np.asarray(palm_walker_m, float)
    h = np.asarray(handle_ends_walker_m, float)
    if p.ndim != 3 or h.shape != (len(p), 2, 3):
        raise ValueError('invalid sequence shapes')
    valid = np.isfinite(p).all(axis=(1,2)) & np.isfinite(h).all(axis=(1,2))
    ids = np.flatnonzero(valid)
    if len(ids) < 3:
        return {'status':'unavailable','rejected_frames':np.flatnonzero(~valid).tolist()}
    p = p[valid]; h = h[valid]
    if not np.allclose(h,h[0],atol=1e-6):
        raise ValueError('rigid search requires walker-frame fixed handle endpoints')
    target = np.median(p,axis=0)
    medoid = int(np.argmin(np.mean(np.linalg.norm(p-target,axis=-1),axis=1)))
    base = p[medoid]; center = base.mean(axis=0)
    a,b = h[0]; axis = b-a
    if axis@axis < 1e-10: raise ValueError('degenerate handle')
    def surface(x):
        return (base-center)@Rotation.from_rotvec(x[3:]).as_matrix().T+center+x[:3]
    def gap(q):
        u=np.clip((q-a)@axis/(axis@axis),0,1)
        return np.linalg.norm(q-a-u[:,None]*axis,axis=-1)-radius_m
    def objective(x):
        r=np.linalg.norm(surface(x)-target,axis=-1)/.03
        return float(np.mean(np.sqrt(1+r*r)-1))
    def constraints(x):
        g=gap(surface(x))
        return np.r_[g+.0005,.003-g.min()]
    rng=np.random.default_rng(seed); candidates=[]
    for k in range(starts):
        x0=np.zeros(6) if k==0 else np.r_[rng.normal(0,.025,3),rng.normal(0,.2,3)]
        sol=minimize(objective,x0,method='SLSQP',bounds=[(-.10,.10)]*3+[(-.65,.65)]*3,
                     constraints=[{'type':'ineq','fun':constraints}],
                     options={'maxiter':500,'ftol':1e-10})
        q=surface(sol.x); g=gap(q)
        feasible=bool(g.min()>=-.00051 and g.min()<=.00301)
        candidates.append({'start':k,'success':bool(sol.success),'feasible':feasible,
                           'message':str(sol.message),'objective':objective(sol.x),
                           'parameters':sol.x.tolist(),'min_gap_mm':float(g.min()*1000)})
    good=[c for c in candidates if c['feasible'] and c['success']]
    if not good:
        return {'status':'no_feasible_solution','candidates':candidates}
    good.sort(key=lambda c:c['objective']); best=good[0]; q=surface(np.asarray(best['parameters'])); g=gap(q)
    errors=np.linalg.norm(p-q,axis=-1)*1000
    half_targets=[np.median(p[:len(p)//2],axis=0),np.median(p[len(p)//2:],axis=0)]
    modes=[]
    for c in good:
        qc=surface(np.asarray(c['parameters']))
        if all(np.sqrt(np.mean((qc-surface(np.asarray(m['parameters'])))**2))>.002 for m in modes): modes.append(c)
    return {'status':'engineering_rigid_surface_candidate','reference_frame':int(ids[medoid]),
            'method':'multistart_SLSQP_rigid_surface_to_temporal_median',
            'not_an_articulated_grasp_solver':True,'valid_frames':ids.tolist(),
            'rejected_frames':np.flatnonzero(~valid).tolist(),'starts':starts,
            'contact_tolerance_mm':3,'penetration_tolerance_mm':.5,
            'best':best,'distinct_candidates':modes,'all_starts':candidates,
            'candidate_margin':None if len(modes)<2 else modes[1]['objective']-modes[0]['objective'],
            'shared_surface_walker_m':q.tolist(),
            'diagnostics':{'min_gap_mm':float(g.min()*1000),
                           'max_penetration_mm':float(np.maximum(-g,0).max()*1000),
                           'within_3mm_vertex_fraction':float(np.mean(abs(g)<=.003)),
                           'observation_error_median_mm':float(np.median(errors)),
                           'observation_error_p95_mm':float(np.percentile(errors,95)),
                           'half_sequence_target_difference_median_mm':float(np.median(np.linalg.norm(half_targets[0]-half_targets[1],axis=-1))*1000),
                           'initial_max_penetration_mm':float(np.maximum(-gap(base),0).max()*1000)}}
