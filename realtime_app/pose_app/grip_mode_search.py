"""Independent search for plausible fixed grip modes.

The search does not read a previously solved hand pose.  It uses only hand
surface observations in the walker frame and the walker handle geometry.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

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
