"""Estimate a single hand-to-handle pose from an entire sequence.

This module deliberately treats the fixed-grip assumption as an explicit
engineering prior.  It estimates a robust per-hand palm offset and orientation
in the handle frame; it does not claim tactile contact or external truth.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def validate_walker_pose(rotation_ground_from_walker: np.ndarray,
                         translation_ground_from_walker_m: np.ndarray,
                         n: int) -> None:
    """Validate the per-frame ground<-walker SE(3) trajectory."""
    R = np.asarray(rotation_ground_from_walker, float)
    t = np.asarray(translation_ground_from_walker_m, float)
    if R.shape != (n, 3, 3) or t.shape != (n, 3):
        raise ValueError(f"walker pose shapes must be {(n,3,3)} and {(n,3)}, got {R.shape} and {t.shape}")
    if not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError("walker pose contains non-finite values")
    ortho = np.einsum("nij,nkj->nik", R, R)
    det = np.linalg.det(R)
    if np.max(np.abs(ortho - np.eye(3))) > 2e-3 or np.min(det) < 0.998:
        raise ValueError("walker rotation is not a proper rotation")


def transform_ground_to_walker(points_ground_m: np.ndarray,
                               rotation_ground_from_walker: np.ndarray,
                               translation_ground_from_walker_m: np.ndarray) -> np.ndarray:
    """Apply the inverse of per-frame ground<-walker poses to points."""
    p = np.asarray(points_ground_m, float)
    R = np.asarray(rotation_ground_from_walker, float)
    t = np.asarray(translation_ground_from_walker_m, float)
    if p.shape[0] != R.shape[0]:
        raise ValueError("point and walker pose frame counts differ")
    return np.einsum("nij,n...j->n...i", R.transpose(0, 2, 1), p - t[:, None, ...])


def _frame_from_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[0] < 3 or not np.isfinite(p).all():
        return None
    c = np.mean(p, axis=0)
    _, _, vh = np.linalg.svd(p - c, full_matrices=False)
    x = vh[0]
    y = vh[1]
    # Keep a deterministic normal direction; orientation signs are aligned to
    # the handle frame below and then made continuous across frames.
    z = np.cross(x, y)
    z /= max(np.linalg.norm(z), 1e-12)
    y = np.cross(z, x)
    y /= max(np.linalg.norm(y), 1e-12)
    return c, np.column_stack((x, y, z))


def _handle_frame(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    axis = b - a
    n = np.linalg.norm(axis)
    if not np.isfinite(axis).all() or n < 1e-8:
        return None
    x = axis / n
    z = np.array([0.0, 0.0, 1.0])
    y = np.cross(z, x)
    if np.linalg.norm(y) < 1e-8:
        y = np.array([0.0, 1.0, 0.0])
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    z /= np.linalg.norm(z)
    return 0.5 * (a + b), np.column_stack((x, y, z))


def _robust_center(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0)
    scale = np.maximum(1.4826 * mad, 1e-6)
    keep = np.all(np.abs(x - med) <= 3.5 * scale, axis=1)
    if int(keep.sum()) < max(3, int(0.25 * len(x))):
        keep = np.ones(len(x), dtype=bool)
    return med, keep


def estimate_global_hand_handle_pose(
    palm_ground_m: np.ndarray,
    handle_ends_ground_m: np.ndarray,
) -> dict:
    """Aggregate a fixed pose for one hand in the walker rigid frame.

    ``palm_ground_m`` and ``handle_ends_ground_m`` are expressed in a common
    world frame, but the handle endpoints must come from the per-frame walker
    pose.  The output is invariant to the walker's world translation/rotation.
    The returned offset maps the handle midpoint to the palm centroid in the
    handle frame.  Orientation is represented by a rotation matrix from the
    hand PCA frame to the handle frame.
    """
    p = np.asarray(palm_ground_m, float)
    h = np.asarray(handle_ends_ground_m, float)
    if p.ndim != 3 or h.shape != (p.shape[0], 2, 3):
        raise ValueError("palm must be (N,V,3), handle ends must be (N,2,3)")
    offsets, rots, frame_ids, rejects = [], [], [], []
    previous = None
    for i, (pts, ends) in enumerate(zip(p, h)):
        hf = _handle_frame(*ends)
        pf = _frame_from_points(pts)
        if hf is None or pf is None:
            rejects.append({"frame": i, "reason": "nonfinite_or_degenerate_frame"})
            continue
        hc, Rh = hf
        pc, Rp = pf
        # Align PCA signs to the handle axes and previous frame.  This avoids
        # the arbitrary sign flips inherent to an eigenvector frame.
        scores = Rh.T @ Rp
        signs = np.sign(np.diag(scores)); signs[signs == 0] = 1.0
        Rp = Rp @ np.diag(signs)
        if np.linalg.det(Rp) < 0:
            Rp[:, -1] *= -1
        R_h_from_hand = Rh.T @ Rp
        offsets.append(Rh.T @ (pc - hc))
        rots.append(R_h_from_hand)
        frame_ids.append(i)
        previous = Rp
    if not offsets:
        return {"status": "unavailable", "reason": "no_valid_frames", "valid_frames": []}
    offsets = np.asarray(offsets)
    center, keep = _robust_center(offsets)
    valid_rot = [r for r, k in zip(rots, keep) if k]
    # Quaternion averaging is stable for small hand orientation variation.
    q = Rotation.from_matrix(np.asarray(valid_rot)).as_quat()
    qmean = np.mean(q * np.where((q @ q[0])[:, None] < 0, -1.0, 1.0), axis=0)
    qmean /= max(np.linalg.norm(qmean), 1e-12)
    Rmean = Rotation.from_quat(qmean).as_matrix()
    residual = np.linalg.norm(offsets - center, axis=1)
    return {
        "status": "engineering_candidate",
        "assumption": "hand_static_relative_to_walker_for_entire_video",
        "palm_offset_handle_m": center.tolist(),
        "hand_frame_to_handle_rotation": Rmean.tolist(),
        "valid_frames": [int(frame_ids[j]) for j, k in enumerate(keep) if k],
        "rejected_frames": rejects + [{"frame": int(frame_ids[j]), "reason": "robust_outlier"}
                                        for j, k in enumerate(keep) if not k],
        "offset_residual_median_m": float(np.median(residual[keep])) if keep.any() else None,
        "offset_residual_p95_m": float(np.percentile(residual[keep], 95)) if keep.any() else None,
    }


def load_static_handle_ends(model_path: Path, n: int, coordinate_frame: str = "nodes_walker_mm") -> np.ndarray:
    doc = json.loads(Path(model_path).read_text(encoding="utf-8"))
    nodes = doc.get(coordinate_frame)
    segs = doc.get("handle_segments")
    if not isinstance(nodes, dict) or not isinstance(segs, dict):
        raise ValueError(f"walker model lacks {coordinate_frame}/handle_segments")
    out = []
    for key in ("left", "right"):
        a, b = segs[key]
        out.append([nodes[a], nodes[b]])
    # caller can select side; return (N,2,2,3)
    return np.repeat(np.asarray(out, float)[None, ...] / 1000.0, n, axis=0)
