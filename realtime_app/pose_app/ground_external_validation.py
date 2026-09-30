"""Independent validation metrics for Stage-2 ground pose.

The estimator is never used to manufacture the reference.  Reference poses
must come from an external rigid marker/IMU system and are passed separately.
"""
from __future__ import annotations

import numpy as np


def _check_pose_array(name: str, poses: np.ndarray) -> np.ndarray:
    arr = np.asarray(poses, dtype=float)
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape [T,4,4]")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def rotation_angle_rad(rotation: np.ndarray) -> np.ndarray:
    """Return SO(3) angle for [T,3,3] relative rotations."""
    r = np.asarray(rotation, dtype=float)
    if r.ndim != 3 or r.shape[1:] != (3, 3):
        raise ValueError("rotation must have shape [T,3,3]")
    cosine = np.clip((np.trace(r, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cosine)


def ground_normal_drift(normals: np.ndarray) -> dict:
    """Measure normal drift relative to the first externally defined normal.

    ``normals`` must be supplied by the ground estimator or an independent
    plane tracker.  This function never infers a reference normal from the
    foot or ankle trajectory.
    """
    n = np.asarray(normals, dtype=float)
    if n.ndim != 2 or n.shape[1] != 3 or len(n) == 0:
        raise ValueError("normals must have shape [T,3] with T > 0")
    if not np.isfinite(n).all():
        raise ValueError("normals contains non-finite values")
    length = np.linalg.norm(n, axis=1)
    if np.any(length <= 1e-8):
        raise ValueError("normals contains a zero vector")
    unit = n / length[:, None]
    cosine = np.clip(unit @ unit[0], -1.0, 1.0)
    angles = np.degrees(np.arccos(cosine))
    return {
        "frames": int(len(n)),
        "median_deg": float(np.median(angles)),
        "p95_deg": float(np.percentile(angles, 95)),
        "max_deg": float(np.max(angles)),
        "angles_deg": angles.tolist(),
        "reference": "first_frame_normal",
    }


def failure_reason_counts(reasons: list[str] | np.ndarray) -> dict:
    """Count explicit frame rejection reasons without dropping failures."""
    values = np.asarray(reasons, dtype=object).reshape(-1)
    counts: dict[str, int] = {}
    for value in values.tolist():
        key = "<missing>" if value is None or str(value) == "" else str(value)
        counts[key] = counts.get(key, 0) + 1
    return {"frames": int(len(values)), "counts": dict(sorted(counts.items()))}


def compare_ground_poses(estimated: np.ndarray, reference: np.ndarray) -> dict:
    """Compare estimated and external poses in the same frame convention."""
    est = _check_pose_array("estimated", estimated)
    ref = _check_pose_array("reference", reference)
    if est.shape[0] != ref.shape[0]:
        raise ValueError("estimated and reference frame counts differ")
    rel = np.linalg.inv(ref) @ est
    trans_m = np.linalg.norm(rel[:, :3, 3], axis=1)
    angles = rotation_angle_rad(rel[:, :3, :3])
    return {
        "frames": int(len(est)),
        "translation_median_m": float(np.median(trans_m)),
        "translation_p95_m": float(np.percentile(trans_m, 95)),
        "rotation_median_deg": float(np.degrees(np.median(angles))),
        "rotation_p95_deg": float(np.degrees(np.percentile(angles, 95))),
        "translation_errors_m": trans_m.tolist(),
        "rotation_errors_deg": np.degrees(angles).tolist(),
        "reference_source": "external_required",
    }


def foot_support_speed(foot_points_ground_m: np.ndarray, dt_s: np.ndarray,
                       support_mask: np.ndarray) -> dict:
    """Report speeds only on adjacent frames explicitly marked support."""
    points = np.asarray(foot_points_ground_m, dtype=float)
    dt = np.asarray(dt_s, dtype=float)
    support = np.asarray(support_mask, dtype=bool)
    if points.ndim != 3 or points.shape[1:] != (2, 3):
        raise ValueError("foot_points_ground_m must have shape [T,2,3]")
    if dt.shape != (len(points) - 1,) or support.shape != (len(points), 2):
        raise ValueError("dt or support_mask shape is inconsistent with points")
    if np.any(dt <= 0) or not np.isfinite(points).all():
        raise ValueError("dt must be positive and points finite")
    speeds = np.linalg.norm(np.diff(points, axis=0), axis=2) / dt[:, None]
    pair_mask = support[:-1] & support[1:]
    values = speeds[pair_mask]
    return {
        "support_pairs": int(pair_mask.sum()),
        "median_speed_mps": None if values.size == 0 else float(np.median(values)),
        "p95_speed_mps": None if values.size == 0 else float(np.percentile(values, 95)),
        "unavailable": bool(values.size == 0),
    }


def stage_switch_jumps(poses: np.ndarray, stages: list[str] | np.ndarray) -> dict:
    """Measure pose jumps at observed Stage labels; no relabeling is done."""
    p = _check_pose_array("poses", poses)
    labels = np.asarray(stages, dtype=str)
    if labels.shape != (len(p),):
        raise ValueError("stages must have one label per pose")
    switches = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    if len(switches) == 0:
        return {"switches": 0, "translation_jumps_m": [], "rotation_jumps_deg": []}
    delta = np.linalg.inv(p[switches - 1]) @ p[switches]
    return {
        "switches": int(len(switches)),
        "switch_indices": switches.tolist(),
        "translation_jumps_m": np.linalg.norm(delta[:, :3, 3], axis=1).tolist(),
        "rotation_jumps_deg": np.degrees(rotation_angle_rad(delta[:, :3, :3])).tolist(),
    }


__all__ = ["compare_ground_poses", "failure_reason_counts", "foot_support_speed",
           "ground_normal_drift", "rotation_angle_rad", "stage_switch_jumps"]
