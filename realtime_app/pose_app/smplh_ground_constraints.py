"""SMPL-H surface geometry for explicit Stage-conditioned ground assumptions.

These functions consume model vertices in one fixed ground frame. They do not
infer contact, move vertices, select stages, or turn Stage into independent world
truth. In particular the historical heel vertices are above the sole: their
anchors must retain their fitted height, rather than being projected to z=0.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HEEL_VERTEX_IDS = (3387, 6787)


def _sole_indices(sole_indices) -> tuple[np.ndarray, np.ndarray]:
    if len(sole_indices) != 2:
        raise ValueError("expected left and right sole candidate indices")
    result = []
    for values in sole_indices:
        indices = np.asarray(values)
        if (indices.ndim != 1 or not indices.size or indices.dtype.kind not in "iu"
                or np.any(indices < 0) or np.any(indices >= 6890)):
            raise ValueError("sole candidates must be nonempty SMPL-H integer indices")
        result.append(np.unique(indices))
    if len(result[0]) != len(result[1]):
        raise ValueError("left and right sole candidates must have equal vertex counts")
    return tuple(result)


def load_sole_vertex_indices(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the existing heel/ball/toe surface candidates, in left/right order."""
    sets = json.loads(Path(path).read_text(encoding="utf-8"))["sets"]
    return _sole_indices([
        sum(sets[f"{side}_sole_surface_candidate"].values(), [])
        for side in ("left", "right")
    ])


def foot_geometry(vertices_ground_m: np.ndarray, sole_indices) -> dict[str, np.ndarray]:
    """Return actual heels, sole minima, and the original soft surface objective.

    Input is [...,6890,3], all in metres. The original surface objective uses
    centered softmin (temperature 5 mm), pseudo-Huber (delta 10 mm), and mean
    squared negative vertex heights with weight 1. Its zero is not a certificate
    that the exact minimum vertex height is zero.
    """
    vertices = np.asarray(vertices_ground_m, dtype=float)
    if vertices.shape[-2:] != (6890, 3) or not np.isfinite(vertices).all():
        raise ValueError("expected finite [...,6890,3] SMPL-H ground vertices")
    soles = _sole_indices(sole_indices)
    minimum, softminimum, surface_loss, points = [], [], [], []
    temperature, delta = .005, .010
    for indices in soles:
        selected = np.take(vertices, indices, axis=-2)
        points.append(selected)
        z = selected[..., 2]
        lowest = z.min(axis=-1)
        centered = lowest - temperature * np.log(
            np.exp(-(z - lowest[..., None]) / temperature).mean(axis=-1)
        )
        robust = delta * delta * (np.hypot(1., centered / delta) - 1.)
        minimum.append(lowest)
        softminimum.append(centered)
        surface_loss.append(robust + np.square(np.minimum(z, 0.)).mean(axis=-1))
    sole_min = np.stack(minimum, axis=-1)
    heels = np.take(vertices, HEEL_VERTEX_IDS, axis=-2)
    return {
        "heel_m": heels,
        "sole_points_m": np.stack(points, axis=-3),
        "sole_min_m": sole_min,
        "sole_softmin_m": np.stack(softminimum, axis=-1),
        "sole_surface_loss_m2": np.stack(surface_loss, axis=-1),
        "sole_penetration_m": np.maximum(-sole_min, 0.),
        "heel_height_above_sole_m": heels[..., 2] - sole_min,
    }


def ground_inequalities(
    geometry: dict[str, np.ndarray], *, stage2: bool,
    heel_anchors_m: np.ndarray | None = None,
    sole_anchors_m: np.ndarray | None = None,
    ground_tolerance_m: float = .005, max_heel_displacement_m: float = .010,
) -> np.ndarray:
    """Return c >= 0: sole lower2, Stage2 upper2, optional heel2 and sole2 balls.

    Both feet may rise outside Stage2. During Stage2 the exact selected sole
    minima stay within the ground tolerance. Heel movement is measured from a
    fixed episode anchor in a 3D Euclidean ball, not a per-axis box or a previous
    frame. With sole anchors each selected surface vertex also stays in its own
    ball, which prevents rotating the foot around a fixed heel. The same 10 mm
    radius applies to both anchor sets. Callers must preserve episode anchors
    across ambiguous stage gaps.
    No anchor is imposed before a first contact-feasible pose has been solved.
    """
    if (not np.isfinite(ground_tolerance_m) or ground_tolerance_m < 0
            or not np.isfinite(max_heel_displacement_m) or max_heel_displacement_m <= 0):
        raise ValueError("ground tolerance must be nonnegative and heel radius positive")
    lowest = np.asarray(geometry["sole_min_m"], dtype=float)
    heels = np.asarray(geometry["heel_m"], dtype=float)
    if (lowest.shape[-1:] != (2,) or heels.shape != lowest.shape + (3,)
            or not np.isfinite(lowest).all() or not np.isfinite(heels).all()):
        raise ValueError("invalid finite left/right foot geometry")
    terms = [lowest + ground_tolerance_m]
    if stage2:
        terms.append(ground_tolerance_m - lowest)
        if heel_anchors_m is not None:
            raw_anchors = np.asarray(heel_anchors_m, dtype=float)
            if raw_anchors.shape[-2:] != (2, 3):
                raise ValueError("heel anchors must have final shape [2,3]")
            try:
                anchors = np.broadcast_to(raw_anchors, heels.shape)
            except ValueError as error:
                raise ValueError("heel anchors must broadcast to [...,2,3] heels") from error
            if not np.isfinite(anchors).all():
                raise ValueError("heel anchors must be finite")
            terms.append(max_heel_displacement_m - np.linalg.norm(heels - anchors, axis=-1))
        if sole_anchors_m is not None:
            points = np.asarray(geometry["sole_points_m"], dtype=float)
            raw_anchors = np.asarray(sole_anchors_m, dtype=float)
            if raw_anchors.shape[-3:] != points.shape[-3:]:
                raise ValueError("sole anchors must have final shape [2,K,3]")
            try:
                anchors = np.broadcast_to(raw_anchors, points.shape)
            except ValueError as error:
                raise ValueError("sole anchors must broadcast to [...,2,K,3] sole points") from error
            if (points.shape[:-2] != lowest.shape or points.shape[-1:] != (3,)
                    or points.shape[-2] <= 0 or not np.isfinite(points).all()
                    or not np.isfinite(anchors).all()):
                raise ValueError("sole anchors must match finite [...,2,K,3] sole points")
            movement = np.linalg.norm(points - anchors, axis=-1).max(axis=-1)
            terms.append(max_heel_displacement_m - movement)
    elif heel_anchors_m is not None or sole_anchors_m is not None:
        raise ValueError("foot anchors may only be imposed during Stage2")
    return np.concatenate(terms, axis=-1)
