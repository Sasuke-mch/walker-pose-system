"""Shared 2D-reprojection helpers for the SMPL-X and skeleton factor-graph fits.

The observation layer is the frozen per-view partition of
``pose_observation_partition``: every L/R keypoint is classified as
accepted_strict / accepted_relaxed / rejected-with-reason. Weights are derived
from the partition only -- never from any 3D target position.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pose_app.pose_observation_partition import (
    REASON_ACCEPTED_RELAXED,
    REASON_ACCEPTED_STRICT,
    classify_view_observations_batch,
)


def acceleration_p95_mm(points_m: np.ndarray) -> float:
    acceleration = points_m[2:] - 2.0 * points_m[1:-1] + points_m[:-2]
    return float(np.percentile(np.linalg.norm(acceleration, axis=2), 95) * 1000.0)


def robust_gm(residual_m: "np.ndarray | Any", scale_m: float = 0.03) -> "Any":
    """Geman-McClure robust loss: r^2 / (r^2 + c^2), bounded, outlier-robust."""
    squared = (residual_m * residual_m).sum(dim=-1)
    return squared / (squared + scale_m * scale_m)


def view_2d_weights(
    scores: np.ndarray,
    accepted: np.ndarray,
    strict: np.ndarray,
    errors_px: np.ndarray,
    relaxed_weight: float = 0.25,
    strict_error_scale_px: float = 5.0,
) -> np.ndarray:
    """Per-(frame, body-joint) 2D observation weight for one view.

    Weights come from the frozen per-view partition and frozen upstream
    quality only:
      - strict accepted points:  score^2 / (1 + (err / 5)^2)
      - relaxed accepted points: 0.25 * score^2  (flat, strictly lower)
      - rejected points:         0.0
    No 3D target position is read here.
    """
    scores = np.asarray(scores, dtype=np.float64)
    accepted = np.asarray(accepted, dtype=bool)
    strict = np.asarray(strict, dtype=bool)
    errors = np.asarray(errors_px, dtype=np.float64)
    safe = np.where(np.isfinite(errors), errors, 30.0)
    score_factor = np.clip(scores, 0.0, 1.0) ** 2
    strict_weight = score_factor / (1.0 + (safe / strict_error_scale_px) ** 2)
    relaxed_weight_v = relaxed_weight * score_factor
    weights = np.where(accepted & strict, strict_weight, 0.0)
    weights = np.where(accepted & ~strict, relaxed_weight_v, weights)
    return weights


def build_view_2d_weights_from_obs(obs2d: dict, relaxed_weight: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: weights for both views from a build_2d_observations dict."""
    left = view_2d_weights(
        obs2d["score_left"], obs2d["accepted_left"], obs2d["strict_left"], obs2d["errors_left_px"],
        relaxed_weight=relaxed_weight,
    )
    right = view_2d_weights(
        obs2d["score_right"], obs2d["accepted_right"], obs2d["strict_right"], obs2d["errors_right_px"],
        relaxed_weight=relaxed_weight,
    )
    return left, right


def build_2d_observations(
    stereo_rows_by_pair: dict[int, dict],
    frames: list[dict],
    window_indices: np.ndarray,
    body_coco: np.ndarray,
    image_width: int,
    image_height: int,
    minimum_score: float = 0.20,
) -> dict:
    """Collect per-frame L/R raw-fisheye 2D keypoints and per-view partition.

    Each view is classified independently. Raw pixels, scores, strict flags,
    upstream errors and rejection reasons are all preserved. ``valid_*`` is an
    alias of ``accepted_*`` for backward compatibility.
    """
    kp_left = np.full((len(window_indices), len(body_coco), 2), np.nan)
    kp_right = np.full_like(kp_left, np.nan)
    score_left = np.full((len(window_indices), len(body_coco)), np.nan)
    score_right = np.full_like(score_left, np.nan)
    strict_left = np.zeros_like(score_left, dtype=bool)
    strict_right = np.zeros_like(score_left, dtype=bool)
    errors_left = np.full_like(score_left, np.nan)
    errors_right = np.full_like(score_left, np.nan)
    pair_ids = []
    for local, frame_index in enumerate(window_indices):
        pair_id = int(frames[frame_index]["pair"])
        pair_ids.append(pair_id)
        row = stereo_rows_by_pair.get(pair_id)
        if row is None or not row["left"]["persons"] or not row["right"]["persons"]:
            continue
        left_kp = np.asarray(row["left"]["persons"][0]["keypoints"], dtype=np.float64)
        right_kp = np.asarray(row["right"]["persons"][0]["keypoints"], dtype=np.float64)
        for position, coco_index in enumerate(body_coco):
            lv = left_kp[coco_index]
            rv = right_kp[coco_index]
            if np.all(np.isfinite(lv[:2])):
                kp_left[local, position] = lv[:2]
                score_left[local, position] = float(lv[2])
            if np.all(np.isfinite(rv[:2])):
                kp_right[local, position] = rv[:2]
                score_right[local, position] = float(rv[2])
            strict_l = bool(frames[frame_index]["strict"][coco_index])
            strict_r = bool(frames[frame_index]["strict"][coco_index])
            err_l = frames[frame_index]["errors_px"][coco_index]
            err_r = frames[frame_index]["errors_px"][coco_index]
            strict_left[local, position] = strict_l
            strict_right[local, position] = strict_r
            if err_l is not None:
                errors_left[local, position] = float(err_l)
            if err_r is not None:
                errors_right[local, position] = float(err_r)

    part_l = classify_view_observations_batch(
        kp_left, score_left, strict_left, errors_left, image_width, image_height, minimum_score
    )
    part_r = classify_view_observations_batch(
        kp_right, score_right, strict_right, errors_right, image_width, image_height, minimum_score
    )
    weights_left = view_2d_weights(score_left, part_l["accepted"], strict_left, errors_left)
    weights_right = view_2d_weights(score_right, part_r["accepted"], strict_right, errors_right)
    return {
        "pair_ids": pair_ids,
        "kp_left_px": kp_left,
        "kp_right_px": kp_right,
        "score_left": score_left,
        "score_right": score_right,
        "strict_left": strict_left,
        "strict_right": strict_right,
        "errors_left_px": errors_left,
        "errors_right_px": errors_right,
        "valid_left": part_l["accepted"],
        "valid_right": part_r["accepted"],
        "accepted_left": part_l["accepted"],
        "accepted_right": part_r["accepted"],
        "reasons_left": part_l["reasons"],
        "reasons_right": part_r["reasons"],
        "rejection_counts_left": part_l["rejection_counts"],
        "rejection_counts_right": part_r["rejection_counts"],
        "weights_left": weights_left,
        "weights_right": weights_right,
    }
