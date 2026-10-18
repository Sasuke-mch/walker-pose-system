"""Per-view 2D observation partition for the dual-fisheye pose fits.

Every L/R raw-fisheye keypoint is classified independently into one of:

  - accepted_strict             (upstream strict flag, otherwise valid)
  - accepted_relaxed            (non-strict but usable, lower weight)
  - missing_or_nonfinite        (pixel not finite)
  - outside_raw_image           (pixel outside the raw image bounds)
  - low_score                   (detection score below the minimum)
  - upstream_non_strict_high_error (non-strict with missing or >10 px error)

Rules (frozen; do not reorder silently):
  1. non-finite pixel -> reject (missing_or_nonfinite);
  2. pixel outside the raw fisheye image -> reject (outside_raw_image);
  3. detection score < minimum_score -> reject (low_score);
  4. strict point -> accept (accepted_strict);
  5. non-strict with empty/None upstream error or >10 px -> reject
     (upstream_non_strict_high_error);
  6. other non-strict points -> accept with a low weight
     (accepted_relaxed).

The two views are decided independently: a good point in one view never
silently upgrades the other view. All rejections keep their reason; nothing is
deleted. A post-fit reprojection outlier may only be used as a *next-frame*
downweight, never to delete points inside the same optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Fixed rejection / acceptance reason strings (contract values).
REASON_ACCEPTED_STRICT = "accepted_strict"
REASON_ACCEPTED_RELAXED = "accepted_relaxed"
REASON_MISSING_OR_NONFINITE = "missing_or_nonfinite"
REASON_OUTSIDE_RAW_IMAGE = "outside_raw_image"
REASON_LOW_SCORE = "low_score"
REASON_UPSTREAM_NON_STRICT_HIGH_ERROR = "upstream_non_strict_high_error"

ALL_REASONS = (
    REASON_ACCEPTED_STRICT,
    REASON_ACCEPTED_RELAXED,
    REASON_MISSING_OR_NONFINITE,
    REASON_OUTSIDE_RAW_IMAGE,
    REASON_LOW_SCORE,
    REASON_UPSTREAM_NON_STRICT_HIGH_ERROR,
)

# Default upstream error threshold (px) above which a non-strict point is
# rejected. Frozen for this experiment.
UPSTREAM_ERROR_THRESHOLD_PX = 10.0


@dataclass(frozen=True)
class ViewObservationDecision:
    """Classification of one keypoint in one view."""

    accepted: bool
    reason: str
    score: Optional[float]
    upstream_strict: bool
    upstream_reprojection_error_px: Optional[float]


def _as_optional_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def classify_view_observation(
    pixel_xy: np.ndarray,
    score: float,
    upstream_strict: bool,
    upstream_reprojection_error_px: Optional[float],
    image_width: int,
    image_height: int,
    minimum_score: float = 0.20,
) -> ViewObservationDecision:
    """Classify a single L or R keypoint for the current view.

    Args:
        pixel_xy: (2,) pixel coordinates (x, y) in the raw fisheye image.
        score: detection score for this keypoint (None tolerated).
        upstream_strict: frozen per-joint strict flag of the upstream stereo
            gate. The same flag is applied to each view independently.
        upstream_reprojection_error_px: frozen per-joint upstream reprojection
            error in px (None means unknown).
        image_width: raw image width in pixels.
        image_height: raw image height in pixels.
        minimum_score: minimum accepted detection score.
    """
    error = _as_optional_float(upstream_reprojection_error_px)
    score_f = _as_optional_float(score)
    pixels = np.asarray(pixel_xy, dtype=np.float64).reshape(-1)
    finite = pixels.shape == (2,) and bool(np.all(np.isfinite(pixels)))
    if not finite:
        return ViewObservationDecision(False, REASON_MISSING_OR_NONFINITE, score_f, bool(upstream_strict), error)
    x, y = float(pixels[0]), float(pixels[1])
    in_bounds = 0.0 <= x < float(image_width) and 0.0 <= y < float(image_height)
    if not in_bounds:
        return ViewObservationDecision(False, REASON_OUTSIDE_RAW_IMAGE, score_f, bool(upstream_strict), error)
    if score_f is None or score_f < minimum_score:
        return ViewObservationDecision(False, REASON_LOW_SCORE, score_f, bool(upstream_strict), error)
    if bool(upstream_strict):
        return ViewObservationDecision(True, REASON_ACCEPTED_STRICT, score_f, True, error)
    if error is None or error > UPSTREAM_ERROR_THRESHOLD_PX:
        return ViewObservationDecision(False, REASON_UPSTREAM_NON_STRICT_HIGH_ERROR, score_f, False, error)
    return ViewObservationDecision(True, REASON_ACCEPTED_RELAXED, score_f, False, error)


def classify_view_observations_batch(
    pixel_xy: np.ndarray,
    scores: np.ndarray,
    upstream_strict: np.ndarray,
    upstream_reprojection_error_px: np.ndarray,
    image_width: int,
    image_height: int,
    minimum_score: float = 0.20,
) -> dict[str, np.ndarray]:
    """Vectorised per-view classification over (N, J) keypoint arrays.

    Args:
        pixel_xy: (N, J, 2) pixel coordinates; NaN marks missing points.
        scores: (N, J) detection scores.
        upstream_strict: (N, J) bool frozen strict flags.
        upstream_reprojection_error_px: (N, J) upstream errors; NaN = unknown.
    Returns a dict with accepted (N,J) bool, reasons (N,J) object array and
    per-reason counts.
    """
    pixels = np.asarray(pixel_xy, dtype=np.float64)
    scores_a = np.asarray(scores, dtype=np.float64)
    strict_a = np.asarray(upstream_strict, dtype=bool)
    errors_a = np.asarray(upstream_reprojection_error_px, dtype=np.float64)
    if pixels.ndim != 3 or pixels.shape[2] != 2:
        raise ValueError("pixel_xy must have shape (N, J, 2)")
    shape = pixels.shape[:2]
    accepted = np.zeros(shape, dtype=bool)
    reasons = np.empty(shape, dtype=object)
    finite = np.all(np.isfinite(pixels), axis=2)
    x, y = pixels[..., 0], pixels[..., 1]
    in_bounds = (
        (x >= 0.0) & (x < float(image_width)) & (y >= 0.0) & (y < float(image_height))
    )
    score_ok = np.isfinite(scores_a) & (scores_a >= minimum_score)
    error_known = np.isfinite(errors_a)
    high_error = error_known & (errors_a > UPSTREAM_ERROR_THRESHOLD_PX)

    reasons[~finite] = REASON_MISSING_OR_NONFINITE
    reasons[finite & ~in_bounds] = REASON_OUTSIDE_RAW_IMAGE
    reasons[finite & in_bounds & ~score_ok] = REASON_LOW_SCORE
    reasons[finite & in_bounds & score_ok & ~strict_a & (~error_known | high_error)] = (
        REASON_UPSTREAM_NON_STRICT_HIGH_ERROR
    )
    reasons[finite & in_bounds & score_ok & strict_a] = REASON_ACCEPTED_STRICT
    reasons[finite & in_bounds & score_ok & ~strict_a & error_known & ~high_error] = (
        REASON_ACCEPTED_RELAXED
    )
    accepted[:] = (reasons == REASON_ACCEPTED_STRICT) | (reasons == REASON_ACCEPTED_RELAXED)

    counts = {reason: int(np.count_nonzero(reasons == reason)) for reason in ALL_REASONS}
    return {
        "accepted": accepted,
        "reasons": reasons,
        "rejection_counts": counts,
    }
