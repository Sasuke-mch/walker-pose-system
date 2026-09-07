from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np


OUT_OF_RAW_IMAGE_BOUNDS = "out_of_raw_image_bounds"
NON_FINITE_2D_POINT = "non_finite_2d_point"
MISSING_2D_POINT = "missing_2d_point"


class RawImageBoundsError(ValueError):
    """Raised before an invalid raw-camera point reaches calibrated geometry."""


def raw_point_rejection_reason(
    point: Sequence[float] | np.ndarray | None,
    image_size: tuple[int, int],
) -> str | None:
    """Return the coordinate-domain rejection reason for one raw pixel point.

    Pixel coordinates use the closed sensor domain ``[0, width - 1]`` by
    ``[0, height - 1]``.  Scores are deliberately not inspected here: callers
    must preserve an out-of-bounds observation as a distinct geometry failure,
    rather than relabelling it as a low-confidence point.
    """

    try:
        missing = point is None or len(point) < 2
    except TypeError:
        missing = True
    if missing:
        return MISSING_2D_POINT
    try:
        x, y = float(point[0]), float(point[1])
    except (TypeError, ValueError):
        return NON_FINITE_2D_POINT
    if not math.isfinite(x) or not math.isfinite(y):
        return NON_FINITE_2D_POINT
    width, height = image_size
    if not (0.0 <= x <= width - 1 and 0.0 <= y <= height - 1):
        return OUT_OF_RAW_IMAGE_BOUNDS
    return None


def paired_raw_point_rejection_reason(
    left_point: Sequence[float] | np.ndarray | None,
    right_point: Sequence[float] | np.ndarray | None,
    left_image_size: tuple[int, int],
    right_image_size: tuple[int, int],
) -> str | None:
    """Return the first raw-coordinate rejection reason for a stereo pair."""

    left_reason = raw_point_rejection_reason(left_point, left_image_size)
    right_reason = raw_point_rejection_reason(right_point, right_image_size)
    if OUT_OF_RAW_IMAGE_BOUNDS in (left_reason, right_reason):
        return OUT_OF_RAW_IMAGE_BOUNDS
    if NON_FINITE_2D_POINT in (left_reason, right_reason):
        return NON_FINITE_2D_POINT
    if MISSING_2D_POINT in (left_reason, right_reason):
        return MISSING_2D_POINT
    return None


def count_out_of_raw_image_bounds(
    persons: Iterable[object], image_size: tuple[int, int]
) -> int:
    """Count saved 2-D observations rejected by the raw-camera domain guard."""

    return sum(
        raw_point_rejection_reason(point, image_size) == OUT_OF_RAW_IMAGE_BOUNDS
        for person in persons
        for point in getattr(person, "keypoints", [])
    )


def require_raw_image_points_in_bounds(
    points: np.ndarray, image_size: tuple[int, int], side: str
) -> None:
    """Hard-stop generic calibration calls that bypass a model-specific audit."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    for index, point in enumerate(points):
        reason = raw_point_rejection_reason(point, image_size)
        if reason is not None:
            raise RawImageBoundsError(
                f"{side} raw point {index} rejected before calibrated geometry: {reason}"
            )
