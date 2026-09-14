"""Conservative image-space propagation for manually audited floor masks.

This module transports a seed mask one adjacent upright-video frame at a time
with backward dense optical flow.  It deliberately has no stereo, calibration,
metric-depth, contact, or ground-plane dependency.  A propagated mask is only
an image-space candidate until an independent manual label validates it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class PropagationThresholds:
    """Quality gates for one adjacent-frame propagation step."""

    maximum_forward_backward_error_px: float = 1.5
    minimum_candidate_flow_consistency: float = 0.70
    maximum_candidate_photometric_median: float = 35.0
    minimum_candidate_pixels: int = 200

    def validate(self) -> None:
        if self.maximum_forward_backward_error_px <= 0:
            raise ValueError("maximum_forward_backward_error_px must be positive")
        if not 0.0 <= self.minimum_candidate_flow_consistency <= 1.0:
            raise ValueError("minimum_candidate_flow_consistency must be in [0, 1]")
        if self.maximum_candidate_photometric_median < 0:
            raise ValueError("maximum_candidate_photometric_median must be non-negative")
        if self.minimum_candidate_pixels < 1:
            raise ValueError("minimum_candidate_pixels must be positive")


def grayscale_u8(image: np.ndarray) -> np.ndarray:
    """Convert a BGR uint8 image to grayscale, rejecting ambiguous inputs."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be an HxWx3 uint8 BGR raster")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def dense_flow(previous_gray: np.ndarray, current_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return adjacent-frame forward (previous->current) and backward flows."""
    if previous_gray.shape != current_gray.shape or previous_gray.ndim != 2:
        raise ValueError("previous/current grayscale images must share an HxW shape")
    if previous_gray.dtype != np.uint8 or current_gray.dtype != np.uint8:
        raise ValueError("grayscale images must be uint8")
    options = dict(pyr_scale=0.5, levels=3, winsize=21, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    forward = cv2.calcOpticalFlowFarneback(previous_gray, current_gray, None, **options)
    backward = cv2.calcOpticalFlowFarneback(current_gray, previous_gray, None, **options)
    return forward.astype(np.float32), backward.astype(np.float32)


def _backward_maps(backward_flow: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if backward_flow.ndim != 3 or backward_flow.shape[2] != 2 or not np.isfinite(backward_flow).all():
        raise ValueError("backward_flow must be a finite HxWx2 array")
    height, width = backward_flow.shape[:2]
    xx, yy = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x, map_y = xx + backward_flow[..., 0], yy + backward_flow[..., 1]
    in_bounds = (map_x >= 0.0) & (map_x <= width - 1) & (map_y >= 0.0) & (map_y <= height - 1)
    return map_x, map_y, in_bounds


def warp_from_previous(source: np.ndarray, backward_flow: np.ndarray, interpolation: int) -> tuple[np.ndarray, np.ndarray]:
    """Pull a previous-frame raster into the current frame via current->previous flow."""
    if source.shape != backward_flow.shape[:2]:
        raise ValueError("source shape must match flow height and width")
    map_x, map_y, in_bounds = _backward_maps(backward_flow)
    warped = cv2.remap(source, map_x, map_y, interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, in_bounds


def flow_consistency(forward_flow: np.ndarray, backward_flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return forward/backward residual and validity in current-frame pixels."""
    if forward_flow.shape != backward_flow.shape:
        raise ValueError("forward and backward flow shapes must match")
    map_x, map_y, in_bounds = _backward_maps(backward_flow)
    forward_at_previous = cv2.remap(forward_flow, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    residual = np.linalg.norm(backward_flow + forward_at_previous, axis=2)
    return residual.astype(np.float32), in_bounds


def propagate_adjacent(
    previous_bgr: np.ndarray,
    current_bgr: np.ndarray,
    source_mask: np.ndarray,
    thresholds: PropagationThresholds,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Propagate one mask and fail closed when the adjacent-flow evidence is weak."""
    thresholds.validate()
    previous_gray, current_gray = grayscale_u8(previous_bgr), grayscale_u8(current_bgr)
    if source_mask.shape != previous_gray.shape or source_mask.dtype != bool:
        raise ValueError("source_mask must be a boolean array matching the previous image")
    forward, backward = dense_flow(previous_gray, current_gray)
    warped_mask_u8, in_bounds = warp_from_previous(source_mask.astype(np.uint8), backward, cv2.INTER_NEAREST)
    propagated_mask = warped_mask_u8.astype(bool) & in_bounds
    warped_previous_gray, _ = warp_from_previous(previous_gray, backward, cv2.INTER_LINEAR)
    residual, consistency_domain = flow_consistency(forward, backward)
    candidate_domain = propagated_mask & consistency_domain
    candidate_pixels = int(propagated_mask.sum())
    consistency_fraction = float((residual[candidate_domain] <= thresholds.maximum_forward_backward_error_px).mean()) if candidate_domain.any() else 0.0
    photometric_median = float(np.median(np.abs(current_gray[candidate_domain].astype(np.int16) - warped_previous_gray[candidate_domain].astype(np.int16)))) if candidate_domain.any() else None
    reasons: list[str] = []
    if candidate_pixels < thresholds.minimum_candidate_pixels:
        reasons.append("too_few_propagated_floor_pixels")
    if consistency_fraction < thresholds.minimum_candidate_flow_consistency:
        reasons.append("forward_backward_flow_inconsistent")
    if photometric_median is None or photometric_median > thresholds.maximum_candidate_photometric_median:
        reasons.append("candidate_photometric_change_too_large")
    status = "candidate" if not reasons else "unavailable"
    record: dict[str, Any] = {
        "status": status,
        "reasons": reasons,
        "propagated_floor_pixel_count": candidate_pixels,
        "candidate_flow_consistency_fraction": consistency_fraction,
        "candidate_photometric_median_abs_gray": photometric_median,
        "flow_method": "Farneback adjacent bidirectional optical flow",
        "maximum_forward_backward_error_px": thresholds.maximum_forward_backward_error_px,
        "minimum_candidate_flow_consistency": thresholds.minimum_candidate_flow_consistency,
        "maximum_candidate_photometric_median": thresholds.maximum_candidate_photometric_median,
        "minimum_candidate_pixels": thresholds.minimum_candidate_pixels,
        "interpretation": "image-space propagated candidate only; not manually audited ground, stereo evidence, geometry, contact, or gait",
    }
    return propagated_mask, record
