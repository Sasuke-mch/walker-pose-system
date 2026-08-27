from __future__ import annotations

from collections.abc import Callable
from typing import Final

import cv2
import numpy as np

from .schema import InferenceResult, PersonPose


ROTATION_CHOICES: Final[tuple[str, ...]] = ("none", "cw90", "ccw90", "180")


def _check_rotation(rotation: str) -> str:
    if rotation not in ROTATION_CHOICES:
        raise ValueError(
            f"Unsupported model-input rotation {rotation!r}; "
            f"expected one of {ROTATION_CHOICES}."
        )
    return rotation


def model_image_size(
    raw_width: int, raw_height: int, rotation: str
) -> tuple[int, int]:
    """Return the image size that the 2D model receives after rotation."""

    _check_rotation(rotation)
    if raw_width <= 0 or raw_height <= 0:
        raise ValueError("Raw image width and height must be positive.")
    if rotation in {"cw90", "ccw90"}:
        return raw_height, raw_width
    return raw_width, raw_height


def rotate_image_for_model(image: np.ndarray, rotation: str) -> np.ndarray:
    """Rotate a raw camera image only for 2D model inference."""

    _check_rotation(rotation)
    if image.ndim < 2 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("Model input image must have a positive height and width.")
    if rotation == "none":
        return image
    if rotation == "cw90":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "ccw90":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return cv2.rotate(image, cv2.ROTATE_180)


def raw_to_model_point(
    x: float,
    y: float,
    raw_width: int,
    raw_height: int,
    rotation: str,
) -> tuple[float, float]:
    """Map one raw-camera pixel coordinate into the rotated model image."""

    _check_rotation(rotation)
    model_image_size(raw_width, raw_height, rotation)
    x = float(x)
    y = float(y)
    if rotation == "none":
        return x, y
    if rotation == "cw90":
        return float(raw_height - 1) - y, x
    if rotation == "ccw90":
        return y, float(raw_width - 1) - x
    return float(raw_width - 1) - x, float(raw_height - 1) - y


def model_to_raw_point(
    x: float,
    y: float,
    raw_width: int,
    raw_height: int,
    rotation: str,
) -> tuple[float, float]:
    """Inverse-map one model-output coordinate back to raw camera pixels."""

    _check_rotation(rotation)
    model_image_size(raw_width, raw_height, rotation)
    x = float(x)
    y = float(y)
    if rotation == "none":
        return x, y
    if rotation == "cw90":
        return y, float(raw_height - 1) - x
    if rotation == "ccw90":
        return float(raw_width - 1) - y, x
    return float(raw_width - 1) - x, float(raw_height - 1) - y


def _restore_bbox(
    bbox: list[float],
    raw_width: int,
    raw_height: int,
    rotation: str,
    undistorted_to_raw: Callable[[float, float], tuple[float, float]] | None = None,
) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    corners = [
        model_to_raw_point(x1, y1, raw_width, raw_height, rotation),
        model_to_raw_point(x1, y2, raw_width, raw_height, rotation),
        model_to_raw_point(x2, y1, raw_width, raw_height, rotation),
        model_to_raw_point(x2, y2, raw_width, raw_height, rotation),
    ]
    if undistorted_to_raw is not None:
        corners = [undistorted_to_raw(x, y) for x, y in corners]
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return [min(xs), min(ys), max(xs), max(ys)]


def restore_model_result_to_raw(
    result: InferenceResult,
    *,
    raw_width: int,
    raw_height: int,
    rotation: str,
    undistorted_to_raw: Callable[[float, float], tuple[float, float]] | None = None,
) -> InferenceResult:
    """Return model results expressed in the original camera-pixel system.

    Calibration, epipolar association, triangulation and visualization all use
    raw camera coordinates.  This function is therefore mandatory whenever a
    rotated image was supplied to a 2D pose model.
    """

    _check_rotation(rotation)
    expected_width, expected_height = model_image_size(
        raw_width, raw_height, rotation
    )
    if (result.image_width, result.image_height) != (
        expected_width,
        expected_height,
    ):
        raise RuntimeError(
            "Model service returned image dimensions inconsistent with the "
            "rotated request: "
            f"got {(result.image_width, result.image_height)}, expected "
            f"{(expected_width, expected_height)}."
        )

    restored_persons: list[PersonPose] = []
    for person in result.persons:
        keypoints = []
        for x, y, score in person.keypoints:
            undistorted_x, undistorted_y = model_to_raw_point(
                x, y, raw_width, raw_height, rotation
            )
            if undistorted_to_raw is None:
                raw_x, raw_y = undistorted_x, undistorted_y
            else:
                raw_x, raw_y = undistorted_to_raw(undistorted_x, undistorted_y)
            keypoints.append([raw_x, raw_y, float(score)])
        restored_persons.append(
            PersonPose(
                person_id=person.person_id,
                bbox=_restore_bbox(
                    person.bbox,
                    raw_width,
                    raw_height,
                    rotation,
                    undistorted_to_raw,
                ),
                bbox_score=person.bbox_score,
                pose_score=person.pose_score,
                keypoints=keypoints,
            )
        )

    return InferenceResult(
        source_frame_id=result.source_frame_id,
        source_timestamp_sec=result.source_timestamp_sec,
        image_width=raw_width,
        image_height=raw_height,
        model_name=result.model_name,
        model_ms=result.model_ms,
        roundtrip_ms=result.roundtrip_ms,
        persons=restored_persons,
        dropped_before=result.dropped_before,
        stage_times_ms=dict(result.stage_times_ms),
    )
