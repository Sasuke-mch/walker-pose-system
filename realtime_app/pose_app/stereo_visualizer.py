from __future__ import annotations

import cv2
import numpy as np

from .schema import InferenceResult
from .stereo_sources import StereoFramePair
from .triangulation import TriangulatedPerson
from .visualizer import draw


def _fit_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    scale = height / image.shape[0]
    width = max(1, round(image.shape[1] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def draw_stereo(
    pair: StereoFramePair,
    left_result: InferenceResult,
    right_result: InferenceResult,
    persons_3d: list[TriangulatedPerson],
    threshold: float,
    processed: int,
    display_width: int = 1920,
) -> np.ndarray:
    left = draw(
        pair.left.image,
        left_result,
        threshold,
        processed,
        pair.dropped_left,
        "stereo-left",
    )
    right = draw(
        pair.right.image,
        right_result,
        threshold,
        processed,
        pair.dropped_right,
        "stereo-right",
    )
    target_height = min(left.shape[0], right.shape[0])
    left = _fit_height(left, target_height)
    right = _fit_height(right, target_height)
    composite = np.concatenate([left, right], axis=1)
    if display_width > 0 and composite.shape[1] > display_width:
        scale = display_width / composite.shape[1]
        composite = cv2.resize(
            composite,
            (display_width, max(1, round(composite.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )

    valid_3d = sum(person.valid_keypoints for person in persons_3d)
    lines = [
        f"pair={pair.pair_id}  skew={pair.timestamp_skew_sec * 1000.0:.2f} ms",
        f"3D persons={len(persons_3d)}  valid 3D keypoints={valid_3d}",
        f"3D coordinates: left-camera frame; timestamp={pair.timestamp_type}",
    ]
    overlay = composite.copy()
    box_width = min(composite.shape[1] - 16, 1000)
    box_height = 32 * len(lines) + 12
    cv2.rectangle(overlay, (8, 8), (8 + box_width, 8 + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, composite, 0.35, 0, composite)
    for index, line in enumerate(lines):
        cv2.putText(
            composite,
            line,
            (18, 36 + 32 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return composite
