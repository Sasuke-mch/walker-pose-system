from __future__ import annotations

from typing import Any, Mapping

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


def lower_limb_status_lines(status: Mapping[str, Any]) -> list[str]:
    """Format only conservative, downstream status for the operator overlay."""

    trajectory = status.get("t1_trajectory", {})
    direct = trajectory.get("direct_observed_joint_names", [])
    metrics = status.get("t2_kinematics", {})

    def scalar(name: str, label: str) -> str:
        metric = metrics.get(name, {})
        if metric.get("available"):
            return f"{label}={float(metric['value']):.1f} deg"
        return f"{label}=unavailable"

    coordinate = status.get("t3_coordinates", {})
    candidates = status.get("t4_noncontact_candidates", {}).get(
        "newly_confirmed_candidates", []
    )
    return [
        "lower-limb live: downstream status only",
        f"direct lower-limb joints={len(direct)}/6  frame={trajectory.get('frame_status', 'unknown')}",
        f"{scalar('left_knee_angle_deg', 'left knee')}  {scalar('right_knee_angle_deg', 'right knee')}",
        f"target coordinates={coordinate.get('status', 'unknown')}",
        f"new non-contact candidates={len(candidates)}  accepted contact events=0",
    ]


def stage_walker_status_lines(status: Mapping[str, Any]) -> list[str]:
    stage = status.get("stage", {})
    reconstruction = status.get("walker_reconstruction", {})
    background = stage.get("background", {})
    view = stage.get("evidence_view")
    selected = background.get(view, {}) if view else {}
    motion = selected.get("median_motion_px")
    motion_text = f"{float(motion):.2f}px" if motion is not None else "unavailable"
    return [
        (
            f"two-stage={stage.get('operational', stage.get('confirmed', 'unknown'))}  "
            f"candidate={stage.get('candidate', 'unknown')}  "
            f"vote={float(stage.get('vote_confidence', 0.0)):.2f}"
        ),
        f"background motion={motion_text}  evidence view={view or 'none'}",
        (
            f"walker reconstruction={reconstruction.get('status', 'unknown')}  "
            f"3D points={len(reconstruction.get('points_3d', []))}  "
            f"2D lines L/R={len(reconstruction.get('left_stable_line_segments', []))}/"
            f"{len(reconstruction.get('right_stable_line_segments', []))}"
        ),
        "cyan lines=walker structure candidates; yellow skeleton=human pose",
        f"stage/walker processing={float(status.get('processing_ms', 0.0)):.1f} ms",
    ]


def _draw_status_panel(image: np.ndarray, lines: list[str], top: int) -> None:
    overlay = image.copy()
    box_width = min(image.shape[1] - 16, 1300)
    box_height = 28 * len(lines) + 12
    cv2.rectangle(overlay, (8, top), (8 + box_width, top + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (18, top + 26 + 28 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def _draw_walker_segments(
    image: np.ndarray, segments: list[list[int]], processing_width: int
) -> None:
    if processing_width <= 0:
        return
    scale = image.shape[1] / float(processing_width)
    for line in segments:
        if len(line) != 4:
            continue
        x1, y1, x2, y2 = [round(float(value) * scale) for value in line]
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 0), 3, cv2.LINE_AA)


def draw_stereo(
    pair: StereoFramePair,
    left_result: InferenceResult,
    right_result: InferenceResult,
    persons_3d: list[TriangulatedPerson],
    threshold: float,
    processed: int,
    display_width: int = 1920,
    lower_limb_status: Mapping[str, Any] | None = None,
    stage_walker_status: Mapping[str, Any] | None = None,
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
    if stage_walker_status is not None:
        reconstruction = stage_walker_status.get("walker_reconstruction", {})
        processing_width = int(reconstruction.get("processing_width", 0))
        _draw_walker_segments(
            left, reconstruction.get("left_stable_line_segments", []), processing_width
        )
        _draw_walker_segments(
            right, reconstruction.get("right_stable_line_segments", []), processing_width
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
    _draw_status_panel(composite, lines, 8)
    if lower_limb_status is not None:
        _draw_status_panel(composite, lower_limb_status_lines(lower_limb_status), 8 + 32 * len(lines) + 20)
    if stage_walker_status is not None:
        top = 8 + 32 * len(lines) + 20
        if lower_limb_status is not None:
            top += 28 * len(lower_limb_status_lines(lower_limb_status)) + 24
        _draw_status_panel(composite, stage_walker_status_lines(stage_walker_status), top)
    return composite
