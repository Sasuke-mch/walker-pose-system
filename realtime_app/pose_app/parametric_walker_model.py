from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WalkerDimensions:
    outer_width_mm: float = 450.0
    depth_mm: float = 300.0
    height_mm: float = 840.0
    middle_rail_height_mm: float = 445.0
    front_rail_height_mm: float = 675.0

    def validate(self) -> None:
        if not 400.0 <= self.outer_width_mm <= 550.0:
            raise ValueError("walker width is outside the supported coarse-model range")
        if not 220.0 <= self.depth_mm <= 400.0:
            raise ValueError("walker depth is outside the supported coarse-model range")
        if not 750.0 <= self.height_mm <= 930.0:
            raise ValueError("walker height must remain inside the product adjustment range")
        if not 250.0 <= self.middle_rail_height_mm < self.height_mm:
            raise ValueError("middle rail height is invalid")
        if not self.middle_rail_height_mm < self.front_rail_height_mm < self.height_mm:
            raise ValueError("front rail must be above the side rails and below the handles")


def build_coarse_frame(dimensions: WalkerDimensions) -> dict[str, Any]:
    """Build a four-foot open-back walking-frame skeleton.

    Walker coordinates use x=lateral right, y=toward the open/user side and
    z=up.  The front crossbars are at negative y; the user enters from +y.
    """
    dimensions.validate()
    x = dimensions.outer_width_mm / 2.0
    y = dimensions.depth_mm / 2.0
    h = dimensions.height_mm
    m = dimensions.middle_rail_height_mm
    f = dimensions.front_rail_height_mm
    nodes = {
        "front_right_foot": [x, -y, 0.0],
        "front_left_foot": [-x, -y, 0.0],
        "rear_right_foot": [x, y, 0.0],
        "rear_left_foot": [-x, y, 0.0],
        "front_right_mid": [x, -y, m],
        "front_left_mid": [-x, -y, m],
        "rear_right_mid": [x, y, m],
        "rear_left_mid": [-x, y, m],
        "front_right_rail": [x, -y, f],
        "front_left_rail": [-x, -y, f],
        "front_right_top": [x, -y, h],
        "front_left_top": [-x, -y, h],
        "rear_right_top": [x, y, h],
        "rear_left_top": [-x, y, h],
        "front_top_center": [0.0, -y, h],
    }
    # Keep the frame as connected structural members.  The previous version
    # used foot->top members, which hid the intermediate joints and omitted
    # the front middle crossbar.  The rear remains open for user entry.
    edges = [
        # four segmented legs/uprights
        ["front_right_foot", "front_right_mid"],
        ["front_right_mid", "front_right_rail"],
        ["front_right_rail", "front_right_top"],
        ["front_left_foot", "front_left_mid"],
        ["front_left_mid", "front_left_rail"],
        ["front_left_rail", "front_left_top"],
        ["rear_right_foot", "rear_right_mid"],
        ["rear_right_mid", "rear_right_top"],
        ["rear_left_foot", "rear_left_mid"],
        ["rear_left_mid", "rear_left_top"],
        # side rails and handles
        ["front_right_mid", "rear_right_mid"],
        ["front_left_mid", "rear_left_mid"],
        ["front_right_top", "rear_right_top"],
        ["front_left_top", "rear_left_top"],
        # front crossbars; rear stays open
        ["front_left_mid", "front_right_mid"],
        ["front_left_rail", "front_right_rail"],
    ]
    # COCO body-left appears on positive ground X in this installation.
    handles = {
        "left": ["front_right_top", "rear_right_top"],
        "right": ["front_left_top", "rear_left_top"],
    }
    return {
        "nodes_walker_mm": nodes,
        "edges": edges,
        "handle_segments": handles,
        "foot_nodes": [
            "front_right_foot", "front_left_foot", "rear_right_foot", "rear_left_foot"
        ],
    }


def transform_nodes(
    nodes: dict[str, list[float]], rotation: np.ndarray, translation: np.ndarray
) -> dict[str, list[float]]:
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("rigid transform must contain a 3x3 rotation and 3-vector translation")
    return {
        name: (rotation @ np.asarray(point, dtype=np.float64) + translation).tolist()
        for name, point in nodes.items()
    }


def point_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> tuple[float, np.ndarray, float]:
    point = np.asarray(point, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    delta = end - start
    denominator = float(delta @ delta)
    fraction = 0.0 if denominator <= 1e-12 else float(np.clip((point - start) @ delta / denominator, 0.0, 1.0))
    closest = start + fraction * delta
    return float(np.linalg.norm(point - closest)), closest, fraction
