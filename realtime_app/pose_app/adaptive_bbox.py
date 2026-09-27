"""Deterministic adaptive person-box expansion for pose model inputs.

The policy is shared with the audited offline continuous foot-inclusive ROI
builder.  It operates in the current model-input image coordinate system; the
existing rotation/local-view restoration then maps pose points back to raw
fisheye pixels before stereo geometry.
"""

from __future__ import annotations

from typing import Any


DEFAULT_THRESHOLD = 0.37
DEFAULT_POWER = 0.75
HORIZONTAL_PAD_MAX = 0.20
TOP_PAD_MAX = 0.08
BOTTOM_PAD_MAX = 1.30


def _growth(width_fraction: float, threshold: float, power: float) -> float:
    if width_fraction <= threshold:
        return 0.0
    normalized = min(1.0, max(0.0, (width_fraction - threshold) / (1.0 - threshold)))
    return normalized**power


def expand_bbox(
    bbox: list[float] | tuple[float, ...],
    image_width: int,
    image_height: int,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    power: float = DEFAULT_POWER,
) -> tuple[list[float], dict[str, float | bool]]:
    """Return a clipped adaptive ROI and auditable rule statistics."""

    if image_width <= 1 or image_height <= 1 or len(bbox) < 4:
        return list(map(float, bbox[:4])), {"expanded": False, "growth": 0.0}
    x1, y1, x2, y2 = map(float, bbox[:4])
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0.0, min(float(image_width), x1))
    x2 = max(0.0, min(float(image_width), x2))
    y1 = max(0.0, min(float(image_height), y1))
    y2 = max(0.0, min(float(image_height), y2))
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    width_fraction = box_width / float(image_width)
    growth = _growth(width_fraction, threshold, power)
    side_pad = HORIZONTAL_PAD_MAX * growth
    top_pad = TOP_PAD_MAX * growth
    bottom_pad = BOTTOM_PAD_MAX * growth
    output = [
        max(0.0, x1 - side_pad * box_width),
        max(0.0, y1 - top_pad * box_height),
        min(float(image_width), x2 + side_pad * box_width),
        min(float(image_height), y2 + bottom_pad * box_height),
    ]
    return output, {
        "expanded": bool(growth > 0.0),
        "width_fraction": width_fraction,
        "growth": growth,
        "side_pad": side_pad,
        "top_pad": top_pad,
        "bottom_pad": bottom_pad,
    }


def expand_detections(
    detections: list[Any],
    image_width: int,
    image_height: int,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    power: float = DEFAULT_POWER,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    """Expand detector boxes while preserving scores/classes and audit counts."""

    output: list[dict[str, Any]] = []
    expanded_count = 0
    person_count = 0
    for item in detections:
        if not isinstance(item, dict):
            continue
        copied = dict(item)
        box = item.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            output.append(copied)
            continue
        class_name = item.get("class_name")
        if class_name is not None and str(class_name).lower() != "person":
            output.append(copied)
            continue
        person_count += 1
        expanded, stats = expand_bbox(
            list(box), image_width, image_height, threshold=threshold, power=power
        )
        copied["bbox"] = expanded
        copied["adaptive_original_bbox"] = [float(value) for value in box[:4]]
        copied["adaptive_box_stats"] = stats
        expanded_count += int(stats["expanded"])
        output.append(copied)
    return output, {
        "detections": len(output),
        "person_detections": person_count,
        "expanded_detections": expanded_count,
    }
