"""The existing continuous foot-inclusive ROI shared by online/offline callers.

Input/output boxes are xyxy pixels in model-input coordinates. This module
only contains the original rule and arithmetic; camera rotation and inverse
keypoint mapping remain with their existing callers. No model is imported.
"""
from __future__ import annotations

RULE = {
    "threshold": 0.37,
    "power": 0.75,
    "horizontal_pad_max_fraction_of_box_width": 0.20,
    "top_pad_max_fraction_of_box_height": 0.08,
    "bottom_pad_max_fraction_of_box_height": 1.30,
    "intent": "aggressive pose-input ROI; preserve far boxes and prefer background context over missing feet",
}


def expansion_fraction(width_fraction: float, threshold: float, power: float) -> float:
    if width_fraction <= threshold:
        return 0.0
    t = min(1.0, max(0.0, (width_fraction - threshold) / (1.0 - threshold)))
    return t ** power


def foot_inclusive_box(box: list[float], width: int, height: int, threshold: float, power: float) -> tuple[list[float], dict[str, float]]:
    x1, y1, x2, y2 = map(float, box)
    bw, bh = x2 - x1, y2 - y1
    r = bw / float(width)
    g = expansion_fraction(r, threshold, power)
    side = RULE["horizontal_pad_max_fraction_of_box_width"] * g
    top = RULE["top_pad_max_fraction_of_box_height"] * g
    bottom = RULE["bottom_pad_max_fraction_of_box_height"] * g
    output = [
        max(0.0, x1 - side * bw),
        max(0.0, y1 - top * bh),
        min(float(width - 1), x2 + side * bw),
        min(float(height - 1), y2 + bottom * bh),
    ]
    return output, {"width_fraction": r, "growth": g, "side_pad": side, "top_pad": top, "bottom_pad": bottom}
