import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.fit_smpl_coco_temporal_guardrail import load_force_all_guardrails


def _joint(index: int, error: float) -> dict:
    return {
        "index": index,
        "xyz": [100.0 + index, 200.0, 1000.0],
        "reprojection_error_mean_px": error,
        "quality_flags": [] if error <= 10.0 else ["high_reprojection_error"],
    }


def test_force_all_points_are_retained_with_quality_dependent_dead_zones() -> None:
    rows = {60: {"persons_3d": [{"keypoints_3d": [_joint(i, 5.0 if i < 8 else 40.0) for i in range(17)]}]}}
    points, radii, errors, available, _ = load_force_all_guardrails(rows, [60])
    assert points.shape == (1, 17, 3)
    assert np.isfinite(points).all()
    assert available.all()
    assert radii[0, 0] == np.float32(0.35)
    assert radii[0, 9] == np.float32(0.55)
    assert radii[0, 14] == np.float32(0.65)
    assert errors[0, 9] == np.float32(40.0)
