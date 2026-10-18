import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.fit_smpl_coco_prefit_trajectory import knot_layout


def test_five_frame_knot_layout_covers_31_frames() -> None:
    knots, indices, alpha = knot_layout(31, 5)
    assert knots.tolist() == [0, 5, 10, 15, 20, 25, 30]
    values = knots.astype(np.float32)[:, None]
    expanded = values[indices[:, 0]] * (1 - alpha[:, None]) + values[indices[:, 1]] * alpha[:, None]
    assert np.allclose(expanded[:, 0], np.arange(31))
