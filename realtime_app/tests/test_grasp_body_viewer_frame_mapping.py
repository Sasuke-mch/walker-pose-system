"""Window mesh frame 0 must select scene frame 60, not scene frame 0."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from build_grasp_body_canvas_viewer import validate_frame_ids


def test_window_keeps_original_scene_indices():
    ids=validate_frame_ids(np.arange(60,91),448)
    scene=np.arange(448)*10
    assert scene[ids][0]==600 and scene[ids][-1]==900
    assert len(scene[ids])==31


@pytest.mark.parametrize('ids', [np.array([60,62]),np.array([60.,61.]),np.array([-1,0]),np.array([447,448])])
def test_bad_frame_identity_rejected(ids):
    with pytest.raises(ValueError):
        validate_frame_ids(ids,448)
