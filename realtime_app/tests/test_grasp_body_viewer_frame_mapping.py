"""Window mesh frame 0 must select scene frame 60, not scene frame 0."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from build_grasp_body_canvas_viewer import validate_frame_ids, recorded_strict_gate


def test_window_keeps_original_scene_indices():
    ids=validate_frame_ids(np.arange(60,91),448)
    scene=np.arange(448)*10
    assert scene[ids][0]==600 and scene[ids][-1]==900
    assert len(scene[ids])==31


@pytest.mark.parametrize('ids', [np.array([60,62]),np.array([60.,61.]),np.array([-1,0]),np.array([447,448])])
def test_bad_frame_identity_rejected(ids):
    with pytest.raises(ValueError):
        validate_frame_ids(ids,448)


def test_strict_association_failure_not_promoted_by_finite_fit_points(tmp_path):
    import json
    p=tmp_path/'strict.jsonl'
    rows=[{'index':0,'persons_3d':[]},
          {'index':1,'persons_3d':[{'left_person_id':0,'right_person_id':0,
             'keypoints_3d':[{'index':14,'valid':False,'reason':'high_reprojection_error'},
                             {'index':15,'valid':True,'reason':None}]}]}]
    p.write_text('\n'.join(json.dumps(r) for r in rows))
    mask,reasons=recorded_strict_gate(p,np.array([0,1]))
    assert not mask[0].any() and reasons[0,14]=='association_failed'
    assert not mask[1,14] and reasons[1,14]=='high_reprojection_error'
    assert mask[1,15]
