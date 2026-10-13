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


def export_options(**changes):
    from walker_tools.body.fit_smplh_wilor_sequence import build_parser

    flags = ('left-raw', 'right-raw', 'wilor-left', 'wilor-right',
             'calibration-dir', 'regressor', 'smplh-model', 'output-dir')
    options = build_parser().parse_args([value for flag in flags
                                        for value in ('--' + flag, 'placeholder')])
    for name, value in changes.items():
        setattr(options, name, value)
    return options


@pytest.mark.parametrize('changes', [
    {},
    {'body_observation_mode': 'reprojection_only', 'body_3d_weight': 0,
     'body_reprojection_weight': .004, 'body_ankle_reprojection_multiplier': 2},
    {'fit_policy': 'legacy', 'body_weighting': 'reference'},
])
def test_export_accepts_current_body_policy_and_historical_reference25(changes):
    from build_current_mainline_view import require_exportable_body

    require_exportable_body('pmpose', export_options(**changes))


@pytest.mark.parametrize('changes', [
    {'body_reprojection_weight': .004},
    {'body_weighting': 'reference'},
    {'fit_policy': 'legacy'},
])
def test_export_rejects_mixed_or_unsupported_body_policy(changes):
    from build_current_mainline_view import require_exportable_body

    with pytest.raises(ValueError):
        require_exportable_body('pmpose', export_options(**changes))


def test_export_keeps_same_detector_contract():
    from build_current_mainline_view import require_exportable_body

    with pytest.raises(ValueError, match='same-source PMPose'):
        require_exportable_body('sapiens2', export_options())
