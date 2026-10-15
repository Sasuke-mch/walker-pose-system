"""Window mesh frame 0 must select scene frame 60, not scene frame 0."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from build_grasp_body_canvas_viewer import validate_frame_ids, recorded_strict_gate, build_canvas_page


def test_motion_display_keeps_finite_rejected_points_and_strict_trails():
    from build_smplh_motion_viewer import scene_observations

    raw = np.arange(102, dtype=float).reshape(2, 17, 3)
    strict = raw.copy()
    strict[:, 14:17] = np.nan
    display, ankles = scene_observations({
        'detector_only_triangulated_points_mm': raw,
        'raw_triangulated_points_mm': strict,
    }, np.array([1]))
    np.testing.assert_array_equal(display, raw[1:2])
    assert np.isnan(ankles).all()
    assert np.isnan(strict[:, 14:17]).all()
    from walker_tools.visualization.build_grasp_body_canvas_viewer import COCO_EDGES
    assert (12, 14) in COCO_EDGES and (14, 16) in COCO_EDGES
    assert (3, 5) in COCO_EDGES and (4, 6) in COCO_EDGES
    page = build_canvas_page({'n': 1, 'ids': [1]}, np.zeros((1, 6890, 3)), raw[1:2], {})
    assert "seg(A,B,'#000000',1.5)" in page
    assert 'observationEdgesDrawn++' in page
    assert 'if(A.every(Number.isFinite)&&B.every(Number.isFinite))' in page


def test_failed_mesh_frames_do_not_poison_fixed_view_center():
    coco = np.zeros((3, 17, 3))
    coco[1] = np.nan
    page = build_canvas_page({'n': 3, 'ids': [0, 1, 2]}, np.zeros((3, 6890, 3)), coco, {})
    assert 'const px=p[0]-(0)' in page
    assert 'p[0]-(nan)' not in page
    assert "x===null?'—':x.toFixed(1)" in page
    with pytest.raises(ValueError, match='no finite pelvis'):
        build_canvas_page({'n': 3, 'ids': [0, 1, 2]}, np.zeros((3, 6890, 3)), np.full_like(coco, np.nan), {})


def test_comparison_center_changes_framing_only():
    coco = np.zeros((1, 17, 3))
    vertices = np.ones((1, 6890, 3))
    page = build_canvas_page({'n': 1, 'ids': [0]}, vertices, coco,
                             {'view_center_ground_m': [1.25, -2.5]})
    assert 'const px=p[0]-(1.25),py=p[1]-(2.5)' in page
    assert np.all(vertices == 1) and np.all(coco == 0)
    with pytest.raises(ValueError, match='finite ground XY'):
        build_canvas_page({'n': 1, 'ids': [0]}, vertices, coco,
                          {'view_center_ground_m': [np.nan, 0]})


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
