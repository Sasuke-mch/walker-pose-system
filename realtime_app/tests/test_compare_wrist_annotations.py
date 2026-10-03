import json

import numpy as np
import pytest

from compare_wrist_annotations import compare_point, stats, summarize, unique_wrists
from walker_tools.hands.solve_annotated_wrist_targets import upright_to_raw
from walker_tools.hands.solve_multiframe_wrist_targets import raw_to_upright


@pytest.mark.parametrize('camera', ['left', 'right'])
def test_manual_coordinate_rotation_roundtrip_and_preserved_distance(camera):
    a, b = np.array([123.25, 987.5]), np.array([128.25, 999.5])
    assert np.allclose(raw_to_upright(upright_to_raw(a, camera), camera), a)
    assert np.linalg.norm(upright_to_raw(a, camera)-upright_to_raw(b, camera)) == 13


def test_missing_and_outside_predictions_are_not_hidden():
    assert compare_point([1, 2], {'state': 'unavailable'})['state'] == 'manual_unavailable'
    label = {'state': 'present', 'raw_px': [0, 0]}
    assert compare_point([np.nan, 0], label, 'no_candidate')['reason'] == 'no_candidate'
    r = compare_point([-3, -4], label)
    assert r['error_px'] == 5 and not r['in_raw_bounds']


def test_common_support_excludes_missing_prediction_for_every_method():
    label = {'state': 'present', 'raw_px': [0, 0]}
    rows = [{'camera': 'left', 'hand': 'right', 'manual': label,
             'methods': {'a': compare_point([3, 4], label), 'b': compare_point([0, 0], label)}},
            {'camera': 'left', 'hand': 'right', 'manual': label,
             'methods': {'a': compare_point([300, 400], label), 'b': compare_point([np.nan, 0], label)}}]
    report = summarize(rows, ['a', 'b'], True)
    assert report['a']['all']['count'] == report['b']['all']['count'] == 1
    assert report['a']['all']['mean'] == 5


def test_unique_candidates_never_select_by_manual_proximity(tmp_path):
    c = {'side': 'left', 'pixel_frame': 'raw_fisheye', 'keypoints_2d_raw_fisheye': [[1, 2]]*21}
    p = tmp_path/'wilor.jsonl'
    p.write_text(json.dumps({'frame_index': 0, 'image': 'pair_0000.png', 'records': [c, c]}))
    points, reasons = unique_wrists(p, 1)
    assert np.isnan(points).all()
    assert reasons[0, 'left'] == 'multiple_candidates_no_label_selection'
    assert reasons[0, 'right'] == 'no_candidate'


def test_stats_distinguishes_empty_from_zero_error():
    assert stats([])['mean'] is None
    assert stats([0])['mean'] == 0
    with pytest.raises(ValueError):
        stats([np.nan])
