from pathlib import Path
import pytest

from walker_tools.body.compare_cold_detector_fits import fit_directories, viewer_url, require_same_scene_input


def test_default_directories_preserve_frozen_results(tmp_path):
    paths = fit_directories(tmp_path)
    assert paths['sapiens2'] == tmp_path/'smplh_cold_start_v2_joint300'
    assert paths['pmpose'] == tmp_path/'pmpose_cold_start_v1_joint300'


def test_explicit_directories_and_viewer_paths_follow_selected_inputs(tmp_path):
    paths = fit_directories(tmp_path, tmp_path/'new sapien #1', tmp_path/'new pm')
    assert paths['sapiens2'] == tmp_path/'new sapien #1'
    assert viewer_url(paths['sapiens2'], tmp_path/'reports'/'comparison') == '../../new%20sapien%20%231/body_viewer.html'
    assert viewer_url(paths['pmpose'], tmp_path/'comparison') == '../new%20pm/body_viewer.html'


def test_relative_input_directory_is_resolved_from_callers_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert fit_directories(tmp_path, Path('new'))['sapiens2'] == tmp_path/'new'


def test_new_result_cannot_use_scene_from_other_body_inputs(tmp_path):
    metadata = {'inputs': {'left_raw': str(tmp_path/'left'), 'right_raw': str(tmp_path/'right')}}
    source = {'raw_left_pmpose': str(tmp_path/'left'), 'raw_right_pmpose': str(tmp_path/'right'),
              'old_stage_jsonl_read': False, 'old_dynamic_ground_read': False, 'old_walker_pose_read': False}
    require_same_scene_input(metadata, source)
    source['raw_left_pmpose'] = str(tmp_path/'other')
    with pytest.raises(ValueError, match='mismatch'):
        require_same_scene_input(metadata, source)
    source['raw_left_pmpose'] = metadata['inputs']['left_raw']
    source['old_dynamic_ground_read'] = True
    with pytest.raises(ValueError, match='old dynamic'):
        require_same_scene_input(metadata, source)
