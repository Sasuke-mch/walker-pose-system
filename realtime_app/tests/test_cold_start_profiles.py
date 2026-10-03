import json
from pathlib import Path

import pytest

from pose_app.project_paths import PROJECT_ROOT
from pose_app.smplh_fitting.stages import build_stage_schedule
from walker_tools.body.run_sapiens_cold_smplh import fit_command


def command(profile, joint_steps=None):
    return fit_command(PROJECT_ROOT / 'realtime_app/configs' / profile,
                       {'left': Path('left.json'), 'right': Path('right.json')},
                       Path('new_output'), Path('wrist.json'), joint_steps)


@pytest.mark.parametrize('profile,budget', [('cold_smplh_legacy30.json', 30),
                                         ('cold_smplh_joint300.json', 300)])
def test_profile_preserves_cold_parameterization_and_stage_trainability(profile, budget):
    values, options = command(profile)
    assert values.count('--wrist-reference') == 1
    assert values.count('--allow-diagnostic-wrist-reference') == 1
    assert options.left_raw == Path('left.json')
    assert options.output_dir == Path('new_output/fit')
    assert options.mano_pose_init and options.shared_hand_pose
    assert options.hand_pca_comps == 12
    assert options.wrist_reference_weight == 1
    assert options.hand_2d_weight == options.body_temporal_weight == options.body_reprojection_weight == 0
    names, train, steps = build_stage_schedule(options, False)
    assert steps == [30, 30, budget, 100, 100, 0]
    assert train[1] == ['beta']
    assert train[2] == ['beta', 'root', 'transl', 'latent']
    assert train[3:] == [['lhand', 'rhand']] * 3
    assert names[4] == 'D2_hand_refine_no_contact'


def test_joint_override_changes_only_saved_joint_budget():
    default, _ = command('cold_smplh_legacy30.json')
    override, _ = command('cold_smplh_legacy30.json', 300)
    changes = [i for i, (a, b) in enumerate(zip(default, override)) if a != b]
    assert changes == [default.index('--joint-steps') + 1]


def test_profile_has_no_old_fitted_parameter_input():
    for name in ('cold_smplh_legacy30.json', 'cold_smplh_joint300.json'):
        values = json.loads((PROJECT_ROOT / 'realtime_app/configs' / name).read_text())
        assert not any('result.npz' in value for value in values)
        assert '--pose-reference-result' not in values


def test_current_route_scalar_snapshot_matches_the_same_parser():
    manifest = json.loads((PROJECT_ROOT / 'realtime_app/configs/current_mainline.json').read_text())
    _, options = command('cold_smplh_joint300.json')
    scalars = {key: value for key, value in vars(options).items()
               if not isinstance(value, Path) and value is not None}
    assert scalars == manifest['frozen_scalar_options']
    assert manifest['body_candidate']['accepted_for_main_fit'] is False
    assert manifest['matched_control']['accepted_for_main_fit'] is False


def test_invalid_profile_is_rejected(tmp_path):
    profile = tmp_path / 'settings.json'
    profile.write_text('{}')
    with pytest.raises(ValueError, match='array'):
        fit_command(profile, {}, tmp_path, tmp_path)


@pytest.mark.parametrize('extra,message', [
    (['--joint-steps', '1'], 'duplicate'),
    (['--contact-labels', 'labels.json'], 'contact inputs'),
    (['--beta-policy', 'freeze_after_b'], 'ablation controls'),
    (['--body-weighting', 'uniform'], 'ablation controls'),
    (['--body-3d-weight', '0'], 'ablation controls'),
    (['--beta-prior-weight', '0'], 'ablation controls'),
    (['--allow-hand-prior-ablation'], 'ablation controls'),
])
def test_custom_profile_cannot_misrepresent_cold_protocol(tmp_path, extra, message):
    values = json.loads((PROJECT_ROOT/'realtime_app/configs/cold_smplh_joint300.json').read_text())
    profile = tmp_path/'profile.json'
    profile.write_text(json.dumps(values + extra))
    with pytest.raises(ValueError, match=message):
        fit_command(profile, {'left': Path('left'), 'right': Path('right')},
                    tmp_path/'output', Path('wrist'))


def test_zero_joint_budget_in_profile_is_rejected(tmp_path):
    values = json.loads((PROJECT_ROOT/'realtime_app/configs/cold_smplh_joint300.json').read_text())
    values[values.index('--joint-steps') + 1] = '0'
    profile = tmp_path/'profile.json'
    profile.write_text(json.dumps(values))
    with pytest.raises(ValueError, match='positive'):
        fit_command(profile, {'left': Path('left'), 'right': Path('right')},
                    tmp_path/'output', Path('wrist'))
