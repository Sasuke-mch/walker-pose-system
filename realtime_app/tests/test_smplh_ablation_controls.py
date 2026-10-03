import pytest
import numpy as np

from pose_app.smplh_fitting.stages import build_stage_schedule, validate_optimization_options
from walker_tools.body.fit_smplh_wilor_sequence import build_parser


def options(*extra):
    return build_parser().parse_args([
        "--left-raw", "left", "--right-raw", "right", "--wilor-left", "wl",
        "--wilor-right", "wr", "--calibration-dir", "cal", "--regressor", "reg",
        "--smplh-model", "model", "--output-dir", "out", *extra])


def test_freezing_after_b_preserves_shared_beta_stage_and_other_stages():
    default = build_stage_schedule(options(), False)
    frozen = build_stage_schedule(options("--beta-policy", "freeze_after_b"), False)
    assert default[0] == frozen[0] and default[2] == frozen[2]
    assert frozen[1][1] == ["beta"] and frozen[1][2] == ["root", "transl", "latent"]
    assert all(a == b for index, (a, b) in enumerate(zip(default[1], frozen[1])) if index != 2)


def test_ablation_defaults_preserve_original_controls():
    args = options()
    assert (args.beta_policy, args.beta_prior_weight, args.body_3d_weight, args.body_weighting,
            args.allow_hand_prior_ablation) == ("joint", .001, 1., "quality", False)


@pytest.mark.parametrize("flag", ["--beta-prior-weight", "--body-3d-weight"])
@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_bad_ablation_weights_stop_before_asset_load(flag, value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        validate_optimization_options(options(flag, value))


@pytest.mark.parametrize("mode,expected", [("no_confidence", .02), ("no_ray", .05), ("no_reprojection", .1)])
def test_quality_component_ablation_keeps_mask_and_original_quality(mode, expected):
    from pose_app.smplh_fitting.losses import observation_weights
    terms = tuple(np.full((1, 2), v) for v in (.5, .2, .1, 1., 1.))
    quality = np.array([[.01, 0.]])
    accepted = np.array([[True, False]])
    weights = observation_weights(quality, terms, accepted, mode)
    np.testing.assert_allclose(weights, [[expected, 0.]])
    np.testing.assert_array_equal(accepted, [[True, False]])
    np.testing.assert_array_equal(quality, [[.01, 0.]])
