"""Data-loss isolation and the differentiable surface-to-pixel contract."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pose_app.fisheye_camera import fisheye_project_torch
from pose_app.smpl_coco_observation import regress_coco17_torch
from pose_app.smplh_fitting.losses import (
    body_reprojection_loss,
    enabled_body_observation,
    raw_pixel_support,
    supported_camera_points,
)


def test_disabled_3d_diagnostic_has_no_gradient_or_nan_edge():
    parameter = torch.tensor(1.0, requires_grad=True)
    diagnostic = parameter * float("nan")
    term = enabled_body_observation(diagnostic, 0, 0)
    assert term == 0 and not term.requires_grad
    (parameter.square() + term).backward()
    assert parameter.grad == 2


def test_independent_view_support_preserves_bounds_and_missing_reasons():
    raw = np.array([[[10, 20, .8], [-1, 20, .9], [10, 20, np.nan], [10, 20, 0]]])
    valid, reasons = raw_pixel_support(raw, (1920, 1080))
    assert valid.tolist() == [[True, False, False, False]]
    assert reasons.tolist() == [["accepted", "out_of_raw_image_bounds", "nonfinite_source", "nonpositive_confidence"]]


def test_reprojection_selects_support_before_arithmetic_and_uses_one_available_view():
    projection = torch.tensor([[[25., 0.], [float("nan"), float("nan")]]], requires_grad=True)
    valid = torch.tensor([[True, False]])
    args = SimpleNamespace(body_reprojection_weight=.002, body_reprojection_scale_px=25.,
                           body_observation_mode="reprojection_only")
    kwargs = {"args": args, "stage_index": 0, "body_loss": torch.tensor(0.),
                  "proj_cam0": projection, "proj_cam1": projection,
                  "raw_left_2d": torch.zeros_like(projection), "raw_right_2d": torch.zeros_like(projection),
                  "raw_left_conf": torch.ones(1, 2), "raw_right_conf": torch.ones(1, 2),
                  "raw_left_valid": valid, "raw_right_valid": torch.zeros_like(valid)}
    loss, pixels = body_reprojection_loss(**kwargs)
    assert loss == .5 and pixels == 25
    gradient = torch.autograd.grad(loss, projection)[0]
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient[:, 1]) == 0
    kwargs["raw_left_valid"] = torch.zeros_like(valid)
    with pytest.raises(RuntimeError, match="no supported"):
        body_reprojection_loss(**kwargs)


def test_surface_regression_precedes_nonlinear_projection_and_translation_survives():
    vertices = torch.zeros(1, 6890, 3, dtype=torch.float64)
    vertices[..., 2] = 2
    vertices[0, 0] = torch.tensor([1., 0., 1.])
    vertices[0, 1] = torch.tensor([0., 1., 3.])
    translation = torch.tensor([.1, .2, .3], dtype=torch.float64, requires_grad=True)
    regressor = np.zeros((17, 6890))
    regressor[:, :2] = .5
    points = regress_coco17_torch(vertices + translation, regressor)
    torch.testing.assert_close(points[0, 0], torch.tensor([.6, .7, 2.3], dtype=torch.float64))
    camera = np.array([[500., 0., 960.], [0., 500., 540.], [0., 0., 1.]])
    pixels = fisheye_project_torch(points, camera, np.zeros(4))
    wrong = fisheye_project_torch((vertices + translation)[:, :2], camera, np.zeros(4)).mean(1)
    assert (pixels[:, 0] - wrong).norm() > 10
    assert torch.isfinite(torch.autograd.grad(pixels.sum(), translation)[0]).all()
    on_axis = torch.tensor([[0., 0., 2.]], dtype=torch.float64, requires_grad=True)
    torch.autograd.gradcheck(lambda x: fisheye_project_torch(x, camera, np.zeros(4)), (on_axis,))


def test_missing_camera_origin_does_not_poison_supported_projection_gradient():
    points = torch.tensor([[[.1, .2, 1.], [0., 0., 0.]]], requires_grad=True)
    valid = torch.tensor([[True, False]])
    pixels = fisheye_project_torch(supported_camera_points(points, valid), np.eye(3), np.zeros(4))
    gradient = torch.autograd.grad(pixels[valid].sum(), points)[0]
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient[:, 1]) == 0


@pytest.mark.parametrize("flags", [
    ["--body-3d-weight", "1", "--body-reprojection-weight", ".002"],
    ["--body-3d-weight", "0", "--body-reprojection-weight", "0"],
    ["--body-3d-weight", "0", "--body-reprojection-weight", ".002", "--wrist-reference-weight", "1"],
])
def test_2d_mode_rejects_mixed_or_missing_objective_before_asset_loading(tmp_path, flags):
    from pose_app.smplh_fitting.pipeline import run_fit
    from walker_tools.body.fit_smplh_wilor_sequence import build_parser

    argv = []
    for name in ("left-raw", "right-raw", "wilor-left", "wilor-right", "calibration-dir", "regressor", "smplh-model"):
        argv += [f"--{name}", str(tmp_path / name)]
    argv += ["--output-dir", str(tmp_path / "fit"), "--body-observation-mode", "reprojection_only", *flags]
    with pytest.raises(ValueError, match="reprojection_only"):
        run_fit(build_parser().parse_args(argv))
    assert not (tmp_path / "fit").exists()


def test_ankle_multiplier_changes_only_ankle_gradient_not_confidence_denominator():
    projection = torch.full((1, 17, 2), 10., requires_grad=True)
    confidence = torch.ones(1, 17)
    valid = torch.ones(1, 17, dtype=torch.bool)
    args = SimpleNamespace(body_reprojection_weight=.004, body_reprojection_scale_px=25.,
                           body_observation_mode="reprojection_only", body_ankle_reprojection_multiplier=1.)
    kwargs = dict(args=args, stage_index=2, body_loss=torch.tensor(0.),
                  proj_cam0=projection, proj_cam1=projection,
                  raw_left_2d=torch.zeros_like(projection), raw_right_2d=torch.zeros_like(projection),
                  raw_left_conf=confidence, raw_right_conf=confidence,
                  raw_left_valid=valid, raw_right_valid=valid)
    base, pixels = body_reprojection_loss(**kwargs)
    gradient = torch.autograd.grad(base, projection)[0]
    args.body_ankle_reprojection_multiplier = 2.
    changed, changed_pixels = body_reprojection_loss(**kwargs)
    updated = torch.autograd.grad(changed, projection)[0]
    torch.testing.assert_close(updated[:, :15], gradient[:, :15])
    torch.testing.assert_close(updated[:, 15:], 2 * gradient[:, 15:])
    torch.testing.assert_close(changed_pixels, pixels)
    torch.testing.assert_close(changed / base, torch.tensor(19 / 17))


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf")])
def test_invalid_ankle_multiplier_fails_before_assets(tmp_path, value):
    from pose_app.smplh_fitting.stages import validate_optimization_options
    with pytest.raises(ValueError, match="ankle reprojection multiplier"):
        validate_optimization_options(SimpleNamespace(body_ankle_reprojection_multiplier=value))


def test_nondefault_ankle_multiplier_rejects_legacy_mode():
    from pose_app.smplh_fitting.stages import validate_optimization_options
    with pytest.raises(ValueError, match="requires reprojection_only"):
        validate_optimization_options(SimpleNamespace(body_ankle_reprojection_multiplier=2., body_observation_mode="legacy"))
