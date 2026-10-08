"""Semantic mismatches must fail even when tensor dimensions look compatible."""
from dataclasses import replace

import cv2
import numpy as np
import pytest
import torch

from realtime_app.learning_flow_mesh.interfaces import (
    JointSchema,
    MaskMeaning,
    PixelDomain,
    SixDLayout,
    camera_observations,
    convert_rotation,
    decode_rotation,
    encode_rotation,
    metric_coordinates,
    presence_mask,
    require_joint_schema,
)
from realtime_app.pose_app.calibration import StereoCalibration
from realtime_app.pose_app.rotation import raw_to_model_point


def rotations():
    values = [np.array([0.3, -0.5, 0.7]), np.array([-1.0, 0.2, 0.4])]
    return torch.tensor(np.stack([cv2.Rodrigues(x)[0] for x in values]), dtype=torch.float64)


@pytest.mark.parametrize("layout", list(SixDLayout))
def test_rotation_layout_roundtrip_and_invalid_state(layout):
    expected = rotations()
    packed = encode_rotation(expected, layout)
    decoded, valid = decode_rotation(packed, layout)
    assert valid.all()
    torch.testing.assert_close(decoded, expected, atol=1e-12, rtol=0)
    damaged = torch.cat((packed, torch.zeros(1, 6, dtype=packed.dtype),
                         torch.full((1, 6), float("nan"), dtype=packed.dtype)))
    for target in SixDLayout:
        changed, admitted = convert_rotation(damaged, source=layout, target=target)
        assert admitted.tolist() == [True, True, False, False]
        assert torch.isnan(changed[~admitted]).all()
        recovered, _ = decode_rotation(changed[:2], target)
        torch.testing.assert_close(recovered, expected, atol=1e-12, rtol=0)


def test_wrong_layout_can_be_proper_but_transposed_and_conversion_has_gradient():
    expected = rotations()
    row = encode_rotation(expected, SixDLayout.PACKED_ROWS).requires_grad_()
    wrong, valid = decode_rotation(row, SixDLayout.PACKED_COLUMNS)
    assert valid.all() and (wrong - expected).abs().max() > 0.1
    torch.testing.assert_close(wrong, expected.mT, atol=1e-12, rtol=0)
    converted, _ = convert_rotation(row, source=SixDLayout.PACKED_ROWS,
                                    target=SixDLayout.INTERLEAVED_COLUMNS)
    converted[..., 0].sum().backward()
    assert torch.isfinite(row.grad).all() and row.grad.abs().sum() > 0


def test_equal_count_is_not_equal_joint_semantics():
    points = torch.zeros(2, 17, 3)
    require_joint_schema(points, actual=JointSchema.COCO17_SURFACE,
                          expected=JointSchema.COCO17_SURFACE)
    with pytest.raises(ValueError, match="semantics"):
        require_joint_schema(points, actual=JointSchema.COCO17_SURFACE,
                              expected=JointSchema.MOTIONBERT_H36M17)
    with pytest.raises(ValueError, match="semantics"):
        require_joint_schema(points, actual=JointSchema.COCO17_SURFACE,
                              expected=JointSchema.SMPLH52_ROTATIONS)
    with pytest.raises(TypeError):
        require_joint_schema(points, actual="coco17", expected=JointSchema.COCO17_SURFACE)


def test_mask_polarity_is_explicit_and_rejects_numeric_mask():
    missing = torch.tensor([[True, False, True]])
    result = presence_mask(missing, MaskMeaning.MISSING)
    assert result.tolist() == [[False, True, False]]
    assert presence_mask(result, MaskMeaning.PRESENT).tolist() == result.tolist()
    assert missing.tolist() == [[True, False, True]]
    with pytest.raises(TypeError):
        presence_mask(missing.float(), MaskMeaning.MISSING)
    with pytest.raises(TypeError):
        presence_mask(missing, "missing")


@pytest.fixture
def calibration():
    K = np.array([[300., 0, 319.5], [0, 300., 239.5], [0, 0, 1]])
    return StereoCalibration(camera_model="fisheye", left_image_size=(640, 480),
                             right_image_size=(640, 480), left_K=K, right_K=K,
                             left_D=np.zeros(4), right_D=np.zeros(4),
                             R=cv2.Rodrigues(np.array([0.1, -0.2, 0.05]))[0],
                             T=np.array([-240., 20., 5.]), length_unit="mm")


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("rotation", ["none", "cw90", "ccw90", "180"])
def test_model_pixel_inverse_and_cam0_metric_ray(calibration, side, rotation):
    point_m = np.array([[0.3, 0.1, 2.0], [-0.2, -0.3, 2.5]])
    projector = calibration.project_left if side == "left" else calibration.project_right
    pixels = projector(point_m * 1000)
    model_pixels = np.array([raw_to_model_point(*p, 640, 480, rotation) for p in pixels])
    result = camera_observations(model_pixels, calibration, side=side,
                                 domain=PixelDomain.MODEL, model_rotation=rotation)
    np.testing.assert_allclose(result.raw_xy, pixels, atol=1e-12, rtol=0)
    expected_origin = np.zeros(3) if side == "left" else -calibration.R.T @ calibration.T * 0.001
    np.testing.assert_allclose(result.ray_origins_m, np.broadcast_to(expected_origin, (2, 3)), atol=1e-12)
    direction = point_m - expected_origin
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    np.testing.assert_allclose(result.ray_directions_cam0, direction, atol=1e-12, rtol=0)
    assert result.point_valid.all() and result.ray_valid.all()
    # Already raw pixels must not receive a second inverse rotation.
    raw = camera_observations(pixels, calibration, side=side,
                              domain=PixelDomain.RAW, model_rotation=rotation)
    np.testing.assert_array_equal(raw.raw_xy, pixels)
    metre_calibration = replace(calibration, T=calibration.T * 0.001, length_unit="meter")
    metre = camera_observations(pixels, metre_calibration, side=side,
                                domain=PixelDomain.RAW, model_rotation=rotation)
    np.testing.assert_allclose(metre.ray_origins_m, result.ray_origins_m, atol=1e-12)
    np.testing.assert_allclose(metre.ray_directions_cam0, result.ray_directions_cam0, atol=1e-12)


def test_bad_pixels_and_inverse_failure_have_separate_states(calibration, monkeypatch):
    pixels = np.array([[10., 20.], [np.nan, 2.], [640., 40.], [639., 479.]])
    result = camera_observations(pixels, calibration, side="left", domain=PixelDomain.RAW,
                                 model_rotation="none")
    assert result.point_valid.tolist() == [True, False, False, True]
    assert result.point_reasons.tolist() == [None, "non_finite_2d_point", "out_of_raw_image_bounds", None]
    assert np.isnan(result.ray_directions_cam0[~result.ray_valid]).all()
    monkeypatch.setattr(StereoCalibration, "undistort_normalized",
                        lambda self, p, side: np.full_like(p, -1e6))
    damaged = camera_observations(pixels[:1], calibration, side="left", domain=PixelDomain.RAW,
                                  model_rotation="none")
    assert damaged.point_valid.all() and not damaged.ray_valid.any()
    assert damaged.ray_reasons.tolist() == ["unprojection_inconsistent"]
    assert np.isfinite(damaged.image_normalized_xy).all()


def test_undeclared_units_domains_and_weak_camera_are_rejected(calibration):
    with pytest.raises(ValueError):
        camera_observations([[10., 20.]], calibration, side="left", domain="raw",
                             model_rotation="none")
    with pytest.raises(ValueError):
        metric_coordinates([1, 2, 3], camera_kind="weak_perspective", unit="meter", frame="cam0")
    with pytest.raises(ValueError):
        metric_coordinates([1, 2, 3], camera_kind="metric_xyz", unit="inch", frame="cam0")
    np.testing.assert_allclose(metric_coordinates([1000, 2000, 3000], camera_kind="metric_xyz",
                                                  unit="mm", frame="cam0"), [1, 2, 3])
    invalid = replace(calibration, R=-np.eye(3))
    with pytest.raises(ValueError, match="extrinsics"):
        camera_observations([[10., 20.]], invalid, side="left", domain=PixelDomain.RAW,
                             model_rotation="none")
