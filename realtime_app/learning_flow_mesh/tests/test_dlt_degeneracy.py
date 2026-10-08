"""R14 author geometry counterexamples, with independent stereo control."""
import numpy as np
import pytest
import torch

from realtime_app.learning_flow_mesh.reproduction.check_dlt_degeneracy import (
    fixture,
    load_geometry,
)


def test_original_dlt_matches_independent_stereo_and_confidence_scaling():
    geometry = load_geometry()
    cameras, point, pixels = fixture()
    p, xy = torch.from_numpy(cameras), torch.from_numpy(pixels)
    fn = geometry.triangulate_point_from_multiple_views_linear_torch
    z = 384. * 600. / (pixels[0, 0] - pixels[1, 0])
    independent = torch.tensor([(pixels[0, 0] - 192.) * z / 384. - 300.,
                                (pixels[0, 1] - 192.) * z / 384., z])
    result = fn(p, xy)
    assert torch.allclose(result, independent, atol=1e-7, rtol=0)
    assert np.max(np.abs(result.numpy() - point)) < 1e-7
    assert torch.allclose(fn(p, xy, torch.tensor([3., 3.])), result, atol=1e-7, rtol=0)


def test_zero_confidence_finite_origin_is_not_a_valid_reconstruction():
    geometry = load_geometry()
    cameras, _point, pixels = fixture()
    result = geometry.triangulate_point_from_multiple_views_linear_torch(
        torch.from_numpy(cameras), torch.from_numpy(pixels), torch.zeros(2))
    assert torch.equal(result, torch.zeros(3, dtype=torch.float64))
    # Zero-depth projection demonstrates why an isfinite(xyz) gate is insufficient.
    projected = geometry.project_3d_points_to_image_plane_without_distortion(
        torch.from_numpy(cameras[0]), result[None])
    assert not torch.isfinite(projected).all()


def test_author_missing_views_and_nan_raise_without_filling():
    geometry = load_geometry()
    cameras, _point, pixels = fixture()
    fn = geometry.triangulate_point_from_multiple_views_linear_torch
    for count in (0, 1):
        with pytest.raises(IndexError):
            fn(torch.from_numpy(cameras[:count]), torch.from_numpy(pixels[:count]))
    bad = torch.from_numpy(pixels.copy())
    bad[0, 0] = torch.nan
    with pytest.raises(torch.linalg.LinAlgError):
        fn(torch.from_numpy(cameras), bad)
