"""Original P16 geometry controls, without allocating the image networks."""
import copy
import importlib.util
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.optimize import least_squares

from realtime_app.learning_flow_mesh.reproduction.check_interfaces import (
    original_function,
)
from realtime_app.learning_flow_mesh.reproduction.run_multiview import (
    SequenceRandom,
    camera_fixture,
    ransac_controls,
)
from realtime_app.learning_flow_mesh.reproduction.run_original import ROOT
from realtime_app.learning_flow_mesh.reproduction.run_sampling import original_method


@pytest.fixture
def geometry():
    path = ROOT / "LearnableTriangulation/mvn/utils/multiview.py"
    spec = importlib.util.spec_from_file_location("p16_geometry_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def preserve_random_state():
    state = random.getstate()
    try:
        yield
    finally:
        random.setstate(state)


def ransac_method(geometry, sampler):
    path = ROOT / "LearnableTriangulation/mvn/models/triangulation.py"
    method = original_method(path, "RANSACTriangulationNet", "triangulate_ransac",
                             {"np": np, "multiview": geometry, "random": sampler,
                              "least_squares": least_squares})
    return SimpleNamespace(triangulate_ransac=lambda *a, **kw: method(None, *a, **kw))


def test_compatibility_facade_retains_seeded_sequence_sampling_and_module():
    original_sample = random.sample
    random.seed(51)
    expected = [random.sample([0, 1, 2, 3], 2) for _ in range(10)]
    random.seed(51)
    actual = [SequenceRandom.sample({0, 1, 2, 3}, 2) for _ in range(10)]
    assert actual == expected and random.sample is original_sample
    with pytest.raises(TypeError, match="integer view sets"):
        SequenceRandom.sample({"a", "b"}, 2)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="set rejection introduced in Python3.11")
def test_unmodified_original_ransac_preserves_python_set_failure(geometry):
    model = ransac_method(geometry, random)
    _cameras, projection = camera_fixture(geometry, "cpu", [384, 384])
    with pytest.raises(TypeError, match="Population must be a sequence"):
        ransac_controls(model, geometry, projection)


def test_original_ransac_outlier_and_two_view_no_redundancy_controls(geometry):
    model = ransac_method(geometry, SequenceRandom)
    _cameras, projection = camera_fixture(geometry, "cpu", [384, 384])
    report = ransac_controls(model, geometry, projection)
    assert report["robust_inliers"] == [0, 1, 2]
    assert report["robust_error_mm"] < 1e-5
    assert report["two_view_forced_inliers"] == [0, 1]
    assert report["two_view_outlier_xyz_error_mm_fixture"] > 1000


def test_original_threshold_is_half_distance_and_strict_less_than(geometry):
    # Force a single clean sampled pair; third view lies exactly on the boundary.
    sampler = SimpleNamespace(sample=lambda _population, _k: [0, 1])
    model = ransac_method(geometry, sampler)
    _cameras, projection = camera_fixture(geometry, "cpu", [384, 384])
    matrices = projection[0].numpy().astype(np.float64)[:3]
    truth = np.array([[120., -40., 3000.]])
    points = np.stack([geometry.project_3d_points_to_image_plane_without_distortion(p, truth)[0]
                       for p in matrices])
    # 29.99px is inside; 30.01px is outside, with room for DLT roundoff.
    for offset, expected in ((29.99, [0, 1, 2]), (30.01, [0, 1])):
        altered = points.copy()
        altered[2, 0] += offset
        errors = geometry.calc_reprojection_error_matrix(truth, altered, matrices)
        assert errors[0, 2] == pytest.approx(offset / 2)
        _xyz, inliers = model.triangulate_ransac(matrices, altered, n_iters=1,
                                                direct_optimization=False)
        assert inliers.tolist() == expected


def test_camera_resize_preserves_original_and_non_square_axis_scaling(geometry):
    cameras, _projection = camera_fixture(geometry, "cpu", [384, 384])
    original = cameras[0][0]
    before = original.K.copy()
    resized = copy.deepcopy(original)
    resized.update_after_resize([384, 384], [96, 192])
    assert np.array_equal(original.K, before)
    np.testing.assert_array_equal(resized.K[0], before[0] / 2)
    np.testing.assert_array_equal(resized.K[1], before[1] / 4)
    np.testing.assert_array_equal(resized.R, original.R)
    np.testing.assert_array_equal(resized.t, original.t)


def test_original_volume_coordinate_integration_retains_metric_axes_and_shift():
    integrate = original_function(ROOT / "LearnableTriangulation/mvn/utils/op.py",
                                  "integrate_tensor_3d_with_coordinates")
    # The isolated AST needs the same module symbol as the original op.py import.
    integrate.__globals__["nn"] = torch.nn
    axes = torch.meshgrid(torch.tensor([-20., 80.]), torch.tensor([4., 8., 12.]),
                          torch.tensor([1000., 2000., 3000., 4000.]), indexing="ij")
    coordinates = torch.stack(axes, -1)[None]
    logits = torch.zeros(1, 1, 2, 3, 4)
    xyz, probabilities = integrate(logits, coordinates)
    torch.testing.assert_close(xyz, torch.tensor([[[30., 8., 2500.]]]))
    torch.testing.assert_close(probabilities.sum((2, 3, 4)), torch.ones(1, 1))
    shifted, _ = integrate(logits, coordinates + torch.tensor([100., -30., 500.]))
    torch.testing.assert_close(shifted - xyz, torch.tensor([[[100., -30., 500.]]]))
    logits[0, 0, 0, 0, 0] = torch.nan
    bad_xyz, bad_probabilities = integrate(logits, coordinates)
    assert not torch.isfinite(bad_xyz).any() and not torch.isfinite(bad_probabilities).any()
