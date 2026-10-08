"""Controls for original source sampling arithmetic and actual deployment inputs."""
from types import SimpleNamespace

import pytest
import torch

from realtime_app.learning_flow_mesh.reproduction.check_interfaces import (
    original_function,
)
from realtime_app.learning_flow_mesh.reproduction.run_original import ROOT
from realtime_app.learning_flow_mesh.reproduction.run_sampling import (
    aggregation_functions,
    original_method,
    root_dependency_probe,
)


@pytest.mark.parametrize("steps", [1, 3, 5])
def test_original_euler_matches_analytic_linear_field_and_preserves_noise(steps):
    sampler = original_function(ROOT / "FMPose3D/fmpose3d/common/utils.py", "euler_sample")
    noise = torch.arange(51, dtype=torch.float64).reshape(1, 1, 17, 3) / 100
    before = noise.clone()
    condition = torch.zeros(1, 1, 17, 2, dtype=torch.float64)
    times = []

    def field(_condition, y, t):
        times.append(float(t[0, 0, 0, 0]))
        return y

    result = sampler(condition, noise, steps, field)
    torch.testing.assert_close(result, noise * (1 + 1 / steps) ** steps, atol=1e-12, rtol=0)
    torch.testing.assert_close(noise, before, atol=0, rtol=0)
    assert times == pytest.approx([s / steps for s in range(steps)], abs=1e-15, rel=0)


def test_released_rpea_root_dependency_and_distance_saturation_are_retained():
    report = root_dependency_probe()
    assert report["docstring_shape_only_claim_contradicted"]
    for branch in report["functions"].values():
        assert branch["root_change_max_output_delta_m_fixture"] > 0.1
        assert branch["nonroot_truth_change_max_output_delta"] == 0
    assert report["far_error_clamp_to_uniform_mean_max_difference"] < 1e-12
    # Original eval code's unsupported backward is documented, not repaired here.
    assert report["optional_backward_probe"]["status"] in ("passed", "original_backward_failed")


def test_aggregation_really_requires_camera_and_root_values_not_just_shape():
    _hard, soft, _projection = aggregation_functions()
    args = SimpleNamespace(pad=0, topk=3, exp_temp=0.002)
    hypotheses = [torch.zeros(1, 1, 17, 3)]
    camera = torch.tensor([[1., 1., 0., 0., 0., 0., 0., 0., 0.]])
    target = torch.zeros(1, 1, 17, 2)
    with pytest.raises(AttributeError):
        soft(args, hypotheses, camera, target, None)
    # Nonfinite supplied root must not be mistaken for valid absolute geometry.
    gt = torch.zeros(1, 1, 17, 3)
    gt[:, :, 0, 2] = float("nan")
    output = soft(args, hypotheses, camera, target, gt)
    # Released RPEA replaces NaN weights with uniform weights and returns finite
    # candidates despite missing root. This behavior is unsafe as a validity gate.
    assert torch.isfinite(output).all()


def test_original_human_postprocess_root_zero_with_explicit_world_rotation_off():
    import numpy as np

    source = ROOT / "FMPose3D/fmpose3d/inference_api/fmpose3d.py"
    function = original_method(source, "HumanPostProcessor", "__call__", {"torch": torch, "np": np})
    raw = torch.arange(51, dtype=torch.float32).reshape(1, 1, 17, 3)
    before = raw.clone()
    pose, world = function(SimpleNamespace(), raw, camera_rotation=None)
    assert pose.shape == (17, 3) and np.array_equal(pose, world)
    assert not np.shares_memory(pose, world)
    assert np.array_equal(pose[0], np.zeros(3))
    # Original root-zeroing is in-place; other already root-relative joints stay unchanged.
    torch.testing.assert_close(raw[:, :, 1:], before[:, :, 1:], atol=0, rtol=0)
    assert torch.equal(raw[:, :, 0], torch.zeros(1, 1, 3))
