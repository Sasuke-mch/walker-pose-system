"""Original HuManiFlow input semantics; missing channels are not validity gates."""
import pytest
import torch

from realtime_app.learning_flow_mesh.reproduction.check_original_proxy import (
    forced_visibility,
    input_semantics_probes,
    original_proxy_statements,
)


def test_original_gaussian_affine_and_missing_semantics_against_independent_controls():
    result = input_semantics_probes()
    assert result["gaussian_row_column_control_error"] < 1e-6
    assert result["affine_pixel_control_error"] < 2e-5
    assert result["asymmetric_argmax_exact"]
    assert result["crop_mutates_bbox_height_for_aspect_ratio"]
    assert result["zero_heatmap_nonzero_gaussian_channels_after_original_mask"] == 7
    assert result["masked_nan_joint_heatmap_remains_nan"]


@pytest.mark.parametrize("confidence,expected", [(-1., False), (0., False), (.75, False), (.751, True)])
def test_original_visibility_forces_head_and_shoulders_even_when_confidence_missing(confidence, expected):
    scores = torch.full((17,), confidence)
    before = scores.clone()
    result = forced_visibility(scores)
    assert result[:7].all()
    assert result[7:].eq(expected).all()
    assert torch.equal(before, scores)


def test_original_ast_subpath_selection_preserves_prediction_boundary():
    _full, full_lines, full_count = original_proxy_statements()
    _pre, pre_lines, pre_count = original_proxy_statements(include_prediction=False)
    _vis, vis_lines, vis_count = original_proxy_statements(visibility_only=True)
    assert full_lines[0] == pre_lines[0] and full_lines[1] > pre_lines[1]
    assert full_count == pre_count + 1 and vis_count == 2
    assert full_lines[0] < vis_lines[0] < vis_lines[1] < pre_lines[1]
