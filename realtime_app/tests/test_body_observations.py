"""Protect the existing soft observation contract during helper extraction."""

import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from pose_app import body_observations as observations
from pose_app.fisheye_camera import fisheye_project_numpy
from pose_app.project_paths import PROJECT_ROOT


def historical_functions():
    path = PROJECT_ROOT / "research_records/engineering_validation/G20260923_smpl_clean_full_sequence_v1/run_clean_full_sequence.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {"raw_side", "raw_triangulate"}]
    namespace = dict(np=np, cv2=cv2, math=math, json=json, Path=Path,
                     fisheye_project_numpy=fisheye_project_numpy)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("channels", [17, 23])
def test_reader_preserves_frame_identity_rotation_and_channel_contract(tmp_path, side, channels):
    points = np.arange(channels * 3, dtype=float).reshape(channels, 3)
    path = tmp_path / "raw.json"
    path.write_text(json.dumps({"images": [{"image_id": 60, "keypoints": [points.tolist()]}]}))
    result = observations.raw_side(path, side)
    assert list(result) == [60]
    assert result[60].shape == (17, 3)
    np.testing.assert_equal(result[60], historical_functions()["raw_side"](path, side)[60])
    np.testing.assert_equal(result[60][:, 2], points[:17, 2])


def test_reader_keeps_invalid_shape_failure(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"images": [{"keypoints": [np.zeros((16, 3)).tolist()]}]}))
    with pytest.raises(ValueError, match="expected 17x3 or 23x3"):
        observations.raw_side(path, "left")


def test_soft_triangulation_retains_weights_masks_rejections_and_units():
    cal = SimpleNamespace(K0=np.array([[800., 0., 960.], [0., 800., 540.], [0., 0., 1.]]),
                          K1=np.array([[800., 0., 960.], [0., 800., 540.], [0., 0., 1.]]),
                          D0=np.zeros(4), D1=np.zeros(4), R_cam0_to_cam1=np.eye(3),
                          T_cam0_to_cam1_mm=np.array([-120., 0., 0.]),
                          image_width=1920, image_height=1080)
    xyz = np.tile([25., 30., 1500.], (17, 1))
    left = np.column_stack((fisheye_project_numpy(xyz, cal.K0, cal.D0), np.ones(17)))[None]
    right = np.column_stack((fisheye_project_numpy(xyz + cal.T_cam0_to_cam1_mm, cal.K1, cal.D1), np.ones(17)))[None]
    left[0, 1, 0] = np.nan
    left[0, 2, 0] = -1.
    left[0, 3, 2] = .16  # soft fitting does not acquire the strict per-view gate
    right[0, 3, 2] = .64
    with np.errstate(divide="ignore", invalid="ignore"):
        before = historical_functions()["raw_triangulate"](left, right, cal)
        after = observations.raw_triangulate(left, right, cal)
    for actual, expected in zip(after, before):
        if isinstance(actual, tuple):
            for a, b in zip(actual, expected):
                np.testing.assert_equal(a, b)
        else:
            np.testing.assert_equal(actual, expected)
    np.testing.assert_allclose(after[0][0, 0], xyz[0], atol=1e-8)
    assert after[7][0, 1] == "nan_inf"
    assert after[7][0, 2] == "out_of_raw_image_bounds"
    assert after[6][0, 3]
    assert after[9][0][0, 3] == pytest.approx(.32)
