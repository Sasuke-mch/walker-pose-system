"""Prevent unintended weight loading and reflected rotation acceptance."""
import importlib

import pytest
import torch

from realtime_app.learning_flow_mesh.reproduction.run_probability import (
    install_download_guard,
    rotation_check,
)


def test_original_model_weight_download_is_blocked(monkeypatch):
    zoo = importlib.import_module("torch.utils.model_zoo")
    # Save originals in pytest's cleanup stack before installing process-local guard.
    for module, name in ((zoo, "load_url"), (torch.hub, "load_state_dict_from_url"),
                         (torch.hub, "download_url_to_file")):
        monkeypatch.setattr(module, name, getattr(module, name))
    install_download_guard()
    for loader in (zoo.load_url, torch.hub.load_state_dict_from_url, torch.hub.download_url_to_file):
        with pytest.raises(RuntimeError, match="Weight download attempted"):
            loader("https://invalid.example/model.pth")


def test_orthogonal_reflection_is_not_a_pose_rotation():
    reflection = torch.diag(torch.tensor([1., 1., -1.]))[None]
    with pytest.raises(AssertionError):
        rotation_check(reflection)
    assert rotation_check(torch.eye(3)[None])["determinant_max_error"] == 0
