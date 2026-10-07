"""Behavioral checks for temporal leakage, missing data and real mesh differentiation."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from realtime_app.learning_flow_mesh.body import BodyDecoder
from realtime_app.learning_flow_mesh.data import (
    assert_disjoint,
    load_dataset,
    masked_mean,
)
from realtime_app.learning_flow_mesh.infer import load_checkpoint
from realtime_app.learning_flow_mesh.model import FlowMesh, ModelConfig
from realtime_app.learning_flow_mesh.rotations import (
    matrix_to_six,
    six_to_matrix,
    validate_rotation_matrices,
)
from realtime_app.learning_flow_mesh.train import loss_for


@pytest.fixture(scope="module")
def body():
    config = json.loads(Path("realtime_app/learning_flow_mesh/config.json").read_text())
    return BodyDecoder(**config["assets"])


@pytest.fixture
def network(body):
    torch.manual_seed(4)
    torch.set_num_threads(2)
    return FlowMesh(ModelConfig(width=32, layers=1), body.parents).eval()


def observations(b=2, t=4):
    xy = torch.randn(b, t, 2, 22, 2) * 0.1
    rays = torch.cat((xy, torch.ones_like(xy[..., :1])), dim=-1)
    rays = rays / rays.norm(dim=-1, keepdim=True)
    return {"xy": xy, "confidence": torch.full(xy.shape[:-1], 0.7),
            "view_mask": torch.ones(b, t, 2, dtype=torch.bool),
            "observation_mask": torch.ones(xy.shape[:-1], dtype=torch.bool),
            "geometry_mask": torch.ones(xy.shape[:-1], dtype=torch.bool),
            "ray_origins": torch.zeros(b, t, 2, 22, 3), "ray_directions": rays,
            "image_features": torch.randn(b, t, 2, 32),
            "image_mask": torch.ones(b, t, 2, dtype=torch.bool)}


@pytest.mark.parametrize("scale", [0, 0.2, 3.141592])
def test_rotation_roundtrip(scale):
    torch.manual_seed(8)
    vector = torch.randn(60, 3, dtype=torch.float64)
    vector = vector / vector.norm(dim=-1, keepdim=True) * scale
    skew = torch.zeros(60, 3, 3, dtype=torch.float64)
    skew[:, 0, 1], skew[:, 1, 0] = -vector[:, 2], vector[:, 2]
    skew[:, 0, 2], skew[:, 2, 0] = vector[:, 1], -vector[:, 1]
    skew[:, 1, 2], skew[:, 2, 1] = -vector[:, 0], vector[:, 0]
    rotation = torch.matrix_exp(skew)
    decoded, valid = six_to_matrix(matrix_to_six(rotation))
    assert valid.all()
    validate_rotation_matrices(decoded, atol=1e-8)
    torch.testing.assert_close(rotation, decoded, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("bad", [[0, 0, 0, 1, 0, 0], [1, 0, 0, 2, 0, 0], [float("nan"), 0, 0, 0, 1, 0]])
def test_degenerate_rotations_retained(bad):
    matrix, valid = six_to_matrix(torch.tensor(bad, dtype=torch.float32))
    assert not valid
    assert torch.isnan(matrix).all()


def test_no_future_leakage_and_stream_equivalence(network):
    batch = observations()
    changed = copy.deepcopy(batch)
    for key in ("xy", "confidence", "ray_origins", "ray_directions", "image_features"):
        changed[key][:, 2:] = batch[key][:, 2:].flip(0)
    with torch.no_grad():
        full = network.encoder(batch)
        altered = network.encoder(changed)
        prefix = network.encoder({k: value[:, :2] for k, value in batch.items()})
        tail = network.encoder({k: value[:, 2:] for k, value in batch.items()},
                               hidden=prefix["hidden"], fixed_betas=prefix["betas"])
        sampled = network.transport(full, steps=3, seed=10)
        sample_prefix = network.transport(prefix, steps=3, seed=10)
    for key in ("condition", "translation"):
        torch.testing.assert_close(full[key][:, :2], altered[key][:, :2], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(full[key][:, :2], prefix[key], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(full[key][:, 2:], tail[key], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(full["betas"], altered["betas"])
    torch.testing.assert_close(full["betas"], tail["betas"])
    torch.testing.assert_close(sampled["state"][:, :2], sample_prefix["state"], atol=1e-6, rtol=1e-6)


def test_invalid_masked_values_cannot_change_prediction(network):
    batch = observations()
    batch["observation_mask"][..., 4:9] = False
    batch["image_mask"][..., 1] = False
    other = copy.deepcopy(batch)
    for key in ("xy", "ray_origins", "ray_directions"):
        other[key][..., 4:9, :] = float("nan")
    other["confidence"] = batch["confidence"].clone()
    other["confidence"][..., 4:9] = float("nan")
    other["image_features"][..., 1, :] = float("nan")
    with torch.no_grad():
        first, second = network.encoder(batch), network.encoder(other)
    torch.testing.assert_close(first["condition"], second["condition"])


def test_all_missing_is_finite_but_unavailable(network):
    batch = observations()
    for key in ("observation_mask", "view_mask", "image_mask", "geometry_mask"):
        batch[key].zero_()
    for key in ("xy", "confidence", "ray_origins", "ray_directions", "image_features"):
        batch[key].fill_(float("nan"))
    encoded = network.encoder(batch)
    assert torch.isfinite(encoded["condition"]).all()
    assert not encoded["shape_available"].any()
    assert not encoded["frame_observed"].any()


def test_zero_confidence_does_not_remove_present_coordinate(network):
    batch = observations()
    batch["confidence"].zero_()
    other = copy.deepcopy(batch)
    other["xy"] += 0.4
    first, second = network.encoder(batch), network.encoder(other)
    assert not torch.allclose(first["condition"], second["condition"], atol=1e-7)


def test_present_nonfinite_and_bad_rays_fail(network):
    batch = observations()
    batch["xy"][0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        network.encoder(batch)
    batch = observations()
    batch["ray_directions"] *= 2
    with pytest.raises(ValueError, match="unit"):
        network.encoder(batch)


def test_actual_rgb_causal_and_differentiable(network):
    batch = observations(b=1)
    del batch["image_features"]
    batch["images"] = torch.rand(1, 4, 2, 3, 24, 24, requires_grad=True)
    encoded = network.encoder(batch)
    encoded["translation"][:, 1].square().sum().backward()
    assert batch["images"].grad[:, :2].abs().max() > 1e-8
    assert batch["images"].grad[:, 2:].abs().max() == 0
    assert torch.isfinite(batch["images"].grad).all()
    batch["images"] = batch["images"].detach() * 255
    with pytest.raises(ValueError, match="RGB"):
        network.encoder(batch)


def test_missing_initial_shape_not_invented_from_future(network):
    batch = observations()
    batch["observation_mask"][:, 0] = False
    batch["image_mask"][:, 0] = False
    encoded = network.encoder(batch)
    assert not encoded["shape_available"].any()
    assert encoded["frame_observed"][:, 1:].all()
    resumed = network.encoder({k: value[:, 1:] for k, value in batch.items()})
    assert resumed["shape_available"].all()


def test_same_spatial_parameters_receive_regression_gradients(network):
    loss = network.regression(network.encoder(observations())).square().mean()
    loss.backward()
    gradient = network.blocks[0].graph.weight.grad
    assert gradient is not None and gradient.abs().max() > 1e-8


def test_real_body_gradients_and_faces(body):
    rotations = matrix_to_six(torch.eye(3).expand(1, 2, 52, 3, 3)).clone().requires_grad_(True)
    beta = torch.zeros(1, 10, requires_grad=True)
    trans = torch.zeros(1, 2, 3, requires_grad=True)
    matrices, valid = six_to_matrix(rotations)
    assert valid.all()
    output = body(matrices, beta, trans)
    assert output["vertices"].shape == (1, 2, 6890, 3)
    assert output["faces"].shape == (13776, 3)
    loss = output["vertices"].square().mean() + output["vertices"][..., 0].mean()
    loss.backward()
    for param in (rotations, beta, trans):
        assert torch.isfinite(param.grad).all() and param.grad.abs().max() > 1e-7
    assert rotations.grad[:, :, 22:37].abs().max() > 1e-7
    assert rotations.grad[:, :, 37:52].abs().max() > 1e-7
    shift = trans.detach() + torch.tensor([1., 2., 3.])
    moved = body(matrices.detach(), beta.detach(), shift)
    torch.testing.assert_close(moved["vertices"] - output["vertices"].detach(),
                               torch.tensor([1., 2., 3.]).expand(1, 2, 6890, 3))


def test_network_ode_mesh_loss_backprop(network, body):
    prediction = network.transport(network.encoder(observations(b=1, t=1)), steps=2)
    assert prediction["rotation_valid"].all()
    mesh = body(prediction["rotations"], prediction["betas"], prediction["translation"])
    mesh["vertices"].square().mean().backward()
    for name in ("velocity_head.1.weight", "encoder.observation.0.weight", "encoder.shape_head.weight"):
        gradient = dict(network.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().max() > 1e-8


def test_checkpoint_seed_replay(network, tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save({"config": {"model": network.cfg.to_dict()}, "model_state": network.state_dict()}, path)
    reloaded, _ = load_checkpoint(path, "cpu")
    batch = observations()
    with torch.no_grad():
        first = network.transport(network.encoder(batch), steps=3, seed=8)
        second = reloaded.transport(reloaded.encoder(batch), steps=3, seed=8)
        third = reloaded.transport(reloaded.encoder(batch), steps=3, seed=9)
    assert torch.equal(first["state"], second["state"])
    assert not torch.equal(first["state"], third["state"])


def test_split_overlap_rejected():
    metadata = {"subject_ids": ["p1"], "sequence_ids": ["s1"], "reference_frame": "cam0",
                "units": "metres", "rotation_convention": "local", "observation_topology": "body"}
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(metadata, metadata)


def test_dataset_rejects_missing_required_fields(tmp_path):
    path = tmp_path / "wrong.npz"
    np.savez(path, metadata=np.array('{}'))
    with pytest.raises(ValueError, match="missing"):
        load_dataset(path, 22, 32)


def test_label_free_dataset_and_boolean_contract(tmp_path):
    batch = observations(b=1, t=2)
    metadata = {"label_source": "none", "reference_frame": "cam0", "units": "metres",
                "rotation_convention": "absolute_local_matrix_columns", "subject_ids": ["p"],
                "sequence_ids": ["s"], "split": "test", "observation_topology": "smplh22",
                "geometry_source": "calibrated", "image_source": "test_features"}
    arrays = {k: value.numpy() for k, value in batch.items()}
    arrays["metadata"] = np.array(json.dumps(metadata))
    path = tmp_path / "observations.npz"
    np.savez(path, **arrays)
    data, _ = load_dataset(path, 22, 32, require_targets=False)
    assert not data["pose_mask"].any() and torch.isnan(data["rotations"]).all()
    arrays["observation_mask"] = arrays["observation_mask"].astype(np.float32)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="boolean"):
        load_dataset(path, 22, 32, require_targets=False)


def test_masked_loss_and_missing_hand_labels(network):
    values = torch.tensor([1., float("nan")], requires_grad=True)
    loss = masked_mean(values, torch.tensor([True, False]))
    assert loss == 1
    loss.backward()
    assert torch.equal(values.grad, torch.tensor([1., 0.]))
    batch = observations()
    batch.update(rotations=torch.eye(3).expand(2, 4, 52, 3, 3).clone(),
                 pose_mask=torch.ones(2, 4, 52, dtype=torch.bool), betas=torch.zeros(2, 10),
                 shape_mask=torch.ones(2, dtype=torch.bool), translation=torch.zeros(2, 4, 3),
                 translation_mask=torch.ones(2, 4, dtype=torch.bool))
    batch["pose_mask"][..., 22:] = False
    settings = {"shape_weight": 0.1, "translation_weight": 1}
    first, _ = loss_for(network, batch, "flow", settings, random_seed=123)
    batch["rotations"][..., 22:, :, :] = float("nan")
    second, _ = loss_for(network, batch, "flow", settings, random_seed=123)
    torch.testing.assert_close(first, second)
    second.backward()
    assert torch.isfinite(network.velocity_head[1].weight.grad).all()
