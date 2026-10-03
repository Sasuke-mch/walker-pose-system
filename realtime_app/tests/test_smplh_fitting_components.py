"""Fitting contracts: trainability, missing observations and geometry gradients."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pose_app.smplh_fitting.initialization import make_body_basis
from pose_app.smplh_fitting.losses import (
    body_temporal_loss,
    hand_temporal_loss,
    body_reprojection_loss,
)
from pose_app.smplh_fitting.contact import load_surface_targets, surface_contact_losses
from pose_app.smplh_fitting.stages import build_stage_schedule, set_trainable_parameters
from walker_tools.body.fit_smplh_wilor_sequence import build_parser, read_wilor


def options(**overrides):
    args = SimpleNamespace(
        steps=180,
        base_steps=None,
        beta_steps=None,
        joint_steps=None,
        hand_steps=None,
        contact_steps=None,
        contact_refine_steps=None,
        shared_hand_pose=False,
        hand_2d_weight=1e-7,
        body_temporal_weight=1.0,
        body_reprojection_weight=1.0,
        body_reprojection_scale_px=100.0,
        contact_labels=None,
    )
    vars(args).update(overrides)
    return args


@pytest.mark.parametrize("contact", [False, True])
def test_stage_budget_and_freezing_preserve_one_shared_beta(contact):
    names, train, steps = build_stage_schedule(options(), contact)
    assert len(names) == len(train) == len(steps) == 6
    assert steps == [30, 30, 30, 30, 30, 30 if contact else 0]
    assert train[:4] == [
        ["transl", "latent"],
        ["beta"],
        ["beta", "root", "transl", "latent"],
        ["lhand", "rhand"],
    ]
    assert (
        train[4:]
        == [["lhand", "rhand"] + (["root", "transl", "latent"] if contact else [])] * 2
    )
    params = {
        k: torch.nn.Parameter(torch.zeros(1, 10) if k == "beta" else torch.zeros(4, 3))
        for k in ("beta", "root", "transl", "latent", "lhand", "rhand")
    }
    beta = params["beta"]
    for active in train:
        set_trainable_parameters(params, active)
        assert {k for k, v in params.items() if v.requires_grad} == set(active)
        assert params["beta"] is beta and beta.shape == (1, 10)


def test_explicit_zero_budget_is_not_replaced_by_default():
    _, _, steps = build_stage_schedule(options(base_steps=0, joint_steps=300), False)
    assert steps == [0, 30, 300, 30, 30, 0]


def test_body_basis_is_rigid_equivariant_and_rejects_degenerate_input():
    from scipy.spatial.transform import Rotation

    p = np.zeros((17, 3))
    p[[5, 6, 11, 12]] = [[1, 1, 0], [-1, 1, 0], [1, 0, 0], [-1, 0, 0]]
    rotation = Rotation.from_rotvec([0.2, -0.4, 0.1]).as_matrix()
    np.testing.assert_allclose(
        make_body_basis(p @ rotation.T + [2, 3, 4]), rotation, atol=1e-15
    )
    assert make_body_basis(np.zeros((17, 3))) is None
    p[5, 0] = np.nan
    assert make_body_basis(p) is None


@pytest.mark.parametrize("stage", range(6))
def test_body_temporal_excludes_gaps_and_obeys_stage_activation(stage):
    coco = torch.zeros(5, 17, 3, requires_grad=True)
    with torch.no_grad():
        coco[2, 0, 0] = 0.1
    mask = torch.ones(5, 17, dtype=torch.bool)
    inputs = dict(
        args=options(),
        n=5,
        stage_index=stage,
        body_loss=coco.sum() * 0,
        coco=coco,
        mask_body=mask,
    )
    value = body_temporal_loss(**inputs)
    if stage in (0, 2):
        assert value > 0
        assert torch.isfinite(torch.autograd.grad(value, coco)[0]).all()
    else:
        assert value == 0 and not value.requires_grad
    mask[2, 1] = False  # Every triplet crosses the unsupported frame.
    assert body_temporal_loss(**inputs) == 0


@pytest.mark.parametrize("shared", [False, True])
def test_hand_temporal_uses_native_support_when_pixels_are_disabled(shared):
    left = torch.tensor([[0.0], [1.0], [0.0]], requires_grad=True)
    right = torch.zeros_like(left, requires_grad=True)
    empty = torch.zeros(3, 21, dtype=torch.bool)
    inputs = dict(
        args=options(shared_hand_pose=shared, hand_2d_weight=0),
        n=3,
        lhand=left,
        rhand=right,
        mask_ll=empty,
        mask_rl=empty,
        mask_lr=empty,
        mask_rr=empty,
        mano_enabled=True,
        mano_weights={"left": torch.ones(3, 2), "right": torch.ones(3, 2)},
    )
    value = hand_temporal_loss(**inputs)
    assert value.item() == (0 if shared else 2)
    if not shared:
        np.testing.assert_array_equal(
            torch.autograd.grad(value, left)[0], [[-2.0], [4.0], [-2.0]]
        )
    inputs["mano_weights"]["left"][1] = 0
    assert hand_temporal_loss(**inputs) == 0


def test_reprojection_scale_and_confidence_mask_have_physical_units():
    projected = torch.tensor([[[300.0, 400.0], [800.0, 600.0]]], requires_grad=True)
    confidence = torch.tensor([[1.0, 0.0]])
    value, pixels = body_reprojection_loss(
        args=options(),
        stage_index=0,
        body_loss=projected.sum() * 0,
        proj_cam0=projected,
        proj_cam1=projected,
        raw_left_2d=torch.zeros_like(projected),
        raw_right_2d=torch.zeros_like(projected),
        raw_left_conf=confidence,
        raw_right_conf=confidence,
        raw_left_valid=confidence.bool(),
        raw_right_valid=confidence.bool(),
    )
    assert value.item() == 4.5 and pixels.item() == 500
    gradient = torch.autograd.grad(value, projected)[0]
    np.testing.assert_allclose(gradient[0, 0], [0.006, 0.008], rtol=1e-6)
    assert torch.count_nonzero(gradient[0, 1]) == 0


def test_disabled_contact_needs_no_surface_assets():
    contact = load_surface_targets(options(), 4, "cpu")
    assert all(v is None for v in vars(contact).values())
    points = torch.ones(4, 21, 2, requires_grad=True)
    terms = surface_contact_losses(
        left_in_cam0=points,
        contact_enabled=False,
        stage_index=5,
        vertices=None,
        contact=contact,
    )
    assert all(t.item() == 0 for t in terms)
    assert torch.count_nonzero(torch.autograd.grad(sum(terms), points)[0]) == 0


def test_contact_weights_preserve_selected_surface_gradients():
    from dataclasses import replace
    from pose_app.smplh_fitting.contact import SurfaceTargets

    vertices = torch.tensor(
        [
            [
                [0.0, 0.032, 0.0],
                [0.0, 0.05, 0.0],
                [0.0, 0.0, -0.01],
                [0.0, 0.0, -0.03],
                [1.0, 1.0, 1.0],
            ]
        ],
        requires_grad=True,
    )
    target = SurfaceTargets(
        hand_weight=torch.tensor([[1.0, 0.0]]),
        foot_weight=torch.tensor([[1.0, 0.0]]),
        handle=torch.tensor(
            [[[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]]
        ),
        rotation=torch.eye(3)[None],
        translation=torch.zeros(1, 3),
        palm_indices={"left": [0], "right": [1]},
        sole_indices={"left": [2], "right": [3]},
        global_hand_offset=None,
        global_hand_surface=None,
        camera_rotation=None,
        camera_translation=None,
    )
    inputs = dict(
        left_in_cam0=vertices[:, :2, :2],
        contact_enabled=True,
        stage_index=4,
        vertices=vertices,
    )
    terms = surface_contact_losses(**inputs, contact=target)
    scaled = surface_contact_losses(
        **inputs,
        contact=replace(
            target,
            hand_weight=target.hand_weight * 10,
            foot_weight=target.foot_weight * 10,
        ),
    )
    torch.testing.assert_close(torch.stack(terms), torch.stack(scaled))
    gradient = torch.autograd.grad(sum(terms), vertices)[0]
    assert gradient[0, 0, 1] > 0  # Descent moves the palm toward the handle surface.
    assert gradient[0, 2, 2] < 0  # Descent lifts the penetrating sole.
    assert torch.count_nonzero(gradient[:, [1, 3, 4]]) == 0


def test_contact_loader_rejects_mismatched_frames_before_optimization(tmp_path):
    labels, scene = tmp_path / "labels.npz", tmp_path / "scene.npz"
    np.savez(labels, handle_ends_ground_m=np.zeros((3, 2, 2, 3)))
    np.savez(scene)
    with pytest.raises(ValueError, match="frame count"):
        load_surface_targets(
            options(contact_labels=labels, scene_transforms=scene), 4, "cpu"
        )


def test_hand_reader_remains_shared_and_disabled_pixels_remain_unavailable(tmp_path):
    from pose_app.smplh_hand_observation import read_wilor as shared_reader

    assert read_wilor is shared_reader
    obs, valid, bounds, weights, audit = read_wilor(
        tmp_path / "missing.jsonl", 4, "left", None, (1920, 1080), "left", False
    )
    assert np.isnan(obs).all() and obs.shape == (4, 21, 2)
    assert not valid.any() and not bounds.any() and not weights.any()
    assert audit[0]["reason"] == "hand_2d_disabled_not_consumed"


def test_cli_retains_mainline_defaults():
    parser = build_parser()
    required = [a.option_strings[0] for a in parser._actions if a.required]
    args = parser.parse_args([item for flag in required for item in (flag, "dummy")])
    assert args.steps == 180 and args.lr == 0.02
    assert args.body_temporal_weight == args.body_reprojection_weight == 0
    assert args.hand_pca_comps == 12 and not args.shared_hand_pose
