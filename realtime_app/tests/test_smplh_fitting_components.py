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


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--joint-steps", "-1"),
        ("--contact-refine-steps", "-1"),
        ("--steps", "-1"),
        ("--lr", "nan"),
        ("--lr", "0"),
        ("--body-temporal-weight", "nan"),
        ("--surface-foot-contact-weight", "inf"),
        ("--max-init-body-rms-mm", "nan"),
        ("--bone-weight", "-1"),
    ],
)
def test_invalid_options_stop_before_creating_output_or_loading_assets(
    tmp_path, flag, value
):
    from pose_app.smplh_fitting.pipeline import run_fit

    parser = build_parser()
    required = [a.option_strings[0] for a in parser._actions if a.required]
    argv = [item for f in required for item in (f, "nonexistent")]
    argv += ["--output-dir", str(tmp_path / "fit"), flag, value]
    with pytest.raises(ValueError, match="finite|non-negative"):
        run_fit(parser.parse_args(argv))
    assert not (tmp_path / "fit").exists()


@pytest.fixture
def surface_input_files(tmp_path):
    import json

    args = options(
        contact_labels=tmp_path / "labels.npz",
        scene_transforms=tmp_path / "scene.npz",
        contact_vertex_sets=tmp_path / "sets.json",
        walker_topology=tmp_path / "topology.json",
        global_hand_handle_pose=None,
    )
    labels = dict(
        handle_ends_ground_m=np.zeros((2, 2, 2, 3)),
        hand_contact_weight=np.ones((2, 2)),
        foot_contact_weight=np.ones((2, 2)),
    )
    scene = dict(
        rotation_ground_from_left=np.tile(np.eye(3), (2, 1, 1)),
        translation_ground_from_left_mm=np.zeros((2, 3)),
    )
    sets = {
        f"{side}_{part}_surface_candidate": {"vertices": [0, 1]}
        for side in ("left", "right")
        for part in ("palm", "sole")
    }
    topology = dict(
        handle_segments={"left": [0, 1], "right": [2, 3]},
        rotation_left_camera_from_walker=np.eye(3).tolist(),
        translation_left_camera_from_walker_mm=[0, 0, 0],
    )
    np.savez(args.contact_labels, **labels)
    np.savez(args.scene_transforms, **scene)
    args.contact_vertex_sets.write_text(json.dumps({"sets": sets}), encoding="utf-8")
    args.walker_topology.write_text(json.dumps(topology), encoding="utf-8")
    return args, labels, scene, sets


@pytest.mark.parametrize("indices", [[-1], [6890], [1.2], []])
def test_sole_indices_do_not_wrap_or_truncate(surface_input_files, indices):
    import json

    args, _, _, sets = surface_input_files
    sets["left_sole_surface_candidate"]["vertices"] = indices
    args.contact_vertex_sets.write_text(json.dumps({"sets": sets}), encoding="utf-8")
    with pytest.raises(ValueError, match="left sole vertices"):
        load_surface_targets(args, 2, "cpu")


def test_surface_transform_rejects_nonfinite_translation(surface_input_files):
    args, _, scene, _ = surface_input_files
    scene["translation_ground_from_left_mm"][0, 0] = np.nan
    np.savez(args.scene_transforms, **scene)
    with pytest.raises(ValueError, match="translation_ground_from_left_mm"):
        load_surface_targets(args, 2, "cpu")


def test_valid_surface_inputs_preserve_units_and_vertex_order(surface_input_files):
    args, _, scene, _ = surface_input_files
    scene["translation_ground_from_left_mm"][:] = [1000.0, 2000.0, 3000.0]
    np.savez(args.scene_transforms, **scene)
    target = load_surface_targets(args, 2, "cpu")
    torch.testing.assert_close(target.translation, torch.tensor([[1.0, 2.0, 3.0]] * 2))
    assert target.sole_indices["left"].tolist() == [0, 1]
    assert target.hand_weight.shape == (2, 2)


@pytest.mark.parametrize("diagonal", [[-1.0, 1.0, 1.0], [2.0, 1.0, 1.0]])
def test_surface_transform_rejects_reflection_and_scale(surface_input_files, diagonal):
    args, _, scene, _ = surface_input_files
    scene["rotation_ground_from_left"][0] = np.diag(diagonal)
    np.savez(args.scene_transforms, **scene)
    with pytest.raises(ValueError, match="invalid proper rotation"):
        load_surface_targets(args, 2, "cpu")


@pytest.mark.parametrize("defect", ["broadcast", "index", "nonfinite"])
def test_global_surface_targets_require_exact_vertex_correspondence(
    surface_input_files, defect
):
    import json

    args, _, _, _ = surface_input_files
    prior = dict(
        assumption="hand_static_relative_to_walker_for_entire_video",
        status="engineering_candidate",
        geometry_source="camera_rigid_mount_assumption",
        hands={
            side: dict(
                vertex_indices=[0, 1],
                shared_surface_walker_m=[[0, 0, 0], [0, 0, 0]],
                palm_offset_handle_m=[0, 0, 0],
            )
            for side in ("left", "right")
        },
    )
    if defect == "broadcast":
        prior["hands"]["left"]["shared_surface_walker_m"] = [[0, 0, 0]]
    elif defect == "index":
        prior["hands"]["left"]["vertex_indices"] = [-1, 1]
    else:
        prior["hands"]["left"]["shared_surface_walker_m"][0][0] = float("inf")
    args.global_hand_handle_pose = args.contact_labels.parent / "prior.json"
    args.global_hand_handle_pose.write_text(json.dumps(prior), encoding="utf-8")
    with pytest.raises(ValueError, match="left global palm|left shared_surface"):
        load_surface_targets(args, 2, "cpu")


def test_finite_loss_with_infinite_gradient_is_rejected():
    from pose_app.smplh_fitting.stages import require_finite_tensors

    parameter = torch.nn.Parameter(torch.tensor(0.0))
    loss = torch.sqrt(parameter)
    assert torch.isfinite(loss)
    loss.backward()
    with pytest.raises(RuntimeError, match="gradient.*beta"):
        require_finite_tensors({"beta": parameter.grad, "inactive": None}, "gradient")
    with pytest.raises(RuntimeError, match="final output.*vertices"):
        require_finite_tensors(
            {"vertices": torch.tensor([float("nan")])}, "final output"
        )


def test_nonfinite_pixels_do_not_poison_zero_weight_residuals():
    from pose_app.smplh_hand_observation import finite_pixel_values
    from pose_app.smplh_fitting.losses import weighted_hand_residual

    raw = np.asarray([[[1.0, 2.0], [float("inf"), float("nan")]]], np.float32)
    prediction = torch.ones(1, 2, 2, requires_grad=True)
    mask = torch.tensor(np.isfinite(raw).all(-1))
    residual = (prediction - torch.tensor(finite_pixel_values(raw))).pow(2).sum(-1)
    loss = weighted_hand_residual(residual, mask.float(), mask)
    assert loss.item() == 1
    gradient = torch.autograd.grad(loss, prediction)[0]
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient[:, 1]) == 0
    assert np.isinf(raw[0, 1, 0]) and np.isnan(raw[0, 1, 1])


def test_selected_hand_confidence_cannot_poison_weights(tmp_path):
    import json
    from pose_app.smplh_hand_observation import read_view

    body = np.zeros((1, 17, 3), np.float32)
    body[0, 9] = [10, 20, 0.9]
    row = dict(
        frame_index=0,
        image="pair_0000.jpg",
        records=[
            dict(
                side="left",
                pixel_frame="raw_fisheye",
                detector_confidence=float("nan"),
                keypoints_2d_raw_fisheye=[[10, 20]] * 21,
            )
        ],
    )
    path = tmp_path / "wilor.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(
        ValueError, match="nonfinite detector confidence.*frame 0 hand left"
    ):
        read_view(path, body, (1920, 1080), "left")
