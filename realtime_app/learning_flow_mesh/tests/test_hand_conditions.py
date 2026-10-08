"""H4 contracts: missing native hands, anatomical identity and causal inputs."""
import copy
import json

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from smplx.lbs import batch_rodrigues

from realtime_app.learning_flow_mesh.hand_conditions import (
    HAND_NODES,
    SIDE_ORDER,
    HandConvention,
    native_hand_tokens,
)
from realtime_app.learning_flow_mesh.interfaces import SixDLayout, decode_rotation
from realtime_app.learning_flow_mesh.reproduction.check_hand_conditions import (
    original_shared_initialization,
)
from realtime_app.pose_app.wilor_mano_prior import (
    CONVENTION,
    MIRROR,
    convert_record,
    read_parameter_view,
)


def parents():
    result = torch.zeros(52, dtype=torch.long)
    result[0] = -1
    relative = torch.tensor([0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14])
    for start, wrist in ((22, 20), (37, 21)):
        result[start:start + 15] = torch.where(relative == 0, wrist, relative + start - 1)
    return result


def arguments():
    ids = torch.arange(3, dtype=torch.long)[None]
    return {"rotations": torch.eye(3).expand(1, 3, 2, 2, 15, 3, 3).clone(),
            "box_confidence": torch.full((1, 3, 2, 2), .7),
            "source_frame_ids": ids, "target_frame_ids": ids.clone(),
            "view_available": torch.ones(1, 3, 2, dtype=torch.bool),
            "convention": HandConvention.SMPLH_PARENT_LOCAL_NATIVE,
            "side_order": SIDE_ORDER, "parents": parents()}


def native_record(side="left"):
    rotations = Rotation.from_rotvec(np.tile([.2, -.3, .1], (15, 1))).as_matrix()
    return {"side": side, "mano_parameter_convention": CONVENTION,
            "mano_parameter_metadata": {"joint_order": ["index", "middle", "pinky", "ring", "thumb"],
                                        "rotation_type": "parent_relative_rotmat",
                                        "pose_mean_added_by_layer": False,
                                        "left_crop_mirrored": side == "left"},
            "mano_parameters": {"hand_pose": rotations.tolist(),
                                "global_orient": np.eye(3)[None].tolist(), "betas": [0.] * 10},
            "pixel_frame": "raw_fisheye", "detector_bbox_xyxy_raw": [90, 70, 110, 90],
            "detector_confidence": .7}


def test_zero_weight_identity_and_unassociated_views_remain_missing():
    args = arguments()
    args["box_confidence"][0, 0, 0, 0] = 0
    args["view_available"][0, 1, 1] = False
    value = native_hand_tokens(**args)
    assert not value.source_present[0, 0, 0, 0]
    assert value.source_present[0, 1, 1].all()
    assert not value.admitted[0, 1, 1].any()
    assert torch.isnan(value.tokens[~value.present]).all()
    assert torch.isnan(value.local_rotations[~value.present]).all()
    assert value.present[0, 0, 1].all()  # independent other camera stays present
    assert torch.equal(value.node_ids, torch.tensor(HAND_NODES))


def test_both_views_and_native_rotation_layout_stay_distinct():
    args = arguments()
    second = torch.tensor(Rotation.from_rotvec([.2, -.1, .3]).as_matrix(), dtype=torch.float32)
    args["rotations"][:, :, 1] = second
    value = native_hand_tokens(**args)
    restored, valid = decode_rotation(value.tokens[..., :6], SixDLayout.PACKED_COLUMNS)
    assert torch.equal(valid, value.present)
    torch.testing.assert_close(restored, args["rotations"], atol=1e-6, rtol=0)
    assert not torch.equal(value.tokens[:, :, 0], value.tokens[:, :, 1])
    torch.testing.assert_close(value.tokens[..., 6], torch.full((1, 3, 2, 2, 15), .7))


@pytest.mark.parametrize("fault", ["side", "convention", "frame_mismatch", "frame_duplicate",
                                  "float_ids", "numeric_mask", "confidence", "reflection", "tree"])
def test_invalid_identity_or_source_rejected(fault):
    args = arguments()
    if fault == "side":
        args["side_order"] = ("right", "left")
    elif fault == "convention":
        args["convention"] = "native_MANO_anatomical_parent_local_after_existing_mirror"
    elif fault == "frame_mismatch":
        args["target_frame_ids"] += 1
    elif fault == "frame_duplicate":
        args["source_frame_ids"][0, 1] = 0
    elif fault == "float_ids":
        args["source_frame_ids"] = args["source_frame_ids"].float()
    elif fault == "numeric_mask":
        args["view_available"] = args["view_available"].float()
    elif fault == "confidence":
        args["box_confidence"][0, 0, 0, 0] = torch.nan
    elif fault == "reflection":
        args["rotations"][0, 0, 0, 0, 0, 0, 0] = -1
        args["view_available"][0, 0, 0] = False  # still audit source selections
    else:
        args["parents"][25] = 22
    with pytest.raises(ValueError):
        native_hand_tokens(**args)


def test_source_native_left_mirror_and_excluded_fields():
    left = native_record()
    original = np.array(left["mano_parameters"]["hand_pose"])
    np.testing.assert_array_equal(convert_record(left), MIRROR @ original @ MIRROR)
    np.testing.assert_array_equal(convert_record(native_record("right")), original)
    changed = copy.deepcopy(left)
    changed["mano_parameters"]["global_orient"] = Rotation.from_rotvec([[.7, .2, -.1]]).as_matrix().tolist()
    changed["mano_parameters"]["betas"] = [2.] * 10
    changed["pred_cam"] = [900, -12, 5]
    changed["keypoints_2d"] = [[999, 888]] * 21
    np.testing.assert_array_equal(convert_record(changed), convert_record(left))
    changed["mano_parameter_metadata"]["joint_order"] = ["thumb", "index", "middle", "ring", "pinky"]
    with pytest.raises(ValueError, match="metadata"):
        convert_record(changed)


def test_original_reader_future_changes_do_not_modify_prefix(tmp_path):
    rows = [{"frame_index": i, "image": f"pair_{i:04d}.jpg", "records": [native_record()]} for i in range(3)]
    body = np.tile(np.array([100, 80, .8], dtype=np.float32), (3, 17, 1))
    path = tmp_path / "original.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    r, w, audit = read_parameter_view(path, body, (1920, 1080))
    rows[2]["records"][0]["mano_parameters"]["hand_pose"] = Rotation.from_rotvec(
        np.tile([-.5, .1, .4], (15, 1))).as_matrix().tolist()
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    r2, w2, audit2 = read_parameter_view(path, body, (1920, 1080))
    np.testing.assert_array_equal(r[:2], r2[:2])
    np.testing.assert_array_equal(w, w2)
    assert audit == audit2
    assert not np.array_equal(r[2, 0], r2[2, 0])
    assert np.all(w[:, 0] == .7) and np.all(w[:, 1] == 0)


def test_original_shared_initialization_is_explicitly_offline():
    coeff = np.zeros((3, 2, 12))  # original encode_pca already selects one side
    weights = np.ones((3, 2, 2))
    initial, location = original_shared_initialization(coeff, weights, 0)
    coeff[2] = 3
    changed, _ = original_shared_initialization(coeff, weights, 0)
    np.testing.assert_array_equal(initial, np.zeros((1, 12)))
    np.testing.assert_array_equal(changed, np.ones((1, 12)))
    assert location["path"].endswith("smplh_fitting/pipeline.py")


def test_native_condition_gradient_and_future_prefix_without_optimizer():
    args = arguments()
    theta = torch.tensor([[.2, -.3, .1]], requires_grad=True)
    score = torch.tensor(.7, requires_grad=True)
    args["rotations"] = batch_rodrigues(theta).expand(1, 3, 2, 2, 15, 3, 3)
    args["box_confidence"] = score.expand(1, 3, 2, 2)
    value = native_hand_tokens(**args)
    value.tokens[..., 0].sum().add(value.tokens[..., 6].sum()).backward()
    assert theta.grad is not None and torch.isfinite(theta.grad).all() and theta.grad.abs().sum() > 0
    assert score.grad is not None and score.grad > 0
    changed = dict(args, rotations=args["rotations"].detach().clone())
    changed["rotations"][:, 2] = torch.eye(3)
    alternative = native_hand_tokens(**changed)
    torch.testing.assert_close(value.tokens[:, :2], alternative.tokens[:, :2], rtol=0, atol=0)
    assert not torch.equal(value.tokens[:, 2], alternative.tokens[:, 2])
