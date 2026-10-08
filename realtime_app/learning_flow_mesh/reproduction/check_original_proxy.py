"""R3 original HRNet/crop/Canny/proxy AST path; no checkpoint or training."""
import argparse
import ast
import importlib
import json
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

from .audit_mesh_resources import mesh_resource_audit
from .check_interfaces import original_function
from .run_original import ROOT
from .run_probability import (
    forbid_weights,
    install_download_guard,
    rotation_check,
    smpl_parents,
)

SOURCE = ROOT / "HuManiFlow/predict/predict_humaniflow.py"
INPUT = Path("research_records/engineering_validation/G20261003_repositioned360_cold_smplh_v2/input/left_ccw90/pair_0000.png")


def assigned_name(node):
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return None
    target = node.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def original_proxy_statements(*, include_prediction=True, visibility_only=False):
    """Select exact contiguous source AST statements; skip unrelated renderer setup."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "predict_humaniflow")
    loop = next(node for node in function.body if isinstance(node, ast.For))
    block = next(node for node in loop.body if isinstance(node, ast.With)).body
    start = next(index for index, node in enumerate(block) if assigned_name(node) == "hrnet_output")
    end = next(index for index, node in enumerate(block) if assigned_name(node) == "pred") + 1
    if visibility_only:
        start = next(index for index, node in enumerate(block) if assigned_name(node) == "hrnet_joints2Dvisib")
        end = start + 2
    elif not include_prediction:
        end -= 1
    nodes = block[start:end]
    return compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), [nodes[0].lineno, nodes[-1].end_lineno], len(nodes)


def forced_visibility(confidences, threshold=.75):
    code, _lines, _count = original_proxy_statements(visibility_only=True)
    space = {"hrnet_output": {"joints2Dconfs": confidences}, "joints2Dvisib_threshold": threshold}
    exec(code, space)  # noqa: S102 -- exact inspected original source statements
    return space["hrnet_joints2Dvisib"]


def input_semantics_probes():
    labels = ROOT / "HuManiFlow/utils/label_conversions.py"
    gaussian = original_function(labels, "convert_2Djoints_to_gaussian_heatmaps_torch")
    argmax = original_function(ROOT / "HuManiFlow/predict/predict_hrnet.py", "get_kp_locations_confs_from_heatmaps")
    crop = original_function(ROOT / "HuManiFlow/utils/image_utils.py", "batch_crop_pytorch_affine")
    joints = torch.tensor([[[31., 83.], [113., 47.]]])
    heatmaps = gaussian(joints, img_wh=128, std=4)
    yy, xx = torch.meshgrid(torch.arange(128), torch.arange(128), indexing="ij")
    expected = torch.exp(-((xx - joints[..., 0, None, None]) ** 2 +
                           (yy - joints[..., 1, None, None]) ** 2) / 32)
    gaussian_error = float((heatmaps - expected).abs().max())
    decoded, _confidence = argmax(heatmaps)
    assert torch.equal(decoded, joints) and gaussian_error < 1e-6
    height, width = torch.tensor([100.]), torch.tensor([200.])
    cropped = crop(input_wh=(200, 100), output_wh=(256, 256), num_to_crop=1, device="cpu",
                   joints2D=torch.tensor([[[30., 70.], [140., 20.]]]),
                   bbox_centres=torch.tensor([[50., 100.]]), bbox_heights=height, bbox_widths=width,
                   orig_scale_factor=1.)
    affine_expected = torch.tensor([[[38.4, 153.6], [179.2, 89.6]]])
    affine_error = float((cropped["joints2D"] - affine_expected).abs().max())
    assert affine_error < 2e-5 and height.item() == 200  # Author mutates aspect-ratio inputs.
    zeros = torch.zeros(1, 17, 96, 72)
    locations, scores = argmax(zeros)
    visible = forced_visibility(scores[0])
    empty_proxy = gaussian(locations * 4, img_wh=256, std=4) * visible[None, :, None, None]
    assert locations.eq(0).all() and scores.eq(0).all() and visible[:7].all() and not visible[7:].any()
    assert (empty_proxy.amax((-1, -2)) > 0).sum() == 7
    negative_locations, negative_scores = argmax(-torch.ones_like(zeros))
    assert negative_locations.eq(0).all() and forced_visibility(negative_scores[0])[:7].all()
    nan_joints = locations.clone()
    nan_joints[:, 8] = torch.nan
    masked_nan = gaussian(nan_joints, img_wh=256, std=4) * visible[None, :, None, None]
    assert torch.isnan(masked_nan[:, 8]).all()
    return {"gaussian_row_column_control_error": gaussian_error, "asymmetric_argmax_exact": True,
            "affine_pixel_control_error": affine_error, "crop_mutates_bbox_height_for_aspect_ratio": True,
            "zero_heatmap_argmax_xy": [0, 0], "zero_heatmap_confidence": 0,
            "zero_or_negative_heatmap_forced_visible_head_shoulder_channels": 7,
            "zero_heatmap_nonzero_gaussian_channels_after_original_mask": 7,
            "masked_nan_joint_heatmap_remains_nan": True,
            "boundary": "Original forced visibility and multiplication are not observation validity gates; no source fix or mainline adaptation"}


def run(output):
    torch.set_num_threads(2)
    sys.path.insert(0, str((ROOT / "_deps").resolve()))
    sys.path.insert(0, str((ROOT / "HuManiFlow").resolve()))
    install_download_guard()
    torch.load = forbid_weights
    parents = smpl_parents()
    cfg = importlib.import_module("configs.humaniflow_config").get_humaniflow_cfg_defaults()
    hr_cfg = importlib.import_module("configs.pose2D_hrnet_config").get_pose2D_hrnet_cfg_defaults()
    torch.manual_seed(41)
    hrnet = importlib.import_module("models.pose2D_hrnet").PoseHighResolutionNet(hr_cfg).eval()
    torch.manual_seed(31)
    model = importlib.import_module("models.humaniflow_model").HumaniflowModel("cpu", cfg.MODEL, parents).eval()
    canny = importlib.import_module("models.canny_edge_detector").CannyEdgeDetector(
        non_max_suppression=cfg.DATA.EDGE_NMS, gaussian_filter_std=cfg.DATA.EDGE_GAUSSIAN_STD,
        gaussian_filter_size=cfg.DATA.EDGE_GAUSSIAN_SIZE, threshold=cfg.DATA.EDGE_THRESHOLD).eval()
    states = [{key: value.clone() for key, value in module.state_dict().items()} for module in (hrnet, model, canny)]
    paths = importlib.import_module("configs.paths")
    for key, filename in (("J_REGRESSOR_EXTRA", "J_regressor_extra.npy"),
                          ("COCOPLUS_REGRESSOR", "cocoplus_regressor.npy"), ("H36M_REGRESSOR", "J_regressor_h36m.npy")):
        setattr(paths, key, str((ROOT / "HuManiFlow/model_files" / filename).resolve()))
    asset = Path("models/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")
    body = importlib.import_module("models.smpl").SMPL(str(asset.resolve()), gender="male", num_betas=10).eval()
    image_array = cv2.imread(str(INPUT))
    if image_array is None:
        raise FileNotFoundError(INPUT)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image_array.transpose(2, 0, 1).copy()).float() / 255
    captured = []
    hook = hrnet.register_forward_hook(lambda _module, _args, result: captured.append(result.detach().clone()))
    space = {"torch": torch, "image": image, "device": "cpu", "hrnet_model": hrnet, "hrnet_cfg": hr_cfg,
             "humaniflow_cfg": cfg, "humaniflow_model": model, "edge_detect_model": canny,
             "object_detect_model": None, "joints2Dvisib_threshold": .75, "num_pred_samples": 50,
             "predict_hrnet": importlib.import_module("predict.predict_hrnet").predict_hrnet,
             "batch_crop_pytorch_affine": importlib.import_module("utils.image_utils").batch_crop_pytorch_affine,
             "convert_2Djoints_to_gaussian_heatmaps_torch": importlib.import_module("utils.label_conversions").convert_2Djoints_to_gaussian_heatmaps_torch}
    code, source_lines, statements = original_proxy_statements()
    try:
        with torch.no_grad():
            exec(code, space)  # noqa: S102 -- exact original RGB/proxy/prediction subpath
    finally:
        hook.remove()
    assert len(captured) == 1 and captured[0].shape == (1, 17, 96, 72)
    pred, proxy = space["pred"], space["proxy_rep_input"]
    assert proxy.shape == (1, 18, 256, 256) and torch.isfinite(proxy).all()
    assert torch.equal(pred["shape_samples"], pred["shape_mode"][:, None].expand(-1, 50, -1))
    with torch.no_grad():
        point = body(body_pose=pred["pose_rotmats_point_est"], global_orient=pred["glob_rotmat"][:, None],
                     betas=pred["shape_mode"], pose2rot=False)
        roots = pred["glob_rotmat"][:, None].expand(50, -1, -1, -1)
        samples = body(body_pose=pred["pose_rotmats_samples"][0], global_orient=roots,
                       betas=pred["shape_samples"][0], pose2rot=False)
        assert torch.isfinite(samples.vertices).all() and samples.joints.shape == (50, 90, 3)
        repeated = dict(space)
        repeat_code, _lines, _count = original_proxy_statements(include_prediction=False)
        exec(repeat_code, repeated)  # noqa: S102 -- same original preprocessing only, no new samples
        assert torch.equal(proxy, repeated["proxy_rep_input"])
        repeat_pred = model(None, input_feats=pred["input_feats"], num_samples=0, compute_point_est=True)
        assert torch.equal(pred["pose_rotmats_point_est"], repeat_pred["pose_rotmats_point_est"])
        probes = input_semantics_probes()
    for module, before in zip((hrnet, model, canny), states, strict=True):
        assert all(torch.equal(before[key], value) for key, value in module.state_dict().items())
    audit = mesh_resource_audit()
    gaps = sum(not item["available"] for paper in audit["author_mesh_paths"] for item in paper["requirements"])
    assert gaps == 14
    report = {"status": "R3_original_RGB_proxy_male_mesh_subpath_passed", "protocol": "architecture_r3_protocol.json",
              "whole_goal_complete": False, "training_executed": False, "checkpoint_loaded": False, "network_requests": 0,
              "weights_unchanged": True, "input_path": INPUT.as_posix(), "frame_id": 0, "original_pair_id": 1157,
              "input_shape": list(image.shape), "camera": "cam0_LEFT", "orientation": "existingCCW90modelinput, not raw geometry pixels",
              "detector_executed": False, "crop_boundary": "Original no-detector branch on full saved view; no verified single-person centered crop or association",
              "exact_original_AST_source": SOURCE.as_posix(), "source_lines": source_lines, "source_statements": statements,
              "HRNet_random_seed": 41, "flow_random_seed": 31,
              "HRNet_parameters": sum(parameter.numel() for parameter in hrnet.parameters()),
              "HRNet_heatmap_shape": list(captured[0].shape), "HRNet_image_size_WH": list(hr_cfg.MODEL.IMAGE_SIZE),
              "proxy_shape": list(proxy.shape), "visibility_threshold": .75,
              "visible_joints_after_original_rule": int(space["hrnet_joints2Dvisib"].sum()),
              "candidate_count": 50, "released_image_path_shared_shape_mode": True, "R2_library_default_shape_mode": False,
              "sample_vertices_shape": list(samples.vertices.shape), "sample_joints_shape": list(samples.joints.shape),
              "faces_shape": list(body.faces.shape), "asset": asset.as_posix(), "asset_gender": "male",
              "default_neutral_reproduced": False, "proxy_repeat_exact": True, "point_repeat_exact": True,
              "rotation_check": rotation_check(torch.cat((roots, pred["pose_rotmats_samples"][0]), 1)),
              "input_semantics_probes": probes, "default_resources_missing_still": gaps,
              "not_executed": ["original top-level function including renderer", "defaultneutral/CLI/checkpoint", "object detector", "original dataset evaluation"],
              "boundary": "Actual savedRGB through original random HRNet/proxy/flow to alternatemale library mesh; no human input/pose/uncertainty validity or fisheye accuracy claim"}
    arrays = {"rgb_u8": image_array, "hrnet_heatmaps": captured[0].numpy(),
              "hrnet_joints2D": space["hrnet_output"]["joints2D"].numpy(),
              "hrnet_confidences": space["hrnet_output"]["joints2Dconfs"].numpy(),
              "cropped_rgb": space["cropped_for_proxy"]["rgb"].numpy(),
              "proxy_joints2D": space["cropped_for_proxy"]["joints2D"].numpy(),
              "visibility": space["hrnet_joints2Dvisib"].numpy(), "proxy": proxy.numpy(),
              "point_vertices": point.vertices.numpy(), "sample_vertices": samples.vertices.numpy(),
              "sample_joints": samples.joints.numpy(), "sample_body_rotations": pred["pose_rotmats_samples"][0].numpy(),
              "sample_roots": roots.numpy(), "sample_betas": pred["shape_samples"][0].numpy(),
              "faces": body.faces, "metadata": np.array(json.dumps(report, ensure_ascii=False))}
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError("retain previous input-path evidence; choose new version")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args.output)
    except Exception as error:  # noqa: BLE001 -- retain actual source/input failures
        result = {"status": "R3_original_RGB_proxy_failed", "error": f"{type(error).__name__}: {error}",
                  "traceback": traceback.format_exc(), "training_executed": False}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["status"].endswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
