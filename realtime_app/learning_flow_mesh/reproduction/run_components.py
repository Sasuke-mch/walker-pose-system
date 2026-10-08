"""Original video components and the original algebraic multiview network.

Component checks preserve explicit coverage gaps. No fake SMPL layer is used to
claim a complete mesh pipeline. External author source stays outside Git.
"""
import argparse
import ast
import importlib
import json
import platform
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from yacs.config import CfgNode

from .run_original import ROOT, namespace


def vibe():
    root = ROOT / "VIBE"
    source = root / "lib/models/vibe.py"
    config_path = root / "configs/vibe.yaml"
    if not config_path.is_file():
        # File names are part of the snapshot, never replace with guessed defaults.
        config_path = root / "configs/config.yaml"
    config = yaml.safe_load(config_path.read_text())
    c = config["MODEL"]["TGRU"]
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TemporalEncoder")
    scope = {"nn": torch.nn, "F": F, "__name__": "original_vibe_temporal_component"}
    # Execute the exact original class; bypass unrelated asset-dependent imports.
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), scope)  # noqa: S102 -- exact reviewed original class
    options = {"n_layers": c["NUM_LAYERS"], "hidden_size": c["HIDDEN_SIZE"],
               "add_linear": c["ADD_LINEAR"], "bidirectional": c["BIDIRECTIONAL"], "use_residual": c["RESIDUAL"]}
    model = scope["TemporalEncoder"](**options).eval()
    x = torch.randn(1, config["DATASET"]["SEQLEN"], 2048)
    with torch.inference_mode():
        y = model(x)
        assert y.shape == x.shape and torch.isfinite(y).all()
        altered = x.clone()
        split = x.shape[1] // 2
        altered[:, split:] += 1
        difference = float((model(altered)[:, :split] - y[:, :split]).abs().max())
        bidirectional = scope["TemporalEncoder"](**(options | {"bidirectional": True})).eval()
        by = bidirectional(x)
        bidirectional_difference = float((bidirectional(altered)[:, :split] - by[:, :split]).abs().max())
    return {"status": "original_component_forward_passed", "source": source.as_posix(), "config": config_path.as_posix(),
            "class_source_lines": [cls.lineno, cls.end_lineno], "constructor": options,
            "input_shape": list(x.shape), "output_shape": list(y.shape),
            "official_config_past_change_when_future_changed": difference,
            "paper_bidirectional_variant_past_change": bidirectional_difference,
            "coverage": "Original TemporalEncoder only, exact AST class, not complete VIBE/SMPL; bidirectional diagnostic explicitly differs from released configuration",
            "unresolved": ["Original SMPL neutral/extra joint regressors and mean-parameter resources needed for complete mesh path", "Paper describes clip-averaged beta, released VIBE.forward flattens temporal features and its Regressor predicts beta per frame"]}


def wham():
    root = ROOT / "WHAM"
    # Skip pretrained builder facade; original modules and their constants/utilities run unchanged.
    namespace("lib", root / "lib")
    namespace("lib.models", root / "lib/models")
    sys.path.insert(0, str(root.resolve()))
    module = importlib.import_module("lib.models.wham")
    config_path = root / "configs/yamls/model_base.yaml"
    config = yaml.safe_load(config_path.read_text())
    model = module.Network(smpl=None, **config).eval()
    b, t, j = 1, 8, 17
    x = torch.randn(b, t, j * 2 + 3) * 0.1
    init_kp = torch.cat((torch.randn(b, 1, j * 3) * 0.1, x[:, :1]), dim=-1)
    init_pose = torch.tensor([1., 0, 0, 0, 1., 0]).expand(b, 1, 24, 6).clone()
    with torch.inference_mode():
        kp, context = model.motion_encoder(x, init_kp)
        pose, shape, camera, contact = model.motion_decoder(context, init_pose)
        trajectory_root, trajectory_velocity = model.trajectory_decoder(context, init_pose[:, :, 0], torch.zeros(b, t, 6))
        assert kp.shape == (b, t, j, 3) and shape.shape == (b, t, 10)
        for output in (kp, context, pose, shape, camera, contact, trajectory_root, trajectory_velocity):
            assert torch.isfinite(output).all()
        altered = x.clone()
        altered[:, t // 2:] += 0.5
        changed_kp, changed_context = model.motion_encoder(altered, init_kp)
        changed_pose, changed_shape, _, _ = model.motion_decoder(changed_context, init_pose)
        past_change = float((changed_pose[:, :t // 2] - pose[:, :t // 2]).abs().max())
        kp_change = float((changed_kp[:, :t // 2] - kp[:, :t // 2]).abs().max())
        shape_change = float((changed_shape[:, :t // 2] - shape[:, :t // 2]).abs().max())
        masked_x = x.clone()
        masked_x[0, 0, :2] = torch.nan
        mask = torch.zeros(b, t, j, dtype=torch.bool)
        mask[0, 0, 0] = True  # WHAM mask True means missing, opposite project presence mask.
        prepared = model.preprocess(masked_x, mask)
        assert torch.isfinite(prepared).all()
    return {"status": "original_components_forward_passed", "source": (root / "lib/models/wham.py").as_posix(),
            "config": config_path.as_posix(), "constructor": config, "input_shape": list(x.shape),
            "pose_shape": list(pose.shape), "beta_shape": list(shape.shape), "root_shape": list(trajectory_root.shape),
            "velocity_shape": list(trajectory_velocity.shape), "past_pose_change": past_change,
            "past_keypoint_change": kp_change, "past_shape_change": shape_change,
            "same_sequence_beta_max_change": float((shape - shape[:, :1]).abs().max()),
            "masked_nan_cleaned_by_original_preprocess": True,
            "coverage": "Original full Network constructor without body asset, plus original motion encoder, motion decoder, trajectory decoder and missing-point preprocessing. No SMPL mesh/global-feet/refiner execution",
            "unresolved": ["Original neutral SMPL and WHAM/H36M/feet regressors required before full Network.forward", "Released decoder predicts per-frame beta; streaming hidden/previous predictions require an explicit adapter"]}


def triangulation():
    root = ROOT / "LearnableTriangulation"
    sys.path.insert(0, str(root.resolve()))
    module = importlib.import_module("mvn.models.triangulation")
    geometry = importlib.import_module("mvn.utils.multiview")
    config_path = root / "experiments/human36m/train/human36m_alg.yaml"
    config = CfgNode(yaml.safe_load(config_path.read_text()))
    # Initialization-only switch: architecture and published resolution remain unchanged.
    config.model.backbone.init_weights = False
    model = module.AlgebraicTriangulationNet(config, device="cpu").eval()
    images = torch.randn(1, 2, 3, *config.image_shape) * 0.1
    k = torch.tensor([[384., 0, 192.], [0, 384., 192.], [0, 0, 1.]])
    extrinsics = torch.eye(3, 4).repeat(2, 1, 1)
    extrinsics[1, 0, 3] = -0.3  # metres; camera1 at +0.3m along X in cam0 frame.
    projection = k @ extrinsics
    truth = torch.tensor([[0.1, 0.2, 3.], [-0.2, 0.1, 2.], [0.1, -0.1, 4.]])
    pixels = torch.stack([geometry.project_3d_points_to_image_plane_without_distortion(p, truth) for p in projection])
    with torch.inference_mode():
        recovered = geometry.triangulate_batch_of_points(projection[None], pixels[None])[0]
        maximum_error = float((recovered - truth).abs().max())
        assert maximum_error < 1e-5
        xyz, xy, heatmaps, confidence = model(images, projection[None], {})
        assert xyz.shape == (1, 17, 3) and xy.shape == (1, 2, 17, 2)
        assert torch.isfinite(xyz).all() and torch.isfinite(confidence).all()
    return {"status": "original_algebraic_network_forward_passed", "source": (root / "mvn/models/triangulation.py").as_posix(),
            "config": config_path.as_posix(), "input_shape": list(images.shape), "xyz_shape": list(xyz.shape),
            "heatmap_shape": list(heatmaps.shape), "confidence_shape": list(confidence.shape),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "exact_geometry_control_max_error_m": maximum_error,
            "adaptations": ["Only backbone.init_weights=True changed to False; ResNet152, 384x384 input and confidence head unchanged"],
            "coverage": "Original algebraic network and original pinhole DLT; volumetric/RANSAC variants not run",
            "unresolved": ["Raw fish-eye pixels cannot be used with pinhole projection matrices; project ray adapter and strict gates must remain separate", "No native point-rejection/missing-state contract; cannot replace strict triangulation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", choices=("P09", "P10", "P16"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(19)
    torch.set_num_threads(2)
    try:
        with torch.inference_mode():
            result = {"P09": vibe, "P10": wham, "P16": triangulation}[args.paper]()
    except Exception as error:  # noqa: BLE001 -- persist external-model failure and exit nonzero
        result = {"status": "original_check_failed", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
    result.update(id=args.paper, training_executed=False, weights="random_initialization",
                  environment={"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
                  evidence_boundary="Untrained structure/interface checks only; no accuracy or deployment-speed conclusion")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["status"].endswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
