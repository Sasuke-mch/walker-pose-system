"""R5 exact original ProHMR initialize/forward_step, no trained weights."""
import argparse
import ast
import importlib
import json
import sys
import traceback
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import smplx
import torch
import yaml
from yacs.config import CfgNode

from .run_original import ROOT, namespace
from .run_probability import (
    forbid_weights,
    install_download_guard,
    rotation_check,
    smpl_parents,
)


def run(output):
    torch.set_num_threads(2)
    torch.manual_seed(47)
    install_download_guard()
    torch.load = forbid_weights
    smpl_parents()
    root = ROOT / "ProHMR"
    sys.path.insert(0, str((ROOT / "_deps").resolve()))
    sys.path.insert(0, str((ROOT / "ProHMR_nflows").resolve()))
    for part in ("", ".models", ".utils", ".models.heads", ".models.backbones"):
        namespace("prohmr" + part, root / ("prohmr" + part.replace(".", "/")))
    cfg = CfgNode(yaml.safe_load((root / "prohmr/configs/prohmr.yaml").read_text()))
    cfg.SMPL.MEAN_PARAMS = str((root / "data/smpl_mean_params.npz").resolve())
    backbone = importlib.import_module("prohmr.models.backbones.resnet").resnet(cfg, pretrained=False).eval()
    flow = importlib.import_module("prohmr.models.heads.smpl_flow").SMPLFlow(cfg).eval()
    wrapper = importlib.import_module("prohmr.models.smpl_wrapper").SMPL
    asset = Path("models/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl").resolve()
    body = wrapper(str(asset), gender="male", joint_regressor_extra=str((root / "data/SMPL_to_J19.pkl").resolve())).eval()
    independent = smplx.SMPLLayer(str(asset), gender="male").eval()
    source = root / "prohmr/models/prohmr.py"
    tree = ast.parse(source.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ProHMR")
    methods = [next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)
               for name in ("initialize", "forward_step")]
    geometry = importlib.import_module("prohmr.utils.geometry")
    symbols = {"torch": torch, "Dict": dict, "aa_to_rotmat": geometry.aa_to_rotmat,
               "perspective_projection": geometry.perspective_projection}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), symbols)  # noqa: S102 -- exact archived methods
    harness = SimpleNamespace(cfg=cfg, backbone=backbone, flow=flow, smpl=body, initialized=torch.tensor(False))
    harness.initialize = MethodType(symbols["initialize"], harness)
    forward = MethodType(symbols["forward_step"], harness)
    states = {name: {k: v.clone() for k, v in module.state_dict().items()}
              for name, module in (("backbone", backbone), ("flow", flow), ("body", body))}
    actnorm = [m for m in flow.modules() if type(m).__name__ == "ActNorm"]
    image = torch.randn(1, 3, 224, 224) * .1
    batch = {"img": image, "smpl_params": {"body_pose": torch.zeros(1, 69),
              "global_orient": torch.zeros(1, 3), "betas": torch.zeros(1, 10)},
             "has_smpl_params": {"body_pose": torch.ones(1)}}
    with torch.no_grad():
        try:
            forward({"img": image})
        except KeyError as error:
            cold_error = repr(error)
            assert error.args == ("smpl_params",)
        else:
            raise AssertionError("Expected original fresh image-only initialization dependency")
        assert not harness.initialized.item()
        torch.manual_seed(59)
        result = forward(batch)
        assert harness.initialized.item()
        torch.manual_seed(59)
        repeat = forward({"img": image})  # Original top flag now bypasses label lookup.
        for key in ("pred_vertices", "pred_keypoints_3d", "pred_keypoints_2d", "pred_pose_6d", "log_prob"):
            assert torch.equal(result[key], repeat[key])
        assert result["pred_vertices"].shape == (1, 64, 6890, 3)
        assert result["pred_keypoints_3d"].shape == (1, 64, 44, 3)
        params = {k: v.reshape(64, *v.shape[2:]) for k, v in result["pred_smpl_params"].items()}
        reference = independent(**params, pose2rot=False)
        vertices = result["pred_vertices"][0]
        errors = {"mesh_m": float((vertices - reference.vertices).abs().max())}
        extra = np.einsum("jv,bvc->bjc", body.joint_regressor_extra.numpy(), vertices.numpy())
        expected = np.concatenate((reference.joints[:, body.joint_map].numpy(), extra), axis=1)
        errors["joints_m"] = float(np.abs(expected - result["pred_keypoints_3d"][0].numpy()).max())
        cam = result["pred_cam"][0].numpy()
        camera = np.stack((cam[:, 1], cam[:, 2], 10000 / (224 * cam[:, 0] + 1e-9)), -1)
        translated = result["pred_keypoints_3d"][0].numpy() + camera[:, None]
        projected = translated[..., :2] / translated[..., 2:] * (5000 / 224)
        errors["projection_normalized"] = float(np.abs(projected - result["pred_keypoints_2d"][0].numpy()).max())
        assert max(errors.values()) < 2e-5 and np.array_equal(body.faces, independent.faces)
        zero = flow(result["conditioning_feats"], z=torch.zeros(1, 1, 144))
        assert torch.equal(zero[4][:, 0], result["pred_pose_6d"][:, 0])
        beta_spread = float((params["betas"] - params["betas"][:1]).abs().max())
        assert beta_spread == 0
        # Repeat actual author initialization with only labels changed, no fake True flag.
        harness.initialized.fill_(False)
        changed = {**batch, "smpl_params": {k: v + .2 for k, v in batch["smpl_params"].items()}}
        torch.manual_seed(59)
        label_changed = forward(changed)
        label_error = float((label_changed["pred_vertices"] - result["pred_vertices"]).abs().max())
        assert label_error == 0
        missing = image.clone()
        missing[0, 0, 0, 64] = torch.nan
        bad = forward({"img": missing})
        invalid = {k: int((~torch.isfinite(v)).sum()) for k, v in bad.items() if isinstance(v, torch.Tensor)}
    for name, module in (("backbone", backbone), ("flow", flow), ("body", body)):
        assert all(torch.equal(v, states[name][k]) for k, v in module.state_dict().items())
    assert all(not m.initialized.item() for m in actnorm)
    rotations = rotation_check(torch.cat((params["global_orient"], params["body_pose"]), dim=1))
    for key in ("pred_vertices", "pred_keypoints_3d", "pred_keypoints_2d", "log_prob"):
        assert torch.isfinite(result[key]).all()
    arrays = {k: v.numpy() for k, v in result.items() if isinstance(v, torch.Tensor)}
    arrays.update({"smpl_" + k: v.numpy() for k, v in params.items()})
    arrays["faces"] = body.faces
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    return {"status": "R5_original_methods_male_passed", "training_executed": False, "checkpoint_loaded": False,
            "source": str(source), "source_method_lines": {m.name: [m.lineno, m.end_lineno] for m in methods},
            "asset": str(asset), "gender": "male", "mean": cfg.SMPL.MEAN_PARAMS,
            "default_neutral_executed": False, "top_level_initializer_executed": False,
            "parameters": sum(p.numel() for m in (backbone, flow) for p in m.parameters()),
            "input": "Artificial RGB seed47 and label fixture; sample seed59; CPU float32",
            "image_only_coldstart_error": cold_error, "top_flag_after_author_initialize": harness.initialized.item(),
            "internal_actnorm_count": len(actnorm), "internal_actnorm_initialized": [m.initialized.item() for m in actnorm],
            "all_network_body_state_dict_unchanged": True, "changed_labels_same_seed_mesh_error_m": label_error,
            "repeat_exact": True, "mode_first_zero_latent_exact": True, "candidate_beta_spread": beta_spread,
            "independent_errors": errors, "rotation_check": rotations,
            "NaN_input_nonfinite_counts": invalid, "output_shapes": {k: list(v.shape) for k, v in arrays.items()},
            "joint_semantics": "25OpenPose plus19 regressions; not COCO17/SMPL-H52",
            "boundary": "Original initialize and forward_step AST methods executed with alternate male body. Eval-mode topflag changes but actual ActNorm remains uninitialized identity parameters; artificial labels test dependency only, no learned or data-calibrated density/accuracy. Original Lightning initializer/renderer/neutral path not run."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError("retain previous evidence; choose new version")
    try:
        report = run(args.output)
    except Exception as error:  # noqa: BLE001 -- retain author failures
        report = {"status": "R5_original_methods_failed", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if report["status"].endswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
