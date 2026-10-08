"""R4 exact HMR2 forward_step, explicit existing male asset, random weights."""
import argparse
import ast
import importlib
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

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


def run(output, *, normalized_image=None):
    torch.set_num_threads(2)
    torch.manual_seed(43)
    install_download_guard()
    torch.load = forbid_weights
    smpl_parents()  # Process-local compatibility for existing licensed legacy pickle.
    root = ROOT / "HMR2"
    for part in ("", ".models", ".utils", ".models.heads", ".models.components", ".models.backbones"):
        namespace("hmr2" + part, root / ("hmr2" + part.replace(".", "/")))
    cfg = CfgNode(yaml.safe_load((root / "hmr2/configs_hydra/experiment/hmr_vit_transformer.yaml").read_text()))
    defaults = yaml.safe_load((root / "hmr2/configs_hydra/experiment/default.yaml").read_text())
    cfg.EXTRA = CfgNode(defaults["EXTRA"])
    cfg.SMPL = CfgNode(defaults["SMPL"])
    cache = Path.home() / ".cache/4DHumans/data"
    cfg.SMPL.MEAN_PARAMS = str(cache / "smpl_mean_params.npz")
    asset = Path("models/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl").resolve()
    wrapper = importlib.import_module("hmr2.models.smpl_wrapper").SMPL
    body = wrapper(str(asset), gender="male", joint_regressor_extra=str(cache / "SMPL_to_J19.pkl")).cuda().eval()
    independent = smplx.SMPLLayer(str(asset), gender="male").cuda().eval()
    with torch.device("cuda"):
        backbone = importlib.import_module("hmr2.models.backbones.vit").vit(cfg)
        head = importlib.import_module("hmr2.models.heads.smpl_head").SMPLTransformerDecoderHead(cfg)
    backbone.eval()  # Original override returns None.
    head.cuda()  # Mean arrays loaded from NumPy initially remain CPU buffers.
    head.eval()
    source = root / "hmr2/models/hmr2.py"
    tree = ast.parse(source.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HMR2")
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "forward_step")
    projection = importlib.import_module("hmr2.utils.geometry").perspective_projection
    symbols = {"torch": torch, "Dict": dict, "perspective_projection": projection}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), symbols)  # noqa: S102 -- exact archived author method
    harness = SimpleNamespace(cfg=cfg, backbone=backbone, smpl_head=head, smpl=body)
    versions = [p._version for m in (backbone, head, body) for p in m.parameters()]
    if normalized_image is None:
        image = torch.randn(1, 3, 256, 256, device="cuda") * .1
    else:
        if normalized_image.shape != (1, 3, 256, 256) or normalized_image.dtype != torch.float32:
            raise ValueError("explicit normalized RGB must be float32 1x3x256x256")
        if not torch.isfinite(normalized_image).all():
            raise ValueError("declared original crop must be finite")
        image = normalized_image.to("cuda")
    with torch.no_grad():
        result = symbols["forward_step"](harness, {"img": image})
        repeat = symbols["forward_step"](harness, {"img": image})
        assert all(torch.equal(result[k], repeat[k]) for k in ("pred_cam", "pred_vertices", "pred_keypoints_3d", "pred_keypoints_2d"))
        params = result["pred_smpl_params"]
        reference = independent(**params, pose2rot=False)
        vertices = result["pred_vertices"]
        assert vertices.shape == (1, 6890, 3) and result["pred_keypoints_3d"].shape == (1, 44, 3)
        assert body.faces.shape == (13776, 3) and np.array_equal(body.faces, independent.faces)
        errors = {"mesh_m": float((reference.vertices - vertices).abs().max())}
        base = reference.joints[:, body.joint_map].clone()
        if body.update_hips:
            right, left, pelvis = base[:, 9].clone(), base[:, 12].clone(), base[:, 8].clone()
            base[:, 9] = right + .25 * (right - left) + .5 * (pelvis - .5 * (right + left))
            base[:, 12] = left + .25 * (left - right) + .5 * (pelvis - .5 * (right + left))
        extra = np.einsum("jv,bvc->bjc", body.joint_regressor_extra.cpu().numpy(), vertices.cpu().numpy())
        expected = np.concatenate((base.cpu().numpy(), extra), axis=1)
        errors["joints_m"] = float(np.abs(expected - result["pred_keypoints_3d"].cpu().numpy()).max())
        cam = result["pred_cam"].cpu().numpy()
        camera = np.stack((cam[:, 1], cam[:, 2], 2 * 5000 / (256 * cam[:, 0] + 1e-9)), axis=-1)
        translated = result["pred_keypoints_3d"].cpu().numpy() + camera[:, None]
        projected = translated[..., :2] / translated[..., 2:] * (5000 / 256)
        errors["projection_normalized"] = float(np.abs(projected - result["pred_keypoints_2d"].cpu().numpy()).max())
        assert max(errors.values()) < 2e-5
        rotations = rotation_check(torch.cat((params["global_orient"], params["body_pose"]), dim=1))
        assert all(torch.isfinite(v).all() for k, v in result.items() if k != "pred_smpl_params")
        missing = image.clone()
        missing[0, 0, 0, 64] = torch.nan  # Inside original 32:-32 crop.
        bad = symbols["forward_step"](harness, {"img": missing})
        invalid = {k: int((~torch.isfinite(v)).sum()) for k, v in bad.items() if k != "pred_smpl_params"}
        scale_probes = []
        for scale in (0., -1.):
            depth = 10000 / (256 * scale + 1e-9)
            translated_probe = torch.tensor([[0., 0., depth]], device="cuda")
            probe = projection(result["pred_keypoints_3d"], translation=translated_probe,
                               focal_length=result["focal_length"] / 256)
            scale_probes.append({"scale": scale, "depth": depth, "finite": bool(torch.isfinite(probe).all())})
    assert versions == [p._version for m in (backbone, head, body) for p in m.parameters()]
    arrays = {k: v.cpu().numpy() for k, v in result.items() if k != "pred_smpl_params"}
    arrays.update({"smpl_" + k: v.cpu().numpy() for k, v in params.items()})
    arrays["faces"] = body.faces
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    return {"status": "R4_original_forward_method_male_passed", "training_executed": False,
            "checkpoint_loaded": False, "source_method": str(source), "source_lines": [method.lineno, method.end_lineno],
            "input": "Artificial normalized RGB 1x3x256x256, seed43" if normalized_image is None else "Explicit original ViTDetDataset normalized RGB; external R16 provenance",
            "asset": str(asset), "gender": "male",
            "original_default_neutral_executed": False, "top_level_initializer_executed": False,
            "mean": cfg.SMPL.MEAN_PARAMS, "regressor": str(cache / "SMPL_to_J19.pkl"),
            "parameter_count": sum(p.numel() for m in (backbone, head) for p in m.parameters()),
            "output_shapes": {k: list(v.shape) for k, v in arrays.items()}, "independent_errors": errors,
            "joint_semantics": "25 OpenPose plus19 regressed joints; not COCO17 or SMPL-H52", "update_hips": body.update_hips,
            "rotation_check": rotations, "repeat_exact": True, "parameter_versions_unchanged": True,
            "NaN_input_nonfinite_output_counts": invalid,
            "zero_negative_scale_projection_probes": scale_probes,
            "camera_scale": float(cam[0, 0]), "camera_depth": float(camera[0, 2]),
            "boundary": "Exact original forward_step method via namespace/AST harness and alternate male SMPLLayer. No original initializer, detector/crop, neutral default, training, physical calibration or accuracy evaluation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError("retain prior evidence; choose new version")
    try:
        report = run(args.output)
    except Exception as error:  # noqa: BLE001 -- retain original path failures
        report = {"status": "R4_original_forward_method_failed", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if report["status"].endswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
