"""R2 original HuManiFlow library mesh subpath with explicit existing male asset."""
import argparse
import ast
import importlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import smplx
import torch

from .audit_mesh_resources import mesh_resource_audit
from .run_original import ROOT
from .run_probability import (
    forbid_weights,
    install_download_guard,
    rotation_check,
    smpl_parents,
)

ASSET = Path("models/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")


def run(output):
    torch.set_num_threads(2)
    torch.manual_seed(31)
    sys.path.insert(0, str((ROOT / "_deps").resolve()))
    sys.path.insert(0, str((ROOT / "HuManiFlow").resolve()))
    install_download_guard()
    torch.load = forbid_weights
    # Existing licensed legacy model deserialization compatibility, process-local.
    tree = smpl_parents()
    cfg = importlib.import_module("configs.humaniflow_config").get_humaniflow_cfg_defaults()
    paths = importlib.import_module("configs.paths")
    for name, filename in (("J_REGRESSOR_EXTRA", "J_regressor_extra.npy"),
                           ("COCOPLUS_REGRESSOR", "cocoplus_regressor.npy"),
                           ("H36M_REGRESSOR", "J_regressor_h36m.npy")):
        setattr(paths, name, str((ROOT / "HuManiFlow/model_files" / filename).resolve()))
    wrapper = importlib.import_module("models.smpl").SMPL
    body = wrapper(str(ASSET.resolve()), batch_size=1, gender="male", num_betas=cfg.MODEL.NUM_SMPL_BETAS).eval()
    assert body.parents.tolist() == tree and len(tree) == 24
    network = importlib.import_module("models.humaniflow_model").HumaniflowModel("cpu", cfg.MODEL, tree).eval()
    before = {key: value.clone() for key, value in network.state_dict().items()}
    body_before = {key: value.clone() for key, value in body.state_dict().items()}
    independent = smplx.SMPL(str(ASSET.resolve()), batch_size=1, gender="male", num_betas=cfg.MODEL.NUM_SMPL_BETAS).eval()
    proxy = torch.randn(1, 18, cfg.DATA.PROXY_REP_SIZE, cfg.DATA.PROXY_REP_SIZE) * .1
    cli_path = ROOT / "HuManiFlow/scripts/run_predict.py"
    cli = ast.parse(cli_path.read_text(encoding="utf-8"))
    guarded = any(isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                  and ast.unparse(node.test) == "args.gender != 'neutral'"
                  and any(isinstance(child, ast.Raise) and isinstance(child.exc, ast.Name)
                          and child.exc.id == "NotImplementedError" for child in node.body)
                  for node in ast.walk(cli))
    assert guarded
    with torch.no_grad():
        pred = network(proxy, num_samples=50, compute_point_est=True, return_input_feats=True)
        point = body(body_pose=pred["pose_rotmats_point_est"], global_orient=pred["glob_rotmat"][:, None],
                     betas=pred["shape_mode"], pose2rot=False)
        # Same original predict_humaniflow.py sample-to-body arguments (budget50).
        poses = pred["pose_rotmats_samples"][0]
        roots = pred["glob_rotmat"][:, None].expand(50, -1, -1, -1)
        shapes = pred["shape_samples"][0]
        samples = body(body_pose=poses, global_orient=roots, betas=shapes, pose2rot=False)
        reference = independent(body_pose=poses, global_orient=roots, betas=shapes, pose2rot=False)
        assert point.vertices.shape == (1, 6890, 3) and samples.vertices.shape == (50, 6890, 3)
        assert point.joints.shape == (1, 90, 3) and samples.joints.shape == (50, 90, 3)
        assert body.faces.shape == (13776, 3)
        assert torch.isfinite(samples.vertices).all() and torch.isfinite(samples.joints).all()
        independent_error = float((samples.vertices - reference.vertices).abs().max())
        assert independent_error == 0
        joint_errors = {"base45": float((samples.joints[:, :45] - reference.joints).abs().max())}
        offset = 45
        for key, rows in (("J_regressor_extra", 9), ("J_regressor_cocoplus", 19), ("J_regressor_h36m", 17)):
            matrix = getattr(body, key).numpy()
            expected = np.einsum("jv,bvc->bjc", matrix, samples.vertices.numpy())
            error = float(np.abs(expected - samples.joints[:, offset:offset + rows].numpy()).max())
            assert error < 2e-5
            joint_errors[key] = error
            offset += rows
        repeated = body(body_pose=pred["pose_rotmats_point_est"], global_orient=pred["glob_rotmat"][:, None],
                        betas=pred["shape_mode"], pose2rot=False)
        assert torch.equal(repeated.vertices, point.vertices)
        assert all(torch.equal(before[key], value) for key, value in network.state_dict().items())
        assert all(torch.equal(body_before[key], value) for key, value in body.state_dict().items())
        rotations = torch.cat((roots, poses), dim=1)
        rotation_report = rotation_check(rotations)
    audit = mesh_resource_audit()
    gaps = sum(not item["available"] for paper in audit["author_mesh_paths"] for item in paper["requirements"])
    assert gaps == 14 and not any(paper["default_mesh_ready"] for paper in audit["author_mesh_paths"])
    inventories = {path: [{"name": item.name, "bytes": item.stat().st_size}
                          for item in Path(path).iterdir() if item.is_file()]
                   for path in ("third_party/WiLoR/mano_data/models", "models/smplx", "models/smpl")}
    report = {"status": "R2_original_library_male_mesh_subpath_passed", "protocol": "architecture_r2_protocol.json",
              "whole_goal_complete": False, "training_executed": False, "checkpoint_loaded": False,
              "network_requests": 0, "parameters_unchanged": True, "random_model_seed": 31,
              "asset": ASSET.as_posix(), "asset_gender": "male", "default_neutral_reproduced": False,
              "author_CLI_non_neutral_NotImplementedError_verified": guarded,
              "input_semantics": "Artificial random18channel256proxy; no detector/edge-generated input",
              "sample_budget": 50, "parameter_count": sum(p.numel() for p in network.parameters()),
              "point_vertices_shape": list(point.vertices.shape), "samples_vertices_shape": list(samples.vertices.shape),
              "samples_joints_shape": list(samples.joints.shape), "faces_shape": list(body.faces.shape), "parents": tree,
              "joint_semantics": "45 original smplx landmarks plus9 extra,19COCOplus,17H36M; not COCO17 or SMPL-H52",
              "same_asset_independent_smplx_mesh_max_error_m": independent_error, "joint_block_einsum_errors_m": joint_errors,
              "point_repeat_exact": True, "rotation_check": rotation_report,
              "default_missing_still": gaps, "local_asset_directory_inventory": inventories,
              "adaptations": ["Exact original regressor paths resolved in process-local configs.paths",
                              "Existing male file explicitly passed to original library wrapper; no asset copies/renaming",
                              "Legacy NumPy/inspect aliases only in this subprocess for licensed pickle"],
              "coverage": "Original fresh ResNet18/hierarchical flow point and50sample parameters to original SMPL wrapper mesh; alternate male library subpath only",
              "not_executed": ["default neutral body path", "author CLI", "image crop/detector/HRNet/edge proxy", "renderer/world visualization", "checkpoint/performance evaluation"],
              "boundary": "Alternate asset original library engineering, no default/full-image method completion or real pose/shape/uncertainty accuracy."}
    np.savez_compressed(output.with_suffix(".npz"), point_vertices=point.vertices.numpy(), point_joints=point.joints.numpy(),
                        sample_vertices=samples.vertices.numpy(), sample_joints=samples.joints.numpy(),
                        sample_body_rotations=poses.numpy(), sample_roots=roots.numpy(), sample_betas=shapes.numpy(),
                        faces=body.faces, metadata=np.array(json.dumps(report, ensure_ascii=False)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError("retain previous variant evidence; choose new output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args.output)
    except Exception as error:  # noqa: BLE001 -- retain exact author path failures
        result = {"status": "R2_original_library_male_mesh_failed", "error": f"{type(error).__name__}: {error}",
                  "traceback": traceback.format_exc(), "training_executed": False}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["status"].endswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
