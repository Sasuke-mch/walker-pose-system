"""Check original probability/mesh heads without training or mainline changes.

Run one paper in each process. Isolated dependencies live under the ignored
source root; licensed assets and external code are never copied into Git.
"""
import argparse
import ast
import importlib
import inspect
import json
import pickle
import platform
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from yacs.config import CfgNode

from .run_original import ROOT, namespace

MEAN_PARAMS = ROOT / "smpl_mean_params_mmpose.npz"
SMPL_PARENTS_ASSET = Path("models/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")


def forbid_weights(*_args, **_kwargs):
    raise RuntimeError("Weight download attempted during random-initialization check")


def install_download_guard():
    """Process-local guard, installed before importing any author's model."""
    importlib.import_module("torch.utils.model_zoo").load_url = forbid_weights
    torch.hub.load_state_dict_from_url = forbid_weights
    torch.hub.download_url_to_file = forbid_weights


def rotation_check(rotations):
    """Check proper rotations, including determinant (orthogonality alone is insufficient)."""
    eye = torch.eye(3, device=rotations.device, dtype=rotations.dtype)
    orthogonal = float((rotations.transpose(-1, -2) @ rotations - eye).abs().max())
    determinant = float((torch.linalg.det(rotations) - 1).abs().max())
    assert torch.isfinite(rotations).all() and orthogonal < 1e-4 and determinant < 1e-4
    return {"orthogonality_max_error": orthogonal, "determinant_max_error": determinant}


def smpl_parents():
    # Compatibility names needed only to read this existing licensed legacy asset.
    # The subprocess ends after the check; no asset/site-package is changed.
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    for name, value in {"bool": np.bool_, "int": np.int64, "float": np.float64,
                        "complex": np.complex128, "object": np.object_,
                        "unicode": np.str_, "str": np.str_}.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    with SMPL_PARENTS_ASSET.open("rb") as handle:
        data = pickle.load(handle, encoding="latin1")
    table = np.asarray(data["kintree_table"])
    identifiers = {int(identifier): i for i, identifier in enumerate(table[1])}
    parents = [-1] + [identifiers[int(parent)] for parent in table[0, 1:]]
    assert len(parents) == 24 and all(0 <= p < i for i, p in enumerate(parents[1:], 1))
    return parents


def prohmr(device):
    root = ROOT / "ProHMR"
    sys.path.insert(0, str((ROOT / "ProHMR_nflows").resolve()))
    for name, path in {"prohmr": root, "prohmr.models": root / "prohmr/models",
                       "prohmr.models.heads": root / "prohmr/models/heads",
                       "prohmr.models.backbones": root / "prohmr/models/backbones",
                       "prohmr.utils": root / "prohmr/utils"}.items():
        namespace(name, root / "prohmr" if name == "prohmr" else path)
    source_config = root / "prohmr/configs/prohmr.yaml"
    cfg = CfgNode(yaml.safe_load(source_config.read_text()))
    cfg.SMPL.MEAN_PARAMS = str(MEAN_PARAMS.resolve())
    backbone = importlib.import_module("prohmr.models.backbones.resnet").resnet(cfg, pretrained=False).to(device).eval()
    head = importlib.import_module("prohmr.models.heads.smpl_flow").SMPLFlow(cfg).to(device).eval()
    x = torch.randn(1, 3, 224, 224, device=device) * 0.1
    features = backbone(x)
    # Exact released inference order: 63 random candidates and one zero latent.
    # The fork's explicit-noise path does not repeat context before base log_prob
    # for N>1. Keep that observed failure; do not patch the author implementation.
    sampled = head(features, num_samples=cfg.TRAIN.NUM_TEST_SAMPLES - 1)
    zero = torch.zeros(1, 1, cfg.MODEL.FLOW.DIM, device=device)
    mode = head(features, z=zero)
    params = {k: torch.cat((mode[0][k], sampled[0][k]), dim=1) for k in mode[0]}
    camera = torch.cat((mode[1], sampled[1]), dim=1)
    density = torch.cat((mode[2], sampled[2]), dim=1)
    noise = torch.cat((zero, sampled[3]), dim=1)
    pose6d = torch.cat((mode[4], sampled[4]), dim=1)
    repeat = head(features, z=zero)
    assert torch.equal(mode[-1], repeat[-1])
    recovered_density, recovered_noise = head.log_prob(
        {"global_orient": pose6d[..., :6], "body_pose": pose6d[..., 6:]}, features)
    inverse_error = float((recovered_noise - noise).abs().max())
    density_error = float((recovered_density - density).abs().max())
    assert inverse_error < 1e-4 and density_error < 1e-3 and torch.isfinite(density).all()
    rotations = torch.cat((params["global_orient"], params["body_pose"]), dim=2)
    beta_delta = float((params["betas"] - params["betas"][:, :1]).abs().max())
    assert beta_delta == 0
    return {"status": "original_backbone_probability_head_passed", "config": source_config.as_posix(),
            "parameter_count": sum(p.numel() for m in (backbone, head) for p in m.parameters()),
            "input_shape": list(x.shape), "feature_shape": list(features.shape),
            "rotation_shape": list(rotations.shape), "beta_shape": list(params["betas"].shape),
            "camera_shape": list(camera.shape), "density_shape": list(density.shape),
            "flow_inverse_max_error": inverse_error, "density_roundtrip_max_error": density_error,
            "rotation_check": rotation_check(rotations), "zero_latent_repeat_exact": True,
            "candidate_beta_max_difference": beta_delta, "mean_params": MEAN_PARAMS.as_posix(),
            "adaptations": ["Namespace package facades; original ResNet50/ConditionalGlow unchanged, pretrained=False explicitly",
                            "Public MMPose mean initialization fixture; equivalence to original release bundle unconfirmed",
                            "ActNorm stays at constructor identity in eval; no GT-dependent initialization or training"],
            "known_author_interface_issue": "Explicit noise with N>1 fails context-row check; released 63 random + 1 zero path is retained",
            "coverage": "Original RGB backbone plus pose-flow/beta/weak-camera head, released 63 random + 1 zero candidates, invertible-flow density roundtrip; original neutral SMPL and full loss/evaluation not run"}


def humaniflow(device):
    root = ROOT / "HuManiFlow"
    sys.path.insert(0, str(root.resolve()))
    cfg = importlib.import_module("configs.humaniflow_config").get_humaniflow_cfg_defaults()
    model = importlib.import_module("models.humaniflow_model").HumaniflowModel(
        device, cfg.MODEL, smpl_parents()).to(device).eval()
    x = torch.randn(1, cfg.MODEL.NUM_IN_CHANNELS, cfg.DATA.PROXY_REP_SIZE,
                    cfg.DATA.PROXY_REP_SIZE, device=device) * 0.1
    # input is a structural proxy fixture, not RGB or real detector/edge output.
    output = model(x, num_samples=4, compute_point_est=True, return_input_feats=True)
    point = output["pose_rotmats_point_est"]
    samples = output["pose_rotmats_samples"]
    target = model(None, input_feats=output["input_feats"], num_samples=0,
                   compute_point_est=False, compute_for_loglik=True,
                   shape_for_loglik=output["shape_mode"], pose_R_for_loglik=point,
                   glob_R_for_loglik=output["glob_rotmat"])
    logprob = torch.stack([dist.log_prob(point[:, i].double()) for i, dist in enumerate(
        target["conditioned_pose_SO3flow_dists_for_loglik"])], dim=-1)
    assert torch.isfinite(logprob).all()
    repeat = model(None, input_feats=output["input_feats"], num_samples=0, compute_point_est=True)
    assert torch.equal(point, repeat["pose_rotmats_point_est"])
    shape_spread = float((output["shape_samples"] - output["shape_samples"][:, :1]).abs().max())
    mode_only = model(None, input_feats=output["input_feats"], num_samples=4,
                      compute_point_est=False, use_shape_mode_for_samples=True)
    assert torch.equal(mode_only["shape_samples"], output["shape_mode"][:, None].expand(-1, 4, -1))
    return {"status": "original_hierarchical_probability_network_passed",
            "config": (root / "configs/humaniflow_config.py").as_posix(),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "proxy_input_shape": list(x.shape), "input_semantics": "17 COCO heatmaps plus edge channel; fixture is random, not RGB",
            "body_rotations_shape": list(point.shape), "sample_rotations_shape": list(samples.shape),
            "beta_sample_shape": list(output["shape_samples"].shape), "joint_logprob_shape": list(logprob.shape),
            "point_rotation_check": rotation_check(point), "sample_rotation_check": rotation_check(samples),
            "global_rotation_check": rotation_check(output["glob_rotmat"]),
            "point_estimate_repeat_exact": True, "candidate_beta_max_difference": shape_spread,
            "use_shape_mode_for_samples_produces_shared_candidate_beta": True,
            "parent_asset": SMPL_PARENTS_ASSET.as_posix(), "parents": smpl_parents(),
            "coverage": "Original ResNet18/23 hierarchical SO(3) flows, shape distribution, weak camera, point/sampling/log-density paths; no detector/edge generation, SMPL neutral mesh or full evaluation",
            "adaptations": ["Actual local male SMPL kinematic tree (24 nodes) used only as topology; no SMPL-H substitution", "Random 18-channel proxy and four candidates for structural checks; no checkpoint"]}


def hmr2(device, backbone_enabled):
    root = ROOT / "HMR2"
    for name, path in {"hmr2": "hmr2", "hmr2.models": "hmr2/models", "hmr2.utils": "hmr2/utils",
                       "hmr2.models.heads": "hmr2/models/heads", "hmr2.models.components": "hmr2/models/components",
                       "hmr2.models.backbones": "hmr2/models/backbones"}.items():
        namespace(name, root / path)
    source_config = root / "hmr2/configs_hydra/experiment/hmr_vit_transformer.yaml"
    cfg = CfgNode(yaml.safe_load(source_config.read_text()))
    cfg.SMPL = CfgNode(yaml.safe_load((root / "hmr2/configs_hydra/experiment/default.yaml").read_text())["SMPL"])
    cfg.SMPL.MEAN_PARAMS = str(MEAN_PARAMS.resolve())
    # Construct the published ViT-H directly on GPU to avoid >2.5 GB host allocation.
    with torch.device(device):
        head = importlib.import_module("hmr2.models.heads.smpl_head").SMPLTransformerDecoderHead(cfg)
        backbone = importlib.import_module("hmr2.models.backbones.vit").vit(cfg) if backbone_enabled else None
    head = head.to(device).eval()
    if backbone is not None:
        backbone.to(device)
        # Author ViT.train() has no return value; do not assume fluent eval().
        backbone.eval()
        image = torch.randn(1, 3, 256, 256, device=device) * 0.1
        features = backbone(image[:, :, :, 32:-32])  # original HMR2.forward_step crop.
    else:
        features = torch.randn(1, 1280, 16, 12, device=device) * 0.1
    params, camera, history = head(features)
    repeat = head(features)
    assert torch.equal(params["betas"], repeat[0]["betas"])
    rotations = torch.cat((params["global_orient"], params["body_pose"]), dim=1)
    assert params["betas"].shape == (1, 10) and camera.shape == (1, 3)
    assert torch.isfinite(camera).all() and torch.isfinite(params["betas"]).all()
    count = sum(p.numel() for m in (head, backbone) if m is not None for p in m.parameters())
    return {"status": "original_vit_and_mesh_head_passed" if backbone_enabled else "original_mesh_head_passed",
            "config": source_config.as_posix(), "parameter_count": count, "backbone_enabled": backbone_enabled,
            "rgb_input_shape": [1, 3, 256, 256] if backbone_enabled else None,
            "backbone_input_after_original_crop": [1, 3, 256, 192] if backbone_enabled else None,
            "feature_shape": list(features.shape), "rotation_shape": list(rotations.shape),
            "beta_shape": list(params["betas"].shape), "camera_shape": list(camera.shape),
            "rotation_check": rotation_check(rotations), "finite_repeat_exact": True,
            "ief_iters": cfg.MODEL.SMPL_HEAD.get("IEF_ITERS", 1), "history_beta_shape": list(history["betas"].shape),
            "mean_params": MEAN_PARAMS.as_posix(),
            "coverage": "Original ViT-H/16 (when enabled) and six-layer Transformer SMPL head; neutral SMPL mesh, detector/crop estimation, checkpoint and full evaluation not run",
            "adaptations": ["Namespace facades bypass Lightning/downloads; source unchanged", "Public MMPose initialization fixture; original release-bundle equivalence unconfirmed", "GPU allocation only; original width/depth/float32 retained"]}


def riemannian(device):
    root = ROOT / "RiemannianFM"
    sys.path.insert(0, str(root.resolve()))
    module = importlib.import_module("manifm.model.arch")
    manifolds = importlib.import_module("manifm.manifolds")
    c = yaml.safe_load((root / "configs/train.yaml").read_text())
    manifold = manifolds.Sphere()
    net = module.ProjectToTangent(module.tMLP(3, d_model=c["model"]["d_model"],
        num_layers=c["model"]["num_layers"], actfn=c["model"]["actfn"],
        fourier=c["model"]["fourier"]), manifold, c["model"]["metric_normalize"]).to(device).eval()
    x0 = manifold.random_base(4, 3, device=device)
    x1 = manifold.random_base(4, 3, device=device)
    t = torch.tensor([0., 0.4, 1.], device=device)
    path = manifolds.geodesic(manifold, x0, x1)
    xt, ut = torch.func.jvp(path, (t,), (torch.ones_like(t),))
    endpoint_error = float(torch.maximum((xt[:, 0] - x0).abs().max(), (xt[:, -1] - x1).abs().max()))
    norm_error = float((xt.norm(dim=-1) - 1).abs().max())
    target_tangent_error = float((xt * ut).sum(-1).abs().max())
    v = net(torch.full((4, 1), 0.4, device=device), xt[:, 1])
    tangent_error = float((xt[:, 1] * v).sum(-1).abs().max())
    assert max(endpoint_error, norm_error, target_tangent_error, tangent_error) < 1e-4
    # Exact author's method, without importing the Lightning/data/visualization facade.
    source = root / "manifm/model_pl.py"
    tree = ast.parse(source.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ManifoldFMLitModule")
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "rfm_loss_fn")
    symbols = {"torch": torch, "Mesh": manifolds.Mesh, "SPD": manifolds.SPD,
               "geodesic": manifolds.geodesic, "jvp": torch.func.jvp, "vmap": torch.func.vmap}
    code = compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), "exec")
    exec(code, symbols)  # noqa: S102 -- exact reviewed author method, no source mutation
    host = SimpleNamespace(manifold=manifold, dim=3, vecfield=net)
    loss = symbols["rfm_loss_fn"](host, {"x0": x0, "x1": x1})
    assert torch.isfinite(loss) and loss >= 0
    return {"status": "original_sphere_vector_field_and_loss_passed",
            "config": (root / "configs/train.yaml").as_posix(), "source_loss": source.as_posix(),
            "parameter_count": sum(p.numel() for p in net.parameters()), "manifold": "Sphere S2, ambient dim3",
            "path_shape": list(xt.shape), "path_endpoint_max_error": endpoint_error,
            "path_unit_norm_max_error": norm_error, "target_tangent_max_error": target_tangent_error,
            "vector_field_tangent_max_error": tangent_error, "untrained_loss": float(loss),
            "coverage": "Original sphere geodesic/JVP, configured 512-wide eight-layer time-MLP/tangent projection and exact author rfm_loss_fn; no training, density/ODE evaluation, meshes or SO(3) human pose",
            "adaptations": ["Sphere geometry fixture, not original volcano dataset", "Exact loss method AST only bypasses Lightning/data facade; original manifold/network source unchanged"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", choices=("P04", "P05", "P07", "P02"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hmr2-backbone", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str((ROOT / "_deps").resolve()))
    # Some author factories default to pretrained=True. Enforce the declared scope.
    torch.manual_seed(19)
    torch.set_num_threads(2)
    start = time.perf_counter()
    try:
        install_download_guard()
        # JVP of manifold paths is an interface check, not an optimizer/training step.
        with torch.no_grad():
            result = hmr2(args.device, args.hmr2_backbone) if args.paper == "P07" else {
                "P04": prohmr, "P05": humaniflow, "P02": riemannian}[args.paper](args.device)
    except Exception as error:  # noqa: BLE001 -- retain failures instead of claiming partial success
        result = {"status": "original_probability_check_failed", "error": f"{type(error).__name__}: {error}",
                  "traceback": traceback.format_exc()}
    result.update(id=args.paper, training_executed=False, weights="random_initialization",
                  environment={"python": platform.python_version(), "torch": torch.__version__, "device": args.device,
                               "elapsed_total_seconds_not_latency": time.perf_counter() - start},
                  evidence_boundary="Untrained structural/probability contracts only; no accuracy or deployment conclusion")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["status"].endswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
