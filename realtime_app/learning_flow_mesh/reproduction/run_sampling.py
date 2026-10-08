"""Run original sampling paths and expose required inputs; no training."""
import argparse
import ast
import copy
import importlib
import json
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

from .check_interfaces import original_function
from .run_original import ROOT, build
from .run_probability import install_download_guard


def original_method(path, owner, name, symbols):
    """Exact reviewed method, bypassing only unrelated framework/package facades."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = dict(symbols)
    code = compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec")
    exec(code, scope)  # noqa: S102 -- exact reviewed author function, no source mutation
    return scope[name]


def aggregation_functions():
    path = ROOT / "FMPose3D/fmpose3d/aggregation_methods.py"
    projection = original_function(ROOT / "FMPose3D/fmpose3d/common/utils.py", "project_to_2d")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = ("aggregation_select_single_best_hypothesis_by_2D_error", "aggregation_RPEA_joint_level")
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(selected) != 2:
        raise ValueError("expected exact original aggregation functions")
    scope = {"torch": torch, "project_to_2d": projection}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)  # noqa: S102
    return scope[names[0]], scope[names[1]], projection


def root_dependency_probe():
    hard, soft, projection = aggregation_functions()
    args = SimpleNamespace(pad=0, topk=3, exp_temp=0.002)
    hypotheses = []
    for x, z in ((0.2, 0.1), (0.4, 0.5), (-0.3, 0.2)):
        h = torch.zeros(1, 1, 17, 3, dtype=torch.float64)
        h[:, :, 1:, 0], h[:, :, 1:, 2] = x, z
        hypotheses.append(h)
    camera = torch.tensor([[1., 1., 0., 0., 0., 0., 0., 0., 0.]], dtype=torch.float64)
    target = torch.zeros(1, 1, 17, 2, dtype=torch.float64)
    target[:, :, 1:, 0] = 0.1
    gt = torch.zeros(1, 1, 17, 3, dtype=torch.float64)
    gt[:, :, 0, 2] = 1
    changed_root = gt.clone()
    changed_root[:, :, 0, 2] = 3
    changed_other = gt.clone()
    changed_other[:, :, 1:] = 100
    results = {}
    for name, function in (("hard_joint_selection", hard), ("topk3_rpea", soft)):
        original = function(args, hypotheses, camera, target, gt)
        root = function(args, hypotheses, camera, target, changed_root)
        other = function(args, hypotheses, camera, target, changed_other)
        root_delta = float((root - original).abs().max())
        other_delta = float((other - original).abs().max())
        assert root_delta > 0.01 and other_delta == 0 and torch.equal(original[:, :, 0], torch.zeros(1, 1, 3))
        results[name] = {"root_change_max_output_delta_m_fixture": root_delta,
                         "nonroot_truth_change_max_output_delta": other_delta,
                         "joint1_root_z1_result": original[0, 0, 1].tolist(),
                         "joint1_root_z3_result": root[0, 0, 1].tolist()}
    # Optional backward capability is distinct from required inference dependency.
    # Preserve original in-place failures; do not patch the author's aggregator.
    backward = {}
    try:
        with torch.enable_grad():
            differentiable_root = changed_root.clone().requires_grad_()
            soft(args, hypotheses, camera, target, differentiable_root).sum().backward()
            gradient = differentiable_root.grad
        backward = {"status": "passed", "root_gradient_l1": float(gradient[:, :, 0].abs().sum()),
                    "nonroot_truth_gradient_l1": float(gradient[:, :, 1:].abs().sum())}
    except RuntimeError as error:
        backward = {"status": "original_backward_failed", "error": str(error),
                    "required_for_eval_forward": False}
    # Released clamp loses distance ranking when all selected errors exceed 20*temperature.
    far_target = target + 2
    far = soft(args, hypotheses, camera, far_target, changed_root)
    average = torch.stack(hypotheses).mean(0)
    saturation_error = float((far - average).abs().max())
    assert saturation_error < 1e-12
    missing_root = changed_root.clone()
    missing_root[:, :, 0] = torch.nan
    missing = soft(args, hypotheses, camera, target, missing_root)
    missing_mean_error = float((missing - average).abs().max())
    assert torch.isfinite(missing).all() and missing_mean_error < 1e-12
    return {"controlled_fixture": "Three unchanged root-relative 17-joint candidates; normalized pinhole camera and fixed 2D; only GT root z changes 1m to3m",
            "functions": results, "optional_backward_probe": backward,
            "far_error_clamp_to_uniform_mean_max_difference": saturation_error,
            "nan_root_still_returns_finite_pose": True,
            "nan_root_to_uniform_mean_max_difference": missing_mean_error,
            "camera_projection_function": projection.__name__,
            "docstring_shape_only_claim_contradicted": True,
            "boundary": "Controlled dependency probe, not pose error/accuracy; root is actual function input, cannot replace it silently for project deployment"}


def fmpose(device):
    model, _, config, options, notes = build("P03")
    model.to(device).eval()
    sampler = original_function(ROOT / "FMPose3D/fmpose3d/common/utils.py", "euler_sample")
    torch.manual_seed(47)
    condition = torch.randn(1, 1, 17, 2, device=device) * 0.1
    noise = torch.randn(1, 1, 17, 3, device=device)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    calls = []

    def record_call(_module, inputs):
        calls.append(tuple(x.detach().clone() for x in inputs))

    handle = model.register_forward_pre_hook(record_call)
    steps_results = []
    try:
        for steps in (1, 3, 5):
            calls.clear()
            sampled = sampler(condition, noise, steps, model)
            nfe = len(calls)
            assert nfe == steps and torch.isfinite(sampled).all()
            reference = noise.clone()
            for step in range(steps):
                t = torch.full((1, 1, 1, 1), step / steps, device=device, dtype=condition.dtype)
                reference = reference + model(condition, reference, t) / steps
            error = float((sampled - reference).abs().max())
            assert error < 1e-6
            repeat = sampler(condition, noise, steps, model)
            assert torch.equal(sampled, repeat)
            steps_results.append({"steps": steps, "nfe": nfe, "shape": list(sampled.shape),
                                  "independent_euler_arithmetic_max_error": error, "same_noise_repeat_exact": True})
        method = original_method(ROOT / "FMPose3D/fmpose3d/inference_api/fmpose3d.py",
            "FMPose3DInference", "_run_euler_sample", {"torch": torch, "np": np,
            "copy": copy, "euler_sample": sampler})
        host = SimpleNamespace(_joints_left=[4, 5, 6, 11, 12, 13],
                               _joints_right=[1, 2, 3, 14, 15, 16], _pad=0,
                               inference_cfg=SimpleNamespace(test_augmentation=True), device=torch.device(device))
        normed = condition[0].cpu().numpy()
        calls.clear()
        torch.manual_seed(48)
        augmented = method(host, normed, model, 3)
        augmentation_nfe = len(calls)
        independent_noise = not torch.equal(calls[0][1], calls[3][1])
        first_condition, flip_condition = calls[0][0], calls[3][0]
        expected_flip = first_condition.clone()
        expected_flip[..., 0] *= -1
        left_right = host._joints_left + host._joints_right
        right_left = host._joints_right + host._joints_left
        expected_flip[:, :, left_right] = expected_flip[:, :, right_left]
        assert torch.equal(flip_condition, expected_flip) and independent_noise and augmentation_nfe == 6
        torch.manual_seed(48)
        augmented_repeat = method(host, normed, model, 3)
        assert torch.equal(augmented, augmented_repeat) and torch.isfinite(augmented).all()
        damaged = condition.clone()
        damaged[:, :, 9] = torch.nan
        failed = sampler(damaged, noise, 3, model)
        nonfinite = int((~torch.isfinite(failed)).sum())
        assert nonfinite > 0  # preserve author missing-data behavior
    finally:
        handle.remove()
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    return {"status": "original_sampling_and_aggregation_dependency_audit_passed",
            "config": config.as_posix(), "model_options": options,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "sampling": steps_results, "released_default_steps": 3,
            "flip_tta": {"nfe": augmentation_nfe, "independent_branch_noise": independent_noise,
                         "condition_flip_and_side_reindex_exact": True, "fixed_seed_repeat_exact": True,
                         "output_shape": list(augmented.shape)},
            "single_missing_wrist_nonfinite_output_values": nonfinite,
            "state_dict_unchanged": True, "rpea_dependency": root_dependency_probe(),
            "adaptations": notes[:1] + ["Single-frame sampling only; 1/3/5 denote Euler steps, not temporal clip length",
                "Exact API method uses explicitly supplied original self members; no inference constructor/weight loader, video detector or world quaternion"],
            "coverage": "Original human velocity network, Euler, single-frame flip-TTA and exact aggregation functions; image detector, trained evaluation, video/camera-to-world and SMPL-H not run"}


def riemannian(device):
    root = ROOT / "RiemannianFM"
    sys.path.insert(0, str(root.resolve()))
    arch = importlib.import_module("manifm.model.arch")
    manifolds = importlib.import_module("manifm.manifolds")
    solvers = importlib.import_module("manifm.solvers")
    cfg = CfgNode(yaml.safe_load((root / "configs/train.yaml").read_text(encoding="utf-8")))
    manifold = manifolds.Sphere()
    net = arch.ProjectToTangent(arch.tMLP(3, d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layers, actfn=cfg.model.actfn, fourier=cfg.model.fourier),
        manifold, cfg.model.metric_normalize).to(device).eval()
    x0 = manifold.random_base(4, 3, device=device)
    calls = [0]

    def counted_field(t, x):
        calls[0] += 1
        return net(t, x)

    scope = {"torch": torch, "projx_integrator": solvers.projx_integrator,
             "projx_integrator_return_last": solvers.projx_integrator_return_last}
    source = root / "manifm/model_pl.py"
    sample = original_method(source, "ManifoldFMLitModule", "sample", scope)
    sample_all = original_method(source, "ManifoldFMLitModule", "sample_all", scope)
    host = SimpleNamespace(manifold=manifold, dim=3, vecfield=counted_field, cfg=cfg)
    before = {k: v.clone() for k, v in net.state_dict().items()}
    final = sample(host, 4, device, x0=x0)
    last_nfe = calls[0]
    calls[0] = 0
    trajectory = sample_all(host, 4, device, x0=x0)
    full_nfe = calls[0]
    assert last_nfe == 1000 and full_nfe == 1001
    assert trajectory.shape == (1001, 4, 3) and torch.isfinite(trajectory).all()
    endpoint_error = float((final - trajectory[-1]).abs().max())
    norm_error = float((trajectory.norm(dim=-1) - 1).abs().max())
    assert endpoint_error == 0 and norm_error < 1e-6
    repeat = sample(host, 4, device, x0=x0)
    assert torch.equal(final, repeat)
    # Analytic field rotates the equator by one radian in one time unit.
    def analytic_field(_t, x):
        return torch.stack((-x[..., 1], x[..., 0], torch.zeros_like(x[..., 0])), dim=-1)

    start = torch.tensor([[1., 0., 0.]], dtype=torch.float64, device=device)
    control = solvers.projx_integrator_return_last(manifold, analytic_field, start,
                  torch.linspace(0, 1, 1001, device=device, dtype=torch.float64), method="euler", projx=True)
    target = torch.tensor([[np.cos(1.), np.sin(1.), 0.]], dtype=torch.float64, device=device)
    analytic_error = float((control - target).abs().max())
    assert analytic_error < 1e-6
    assert all(torch.equal(before[k], v) for k, v in net.state_dict().items())
    return {"status": "original_sphere_sampling_paths_passed", "config": (root / "configs/train.yaml").as_posix(),
            "sample_source": source.as_posix(), "solver_source": (root / "manifm/solvers.py").as_posix(),
            "parameter_count": sum(p.numel() for p in net.parameters()),
            "eval_projx": cfg.eval_projx, "local_coords": cfg.local_coords, "steps": 1000,
            "last_only_nfe": last_nfe, "trajectory_nfe_including_endpoint_velocity": full_nfe,
            "trajectory_shape": list(trajectory.shape), "endpoint_max_difference": endpoint_error,
            "trajectory_unit_norm_max_error": norm_error, "same_initial_state_repeat_exact": True,
            "analytic_rotation_control_max_error": analytic_error, "state_dict_unchanged": True,
            "coverage": "Original configured projected Euler sample/sample_all with original Sphere/network; not original volcano data, likelihood, other manifolds or human SO(3)"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", choices=("P03", "P02"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str((ROOT / "_deps").resolve()))
    torch.set_num_threads(2)
    torch.manual_seed(19)
    start = time.perf_counter()
    try:
        install_download_guard()
        with torch.no_grad():
            result = fmpose(args.device) if args.paper == "P03" else riemannian(args.device)
    except Exception as error:  # noqa: BLE001 -- record external failure and exit nonzero
        result = {"status": "original_sampling_check_failed", "error": f"{type(error).__name__}: {error}",
                  "traceback": traceback.format_exc()}
    result.update(id=args.paper, training_executed=False, weights="random_initialization",
                  protocol="architecture_sampling_protocol.json",
                  environment={"python": platform.python_version(), "torch": torch.__version__, "device": args.device,
                               "elapsed_total_seconds_not_latency": time.perf_counter() - start},
                  evidence_boundary="Untrained structure and dependency audit; no pose accuracy or realtime deployment evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["status"].endswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
