"""R6 original sphere likelihood and exception semantics, no trained weights."""
import argparse
import ast
import contextlib
import importlib
import io
import json
import math
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from yacs.config import CfgNode

from .run_original import ROOT
from .run_probability import forbid_weights, install_download_guard
from .run_sampling import original_method


def run():
    torch.set_num_threads(2)
    torch.manual_seed(53)
    sys.path.insert(0, str((ROOT / "_deps").resolve()))
    root = ROOT / "RiemannianFM"
    sys.path.insert(0, str(root.resolve()))
    install_download_guard()
    torch.load = forbid_weights
    manifolds = importlib.import_module("manifm.manifolds")
    arch = importlib.import_module("manifm.model.arch")
    solvers = importlib.import_module("manifm.solvers")
    odeint = importlib.import_module("torchdiffeq").odeint
    cfg = CfgNode(yaml.safe_load((root / "configs/train.yaml").read_text()))
    sphere = manifolds.Sphere()
    net = arch.Unbatch(arch.ProjectToTangent(arch.tMLP(3, d_model=cfg.model.d_model,
          num_layers=cfg.model.num_layers, actfn=cfg.model.actfn, fourier=cfg.model.fourier),
          sphere, cfg.model.metric_normalize)).eval()
    source = root / "manifm/model_pl.py"
    symbols = {"torch": torch, "jacrev": torch.func.jacrev, "vmap": torch.func.vmap,
               "vjp": torch.func.vjp, "jvp": torch.func.jvp, "traceback": traceback,
               "ProductManifold": manifolds.ProductManifold, "Euclidean": manifolds.Euclidean,
               "SPD": manifolds.SPD, "odeint": odeint,
               "projx_integrator_return_last": solvers.projx_integrator_return_last}
    tree = ast.parse(source.read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("div_fn", "output_and_div")]
    assert len(funcs) == 2
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(source), "exec"), symbols)  # noqa: S102 -- exact author functions
    density = original_method(source, "ManifoldFMLitModule", "compute_exact_loglikelihood", symbols)
    points = torch.tensor([[1., 0., 0.], [.6, 0., .8]])
    before = {k: v.clone() for k, v in net.state_dict().items()}

    def evaluate(field, x, config=cfg):
        logs = {}

        def record(name, value, **_kwargs):
            number = float(value)
            logs[name] = number if math.isfinite(number) else str(number)

        host = SimpleNamespace(cfg=config, manifold=sphere, dim=3, vecfield=field, log=record)
        return density(host, x, return_projx_error=True), logs

    result, logs = evaluate(net, points)
    repeat, _ = evaluate(net, points)
    assert isinstance(result, tuple) and torch.isfinite(result[0]).all()
    assert torch.equal(result[0], repeat[0]) and logs["nfe"] == 1000
    assert all(torch.equal(v, before[k]) for k, v in net.state_dict().items())
    x = torch.tensor([[1., 0., 0.], [.6, 0., .8]], dtype=torch.float64)
    base = -math.log(4 * math.pi)

    def rotation(_t, y):
        return torch.stack((-y[..., 1], y[..., 0], y[..., 2] * 0), -1)

    rot_result, rot_logs = evaluate(rotation, x)
    rotation_error = float((rot_result[0] - base).abs().max())
    assert rotation_error < 1e-8
    zero_result, _ = evaluate(lambda _t, y: y * 0, x)
    zero_error = float((zero_result[0] - base).abs().max())
    assert zero_error < 1e-8

    # Original ProjectToTangent normalizes x before projection, so ambient divergence
    # equals intrinsic sphere divergence; test the off-sphere extension too.
    class ConstantAxis(torch.nn.Module):
        def forward(self, _t, y):
            return y * 0 + y.new_tensor([0., 0., .3])

    gradient = arch.ProjectToTangent(ConstantAxis(), sphere, False)
    _, div = symbols["output_and_div"](lambda y: gradient(torch.tensor(0.), y), x)
    divergence_error = float((div - (-.6 * x[:, 2])).abs().max())
    assert divergence_error < 1e-8, f"analytic divergence error {divergence_error}"
    grad_result, grad_logs = evaluate(gradient, x)
    z = torch.atanh(x[:, 2])
    expected = base + 2 * (torch.log(torch.cosh(z)) - torch.log(torch.cosh(z - .3)))
    gradient_density_error = float((grad_result[0] - expected).abs().max())
    assert gradient_density_error < 3e-3
    normalized = cfg.clone()
    normalized.normalize_loglik = True
    norm_result, _ = evaluate(rotation, x, normalized)
    normalization_error = float((norm_result[0] - base / 3).abs().max())
    assert normalization_error < 1e-8

    def fail(_t, _x):
        raise RuntimeError("R6 intentional vectorfield failure")

    failure_trace = io.StringIO()
    with contextlib.redirect_stderr(failure_trace):
        failed_result, failed_logs = evaluate(fail, points)
    assert isinstance(failed_result, torch.Tensor) and torch.equal(failed_result, torch.zeros(2))
    assert "R6 intentional vectorfield failure" in failure_trace.getvalue()
    nan_trace = io.StringIO()
    with contextlib.redirect_stderr(nan_trace):
        nan_result, nan_logs = evaluate(lambda _t, y: y * torch.nan, points)
    nan_values = nan_result[0] if isinstance(nan_result, tuple) else nan_result
    return {"status": "R6_original_sphere_density_verified", "training_executed": False, "checkpoint_loaded": False,
            "source": str(source), "configured_network_parameters": sum(p.numel() for p in net.parameters()),
            "input_points": points.tolist(), "seed": 53, "div_mode": cfg.div_mode, "eval_projx": cfg.eval_projx,
            "random_density": result[0].tolist(), "random_logs": logs, "repeat_exact": True, "network_state_unchanged": True,
            "rotation_control_error": rotation_error, "rotation_logs": rot_logs, "zero_field_error": zero_error,
            "sphere_gradient_divergence_error": divergence_error, "gradient_analytic_density_error": gradient_density_error,
            "gradient_logs": grad_logs, "normalize_by_ambient_dim3_error": normalization_error,
            "uniform_sphere_log_density": base,
            "exception_fallback": {"result": failed_result.tolist(), "finite": bool(torch.isfinite(failed_result).all()),
                                   "return_tuple_requested_but_tensor_returned": True, "logs": failed_logs,
                                   "stderr_trace": failure_trace.getvalue()},
            "nan_field": {"nonfinite_density_count": int((~torch.isfinite(nan_values)).sum()), "logs": nan_logs,
                          "returns_tuple": isinstance(nan_result, tuple), "stderr": nan_trace.getvalue()},
            "boundary": "Original SphereS2 likelihood method and exact divergence executed with projectedEuler1000 and randomconfigured network. Function returns logdensity despite negative-loglikelihood docstring. Exception fallback finitezero logdensity/exponentiated density1 is invalid evidence; not dataset likelihood or humanSO3."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("retain previous evidence; choose new version")
    try:
        result = run()
    except Exception as error:  # noqa: BLE001 -- retain source failures
        result = {"status": "R6_failed", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["status"].endswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
