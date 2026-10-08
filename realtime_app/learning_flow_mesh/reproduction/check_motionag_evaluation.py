"""R9 original default MotionAGFormer and evaluation on explicit synthetic data."""
import argparse
import ast
import contextlib
import importlib
import io
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

from .check_motionbert_wild import compile_nodes
from .run_original import ROOT
from .run_probability import forbid_weights, install_download_guard


def run(output):
    torch.set_num_threads(2)
    torch.manual_seed(67)
    install_download_guard()
    torch.load = forbid_weights
    root = ROOT / "MotionAGFormer"
    sys.path.insert(0, str(root.resolve()))
    config = root / "configs/h36m/MotionAGFormer-base.yaml"
    args = SimpleNamespace(**yaml.safe_load(config.read_text()))
    model = importlib.import_module("utils.learning").load_model(args).cuda().eval()
    flip = importlib.import_module("utils.data").flip_data
    metrics = importlib.import_module("loss.pose3d")
    reader_cls = importlib.import_module("data.reader.h36m").DataReaderH36M
    source = root / "train.py"
    method = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
    scope = {"torch": torch, "np": np, "tqdm": iter, "flip_data": flip,
             "calculate_mpjpe": metrics.mpjpe, "calculate_p_mpjpe": metrics.p_mpjpe,
             "calculate_jpe": metrics.jpe, "calculate_acc_err": metrics.acc_error}
    compile_nodes(source, [method], scope)
    x = torch.randn(1, 243, 17, 3) * .1
    x[..., 2] = .8
    ground = np.random.default_rng(67).normal(0, 50, (245, 17, 3))
    # Dataset-defined image dimensions, not project camera identities.
    def reader(total, factor=1., reference=ground):
        host = reader_cls.__new__(reader_cls)
        host.test_hw = None
        host.sample_stride = 1
        host.split_id_train = np.arange(243)[None]
        host.split_id_test = np.arange(243)[None]
        host.dt_dataset = {"test": {"action": ["synthetic"] * total,
                           "2.5d_factor": np.full(total, factor), "joints_2.5d_image": reference[:total].copy(),
                           "source": ["synthetic_fixture000001"] * total, "camera_name": ["55011271"] * total}}
        return host

    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        image = x.cuda()
        a, b = model(image), model(flip(image))
        index = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]
        back = b[:, :, index].clone()
        back[..., 0] *= -1
        combined = ((a + back) / 2).cpu().numpy()
        combined[:, :, 0] = 0
        denorm = reader(245).denormalize(combined.copy())
        independent_denorm = combined.copy()
        independent_denorm[..., :2] = (combined[..., :2] + [1, 1]) * 500
        independent_denorm[..., 2] = combined[..., 2] * 500
        affine_error = float(np.abs(denorm - independent_denorm).max())
        assert affine_error == 0
        changed = image.clone()
        changed[:, 121:, :, :2] += .5
        future_delta = float((model(changed)[:, :120] - a[:, :120]).abs().max())
        zero_conf = image.clone()
        zero_conf[..., 2] = 0
        zero_output = model(zero_conf)
        missing = image.clone()
        missing[0, 0, 0, 0] = torch.nan
        missing_output = model(missing)
        assert torch.equal(a, model(image))
    logs = io.StringIO()
    observed_denormalization = []
    audited_reader = reader(245)
    original_denormalize = audited_reader.denormalize

    def observe_denormalize(*positional, **keywords):
        data = original_denormalize(*positional, **keywords)
        observed_denormalization.append(data.copy())
        return data

    audited_reader.denormalize = observe_denormalize  # Observer only; original method executed, return untouched.
    with contextlib.redirect_stdout(logs):
        try:
            scope["evaluate"](args, model, [(x, torch.zeros(1, 243, 17, 3))], reader(243), "cuda")
        except IndexError as error:
            edge_error = f"{type(error).__name__}: {error}"
        else:
            raise AssertionError("Expected full-coverage acceleration tail indexing failure")
        values = scope["evaluate"](args, model, [(x, torch.zeros(1, 243, 17, 3))], audited_reader, "cuda")
        scaled = scope["evaluate"](args, model, [(x, torch.zeros(1, 243, 17, 3))], reader(245, factor=2.), "cuda")
    assert len(observed_denormalization) == 1 and np.array_equal(observed_denormalization[0], denorm)
    pred = denorm[0] - denorm[0, :, :1]
    gt = ground[:243] - ground[:243, :1]
    joint_errors = np.sqrt(np.sum((pred - gt) ** 2, axis=-1))
    direct = float(joint_errors.mean())
    acceleration = np.sqrt(np.sum(np.diff(pred - gt, n=2, axis=0) ** 2, axis=-1)).mean(axis=1)
    # Original evaluator includes two initialized zero tail entries in action mean.
    direct_acc = float(acceleration.sum() / 243)
    procrustes = []
    for p, g in zip(pred.astype(np.float64), gt):
        pc, gc = p - p.mean(0), g - g.mean(0)
        rotation, _ = Rotation.align_vectors(gc, pc)
        rotated = rotation.apply(pc)
        scale = np.sum(rotated * gc) / np.sum(pc * pc)
        procrustes.append(np.linalg.norm(rotated * scale - gc, axis=-1).mean())
    residuals = {"mean_joint_error": abs(float(values[0]) - direct),
                 "per_joint_error": float(np.abs(values[2] - joint_errors.mean(0)).max()),
                 "acceleration_aggregation": abs(float(values[3]) - direct_acc),
                 "Procrustes_independent_scipy": abs(float(values[1]) - float(np.mean(procrustes)))}
    assert max(residuals.values()) < 2e-5, residuals
    perfect_reference = np.concatenate((denorm[0].astype(np.float64), ground[243:]), axis=0)
    perfect_trace = io.StringIO()
    with contextlib.redirect_stdout(perfect_trace), contextlib.redirect_stderr(perfect_trace):
        try:
            perfect = scope["evaluate"](args, model, [(x, torch.zeros(1, 243, 17, 3))], reader(245, reference=perfect_reference), "cuda")
        except AssertionError as error:
            perfect_error = f"{type(error).__name__}: {error}"
        else:
            perfect_error = "returned " + str(perfect)
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    np.savez_compressed(output.with_suffix(".npz"), normalized_input=x.numpy(), TTA_root_zero=combined,
                        image_relative_output=denorm, synthetic_reference=ground,
                        zero_conf_output=zero_output.cpu().numpy(), NaN_output=missing_output.cpu().numpy())
    return {"status": "R9_original_base_evaluation_contract_verified", "training_executed": False, "checkpoint_loaded": False,
            "config": str(config), "layers": args.n_layers, "frames": args.n_frames,
            "parameters": sum(p.numel() for p in model.parameters()), "source_evaluate_lines": [method.lineno, method.end_lineno],
            "synthetic_reader_initialization": "Exact original class/methods, __new__ skips fileIO; explicit synthetic dt_dataset and one clip split caches",
            "243_fullcoverage_failure": edge_error, "245_total_one243clip_execution": True, "uncovered_tail_frames": [243, 244],
            "independent_residuals": residuals, "pixel_denormalize_error": affine_error,
            "full_evaluator_TTA_root_denormalization_elementwise_exact": True,
            "observer": "Process-local reader.denormalize observer invokes unchanged original and copies result without altering return",
            "synthetic_reference_diagnostics": {"factor1": [float(values[0]), float(values[1]), float(values[3])],
                                               "factor2": [float(scaled[0]), float(scaled[1]), float(scaled[3])]},
            "acceleration_original_tail_zero_count": 2, "acceleration_fps_normalization": False,
            "perfect_reference_behavior": perfect_error, "perfect_reference_stdout_stderr": perfect_trace.getvalue(),
            "evaluation_stdout": logs.getvalue(), "repeat_exact": True, "state_dict_unchanged": True,
            "future_prefix_network_max_delta": future_delta, "zero_conf_output_finite": bool(torch.isfinite(zero_output).all()),
            "zero_conf_output_nonzero_count": int(torch.count_nonzero(zero_output)),
            "NaN_output_nonfinite_count": int((~torch.isfinite(missing_output)).sum()),
            "boundary": "Original CLI-defaultBASE fullnetwork/full evaluation function on declared artificial references only; no authenticH36M/detector/video/weights or accuracy claim. Factor2.5D from dataset controls metric scale; acceleration second differences lackFPS despite printed mm/s². Original edge/perfect-error failures retained, not patched."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError("retain previous evidence; choose new version")
    try:
        report = run(args.output)
    except Exception as error:  # noqa: BLE001 -- preserve source failures
        report = {"status": "R9_failed", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True), flush=True)
    if report["status"].endswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
