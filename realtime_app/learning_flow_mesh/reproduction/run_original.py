"""Run an author architecture untrained, in a fresh subprocess per paper.

No pretrained weights, optimizer, training step or project observation data are
used. Package facades can import detector/download pipelines; only for FMPose3D
we bypass those __init__ facades via namespace paths. Model source is unchanged.
"""
import argparse
import importlib
import inspect
import json
import platform
import sys
import time
import traceback
import types
from pathlib import Path

import torch
import yaml

ROOT = Path("third_party/architecture_reproduction")


def namespace(name, directory):
    module = types.ModuleType(name)
    module.__path__ = [str(directory.resolve())]
    sys.modules[name] = module


def build(paper):
    if paper == "P12":
        root = ROOT / "MotionBERT"
        sys.path.insert(0, str(root.resolve()))
        source_config = root / "configs/pose3d/MB_ft_h36m.yaml"
        c = yaml.safe_load(source_config.read_text())
        module = importlib.import_module("lib.model.DSTformer")
        options = {k: c[k] for k in ("dim_feat", "dim_rep", "depth", "num_heads", "mlp_ratio", "num_joints", "maxlen", "att_fuse")}
        model = module.DSTformer(**options)
        return model, (1, c["clip_len"], 17, 3), source_config, options, []
    if paper == "P14":
        root = ROOT / "MotionAGFormer"
        sys.path.insert(0, str(root.resolve()))
        source_config = root / "configs/h36m/MotionAGFormer-xsmall.yaml"
        c = yaml.safe_load(source_config.read_text())
        module = importlib.import_module("model.MotionAGFormer")
        allowed = inspect.signature(module.MotionAGFormer).parameters
        options = {k: v for k, v in c.items() if k in allowed and k != "act_layer"}
        model = module.MotionAGFormer(**options)
        return model, (1, c["n_frames"], 17, 3), source_config, options, []
    if paper == "P03":
        root = ROOT / "FMPose3D"
        namespace("fmpose3d", root / "fmpose3d")
        namespace("fmpose3d.models", root / "fmpose3d/models")
        namespace("fmpose3d.models.fmpose3d", root / "fmpose3d/models/fmpose3d")
        module = importlib.import_module("fmpose3d.models.fmpose3d.model_GAMLP")
        # Values checked against scripts/FMPose3D_train.sh and common/config.py.
        options = {"channel": 512, "d_hid": 1024, "token_dim": 256, "layers": 5, "n_joints": 17}
        model = module.Model(types.SimpleNamespace(**options))
        return model, (1, 1, 17, 2), root / "scripts/FMPose3D_train.sh", options, [
            "Bypass top-level detector/inference/animal package facades only; original velocity network and graph files unchanged",
            "Primary check uses published one-frame setting; extra five-frame test checks frame independence, not temporal training"]
    raise ValueError(f"unsupported paper {paper}")


def forward(model, x, paper, state=None, flow_time=None):
    return model(x, state, flow_time) if paper == "P03" else model(x)


def check(paper, device):
    torch.manual_seed(19)
    torch.set_num_threads(2)
    model, shape, cfg_path, options, notes = build(paper)
    model = model.to(device).eval()
    x = torch.randn(shape, device=device) * 0.1
    if shape[-1] == 3:
        x[..., 2] = 0.7  # 2D detector confidence, not a third coordinate.
    state = torch.randn(*shape[:-1], 3, device=device) if paper == "P03" else None
    flow_time = torch.full((*shape[:2], 1, 1), 0.4, device=device) if paper == "P03" else None
    with torch.inference_mode():
        y = forward(model, x, paper, state, flow_time)
        assert y.shape == (*shape[:-1], 3)
        assert torch.isfinite(y).all()
        repeat = forward(model, x, paper, state, flow_time)
        assert torch.equal(y, repeat)
        # Published FMPose model is frame-local; test more frames without changing the network.
        if paper == "P03":
            x = x.expand(-1, 5, -1, -1).clone()
            state = state.expand(-1, 5, -1, -1).clone()
            flow_time = flow_time.expand(-1, 5, -1, -1).clone()
            y = forward(model, x, paper, state, flow_time)
        split = x.shape[1] // 2
        changed = x.clone()
        changed[:, split:, :, :2] += 0.5
        future = forward(model, changed, paper, state, flow_time)
        past_delta = float((future[:, :split] - y[:, :split]).abs().max())
        # NaN is deliberately not cleaned. Record the original missing-data behavior.
        missing = x.clone()
        missing[:, 0, 0, :2] = torch.nan
        missing_y = forward(model, missing, paper, state, flow_time)
        nan_elements = int((~torch.isfinite(missing_y)).sum())
    source = Path(inspect.getfile(type(model))).resolve()
    return {"id": paper, "status": "original_network_forward_passed", "source": str(source),
            "config": cfg_path.as_posix(), "constructor": options, "parameter_count": sum(p.numel() for p in model.parameters()),
            "published_input_shape": list(shape), "published_output_shape": [*shape[:-1], 3],
            "future_probe_frames": x.shape[1], "past_output_max_change_when_future_changed": past_delta,
            "finite_input_repeat_exact": True, "nonfinite_output_elements_after_one_missing_point": nan_elements,
            "original_observation_mask_api": False,
            "weights": "random_initialization", "training_executed": False, "adaptations": notes,
            "coverage": "Original pose-lifting network (P12/P14) or velocity network (P03); mesh heads, image detectors, learned accuracy and full evaluation pipelines are not covered",
            "evidence_boundary": "Architecture/interface behavior only; random outputs are not accuracy or real-time deployment evidence"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", choices=("P03", "P12", "P14"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    start = time.perf_counter()
    try:
        result = check(args.paper, args.device)
    except Exception as error:  # noqa: BLE001 -- persist external-model failure and exit nonzero
        result = {"id": args.paper, "status": "original_network_forward_failed", "error": f"{type(error).__name__}: {error}",
                  "traceback": traceback.format_exc(), "training_executed": False}
    result["environment"] = {"python": platform.python_version(), "torch": torch.__version__, "device": args.device,
                             "elapsed_total_seconds_not_inference_latency": time.perf_counter() - start}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["status"].endswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
