"""R12 original temporal semantics, not complete VIBE/SMPL reproduction."""
import argparse
import ast
import inspect
import json
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from .run_original import ROOT


def original_encoder():
    path = ROOT / "VIBE/lib/models/vibe.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TemporalEncoder")
    scope = {"nn": nn, "F": F}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)  # noqa: S102 -- reviewed original class only
    config = yaml.safe_load((ROOT / "VIBE/configs/config.yaml").read_text(encoding="utf-8"))
    opts = config["MODEL"]["TGRU"]
    options = {"n_layers": opts["NUM_LAYERS"], "hidden_size": opts["HIDDEN_SIZE"],
               "add_linear": opts["ADD_LINEAR"], "bidirectional": opts["BIDIRECTIONAL"],
               "use_residual": opts["RESIDUAL"]}
    return scope["TemporalEncoder"](**options).eval(), options, [cls.lineno, cls.end_lineno]


def carried_forward(model, features, hidden=None):
    """Explicit diagnostic state adapter; returns unchanged author's projection.

    Original forward discards the GRU hidden. This adapter supplies/reads that
    same GRU's state using scoped hooks; it is not an author streaming API.
    Shared-module concurrent/compiled calls are unsupported.
    """
    if model.gru.bidirectional or model.training:
        raise ValueError("carry diagnostic requires original eval one-way GRU")
    captures = []

    def carry(_module, args):
        return (args[0], hidden) if hidden is not None else None

    def read(_module, _args, output):
        captures.append(output[1])

    pre = model.gru.register_forward_pre_hook(carry)
    post = model.gru.register_forward_hook(read)
    try:
        output = model(features)
    finally:
        pre.remove()
        post.remove()
    if len(captures) != 1:
        raise ValueError("expected one original GRU state")
    return output, captures[0]


def run(output):
    torch.set_num_threads(2)
    torch.manual_seed(113)
    model, options, lines = original_encoder()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    features = torch.randn(1, 48, 2048) * .1
    with torch.no_grad():
        whole = model(features)
        repeated = model(features)
        assert torch.equal(whole, repeated)
        windows = [features[:, i:i + 16] for i in range(0, 48, 16)]
        reset = torch.cat([model(chunk) for chunk in windows], 1)
        carried, hidden = [], None
        for chunk in windows:
            value, hidden = carried_forward(model, chunk, hidden)
            carried.append(value)
        carried = torch.cat(carried, 1)
        carry_error = float((whole - carried).abs().max())
        reset_delta = float((whole[:, 16:] - reset[:, 16:]).abs().max())
        assert carry_error < 2e-5 and reset_delta > 0
        assert torch.equal(reset[:, :16], whole[:, :16])
        future = features.clone()
        future[:, 24:] += .5
        future_delta = float((model(future)[:, :24] - whole[:, :24]).abs().max())
        assert future_delta == 0
        zero = model(torch.zeros_like(features))
        assert torch.isfinite(zero).all() and zero.abs().sum() > 0
        missing = features.clone()
        missing[:, 17, 0] = torch.nan
        nan_output = model(missing)
        assert torch.equal(nan_output[:, :17], whole[:, :17])
        invalid_per_frame = (~torch.isfinite(nan_output)).sum(-1)[0]
        assert invalid_per_frame[:17].eq(0).all() and invalid_per_frame[17:].eq(2048).all()
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    assert not model.gru._forward_hooks and not model.gru._forward_pre_hooks
    report = {"status": "R12_original_temporal_missing_and_state_contract_verified",
              "training_executed": False, "checkpoint_loaded": False, "network_requests": 0,
              "whole_goal_complete": False, "config": "VIBE/configs/config.yaml", "constructor": options,
              "original_class_lines": lines, "seed": 113, "published_window": 16,
              "input_shape": list(features.shape), "output_shape": list(whole.shape),
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "native_forward_signature": str(inspect.signature(model.forward)),
              "native_hidden_mask_frameID_API": False, "state_dict_unchanged": True,
              "repeat_exact": True, "reset_first_window_exact": True,
              "reset_windows_later_delta": reset_delta, "explicit_carry_vs_whole_max_error": carry_error,
              "future24_prefix_delta": future_delta, "zero_features_finite": True,
              "zero_features_nonzero_output_values": int(torch.count_nonzero(zero)),
              "one_NaN_nonfinite_values_per_frame": invalid_per_frame.tolist(),
              "NaN_prefix17_exact": True,
              "adapter": "Original GRU receives explicit carried state through scoped hook; native TemporalEncoder discardshidden and resets each invocation. No concurrent/export or elapsedtime claim.",
              "boundary": "Temporal component only; cannot mark zero-feature or finite output as observation availability. Complete VIBE Regressor/neutral/defaultmesh/detector still not executed."}
    np.savez_compressed(output.with_suffix(".npz"), input_features=features.numpy(), whole=whole.numpy(),
                        reset_windows=reset.numpy(), carried_windows=carried.numpy(), final_hidden=hidden.numpy(),
                        zero_features_output=zero.numpy(), NaN_features=missing.numpy(), NaN_output=nan_output.numpy())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError("retain prior outputs; select a new version")
    try:
        result = run(args.output)
    except Exception as error:  # noqa: BLE001 -- preserve exact source/runtime failure
        result = {"status": "R12_original_temporal_failed", "error": f"{type(error).__name__}: {error}",
                  "traceback": traceback.format_exc(), "training_executed": False}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)
    if result["status"].endswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
