"""Preparation and losses only. No optimizer run or checkpoint loader is exposed."""
import argparse
import json
from pathlib import Path

import torch

from .data_firewall import validate_manifest
from .joint_flow import JointFlow, ModelConfig, cfm_loss, regression_loss


def configuration():
    return {"authorization": "training_disabled", "optimizer": "AdamW", "lr": 3e-4,
            "weight_decay": 0.01, "gradient_clip": 1, "warmup_steps": 1000,
            "schedule": "cosine", "effective_batch": 16, "unroll": 32,
            "smoke_steps": 2000, "main_steps": 50000, "validation_every": 1000,
            "seeds": [7, 17, 27], "objective": "block_equal_weight_linear_CFM",
            "checkpoint_selection": "validation_energy_score_with_point_and_legality_guards",
            "normalization": "train_only_required", "no_endpoint_mse": True,
            "test_locked": True}


def prepare(manifest, nodes=52):
    gates = validate_manifest(manifest)
    model = JointFlow(ModelConfig(nodes=nodes))
    return {"config": configuration(), "model": model.config.to_dict(), "data_gates": gates,
            "parameters": sum(p.numel() for p in model.parameters()),
            "regression_parameters": sum(p.numel() for p in model.parameters()),
            "G0_G1_core_capacity_matched": True, "training_executed": False}


def objective(mode, model, context, labels, normalization, **kwargs):
    if mode == "CFM":
        return cfm_loss(model, context, labels, normalization, **kwargs)
    if mode == "regression":
        return regression_loss(model, context, labels, normalization, **kwargs), {}
    raise ValueError("unregistered objective")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--nodes", type=int, choices=(22, 52), default=52)
    args = parser.parse_args()
    torch.manual_seed(7)
    print(json.dumps(prepare(json.loads(args.manifest.read_text(encoding="utf-8")), args.nodes), indent=2))


if __name__ == "__main__":
    main()
