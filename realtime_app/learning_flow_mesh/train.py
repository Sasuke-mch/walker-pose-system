"""Reproducible training CLI; identical condition encoder for regression control."""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .body import BodyDecoder
from .data import assert_disjoint, load_dataset, masked_mean, observed_inputs, select
from .model import FlowMesh, ModelConfig


def seed_all(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def loss_for(model, batch, mode, training, decoder=None, random_seed=None):
    encoded = model.encoder(observed_inputs(batch))
    pose_mask = batch["pose_mask"]
    target_matrix = torch.where(pose_mask[..., None, None], batch["rotations"],
                                torch.eye(3, device=pose_mask.device))
    target = model.encode_rotations(target_matrix)
    if mode == "flow":
        generator = None if random_seed is None else torch.Generator(device=target.device).manual_seed(random_seed)
        noise = torch.randn(target.shape, device=target.device, generator=generator)
        tau = torch.rand(target.shape[:2], device=target.device, generator=generator)
        # Unknown target nodes retain noise; no artificial identity hand labels.
        destination = torch.where(pose_mask[..., None], target, noise)
        state = (1 - tau[..., None, None]) * noise + tau[..., None, None] * destination
        velocity = model.velocity(state, tau, encoded["condition"])
        pose_loss = masked_mean((velocity - (destination - noise)).square(), pose_mask[..., None])
        estimated = state + (1 - tau[..., None, None]) * velocity
    else:
        estimated = model.regression(encoded)
        pose_loss = masked_mean((estimated - target).square(), pose_mask[..., None])
    shape_loss = masked_mean((encoded["betas"] - torch.nan_to_num(batch["betas"])).square(),
                             (batch["shape_mask"] & encoded["shape_available"])[..., None])
    translation_loss = masked_mean((encoded["translation"] - torch.nan_to_num(batch["translation"])).square(),
                                   batch["translation_mask"][..., None])
    total = pose_loss + training["shape_weight"] * shape_loss + training["translation_weight"] * translation_loss
    metrics = {"pose": pose_loss, "shape": shape_loss, "translation": translation_loss}
    if decoder is not None and training["mesh_weight"]:
        rotations, valid = model.decode_state(estimated)
        if not valid.all():
            raise FloatingPointError("degenerate estimated endpoint; no silent replacement")
        predicted = decoder(rotations, encoded["betas"], encoded["translation"])["vertices"]
        with torch.no_grad():
            reference = decoder(target_matrix, torch.nan_to_num(batch["betas"]),
                                torch.nan_to_num(batch["translation"]))["vertices"]
        # No mesh loss on any frame with partial pose/shape/translation labels.
        known = pose_mask.all(dim=-1) & batch["shape_mask"][:, None] & batch["translation_mask"]
        known &= encoded["shape_available"][:, None]
        mesh_loss = masked_mean((predicted - reference).abs(), known[..., None, None])
        total = total + training["mesh_weight"] * mesh_loss
        metrics["mesh_metres"] = mesh_loss
    return total, {k: float(v.detach()) for k, v in metrics.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("flow", "regression"), default="flow")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    try:
        run(config, args, output)
    except Exception as exc:
        (output / "failure.json").write_text(json.dumps({"type": type(exc).__name__, "message": str(exc)}) + "\n")
        raise


def run(config, args, output):
    seed_all(config["seed"])
    cfg = ModelConfig(**config["model"])
    device = config["device"]
    decoder = BodyDecoder(**config["assets"]).to(device)
    model = FlowMesh(cfg, decoder.parents.detach().cpu()).to(device)
    train, train_meta = load_dataset(Path(args.data) / "train.npz", cfg.observation_joints, cfg.image_feature_dim)
    val, val_meta = load_dataset(Path(args.data) / "validation.npz", cfg.observation_joints, cfg.image_feature_dim)
    if train_meta["split"] != "train" or val_meta["split"] != "validation":
        raise ValueError("training needs explicit train and validation files")
    assert_disjoint(train_meta, val_meta)
    for source in (train, val):
        if not source["pose_mask"].any():
            raise ValueError("no pose supervision; do not train a fake successful model")
    settings = config["training"]
    for key in ("steps", "batch_size", "sample_steps"):
        if not isinstance(settings[key], int) or isinstance(settings[key], bool) or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not np.isfinite(settings["learning_rate"]) or settings["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive and finite")
    for key in ("mesh_weight", "shape_weight", "translation_weight"):
        if not np.isfinite(settings[key]) or settings[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=1e-4)
    val_batch = select(val, torch.arange(len(val["xy"])), device)
    model.eval()
    with torch.no_grad():
        initial_loss, initial_metrics = loss_for(model, val_batch, args.mode, settings, random_seed=123)
    started = time.perf_counter()
    logs = []
    sampler = torch.Generator().manual_seed(config["seed"])
    for step in range(settings["steps"]):
        model.train()
        indices = torch.randint(len(train["xy"]), (settings["batch_size"],), generator=sampler)
        batch = select(train, indices, device)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = loss_for(model, batch, args.mode, settings, decoder=decoder)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at step {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        if step % 20 == 0 or step == settings["steps"] - 1:
            entry = {"step": step + 1, "loss": float(loss.detach()), "gradient_norm": float(norm), **metrics}
            logs.append(entry)
            with (output / "training.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
            print(json.dumps(entry), flush=True)
    model.eval()
    with torch.no_grad():
        final_loss, final_metrics = loss_for(model, val_batch, args.mode, settings, random_seed=123)
    if not torch.isfinite(final_loss):
        raise FloatingPointError("nonfinite final validation loss; checkpoint not accepted")
    checkpoint = {"model_state": model.state_dict(), "config": config,
                  "mode": args.mode, "train_metadata": train_meta, "validation_metadata": val_meta,
                  "step": settings["steps"], "seed": config["seed"]}
    torch.save(checkpoint, output / "checkpoint.pt")
    report = {"mode": args.mode, "initial_validation_loss": float(initial_loss),
              "final_validation_loss": float(final_loss), "initial_metrics": initial_metrics,
              "final_metrics": final_metrics, "steps": settings["steps"], "logs": logs,
              "training_seconds": time.perf_counter() - started,
              "parameters": sum(p.numel() for p in model.parameters()),
              "mesh_auxiliary": ("flow single-step endpoint estimate, not integrated terminal distribution"
                                 if args.mode == "flow" else "deterministic regression terminal mesh"),
              "train": train_meta, "validation": val_meta,
              "evidence_boundary": "Label-source consistency only; synthetic inputs do not demonstrate real image accuracy"}
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("mode", "initial_validation_loss", "final_validation_loss", "training_seconds", "parameters")}))


if __name__ == "__main__":
    main()
