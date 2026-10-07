"""Label-free model input; targets only enter separately reported evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .body import BodyDecoder
from .data import assert_disjoint, load_dataset, observed_inputs, select
from .model import FlowMesh, ModelConfig
from .train import seed_all


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    cfg = ModelConfig(**checkpoint["config"]["model"])
    model = FlowMesh(cfg, checkpoint["model_state"]["parents"].cpu()).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint


def predict(model, observations, mode, steps, seed):
    encoded = model.encoder(observations)
    if mode == "flow":
        return model.transport(encoded, steps=steps, seed=seed)
    state = model.regression(encoded)
    rotations, valid = model.decode_state(state)
    return {"state": state, "rotations": rotations, "rotation_valid": valid,
            "betas": encoded["betas"], "translation": encoded["translation"],
            "shape_available": encoded["shape_available"], "frame_observed": encoded["frame_observed"], "nfe": 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        run(args, output)
    except Exception as exc:
        (output / "failure.json").write_text(json.dumps({"type": type(exc).__name__,
                                                       "message": str(exc)}) + "\n", encoding="utf-8")
        raise


def run(args, output):
    seed_all(args.seed)
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    cfg = model.cfg
    dataset, metadata = load_dataset(args.input, cfg.observation_joints, cfg.image_feature_dim,
                                     require_targets=False)
    assert_disjoint(checkpoint["train_metadata"], metadata)
    batch = select(dataset, torch.arange(len(dataset["xy"])), args.device)
    decoder = BodyDecoder(**checkpoint["config"]["assets"]).to(args.device)
    with torch.no_grad():
        prediction = predict(model, observed_inputs(batch), checkpoint["mode"], args.steps, args.seed)
        frame_valid = prediction["rotation_valid"].all(-1) & prediction["shape_available"][:, None]
        frame_valid &= torch.isfinite(prediction["betas"]).all(-1)[:, None]
        frame_valid &= torch.isfinite(prediction["translation"]).all(-1)
        vertices = prediction["translation"].new_full((*frame_valid.shape, 6890, 3), float("nan"))
        joints = vertices.new_full((*frame_valid.shape, 52, 3), float("nan"))
        # Keep all failures; decode only valid frames, leave others NaN.
        if frame_valid.any():
            matrices = prediction["rotations"][frame_valid][:, None]
            beta = prediction["betas"][:, None].expand(-1, frame_valid.shape[1], -1)[frame_valid]
            trans = prediction["translation"][frame_valid][:, None]
            decoded = decoder(matrices, beta, trans)
            vertices[frame_valid] = decoded["vertices"][:, 0]
            joints[frame_valid] = decoded["joints"][:, 0]
        known = batch["pose_mask"].all(-1) & batch["shape_mask"][:, None] & batch["translation_mask"]
        evaluation = known & frame_valid
        metrics = {}
        if evaluation.any():
            reference = decoder(batch["rotations"][evaluation][:, None],
                                batch["betas"][:, None].expand(-1, known.shape[1], -1)[evaluation],
                                batch["translation"][evaluation][:, None])
            metrics["vertex_error_metres_label_consistency"] = float(
                (vertices[evaluation] - reference["vertices"][:, 0]).norm(dim=-1).mean())
            permutation = torch.arange(len(batch["xy"]), device=args.device).roll(1)
            shuffled = {k: value[permutation] for k, value in observed_inputs(batch).items()}
            other = predict(model, shuffled, checkpoint["mode"], args.steps, args.seed)
            other_valid = other["rotation_valid"].all(-1) & other["shape_available"][:, None]
            paired = evaluation & other_valid
            if paired.any():
                wrong_mesh = decoder(other["rotations"][paired][:, None],
                                     other["betas"][:, None].expand(-1, known.shape[1], -1)[paired],
                                     other["translation"][paired][:, None])["vertices"][:, 0]
                reference_pair = decoder(batch["rotations"][paired][:, None],
                                         batch["betas"][:, None].expand(-1, known.shape[1], -1)[paired],
                                         batch["translation"][paired][:, None])["vertices"][:, 0]
                metrics["shuffled_condition_vertex_error_metres"] = float((wrong_mesh - reference_pair).norm(dim=-1).mean())
        report = {"input": metadata, "checkpoint": str(Path(args.checkpoint).resolve()),
                  "seed": args.seed, "steps": prediction["nfe"], "candidates": 1,
                  "frames": int(frame_valid.numel()), "valid_frames": int(frame_valid.sum()),
                  "unobserved_frames": int((~prediction["frame_observed"]).sum()),
                  "topology": {"vertices": 6890, "triangles": int(decoder.faces.shape[0])},
                  "metrics": metrics, "oracle_selection": False,
                  "evidence_boundary": "Synthetic/label-source consistency; not independent physical accuracy"}
        arrays = {k: value.cpu().numpy() for k, value in prediction.items() if isinstance(value, torch.Tensor)}
        arrays.update(vertices=vertices.cpu().numpy(), joints=joints.cpu().numpy(),
                      faces=decoder.faces.cpu().numpy(), frame_valid=frame_valid.cpu().numpy())
        arrays["metadata"] = np.array(json.dumps(report))
        np.savez_compressed(output / "prediction.npz", **arrays)
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("frames", "valid_frames", "topology", "metrics")}))


if __name__ == "__main__":
    main()
