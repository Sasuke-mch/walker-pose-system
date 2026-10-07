"""Synthetic engineering data from the real decoder; not a human-motion benchmark."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from smplx.lbs import batch_rodrigues

from .body import BodyDecoder


def generate(decoder, count, frames, seed, split, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn(count, 1, 3, generator=generator, device=device) * 0.45
    clock = torch.linspace(0, 1, frames, device=device)[None, :, None]
    motion = latent + 0.25 * torch.sin(clock * 2 + latent)
    joint_phase = torch.arange(52, device=device)[None, None, :, None] / 9
    angles = 0.25 * motion[:, :, None] * torch.cos(joint_phase)
    rotations = batch_rodrigues(angles.reshape(-1, 3)).reshape(count, frames, 52, 3, 3)
    betas = torch.randn(count, 10, generator=generator, device=device) * 0.1
    translation = torch.cat((motion[..., :2] * 0.3, 3 + motion[..., 2:] * 0.2), dim=-1)
    with torch.no_grad():
        decoded = decoder(rotations, betas, translation)
    joints = decoded["joints"][:, :, :22]
    origins = joints.new_tensor([[0, 0, 0], [0.3, 0, 0]])
    relative = joints[:, :, None] - origins[None, None, :, None]
    rays = relative / relative.norm(dim=-1, keepdim=True)
    xy = relative[..., :2] / relative[..., 2:]
    noise = torch.randn(xy.shape, generator=generator, device=device) * 0.001
    confidence = 0.25 + 0.75 * torch.rand(xy.shape[:-1], generator=generator, device=device)
    observed = torch.rand(confidence.shape, generator=generator, device=device) > 0.1
    views = torch.rand(count, frames, 2, generator=generator, device=device) > 0.04
    observed &= views[..., None]
    # Explicit proxy image of projected joints. This is NOT a rendered human RGB image.
    coordinates = torch.linspace(-0.6, 0.6, 32, device=device)
    gy, gx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    delta = (xy[..., 0, None, None] - gx).square() + (xy[..., 1, None, None] - gy).square()
    heat = torch.exp(-delta / 0.001) * observed[..., None, None]
    image = torch.stack([heat[..., c::3, :, :].sum(dim=-3).clamp(max=1) for c in range(3)], dim=3)
    arrays = {"xy": xy + noise, "confidence": confidence, "observation_mask": observed,
              "view_mask": views, "ray_origins": origins[None, None, :, None].expand(count, frames, 2, 22, 3),
              "ray_directions": rays, "geometry_mask": observed.clone(),
              "image_mask": views.clone(), "images": image,
              "rotations": rotations, "pose_mask": torch.ones(count, frames, 52, device=device, dtype=torch.bool),
              "betas": betas, "shape_mask": torch.ones(count, device=device, dtype=torch.bool),
              "translation": translation, "translation_mask": torch.ones(count, frames, device=device, dtype=torch.bool)}
    # Rays must derive from the noisy xy as they would from observed pixels.
    noisy_dirs = torch.cat((arrays["xy"], torch.ones_like(arrays["xy"][..., :1])), dim=-1)
    arrays["ray_directions"] = noisy_dirs / noisy_dirs.norm(dim=-1, keepdim=True)
    arrays["xy"] = torch.where(observed[..., None], arrays["xy"], float("nan"))
    arrays["ray_directions"] = torch.where(observed[..., None], arrays["ray_directions"], float("nan"))
    metadata = {"label_source": "synthetic_real_smplh_artificial_correlated_rotations",
                "reference_frame": "synthetic_cam0_right_handed", "units": "metres",
                "rotation_convention": "absolute_local_matrix_columns", "split": split,
                "subject_ids": [f"{split}_synthetic_subject_{i}" for i in range(count)],
                "sequence_ids": [f"{split}_synthetic_sequence_{i}" for i in range(count)],
                "observation_topology": "smplh_first22_kinematic_joints_not_coco17",
                "geometry_source": "synthetic_pinhole_f1_baseline0.3m_noisy_observations",
                "image_source": "synthetic_projected_joint_heatmap_rgb_proxy_not_real_rgb",
                "seed": seed, "frames": frames, "assets": decoder.asset_paths,
                "evidence_boundary": "Artificial mapping engineering check; not natural motion or physical accuracy"}
    output = {k: value.detach().cpu().numpy() for k, value in arrays.items()}
    output["metadata"] = np.array(json.dumps(metadata, ensure_ascii=False))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    device = config["device"]
    decoder = BodyDecoder(**config["assets"]).to(device)
    for i, (split, count) in enumerate((("train", 64), ("validation", 16), ("test", 16))):
        arrays = generate(decoder, count, 4, config["seed"] + i, split, device)
        np.savez_compressed(output / f"{split}.npz", **arrays)
    print(json.dumps({"output": str(output.resolve()), "counts": [64, 16, 16],
                      "label_source": "synthetic", "mesh": [6890, 13776]}))


if __name__ == "__main__":
    main()
