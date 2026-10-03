from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


COCO17_TO_SMPLX = {
    0: 55,   # nose
    1: 57,   # left eye
    2: 56,   # right eye
    3: 59,   # left ear
    4: 58,   # right ear
    5: 16,   # left shoulder
    6: 17,   # right shoulder
    7: 18,   # left elbow
    8: 19,   # right elbow
    9: 20,   # left wrist
    10: 21,  # right wrist
    11: 1,   # left hip
    12: 2,   # right hip
    13: 4,   # left knee
    14: 5,   # right knee
    15: 7,   # left ankle
    16: 8,   # right ankle
}
BODY_COCO_INDICES = tuple(range(5, 17))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def select_representative_frame(motion: dict[str, Any]) -> dict[str, Any]:
    """Select by quality only; never select by downstream fit success."""
    ranked = []
    for index, frame in enumerate(motion["frames"]):
        strict = np.asarray(frame["strict"], dtype=bool)[list(BODY_COCO_INDICES)]
        errors = np.asarray(
            [value if value is not None else np.inf for value in frame["errors_px"]],
            dtype=np.float64,
        )[list(BODY_COCO_INDICES)]
        valid_errors = errors[strict & np.isfinite(errors)]
        mean_error = float(np.mean(valid_errors)) if len(valid_errors) else float("inf")
        stage1_priority = int(frame.get("stage") == "stage1_walker_static_human_moving")
        ranked.append((int(strict.sum()), stage1_priority, -mean_error, -index, index))
    selected = max(ranked)[-1]
    return {"frame_index": selected, "frame": motion["frames"][selected]}


def initial_world_from_model_rotation(target_coco_mm: np.ndarray) -> np.ndarray:
    """Map SMPL-X +Y up and +X left onto the observed body frame."""
    target = np.asarray(target_coco_mm, dtype=np.float64)
    left_axis = target[5] - target[6]
    hip_mid = 0.5 * (target[11] + target[12])
    shoulder_mid = 0.5 * (target[5] + target[6])
    up_axis = shoulder_mid - hip_mid
    left_axis /= max(np.linalg.norm(left_axis), 1e-9)
    up_axis -= left_axis * float(left_axis @ up_axis)
    up_axis /= max(np.linalg.norm(up_axis), 1e-9)
    forward_axis = np.cross(left_axis, up_axis)
    forward_axis /= max(np.linalg.norm(forward_axis), 1e-9)
    rotation = np.column_stack((left_axis, up_axis, forward_axis))
    if np.linalg.det(rotation) < 0:
        rotation[:, 2] *= -1.0
    return rotation


def observation_weights(errors_px: np.ndarray, strict: np.ndarray) -> np.ndarray:
    errors = np.asarray(errors_px, dtype=np.float64)
    strict = np.asarray(strict, dtype=bool)
    safe = np.where(np.isfinite(errors), errors, 30.0)
    return np.where(strict, 1.0 / (1.0 + (safe / 5.0) ** 2), 0.08 / (1.0 + (safe / 10.0) ** 2))


def load_vposer_explicit(vposer_dir: str | Path, device: str = "cpu"):
    """Load official V02_05 assets without the upstream Windows path bug."""
    import torch
    from human_body_prior.models.vposer_model import VPoser
    from omegaconf import OmegaConf

    root = Path(vposer_dir).resolve()
    configs = sorted(root.glob("*.yaml"))
    checkpoints = sorted((root / "snapshots").glob("*.ckpt"))
    if len(configs) != 1:
        raise ValueError(f"expected one VPoser yaml in {root}, found {len(configs)}")
    if not checkpoints:
        raise ValueError(f"no VPoser checkpoint found in {root / 'snapshots'}")
    config = OmegaConf.load(configs[0])
    model = VPoser(config).to(device)
    checkpoint_path = checkpoints[-1]
    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device(device), weights_only=False
    )
    model_keys = set(model.state_dict())
    state = {}
    for name, value in checkpoint["state_dict"].items():
        clean = name.removeprefix("vp_model.")
        if clean in model_keys:
            state[clean] = value
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("VPoser checkpoint does not exactly match the model definition")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, config, checkpoint_path

