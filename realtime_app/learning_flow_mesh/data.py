"""Explicit NPZ contract and split provenance for the isolated learning module."""
import json
from pathlib import Path

import numpy as np
import torch

from .rotations import validate_rotation_matrices

OBSERVATION_KEYS = ("xy", "confidence", "observation_mask", "view_mask",
                    "ray_origins", "ray_directions", "geometry_mask", "image_mask")
TARGET_KEYS = ("rotations", "pose_mask", "betas", "shape_mask", "translation",
               "translation_mask")
BOOL_KEYS = ("observation_mask", "view_mask", "geometry_mask", "image_mask",
             "pose_mask", "shape_mask", "translation_mask")


def load_dataset(path, joints, feature_dim, require_targets=True):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {k: np.array(archive[k], copy=True) for k in archive.files if k != "metadata"}
    for key in OBSERVATION_KEYS + (TARGET_KEYS if require_targets else ()):
        if key not in arrays:
            raise ValueError(f"dataset missing {key}")
    if arrays["xy"].ndim != 5 or arrays["xy"].shape[-2:] != (joints, 2):
        raise ValueError("xy must be N,T,V,J,2")
    n, t, v = arrays["xy"].shape[:3]
    if not require_targets:
        present_targets = set(TARGET_KEYS) & set(arrays)
        if present_targets and present_targets != set(TARGET_KEYS):
            raise ValueError("evaluation labels must include all target fields and explicit masks")
        if not present_targets:
            arrays.update(rotations=np.full((n, t, 52, 3, 3), np.nan, dtype=np.float32),
                          pose_mask=np.zeros((n, t, 52), dtype=np.bool_),
                          betas=np.full((n, 10), np.nan, dtype=np.float32),
                          shape_mask=np.zeros(n, dtype=np.bool_),
                          translation=np.full((n, t, 3), np.nan, dtype=np.float32),
                          translation_mask=np.zeros((n, t), dtype=np.bool_))
    for key in BOOL_KEYS:
        if arrays[key].dtype != np.bool_:
            raise ValueError(f"{key} must be boolean, not an implicit confidence threshold")
    shapes = {"confidence": (n, t, v, joints), "observation_mask": (n, t, v, joints),
              "view_mask": (n, t, v), "ray_origins": (n, t, v, joints, 3),
              "ray_directions": (n, t, v, joints, 3), "geometry_mask": (n, t, v, joints),
              "image_mask": (n, t, v), "rotations": (n, t, 52, 3, 3),
              "pose_mask": (n, t, 52), "betas": (n, 10), "shape_mask": (n,),
              "translation": (n, t, 3), "translation_mask": (n, t)}
    for key, expected in shapes.items():
        if arrays[key].shape != expected:
            raise ValueError(f"{key} must have shape {expected}")
    if "image_features" in arrays and arrays["image_features"].shape != (n, t, v, feature_dim):
        raise ValueError("image feature shape does not match config")
    if "images" in arrays and (arrays["images"].ndim != 6 or arrays["images"].shape[:4] != (n, t, v, 3)):
        raise ValueError("images must be N,T,V,3,H,W RGB float [0,1]")
    if "image_features" in arrays and "images" in arrays:
        raise ValueError("choose images or supplied features")
    if not n or not t or not v:
        raise ValueError("empty dataset")
    for key in ("label_source", "reference_frame", "units", "rotation_convention",
                "subject_ids", "sequence_ids", "split", "observation_topology",
                "geometry_source", "image_source"):
        if key not in metadata:
            raise ValueError(f"metadata missing {key}")
    if metadata["units"] != "metres" or metadata["rotation_convention"] != "absolute_local_matrix_columns":
        raise ValueError("explicit metric/rotation convention required")
    if len(metadata["subject_ids"]) != n or len(metadata["sequence_ids"]) != n:
        raise ValueError("one subject and sequence identity per sample required")
    if metadata["split"] not in ("train", "validation", "test"):
        raise ValueError("unknown split")
    tensors = {k: torch.as_tensor(a, dtype=torch.bool if k in BOOL_KEYS else torch.float32)
               for k, a in arrays.items()}
    if tensors["pose_mask"].any():
        validate_rotation_matrices(tensors["rotations"][tensors["pose_mask"]])
    for values, mask in ((tensors["betas"], tensors["shape_mask"]),
                         (tensors["translation"], tensors["translation_mask"])):
        if not torch.isfinite(values[mask]).all():
            raise ValueError("nonfinite present labels")
    metadata["input_path"] = str(Path(path).resolve())
    return tensors, metadata


def assert_disjoint(first, second):
    for key in ("subject_ids", "sequence_ids"):
        if set(first[key]) & set(second[key]):
            raise ValueError(f"train/heldout overlap in {key}; adjacent-frame splits forbidden")
    for key in ("reference_frame", "units", "rotation_convention", "observation_topology"):
        if first[key] != second[key]:
            raise ValueError(f"train/heldout disagree on {key}")


def select(batch, indices, device):
    return {k: value[indices].to(device) for k, value in batch.items()}


def observed_inputs(batch):
    return {k: v for k, v in batch.items() if k in OBSERVATION_KEYS + ("images", "image_features")}


def masked_mean(values, mask):
    """No labels => differentiable zero; no implicit replacement with full loss."""
    mask = mask.expand_as(values)
    safe = torch.where(mask, values, torch.zeros_like(values))
    return safe.sum() / mask.sum().clamp_min(1)
