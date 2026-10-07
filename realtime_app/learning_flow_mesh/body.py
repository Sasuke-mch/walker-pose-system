"""Decode actual local licensed assets. No downloads, substitute meshes or PCA priors."""
import inspect
import pickle
from pathlib import Path

import numpy as np
import smplx
import torch
from smplx.utils import Struct
from torch import nn

from .rotations import validate_rotation_matrices


class BodyDecoder(nn.Module):
    def __init__(self, model_path, mano_left, mano_right):
        super().__init__()
        paths = [Path(p).resolve() for p in (model_path, mano_left, mano_right)]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"required local licensed asset missing: {path}")
        # Only legacy unpickling needs these names; do not alter the assets/site-packages.
        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec
        aliases = {"bool": np.bool_, "int": np.int64, "float": np.float64,
                   "complex": np.complex128, "object": np.object_,
                   "unicode": np.str_, "str": np.str_}
        for name, value in aliases.items():
            if name not in np.__dict__:
                setattr(np, name, value)
        with paths[0].open("rb") as handle:
            data = pickle.load(handle, encoding="latin1")
        for suffix, path in zip(("l", "r"), paths[1:]):
            with path.open("rb") as handle:
                hand = pickle.load(handle, encoding="latin1")
            if np.asarray(hand["hands_components"]).shape != (45, 45):
                raise ValueError("matching MANO asset has unexpected PCA dimensions")
            data["hands_components" + suffix] = hand["hands_components"]
            data["hands_mean" + suffix] = hand["hands_mean"]
        self.layer = smplx.SMPLHLayer(str(paths[0]), data_struct=Struct(**data),
                                    use_pca=False, flat_hand_mean=True, num_betas=10)
        self.layer.requires_grad_(False)
        self.asset_paths = [str(p) for p in paths]
        self.register_buffer("faces", torch.as_tensor(self.layer.faces.astype(np.int64)))
        if self.layer.parents.numel() != 52 or self.layer.v_template.shape != (6890, 3):
            raise ValueError("this baseline requires the 52-rotation, 6890-vertex SMPL-H asset")

    @property
    def parents(self):
        return self.layer.parents

    def forward(self, rotations, betas, translation):
        """B,T,52,3,3 absolute local rotations; B,10 shared shape; B,T,3 metres."""
        validate_rotation_matrices(rotations)
        batch, frames = rotations.shape[:2]
        if rotations.shape[2:] != (52, 3, 3):
            raise ValueError("expected 52 absolute rotation matrices")
        if betas.shape != (batch, 10) or translation.shape != (batch, frames, 3):
            raise ValueError("shape must be sequence-shared; translation must be per-frame")
        if not torch.isfinite(betas).all() or not torch.isfinite(translation).all():
            raise ValueError("nonfinite shape/translation")
        flat = rotations.reshape(-1, 52, 3, 3)
        output = self.layer(betas=betas[:, None].expand(-1, frames, -1).reshape(-1, 10),
                            global_orient=flat[:, :1], body_pose=flat[:, 1:22],
                            left_hand_pose=flat[:, 22:37], right_hand_pose=flat[:, 37:52],
                            transl=translation.reshape(-1, 3), pose2rot=False)
        return {"vertices": output.vertices.reshape(batch, frames, 6890, 3),
                "joints": output.joints[:, :52].reshape(batch, frames, 52, 3),
                "faces": self.faces}
