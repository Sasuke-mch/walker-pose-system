"""The declared surface observation operator, with candidate-owned T and beta."""
import torch
from torch import nn

from realtime_app.learning_flow_mesh.rotations import validate_rotation_matrices
from realtime_app.pose_app.smpl_coco_observation import validate_coco17_regressor


class SurfaceDecoder(nn.Module):
    def __init__(self, licensed_decoder, regressor):
        super().__init__()
        validate_coco17_regressor(regressor.detach().cpu().numpy())
        self.decoder = licensed_decoder
        self.register_buffer("regressor", regressor)

    def forward(self, rotations, translation, beta):
        if rotations.shape[-3:] != (52, 3, 3):
            raise ValueError("real mesh decoding requires full SMPLH52; BODY22 is partial")
        validate_rotation_matrices(rotations)
        # Each row is one complete candidate, never a batch with averaged beta.
        decoded = self.decoder(rotations[:, None], beta, translation[:, None])
        vertices = decoded["vertices"][:, 0]
        return {"vertices": vertices, "joints": decoded["joints"][:, 0],
                "coco17": torch.einsum("jv,kvc->kjc", self.regressor.to(vertices), vertices),
                "faces": decoded["faces"]}
