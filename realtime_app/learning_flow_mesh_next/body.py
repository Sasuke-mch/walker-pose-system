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
        self._refresh_observation_buffers()
        self.register_load_state_dict_post_hook(self._after_load)

    def _refresh_observation_buffers(self):
        # The observation regressor is sparse: keep every nonzero coefficient,
        # without a threshold, and skin only vertices that affect COCO17.
        layer, regressor = self.decoder.layer, self.regressor
        support = regressor.ne(0).any(0).nonzero().flatten().to(layer.v_template.device)
        self.register_buffer("observation_vertices", support, persistent=False)
        for name, value in {
            "observation_template": layer.v_template[support],
            "observation_shapedirs": layer.shapedirs[support],
            "observation_posedirs": layer.posedirs.reshape(-1, 6890, 3)[:, support].flatten(1),
            "observation_weights": layer.lbs_weights[support],
            "observation_regressor": regressor[:, support.to(regressor.device)],
            "joint_template": torch.einsum("jv,vc->jc", layer.J_regressor, layer.v_template),
            "joint_shapedirs": torch.einsum("jv,vcl->jcl", layer.J_regressor, layer.shapedirs),
        }.items():
            self.register_buffer(name, value.detach().clone(), persistent=False)
        # smplx traverses this static topology in Python. CUDA scalar parents
        # would force a device-to-host synchronization at every joint.
        self._parents_cpu = layer.parents.detach().cpu().clone()

    def _after_load(self, module, incompatible_keys):
        # Child asset buffers have now loaded too; never retain contractions
        # from the previous asset or regressor after a normal state_dict load.
        validate_coco17_regressor(self.regressor.detach().cpu().numpy())
        self._refresh_observation_buffers()

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Recompute at the new dtype as well as device (e.g. float -> double).
        self._refresh_observation_buffers()
        return self

    def coco17(self, rotations, translation, beta):
        """Same surface operator as forward, without constructing unused mesh.

        Shape-dependent rest joints still use the full 6890-vertex regressor,
        contracted once with the frozen template/shape basis. All 52 rotations
        and all pose blend shapes remain active. Derived buffers are immutable
        asset data, not cached predictions or new learned parameters.
        """
        from smplx.lbs import batch_rigid_transform, blend_shapes

        if rotations.ndim != 4 or rotations.shape[1:] != (52, 3, 3):
            raise ValueError("real mesh decoding requires full SMPLH52; BODY22 is partial")
        validate_rotation_matrices(rotations)
        batch = len(rotations)
        if beta.shape != (batch, 10) or translation.shape != (batch, 3):
            raise ValueError("one complete shape/translation per candidate required")
        if not torch.isfinite(beta).all() or not torch.isfinite(translation).all():
            raise ValueError("nonfinite shape/translation")
        shaped = self.observation_template + blend_shapes(beta, self.observation_shapedirs)
        joints = self.joint_template + blend_shapes(beta, self.joint_shapedirs)
        identity = torch.eye(3, device=rotations.device, dtype=rotations.dtype)
        pose = (rotations[:, 1:] - identity).reshape(batch, -1)
        posed = shaped + (pose @ self.observation_posedirs).reshape(batch, -1, 3)
        _, transforms = batch_rigid_transform(rotations, joints, self._parents_cpu, dtype=rotations.dtype)
        skin = (self.observation_weights @ transforms.reshape(batch, 52, 16)).reshape(batch, -1, 4, 4)
        homogeneous = torch.cat((posed, torch.ones_like(posed[..., :1])), -1)
        vertices = (skin @ homogeneous[..., None])[..., :3, 0] + translation[:, None]
        return torch.einsum("jv,kvc->kjc", self.observation_regressor.to(vertices), vertices)

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


def observation_points(surface, rotations, translation, beta):
    """Use a decoder's observation-only path, preserving analytic callables."""
    decode = getattr(surface, "coco17", None)
    if decode is not None:
        return decode(rotations, translation, beta)
    return surface(rotations, translation, beta)["coco17"]
