"""Conditional parameter transport with causal observations and frame-local velocity."""
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .rotations import matrix_to_six, six_to_matrix


@dataclass
class ModelConfig:
    observation_joints: int = 22
    image_feature_dim: int = 32
    width: int = 64
    heads: int = 4
    layers: int = 2
    use_graph: bool = True
    use_history: bool = True
    rotation_scale: float = 0.25
    root_offset: tuple = (0.0, 0.0, 3.0)

    def __post_init__(self):
        if self.width < 4 or self.width % self.heads or self.width % 2:
            raise ValueError("width must be even and divisible by heads")
        if min(self.observation_joints, self.image_feature_dim, self.layers) < 1:
            raise ValueError("dimensions/layers must be positive")
        if self.rotation_scale <= 0:
            raise ValueError("rotation_scale must be positive")

    def to_dict(self):
        return asdict(self)


class ImageEncoder(nn.Module):
    """Small randomly initialized RGB branch; never called pretrained."""
    def __init__(self, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1),
                                 nn.GroupNorm(4, 16), nn.GELU(),
                                 nn.Conv2d(16, out_dim, 3, stride=2, padding=1),
                                 nn.GELU(), nn.AdaptiveAvgPool2d(1))

    def forward(self, images):
        return self.net(images).flatten(1)


class ConditionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.width
        # xy, confidence, ray origin, ray direction, geometry_present
        self.observation = nn.Sequential(nn.Linear(10, d), nn.GELU(), nn.Linear(d, d))
        self.obs_identity = nn.Parameter(torch.randn(cfg.observation_joints, d) * 0.02)
        self.pose_queries = nn.Parameter(torch.randn(52, d) * 0.02)
        self.null_token = nn.Parameter(torch.zeros(1, d))
        self.image_encoder = ImageEncoder(cfg.image_feature_dim)
        self.image_projection = nn.Linear(cfg.image_feature_dim, d)
        self.cross = nn.MultiheadAttention(d, cfg.heads, batch_first=True, dropout=0)
        self.norm = nn.LayerNorm(d)
        self.history = nn.GRU(d, d, batch_first=True)
        self.shape_head = nn.Linear(d, 10)
        self.translation_head = nn.Linear(d, 3)
        self.register_buffer("root_offset", torch.tensor(cfg.root_offset, dtype=torch.float32))

    def forward(self, batch, hidden=None, fixed_betas=None):
        xy = batch["xy"]
        b, t, v, j, _ = xy.shape
        if j != self.cfg.observation_joints:
            raise ValueError("observation topology does not match config")
        observed = batch["observation_mask"] & batch["view_mask"][..., None]
        if observed.dtype != torch.bool:
            raise ValueError("observation/view masks must be boolean")
        score = batch["confidence"]
        if not torch.isfinite(xy[observed]).all() or not torch.isfinite(score[observed]).all():
            raise ValueError("present observations must be finite")
        if ((score[observed] < 0) | (score[observed] > 1)).any():
            raise ValueError("confidence must be in [0,1]")
        geometry = observed & batch["geometry_mask"]
        origins, rays = batch["ray_origins"], batch["ray_directions"]
        if not torch.isfinite(origins[geometry]).all() or not torch.isfinite(rays[geometry]).all():
            raise ValueError("present geometry must be finite")
        if geometry.any() and (rays[geometry].norm(dim=-1) - 1).abs().max() > 1e-3:
            raise ValueError("rays must be unit vectors in the declared reference frame")
        inputs = torch.cat((torch.where(observed[..., None], xy, 0),
                            torch.where(observed, score, 0)[..., None],
                            torch.where(geometry[..., None], origins, 0),
                            torch.where(geometry[..., None], rays, 0),
                            geometry[..., None].to(xy.dtype)), dim=-1)
        tokens = self.observation(inputs) + self.obs_identity[None, None, None]
        tokens = tokens.reshape(b * t, v * j, -1)
        valid_tokens = observed.reshape(b * t, v * j)
        image_valid = batch.get("image_mask", torch.zeros_like(batch["view_mask"]))
        image_valid = image_valid & batch["view_mask"]
        if "images" in batch and "image_features" in batch:
            raise ValueError("choose RGB or supplied features, not both")
        if "images" in batch:
            imgs = batch["images"]
            if not torch.isfinite(imgs[image_valid]).all():
                raise ValueError("present images must be finite")
            if ((imgs[image_valid] < 0) | (imgs[image_valid] > 1)).any():
                raise ValueError("RGB images must be float in [0,1]")
            imgs = torch.where(image_valid[..., None, None, None], imgs, 0)
            features = self.image_encoder(imgs.reshape(b * t * v, *imgs.shape[3:]))
            features = features.reshape(b, t, v, -1)
        elif "image_features" in batch:
            features = batch["image_features"]
            if not torch.isfinite(features[image_valid]).all():
                raise ValueError("present image features must be finite")
            features = torch.where(image_valid[..., None], features, 0)
        else:
            if image_valid.any():
                raise ValueError("image_mask declares missing images/features present")
            features = xy.new_zeros(b, t, v, self.cfg.image_feature_dim)
        image_tokens = self.image_projection(features).reshape(b * t, v, -1)
        null = self.null_token.expand(b * t, 1, -1)
        tokens = torch.cat((tokens, image_tokens, null), dim=1)
        valid_tokens = torch.cat((valid_tokens, image_valid.reshape(b * t, v),
                                  torch.ones(b * t, 1, device=xy.device, dtype=torch.bool)), dim=1)
        query = self.pose_queries.expand(b * t, -1, -1)
        attended, _ = self.cross(query, tokens, tokens, key_padding_mask=~valid_tokens,
                                 need_weights=False)
        condition = self.norm(query + attended).reshape(b, t, 52, -1)
        pooled = condition.mean(dim=2)
        if self.cfg.use_history:
            global_condition, hidden_out = self.history(pooled, hidden)
        else:
            global_condition, hidden_out = pooled, None
        condition = condition + global_condition[:, :, None]
        # Freeze shape from the initial frame only; caller carries it between windows.
        betas = self.shape_head(global_condition[:, 0]) if fixed_betas is None else fixed_betas
        shape_available = observed[:, 0].any(dim=(-1, -2)) | image_valid[:, 0].any(dim=-1)
        if fixed_betas is not None:
            shape_available = torch.ones(b, device=xy.device, dtype=torch.bool)
        return {"condition": condition, "betas": betas,
                "translation": self.translation_head(global_condition) + self.root_offset,
                "shape_available": shape_available, "hidden": hidden_out,
                "frame_observed": observed.any(dim=(-1, -2)) | image_valid.any(dim=-1)}


class SpatialBlock(nn.Module):
    def __init__(self, cfg, adjacency):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.width)
        self.attention = nn.MultiheadAttention(cfg.width, cfg.heads, batch_first=True, dropout=0)
        self.graph = nn.Linear(cfg.width, cfg.width) if cfg.use_graph else None
        self.register_buffer("adjacency", adjacency)
        self.mlp = nn.Sequential(nn.LayerNorm(cfg.width), nn.Linear(cfg.width, cfg.width * 2),
                                 nn.GELU(), nn.Linear(cfg.width * 2, cfg.width))

    def forward(self, x):
        normalized = self.norm(x)
        attn, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        if self.graph is not None:
            attn = attn + self.graph(torch.einsum("ij,bjd->bid", self.adjacency, normalized))
        x = x + attn
        return x + self.mlp(x)


class FlowMesh(nn.Module):
    def __init__(self, cfg, parents):
        super().__init__()
        self.cfg = cfg
        if len(parents) != 52 or int(parents[0]) != -1:
            raise ValueError("need actual SMPL-H 52-node parent tree")
        adjacency = torch.eye(52)
        for i in range(1, 52):
            p = int(parents[i])
            if not 0 <= p < i:
                raise ValueError("parent tree must be topologically ordered")
            adjacency[i, p] = adjacency[p, i] = 1
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True)
        self.register_buffer("parents", torch.as_tensor(parents).long().clone())
        self.register_buffer("identity_six", matrix_to_six(torch.eye(3)))
        self.encoder = ConditionEncoder(cfg)
        self.state_projection = nn.Linear(6, cfg.width)
        self.time_projection = nn.Sequential(nn.Linear(cfg.width, cfg.width), nn.GELU(),
                                             nn.Linear(cfg.width, cfg.width))
        self.blocks = nn.ModuleList(SpatialBlock(cfg, adjacency) for _ in range(cfg.layers))
        self.velocity_head = nn.Sequential(nn.LayerNorm(cfg.width), nn.Linear(cfg.width, 6))

    def encode_rotations(self, rotations):
        return (matrix_to_six(rotations) - self.identity_six) / self.cfg.rotation_scale

    def decode_state(self, state):
        return six_to_matrix(self.identity_six + state * self.cfg.rotation_scale)

    def velocity(self, state, time, condition):
        b, t = state.shape[:2]
        freq = torch.exp(torch.arange(self.cfg.width // 2, device=state.device,
                                     dtype=state.dtype) * (-math.log(10000) / (self.cfg.width // 2)))
        angle = time[..., None] * freq * 2 * math.pi
        time_emb = self.time_projection(torch.cat((angle.sin(), angle.cos()), dim=-1))
        x = condition + self.state_projection(state) + time_emb[:, :, None]
        x = x.reshape(b * t, 52, -1)
        for block in self.blocks:
            x = block(x)
        return self.velocity_head(x).reshape(b, t, 52, 6)

    def regression(self, encoded):
        # Same spatial decoder/capacity as FM; only state, time and objective differ.
        condition = encoded["condition"]
        b, t = condition.shape[:2]
        return self.velocity(condition.new_zeros(b, t, 52, 6),
                             condition.new_ones(b, t), condition)

    def transport(self, encoded, steps=8, seed=0, initial=None):
        """Euler transport. Same noise across frames prevents artificial seed jitter.

        No no_grad decorator: endpoint losses may backpropagate through this ODE.
        """
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        condition = encoded["condition"]
        b, t = condition.shape[:2]
        if initial is None:
            generator = torch.Generator(device=condition.device).manual_seed(seed)
            initial = torch.randn(b, 1, 52, 6, generator=generator, device=condition.device,
                                  dtype=condition.dtype).expand(-1, t, -1, -1).clone()
        if initial.shape != (b, t, 52, 6) or not torch.isfinite(initial).all():
            raise ValueError("initial transport state has wrong shape/nonfinite values")
        state = initial
        for step in range(steps):
            time = condition.new_full((b, t), step / steps)
            state = state + self.velocity(state, time, condition) / steps
        rotations, valid = self.decode_state(state)
        return {"state": state, "rotations": rotations, "rotation_valid": valid,
                "betas": encoded["betas"], "translation": encoded["translation"],
                "shape_available": encoded["shape_available"],
                "frame_observed": encoded["frame_observed"], "nfe": steps}
