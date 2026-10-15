"""Load the author's PHC+ keypoint primitives and composer without Isaac Gym.

Architecture and composition follow PHC's pnn.py, amp_network_mcp_builder.py,
and humanoid_im_mcp.py. No optimizer, fine tuning, or invented policy weights.
Only load checkpoint files obtained from the author's download_data.sh.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn


def _mlp(state, prefix):
    indices = sorted(int(k[len(prefix):].split('.')[0]) for k in state
                     if k.startswith(prefix) and k.endswith('.weight'))
    if not indices or len(indices) != len(set(indices)):
        raise ValueError(f"invalid sequential checkpoint prefix: {prefix}")
    layers = []
    for i, index in enumerate(indices):
        weight, bias = state[f'{prefix}{index}.weight'], state[f'{prefix}{index}.bias']
        layer = nn.Linear(weight.shape[1], weight.shape[0])
        layer.load_state_dict({'weight': weight.float(), 'bias': bias.float()})
        layers.append(layer)
        if i < len(indices) - 1:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class FrozenPHC(nn.Module):
    def __init__(self, primitive_path: Path, composer_path: Path):
        super().__init__()
        # These are author-distributed PyTorch checkpoints including optimizer
        # metadata. This deliberately does not accept untrusted uploaded pickle.
        primitive = torch.load(primitive_path, map_location='cpu', weights_only=False)
        composer = torch.load(composer_path, map_location='cpu', weights_only=False)
        state = primitive['model']
        self.actors = nn.ModuleList([
            _mlp(state, f'a2c_network.pnn.actors.{i}.') for i in range(3)])
        if any('.u.' in k for k in state):
            raise ValueError('lateral PNN weights are not supported by this adapter')
        self.composer = _mlp(composer['model'], 'a2c_network.composer.')
        for prefix, checkpoint in (('primitive', primitive), ('composer', composer)):
            norm = checkpoint['running_mean_std']
            self.register_buffer(prefix + '_mean', norm['running_mean'].float())
            self.register_buffer(prefix + '_var', norm['running_var'].float())
        if self.primitive_mean.shape != (574,) or self.composer_mean.shape != (574,):
            raise ValueError('expected PHC+ obs_v7 574-dimensional checkpoints')
        self.requires_grad_(False)
        self.eval()
        with torch.inference_mode():
            if self(torch.zeros(1, 574))[0].shape != (1, 69):
                raise ValueError('expected 69 PHC SMPL control actions')

    @staticmethod
    def normalize(observation, mean, variance):
        return ((observation - mean) / torch.sqrt(variance + 1e-5)).clamp(-5, 5)

    def forward(self, observation):
        if observation.shape[-1] != 574 or not torch.isfinite(observation).all():
            raise ValueError('finite PHC observation with dimension 574 required')
        # Author's eval command specifies ending_act=False and has_softmax=False.
        weights = self.composer(self.normalize(
            observation, self.composer_mean, self.composer_var)).clamp(-1, 1)
        normalized = self.normalize(observation, self.primitive_mean, self.primitive_var)
        experts = torch.stack([actor(normalized) for actor in self.actors], dim=1)
        actions = (weights[..., None] * experts).sum(dim=1).clamp(-1, 1)
        return actions, weights

    def act(self, observation):
        with torch.inference_mode():
            action, weights = self(torch.as_tensor(np.asarray(observation)[None], dtype=torch.float32))
        return action[0].numpy(), weights[0].numpy()


def observation_v7(position, rotation, velocity, angular_velocity,
                   reference_position, reference_velocity):
    """Author's self_obs_v1 + obs_v7, upright start, all 24 PHC rigid bodies.

    Rotations are world rotation matrices. This map uses native body origins,
    never COCO regressor points. Ordering is the XML/SMPL_MUJOCO_NAMES order.
    """
    arrays = (position, velocity, angular_velocity, reference_position, reference_velocity)
    if any(np.shape(a) != (24, 3) for a in arrays) or np.shape(rotation) != (24, 3, 3):
        raise ValueError('expected 24 ordered bodies and world rotations')
    if any(not np.isfinite(a).all() for a in (*arrays, rotation)):
        raise ValueError('unavailable bodies cannot become a PHC observation')
    yaw = np.arctan2(rotation[0, 1, 0], rotation[0, 0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    inverse_heading = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    local_rotation = inverse_heading @ rotation
    tangent_normal = np.concatenate([local_rotation[:, :, 0], local_rotation[:, :, 2]], axis=1)
    self_observation = np.r_[position[0, 2],
        ((position - position[0]) @ inverse_heading.T)[1:].ravel(),
        tangent_normal.ravel(), (velocity @ inverse_heading.T).ravel(),
        (angular_velocity @ inverse_heading.T).ravel()]
    task_observation = np.concatenate([
        ((reference_position - position) @ inverse_heading.T).ravel(),
        ((reference_velocity - velocity) @ inverse_heading.T).ravel(),
        ((reference_position - position[0]) @ inverse_heading.T).ravel()])
    return np.r_[self_observation, task_observation].astype(np.float32)


def recovery_reference(position, velocity, reference_position, reference_velocity,
                       close_distance=.25, far_distance=3.):
    """Author humanoid_im.py obs_v7 zero_out_far, without changing saved targets."""
    target, target_velocity = reference_position.copy(), reference_velocity.copy()
    distance = np.linalg.norm(target[0]-position[0])
    if distance > close_distance:
        target[1:] = position[1:]
        target_velocity[:] = velocity
    if distance > far_distance:
        target[0] = position[0] + (target[0]-position[0])/distance*far_distance
    return target, target_velocity
