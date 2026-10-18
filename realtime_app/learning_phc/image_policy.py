"""Direct dual-fisheye policy inputs and exact fixed-shape observation surface.

Mechanism references: PhysHMR (pixel rays / teacher distillation), Mimic2DM
(reprojection-conditioned control). This is an isolated adaptation, not either
paper's released model. No per-frame 3D target enters inference.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .policy import observation_v7
from .reference import SMPLHHeelSurface
from pose_app.fisheye_camera import fisheye_project_numpy


def pixel_rays(pixels, intrinsic, distortion):
    """Inverse OpenCV fisheye on the front hemisphere; invalid rays stay NaN."""
    xy = (np.asarray(pixels)-intrinsic[:2, 2])/np.diag(intrinsic)[:2]
    radius = np.linalg.norm(xy, axis=-1)
    theta = radius.copy()
    k = np.asarray(distortion).ravel()[:4]
    for _ in range(12):
        t2 = theta*theta
        f = theta*(1+k[0]*t2+k[1]*t2**2+k[2]*t2**3+k[3]*t2**4)-radius
        derivative = 1+3*k[0]*t2+5*k[1]*t2**2+7*k[2]*t2**3+9*k[3]*t2**4
        theta -= f/np.maximum(derivative, 1e-8)
    ray = np.concatenate([xy*(np.sin(theta)/np.maximum(radius, 1e-12))[..., None],
                          np.cos(theta)[..., None]], axis=-1)
    ray[radius < 1e-12] = [0, 0, 1]
    residual = theta*(1+k[0]*theta**2+k[1]*theta**4+k[2]*theta**6+k[3]*theta**8)-radius
    ray[(theta < 0) | (theta >= np.pi/2) | (np.abs(residual) > 1e-7)] = np.nan
    return ray


class ImageObservation:
    """574 inputs = PHC self-state358 + 24 ray slots *9, no pose references.

    Each slot: ray-to-model displacement3 /0.3m, heading-local ray3,
    confidence1, raw-pixel residual2 /100px. Right knee is never an input or
    reward observation. Missing rays have all nine entries zero.
    """
    def __init__(self, packet):
        forbidden = {'reference_position_m', 'reference_rotation', 'reference_velocity_m_s',
                     'feedback_coco_local_m', 'position_m', 'rotation_world', 'action'}
        if forbidden.intersection(packet):
            raise ValueError('privileged per-frame3D/action arrays forbidden in inference packet')
        self.data = packet
        self.surface = SMPLHHeelSurface(packet)

    def points(self, position, rotation):
        vertices = self.surface.points(position[0], rotation)
        return self.data['surface_coco_regressor']@vertices

    def project(self, points, index):
        d = self.data
        left = (points-d['camera_translation_m'][index])@d['camera_rotation'][index]
        right = left@d['R01'].T+d['T01_m']
        with np.errstate(divide='ignore', invalid='ignore'):
            pixels = np.stack([fisheye_project_numpy(left, d['K0'], d['D0']),
                               fisheye_project_numpy(right, d['K1'], d['D1'])])
        pixels[np.stack([left[:, 2], right[:, 2]]) <= 1e-5] = np.nan
        return pixels

    def valid(self, index):
        observed = self.data['pixels'][index]
        valid = (np.isfinite(observed).all(-1) & (observed[..., 2] >= .2)
                 & (observed[..., 2] <= 1) & (observed[..., 0] >= 0)
                 & (observed[..., 0] < 1920) & (observed[..., 1] >= 0)
                 & (observed[..., 1] < 1080))
        valid[:, 9] = False
        return valid

    def features(self, index, position, rotation, velocity, angular):
        d = self.data
        points = self.points(position, rotation)
        projected = self.project(points, index)
        observed = d['pixels'][index]
        r = d['camera_rotation'][index]
        t = d['camera_translation_m'][index]
        camera_rotations = (r, r@d['R01'].T)
        centers = (t, t-r@d['R01'].T@d['T01_m'])
        yaw = np.arctan2(rotation[0, 1, 0], rotation[0, 0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        heading = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        slots = np.zeros((2, 12, 9))
        valid = self.valid(index)
        for side in range(2):
            ray = pixel_rays(observed[side, :, :2], d['K'+str(side)], d['D'+str(side)])
            ray = ray@camera_rotations[side].T
            relative = points-centers[side]
            depth = np.sum(relative*ray, axis=-1)
            displacement = centers[side]+depth[:, None]*ray-points
            slots[side] = np.concatenate([displacement@heading.T/.3, ray@heading.T,
                observed[side, :, 2:3], (observed[side, :, :2]-projected[side])/100], axis=-1)
            good = valid[side] & np.isfinite(slots[side]).all(-1)
            slots[side, ~good] = 0
        state = observation_v7(position, rotation, velocity, angular, position, velocity)[:358]
        return np.r_[state, np.clip(slots, -5, 5).ravel()].astype(np.float32)

    def reward(self, index, position, rotation, action):
        valid = self.valid(index)
        error = np.linalg.norm(self.project(self.points(position, rotation), index)
                               - self.data['pixels'][index, :, :, :2], axis=-1)
        # A missing/behind-camera projection receives zero, never a dropped term.
        score = np.zeros((2, 12))
        finite = valid & np.isfinite(error)
        score[finite] = np.exp(-np.minimum((error[finite]/50)**2, 100))
        confidence = np.where(valid, self.data['pixels'][index, :, :, 2], 0)
        image_reward = float(np.sum(score*confidence)/max(confidence.sum(), 1e-12))
        fallen = bool(position[0, 2] < .3)
        return image_reward+.1-float(fallen)-.01*float(np.mean(action**2)), image_reward, fallen


class ImageActorCritic(nn.Module):
    """One standard PPO actor/critic; teacher is absent at deployment."""
    def __init__(self, state_mean=None, state_var=None):
        super().__init__()
        self.register_buffer('state_mean', torch.zeros(358) if state_mean is None else state_mean[:358].clone())
        self.register_buffer('state_var', torch.ones(358) if state_var is None else state_var[:358].clone())
        def mlp(output):
            return nn.Sequential(nn.Linear(574, 256), nn.SiLU(), nn.Linear(256, 256),
                                 nn.SiLU(), nn.Linear(256, output))
        self.actor = mlp(69)
        self.critic = mlp(1)
        self.log_std = nn.Parameter(torch.full((69,), -3.7))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    def normalize(self, observation):
        state = ((observation[..., :358]-self.state_mean)/torch.sqrt(self.state_var+1e-5)).clamp(-5, 5)
        return torch.cat([state, observation[..., 358:].clamp(-5, 5)], -1)

    def distribution(self, observation):
        x = self.normalize(observation)
        normal = torch.distributions.Normal(self.actor(x), self.log_std.clamp(-5, -2).exp())
        return normal, self.critic(x).squeeze(-1)

    @staticmethod
    def log_probability(normal, raw):
        action = raw.tanh()
        return (normal.log_prob(raw)-torch.log(1-action**2+1e-6)).sum(-1)

    def action(self, observation):
        return self.actor(self.normalize(observation)).tanh()


def prepare(root, output):
    """Export deployment packet separately from prefix-only teacher labels."""
    from .analyze import load_model
    output.mkdir(exist_ok=False)
    with np.load(root/'corrections_v1/reference_exact_heel.npz') as z:
        reference = {k: z[k] for k in z.files}
    packet = {k: reference[k] for k in ('pair_id', 'timestamp_s', 'camera_rotation',
        'camera_translation_m', 'stage', 'K0', 'D0', 'K1', 'D1', 'R01', 'T01_m')}
    packet.update(pixels=reference['feedback_pixels'],
        initial_position_m=reference['reference_position_m'][0],
        initial_rotation=reference['reference_rotation'][0])
    model = load_model()
    regressor = np.load(root.parents[2]/'models/smpl/J_regressor_coco.npy')[5:17]
    vertices = np.flatnonzero(np.any(regressor != 0, axis=0))
    rest = model.v_template+torch.einsum('bl,vcl->bvc', torch.tensor(reference['betas']), model.shapedirs)[0]
    packet.update(heel_rest_joints_m=reference['heel_rest_joints_m'],
        heel_rest_vertices_m=rest[vertices].detach().numpy(),
        heel_skin_weights=model.lbs_weights[vertices].detach().numpy(),
        heel_pose_dirs=model.posedirs.reshape(-1, 6890, 3)[:, vertices].detach().numpy(),
        heel_smpl_parents=reference['heel_smpl_parents'],
        heel_hand_rotation=reference['heel_hand_rotation'], surface_coco_regressor=regressor[:, vertices],
        surface_vertex_ids=vertices, betas=reference['betas'])
    np.savez_compressed(output/'inference_packet.npz', **packet)
    # All labels / privileged reference frames are physically restricted to0..255.
    teacher_reference = {k: reference[k][:256] for k in ('reference_position_m',
        'reference_rotation', 'reference_velocity_m_s')}
    with np.load(root/'corrections_v1/matched_body/rollout.npz') as z:
        teacher = {k: z[k][:256] for k in ('position_m', 'rotation_world', 'velocity_m_s',
                   'angular_velocity_rad_s', 'action')}
    np.savez_compressed(output/'teacher_train_only.npz', **teacher, **teacher_reference)
    (output/'protocol.json').write_text(json.dumps(dict(
        scope='single-video feasibility, no independent3D/generalization claim',
        sources=['https://arxiv.org/abs/2510.02566','https://jiann-li.github.io/mimic2dm/'],
        teacher='frozen official PHC, source-beta matched body, assumed wrist support',
        inference='current two raw-fisheye COCO observations, camera conditioning, self-state; no future3D reference',
        train_frames=[0,255], validation_frames=[256,319], test_frames=[320,447],
        train_supervision='PHC pseudo-actions, not human GT; teacher reference only in train packet',
        right_knee14='excluded from actor observation/reward; retained in final evaluation',
        body='fixed beta and hands pre-calibrated upstream; conditional Stage camera trajectory, not independent world truth',
        observations='exact100-vertex reduced SMPL-H LBS and COCO regressor; no source pose attachment proxy',
        action='absolute69D PD target, original action scale, no pose postprocessing',
        schedule=dict(seed=0,bc_steps=600,dagger_rounds=2,dagger_steps=2048,dagger_fit_steps=200,
                      ppo_updates=64,ppo_horizon=128,ppo_epochs=4),
        gates='finite448,0falls, full10752 median/P95 improve5% against60.55/159.30 and25px fraction nondecreasing; validation selection before inspecting heldout',
        stopping='one declared seed/schedule; no test-driven gain tuning or runtime fallback to teacher',
        source=str(root.resolve()),surface_vertices=len(vertices)),ensure_ascii=False,indent=2),encoding='utf-8')
    print('Exported inference packet, train-only teacher labels; exact vertices', len(vertices), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.output.resolve())
