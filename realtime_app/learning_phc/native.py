"""Original Isaac Gym Preview4 / PhysX CPU backend for frozen PHC+.

Requires the separate WSL Python3.8 environment. Import Isaac Gym before torch.
No gymtorch extension/GPU pipeline is needed for single-human numpy state IO.
"""

from isaacgym import gymapi

import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import joblib
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .contract import NAMES, COLLISION_FILTERS
from .policy import FrozenPHC, observation_v7, recovery_reference

ROOT = Path(__file__).resolve().parents[2]


class PhysXBody:
    def __init__(self, xml, root_position, root_rotation):
        self.gym = gymapi.acquire_gym()
        params = gymapi.SimParams()
        params.dt, params.substeps = 1/60, 2
        params.up_axis = gymapi.UP_AXIS_Z
        params.gravity = gymapi.Vec3(0, 0, -9.81)
        params.use_gpu_pipeline = False
        params.physx.use_gpu = False
        params.physx.num_threads = 2
        params.physx.solver_type = 1
        params.physx.num_position_iterations = 4
        params.physx.num_velocity_iterations = 0
        params.physx.contact_offset = .02
        params.physx.rest_offset = 0.
        params.physx.bounce_threshold_velocity = .2
        params.physx.max_depenetration_velocity = 10.
        self.params = params
        self.sim = self.gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
        if self.sim is None:
            raise RuntimeError('Isaac Gym could not create the native PhysX CPU simulator')
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0, 0, 1)
        plane.static_friction = plane.dynamic_friction = 1.
        plane.restitution = 0.
        self.gym.add_ground(self.sim, plane)
        self.env = self.gym.create_env(self.sim, gymapi.Vec3(-5, -5, 0), gymapi.Vec3(5, 5, 5), 1)
        options = gymapi.AssetOptions()
        options.angular_damping = .01
        options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset = self.gym.load_asset(self.sim, str(xml.parent), xml.name, options)
        if asset is None:
            raise RuntimeError('native PhysX failed to load the author humanoid XML')
        names = self.gym.get_asset_rigid_body_names(asset)
        if tuple(names) != NAMES:
            raise ValueError('author asset body ordering differs from the PHC checkpoint contract')
        self.dof_names = self.gym.get_asset_dof_names(asset)
        expected = [f'{n}_{axis}' for n in NAMES[1:] for axis in 'xyz']
        if self.dof_names != expected:
            raise ValueError('author asset DOF ordering differs from the PHC checkpoint contract')
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*root_position)
        pose.r = gymapi.Quat(*Rotation.from_matrix(root_rotation).as_quat())
        self.actor = self.gym.create_actor(self.env, asset, pose, 'humanoid', 0, 0)
        props = self.gym.get_actor_rigid_shape_properties(self.env, self.actor)
        if len(props) != 24:
            raise ValueError('expected one author collision shape per body')
        for prop, mask in zip(props, COLLISION_FILTERS):
            prop.filter = mask
        self.gym.set_actor_rigid_shape_properties(self.env, self.actor, props)
        dofs = self.gym.get_actor_dof_properties(self.env, self.actor)
        dofs['driveMode'] = gymapi.DOF_MODE_POS
        self.gym.set_actor_dof_properties(self.env, self.actor, dofs)
        self.scale = np.empty(69)
        for i in range(23):
            scale = min(1.2 * max(abs(dofs['lower'][3*i:3*i+3]).max(),
                                  abs(dofs['upper'][3*i:3*i+3]).max()), np.pi)
            self.scale[3*i:3*i+3] = scale if scale else np.pi
        self.scale[self.dof_names.index('L_Knee_y')] = 5
        self.scale[self.dof_names.index('R_Knee_y')] = 5
        tree = ET.parse(xml)
        parent_lookup = {child.get('name'): parent.get('name') for parent in tree.findall('.//body')
                         for child in parent.findall('body')}
        self.parents = [-1] + [NAMES.index(parent_lookup[name]) for name in NAMES[1:]]
        self.gym.prepare_sim(self.sim)
        self.properties = dict(stiffness=dofs['stiffness'].tolist(), damping=dofs['damping'].tolist(),
                               effort=dofs['effort'].tolist(), action_scale=self.scale.tolist())
        self.wrist_handles = [self.gym.find_actor_rigid_body_handle(self.env, self.actor, name)
                              for name in ('L_Wrist', 'R_Wrist')]
        self.max_support_force = 0.
        self.support_saturation_substeps = 0

    def reset_joints(self, rotations):
        local = np.stack([rotations[self.parents[i]].T @ rotations[i] for i in range(1, 24)])
        dof = np.zeros(69, dtype=gymapi.DofState.dtype)
        dof['pos'] = Rotation.from_matrix(local).as_rotvec().ravel()
        self.gym.set_actor_dof_states(self.env, self.actor, dof, gymapi.STATE_ALL)
        self.gym.set_actor_dof_position_targets(self.env, self.actor, dof['pos'].copy())

    def state(self):
        state = self.gym.get_actor_rigid_body_states(self.env, self.actor, gymapi.STATE_ALL)
        def xyz(value):
            return np.column_stack([value[k] for k in 'xyz']).astype(float)
        position = xyz(state['pose']['p'])
        quaternion = np.column_stack([state['pose']['r'][k] for k in 'xyzw'])
        return position, Rotation.from_quat(quaternion).as_matrix(), xyz(state['vel']['linear']), xyz(state['vel']['angular'])

    def advance(self, action, dt, wrist_target=None):
        if np.shape(action) != (69,) or not np.isfinite(action).all() or not np.isfinite(dt) or dt <= 0:
            raise ValueError('finite69D action and positive dt required')
        target = (np.clip(action, -1, 1) * self.scale).astype(np.float32)
        self.gym.set_actor_dof_position_targets(self.env, self.actor, target)
        remaining = float(dt)
        self.max_support_force = 0.
        self.support_saturation_substeps = 0
        while remaining > 1e-10:
            h = min(1/60, remaining)
            if abs(self.params.dt - h) > 1e-10:
                self.params.dt = h
                self.gym.set_sim_params(self.sim, self.params)
            if wrist_target is not None:
                p, _, v, _ = self.state()
                for side, index in enumerate((17, 22)):
                    force = 900*(wrist_target[side]-p[index])-45*v[index]
                    magnitude = np.linalg.norm(force)
                    self.support_saturation_substeps += int(magnitude > 300)
                    force *= min(1., 300/max(magnitude, 1e-12))
                    self.max_support_force = max(self.max_support_force, float(np.linalg.norm(force)))
                    self.gym.apply_body_force_at_pos(self.env, self.wrist_handles[side],
                        gymapi.Vec3(*force), gymapi.Vec3(*p[index]), gymapi.ENV_SPACE)
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            remaining -= h
        return True

    def close(self):
        self.gym.destroy_sim(self.sim)


def replay_reference(args, policy):
    with np.load(args.reference, allow_pickle=False) as z:
        reference = {k: z[k] for k in ('timestamp_s', 'pair_id', 'reference_position_m',
                      'reference_rotation', 'reference_velocity_m_s', 'available', 'segment_id',
                      'camera_rotation', 'camera_translation_m')}
    count = len(reference['timestamp_s'])
    if reference['pair_id'].tolist() != list(range(448)):
        raise ValueError('all448 reference IDs are required')
    if not reference['available'][0]:
        raise ValueError('frame0 unavailable; cannot initialize from a future frame')
    body = PhysXBody(args.xml, reference['reference_position_m'][0, 0],
                     reference['reference_rotation'][0, 0])
    try:
        body.reset_joints(reference['reference_rotation'][0])
        positions = np.full((count, 24, 3), np.nan)
        rotations = np.full((count, 24, 3, 3), np.nan)
        velocity = np.full_like(positions, np.nan)
        angular = np.full_like(positions, np.nan)
        error = np.full((count, 24), np.nan)
        actions = np.full((count, 69), np.nan)
        timings = np.zeros(count)
        support_force = np.zeros(count)
        saturation = np.zeros(count, int)
        wrist_camera = ((reference['reference_position_m'][0, [17, 22]]-
                         reference['camera_translation_m'][0]) @ reference['camera_rotation'][0])
        available = np.zeros(count, bool)
        statuses = []
        failed = False
        with (args.output / 'trajectory.jsonl').open('w', encoding='utf-8') as stream:
            for i in range(count):
                start = time.perf_counter()
                reason = 'tracked_reference'
                if not reference['available'][i]:
                    failed, reason = True, 'source_reference_unavailable'
                if failed:
                    reason = reason if reason != 'tracked_reference' else 'unavailable_after_failure'
                else:
                    if i:
                        p, r, v, w = body.state()
                        target, target_velocity = reference['reference_position_m'][i], reference['reference_velocity_m_s'][i]
                        if not args.disable_far_recovery:
                            target, target_velocity = recovery_reference(p, v, target, target_velocity)
                        observation = observation_v7(p, r, v, w, target, target_velocity)
                        action, _ = policy.act(observation)
                        wrist_target = (wrist_camera @ reference['camera_rotation'][i].T +
                                        reference['camera_translation_m'][i]) if args.wrist_support else None
                        body.advance(action, reference['timestamp_s'][i]-reference['timestamp_s'][i-1], wrist_target)
                        support_force[i], saturation[i] = body.max_support_force, body.support_saturation_substeps
                        actions[i] = action
                    p, r, v, w = body.state()
                    if not all(np.isfinite(a).all() for a in (p, r, v, w)):
                        failed, reason = True, 'nonfinite_simulation'
                    else:
                        positions[i], rotations[i], velocity[i], angular[i] = p, r, v, w
                        error[i] = np.linalg.norm(p-reference['reference_position_m'][i], axis=-1)
                        available[i] = True
                        if i == 0:
                            reason = 'initialized_from_frame0'
                        elif p[0, 2] < .3:
                            reason = 'fallen_finite'
                        elif np.mean(error[i]) > .25:
                            reason = 'poor_reference_tracking'
                timings[i] = (time.perf_counter()-start)*1000
                statuses.append(reason)
                row = dict(pair_id=i, timestamp_s=float(reference['timestamp_s'][i]),
                           status=reason, available=bool(available[i]),
                           reference_mean_error_m=float(np.mean(error[i])) if available[i] else None,
                           pelvis_height_m=float(positions[i, 0, 2]) if available[i] else None,
                           assumed_wrist_support=bool(args.wrist_support),
                           wrist_force_max_n=float(support_force[i]),
                           source_segment_id=int(reference['segment_id'][i]),
                           policy_reinitialized=False, world_pose_certified=False)
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                if i % 64 == 0:
                    print('frame', i, reason, 'error', row['reference_mean_error_m'], flush=True)
        mean_error = np.mean(error, axis=1)
        successful = available & (mean_error <= .25) & (positions[:, 0, 2] >= .3)
        np.savez_compressed(args.output/'rollout.npz', pair_id=reference['pair_id'],
            timestamp_s=reference['timestamp_s'], position_m=positions, rotation_world=rotations,
            velocity_m_s=velocity, angular_velocity_rad_s=angular, error_m=error,
            reference_position_m=reference['reference_position_m'], action=actions,
            step_ms=timings, available=available, tracking_within25cm=successful,
            status=np.asarray(statuses))
        np.savez_compressed(args.output/'support_diagnostic.npz', maximum_force_n=support_force,
                            saturated_side_substeps=saturation)
        summary = dict(status='complete448_native_replay;research_candidate',
            backend='IsaacGym Preview4 PhysX CPU', frames=count, finite_frames=int(available.sum()),
            full_sequence_reference_tracking_fraction=float(successful.mean()),
            joint_error_mean_m=float(np.nanmean(error)), joint_error_p95_m=float(np.nanpercentile(error,95)),
            pelvis_min_height_m=float(np.nanmin(positions[:,0,2])),
            fallen_finite_frames=int(np.sum(available & (positions[:,0,2]<.3))),
            step_median_ms=float(np.median(timings[1:])), step_p95_ms=float(np.percentile(timings[1:],95)),
            weights_frozen=True, external_stabilizing_force=bool(args.wrist_support), resets_after_frame0=0,
            assumed_wrist_support=bool(args.wrist_support), support_max_n=float(support_force.max()),
            support_saturated_side_substeps=int(saturation.sum()), author_far_recovery=not args.disable_far_recovery,
            reference=str(args.reference.resolve()), primitive=str(args.primitive.resolve()),
            composer=str(args.composer.resolve()), xml=str(args.xml.resolve()),
            time='actual capture dt;one policy update per incoming frame, not original uniform30Hz',
            conditioning='at observation_i arrival, advance previous state to time_i using current reference_i; no reference_i+1',
            contact='author ground/self collision; optional assumed camera-rigid wrists K900,D45,300N; no added heel hard anchor',
            scope='3D-reference imitation,not raw2D-driven RL or independent human3D accuracy',
            properties=body.properties)
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps({k:v for k,v in summary.items() if k != 'properties'}), flush=True)
    finally:
        body.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--reference', type=Path, help='prepared same-source causal448 reference; omission runs standing')
    parser.add_argument('--wrist-support', action='store_true', help='explicit assumed camera-rigid wrist support contrast')
    parser.add_argument('--disable-far-recovery', action='store_true', help='reproduce the initial adapter without author zero_out_far')
    parser.add_argument('--primitive', type=Path, default=ROOT / 'models/phc/phc_kp_2.pth')
    parser.add_argument('--composer', type=Path, default=ROOT / 'models/phc/phc_comp_kp_2.pth')
    parser.add_argument('--xml', type=Path, default=ROOT / 'third_party/PHC/phc/data/assets/mjcf/smpl_0_humanoid.xml')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    policy = FrozenPHC(args.primitive, args.composer)
    if args.reference is not None:
        replay_reference(args, policy)
        return
    motion = next(iter(joblib.load(ROOT / 'models/phc/standing.pkl').values()))
    reference_rotation = Rotation.from_quat(motion['pose_quat_global'][0]).as_matrix()
    body = PhysXBody(args.xml, np.asarray(motion['root_trans_offset'][0]), reference_rotation[0])
    try:
        body.reset_joints(reference_rotation)
        reference, rotation, _, _ = body.state()
        matrix_error = float(np.max(np.abs(rotation - reference_rotation)))
        print('initial FK matrix error', matrix_error, flush=True)
        positions, errors, actions, timings = [reference], [], [], []
        for i in range(int(round(args.seconds * 30))):
            start = time.perf_counter()
            p, r, v, w = body.state()
            action, _ = policy.act(observation_v7(p, r, v, w, reference, np.zeros((24, 3))))
            body.advance(action, 1/30)
            p = body.state()[0]
            positions.append(p)
            errors.append(np.linalg.norm(p-reference, axis=-1))
            actions.append(action)
            timings.append((time.perf_counter()-start)*1000)
            if (i+1) % 90 == 0:
                print('step', i+1, 'height', p[0, 2], 'error', np.mean(errors[-1]), flush=True)
        positions, errors = np.asarray(positions), np.asarray(errors)
        passed = positions[:, 0, 2].min() >= .5 and errors.mean() <= .1
        np.savez_compressed(args.output/'rollout.npz', position_m=positions,
                            reference_position_m=reference, error_m=errors, action=actions, step_ms=timings)
        summary = dict(status='passed_native_standing_gate' if passed else 'failed_native_standing_gate',
                       backend='Isaac Gym Preview4 PhysX CPU', weights_frozen=True,
                       steps=len(errors), joint_error_mean_m=float(errors.mean()),
                       joint_error_p95_m=float(np.percentile(errors, 95)),
                       pelvis_min_height_m=float(positions[:, 0, 2].min()),
                       initial_rotation_max_matrix_error=matrix_error,
                       step_median_ms=float(np.median(timings)), properties=body.properties)
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary), flush=True)
    finally:
        body.close()


if __name__ == '__main__':
    main()
