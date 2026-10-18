"""Joint humanoid/walker scene, with paired grips and no world wrist support.

The short entry point checks interfaces; it is not a trained tracking policy.
"""
from isaacgym import gymapi

import json
import argparse
import ctypes
from pathlib import Path
import traceback
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .contract import COLLISION_FILTERS, NAMES, motor_effort_limits
from .grasp import free_velocity_predictor, grip_impulse, point_grasp, point_jacobian, skew
from .walker_scene import H, WalkerScene, clean, demand, vec


class HumanoidWalkerScene(WalkerScene):
    def __init__(self, asset_dir, xml, initial_position, initial_rotation, camera_rotation,
                 camera_translation, stiffness, damping, stereo_transform=None, project_initial_contact=False, dt=H,
                 grip_mode='spring', gravity=True, ground=True, physics_substeps=1):
        self.xml = Path(xml)
        self.walker_asset_dir = Path(asset_dir)
        self.raw_initial_position = np.asarray(initial_position).copy()
        self.reference_position = self.raw_initial_position.copy()
        self.reference_rotation = np.asarray(initial_rotation)
        self.stiffness, self.damping = stiffness, damping
        demand(grip_mode in ('spring', 'constraint', 'constraint_predict'), 'unknown grip model')
        self.grip_mode = grip_mode
        self.grip_velocity_history = None
        self.explicit_simulate_calls = 0
        self.contact_initialization = dict(method="raw reference; no contact projection", human_root_z_delta_m=0.)
        if project_initial_contact:
            foot_meta = json.loads(self.xml.with_suffix('.json').read_text(encoding='utf-8'))
            demand(foot_meta.get('status') == 'foot_collision_asset_generated_unvalidated',
                   'contact projection requires the explicitly shoe-matched foot asset')
            minimum = float('inf')
            tree = ET.parse(self.xml)
            for name in ('L_Ankle', 'L_Toe', 'R_Ankle', 'R_Toe'):
                i = NAMES.index(name)
                geom = tree.find('.//body[@name="'+name+'"]/geom')
                demand(geom.get('type') == 'box' and geom.get('quat') == '1 0 0 0', 'expected body-aligned shoe box')
                center, half = (np.fromstring(geom.get(k), sep=' ') for k in ('pos', 'size'))
                r = self.reference_rotation[i]
                minimum = min(minimum, float((self.reference_position[i]+r@center)[2]-np.abs(r[2])@half))
            delta = max(0., -minimum)
            self.reference_position[:, 2] += delta
            self.contact_initialization = dict(method="declared human-only initial root Z feasibility projection",
                raw_foot_collision_min_z_m=minimum, human_root_z_delta_m=delta,
                physical_floor_shift_m=0., walker_camera_shift_m=0., raw_grip_offsets_preserved=True,
                future_frames_used=0, simulation_steps_used=0,
                interpretation="initial state differs from image fit; no claim of exact image initialization")
        self.stereo_transform = None if stereo_transform is None else np.asarray(stereo_transform, dtype=float)
        if self.stereo_transform is not None:
            transform = self.stereo_transform
            demand(transform.shape == (4, 4) and np.isfinite(transform).all(), "finite stereo transform required")
            demand(np.allclose(transform[3], [0, 0, 0, 1]) and
                   np.allclose(transform[:3, :3].T@transform[:3, :3], np.eye(3), atol=1e-6) and
                   abs(np.linalg.det(transform[:3, :3])-1) <= 1e-6, "invalid stereo transform")
        meta = json.loads((Path(asset_dir)/"asset_metadata.json").read_text(encoding="utf-8"))
        camera = np.eye(4)
        camera[:3, :3], camera[:3, 3] = camera_rotation, camera_translation
        walker = camera@np.asarray(meta["left_camera_mount"]["T_C_from_W"])
        super().__init__(asset_dir, meta, dt=dt, gravity=gravity, ground=ground,
                         base_position=walker[:3, 3], quaternion=Rotation.from_matrix(walker[:3, :3]).as_quat(),
                         physics_substeps=physics_substeps)
        try:
            self.reset_humanoid()
            if grip_mode != 'spring':
                self.jacobian_descriptor = self.gym.acquire_jacobian_tensor(self.sim, 'humanoid')
                self.mass_descriptor = self.gym.acquire_mass_matrix_tensor(self.sim, 'humanoid')
            p, r, _, _ = self.human_state()
            w = self.read(False)
            self.grips = []
            for side, index in (("left", 17), ("right", 22)):
                local_handle = np.mean(self.metadata["handles_m"][side], axis=0)
                handle = w["p"]+w["R"]@local_handle
                local_wrist = self.reference_rotation[index].T@(handle-self.raw_initial_position[index])
                demand(np.linalg.norm(local_wrist) < .12, "initial wrist/handle offset needs independent correction")
                self.grips.append(dict(side=side, index=index, local_wrist=local_wrist, local_handle=local_handle))
            self.initialization_contract = dict(fk=self.fk_error, human_mass_kg=float(self.human_mass.sum()),
                body_count=24, joint_dof_count=69, root_velocity_zero=True, joint_velocity_zero=True,
                contact_initialization=self.contact_initialization,
                wrist_local_offsets_m={g["side"]: g["local_wrist"].tolist() for g in self.grips},
                grip_definition="assumed initial contact at handle midpoint; first-frame derived, not measured palm",
                human_dynamics="author collision/PD/density retained; subject inertia and torque not calibrated",
                hand_walker_contact="bilateral point-grip model replaces coarse palm hull contact; "
                    "wrist/hand hulls excluded against walker, other body collisions retained; "
                    "released finger surface contact not represented",
                collision_filter_readback=self.collision_filter_readback,
                effectively_unbounded_effort_count=int(np.sum(self.dof_properties["effort"] > 1e20)),
                imported_effectively_unbounded_effort_count=self.imported_unbounded,
                effort_source="scalar gear * inherited ctrlrange from source MJCF; author fixture, not subject strength",
                T_right_from_left=None if self.stereo_transform is None else self.stereo_transform.tolist(),
                body_mass_kg=self.human_mass.tolist(), body_com_local_m=self.human_local_com.tolist(),
                body_inertia_about_com_kg_m2=self.human_inertia.tolist(),
                foot_inertia_preservation=self.foot_inertia_preservation,
                grip_mode=grip_mode,
                dof_properties={k: self.dof_properties[k].tolist() for k in
                                ("lower", "upper", "stiffness", "damping", "effort", "armature")})
        except Exception:
            self.close()
            raise

    def create_participants(self):
        opt = gymapi.AssetOptions()
        opt.fix_base_link = False
        opt.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        opt.angular_damping = .01
        asset = self.gym.load_asset(self.sim, str(self.xml.parent), self.xml.name, opt)
        demand(asset is not None, "humanoid load failed")
        demand(tuple(self.gym.get_asset_rigid_body_names(asset)) == NAMES, "humanoid body order mismatch")
        dof_names = [name+"_"+axis for name in NAMES[1:] for axis in "xyz"]
        demand(self.gym.get_asset_dof_names(asset) == dof_names, "humanoid DOF order mismatch")
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*self.reference_position[0])
        pose.r = gymapi.Quat(*Rotation.from_matrix(self.reference_rotation[0]).as_quat())
        self.human = self.gym.create_actor(self.env, asset, pose, "humanoid", 0, 0)
        shapes = self.gym.get_actor_rigid_shape_properties(self.env, self.human)
        demand(len(shapes) == 24, "expected author 24 body collision shapes")
        # The author's solid palm capsule/sphere has no cavity for a grasped
        # tube. The paired grip replaces this interaction, rather than adding
        # a constraint on top of an interpenetrating solid hand. Reserve bits
        # absent from the author's human self-collision masks. This excludes
        # only walker--palm pairs; forearm, trunk, leg and ground contacts stay.
        palms = (17, 18, 22, 23)
        palm_bits = {index: 1 << (20+j) for j, index in enumerate(palms)}
        grip_bits = sum(palm_bits.values())
        demand(all(mask & grip_bits == 0 for mask in COLLISION_FILTERS), "grip filter bits already used")
        walker_shapes = self.gym.get_actor_rigid_shape_properties(self.env, self.actor)
        for shape in walker_shapes:
            demand(shape.filter & grip_bits == 0, "walker grip filter bits already used")
            shape.filter |= grip_bits
        self.gym.set_actor_rigid_shape_properties(self.env, self.actor, walker_shapes)
        for i, (shape, mask) in enumerate(zip(shapes, COLLISION_FILTERS)):
            shape.filter = mask | palm_bits.get(i, 0)
            shape.contact_offset, shape.rest_offset = .002, 0.
        self.gym.set_actor_rigid_shape_properties(self.env, self.human, shapes)
        actual = self.gym.get_actor_rigid_shape_properties(self.env, self.human)
        walker_actual = self.gym.get_actor_rigid_shape_properties(self.env, self.actor)
        expected = [mask | palm_bits.get(i, 0) for i, mask in enumerate(COLLISION_FILTERS)]
        demand([p.filter for p in actual] == expected, "human contact filter readback mismatch")
        demand(all(p.filter == grip_bits for p in walker_actual), "walker contact filter readback mismatch")
        self.collision_filter_readback = dict(walker=[p.filter for p in walker_actual], human=expected,
                                             palm_body_names=[NAMES[i] for i in palms])
        self.dof_properties = self.gym.get_actor_dof_properties(self.env, self.human)
        self.imported_unbounded = int(np.sum(self.dof_properties["effort"] > 1e20))
        self.dof_properties["driveMode"] = gymapi.DOF_MODE_POS
        limits = motor_effort_limits(self.xml, dof_names)
        self.dof_properties["effort"] = limits
        self.gym.set_actor_dof_properties(self.env, self.human, self.dof_properties)
        self.dof_properties = self.gym.get_actor_dof_properties(self.env, self.human)
        demand(np.allclose(self.dof_properties["effort"], limits), "motor effort readback mismatch")
        demand(np.isfinite(self.dof_properties['armature']).all() and
               np.all(self.dof_properties['armature'] >= 0), 'invalid joint armature readback')
        props = self.gym.get_actor_rigid_body_properties(self.env, self.human)
        self.human_mass = np.array([p.mass for p in props])
        self.human_local_com = np.array([vec(p.com) for p in props])
        self.human_inertia = np.array([[vec(p.inertia.x), vec(p.inertia.y), vec(p.inertia.z)] for p in props]).transpose(0, 2, 1)
        self.foot_inertia_preservation = None
        if self.xml.with_suffix('.json').exists():
            metadata = json.loads(self.xml.with_suffix('.json').read_text(encoding='utf-8'))
            if metadata.get('status') == 'foot_collision_asset_generated_unvalidated':
                errors = dict(mass_kg=0., com_m=0., inertia_kg_m2=0.)
                for name, expected in metadata['boxes'].items():
                    i = NAMES.index(name)
                    errors['mass_kg'] = max(errors['mass_kg'], abs(self.human_mass[i]-expected['expected_mass_kg']))
                    errors['com_m'] = max(errors['com_m'], float(np.linalg.norm(self.human_local_com[i]-expected['expected_com_m'])))
                    errors['inertia_kg_m2'] = max(errors['inertia_kg_m2'], float(np.linalg.norm(
                        self.human_inertia[i]-expected['expected_inertia_about_com_kg_m2'])))
                demand(errors['mass_kg'] <= 1e-5 and errors['com_m'] <= 1e-6 and errors['inertia_kg_m2'] <= 1e-7,
                       'foot inertial properties changed during collision retargeting: %s' % errors)
                self.foot_inertia_preservation = errors
        self.human_handles = [self.gym.get_actor_rigid_body_handle(self.env, self.human, i) for i in range(24)]

    def human_state(self):
        raw = self.gym.get_actor_rigid_body_states(self.env, self.human, gymapi.STATE_ALL)
        xyz = lambda array: np.column_stack([array[k] for k in "xyz"]).astype(float)
        p, q = xyz(raw["pose"]["p"]), np.column_stack([raw["pose"]["r"][k] for k in "xyzw"])
        return p, Rotation.from_quat(q).as_matrix(), xyz(raw["vel"]["linear"]), xyz(raw["vel"]["angular"])

    def reset_humanoid(self):
        self.grip_velocity_history = None
        self.grip_active = True
        self.motion_time_offset_s = 0.
        parent_lookup = {child.get("name"): parent.get("name") for parent in ET.parse(self.xml).findall(".//body")
                         for child in parent.findall("body")}
        parents = [-1]+[NAMES.index(parent_lookup[n]) for n in NAMES[1:]]
        local = np.stack([self.reference_rotation[parents[i]].T@self.reference_rotation[i] for i in range(1, 24)])
        self.joint_target = Rotation.from_matrix(local).as_rotvec().ravel().astype(np.float32)
        joint = np.zeros(69, dtype=gymapi.DofState.dtype)
        joint["pos"] = self.joint_target
        demand(self.gym.set_actor_dof_states(self.env, self.human, joint, gymapi.STATE_ALL), "DOF reset rejected")
        self.gym.set_actor_dof_position_targets(self.env, self.human, self.joint_target)
        # Compare imported FK before assigning saved body transforms. A state
        # write must never hide a DOF-to-rotation mapping error.
        p, r, _, _ = self.human_state()
        current_local = (p-p[0])@r[0]
        reference_local = (self.reference_position-self.reference_position[0])@self.reference_rotation[0]
        current_rotation = r[0].T@r
        reference_rotation = self.reference_rotation[0].T@self.reference_rotation
        position_error = float(np.linalg.norm(current_local-reference_local, axis=1).max())
        angle_error = float(Rotation.from_matrix(current_rotation@reference_rotation.transpose(0, 2, 1)).magnitude().max())
        self.fk_error = dict(max_root_frame_position_error_m=position_error, max_root_frame_rotation_error_rad=angle_error,
            pre_reset_root_rotation_drift_rad=float(Rotation.from_matrix(r[0]@self.reference_rotation[0].T).magnitude()))
        demand(position_error <= 1e-4 and angle_error <= 1e-4, "actual PhysX FK mismatch: %s" % self.fk_error)
        # The root-state API controls articulations without writing all child
        # poses. A CPU NumPy descriptor follows gymtorch.unwrap_tensor's
        # documented layout; no torch extension or GPU pointer is involved.
        root = np.zeros((2, 13), dtype=np.float32)
        walker = self.read(False)
        root[0, :3], root[0, 3:7] = walker["p"], walker["q"]
        root[1, :3] = self.reference_position[0]
        root[1, 3:7] = Rotation.from_matrix(self.reference_rotation[0]).as_quat()
        descriptor = gymapi.Tensor()
        descriptor.device = -1
        descriptor.dtype = gymapi.DTYPE_FLOAT32
        descriptor.shape = list(root.shape)
        descriptor.data_address = root.ctypes.data
        descriptor.own_data = False
        demand(self.gym.set_actor_root_state_tensor(self.sim, descriptor), "joint root reset rejected")
        p, r, v, w = self.human_state()
        reset_error = float(np.linalg.norm(p-self.reference_position, axis=1).max())
        demand(reset_error <= 1e-5, "human reset position mismatch: %s, root=%s" % (reset_error, p[0].tolist()))
        angle_error = Rotation.from_matrix(r@self.reference_rotation.transpose(0, 2, 1)).magnitude().max()
        demand(angle_error <= 1e-5, "human reset orientation mismatch")
        demand(np.linalg.norm(v, axis=1).max() <= 1e-6 and np.linalg.norm(w, axis=1).max() <= 1e-6,
               "human reset velocities not zero")

    def motion_state(self):
        """Capture at a completed control step, without stepping the engine.

        This covers public motion/controller state, not PhysX contact caches.
        It can only be restored into the same assets and timing configuration.
        """
        p, r, v, w = self.human_state()
        walker = self.read(False)
        roots = np.stack([np.r_[walker['p'], walker['q'], walker['v'], walker['w']],
                          np.r_[p[0], Rotation.from_matrix(r[0]).as_quat(), v[0], w[0]]]).astype(np.float32)
        history = self.grip_velocity_history
        return dict(root_state=roots,
            dof_state=self.gym.get_actor_dof_states(self.env, self.human, gymapi.STATE_ALL).copy(),
            joint_target=self.joint_target.copy(), human_position=p, human_rotation=r,
            human_linear_velocity=v, human_angular_velocity=w,
            grip_local_wrist=np.array([g['local_wrist'] for g in self.grips]),
            grip_local_handle=np.array([g['local_handle'] for g in self.grips]),
            grip_history_velocity=np.array([]) if history is None else history[0].copy(),
            grip_history_response=np.array([]) if history is None else history[1].copy(),
            time_s=np.array(walker['t']+self.motion_time_offset_s), dt_s=np.array(self.dt),
            physics_substeps=np.array(self.physics_timing['physics_substeps']),
            grip_active=np.array(self.grip_active),
            grip_mode=np.array(self.grip_mode), source_xml=np.array(str(self.xml.resolve())),
            source_walker_asset=np.array(str(self.walker_asset_dir.resolve())))

    def restore_motion_state(self, state):
        """Restore both free roots, joints and lagged grip state, then read back.

        No simulation/prewarm or per-child pose writes are used. The native
        simulation clock stays unchanged; a separate logical time is restored.
        """
        demand(str(state['source_xml']) == str(self.xml.resolve()), 'motion restore requires the same humanoid asset')
        demand(str(state['source_walker_asset']) == str(self.walker_asset_dir.resolve()),
               'motion restore requires the same walker asset')
        demand(float(state['dt_s']) == self.dt and str(state['grip_mode']) == self.grip_mode and
               int(state['physics_substeps']) == self.physics_timing['physics_substeps'],
               'motion restore timing/controller mismatch')
        roots = np.array(state['root_state'], dtype=np.float32, copy=True)
        joint = np.array(state['dof_state'], dtype=gymapi.DofState.dtype, copy=True)
        target = np.array(state['joint_target'], dtype=np.float32, copy=True)
        local_wrist, local_handle = (np.asarray(state[k], dtype=float) for k in ('grip_local_wrist', 'grip_local_handle'))
        previous, response = (np.array(state[k], dtype=float, copy=True)
                              for k in ('grip_history_velocity', 'grip_history_response'))
        active = np.asarray(state['grip_active'])
        demand(active.shape == () and active.dtype == np.dtype(bool), 'invalid saved grasp/release mode')
        demand(roots.shape == (2, 13) and joint.shape == target.shape == (69,) and
               local_wrist.shape == local_handle.shape == (2, 3) and
               previous.shape == response.shape and previous.shape in ((0,), (6,)), 'invalid motion-state layout')
        demand(np.isfinite(np.r_[roots.ravel(), joint['pos'], joint['vel'], target,
            local_wrist.ravel(), local_handle.ravel(), previous, response, float(state['time_s'])]).all(),
            'nonfinite motion restore state')
        demand(np.allclose(np.linalg.norm(roots[:, 3:7], axis=1), 1, atol=1e-5), 'invalid restore root quaternion')
        before = float(self.gym.get_sim_time(self.sim))
        demand(self.gym.set_actor_dof_states(self.env, self.human, joint, gymapi.STATE_ALL), 'motion DOF restore rejected')
        self.gym.set_actor_dof_position_targets(self.env, self.human, target)
        descriptor = gymapi.Tensor()
        descriptor.device, descriptor.dtype, descriptor.shape = -1, gymapi.DTYPE_FLOAT32, list(roots.shape)
        descriptor.data_address, descriptor.own_data = roots.ctypes.data, False
        demand(self.gym.set_actor_root_state_tensor(self.sim, descriptor), 'motion root restore rejected')
        self.joint_target = target
        for g, a, b in zip(self.grips, local_wrist, local_handle):
            g['local_wrist'], g['local_handle'] = a.copy(), b.copy()
        self.grip_velocity_history = (previous, response) if len(previous) else None
        self.grip_active = bool(active)
        self.motion_time_offset_s = float(state['time_s'])-before
        actual = self.motion_state()
        error = {key: float(np.max(np.abs(actual[key]-state[key]))) for key in
                 ('root_state', 'human_position', 'human_linear_velocity', 'human_angular_velocity', 'joint_target')}
        for field in ('pos', 'vel'):
            error['dof_'+field] = float(np.max(np.abs(actual['dof_state'][field]-joint[field])))
        error['human_rotation_rad'] = float(Rotation.from_matrix(
            actual['human_rotation']@np.asarray(state['human_rotation']).transpose(0, 2, 1)).magnitude().max())
        demand(float(self.gym.get_sim_time(self.sim)) == before, 'restore advanced native physics time')
        demand(max(error.values()) <= 1e-5, 'motion restore readback mismatch: %s' % error)
        demand(abs(float(actual['time_s'])-float(state['time_s'])) <= 1e-8, 'logical restore time mismatch')
        return dict(errors=error, native_time_s=before, logical_time_s=float(actual['time_s']),
                    simulate_calls_during_restore=0, contact_solver_cache_restored=False)

    @staticmethod
    def tensor_view(descriptor):
        demand(descriptor.device == -1 and descriptor.dtype == gymapi.DTYPE_FLOAT32, 'expected CPU float tensor')
        data = (ctypes.c_float*int(np.prod(descriptor.shape))).from_address(descriptor.data_address)
        return np.ctypeslib.as_array(data).reshape(descriptor.shape)

    def grip_mobility(self, p, r, walker, connections):
        self.gym.refresh_jacobian_tensors(self.sim)
        jac = self.tensor_view(self.jacobian_descriptor)[0].astype(float)
        demand(jac.shape == (24, 6, 75), 'floating-base Jacobian shape mismatch')
        com = p+np.einsum('nij,nj->ni', r, self.human_local_com)
        mass = np.einsum('i,ijk,ijl->kl', self.human_mass, jac[:, :3], jac[:, :3])
        iw = r@self.human_inertia@r.transpose(0, 2, 1)
        mass += np.einsum('ijk,ijl,ilm->km', jac[:, 3:], iw, jac[:, 3:])
        if not hasattr(self, 'mass_contract'):
            self.gym.refresh_mass_matrix_tensors(self.sim)
            engine_mass = self.tensor_view(self.mass_descriptor)[0]
            schur = mass[6:, 6:]-mass[6:, :6]@np.linalg.solve(mass[:6, :6], mass[:6, 6:])
            error = float(np.linalg.norm(schur-engine_mass)/np.linalg.norm(engine_mass))
            demand(error <= 1e-4, 'full inertia / Gym root-eliminated mass mismatch')
            root_error = max(np.linalg.norm(jac[i, :3, 3:6]+skew(com[i]-com[0])) for i in range(24))
            demand(root_error <= 1e-5, 'Jacobian is not at declared body COM')
            self.mass_contract = dict(schur_relative_error=error, root_com_jacobian_error=root_error,
                query_scope='rigid-body inertia only; does not establish armature completeness',
                response_armature_source='get_actor_dof_properties.armature, kg*m^2',
                response_armature_added=True)
        human_jac, walker_jac = [], []
        for g, connection in zip(self.grips, connections):
            i, point = g['index'], connection['point']
            human_jac.append(jac[i, :3]-skew(point-com[i])@jac[i, 3:])
            walker_jac.append(point_jacobian(point, walker['com']))
        human_jac, walker_jac = np.vstack(human_jac), np.vstack(walker_jac)
        # Preview4's queried body mass matrix omits joint armature although
        # forward dynamics use it. Keep the author's read-back armature in
        # both simulation and response; do not zero it to fit a force trace.
        mass[6:, 6:] += np.diag(self.dof_properties['armature']+
            self.dt*self.dof_properties['damping']+self.dt**2*self.dof_properties['stiffness'])
        inverse_walker = np.zeros((6, 6))
        inverse_walker[:3, :3] = np.eye(3)/self.mass
        inverse_walker[3:, 3:] = np.linalg.inv(walker['R']@self.inertia@walker['R'].T)
        mobility = human_jac@np.linalg.solve(mass, human_jac.T)+walker_jac@inverse_walker@walker_jac.T
        return (mobility+mobility.T)/2

    def advance_coupled(self, active=None):
        # No new command means retain the saved release/grasp mode.
        if active is not None:
            self.grip_active = bool(active)
        active = self.grip_active
        p, r, v, w = self.human_state()
        walker = self.read(False)
        connections = []
        for g in self.grips:
            i = g["index"]
            wrist = dict(p=p[i], R=r[i], v=v[i], w=w[i], com=p[i]+r[i]@self.human_local_com[i])
            connection = point_grasp(wrist, g["local_wrist"], walker, g["local_handle"],
                                     self.stiffness, self.damping, active)
            connections.append(connection)
        if self.grip_mode != 'spring' and active:
            mobility = self.grip_mobility(p, r, walker, connections)
            relative = []
            for g, connection in zip(self.grips, connections):
                i, point = g['index'], connection['point']
                relative.append(walker['v']+np.cross(walker['w'], point-walker['com'])-
                    v[i]-np.cross(w[i], point-p[i]-r[i]@self.human_local_com[i]))
            relative = np.concatenate(relative)
            free = relative
            if self.grip_mode == 'constraint_predict':
                free = free_velocity_predictor(relative, *(self.grip_velocity_history or (None, None)))
            impulse, self.last_constraint_diagnostics = grip_impulse(mobility,
                np.concatenate([g['anchor_b']-g['anchor_a'] for g in connections]), free, self.dt)
            self.grip_velocity_history = (relative.copy(), mobility@impulse)
            for connection, force in zip(connections, impulse.reshape(2, 3)/self.dt):
                connection['force_a'], connection['force_b'] = force, -force
        else:
            self.grip_velocity_history = None
        for g, connection in zip(self.grips, connections):
            i = g['index']
            point = gymapi.Vec3(*connection["point"])
            self.gym.apply_body_force_at_pos(self.env, self.human_handles[i],
                gymapi.Vec3(*connection["force_a"]), point, gymapi.ENV_SPACE)
            self.gym.apply_body_force_at_pos(self.env, self.body,
                gymapi.Vec3(*connection["force_b"]), point, gymapi.ENV_SPACE)
        self.explicit_simulate_calls += 1
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        return connections

    def camera_mount(self, side="left"):
        walker = self.read(False)
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = walker["R"], walker["p"]
        left = transform@np.asarray(self.metadata["left_camera_mount"]["T_W_from_C"])
        if side == "left":
            return left
        demand(side == "right" and self.stereo_transform is not None, "right camera requires full stereo calibration")
        return left@np.linalg.inv(self.stereo_transform)


def static_hold_assessment(human, metrics):
    """Keep standing and sustained grasp as separate operational checks."""
    human, metrics = np.asarray(human), np.asarray(metrics)
    drop = float(max(0., human[0, 0, 2, 0]-human[:, 0, 2, 0].min()))
    gaps = metrics[:, 2:4].max(0)
    return dict(passed=bool(drop <= .15 and max(gaps) <= .05),
        pose_hold_passed=bool(drop <= .15), sustained_grip_passed=bool(max(gaps) <= .05),
        max_pelvis_drop_m=drop, max_recorded_grip_gap_m=gaps.tolist(),
        allowed_pelvis_drop_m=.15, grip_diagnostic_limit_m=.05,
        interpretation='engineering diagnostics, not human truth; 5cm grip limit added after initial failure observation')


def check(asset_dir, xml, reference, output, calibration=None, project_initial_contact=False, half_step=False, static_hold=False,
          grip_mode='spring', physics_substeps=1):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with np.load(reference, allow_pickle=False) as saved:
        first = {k: saved[k][0].copy() for k in ("reference_position_m", "reference_rotation", "camera_rotation", "camera_translation_m")}
    dt = H/2 if half_step else H
    duration = 1. if static_hold else .2
    steps = int(round(duration/dt))
    grip_steps = steps if static_hold else int(round(.125/dt))
    report = dict(status="failed", scope="frozen initial PD hold; integration only, not tracking", duration_s=duration,
                  reference_first_frame=str(reference), xml=str(xml), asset=str(asset_dir), dt_s=dt,
                  grip_stiffness_N_m=400. if grip_mode == 'spring' else None,
                  grip_damping_N_s_m=4. if grip_mode == 'spring' else None,
                  training_steps=0, policy_calls=0, simulate_calls=0,
                  initial_geometry="first frame of existing fitted body; full-clip shape pre-calibration retained",
                  training_ready=False, trace_columns=["t_s", "grip_active", "left_gap_m", "right_gap_m",
                      "left_force_N", "right_force_N", "pelvis_z_m", "walker_contact_z_N"],
                  trace_missing="t=0 contact is unavailable after reset; NaN, not measured zero")
    report['grip_mode'] = grip_mode
    report['constraint_prediction'] = ('one-step lagged measured free increment; reset history on release'
        if grip_mode == 'constraint_predict' else 'none')
    report['constraint_work_interpretation'] = 'linearized projection of predicted free velocity; not measured total energy'
    human, walker, cameras, right_cameras, metrics, contacts = [], [], [], [], [], []
    grip_gaps, grip_forces, constraint_work = [], [], []
    try:
        stereo = None
        if calibration is not None:
            with np.load(calibration, allow_pickle=False) as saved:
                stereo = np.eye(4)
                stereo[:3, :3], stereo[:3, 3] = saved["R01"], saved["T01_m"].reshape(3)
            report["stereo_calibration_source"] = str(calibration)
        with HumanoidWalkerScene(asset_dir, xml, first["reference_position_m"], first["reference_rotation"],
                first["camera_rotation"], first["camera_translation_m"], 400., 4., stereo, project_initial_contact, dt, grip_mode,
                physics_substeps=physics_substeps) as s:
            report["initialization"] = s.initialization_contract
            report['timing'] = dict(s.physics_timing, grip_update_interval_s=dt,
                semantics='grip evaluated once per simulate call; applied pair force held across internal substeps')
            for i in range(steps+1):
                p, r, v, w = s.human_state()
                state = s.read(contacts=i > 0)
                demand(abs(state["t"]-i*dt) <= 1e-6, "coupled scene time does not match completed steps")
                human.append(np.concatenate([p[..., None], r], axis=2))
                walker.append(np.r_[state["p"], state["q"], state["v"], state["w"]])
                cameras.append(s.camera_mount())
                right_cameras.append(s.camera_mount("right") if stereo is not None else np.full((4, 4), np.nan))
                if stereo is not None:
                    demand(np.allclose(np.linalg.inv(right_cameras[-1])@cameras[-1], stereo, atol=1e-6),
                           "simulated stereo relative pose changed")
                contact = np.full((25, 3), np.nan) if i == 0 else np.array(
                    [vec(f) if hasattr(f, "x") else [f[k] for k in "xyz"]
                     for f in s.gym.get_env_rigid_contact_forces(s.env)])
                contacts.append(contact)
                grip_gaps.append([float(np.linalg.norm(p[g['index']]+r[g['index']]@g['local_wrist']-
                    state['p']-state['R']@g['local_handle'])) for g in s.grips])
                demand(np.isfinite(np.r_[p.ravel(), r.ravel(), v.ravel(), w.ravel()]).all(), "nonfinite human state")
                if i < steps:
                    grips = s.advance_coupled(active=i < grip_steps)
                    grip_forces.append([g['force_a'].copy() for g in grips])
                    constraint_work.append([s.last_constraint_diagnostics[k] for k in
                        ('stabilization_work_J', 'velocity_projection_dissipation_J')]
                        if grip_mode != 'spring' and i < grip_steps else [np.nan, np.nan])
                    metrics.append([i*dt, float(i < grip_steps), *[g["error_m"] for g in grips],
                                    *[np.linalg.norm(g["force_a"]) for g in grips], p[0, 2], state["contact"][2]])
            metrics = np.asarray(metrics)
            report.update(status="integration_contract_passed", simulate_calls=s.explicit_simulate_calls,
                max_grip_gap_m=metrics[:, 2:4].max(0).tolist(), max_grip_force_N=metrics[:, 4:6].max(0).tolist(),
                active_max_grip_gap_m=metrics[:grip_steps, 2:4].max(0).tolist(),
                walker_contact_peak_z_N=float(np.nanmax(np.asarray(contacts)[:, s.index, 2])),
                released_max_grip_force_N=float(metrics[grip_steps:, 4:6].max()) if grip_steps < steps else None,
                final_root_height_m=float(p[0, 2]), initial_root_height_m=float(human[0][0, 2, 0]),
                raw_reference_initial_root_height_m=float(first["reference_position_m"][0, 2]),
                camera_source="simulated rigid walker mount; external Stage camera used only for initial transform")
            if hasattr(s, 'mass_contract'):
                report['mass_contract'] = s.mass_contract
            demand(report["released_max_grip_force_N"] in (None, 0.), "released humanoid grasp force persists")
            if static_hold:
                report['static_hold_gate'] = static_hold_assessment(human, metrics)
                if grip_mode != 'spring':
                    gap, forces = np.asarray(grip_gaps), np.asarray(grip_forces)
                    peak = float(np.linalg.norm(forces, axis=2).max())
                    after_start = gap[np.arange(steps+1)*dt >= .05].max(0)
                    passed = bool(gap.max() <= .012 and after_start.max() <= .005 and peak <= 1000 and
                                  report['static_hold_gate']['max_pelvis_drop_m'] <= .15)
                    report['constraint_grip_gate'] = dict(passed=passed, full_max_gap_m=gap.max(0).tolist(),
                        after50ms_max_gap_m=after_start.tolist(), force_peak_N=peak,
                        impulse_per_hand_N_s=(forces.sum(0)*dt).tolist(), initial_transient_included=True,
                        limits='full12mm; after50ms5mm; force1000N; pelvis_drop150mm; conditional engineering only')
    except Exception as error:
        report.update(status="failed", reason=str(error), failure_traceback=traceback.format_exc())
    finally:
        if "s" in locals():
            report["simulate_calls"] = s.explicit_simulate_calls
        np.savez_compressed(output/"trajectory.npz", human=np.asarray(human), walker=np.asarray(walker),
                            cameras=np.asarray(cameras), right_cameras=np.asarray(right_cameras),
                            metrics=np.asarray(metrics), contact_forces_N=np.asarray(contacts),
                            grip_gap_m=np.asarray(grip_gaps), grip_force_N=np.asarray(grip_forces),
                            constraint_work_J=np.asarray(constraint_work))
        (output/"results.json").write_text(json.dumps(clean(report), ensure_ascii=False, indent=2,
                                                     allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(clean(report), ensure_ascii=False))
    passed = (report["status"] == "integration_contract_passed" and report.get('static_hold_gate', {}).get('passed', True)
              and report.get('constraint_grip_gate', {}).get('passed', True))
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('asset_dir', 'xml', 'reference', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('calibration', type=Path, nargs='?')
    parser.add_argument('--project-initial-contact', action='store_true',
                        help='explicit human-only initial Z projection; preserve raw pose and raw grip offsets')
    parser.add_argument('--half-step', action='store_true', help='legacy combined refinement: halves simulate AND grip-update intervals')
    parser.add_argument('--physics-substeps', type=int, choices=(1, 2), default=1,
                        help='refine physical integration while keeping the grip update interval unchanged')
    parser.add_argument('--static-hold', action='store_true', help='one-second frozen PD/grip diagnostic; no new image or pose targets')
    parser.add_argument('--grip-mode', choices=('spring', 'constraint', 'constraint_predict'), default='spring')
    args = parser.parse_args()
    raise SystemExit(check(**vars(args)))
