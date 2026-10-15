"""Explicit MuJoCo backend transfer of the author's released upright SMPL XML.

This transfer FAILED the standing gate. Ball-joint/stable-PD, joint-coordinate
and collision differences are disclosed in the experiment record. Retained
for the failed-backend diagnostic and rotation-interface test only.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .contract import NAMES, COLLISION_FILTERS


class PHCBody:
    def __init__(self, source_xml: Path):
        tree = ET.parse(source_xml)
        root = tree.getroot()
        root.find('option').set('timestep', str(1 / 240))
        root.find('option').set('integrator', 'implicitfast')
        root.find('option').set('gravity', '0 0 -9.81')
        joints = list(root.findall('.//worldbody//joint'))
        self.joint_names = [f'{name}_{axis}' for name in NAMES[1:] for axis in 'xyz']
        by_name = {j.get('name'): j for j in joints}
        if set(by_name) != set(self.joint_names):
            raise ValueError('source XML must contain the 69 PHC SMPL hinge joints')
        self.scale = np.full(69, np.pi)
        self.scale[self.joint_names.index('L_Knee_y')] = 5
        self.scale[self.joint_names.index('R_Knee_y')] = 5
        actuator = root.find('actuator')
        actuator.clear()
        self.gains = []
        # Isaac Gym exposes these triples as spherical exp-map DOFs. Keep the
        # released rigid bodies, but replace each XYZ hinge triple with a ball
        # joint so feedback/torque coordinates have the same rotation meaning.
        for name in NAMES[1:]:
            element = root.find(f'.//body[@name="{name}"]')
            joint = by_name[f'{name}_x']
            kp, kd = float(joint.get('stiffness')), float(joint.get('damping'))
            for child in list(element.findall('joint')):
                element.remove(child)
            ET.SubElement(element, 'joint', name=name, type='ball', pos='0 0 0',
                          limited='false', damping='0', stiffness='0', armature='0.02')
            self.gains.extend([(kp, kd)] * 3)
            for axis, vector in zip('xyz', ('1 0 0', '0 1 0', '0 0 1')):
                ET.SubElement(actuator, 'motor', name=f'{name}_{axis}', joint=name,
                              gear=vector + ' 0 0 0', ctrlrange='-500 500', ctrllimited='true')
        # PHC humanoid.py::_process_dof_props overrides collision filters after
        # loading the asset. XML alone otherwise creates spurious Head/Chest
        # contact even at the author's standing reference pose.
        contact = root.find('contact')
        if contact is None:
            contact = ET.SubElement(root, 'contact')
        for i in range(24):
            for j in range(i):
                if COLLISION_FILTERS[i] & COLLISION_FILTERS[j]:
                    ET.SubElement(contact, 'exclude', body1=NAMES[i], body2=NAMES[j])
        self.xml = ET.tostring(root, encoding='unicode')
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.body_ids = np.array([self.model.body(name).id for name in NAMES])
        self.qpos_indices = np.array([self.model.jnt_qposadr[self.model.joint(n).id] + k
                                     for n in NAMES[1:] for k in range(4)])
        self.dof_indices = np.array([self.model.jnt_dofadr[self.model.joint(n).id] + k
                                    for n in NAMES[1:] for k in range(3)])
        self.kp = np.r_[np.zeros(6), np.asarray(self.gains)[:, 0]]
        self.kd = np.r_[np.zeros(6), np.asarray(self.gains)[:, 1]]
        self.parents = [int(np.flatnonzero(self.body_ids == self.model.body_parentid[b])[0])
                        if i else -1 for i, b in enumerate(self.body_ids)]
        self.max_torque = 0.

    def from_global_rotations(self, position, rotations):
        qpos = self.model.qpos0.copy()
        qpos[:3] = position
        qpos[3:7] = Rotation.from_matrix(rotations[0]).as_quat()[[3, 0, 1, 2]]
        local = np.stack([rotations[self.parents[i]].T @ rotations[i] for i in range(1, 24)])
        qpos[self.qpos_indices] = Rotation.from_matrix(local).as_quat()[:, [3, 0, 1, 2]].ravel()
        return qpos

    def reset(self, qpos):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

    def state(self):
        pos = self.data.xpos[self.body_ids].copy()
        rot = self.data.xmat[self.body_ids].reshape(24, 3, 3).copy()
        velocity, omega = [], []
        for body in self.body_ids:
            jac, jacrot = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jac, jacrot, int(body))
            velocity.append(jac @ self.data.qvel)
            omega.append(jacrot @ self.data.qvel)
        return pos, rot, np.array(velocity), np.array(omega)

    def advance(self, action, dt):
        if np.shape(action) != (69,) or not np.isfinite(action).all() or not np.isfinite(dt) or dt <= 0:
            raise ValueError('finite 69D action and positive dt required')
        target_rotation = Rotation.from_rotvec((np.clip(action, -1, 1) * self.scale).reshape(23, 3))
        remaining = dt
        self.max_torque = 0.
        while remaining > 1e-10:
            self.model.opt.timestep = min(remaining, 1 / 240)
            h = self.model.opt.timestep
            current = Rotation.from_quat(self.data.qpos[self.qpos_indices].reshape(23, 4)[:, [1, 2, 3, 0]])
            error = np.r_[np.zeros(6), (current.inv() * target_rotation).as_rotvec().ravel()]
            # Stable PD accounts for the next velocity rather than explicitly
            # applying a stiff spring to the very light hand/toe bodies.
            mass = np.empty((self.model.nv, self.model.nv))
            mujoco.mj_fullM(self.model, self.data, mass)
            base = self.kp * (error - h * self.data.qvel) - self.kd * self.data.qvel
            correction = h * self.kd + h*h * self.kp
            acceleration = np.linalg.solve(mass + np.diag(correction),
                                           base - self.data.qfrc_bias + self.data.qfrc_constraint)
            torque = base - correction * acceleration
            self.data.ctrl[:] = np.clip(torque[self.dof_indices], -500, 500)
            mujoco.mj_step(self.model, self.data)
            remaining -= self.model.opt.timestep
            self.max_torque = max(self.max_torque, float(np.max(np.abs(self.data.actuator_force))))
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                return False
        mujoco.mj_forward(self.model, self.data)
        return True
