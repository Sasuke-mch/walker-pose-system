"""Exact sparse SMPL-H surface observations plus a MuJoCo articulated surrogate.

Observation skinning retains all 52 transforms and pose blend shapes. MuJoCo
simulates the 22 body links; hands retain the frame-zero rotations. The physical
mass/collision model is an explicitly approximate, uncalibrated surrogate.
"""

from pathlib import Path
import pickle
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from pose_app.smpl_coco_observation import (
    _install_legacy_smpl_pickle_compatibility,
    load_coco17_regressor,
)
from .data import ROOT


def wxyz(matrix):
    return Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]


def fmt(values):
    return " ".join(f"{x:.12g}" for x in np.asarray(values).ravel())


class PhysicsBody:
    def __init__(
        self,
        initial_fit: Path,
        world_rotation: np.ndarray,
        world_translation: np.ndarray,
        heel_mode: str = "surface",
    ):
        if heel_mode not in ("surface", "rigid"):
            raise ValueError("heel_mode must be surface or explicit historical rigid")
        self.heel_mode = heel_mode
        _install_legacy_smpl_pickle_compatibility()
        with (ROOT / "third_party/WiLoR/mano_data/models/SMPLH_male.pkl").open(
            "rb"
        ) as stream:
            asset = pickle.load(stream, encoding="latin1")
        reg = load_coco17_regressor(ROOT / "models/smpl/J_regressor_coco.npy")[5:17]
        with np.load(initial_fit, allow_pickle=False) as fitted:
            if fitted["pair_id"].tolist() != [0]:
                raise ValueError("only a frame-zero initializer is permitted")
            self.beta = fitted["betas"][0].astype(float)
            self.initial_rotations = Rotation.from_rotvec(
                fitted["body_pose"][0].reshape(21, 3)
            ).as_matrix()
            self.hands = Rotation.from_rotvec(
                np.concatenate(
                    [fitted["left_hand_pose"][0], fitted["right_hand_pose"][0]]
                ).reshape(30, 3)
            ).as_matrix()
            root_rotation = (
                world_rotation
                @ Rotation.from_rotvec(fitted["global_orient"][0]).as_matrix()
            )
            self.initial_root = (
                world_rotation @ fitted["smplh_joints"][0, 0] + world_translation
            )
            self.expected_initial = (
                fitted["predicted_coco"][0, 5:17] @ world_rotation.T + world_translation
            )
        shaped = np.asarray(asset["v_template"], float) + np.einsum(
            "vcd,d->vc", asset["shapedirs"], self.beta
        )
        self.joints_rest = np.asarray(asset["J_regressor"] @ shaped)
        self.parents = np.asarray(asset["kintree_table"][0], dtype=np.int64)
        self.parents[0] = -1
        self.vertex_ids = np.unique(
            np.r_[np.flatnonzero(np.any(reg > 0, axis=0)), 3387, 6787]
        )
        self.skin_regressor = reg[:, self.vertex_ids]
        self.heel_rows = np.searchsorted(self.vertex_ids, [3387, 6787])
        self.shaped_vertices = shaped[self.vertex_ids]
        self.pose_dirs = np.asarray(asset["posedirs"], float)[self.vertex_ids].reshape(
            -1, 459
        )
        self.weights = np.asarray(asset["weights"], float)[self.vertex_ids]
        self.heel_local = shaped[[3387, 6787]] - self.joints_rest[[10, 11]]
        self.xml = self._xml()
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.body_ids = np.asarray([self.model.body(f"link_{j}").id for j in range(22)])
        self.heel_sites = np.asarray(
            [self.model.site(f"heel_{j}").id for j in range(2)]
        )
        self.wrist_sites = np.asarray(
            [self.model.site(f"wrist_{j}").id for j in range(2)]
        )
        # MuJoCo traverses bodies depth first; SMPL joint IDs are not that order.
        self.qpos_indices = np.asarray(
            [
                self.model.jnt_qposadr[self.model.joint(f"joint_{j}").id] + np.arange(4)
                for j in range(1, 22)
            ]
        )
        self.dof_indices = np.asarray(
            [
                self.model.jnt_dofadr[self.model.joint(f"joint_{j}").id] + np.arange(3)
                for j in range(1, 22)
            ]
        )
        self.qpos0 = self.model.qpos0.copy()
        self.qpos0[:7] = np.r_[self.initial_root, wxyz(root_rotation)]
        self.qpos0[self.qpos_indices] = np.asarray(
            [wxyz(r) for r in self.initial_rotations]
        )
        self.kp = np.repeat(
            np.asarray([220 if j in (1, 2, 4, 5, 7, 8) else 100 for j in range(1, 22)]),
            3,
        )
        self.kd = np.sqrt(self.kp) * 1.5
        self.target = self.initial_rotations.copy()
        self.last_torque = np.zeros(63)
        self.heel_anchor = None
        self.max_heel_error = 0.0
        self.max_projection_delta = 0.0
        self.max_projection_impulse = 0.0
        self.max_wrist_force = 0.0
        self.reset()
        self.initial_mapping_max_m = float(
            np.max(np.linalg.norm(self.points() - self.expected_initial, axis=-1))
        )
        if self.initial_mapping_max_m > 2e-5:
            raise ValueError(
                f"SMPL-H observation mapping differs by {self.initial_mapping_max_m}m"
            )

    def _xml(self):
        tree = ET.Element("mujoco", model="smplh_conditional_tracking_surrogate")
        ET.SubElement(tree, "compiler", angle="radian", inertiafromgeom="true")
        ET.SubElement(
            tree,
            "option",
            timestep=str(1 / 240),
            gravity="0 0 -9.81",
            integrator="implicitfast",
            iterations="30",
        )
        default = ET.SubElement(tree, "default")
        ET.SubElement(default, "joint", damping="1.5", armature="0.025")
        ET.SubElement(
            default, "geom", friction="0.8 0.01 0.001", condim="3", solref="0.015 1"
        )
        world = ET.SubElement(tree, "worldbody")
        ET.SubElement(world, "geom", name="ground", type="plane", size="5 5 .1")
        bodies = []
        for j in range(22):
            parent = world if j == 0 else bodies[self.parents[j]]
            offset = (
                np.zeros(3)
                if j == 0
                else self.joints_rest[j] - self.joints_rest[self.parents[j]]
            )
            body = ET.SubElement(parent, "body", name=f"link_{j}", pos=fmt(offset))
            ET.SubElement(
                body,
                "freejoint" if j == 0 else "joint",
                **(
                    {"name": "root"}
                    if j == 0
                    else {
                        "type": "ball",
                        "name": f"joint_{j}",
                        "limited": "true",
                        "range": "0 2.9",
                    }
                ),
            )
            children = [k for k in range(1, 22) if self.parents[k] == j]
            endpoint = (
                self.joints_rest[children[0]] - self.joints_rest[j]
                if children
                else np.array([0, 0.07, 0])
            )
            radius = (
                0.07 if j in (0, 3, 6, 9) else 0.045 if j in (1, 2, 4, 5) else 0.025
            )
            # Self collisions are omitted in this bounded surrogate; ground contact retained.
            ET.SubElement(
                body,
                "geom",
                type="capsule",
                fromto=fmt(np.r_[np.zeros(3), endpoint]),
                size=str(radius),
                mass=str(8 if j == 0 else 2.6),
                contype="1",
                conaffinity="2",
            )
            bodies.append(body)
            if j in (10, 11):
                side = j - 10
                local = self.heel_local[side]
                ET.SubElement(
                    body, "site", name=f"heel_{side}", pos=fmt(local), size=".005"
                )
                # A capsule through the physical heel point provides a simple sole.
                toe = self.joints_rest[j] - self.joints_rest[self.parents[j]]
                toe = toe / max(np.linalg.norm(toe), 1e-8) * 0.13
                ET.SubElement(
                    body,
                    "geom",
                    type="capsule",
                    fromto=fmt(np.r_[local, local + toe]),
                    size=".015",
                    mass=".4",
                    contype="1",
                    conaffinity="2",
                )
            if j in (20, 21):
                ET.SubElement(
                    body, "site", name=f"wrist_{j - 20}", pos="0 0 0", size=".005"
                )
        # The plane can collide with link geoms, but body-body collisions are disabled.
        world.find("geom").set("contype", "2")
        world.find("geom").set("conaffinity", "1")
        return ET.tostring(tree, encoding="unicode")

    def reset(self, qpos=None, qvel=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.qpos0 if qpos is None else qpos
        self.data.qvel[:] = 0 if qvel is None else qvel
        self.target = self.rotations().copy()
        self.heel_anchor = None
        self.max_heel_error = self.max_projection_delta = (
            self.max_projection_impulse
        ) = self.max_wrist_force = 0.0
        mujoco.mj_forward(self.model, self.data)

    def rotations(self):
        return Rotation.from_quat(
            self.data.qpos[self.qpos_indices][:, [1, 2, 3, 0]]
        ).as_matrix()

    def points(self, include_heels=False):
        local = np.concatenate([self.rotations(), self.hands])
        pose_feature = (local - np.eye(3)).reshape(459)
        vertices = self.shaped_vertices + (self.pose_dirs @ pose_feature).reshape(-1, 3)
        global_r = np.empty((52, 3, 3))
        global_t = np.empty((52, 3))
        global_r[:22] = self.data.xmat[self.body_ids].reshape(22, 3, 3)
        global_t[:22] = self.data.xpos[self.body_ids]
        for j in range(22, 52):
            p = self.parents[j]
            global_r[j] = global_r[p] @ local[j - 1]
            global_t[j] = global_t[p] + global_r[p] @ (
                self.joints_rest[j] - self.joints_rest[p]
            )
        offsets = global_t - np.einsum("jcd,jd->jc", global_r, self.joints_rest)
        blended_r = np.einsum("vj,jcd->vcd", self.weights, global_r)
        skinned = np.einsum("vcd,vd->vc", blended_r, vertices) + self.weights @ offsets
        points = self.skin_regressor @ skinned
        return (points, skinned[self.heel_rows]) if include_heels else points

    def heel_positions(self):
        # The requested anchors belong to actual SMPL-H surface vertices. A
        # rigid site at the corresponding rest location diverges under skinning.
        return (
            self.points(include_heels=True)[1]
            if self.heel_mode == "surface"
            else self.rigid_heel_positions()
        )

    def rigid_heel_positions(self):
        return self.data.site_xpos[self.heel_sites].copy()

    def wrist_positions(self):
        return self.data.site_xpos[self.wrist_sites].copy()

    def enable_heel(self, enabled):
        if enabled and self.heel_anchor is None:
            self.heel_anchor = self.heel_positions()
        elif not enabled:
            self.heel_anchor = None

    def _heel_jacobian(self):
        jac = np.zeros((6, self.model.nv))
        if self.heel_mode == "rigid":
            for side, site in enumerate(self.heel_sites):
                mujoco.mj_jacSite(
                    self.model, self.data, jac[3 * side : 3 * side + 3], None, int(site)
                )
            return jac
        saved = self.data.qpos.copy()
        center = self.heel_positions()
        eps = 1e-5
        for column in range(self.model.nv):
            delta = np.zeros(self.model.nv)
            delta[column] = eps
            self.data.qpos[:] = saved
            mujoco.mj_integratePos(self.model, self.data.qpos, delta, 1.0)
            mujoco.mj_forward(self.model, self.data)
            jac[:, column] = ((self.heel_positions() - center) / eps).ravel()
        self.data.qpos[:] = saved
        mujoco.mj_forward(self.model, self.data)
        return jac

    def _project_heel(self, dt):
        if self.heel_anchor is None:
            return True
        original = self.data.qpos.copy()
        for _ in range(5):
            error = (self.heel_anchor - self.heel_positions()).ravel()
            if np.max(np.abs(error)) < 1e-6:
                break
            jac = self._heel_jacobian()
            delta = np.linalg.lstsq(jac, error, rcond=1e-8)[0]
            mujoco.mj_integratePos(self.model, self.data.qpos, delta, 1.0)
            mujoco.mj_forward(self.model, self.data)
        jac = self._heel_jacobian()
        mass = np.empty((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, self.data, mass)
        minv_jt = np.linalg.solve(mass, jac.T)
        impulse = -np.linalg.solve(
            jac @ minv_jt + np.eye(6) * 1e-10, jac @ self.data.qvel
        )
        self.data.qvel[:] += minv_jt @ impulse
        self.max_projection_impulse = max(
            self.max_projection_impulse, float(np.linalg.norm(impulse))
        )
        correction = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(
            self.model, correction, 1.0, original, self.data.qpos
        )
        self.max_projection_delta = max(
            self.max_projection_delta, float(np.linalg.norm(correction))
        )
        mujoco.mj_forward(self.model, self.data)
        error = float(
            np.max(np.linalg.norm(self.heel_positions() - self.heel_anchor, axis=1))
        )
        self.max_heel_error = max(self.max_heel_error, error)
        return error <= 0.001 and np.isfinite(self.data.qvel).all()

    def advance(self, target: np.ndarray, dt: float, wrist_target=None):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("positive actual dt required")
        self.target = target.copy()
        remaining = dt
        self.max_heel_error = self.max_projection_delta = (
            self.max_projection_impulse
        ) = self.max_wrist_force = 0.0
        energy = 0.0
        while remaining > 1e-10:
            step = min(1 / 240, remaining)
            self.model.opt.timestep = step
            errors = (
                Rotation.from_matrix(self.rotations().transpose(0, 2, 1) @ target)
                .as_rotvec()
                .ravel()
            )
            speed = self.data.qvel[self.dof_indices].ravel()
            self.last_torque = np.clip(self.kp * errors - self.kd * speed, -120, 120)
            self.data.qfrc_applied[:] = 0
            self.data.qfrc_applied[self.dof_indices] = self.last_torque.reshape(21, 3)
            if wrist_target is not None:
                for side, site in enumerate(self.wrist_sites):
                    jac = np.zeros((3, self.model.nv))
                    mujoco.mj_jacSite(self.model, self.data, jac, None, int(site))
                    force = 900 * (
                        wrist_target[side] - self.data.site_xpos[site]
                    ) - 45 * (jac @ self.data.qvel)
                    force *= min(1.0, 300 / max(np.linalg.norm(force), 1e-8))
                    self.max_wrist_force = max(
                        self.max_wrist_force, float(np.linalg.norm(force))
                    )
                    self.data.qfrc_applied[:] += jac.T @ force
            energy += step * np.sum(np.abs(self.last_torque * speed))
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            if (
                not np.isfinite(self.data.qpos).all()
                or not np.isfinite(self.data.qvel).all()
            ):
                return False, energy, "nonfinite_simulation"
            if not self._project_heel(step):
                return False, energy, "heel_projection_failed"
            remaining -= step
        return True, energy, "ok"

    def point_derivatives(self):
        # Derivatives of the exact COCO surface map under the simulator tangent.
        qpos, qvel = self.data.qpos.copy(), self.data.qvel.copy()
        mujoco.mj_forward(self.model, self.data)
        qacc = self.data.qacc.copy()
        if self.heel_anchor is not None:
            # Acceleration must remain tangent to the artificial anchored manifold.
            jac = self._heel_jacobian()
            jac_samples = []
            for sign in (-1, 1):
                self.data.qpos[:] = qpos
                mujoco.mj_integratePos(
                    self.model, self.data.qpos, sign * qvel * 1e-5, 1.0
                )
                mujoco.mj_forward(self.model, self.data)
                jac_samples.append(self._heel_jacobian())
            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)
            jac_dot = (jac_samples[1] - jac_samples[0]) / 2e-5
            mass = np.empty((self.model.nv, self.model.nv))
            mujoco.mj_fullM(self.model, self.data, mass)
            minv_jt = np.linalg.solve(mass, jac.T)
            qacc -= minv_jt @ np.linalg.solve(
                jac @ minv_jt + np.eye(6) * 1e-10, jac @ qacc + jac_dot @ qvel
            )
        center = self.points()
        eps = 1e-4
        samples = []
        for sign in (-1, 1):
            self.data.qpos[:] = qpos
            mujoco.mj_integratePos(
                self.model, self.data.qpos, sign * qvel * eps + 0.5 * qacc * eps**2, 1.0
            )
            mujoco.mj_forward(self.model, self.data)
            samples.append(self.points())
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        return (
            center,
            (samples[1] - samples[0]) / (2 * eps),
            (samples[1] + samples[0] - 2 * center) / eps**2,
        )
