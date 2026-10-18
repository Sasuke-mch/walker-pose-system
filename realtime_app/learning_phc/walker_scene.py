"""Bounded CPU PhysX checks for the isolated synthetic walker asset."""
from isaacgym import gymapi

import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np

from .grasp import free_velocity_predictor, grip_impulse, point_grasp, point_jacobian


H = 1.0 / 240.0
F = np.array([12.0, -8.0, 4.0])
Q = np.array([0.1, -0.2, 0.3, 0.9])
Q = Q / np.linalg.norm(Q)
NAMES = ("load_readback", "com_force_clear", "offset_h", "offset_half_h",
         "freefall", "drop_rest_lift_release")
SIMULATE_CALLS_ISSUED = 0


def vec(value):
    return np.array([value.x, value.y, value.z], dtype=float)


def fields(value, keys=("x", "y", "z")):
    return np.array([value[key] for key in keys], dtype=float)


def rotation(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(q).all() or abs(norm - 1) > 5e-4:
        raise ValueError("invalid xyzw body quaternion: %s" % q.tolist())
    x, y, z, w = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def demand(condition, message):
    if not condition:
        raise RuntimeError(message)


def residual(actual, expected, tolerance, label):
    error = float(np.linalg.norm(np.asarray(actual) - np.asarray(expected)))
    demand(math.isfinite(error) and error <= tolerance,
           "%s actual=%s expected=%s residual=%s tolerance=%s" %
           (label, np.asarray(actual).tolist(), np.asarray(expected).tolist(), error, tolerance))
    return error


class WalkerScene:
    def __init__(self, asset_dir, metadata, dt=H, gravity=False, ground=False,
                 quaternion=(0, 0, 0, 1), base_height=1.0, base_position=None, physics_substeps=1):
        self.gym = gymapi.acquire_gym()
        self.sim = None
        self.metadata, self.dt = metadata, dt
        self.initial_position = np.asarray([0, 0, base_height] if base_position is None else base_position, dtype=float)
        try:
            demand(physics_substeps in (1, 2), 'foundation check supports one or two physical substeps')
            parameters = gymapi.SimParams()
            parameters.dt, parameters.substeps = dt, physics_substeps
            parameters.up_axis = gymapi.UP_AXIS_Z
            parameters.gravity = gymapi.Vec3(0, 0, -9.81 if gravity else 0)
            parameters.use_gpu_pipeline = False
            parameters.physx.use_gpu = False
            parameters.physx.num_threads = 2
            parameters.physx.solver_type = 1
            parameters.physx.num_position_iterations = 8
            parameters.physx.num_velocity_iterations = 2
            parameters.physx.contact_offset = 0.002
            parameters.physx.rest_offset = 0.0
            parameters.physx.bounce_threshold_velocity = 0.2
            self.sim = self.gym.create_sim(0, -1, gymapi.SIM_PHYSX, parameters)
            demand(self.sim is not None, "create_sim failed")
            actual_timing = self.gym.get_sim_params(self.sim)
            demand(abs(actual_timing.dt-dt) <= 1e-8 and actual_timing.substeps == physics_substeps,
                   'simulation timing readback mismatch')
            self.physics_timing = dict(simulate_interval_s=float(actual_timing.dt),
                physics_substeps=int(actual_timing.substeps),
                internal_physics_dt_s=float(actual_timing.dt/actual_timing.substeps))
            if ground:
                plane = gymapi.PlaneParams()
                plane.normal = gymapi.Vec3(0, 0, 1)
                plane.static_friction = plane.dynamic_friction = 0.8
                plane.restitution = 0
                self.gym.add_ground(self.sim, plane)
            self.env = self.gym.create_env(self.sim, gymapi.Vec3(-3, -3, -3),
                                           gymapi.Vec3(3, 3, 3), 1)
            options = gymapi.AssetOptions()
            options.fix_base_link = False
            options.disable_gravity = False
            options.override_com = options.override_inertia = False
            options.linear_damping = options.angular_damping = 0
            options.armature = 0
            options.replace_cylinder_with_capsule = False
            options.slices_per_cylinder = 32
            options.enable_gyroscopic_forces = True
            asset = self.gym.load_asset(self.sim, str(asset_dir), "walker.urdf", options)
            demand(asset is not None, "load_asset failed")
            demand(self.gym.get_asset_rigid_body_count(asset) == 1, "body count is not 1")
            demand(self.gym.get_asset_dof_count(asset) == 0, "unexpected joint DOF")
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(*self.initial_position)
            pose.r = gymapi.Quat(*quaternion)
            self.actor = self.gym.create_actor(self.env, asset, pose, "walker", 0, 0)
            shapes = self.gym.get_actor_rigid_shape_properties(self.env, self.actor)
            demand(len(shapes) == 13, "importer did not preserve 13 collision shapes")
            for shape in shapes:
                shape.friction, shape.restitution = 0.8, 0.0
                shape.contact_offset, shape.rest_offset = 0.002, 0.0
            self.gym.set_actor_rigid_shape_properties(self.env, self.actor, shapes)
            actual_shapes = self.gym.get_actor_rigid_shape_properties(self.env, self.actor)
            shape_settings = np.array([[p.friction, p.restitution, p.contact_offset, p.rest_offset]
                                       for p in actual_shapes])
            demand(np.allclose(shape_settings, [0.8, 0.0, 0.002, 0.0], rtol=1e-5, atol=1e-7),
                   "shape settings did not round-trip: %s" % shape_settings.tolist())
            p = self.gym.get_actor_rigid_body_properties(self.env, self.actor)[0]
            com = p.com
            if hasattr(com, "p"):
                residual(rotation([com.r.x, com.r.y, com.r.z, com.r.w]), np.eye(3),
                         1e-6, "unexpected COM/inertia frame rotation")
                com = com.p
            self.mass, self.com = float(p.mass), vec(com)
            self.inertia = np.array([vec(p.inertia.x), vec(p.inertia.y), vec(p.inertia.z)]).T
            expected = metadata["parameter_provenance"]
            xx, yy, zz, xy, xz, yz = expected["inertia_kg_m2"]["value"]
            tensor = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
            residual(self.mass, expected["mass_kg"]["value"],
                     1e-5 * expected["mass_kg"]["value"], "mass kg")
            residual(self.com, expected["com_m"]["value"], 5e-6, "COM m")
            residual(self.inertia, tensor, 1e-7 + 1e-4*np.linalg.norm(tensor), "inertia kg m2")
            self.body = self.gym.get_actor_rigid_body_handle(self.env, self.actor, 0)
            self.index = self.gym.get_actor_rigid_body_index(self.env, self.actor, 0, gymapi.DOMAIN_ENV)
            self.create_participants()
            demand(self.gym.prepare_sim(self.sim), "prepare_sim failed")
            initial = self.initialize_gravity_state(quaternion, base_height) if gravity else self.read(False)
            residual(initial["p"], self.initial_position, 1e-5, "initial body origin")
            residual(initial["R"], rotation(quaternion), 1e-5, "initial rotation")
            residual(initial["v"], np.zeros(3), 1e-6, "initial linear velocity")
            residual(initial["w"], np.zeros(3), 1e-6, "initial angular velocity")
            self.readback = {"mass_kg": self.mass, "com_m": self.com.tolist(),
                             "inertia_kg_m2_matrix": self.inertia.tolist(), "collision_count": len(shapes),
                             "body_count": 1, "joint_dof_count": 0,
                             "shape_setting_columns": ["friction", "restitution", "contact_offset_m", "rest_offset_m"],
                             "shape_settings_readback": shape_settings.tolist(),
                             "body_property_flags": int(p.flags), "com_binding_type": type(p.com).__name__}
        except Exception:
            self.close()
            raise

    def create_participants(self):
        """Optional extra bodies must be created before prepare_sim."""

    def initialize_gravity_state(self, quaternion, base_height):
        def snapshot(state):
            return {"t_s": state["t"], **{key: state[key].tolist()
                    for key in ("p", "q", "v", "w", "com")}}

        evidence = {"method": "one explicit STATE_ALL write after prepare_sim before simulate",
                    "position_unit": "m", "velocity_units": "m/s and rad/s",
                    "target_p": self.initial_position.tolist(),
                    "target_q_xyzw": list(quaternion), "target_v": [0.0, 0.0, 0.0],
                    "target_w": [0.0, 0.0, 0.0], "explicit_simulate_calls_during_reset": 0}
        self.initialization_evidence = evidence
        try:
            before = self.read(False)
            evidence["before_reset"] = snapshot(before)
            residual(before["t"], 0.0, 1e-9, "reset must occur at scene time zero")
            state = np.zeros(1, dtype=gymapi.RigidBodyState.dtype)
            for key, value in zip("xyz", self.initial_position):
                state["pose"]["p"][key][0] = value
            for key, value in zip("xyzw", quaternion):
                state["pose"]["r"][key][0] = value
            result = self.gym.set_actor_rigid_body_states(self.env, self.actor, state, gymapi.STATE_ALL)
            evidence["set_state_returned"] = bool(result)
            demand(bool(result), "explicit initial STATE_ALL write rejected")
            for index in range(2):
                current = self.read(False)
                evidence["after_reset_read_%d" % index] = snapshot(current)
                residual(current["t"], before["t"], 1e-9, "reset/read changed clock")
                residual(current["p"], self.initial_position, 1e-5, "explicit reset body origin")
                residual(current["R"], rotation(quaternion), 1e-5, "explicit reset rotation")
                residual(current["v"], np.zeros(3), 1e-6, "explicit reset linear velocity")
                residual(current["w"], np.zeros(3), 1e-6, "explicit reset angular velocity")
            return current
        except Exception as error:
            error.initialization_evidence = evidence
            raise

    def close(self):
        if self.sim is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def read(self, contacts=True):
        b = self.gym.get_actor_rigid_body_states(self.env, self.actor, gymapi.STATE_ALL)[0]
        p, q = fields(b["pose"]["p"]), fields(b["pose"]["r"], ("x", "y", "z", "w"))
        v, w = fields(b["vel"]["linear"]), fields(b["vel"]["angular"])
        demand(np.isfinite(np.r_[p, q, v, w]).all(), "nonfinite body state at t=%s: %s" %
               (self.gym.get_sim_time(self.sim), np.r_[p, q, v, w].tolist()))
        r = rotation(q)
        contact = np.full(3, np.nan)
        if contacts:
            contact = fields(self.gym.get_env_rigid_contact_forces(self.env)[self.index])
            demand(np.isfinite(contact).all(), "nonfinite contact readback at t=%s: %s" %
                   (self.gym.get_sim_time(self.sim), contact.tolist()))
        return {"t": float(self.gym.get_sim_time(self.sim)), "p": p, "q": q, "R": r,
                "v": v, "w": w, "com": p+r@self.com, "contact": contact}

    def step(self, force=None, offset=None):
        global SIMULATE_CALLS_ISSUED
        before = self.read(False)
        applied = np.zeros(3) if force is None else np.asarray(force, dtype=float)
        point = np.full(3, np.nan)
        if force is not None:
            point = before["com"] + before["R"] @ (np.zeros(3) if offset is None else offset)
            self.gym.apply_body_force_at_pos(self.env, self.body, gymapi.Vec3(*applied),
                                             gymapi.Vec3(*point), gymapi.ENV_SPACE)
        SIMULATE_CALLS_ISSUED += 1
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        return before, self.read(), applied, point


def row(state, force=None, point=None):
    return np.r_[state["t"], state["p"], state["q"], state["v"], state["w"], state["com"],
                 state["contact"], np.zeros(3) if force is None else force,
                 np.full(3, np.nan) if point is None else point]


def lowest(scene, state):
    result = float("inf")
    for shape in scene.metadata["rods"] + scene.metadata["pads"]:
        a, b = np.asarray(shape["a"]), np.asarray(shape["b"])
        length = np.linalg.norm(b-a)
        axis = state["R"] @ ((b-a)/length)
        center = state["p"] + state["R"] @ ((a+b)/2)
        z = center[2] - length/2*abs(axis[2]) - shape["radius"]*math.sqrt(max(0, 1-axis[2]**2))
        result = min(result, float(z))
    return result


def advance(scene, trace, force=None, offset=None):
    before, state, applied, point = scene.step(force, offset)
    trace.append(row(state, applied, point))
    return before, state


def impulse(asset, meta, traces, label, dt, count, offset=None, clear=False):
    trace = traces.setdefault(label, [])
    with WalkerScene(asset, meta, dt=dt, quaternion=Q) as s:
        start = s.read(False)
        trace.append(row(start))
        expected_j, expected_l = np.zeros(3), np.zeros(3)
        for _ in range(count):
            before, state = advance(s, trace, F, offset)
            expected_j += F*dt
            if offset is not None:
                expected_l += np.cross(before["R"] @ offset, F)*dt
        actual_j = s.mass*(state["v"]-start["v"])
        actual_l = state["R"]@s.inertia@state["R"].T@state["w"]
        actual_l -= start["R"]@s.inertia@start["R"].T@start["w"]
        result = {"J_N_s": expected_j.tolist(), "linear_residual_N_s": residual(
            actual_j, expected_j, 1e-5+0.005*np.linalg.norm(expected_j), "linear impulse"),
            "angular_impulse_N_m_s": expected_l.tolist(), "angular_residual_N_m_s": residual(
                actual_l, expected_l, 1e-6+0.01*np.linalg.norm(expected_l), "angular impulse"),
            "final_omega_rad_s": state["w"].tolist()}
        residual(state["t"]-start["t"], count*dt, 1e-6, "elapsed time s")
        if clear:
            residual(state["w"], np.zeros(3), 1e-4, "COM force angular velocity")
            _, coast = advance(s, trace)
            result["clear_force_residual_N_s"] = residual(
                s.mass*(coast["v"]-state["v"]), np.zeros(3), 1e-5, "force removal")
        return result


def freefall(asset, meta, traces, evidence=None):
    trace = traces.setdefault("freefall", [])
    with WalkerScene(asset, meta, gravity=True) as s:
        if evidence is not None:
            evidence["freefall"] = s.initialization_evidence
        start = s.read(False)
        trace.append(row(start))
        g = np.array([0, 0, -9.81])
        for index in range(24):
            _, state = advance(s, trace)
            residual(state["contact"], np.zeros(3), 0.01, "airborne contact N")
            if index == 0:
                residual(state["t"]-start["t"], H, 1e-6, "first step clock")
                residual(state["v"]-start["v"], g*H, 1e-5, "first step gravity velocity")
                residual(state["com"]-start["com"], 0.5*g*H**2,
                         0.5*9.81*H**2+2e-5, "first step gravity displacement")
        duration = 24*H
        return {"duration_s": duration, "velocity_residual_m_s": residual(
            state["v"]-start["v"], g*duration, 1e-4, "gravity velocity"),
            "position_residual_m": residual(state["com"]-start["com"], 0.5*g*duration**2,
            0.5*9.81*duration*H+2e-5, "gravity position"),
            "elapsed_residual_s": residual(state["t"]-start["t"], duration, 1e-6, "freefall time")}


def ground_check(asset, meta, traces, evidence=None):
    trace = traces.setdefault("drop_rest_lift_release", [])
    with WalkerScene(asset, meta, gravity=True, ground=True, base_height=0.1) as s:
        if evidence is not None:
            evidence["drop_rest_lift_release"] = s.initialization_evidence
        trace.append(row(s.read(False)))
        recent = []
        for i in range(480):
            _, state = advance(s, trace)
            if i >= 420:
                recent.append((state["contact"], np.linalg.norm(state["v"]),
                               np.linalg.norm(state["w"]), lowest(s, state), float(state["R"][2, 2])))
        support = np.mean([item[0] for item in recent], axis=0)
        mg = s.mass*9.81
        result = {"mean_contact_N": support.tolist(), "weight_N": mg,
                  "mean_support_residual_N": residual(support, [0, 0, mg], 0.02*mg+0.1, "static support"),
                  "max_speed_m_s": max(x[1] for x in recent),
                  "max_omega_rad_s": max(x[2] for x in recent),
                  "min_collision_height_m": min(x[3] for x in recent),
                  "max_collision_height_m": max(x[3] for x in recent),
                  "minimum_up_axis_cosine": min(x[4] for x in recent)}
        demand(result["max_speed_m_s"] <= 0.01 and result["max_omega_rad_s"] <= 0.02,
               "walker did not settle within fixed 2 seconds")
        demand(result["min_collision_height_m"] >= -0.001 and result["max_collision_height_m"] <= 0.005,
               "rest collision surface outside stated engineering tolerance")
        demand(result["minimum_up_axis_cosine"] >= math.cos(math.radians(5)),
               "walker settled on its side instead of upright feet")
        initial_lift = state
        external_work = 0.0
        for _ in range(60):
            before, state = advance(s, trace, [0, 0, 2*mg])
            external_work += 2*mg*(state["com"][2]-before["com"][2])
        airborne_contacts = []
        for i in range(24):
            _, state = advance(s, trace)
            if i >= 12:
                airborne_contacts.append(np.linalg.norm(state["contact"]))
        result.update({"lift_force_N": [0, 0, 2*mg], "lift_force_duration_s": 0.25,
                       "declared_external_work_J": external_work,
                       "lift_com_displacement_m": (state["com"]-initial_lift["com"]).tolist(),
                       "released_lowest_height_m": lowest(s, state),
                       "max_released_contact_N": max(airborne_contacts)})
        demand(result["released_lowest_height_m"] > 0.05, "external COM force did not lift walker")
        demand(result["max_released_contact_N"] <= 0.01, "ground contact persists after separation")
        residual(state["t"], 2.35, 1e-6, "drop/lift total time")
        return result


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


class GraspProbeScene(WalkerScene):
    """Two finite-mass probes, not a human or an external support fixture."""

    def __init__(self, *args, centered_lift_probes=False, **kwargs):
        self.centered_lift_probes = centered_lift_probes
        super().__init__(*args, **kwargs)

    def create_participants(self):
        self.probes = []
        for side, delta in (("left", [.012, -.005, .005]), ("right", [-.008, .004, -.003])):
            handle = np.mean(self.metadata["handles_m"][side], axis=0)
            local = np.array([0., 0., -.1])
            radius = .04
            if self.centered_lift_probes:
                local, delta, radius = np.zeros(3), np.array([0., 0., .02]), .005
            position = self.initial_position+handle+delta-local
            opt = gymapi.AssetOptions()
            opt.fix_base_link = False
            opt.linear_damping = opt.angular_damping = 0
            opt.enable_gyroscopic_forces = True
            opt.density = 2.0/(4/3*np.pi*radius**3)
            asset = self.gym.create_sphere(self.sim, radius, opt)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(*position)
            actor = self.gym.create_actor(self.env, asset, pose, side+"_probe", 0, 0)
            props = self.gym.get_actor_rigid_body_properties(self.env, actor)[0]
            self.probes.append(dict(actor=actor, body=self.gym.get_actor_rigid_body_handle(self.env, actor, 0),
                mass=float(props.mass), local_com=vec(props.com),
                inertia=np.array([vec(props.inertia.x), vec(props.inertia.y), vec(props.inertia.z)]).T,
                position=position, local=local, handle=handle))

    def start_probes(self, perturbed=True, asymmetric=False):
        for i, probe in enumerate(self.probes):
            state = np.zeros(1, dtype=gymapi.RigidBodyState.dtype)
            for key, value in zip("xyz", probe["position"]):
                state["pose"]["p"][key][0] = value
            state["pose"]["r"]["w"][0] = 1
            if perturbed:
                state["vel"]["linear"]["x"][0] = .03*(-1)**i
                state["vel"]["angular"]["y"][0] = .1*(-1)**i
            if asymmetric:
                linear = ([.03, .01, -.02], [-.02, -.015, .01])[i]
                angular = ([.1, -.07, .03], [-.04, .08, -.02])[i]
                for key, a, b in zip('xyz', linear, angular):
                    state['vel']['linear'][key][0] = a
                    state['vel']['angular'][key][0] = b
            demand(self.gym.set_actor_rigid_body_states(self.env, probe["actor"], state, gymapi.STATE_ALL),
                   "probe initial state rejected")

    def probe_state(self, probe):
        raw = self.gym.get_actor_rigid_body_states(self.env, probe["actor"], gymapi.STATE_ALL)[0]
        p, q = fields(raw["pose"]["p"]), fields(raw["pose"]["r"], ("x", "y", "z", "w"))
        r = rotation(q)
        return dict(p=p, q=q, R=r, com=p+r@probe["local_com"],
                    v=fields(raw["vel"]["linear"]), w=fields(raw["vel"]["angular"]))

    def system(self):
        states = [self.read(False)]+[self.probe_state(p) for p in self.probes]
        masses = [self.mass]+[p["mass"] for p in self.probes]
        inertias = [self.inertia]+[p["inertia"] for p in self.probes]
        linear, angular, kinetic = np.zeros(3), np.zeros(3), 0.
        for s, mass, tensor in zip(states, masses, inertias):
            iw = s["R"]@tensor@s["R"].T
            linear += mass*s["v"]
            angular += np.cross(s["com"], mass*s["v"])+iw@s["w"]
            kinetic += .5*mass*(s["v"]@s["v"])+.5*s["w"]@iw@s["w"]
        return states, linear, angular, float(kinetic)


def run_grasp_checks(asset_dir, output, constraint=False, predict=False):
    global SIMULATE_CALLS_ISSUED
    asset_dir, output = Path(asset_dir), Path(output)
    meta = json.loads((asset_dir/"asset_metadata.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="failed", scope="two synthetic finite-mass grip probes; not human grasp truth",
                  stiffness_N_m=400., damping_N_s_m=4., dt_s=H, active_steps=120, released_steps=60,
                  human=False, external_support=False, force_application="shared anchor midpoint",
                  angular_dof="free; translational point grip only", input=str(asset_dir))
    report['grip_mode'] = 'constraint_predict' if predict else ('constraint' if constraint else 'spring')
    if constraint:
        report.update(stiffness_N_m=None, damping_N_s_m=None, recovery_time_s=.02,
                      initial_anchors='coincident; nonzero opposing probe velocities retained')
    rows, bodies = [], []
    begun = SIMULATE_CALLS_ISSUED
    try:
        with GraspProbeScene(asset_dir, meta) as s:
            if constraint:
                for probe in s.probes:
                    probe['position'] = s.initial_position+probe['handle']-probe['local']
            s.start_probes(asymmetric=constraint)
            initial, p0, l0, k0 = s.system()
            report['initial_probe_velocities'] = [np.r_[p['v'], p['w']].tolist() for p in initial[1:]]
            grips = [point_grasp(initial[0], p["handle"], initial[i+1], p["local"], 400., 4.)
                     for i, p in enumerate(s.probes)]
            energy0 = k0+(0 if constraint else sum(g["potential_J"] for g in grips))
            stabilization_work = 0.
            maxima = dict(linear=0., angular=0., power_identity=0., positive_energy_step=0., contact=0.)
            previous_energy, released_potential, release_p = energy0, 0., None
            velocity_history = None
            for tick in range(180):
                active = tick < 120
                states, momentum, angular, kinetic = s.system()
                grips = [point_grasp(states[0], p["handle"], states[i+1], p["local"], 400., 4., active)
                         for i, p in enumerate(s.probes)]
                potential = 0 if constraint else sum(g["potential_J"] for g in grips)
                if constraint and active:
                    inverse_mass = np.zeros((18, 18))
                    masses = [s.mass]+[p['mass'] for p in s.probes]
                    inertias = [s.inertia]+[p['inertia'] for p in s.probes]
                    for j, (body, mass, inertia) in enumerate(zip(states, masses, inertias)):
                        inverse_mass[j*6:j*6+3, j*6:j*6+3] = np.eye(3)/mass
                        inverse_mass[j*6+3:j*6+6, j*6+3:j*6+6] = np.linalg.inv(body['R']@inertia@body['R'].T)
                    jacobian, relative = np.zeros((6, 18)), []
                    for j, g in enumerate(grips):
                        a = point_jacobian(g['point'], states[0]['com'])
                        b = point_jacobian(g['point'], states[j+1]['com'])
                        jacobian[j*3:j*3+3, :6] = -a
                        jacobian[j*3:j*3+3, (j+1)*6:(j+2)*6] = b
                        relative.extend(b@np.r_[states[j+1]['v'], states[j+1]['w']]-a@np.r_[states[0]['v'], states[0]['w']])
                    mobility = jacobian@inverse_mass@jacobian.T
                    free = free_velocity_predictor(relative, *(velocity_history or (None, None))) if predict else relative
                    impulse, work = grip_impulse(mobility,
                        np.concatenate([g['anchor_b']-g['anchor_a'] for g in grips]), free, H)
                    velocity_history = (np.asarray(relative), mobility@impulse)
                    stabilization_work += work['stabilization_work_J']
                    for g, force in zip(grips, impulse.reshape(2, 3)/H):
                        g['force_a'], g['force_b'] = force, -force
                if tick == 120:
                    released_potential = 0 if constraint else sum(point_grasp(states[0], p["handle"], states[i+1], p["local"],
                        400., 4.)["potential_J"] for i, p in enumerate(s.probes))
                    release_p = momentum.copy()
                energy = kinetic+potential
                jump = energy-previous_energy
                maxima["positive_energy_step"] = max(maxima["positive_energy_step"], jump)
                previous_energy = energy
                maxima["linear"] = max(maxima["linear"], float(np.linalg.norm(momentum-p0)))
                maxima["angular"] = max(maxima["angular"], float(np.linalg.norm(angular-l0)))
                for i, g in enumerate(grips):
                    if not constraint:
                        maxima["power_identity"] = max(maxima["power_identity"], abs(
                            g["mechanical_power_W"]+g["potential_rate_W"]+g["dissipation_W"]))
                    self_force = gymapi.Vec3(*g["force_a"])
                    peer_force = gymapi.Vec3(*g["force_b"])
                    point = gymapi.Vec3(*g["point"])
                    s.gym.apply_body_force_at_pos(s.env, s.body, self_force, point, gymapi.ENV_SPACE)
                    s.gym.apply_body_force_at_pos(s.env, s.probes[i]["body"], peer_force, point, gymapi.ENV_SPACE)
                rows.append([tick*H, float(active), energy, *momentum, *angular,
                             *[g["error_m"] for g in grips], *[np.linalg.norm(g["force_a"]) for g in grips]])
                bodies.append(np.stack([np.r_[a["p"], a["q"], a["v"], a["w"]] for a in states]))
                SIMULATE_CALLS_ISSUED += 1
                s.gym.simulate(s.sim)
                s.gym.fetch_results(s.sim, True)
                contact = s.gym.get_env_rigid_contact_forces(s.env)
                maxima["contact"] = max(maxima["contact"], max(float(np.linalg.norm(fields(c))) for c in contact))
            states, momentum, angular, kinetic = s.system()
            maxima["linear"] = max(maxima["linear"], float(np.linalg.norm(momentum-p0)))
            maxima["angular"] = max(maxima["angular"], float(np.linalg.norm(angular-l0)))
            report.update(initial_energy_J=energy0, final_kinetic_J=kinetic,
                          release_removed_spring_energy_J=released_potential, maxima=maxima,
                          probe_masses_kg=[p["mass"] for p in s.probes],
                          released_momentum_residual_N_s=float(np.linalg.norm(momentum-release_p)))
            if constraint:
                maxima['power_identity'] = None
                report['numerical_stabilization_work_J'] = stabilization_work
            demand(maxima["linear"] <= 1e-4 and maxima["angular"] <= 1e-4, "internal grasp momentum drift")
            if not constraint:
                demand(maxima["power_identity"] <= 1e-9, "grasp power identity failure")
            tolerance = 1e-5 if constraint else 1e-4+0.01*energy0
            total_tolerance = 1e-5 if constraint else 1e-4
            demand(maxima["positive_energy_step"] <= tolerance and kinetic <= energy0+total_tolerance,
                   "unexplained grasp energy creation")
            demand(maxima["contact"] <= 1e-5, "probe test contains unplanned collisions")
            demand(report["released_momentum_residual_N_s"] <= 1e-4, "released grip still changes momentum")
            report["status"] = "conditional_grasp_conservation_passed"
    except Exception as error:
        report.update(reason=str(error), failure_traceback=traceback.format_exc())
    finally:
        report["explicit_simulate_calls_issued"] = SIMULATE_CALLS_ISSUED-begun
        report["trace_columns"] = ["t", "active", "energy_J", "px", "py", "pz", "Lx", "Ly", "Lz",
                                   "left_gap_m", "right_gap_m", "left_force_N", "right_force_N"]
        np.savez_compressed(output/"traces.npz", diagnostics=np.asarray(rows), bodies=np.asarray(bodies))
        (output/"results.json").write_text(json.dumps(clean(report), ensure_ascii=False, indent=2,
                                                     allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(clean(report), ensure_ascii=False))
    return 0 if report["status"] == "conditional_grasp_conservation_passed" else 1


def run_grasp_lift_check(asset_dir, output):
    """Lift via forces on two finite-mass probes, never directly on the walker."""
    asset_dir, output = Path(asset_dir), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    meta = json.loads((asset_dir/'asset_metadata.json').read_text(encoding='utf-8'))
    report = dict(status='failed', scope='synthetic two-probe load transfer, not human strength',
                  dt_s=H, force_on_walker_N=[0., 0., 0.], grip_stiffness_N_m=400., grip_damping_N_s_m=4.,
                  fixture='two 2kg/5mm-radius probes; COM grip anchors, equal20mm initial Z separation; one-axis lift')
    rows, states_saved = [], []
    try:
        with GraspProbeScene(asset_dir, meta, gravity=True, ground=True, base_height=0., centered_lift_probes=True) as s:
            s.start_probes(perturbed=False)
            states, p0, _, _ = s.system()
            total_mass = s.mass+sum(p['mass'] for p in s.probes)
            gravity = np.array([0., 0., -total_mass*9.81])
            force = np.array([0., 0., total_mass*9.81/2+12.])
            impulse, work, maximum_residual = np.zeros(3), 0., 0.
            airborne_contact, other_contact, actuated_contact, lift_height = [], [], [], None
            for tick in range(168):
                states, _, _, _ = s.system()
                command = force if tick < 120 else np.zeros(3)
                for i, probe in enumerate(s.probes):
                    g = point_grasp(states[0], probe['handle'], states[i+1], probe['local'], 400., 4.)
                    for body, f in ((s.body, g['force_a']), (probe['body'], g['force_b'])):
                        s.gym.apply_body_force_at_pos(s.env, body, gymapi.Vec3(*f), gymapi.Vec3(*g['point']), gymapi.ENV_SPACE)
                    s.gym.apply_body_force_at_pos(s.env, probe['body'], gymapi.Vec3(*command),
                                                  gymapi.Vec3(*states[i+1]['com']), gymapi.ENV_SPACE)
                s.gym.simulate(s.sim)
                s.gym.fetch_results(s.sim, True)
                after, momentum, _, _ = s.system()
                contact = sum((fields(f) for f in s.gym.get_env_rigid_contact_forces(s.env)), np.zeros(3))
                impulse += (gravity+2*command+contact)*H
                maximum_residual = max(maximum_residual, float(np.linalg.norm(momentum-p0-impulse)))
                work += sum(float(command@(after[i+1]['com']-states[i+1]['com'])) for i in range(2))
                walker = s.read()
                if tick == 119:
                    lift_height = lowest(s, walker)
                ground_force, non_ground_force = 0., 0.
                for c in s.gym.get_env_rigid_contacts(s.env):
                    ids = [int(c['body0']), int(c['body1'])]
                    if s.index in ids:
                        if -1 in ids:
                            ground_force += abs(float(c['lambda']))
                        else:
                            non_ground_force += abs(float(c['lambda']))
                if tick < 120:
                    actuated_contact.append(non_ground_force)
                else:
                    airborne_contact.append(ground_force)
                    other_contact.append(non_ground_force)
                rows.append(np.r_[(tick+1)*H, lowest(s, walker), walker['contact'], command, momentum, impulse])
                states_saved.append([np.r_[p['p'], p['q'], p['v'], p['w']] for p in after])
            report.update(simulate_calls=len(rows), force_each_probe_N=force.tolist(), force_duration_s=.5,
                total_mass_kg=total_mass, lift_lowest_height_m=lift_height,
                after_force_removed_lowest_height_m=lowest(s, s.read()),
                max_airborne_walker_ground_normal_N=max(airborne_contact),
                max_airborne_walker_probe_normal_N=max(other_contact), momentum_balance_max_residual_N_s=maximum_residual,
                max_actuated_walker_probe_normal_N=max(actuated_contact),
                external_work_J=work)
            demand(lift_height > .05, 'probe forces did not lift the free walker')
            demand(max(airborne_contact) <= .01, 'airborne walker retains contact')
            demand(max(actuated_contact) <= .01, 'actuated lift proof contains direct probe collisions')
            demand(maximum_residual <= .02, 'gravity/ground/declared input momentum balance failed')
            report['status'] = 'conditional_grasp_load_transfer_passed'
    except Exception as error:
        report.update(reason=str(error), failure_traceback=traceback.format_exc())
    finally:
        report['completed_steps_recorded'] = len(rows)
        report['trace_columns'] = ['t', 'walker_lowest_z', 'contact_x', 'contact_y', 'contact_z',
            'force_each_probe_x', 'force_each_probe_y', 'force_each_probe_z', 'px', 'py', 'pz', 'Jx', 'Jy', 'Jz']
        np.savez_compressed(output/'traces.npz', diagnostics=np.asarray(rows), bodies=np.asarray(states_saved))
        (output/'results.json').write_text(json.dumps(clean(report), indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(clean(report)))
    return 0 if report['status'] == 'conditional_grasp_load_transfer_passed' else 1


def run(asset_dir, output):
    asset_dir, output = Path(asset_dir), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    meta = json.loads((asset_dir/"asset_metadata.json").read_text(encoding="utf-8"))
    report = {"status": "failed", "scope": "synthetic conditional passive physics; not actual equipment",
              "input": str(asset_dir), "contains_synthetic_fixture": meta["contains_synthetic_fixture"],
              "simulator_loaded": None, "inertia_readback_verified": None,
              "dt_s": H, "substeps": 1, "solver_position_iterations": 8, "solver_velocity_iterations": 2,
              "contact_offset_m": 0.002, "rest_offset_m": 0.0, "friction_assumed": 0.8,
              "restitution": 0.0, "linear_angular_damping": 0.0, "cylinder_slices": 32,
              "trace_columns": ["t", "px", "py", "pz", "qx", "qy", "qz", "qw", "vx", "vy", "vz",
              "wx", "wy", "wz", "com_x", "com_y", "com_z", "contact_x", "contact_y", "contact_z",
              "applied_Fx", "applied_Fy", "applied_Fz", "point_x", "point_y", "point_z"],
              "trace_missing": "initial contact and absent application points are NaN, not measured zero",
              "cases": {name: {"status": "not_run"} for name in NAMES}}
    traces, stage = {}, "load_readback"
    try:
        demand(meta["status"] == "asset_generated_unvalidated" and meta["contains_synthetic_fixture"],
               "expected frozen named synthetic asset")
        with WalkerScene(asset_dir, meta, quaternion=Q) as scene:
            report["readback"] = scene.readback
        report["simulator_loaded"] = report["inertia_readback_verified"] = True
        report["cases"][stage] = {"status": "passed"}
        stage = "com_force_clear"
        result = impulse(asset_dir, meta, traces, stage, H, 1, clear=True)
        report["cases"][stage] = {"status": "passed", **result}
        stage = "offset_h"
        offset = np.array([0.10, 0.03, -0.02])
        coarse = impulse(asset_dir, meta, traces, stage, H, 1, offset)
        report["cases"][stage] = {"status": "passed", **coarse}
        stage = "offset_half_h"
        fine = impulse(asset_dir, meta, traces, stage, H/2, 2, offset)
        fine["halfstep_omega_difference_rad_s"] = residual(
            coarse["final_omega_rad_s"], fine["final_omega_rad_s"],
            1e-5+0.01*np.linalg.norm(fine["final_omega_rad_s"]), "step-halving omega")
        report["cases"][stage] = {"status": "passed", **fine}
        stage = "freefall"
        report["cases"][stage] = {"status": "passed", **freefall(asset_dir, meta, traces)}
        stage = "drop_rest_lift_release"
        report["cases"][stage] = {"status": "passed", **ground_check(asset_dir, meta, traces)}
        report["status"] = "conditional_passive_physics_passed"
    except Exception as error:
        report["cases"][stage] = {"status": "failed", "reason": str(error)}
        report["failure_traceback"] = traceback.format_exc()
    finally:
        np.savez_compressed(output/"traces.npz", **{key: np.asarray(value, dtype=float) for key, value in traces.items()})
        (output/"results.json").write_text(json.dumps(clean(report), ensure_ascii=False,
            indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "cases": {k: v["status"] for k, v in report["cases"].items()},
                      "output": str(output)}, ensure_ascii=False))
    return 0 if report["status"] == "conditional_passive_physics_passed" else 1


def run_gravity_retry(asset_dir, output):
    asset_dir, output = Path(asset_dir), Path(output)
    previous_dir = asset_dir.parent / "passive_physx_v1"
    previous = json.loads((previous_dir/"results.json").read_text(encoding="utf-8"))
    historical = ("load_readback", "com_force_clear", "offset_h", "offset_half_h")
    demand(all(previous["cases"][name]["status"] == "passed" for name in historical),
           "expected the four preserved v1 passes; do not silently rerun them")
    demand(previous["cases"]["freefall"]["status"] == "failed" and
           previous["cases"]["drop_rest_lift_release"]["status"] == "not_run",
           "previous evidence differs from dispatched failure")
    meta = json.loads((asset_dir/"asset_metadata.json").read_text(encoding="utf-8"))
    demand(meta["contains_synthetic_fixture"], "expected the original named synthetic asset")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "scope": "gravity reset and two unfinished cases only",
              "input": str(asset_dir), "contains_synthetic_fixture": True,
              "previous_evidence": {"path": str(previous_dir),
                 "status": previous["status"], "cases_not_rerun": list(historical)},
              "fixed_parameters": {key: previous[key] for key in ("dt_s", "substeps",
                 "solver_position_iterations", "solver_velocity_iterations", "contact_offset_m",
                 "rest_offset_m", "friction_assumed", "restitution", "linear_angular_damping", "cylinder_slices")},
              "trace_columns": previous["trace_columns"], "trace_missing": previous["trace_missing"],
              "planned_max_explicit_simulate_calls": 588,
              "cases": {name: {"status": "not_run"} for name in ("freefall", "drop_rest_lift_release")},
              "initialization": {}}
    traces = {}
    started = SIMULATE_CALLS_ISSUED
    stage = "freefall"
    try:
        result = freefall(asset_dir, meta, traces, report["initialization"])
        report["cases"][stage] = {"status": "passed", **result}
        stage = "drop_rest_lift_release"
        result = ground_check(asset_dir, meta, traces, report["initialization"])
        report["cases"][stage] = {"status": "passed", **result}
        report["status"] = "gravity_and_ground_checks_passed"
    except Exception as error:
        report["cases"][stage] = {"status": "failed", "reason": str(error)}
        if hasattr(error, "initialization_evidence"):
            report["initialization"][stage] = error.initialization_evidence
        report["failure_traceback"] = traceback.format_exc()
    finally:
        report["explicit_simulate_calls_issued"] = SIMULATE_CALLS_ISSUED-started
        report["completed_steps_recorded_in_traces"] = sum(max(0, len(v)-1) for v in traces.values())
        np.savez_compressed(output/"traces.npz", **{key: np.asarray(value, dtype=float) for key, value in traces.items()})
        (output/"results.json").write_text(json.dumps(clean(report), ensure_ascii=False,
            indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "cases": {k: v["status"] for k, v in report["cases"].items()},
                      "explicit_simulate_calls_issued": report["explicit_simulate_calls_issued"],
                      "output": str(output)}, ensure_ascii=False))
    return 0 if report["status"] == "gravity_and_ground_checks_passed" else 1


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--predict-grasp-checks":
        raise SystemExit(run_grasp_checks(sys.argv[2], sys.argv[3], constraint=True, predict=True))
    if len(sys.argv) == 4 and sys.argv[1] == "--constraint-grasp-checks":
        raise SystemExit(run_grasp_checks(sys.argv[2], sys.argv[3], constraint=True))
    if len(sys.argv) == 4 and sys.argv[1] == "--grasp-lift":
        raise SystemExit(run_grasp_lift_check(sys.argv[2], sys.argv[3]))
    if len(sys.argv) == 4 and sys.argv[1] == "--grasp-checks":
        raise SystemExit(run_grasp_checks(sys.argv[2], sys.argv[3]))
    if len(sys.argv) == 4 and sys.argv[1] == "--gravity-retry":
        raise SystemExit(run_gravity_retry(sys.argv[2], sys.argv[3]))
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m learning_phc.walker_scene [--gravity-retry] ASSET_DIRECTORY NEW_OUTPUT_DIRECTORY")
    raise SystemExit(run(sys.argv[1], sys.argv[2]))
