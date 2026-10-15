"""Stage-conditioned world q/v/a with model-level ground constraints.

The supplied camera transforms are an explicit capture-protocol assumption, not
independent visual odometry. Pixels are used once per posterior; hard contact
conditions change the complete model, never the displayed vertices alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from pose_app.smplh_motion_tracker import MotionPose, SMPLHMotionTracker, retract, right_jacobian
from pose_app.smplh_ground_constraints import foot_geometry, ground_inequalities


def _minimize_scaled_pose(fun, initial_delta, *, jac, bounds, constraints, covariance, options):
    """Change optimizer coordinates, preserving the full physical MAP problem."""
    scale = np.sqrt(np.diag(covariance))
    scaled_bounds = [(None if lower is None else lower/scale[i],
                      None if upper is None else upper/scale[i])
                     for i, (lower, upper) in enumerate(bounds)]
    result = minimize(lambda coordinates: fun(coordinates*scale), initial_delta/scale,
        jac=lambda coordinates: jac(coordinates*scale)*scale, method="SLSQP", bounds=scaled_bounds,
        constraints={"type": constraints["type"],
                     "fun": lambda coordinates: constraints["fun"](coordinates*scale),
                     "jac": lambda coordinates: constraints["jac"](coordinates*scale)*scale[None]},
        options=options)
    # All callers, feasibility checks, and incumbent deltas use physical units.
    result.x = result.x*scale
    if hasattr(result, "jac"):
        result.jac = result.jac/scale
    return result


@dataclass(frozen=True)
class WorldGroundConfig:
    ground_tolerance_m: float = .005
    max_heel_displacement_m: float = .010
    maximum_iterations: int = 30
    bootstrap_iterations: int = 80
    optimizer_objective_tolerance: float = .01
    feasibility_tolerance_m: float = .0002
    heel_velocity_sigma_m_s: float = .02
    heel_acceleration_sigma_m_s2: float = .20

    def __post_init__(self):
        for key in self.__dataclass_fields__:
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")


class SMPLHWorldMotionTracker(SMPLHMotionTracker):
    def __init__(self, model, cal, config, rotation, translation_m, sole_indices,
                 ground_config=None, constrain_ground=True):
        super().__init__(model, cal, config)
        self.ground_config = ground_config or WorldGroundConfig()
        self.constrain_ground = constrain_ground
        self.sole_indices = sole_indices
        self.camera_rotation = np.asarray(rotation, float)
        self.camera_translation = np.asarray(translation_m, float)
        # SMPLH rotates about its shaped rest pelvis J0, not the origin.
        self.pelvis_offset = model.evaluate(self.pose)["joints_m"][0, 0] - self.pose.translation_m
        self.pose = MotionPose(self.camera_rotation @ (self.pose.translation_m + self.pelvis_offset)
                               + self.camera_translation - self.pelvis_offset,
                               self.camera_rotation @ self.pose.root_rotation, self.pose.latent.copy())
        transform = np.eye(3*self.n)
        transform[:3, :3] = self.camera_rotation
        transform[self.n:self.n+3, self.n:self.n+3] = self.camera_rotation
        transform[2*self.n:2*self.n+3, 2*self.n:2*self.n+3] = self.camera_rotation
        self.covariance = transform @ self.covariance @ transform.T
        self.initial_covariance = self.covariance.copy()
        self.last_accepted_pose = self.pose.copy()
        self.stage2 = False
        self.heel_anchors = None
        self.sole_anchors = None
        self.contact_diagnostic = {"status": "not_requested"}
        self.camera_pose_status = "unknown"

    def set_scene_frame(self, rotation, translation_m, stage, camera_pose_status):
        self.camera_rotation = np.asarray(rotation, float)
        self.camera_translation = np.asarray(translation_m, float)
        if (self.camera_rotation.shape != (3, 3) or self.camera_translation.shape != (3,)
                or not np.isfinite(self.camera_rotation).all() or not np.isfinite(self.camera_translation).all()
                or not np.allclose(self.camera_rotation.T @ self.camera_rotation, np.eye(3), atol=1e-5)):
            raise ValueError("a finite rigid world-from-camera transform is required")
        self.stage2 = stage == "stage2_feet_static_walker_moving"
        # Transition gaps cannot start a new anchor and accumulate hidden slip.
        if stage == "stage1_walker_static_human_moving":
            self.heel_anchors = None
            self.sole_anchors = None
        self.camera_pose_status = camera_pose_status

    def _pixels(self, points):
        camera = (points - self.camera_translation) @ self.camera_rotation
        return super()._pixels(camera)

    def _geometry(self, vertices):
        return foot_geometry(vertices, self.sole_indices)

    def _conditions(self, geometry):
        result = ground_inequalities(geometry, stage2=self.stage2, heel_anchors_m=self.heel_anchors if self.stage2 else None,
                                   sole_anchors_m=self.sole_anchors if self.stage2 else None,
                                   ground_tolerance_m=self.ground_config.ground_tolerance_m,
                                   max_heel_displacement_m=self.ground_config.max_heel_displacement_m)
        if not self.stage2:
            # Controlled walker walking assumption: one supporting foot, while
            # the other can swing. This is not an inferred contact label.
            result = np.concatenate((result, (self.ground_config.ground_tolerance_m
                                     - np.min(geometry["sole_min_m"], axis=-1))[..., None]), axis=-1)
        return result

    def _solver_conditions(self, geometry):
        """Smooth per-vertex form; keep a small interior numerical margin.

        Lower bounds and point balls are smooth per-vertex inequalities. The
        upper bound uses the current sole minimum, allowing its active vertex
        to change with the pose. Outside Stage2 each foot is solved separately
        and the lower-cost feasible branch is selected under the same prior. A
        vertex fixed from the prediction would impose a stronger condition than
        the public physical check and could oppose the current-frame pixels.
        """
        sole = geometry["sole_points_m"]
        tolerance = self.ground_config.ground_tolerance_m-self.ground_config.feasibility_tolerance_m
        radius = self.ground_config.max_heel_displacement_m-self.ground_config.feasibility_tolerance_m
        result = [(sole[..., 2]+tolerance).reshape(sole.shape[:-3]+(-1,))]
        support = geometry["sole_min_m"]
        if not self.stage2:
            side = getattr(self, "_solver_support_side", None)
            support = np.min(support, axis=-1, keepdims=True) if side is None else support[..., side:side+1]
        result.append(tolerance-support)
        if self.stage2 and self.heel_anchors is not None:
            heel_squared = np.sum((geometry["heel_m"]-self.heel_anchors)**2, axis=-1)
            sole_squared = np.sum((sole-self.sole_anchors)**2, axis=-1)
            result.extend(((radius**2-heel_squared)/(2*radius),
                           ((radius**2-sole_squared)/(2*radius)).reshape(sole.shape[:-3]+(-1,))))
        return np.concatenate(result, axis=-1)

    def _update(self, pixels, valid, reasons):
        if not self.constrain_ground:
            return super()._update(pixels, valid, reasons)
        return self._constrained_update(pixels, valid, reasons)

    def _constrained_update(self, pixels, valid, reasons, initialize=False):
        n, P = self.n, self.covariance.copy()
        prior = self.pose.copy()
        initial_geometry = self._geometry(self.model.evaluate(prior, mesh=True)["vertices_m"][0])
        used = valid.copy()
        if not initialize and np.any(used, axis=0).sum() < self.config.min_observed_joints:
            reasons[used] = "insufficient_joint_support"
            self.contact_diagnostic = {"status": "insufficient_joint_support"}
            return False, np.zeros_like(used), "insufficient_joint_support"
        # Frame0 pixels were already used to refine its pose. Do not reuse them.
        selected = np.repeat(used.ravel(), 2) if not initialize else np.zeros(48, bool)
        precision = np.linalg.solve(P[:n, :n], np.eye(n))
        target = pixels[..., :2].reshape(-1)[selected]
        cache = {}
        incumbent = {}

        def evaluate(delta, jacobian=False):
            if not cache or not np.array_equal(cache["delta"], delta):
                pose = retract(prior, delta)
                data = self.model.evaluate(pose, mesh=True)
                projected = self._pixels(data["coco_m"])
                cache.clear()
                cache.update(delta=delta.copy(), pose=pose, data=data,
                             pixels=None if projected is None else projected[0].reshape(-1),
                             geometry=self._geometry(data["vertices_m"][0]))
            if jacobian and "pixel_jacobian" not in cache:
                eps = 1e-3
                offsets = np.r_[delta[None] + np.eye(n)*eps, delta[None] - np.eye(n)*eps]
                data = self.model.evaluate_batch(prior.translation_m + offsets[:, :3],
                         prior.root_rotation @ Rotation.from_rotvec(offsets[:, 3:6]).as_matrix(),
                         prior.latent + offsets[:, 6:], mesh=True)
                projected = self._pixels(data["coco_m"])
                if projected is None:
                    cache["pixel_jacobian"] = np.zeros((48, n))
                else:
                    cache["pixel_jacobian"] = ((projected[:n]-projected[n:])/(2*eps)).reshape(n, -1).T
                conditions = self._solver_conditions(self._geometry(data["vertices_m"]))
                cache["condition_jacobian"] = ((conditions[:n]-conditions[n:])/(2*eps)).T
            return cache

        def objective(delta):
            val = evaluate(delta)
            if val["pixels"] is None:
                return 1e12 + delta @ delta
            errors = (val["pixels"][selected]-target).reshape(-1, 2)
            norm = np.linalg.norm(errors, axis=1)
            data_cost = np.sum(np.where(norm <= self.config.huber_px, .5*norm**2,
                                      self.config.huber_px*(norm-.5*self.config.huber_px)))/self.config.pixel_sigma_px**2
            cost = data_cost + .5*delta @ precision @ delta + (0. if initialize else .5*np.sum(val["pose"].latent**2)/self.config.latent_prior_sigma**2)
            branch_feasible = (self.stage2 or val["geometry"]["sole_min_m"][self._solver_support_side]
                               <= self.ground_config.ground_tolerance_m+1e-6)
            if np.isfinite(cost) and branch_feasible and self._conditions(val["geometry"]).min() >= -1e-6:
                if not incumbent or cost < incumbent["cost"]:
                    incumbent.update(delta=delta.copy(), cost=float(cost))
            return cost

        def gradient(delta):
            val = evaluate(delta, True)
            if val["pixels"] is None:
                return 2*delta
            residual = (val["pixels"][selected]-target).reshape(-1, 2)
            norm = np.linalg.norm(residual, axis=1)
            robust = np.minimum(1., self.config.huber_px/np.maximum(norm, 1e-12))
            result = val["pixel_jacobian"][selected].T @ (residual*robust[:, None]).ravel()/self.config.pixel_sigma_px**2 + precision @ delta
            if not initialize:
                result[6:] += val["pose"].latent/self.config.latent_prior_sigma**2
            return result

        def conditions(delta):
            # SLSQP constraint scale is centimetres, with a physical final check.
            return self._solver_conditions(evaluate(delta)["geometry"])/.01

        def condition_jacobian(delta):
            return evaluate(delta, True)["condition_jacobian"]/.01

        bounds = [(None, None)]*6 + [(-self.config.max_abs_latent-z, self.config.max_abs_latent-z) for z in prior.latent]
        starts = [np.zeros(n), np.r_[self.last_accepted_pose.translation_m-prior.translation_m,
            Rotation.from_matrix(prior.root_rotation.T @ self.last_accepted_pose.root_rotation).as_rotvec(),
            self.last_accepted_pose.latent-prior.latent]]
        if not self.stage2:
            vertical = np.zeros(n)
            vertical[2] = self.ground_config.ground_tolerance_m-self.ground_config.feasibility_tolerance_m-initial_geometry["sole_min_m"].min()
            starts.append(vertical)
        def physically_feasible(delta):
            return self._conditions(evaluate(delta)["geometry"]).min() >= -1e-6
        candidates = []
        branches = (None,) if self.stage2 else (0, 1)
        for side in branches:
            self._solver_support_side = side
            cache.clear()
            incumbent.clear()
            feasible_starts = [(objective(start), start) for start in starts
                               if physically_feasible(start) and conditions(start).min() >= -1e-6]
            bootstrap = None
            if not feasible_starts:
                # Find a contact-only warm start for this support branch. No
                # pixels or covariance are consumed by the feasibility solve.
                bootstrap = _minimize_scaled_pose(lambda delta: .5*delta @ precision @ delta, starts[0],
                    jac=lambda delta: precision @ delta, bounds=bounds, covariance=P[:n, :n],
                    constraints={"type": "ineq", "fun": conditions, "jac": condition_jacobian},
                    options={"maxiter": self.ground_config.bootstrap_iterations,
                             "ftol": self.ground_config.optimizer_objective_tolerance})
                if physically_feasible(bootstrap.x) and conditions(bootstrap.x).min() >= -1e-6:
                    feasible_starts.append((objective(bootstrap.x), bootstrap.x.copy()))
            initial_cost, initial_delta = min(feasible_starts, key=lambda item: item[0]) if feasible_starts else (np.inf, starts[0])
            solution = _minimize_scaled_pose(objective, initial_delta, jac=gradient, bounds=bounds,
                covariance=P[:n, :n],
                constraints={"type": "ineq", "fun": conditions, "jac": condition_jacobian},
                options={"maxiter": self.ground_config.maximum_iterations,
                         "ftol": self.ground_config.optimizer_objective_tolerance})
            val = evaluate(solution.x, True)
            slack = self._conditions(val["geometry"])
            feasible = val["pixels"] is not None and np.isfinite(solution.x).all() and slack.min() >= -1e-6
            optimizer_last_slack = float(slack.min())
            optimizer_last_geometry = {key: val["geometry"][key].tolist() for key in ("sole_min_m", "heel_m")}
            converged = bool(solution.success and feasible)
            chosen_delta, chosen_cost = solution.x, float(solution.fun)
            selected_estimate = "optimizer_last_iterate"
            if (not converged and getattr(solution, "status", None) == 9 and incumbent
                and np.isfinite(initial_cost) and incumbent["cost"] < initial_cost-self.ground_config.optimizer_objective_tolerance):
                chosen_delta, chosen_cost = incumbent["delta"], incumbent["cost"]
                val = evaluate(chosen_delta, True)
                slack = self._conditions(val["geometry"])
                feasible = bool(slack.min() >= -1e-6)
                selected_estimate = "best_feasible_improving_current_frame_iterate"
            # A finite-budget estimate must improve a known feasible start.
            # Failure remains explicit, without information shrink or published
            # derivatives, even if this branch has the best physical objective.
            feasible_budget = bool(not solution.success and getattr(solution, "status", None) == 9
                              and feasible and np.isfinite(initial_cost)
                              and chosen_cost < initial_cost-self.ground_config.optimizer_objective_tolerance)
            accepted = bool(feasible and (converged or feasible_budget))
            candidates.append(dict(side=side, solution=solution, bootstrap=bootstrap,
                initial_cost=initial_cost, delta=chosen_delta.copy(), cost=chosen_cost,
                converged=converged, accepted=accepted, selected_estimate=selected_estimate,
                last_slack=optimizer_last_slack, last_geometry=optimizer_last_geometry))
        chosen = min(candidates, key=lambda item: (not item["accepted"], item["cost"]))
        self._solver_support_side = chosen["side"]
        cache.clear()
        chosen_delta, chosen_cost = chosen["delta"], chosen["cost"]
        solution, bootstrap, initial_cost = chosen["solution"], chosen["bootstrap"], chosen["initial_cost"]
        converged, accepted, selected_estimate = chosen["converged"], chosen["accepted"], chosen["selected_estimate"]
        val = evaluate(chosen_delta, True)
        slack = self._conditions(val["geometry"])
        self.contact_diagnostic = {"status": ("constrained" if converged else "feasible_unconverged") if accepted else "constraint_conflict",
            "optimizer_success": bool(solution.success), "optimizer_message": str(solution.message),
            "iterations": int(solution.nit), "minimum_slack_m": float(slack.min()),
            "objective": float(chosen_cost), "feasible_start_objective": None if not np.isfinite(initial_cost) else float(initial_cost), "stage2": self.stage2,
            "selected_estimate": selected_estimate, "optimizer_last_objective": float(solution.fun),
            "optimizer_last_minimum_slack_m": chosen["last_slack"], "optimizer_last_geometry": chosen["last_geometry"],
            "support_selection": "current_pose_sole_minimum", "selected_support_side": chosen["side"],
            "support_extra_solver_calls": len(candidates)-1,
            "map_solver_calls": len(candidates),
            "feasibility_solver_calls": sum(item["bootstrap"] is not None for item in candidates),
            "support_branch_results": [{"side": item["side"], "accepted": item["accepted"],
                "optimizer_success": bool(item["solution"].success), "objective": float(item["cost"]),
                "iterations": int(item["solution"].nit), "optimizer_message": str(item["solution"].message),
                "minimum_slack_m": float(item["last_slack"]),
                "bootstrap_iterations": 0 if item["bootstrap"] is None else int(item["bootstrap"].nit)}
                for item in candidates],
            "heel_anchors_m": None if self.heel_anchors is None else self.heel_anchors.tolist()}
        if bootstrap is not None:
            self.contact_diagnostic["feasibility_bootstrap"] = {"optimizer_success": bool(bootstrap.success),
                "iterations": int(bootstrap.nit), "message": str(bootstrap.message)}
        if not accepted:
            reasons[used] = "constraint_conflict"
            return False, np.zeros_like(used), "constraint_conflict"
        delta = chosen_delta
        conditional = P[:, :n] @ precision
        correction = conditional @ delta
        self.pose = val["pose"]
        self.velocity += correction[n:2*n]
        self.acceleration += correction[2*n:]
        # One Laplace posterior at a converged MAP, not one shrink per iteration.
        J = val["pixel_jacobian"][selected]
        residual = (val["pixels"][selected]-target).reshape(-1, 2)
        weights = np.repeat(np.minimum(1., self.config.huber_px/np.maximum(np.linalg.norm(residual, axis=1), 1e-12)), 2)
        Hprior = np.zeros((0 if initialize else n-6, n))
        if not initialize:
            Hprior[:, 6:] = np.eye(n-6)
        solver_slack = self._solver_conditions(val["geometry"])
        active = solver_slack < .0005
        Hq = np.r_[J, Hprior, val["condition_jacobian"][active]]
        H = np.c_[Hq, np.zeros((len(Hq), 2*n))]
        variance = np.r_[self.config.pixel_sigma_px**2/weights,
                         np.repeat(self.config.latent_prior_sigma**2, len(Hprior)), np.repeat(.001**2, active.sum())]
        gain = np.linalg.solve(H @ P @ H.T + np.diag(variance), H @ P).T
        A = np.eye(3*n)-gain @ H
        posterior = A @ P @ A.T + (gain*variance) @ gain.T if converged else P
        reset = np.eye(3*n); reset[3:6, 3:6] = right_jacobian(delta[3:6])
        self.covariance = reset @ posterior @ reset.T
        self.covariance = (self.covariance+self.covariance.T)/2
        if self.stage2 and self.heel_anchors is None:
            self.heel_anchors = val["geometry"]["heel_m"].copy()
            self.sole_anchors = val["geometry"]["sole_points_m"].copy()
            self.contact_diagnostic["heel_anchors_m"] = self.heel_anchors.tolist()
        if self.stage2 and converged:
            self._condition_stationary_derivatives()
        self.update_scale = 1.
        self.last_point_jacobian = None
        self._linearize()
        reasons[used] = "used_2d_ground_assumption"
        return True, used if not initialize else np.zeros_like(used), "updated" if converged else "feasible_unconverged"

    def _condition_stationary_derivatives(self):
        """Condition v/a on a zero heel derivative assumption, holding q fixed.

        Acceleration includes model curvature. The joint conditional covariance
        keeps its correlation with q; the new prior cannot move the hard-feasible
        pose or consume the same pixels again.
        """
        from pose_app.smplh_ground_constraints import HEEL_VERTEX_IDS
        n, eps = self.n, self.config.derivative_step_s
        offsets = np.r_[np.eye(n)*1e-3, -np.eye(n)*1e-3]
        data = self.model.evaluate_batch(self.pose.translation_m+offsets[:, :3],
                  self.pose.root_rotation @ Rotation.from_rotvec(offsets[:, 3:6]).as_matrix(),
                  self.pose.latent+offsets[:, 6:], mesh=True)["vertices_m"][:, HEEL_VERTEX_IDS]
        J = ((data[:n]-data[n:])/2e-3).reshape(n, 6).T
        center = self.model.evaluate(self.pose, mesh=True)["vertices_m"][0, HEEL_VERTEX_IDS]
        drive = np.stack((self.velocity*eps+.5*self.acceleration*eps**2,
                          -self.velocity*eps+.5*self.acceleration*eps**2))
        nearby = self.model.evaluate_batch(self.pose.translation_m+drive[:, :3],
                   self.pose.root_rotation @ Rotation.from_rotvec(drive[:, 3:6]).as_matrix(),
                   self.pose.latent+drive[:, 6:], mesh=True)["vertices_m"][:, HEEL_VERTEX_IDS]
        velocity = ((nearby[0]-nearby[1])/(2*eps)).ravel()
        acceleration = ((nearby[0]+nearby[1]-2*center)/eps**2).ravel()
        cross = self.covariance[n:, :n] @ np.linalg.solve(self.covariance[:n, :n], np.eye(n))
        explained = cross @ self.covariance[:n, n:]
        conditional = self.covariance[n:, n:]-explained
        H = np.zeros((12, 2*n)); H[:6, :n] = J; H[6:, n:] = J
        variance = np.r_[np.repeat(self.ground_config.heel_velocity_sigma_m_s**2, 6),
                         np.repeat(self.ground_config.heel_acceleration_sigma_m_s2**2, 6)]
        gain = np.linalg.solve(H @ conditional @ H.T+np.diag(variance), H @ conditional).T
        correction = gain @ -np.r_[velocity, acceleration]
        self.velocity += correction[:n]; self.acceleration += correction[n:]
        A = np.eye(2*n)-gain @ H
        self.covariance[n:, n:] = A @ conditional @ A.T+(gain*variance) @ gain.T+explained
        self.covariance = (self.covariance+self.covariance.T)/2
        self.contact_diagnostic["derivative_prior"] = "zero_heel_v_and_a_conditional_local_approximation;acceleration_includes_model_curvature"
        self.contact_diagnostic["heel_speed_before_prior_m_s"] = np.linalg.norm(velocity.reshape(2, 3), axis=1).tolist()
        self.contact_diagnostic["heel_acceleration_before_prior_m_s2"] = np.linalg.norm(acceleration.reshape(2, 3), axis=1).tolist()

    def step(self, now, left, right):
        first = self.last_time is None
        if first and self.constrain_ground:
            px = np.array([left, right], float)
            valid, reasons = self._supported_pixels(px)
            self._constrained_update(px, valid, reasons, initialize=True)
            self.last_accepted_pose = self.pose.copy()
        output = super().step(now, left, right)
        return output

    def _output(self, now, status, used, reasons, observed_pixels=None):
        output = super()._output(now, status, used, reasons, observed_pixels)
        output["ground_constraint"] = dict(self.contact_diagnostic)
        satisfied = False
        if np.isfinite(output["vertices_m"]).all():
            geometry = self._geometry(output["vertices_m"])
            slack = self._conditions(geometry)
            satisfied = bool(slack.min() >= -1e-6)
            output["ground_constraint"].update(sole_min_m=geometry["sole_min_m"].tolist(),
                heel_m=geometry["heel_m"].tolist(), heel_anchors_m=None if self.heel_anchors is None else self.heel_anchors.tolist(),
                minimum_output_slack_m=float(slack.min()),
                sole_max_displacement_m=None if self.sole_anchors is None else
                    np.linalg.norm(geometry["sole_points_m"]-self.sole_anchors, axis=-1).max(axis=-1).tolist())
        output["ground_constraint"]["constraint_satisfied"] = satisfied if self.constrain_ground else None
        output["ground_constraint"]["stage2"] = self.stage2
        if self.constrain_ground and (not satisfied or self.contact_diagnostic["status"] in ("constraint_conflict", "feasible_unconverged")):
            output["derivative_valid"][:] = False
            output["velocity_mm_s"][:] = output["acceleration_mm_s2"][:] = np.nan
            output["derivative_reason"][:] = self.contact_diagnostic["status"] if satisfied else "constraint_conflict"
        output["world_state_available"] = True
        output["world_pose_certified"] = False
        output["coordinate_frame"] = "stage_conditioned_fixed_ground"
        return output
