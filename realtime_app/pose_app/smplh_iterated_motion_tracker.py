"""Current-frame iterated pixel MAP with one predicted prior and one posterior.

The complete SMPL-H observation map, pixel gates and q/v/a process are inherited
from the baseline. Optional engineering residuals regularize the point estimate;
they are not treated as independent information in the posterior covariance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from pose_app.smplh_motion_tracker import SMPLHMotionTracker, retract, right_jacobian


@dataclass(frozen=True)
class IteratedMAPConfig:
    maximum_iterations: int = 15
    gradient_tolerance: float = .05
    objective_tolerance: float = 1e-3
    step_tolerance: float = 1e-3
    maximum_line_search_steps: int = 12
    condition_accepted_iterate: bool = False  # Opt-in approximate information feedback.

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if name == "condition_accepted_iterate":
                if not isinstance(self.condition_accepted_iterate, bool):
                    raise ValueError("condition_accepted_iterate must be boolean")
                continue
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("maximum_iterations", "maximum_line_search_steps"):
            if not isinstance(getattr(self, name), (int, np.integer)):
                raise ValueError(f"{name} must be an integer")


class SMPLHIteratedMotionTracker(SMPLHMotionTracker):
    """Solve pixel + whole-VPoser + optional regularizer + fixed-prior MAP."""

    def __init__(self, model, cal, config=None, optimizer_config=None):
        super().__init__(model, cal, config)
        self.optimizer_config = optimizer_config or IteratedMAPConfig()
        self.optimizer_diagnostic = {"status": "not_run"}
        self.derivative_estimate_converged = True

    def _engineering_residuals(self, data):
        """Return normalized residuals (batch, terms), empty in the pixel baseline."""
        return np.empty((len(data["coco_m"]), 0))

    def _objective_jacobians(self, prior, delta, selected, engineering_count):
        """Differentiate in the one fixed prediction tangent (default: central FD)."""
        n, eps = self.n, 1e-3
        offsets = np.r_[delta[None]+np.eye(n)*eps,
                        delta[None]-np.eye(n)*eps]
        data = self.model.evaluate_batch(prior.translation_m+offsets[:, :3],
                prior.root_rotation @ Rotation.from_rotvec(offsets[:, 3:6]).as_matrix(),
                prior.latent+offsets[:, 6:])
        projected = self._pixels(data["coco_m"])
        pixel_jacobian = None if projected is None else ((projected[:n]-projected[n:])/(2*eps)).reshape(n, -1).T[selected]
        engineering = np.asarray(self._engineering_residuals(data), dtype=float)
        if engineering.shape != (2*n, engineering_count):
            raise ValueError("engineering residuals must have stable shape (batch, terms)")
        engineering_jacobian = ((engineering[:n]-engineering[n:])/(2*eps)).T
        return pixel_jacobian, engineering_jacobian

    def _update(self, pixels, valid, reasons):
        self.update_scale = 0.
        used = valid.copy()
        self.optimizer_diagnostic = {"status": "not_run", "posterior_updates": 0,
            "engineering_objective_initial": None, "engineering_objective_final": None,
            "engineering_residual_count": 0, "engineering_information_update": False}
        if np.any(used, axis=0).sum() < self.config.min_observed_joints:
            reasons[used] = "insufficient_joint_support"
            self.optimizer_diagnostic["status"] = "insufficient_joint_support"
            return False, np.zeros_like(used), "insufficient_joint_support"
        n, prior, P = self.n, self.pose.copy(), self.covariance.copy()
        # Full whitening includes correlations, unlike parameter-by-parameter
        # rescaling. delta always lives in this one predicted-pose tangent.
        L = np.linalg.cholesky(P[:n, :n])
        selected = np.repeat(used.ravel(), 2)
        target = pixels[..., :2].reshape(-1)[selected]
        config, opt = self.config, self.optimizer_config
        latent_precision = 1 / config.latent_prior_sigma**2
        cache = {}

        def engineering_residuals(data, batch_size, count=None):
            residual = np.asarray(self._engineering_residuals(data), dtype=float)
            if residual.ndim != 2 or residual.shape[0] != batch_size or (count is not None and residual.shape[1] != count):
                raise ValueError("engineering residuals must have stable shape (batch, terms)")
            return residual

        def evaluate(y, jacobian=False):
            if not cache or not np.array_equal(y, cache["y"]):
                delta = L @ y
                pose = retract(prior, delta)
                projected = None
                engineering = None
                if np.isfinite(delta).all() and abs(pose.latent).max(initial=0.) <= config.max_abs_latent:
                    data = self.model.evaluate(pose)
                    projected = self._pixels(data["coco_m"])
                    engineering = engineering_residuals(data, 1)[0]
                cache.clear()
                cache.update(y=y.copy(), delta=delta, pose=pose, projected=projected,
                             engineering=engineering, cost=np.inf, engineering_cost=None)
                if engineering is not None and np.isfinite(engineering).all():
                    energy = float(.5*(engineering@engineering))
                    if np.isfinite(energy):
                        cache["engineering_cost"] = energy
                if projected is not None and cache["engineering_cost"] is not None:
                    residual = (projected[0].reshape(-1)[selected]-target).reshape(-1, 2)
                    norms = np.linalg.norm(residual, axis=1)
                    cost = np.sum(np.where(norms <= config.huber_px, .5*norms**2,
                                          config.huber_px*(norms-.5*config.huber_px))) / config.pixel_sigma_px**2
                    cache.update(residual=residual, robust=np.minimum(1., config.huber_px/np.maximum(norms, 1e-12)),
                                 cost=float(cost+.5*y@y+.5*latent_precision*(pose.latent@pose.latent)
                                            +cache["engineering_cost"]))
            if jacobian and np.isfinite(cache["cost"]) and "J" not in cache:
                # Perturb the fixed prediction's delta, not the repeatedly
                # retracted current root. Thus the prior remains exactly y'y/2.
                cache["J"], cache["J_engineering"] = self._objective_jacobians(
                    prior, cache["delta"], selected, len(cache["engineering"]))
            return cache

        def valid_linearization(val):
            return (np.isfinite(val["cost"]) and val.get("J") is not None
                    and np.isfinite(val["J"]).all() and val.get("J_engineering") is not None
                    and np.isfinite(val["J_engineering"]).all())

        def normal_equations(val):
            Jwhite = val["J"] @ L
            weights = np.repeat(val["robust"], 2) / config.pixel_sigma_px**2
            Z = L[6:]
            gradient = Jwhite.T @ (weights*val["residual"].ravel())+val["y"]+latent_precision*Z.T @ val["pose"].latent
            hessian = (Jwhite.T*weights) @ Jwhite+np.eye(n)+latent_precision*Z.T @ Z
            if len(val["engineering"]):
                Jengineering = val["J_engineering"] @ L
                gradient += Jengineering.T @ val["engineering"]
                hessian += Jengineering.T @ Jengineering
            return gradient, hessian

        y = np.zeros(n)
        val = evaluate(y)
        initial_cost = val["cost"]
        initial_engineering_cost = val["engineering_cost"]
        engineering_count = len(val["engineering"]) if val["engineering"] is not None else 0
        self.optimizer_diagnostic.update(engineering_objective_initial=initial_engineering_cost,
            engineering_objective_final=initial_engineering_cost, engineering_residual_count=engineering_count)
        if not np.isfinite(initial_cost):
            status = "engineering_residual_invalid" if val["engineering"] is not None and initial_engineering_cost is None else "model_projection_invalid"
            reasons[used] = status
            self.optimizer_diagnostic["status"] = status
            return False, np.zeros_like(used), status
        initial_pixel_median = float(np.median(np.linalg.norm(val["residual"], axis=1)))
        history, converged, termination = [], False, "iteration_budget"
        damping, last_step, gradient_norm = 1e-4, np.inf, np.inf
        for iteration in range(opt.maximum_iterations):
            val = evaluate(y, True)
            if not valid_linearization(val):
                termination = "jacobian_invalid"
                break
            gradient, hessian = normal_equations(val)
            if not np.isfinite(gradient).all() or not np.isfinite(hessian).all():
                termination = "normal_equations_invalid"
                break
            gradient_norm = float(np.linalg.norm(gradient, ord=np.inf))
            if gradient_norm <= opt.gradient_tolerance:
                converged, termination = True, "scaled_gradient"
                break
            cost_before = val["cost"]
            accepted = False
            for _ in range(4):
                direction = np.linalg.solve(hessian+damping*np.diag(np.diag(hessian)), -gradient)
                slope = float(gradient@direction)
                alpha = 1.
                for backtrack in range(opt.maximum_line_search_steps):
                    trial = evaluate(y+alpha*direction)
                    if np.isfinite(trial["cost"]) and trial["cost"] <= cost_before+1e-4*alpha*slope:
                        accepted = True
                        break
                    alpha *= .5
                if accepted:
                    break
                damping *= 10
            if not accepted:
                termination = "line_search_failed"
                break
            y = trial["y"].copy()
            last_step = float(np.linalg.norm(alpha*direction, ord=np.inf))
            decrease = float(cost_before-trial["cost"])
            history.append({"iteration": iteration+1, "objective": trial["cost"],
                            "decrease": decrease, "scaled_gradient_inf": gradient_norm,
                            "scaled_step_inf": last_step, "line_search_scale": alpha,
                            "backtracks": backtrack, "damping": damping})
            damping = max(damping*.3, 1e-8)
            # Require both a small scaled step and small objective change; a
            # line-search failure alone is never labelled convergence.
            if last_step <= opt.step_tolerance and decrease <= opt.objective_tolerance:
                final = evaluate(y, True)
                if valid_linearization(final):
                    gradient_norm = float(np.linalg.norm(normal_equations(final)[0], ord=np.inf))
                    if gradient_norm <= opt.gradient_tolerance:
                        converged, termination = True, "scaled_step_objective_and_gradient"
                        break
        val = evaluate(y, True)
        valid_final_jacobian = valid_linearization(val)
        if valid_final_jacobian:
            gradient, hessian = normal_equations(val)
            valid_final_jacobian = bool(np.isfinite(gradient).all() and np.isfinite(hessian).all())
            gradient_norm = float(np.linalg.norm(gradient, ord=np.inf))
            if valid_final_jacobian and gradient_norm <= opt.gradient_tolerance:
                converged, termination = True, "scaled_gradient"
        improvement = bool(np.isfinite(val["cost"]) and val["cost"] < initial_cost-opt.objective_tolerance)
        accepted = bool(valid_final_jacobian and (converged or improvement))
        condition_posterior = accepted and (converged or opt.condition_accepted_iterate)
        self.optimizer_diagnostic = {
            "status": "converged" if converged else ("feasible_unconverged" if accepted else "rejected"),
            "optimizer_success": converged, "termination": termination,
            "iterations": len(history), "linearization_iterations": iteration+1,
            "objective_initial": initial_cost, "objective": float(val["cost"]) if np.isfinite(val["cost"]) else None,
            "scaled_gradient_inf": gradient_norm if np.isfinite(gradient_norm) else None,
            "scaled_step_inf": last_step if np.isfinite(last_step) else None,
            "pixel_error_median_initial_px": initial_pixel_median,
            "pixel_error_median_px": float(np.median(np.linalg.norm(val["residual"], axis=1))) if "residual" in val else None,
            "posterior_updates": int(condition_posterior),
            "engineering_objective_initial": initial_engineering_cost,
            "engineering_objective_final": val["engineering_cost"],
            "engineering_residual_count": engineering_count,
            "engineering_information_update": False,
            "fixed_prior_each_iteration": True,
            "covariance_approximation": ("one_final_IRLS_Gauss_Newton_posterior" if converged else
                "one_accepted_iterate_linearized_posterior_not_converged" if condition_posterior else
                "transported_prediction_no_information_update"),
            "history": history,
        }
        if engineering_count:
            self.optimizer_diagnostic["covariance_approximation"] += ";pixel_and_pose_information_only_engineering_regularizer_omitted"
        if not accepted:
            self.derivative_estimate_converged = False
            reasons[used] = "iterated_map_rejected"
            return False, np.zeros_like(used), "iterated_map_rejected"
        delta = val["delta"]
        # Gaussian conditional of the *same* predicted full state. This adjusts
        # velocity/acceleration through existing cross-covariances, once only.
        conditional = np.linalg.solve(P[:n, :n], P[:n]).T
        correction = conditional @ delta
        self.pose = val["pose"]
        self.velocity += correction[n:2*n]
        self.acceleration += correction[2*n:]
        posterior = P
        if condition_posterior:
            # Information feedback and optimizer certification are separate:
            # this local approximation never enables unconverged derivatives.
            Hprior = np.zeros((n-6, n)); Hprior[:, 6:] = np.eye(n-6)
            Hq = np.r_[val["J"], Hprior]
            H = np.c_[Hq, np.zeros((len(Hq), 2*n))]
            variance = np.r_[config.pixel_sigma_px**2/np.repeat(val["robust"], 2),
                             np.repeat(config.latent_prior_sigma**2, n-6)]
            gain = np.linalg.solve(H @ P @ H.T+np.diag(variance), H @ P).T
            A = np.eye(3*n)-gain @ H
            posterior = A @ P @ A.T+(gain*variance) @ gain.T
        reset = np.eye(3*n); reset[3:6, 3:6] = right_jacobian(delta[3:6])
        self.covariance = reset @ posterior @ reset.T
        self.covariance = (self.covariance+self.covariance.T)/2
        self.derivative_estimate_converged = converged
        self.update_scale = 1.
        self.last_point_jacobian = None
        self._linearize()
        reasons[used] = "used_2d_iterated_map" if converged else "used_2d_feasible_unconverged"
        return True, used, "updated"

    def _output(self, now, status, used, reasons, observed_pixels=None):
        previous_good_fit = self.last_joint_good_fit.copy()
        output = super()._output(now, status, used, reasons, observed_pixels)
        output["iterated_update"] = dict(self.optimizer_diagnostic)
        if not self.derivative_estimate_converged:
            self.last_joint_good_fit[:] = previous_good_fit
            output["derivative_valid"][:] = False
            output["velocity_mm_s"][:] = output["acceleration_mm_s2"][:] = np.nan
            output["derivative_reason"][:] = "iterated_map_unconverged"
        return output
