"""Nonlinear current-frame solve, information accounting and output contracts."""
import unittest

import numpy as np
from pose_app.smplh_iterated_motion_tracker import (
    IteratedMAPConfig,
    SMPLHIteratedMotionTracker,
)
from pose_app.smplh_motion_tracker import SMPLHMotionConfig, right_jacobian
from scipy.spatial.transform import Rotation

from tests.test_smplh_motion_tracker import AnalyticBodyMap, observation, tracker


def iterated(config=None, optimizer=None):
    return SMPLHIteratedMotionTracker(AnalyticBodyMap(), tracker().cal, config, optimizer)


class IteratedMotionTests(unittest.TestCase):
    def test_nonlinear_iteration_reduces_pixel_error_beyond_bounded_ekf_step(self):
        config = SMPLHMotionConfig(pixel_sigma_px=.5)
        baseline, tr = tracker(config), iterated(config)
        initial = observation(tr)
        for instance in (baseline, tr):
            instance.step(0., initial[0], initial[1])
        target_pose = tr.pose.copy()
        target_pose.translation_m += [.25, .08, .1]
        target_pose.root_rotation = Rotation.from_rotvec([.05, .08, .4]).as_matrix()
        target_pose.latent[:] = .7
        pixels = observation(tr, target_pose)
        old = baseline.step(.03, pixels[0], pixels[1])
        result = tr.step(.03, pixels[0], pixels[1])
        before = np.median(np.linalg.norm(old["projected_pixels"]-pixels[..., :2], axis=-1))
        after = np.median(np.linalg.norm(result["projected_pixels"]-pixels[..., :2], axis=-1))
        self.assertLess(after, .1*before)
        self.assertGreater(np.linalg.norm(tr.pose.translation_m-baseline.model.initial_pose.translation_m), config.max_translation_correction_m)
        diag = result["iterated_update"]
        self.assertEqual(diag["status"], "converged")
        self.assertGreater(diag["iterations"], 1)
        costs = [diag["objective_initial"]]+[item["objective"] for item in diag["history"]]
        self.assertTrue(np.all(np.diff(costs) < 0))

    def test_only_one_final_posterior_and_full_conditional_velocity_acceleration(self):
        tr = iterated(SMPLHMotionConfig(pixel_sigma_px=1.))
        n = tr.n
        # Include cross-correlated pose scales so diagonal-only whitening or a
        # correction to q alone cannot satisfy this conditional-mean identity.
        rng = np.random.default_rng(42)
        A = np.eye(3*n)+rng.normal(size=(3*n, 3*n))*.04
        P = A @ np.diag(np.tile([.02, .01, .03, .1, .08, .07, .2], 3)) @ A.T
        tr.covariance = P.copy()
        prior = tr.pose.copy()
        target = prior.copy(); target.translation_m += [.04, -.03, .02]
        target.root_rotation = Rotation.from_rotvec([.03, -.04, .05]).as_matrix()
        target.latent[:] = .15
        pixels = observation(tr, target)
        valid = np.ones((2, 12), bool)
        accepted, _, _ = tr._update(pixels, valid, np.full((2, 12), "available", dtype=object))
        self.assertTrue(accepted)
        self.assertEqual(tr.optimizer_diagnostic["posterior_updates"], 1)
        delta = np.r_[tr.pose.translation_m-prior.translation_m,
                      Rotation.from_matrix(prior.root_rotation.T @ tr.pose.root_rotation).as_rotvec(),
                      tr.pose.latent-prior.latent]
        correction = P[:, :n] @ np.linalg.solve(P[:n, :n], delta)
        np.testing.assert_allclose(tr.velocity, correction[n:2*n], atol=1e-10)
        np.testing.assert_allclose(tr.acceleration, correction[2*n:], atol=1e-10)
        # Independently linearize at the final delta in the *prediction* tangent.
        eps = 1e-3
        from pose_app.smplh_motion_tracker import retract
        samples = [tr._pixels(tr.model.evaluate(retract(prior, delta+sign*np.eye(n)[i]*eps))["coco_m"])[0].ravel()
                   for sign in (1, -1) for i in range(n)]
        J = (np.array(samples[:n])-np.array(samples[n:])).T/(2*eps)
        Hprior = np.zeros((n-6, n)); Hprior[:, 6:] = np.eye(n-6)
        Hq = np.r_[J, Hprior]
        H = np.c_[Hq, np.zeros((len(Hq), 2*n))]
        residual = (tr._pixels(tr.model.evaluate(tr.pose)["coco_m"])[0]-pixels[..., :2]).reshape(-1, 2)
        weights = np.repeat(np.minimum(1., tr.config.huber_px/np.maximum(np.linalg.norm(residual, axis=1), 1e-12)), 2)
        variance = np.r_[tr.config.pixel_sigma_px**2/weights, [tr.config.latent_prior_sigma**2]]
        K = np.linalg.solve(H @ P @ H.T+np.diag(variance), H @ P).T
        B = np.eye(3*n)-K @ H
        posterior = B @ P @ B.T+(K*variance) @ K.T
        reset = np.eye(3*n); reset[3:6, 3:6] = right_jacobian(delta[3:6])
        np.testing.assert_allclose(tr.covariance, reset @ posterior @ reset.T, atol=1e-10)
        self.assertGreater(np.linalg.eigvalsh(tr.covariance).min(), 0)

    def test_budget_improvement_retains_prior_information_and_masks_derivatives(self):
        tr = iterated(SMPLHMotionConfig(pixel_sigma_px=.5, max_position_std_m=100.),
                      IteratedMAPConfig(maximum_iterations=1, gradient_tolerance=1e-8))
        prior_covariance = tr.covariance.copy()
        prior_pose = tr.pose.copy()
        target = tr.pose.copy(); target.translation_m += [.2, .1, .15]
        target.root_rotation = Rotation.from_rotvec([.2, -.1, .5]).as_matrix()
        pixels = observation(tr, target)
        reasons = np.full((2, 12), "available", dtype=object)
        accepted, used, status = tr._update(pixels, np.ones((2, 12), bool), reasons)
        self.assertTrue(accepted)
        self.assertEqual(tr.optimizer_diagnostic["status"], "feasible_unconverged")
        self.assertEqual(tr.optimizer_diagnostic["posterior_updates"], 0)
        root_delta = Rotation.from_matrix(prior_pose.root_rotation.T @ tr.pose.root_rotation).as_rotvec()
        reset = np.eye(3*tr.n); reset[3:6, 3:6] = right_jacobian(root_delta)
        np.testing.assert_allclose(tr.covariance, reset @ prior_covariance @ reset.T, atol=1e-10)
        tr.last_joint_support[:] = .2
        good_before = tr.last_joint_good_fit.copy()
        result = tr._output(.2, status, used, reasons, pixels)
        self.assertFalse(result["derivative_valid"].any())
        self.assertTrue(np.isnan(result["velocity_mm_s"]).all())
        self.assertTrue(np.isfinite(result["raw_velocity_mm_s"]).all())
        np.testing.assert_array_equal(tr.last_joint_good_fit, good_before)

    def test_first_frame_not_reused_and_missing_expiry_is_nan(self):
        tr = iterated()
        px = observation(tr)
        P = tr.covariance.copy()
        first = tr.step(0., px[0], px[1])
        self.assertEqual(first["status"], "initialized")
        np.testing.assert_array_equal(P, tr.covariance)
        self.assertEqual(first["iterated_update"]["status"], "not_run")
        missing = np.full((12, 3), np.nan)
        expired = tr.step(.2, missing, missing)
        self.assertEqual(expired["status"], "expired")
        self.assertTrue(np.isnan(expired["vertices_m"]).all())
        self.assertTrue(np.isnan(expired["raw_acceleration_mm_s2"]).all())
        restored = tr.step(.3, px[0], px[1])
        self.assertEqual(restored["status"], "reinitialized")
        self.assertFalse(restored["derivative_valid"].any())

    def test_accepted_iterate_information_does_not_claim_optimizer_convergence(self):
        # Both policies accept exactly the same first nonlinear iterate. Only
        # information feedback may differ; approximate information must not
        # make an unconverged derivative publishable.
        from dataclasses import replace
        opt = IteratedMAPConfig(maximum_iterations=1, gradient_tolerance=1e-8)
        old = iterated(optimizer=opt)
        tr = iterated(optimizer=replace(opt, condition_accepted_iterate=True))
        target = tr.pose.copy(); target.translation_m += [.2, .1, .15]
        target.root_rotation = Rotation.from_rotvec([.2, -.1, .5]).as_matrix()
        pixels = observation(tr, target)
        for instance in (old, tr):
            instance.last_joint_support[:] = .2
            accepted, used, status = instance._update(pixels, np.ones((2, 12), bool),
                np.full((2, 12), 'available', dtype=object))
            self.assertTrue(accepted)
            self.assertEqual(instance.optimizer_diagnostic['status'], 'feasible_unconverged')
            self.assertFalse(instance.optimizer_diagnostic['optimizer_success'])
        np.testing.assert_allclose(tr.pose.translation_m, old.pose.translation_m)
        np.testing.assert_allclose(tr.velocity, old.velocity)
        np.testing.assert_allclose(tr.acceleration, old.acceleration)
        # Positive information, once, with a positive covariance; the existing
        # exact-posterior test covers the same final linearization equations.
        self.assertEqual(tr.optimizer_diagnostic['posterior_updates'], 1)
        self.assertEqual(old.optimizer_diagnostic['posterior_updates'], 0)
        self.assertGreater(np.linalg.eigvalsh(tr.covariance).min(), 0)
        self.assertLess(np.trace(tr.covariance), np.trace(old.covariance))
        result = tr._output(.2, status, used, np.full((2,12), 'available', object), pixels)
        self.assertFalse(result['derivative_valid'].any())
        self.assertTrue(np.isnan(result['velocity_mm_s']).all())
        self.assertTrue(np.isnan(result['acceleration_mm_s2']).all())

    def test_approximate_policy_still_requires_accepted_pixels(self):
        tr = iterated(optimizer=IteratedMAPConfig(condition_accepted_iterate=True))
        pixels = observation(tr)
        P = tr.covariance.copy()
        accepted, used, _ = tr._update(pixels, np.zeros((2,12), bool),
            np.full((2,12), 'missing', dtype=object))
        self.assertFalse(accepted)
        self.assertFalse(used.any())
        self.assertEqual(tr.optimizer_diagnostic['posterior_updates'], 0)
        np.testing.assert_array_equal(tr.covariance, P)

    def test_line_search_failure_does_not_consume_observations(self):
        tr = iterated()
        pixels = observation(tr)
        target = tr.pose.copy(); target.translation_m[0] += .1
        pixels = observation(tr, target)
        original_pixels = tr._pixels
        initial_points = tr.model.evaluate(tr.pose)["coco_m"]

        def reject_trials(points):
            # Keep the initial projection and its batched Jacobian, but make
            # every nonlinear line-search proposal physically invalid.
            return original_pixels(points) if len(points) > 1 or np.allclose(points, initial_points, atol=1e-14, rtol=0) else None

        tr._pixels = reject_trials
        P = tr.covariance.copy()
        pose = tr.pose.copy()
        reasons = np.full((2, 12), "available", dtype=object)
        accepted, used, _ = tr._update(pixels, np.ones((2, 12), bool), reasons)
        self.assertFalse(accepted)
        self.assertFalse(used.any())
        self.assertEqual(tr.optimizer_diagnostic["termination"], "line_search_failed")
        self.assertEqual(tr.optimizer_diagnostic["posterior_updates"], 0)
        np.testing.assert_array_equal(tr.covariance, P)
        np.testing.assert_array_equal(tr.pose.translation_m, pose.translation_m)
        np.testing.assert_array_equal(tr.velocity, np.zeros(tr.n))

    def test_whole_latent_prior_is_configurable_and_domain_is_preserved(self):
        weak = iterated(SMPLHMotionConfig(latent_prior_sigma=2., pixel_sigma_px=2., max_abs_latent=.4))
        strong = iterated(SMPLHMotionConfig(latent_prior_sigma=.2, pixel_sigma_px=2., max_abs_latent=.4))
        target = weak.pose.copy(); target.latent[:] = 1.
        pixels = observation(weak, target)
        for tr in (weak, strong):
            tr._update(pixels, np.ones((2, 12), bool), np.full((2, 12), "available", dtype=object))
            self.assertLessEqual(abs(tr.pose.latent).max(), .4)
        self.assertLess(abs(strong.pose.latent).max(), abs(weak.pose.latent).max())

    def test_invalid_jacobian_diagnostic_remains_strict_json_and_rejects_update(self):
        import json

        tr = iterated()
        pixels = observation(tr)
        original = tr._pixels
        tr._pixels = lambda points: None if len(points) > 1 else original(points)
        P = tr.covariance.copy()
        accepted, used, _ = tr._update(pixels, np.ones((2, 12), bool),
                                      np.full((2, 12), "available", dtype=object))
        self.assertFalse(accepted)
        self.assertFalse(used.any())
        self.assertEqual(tr.optimizer_diagnostic["termination"], "jacobian_invalid")
        self.assertIsNone(tr.optimizer_diagnostic["scaled_gradient_inf"])
        json.dumps(tr.optimizer_diagnostic, allow_nan=False)
        np.testing.assert_array_equal(P, tr.covariance)

    def test_invalid_optimizer_parameters_are_rejected(self):
        for values in ({"maximum_iterations": 0}, {"maximum_iterations": 1.5}, {"gradient_tolerance": np.nan},
                       {"condition_accepted_iterate": 1}):
            with self.assertRaises(ValueError):
                IteratedMAPConfig(**values)


if __name__ == "__main__":
    unittest.main()
