"""Known camera motion and actual-model contact conditions, not accuracy tests."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize as scipy_minimize
from scipy.spatial.transform import Rotation

from pose_app.smplh_motion_tracker import MotionPose, SMPLHMotionConfig
from pose_app.foot_ground_reference import SHOE_HEIGHT_M
from pose_app.smplh_world_motion_tracker import SMPLHWorldMotionTracker, _minimize_scaled_pose
from tests.test_smplh_motion_tracker import AnalyticBodyMap, tracker


class SurfaceBodyMap(AnalyticBodyMap):
    def __init__(self):
        super().__init__()
        self.base[0] = [.02, .03, .04]
        self.pelvis = self.base[0].copy()
        self.mesh = np.repeat(self.pelvis[None], 6890, axis=0)
        for side, indices in enumerate((np.arange(54), np.arange(54, 108))):
            self.mesh[indices] = np.array([(-.12, .12)[side], 0, -.04]) + np.c_[np.linspace(-.02, .02, 54), np.linspace(-.07, .07, 54), np.zeros(54)]
        self.mesh[[3387, 6787]] = [[-.12, 0, -.01], [.12, 0, -.01]]

    def evaluate_batch(self, translations, roots, latents, mesh=False):
        data = super().evaluate_batch(translations, roots, latents, mesh=False)
        correction = self.pelvis-np.einsum("bij,j->bi", roots, self.pelvis)
        data["coco_m"] += correction[:, None]
        data["joints_m"] = data["coco_m"].copy()
        if mesh:
            data["vertices_m"] = np.einsum("bij,vj->bvi", roots, self.mesh-self.pelvis) + self.pelvis + translations[:, None]
        return data


def world_tracker(contact=False, rotation=None):
    model = SurfaceBodyMap()
    result = SMPLHWorldMotionTracker(model, tracker().cal, SMPLHMotionConfig(pixel_sigma_px=.2),
        np.eye(3) if rotation is None else rotation, [0, 0, -1.96],
        [np.arange(54), np.arange(54, 108)], constrain_ground=contact)
    if contact:
        result.pose.translation_m[2] += SHOE_HEIGHT_M
    result.last_accepted_pose = result.pose.copy()
    return result


def observe(tr, pose=None):
    p = tr._pixels(tr.model.evaluate(pose or tr.pose)["coco_m"])[0]
    return np.concatenate((p, np.ones((2, 12, 1))), axis=-1)


class WorldMotionTests(unittest.TestCase):
    def test_duplicate_hard_inequalities_do_not_add_independent_covariance_information(self):
        posteriors = []
        for copies in (1, 10):
            tr = world_tracker(contact=True)
            tr.pose.translation_m[2] -= .0048
            original_conditions = tr._solver_conditions
            tr._solver_conditions = lambda geometry, original=original_conditions, copies=copies: np.tile(
                original(geometry), (1,)*(geometry["sole_min_m"].ndim-1)+(copies,))
            initial_covariance = tr.covariance.copy()
            pixels = observe(tr)

            def held_feasible_optimum(fun, initial, **kwargs):
                delta = np.zeros(tr.n)
                return SimpleNamespace(x=delta, fun=fun(delta), success=True, status=0,
                                       message="fixed feasible optimum to isolate covariance", nit=1)

            with patch("pose_app.smplh_world_motion_tracker._minimize_scaled_pose", side_effect=held_feasible_optimum):
                accepted, _, _ = tr._constrained_update(pixels, np.ones((2, 12), bool),
                    np.full((2, 12), "available", dtype=object), initialize=True)
            self.assertTrue(accepted)
            self.assertGreaterEqual(tr._conditions(tr._geometry(
                tr.model.evaluate(tr.pose, mesh=True)["vertices_m"][0])).min(), 0.)
            posteriors.append(tr.covariance)
            # Initialization consumes no new pixels or VPoser prior. A hard
            # feasible-set boundary has no calibrated independent noise model.
            np.testing.assert_allclose(tr.covariance, initial_covariance, atol=1e-12)
            self.assertFalse(tr.contact_diagnostic["contact_covariance_certified"])
            self.assertEqual(tr.contact_diagnostic["covariance_observation_policy"],
                             "pixels_and_whole_vposer_only;hard_inequalities_not_independent_observations")
        np.testing.assert_allclose(posteriors[0], posteriors[1], atol=1e-12)

    def test_online_posterior_uses_pixels_and_vposer_independent_of_hard_constraint_duplicates(self):
        posteriors = []
        for copies in (1, 10):
            tr = world_tracker(contact=True)
            tr.pose.translation_m[2] -= .0048
            original_conditions = tr._solver_conditions
            tr._solver_conditions = lambda geometry, original=original_conditions, copies=copies: np.tile(
                original(geometry), (1,)*(geometry["sole_min_m"].ndim-1)+(copies,))
            pixels = observe(tr)
            P = tr.covariance.copy()
            pixel_jacobian = tr._linearize()[1]
            latent_jacobian = np.zeros((tr.n-6, tr.n))
            latent_jacobian[:, 6:] = np.eye(tr.n-6)
            Hq = np.r_[pixel_jacobian, latent_jacobian]
            H = np.c_[Hq, np.zeros((len(Hq), 2*tr.n))]
            variance = np.r_[np.full(48, tr.config.pixel_sigma_px**2),
                             np.full(tr.n-6, tr.config.latent_prior_sigma**2)]
            gain = np.linalg.solve(H @ P @ H.T+np.diag(variance), H @ P).T
            A = np.eye(3*tr.n)-gain @ H
            expected = A @ P @ A.T+(gain*variance) @ gain.T

            def held_feasible_optimum(fun, initial, **kwargs):
                delta = np.zeros(tr.n)
                return SimpleNamespace(x=delta, fun=fun(delta), success=True, status=0,
                                       message="fixed feasible optimum to isolate online covariance", nit=1)

            with patch("pose_app.smplh_world_motion_tracker._minimize_scaled_pose", side_effect=held_feasible_optimum):
                accepted, used, status = tr._constrained_update(pixels, np.ones((2, 12), bool),
                    np.full((2, 12), "available", dtype=object), initialize=False)
            self.assertTrue(accepted)
            self.assertTrue(used.all())
            self.assertEqual(status, "updated")
            np.testing.assert_allclose(tr.covariance, expected, atol=1e-12)
            self.assertLess(np.trace(tr.covariance[:tr.n, :tr.n]), np.trace(P[:tr.n, :tr.n]))
            posteriors.append(tr.covariance)
        np.testing.assert_allclose(posteriors[0], posteriors[1], atol=1e-12)

    def test_current_support_can_switch_feet_without_extra_upper_bound(self):
        tr = world_tracker(contact=True)
        vertices = tr.model.evaluate(tr.pose, mesh=True)["vertices_m"][0].copy()
        vertices[tr.sole_indices[0], 2] = SHOE_HEIGHT_M+.025
        vertices[tr.sole_indices[1], 2] = SHOE_HEIGHT_M+.003
        # The old solver froze this predicted point as the support point.
        tr.support_vertices = [(0, 0)]
        geometry = tr._geometry(vertices)
        self.assertGreater(tr._conditions(geometry).min(), 0.)
        self.assertGreater(tr._solver_conditions(geometry).min(), 0.)

    def test_stage2_current_contact_vertex_can_change_within_each_foot(self):
        tr = world_tracker(contact=True)
        tr.stage2 = True
        vertices = tr.model.evaluate(tr.pose, mesh=True)["vertices_m"][0].copy()
        for indices in tr.sole_indices:
            vertices[indices, 2] = SHOE_HEIGHT_M+.020
            vertices[indices[-1], 2] = SHOE_HEIGHT_M+.003
        tr.support_vertices = [(0, 0), (1, 0)]
        geometry = tr._geometry(vertices)
        self.assertGreater(tr._conditions(geometry).min(), 0.)
        self.assertGreater(tr._solver_conditions(geometry).min(), 0.)

    def test_pixel_fit_can_transfer_support_in_a_tilted_model(self):
        tr = world_tracker(contact=True)
        px = observe(tr)
        tr.step(0., px[0], px[1])
        target = tr.pose.copy()
        target.root_rotation = Rotation.from_rotvec([0., .12, 0.]).as_matrix()
        geometry = tr._geometry(tr.model.evaluate(target, mesh=True)["vertices_m"][0])
        target.translation_m[2] += SHOE_HEIGHT_M+.002-geometry["sole_min_m"].min()
        px = observe(tr, target)
        initial_error = np.linalg.norm(tr._pixels(tr.model.evaluate(tr.pose)["coco_m"])[0]-px[..., :2], axis=-1).mean()
        priors = []
        def solve_with_frozen_prior(*args, **kwargs):
            priors.append(tr.covariance.copy())
            return _minimize_scaled_pose(*args, **kwargs)
        with patch("pose_app.smplh_world_motion_tracker._minimize_scaled_pose", side_effect=solve_with_frozen_prior):
            out = tr.step(.03, px[0], px[1])
        final_error = np.linalg.norm(tr._pixels(tr.model.evaluate(tr.pose)["coco_m"])[0]-px[..., :2], axis=-1).mean()
        self.assertEqual(out["status"], "updated")
        self.assertTrue(out["ground_constraint"]["constraint_satisfied"])
        self.assertEqual(out["ground_constraint"]["support_extra_solver_calls"], 1)
        self.assertEqual(out["ground_constraint"]["map_solver_calls"], 2)
        self.assertEqual(out["ground_constraint"]["feasibility_solver_calls"], 0)
        self.assertEqual(len(priors), 2)
        np.testing.assert_array_equal(priors[0], priors[1])
        sole_min = np.array(out["ground_constraint"]["sole_min_m"])-SHOE_HEIGHT_M
        self.assertEqual(out["ground_constraint"]["selected_support_side"], 1)
        self.assertGreater(sole_min[0], .0055)
        self.assertLess(abs(sole_min[1]), .005)
        self.assertLess(final_error, initial_error*.9)

    def test_opposite_foot_cannot_certify_selected_support_branch(self):
        tr = world_tracker(contact=True)
        px = observe(tr)
        tr.step(0., px[0], px[1])
        target = tr.pose.copy()
        target.root_rotation = Rotation.from_rotvec([0., .12, 0.]).as_matrix()
        geometry = tr._geometry(tr.model.evaluate(target, mesh=True)["vertices_m"][0])
        target.translation_m[2] += SHOE_HEIGHT_M+.003-geometry["sole_min_m"].min()
        geometry = tr._geometry(tr.model.evaluate(target, mesh=True)["vertices_m"][0])
        self.assertGreater(geometry["sole_min_m"][0]-SHOE_HEIGHT_M, .005)
        self.assertLess(geometry["sole_min_m"][1]-SHOE_HEIGHT_M, .005)
        delta = np.r_[target.translation_m-tr.pose.translation_m,
                      Rotation.from_matrix(tr.pose.root_rotation.T @ target.root_rotation).as_rotvec(),
                      target.latent-tr.pose.latent]
        def opposite_support_result(fun, x0, **kwargs):
            return SimpleNamespace(x=delta.copy(), fun=fun(delta), success=True,
                                   status=0, message="test successful proposal", nit=1)
        with patch("pose_app.smplh_world_motion_tracker._minimize_scaled_pose", side_effect=opposite_support_result):
            pixels = observe(tr, target)
            accepted, _, _ = tr._constrained_update(pixels, np.ones((2, 12), bool),
                                                   np.full((2, 12), "available", dtype=object))
        self.assertTrue(accepted)
        branches = tr.contact_diagnostic["support_branch_results"]
        self.assertFalse(branches[0]["accepted"])
        self.assertTrue(branches[1]["accepted"])
        self.assertEqual(tr.contact_diagnostic["selected_support_side"], 1)

    def test_pose_scaling_preserves_problem_and_avoids_mixed_unit_one_step_stop(self):
        # Millimetre translation and order-one latent variables reproduce the
        # premature success observed with the real frame48 full covariance.
        scale = np.array([.001, .25])
        covariance = np.outer(scale, scale)*np.array([[1., .8], [.8, 1.]])
        precision = np.linalg.inv(covariance)
        projection = np.array([[1000., 40.], [0., 20.]])
        pixels = np.array([100., 40.])
        def objective(delta):
            return .5*np.sum((projection @ delta-pixels)**2)/64+.5*delta @ precision @ delta
        def gradient(delta):
            return projection.T @ (projection @ delta-pixels)/64+precision @ delta
        constraints = {"type": "ineq",
            "fun": lambda delta: 100*np.array([.0048-delta[0]-.03*delta[1], delta[0]+.03*delta[1]+.0048]),
            "jac": lambda delta: 100*np.array([[-1., -.03], [1., .03]])}
        bounds, initial = [(None, None), (-6., 6.)], np.zeros(2)
        options = {"maxiter": 30, "ftol": .01}
        unscaled = scipy_minimize(objective, initial, jac=gradient, bounds=bounds,
            constraints=constraints, method="SLSQP", options=options)
        self.assertTrue(unscaled.success)
        self.assertEqual(unscaled.nit, 1)
        captured = {}

        def inspect_scaled_problem(fun, coordinates, **kwargs):
            probe = np.array([.0004, .12])
            scaled_probe = probe/scale
            np.testing.assert_array_equal(coordinates*scale, initial)
            self.assertAlmostEqual(fun(scaled_probe), objective(probe))
            np.testing.assert_allclose(kwargs["jac"](scaled_probe), gradient(probe)*scale)
            eps = 1e-6
            numerical = np.array([(fun(scaled_probe+axis*eps)-fun(scaled_probe-axis*eps))/(2*eps)
                                  for axis in np.eye(2)])
            np.testing.assert_allclose(kwargs["jac"](scaled_probe), numerical, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(kwargs["constraints"]["fun"](scaled_probe), constraints["fun"](probe))
            np.testing.assert_allclose(kwargs["constraints"]["jac"](scaled_probe),
                                       constraints["jac"](probe)*scale[None])
            self.assertEqual(kwargs["bounds"], [(None, None), (-24., 24.)])
            self.assertEqual(kwargs["options"], options)
            result = scipy_minimize(fun, coordinates, **kwargs)
            captured["optimizer_gradient"] = result.jac.copy()
            return result

        with patch("pose_app.smplh_world_motion_tracker.minimize", side_effect=inspect_scaled_problem):
            scaled = _minimize_scaled_pose(objective, initial, jac=gradient, bounds=bounds,
                constraints=constraints, covariance=covariance, options=options)
        self.assertTrue(scaled.success)
        self.assertLess(scaled.fun, unscaled.fun-5.)
        self.assertGreaterEqual(constraints["fun"](scaled.x).min(), -1e-7)
        self.assertAlmostEqual(scaled.fun, objective(scaled.x))
        np.testing.assert_allclose(scaled.jac, captured["optimizer_gradient"]/scale)

    def test_shaped_pelvis_transform_preserves_every_mesh_vertex(self):
        rotation = Rotation.from_rotvec([.13, .05, -.08]).as_matrix()
        tr = world_tracker(rotation=rotation)
        expected = tr.model.evaluate(tr.model.initial_pose, mesh=True)["vertices_m"][0] @ rotation.T + [0, 0, -1.96]
        actual = tr.model.evaluate(tr.pose, mesh=True)["vertices_m"][0]
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_camera_motion_does_not_create_world_body_velocity(self):
        tr = world_tracker()
        fixed = tr.pose.copy()
        px = observe(tr)
        tr.step(0., px[0], px[1])
        for i in range(1, 21):
            R = Rotation.from_rotvec([0, .003*i, .001*i]).as_matrix()
            tr.set_scene_frame(R, [.003*i, 0, -1.96], "transition", "updated")
            px = observe(tr, fixed)
            output = tr.step(i*.03, px[0], px[1])
        np.testing.assert_allclose(tr.pose.translation_m, fixed.translation_m, atol=1e-8)
        np.testing.assert_allclose(output["generalized_velocity"], 0, atol=1e-7)
        np.testing.assert_allclose(output["generalized_acceleration"], 0, atol=1e-6)

    def test_contact_changes_complete_model_and_retains_episode_anchors(self):
        tr = world_tracker(contact=True)
        tr.set_scene_frame(np.eye(3), [0, 0, -1.96], "stage2_feet_static_walker_moving", "updated")
        tr.pose.translation_m[2] += .025
        px = observe(tr)
        out = tr.step(0., px[0], px[1])
        self.assertTrue(out["ground_constraint"]["constraint_satisfied"])
        self.assertEqual(out["ground_constraint"]["map_solver_calls"], 1)
        self.assertEqual(out["ground_constraint"]["support_extra_solver_calls"], 0)
        self.assertLessEqual(max(abs(np.array(out["ground_constraint"]["sole_min_m"])-SHOE_HEIGHT_M)), .0052)
        anchor = tr.sole_anchors.copy()
        tr.set_scene_frame(np.eye(3), [0, 0, -1.96], "transition", "held")
        self.assertIsNotNone(tr.sole_anchors)
        tr._conditions(tr._geometry(tr.model.evaluate(tr.pose, mesh=True)["vertices_m"][0]))
        tr.set_scene_frame(np.eye(3), [0, 0, -1.96], "stage2_feet_static_walker_moving", "updated")
        np.testing.assert_array_equal(tr.sole_anchors, anchor)
        np.testing.assert_allclose(out["vertices_m"], tr.model.evaluate(tr.pose, mesh=True)["vertices_m"][0])
        self.assertFalse(out["used_views"].any())

    def test_expired_frame_and_missing_support_do_not_hide_violation(self):
        tr = world_tracker(contact=True)
        px = observe(tr)
        tr.step(0., px[0], px[1])
        missing = np.full((12, 3), np.nan)
        expired = tr.step(.2, missing, missing)
        self.assertFalse(expired["pose_available"])
        self.assertFalse(expired["ground_constraint"]["constraint_satisfied"])
        self.assertTrue(np.isnan(expired["vertices_m"]).all())

    def test_feasible_but_failed_optimizer_keeps_prediction_and_records_conflict(self):
        tr = world_tracker(contact=True)
        tr.set_scene_frame(np.eye(3), [0, 0, -1.96], "stage2_feet_static_walker_moving", "updated")
        px = observe(tr)
        tr.step(0., px[0], px[1])
        prior = tr.pose.copy()
        failed = SimpleNamespace(x=np.zeros(tr.n), success=False, message="Iteration limit reached",
                                 nit=35, fun=0.)
        with patch("pose_app.smplh_world_motion_tracker.minimize", return_value=failed):
            output = tr.step(.03, px[0], px[1])
        self.assertEqual(output["status"], "predicted")
        self.assertTrue(output["pose_available"])
        self.assertTrue(output["ground_constraint"]["constraint_satisfied"])
        self.assertEqual(output["ground_constraint"]["status"], "constraint_conflict")
        self.assertFalse(output["ground_constraint"]["optimizer_success"])
        self.assertFalse(output["used_views"].any())
        self.assertFalse(output["derivative_valid"].any())
        self.assertEqual(tr.last_support, 0.)
        np.testing.assert_allclose(tr.pose.translation_m, prior.translation_m, atol=1e-10)
        self.assertTrue(np.isfinite(output["raw_velocity_mm_s"]).all())

    def test_short_missing_observation_cannot_certify_prediction_outside_whole_foot_bound(self):
        tr = world_tracker(contact=True)
        tr.set_scene_frame(np.eye(3), [0, 0, -1.96], "stage2_feet_static_walker_moving", "updated")
        px = observe(tr)
        tr.step(0., px[0], px[1])
        anchor = tr.sole_anchors.copy()
        tr.velocity[:] = 0.
        tr.acceleration[:] = 0.
        tr.velocity[0] = 1.
        missing = np.full((12, 3), np.nan)
        output = tr.step(.03, missing, missing)
        self.assertEqual(output["status"], "predicted")
        self.assertTrue(output["pose_available"])
        self.assertFalse(output["ground_constraint"]["constraint_satisfied"])
        self.assertGreater(min(output["ground_constraint"]["sole_max_displacement_m"]), .029)
        self.assertFalse(output["derivative_valid"].any())
        self.assertTrue(np.isnan(output["velocity_mm_s"]).all())
        self.assertTrue(np.isnan(output["acceleration_mm_s2"]).all())
        self.assertTrue(np.isfinite(output["vertices_m"]).all())
        self.assertTrue(np.isfinite(output["raw_velocity_mm_s"]).all())
        np.testing.assert_array_equal(tr.sole_anchors, anchor)

    def test_improved_feasible_budget_result_keeps_failure_and_suppresses_derivatives(self):
        tr = world_tracker(contact=True)
        tr.set_scene_frame(np.eye(3), [0, 0, -1.96], "stage2_feet_static_walker_moving", "updated")
        px = observe(tr); tr.step(0., px[0], px[1])
        true = tr.pose.copy(); true.translation_m[0] += .005
        px = observe(tr, true)
        def budget_result(fun, x0, **kwargs):
            candidate = x0.copy()
            candidate[0] += .001/np.sqrt(tr.covariance[0, 0])
            return SimpleNamespace(x=candidate, fun=fun(candidate), success=False,
                                   status=9, message="Iteration limit reached", nit=30)
        with patch("pose_app.smplh_world_motion_tracker.minimize", side_effect=budget_result):
            out = tr.step(.03, px[0], px[1])
        self.assertEqual(out["status"], "feasible_unconverged")
        self.assertTrue(out["ground_constraint"]["constraint_satisfied"])
        self.assertFalse(out["ground_constraint"]["optimizer_success"])
        self.assertLess(out["ground_constraint"]["objective"], out["ground_constraint"]["feasible_start_objective"])
        self.assertFalse(out["derivative_valid"].any())
        self.assertTrue(np.isfinite(out["raw_velocity_mm_s"]).all())

    def test_earlier_feasible_improvement_survives_an_infeasible_last_iterate(self):
        tr = world_tracker(contact=True)
        tr.set_scene_frame(np.eye(3), [0, 0, -1.96], "stage2_feet_static_walker_moving", "updated")
        px = observe(tr)
        tr.step(0., px[0], px[1])
        prior = tr.pose.copy()
        anchors = tr.sole_anchors.copy()
        target = prior.copy()
        target.translation_m[0] += .006
        px = observe(tr, target)
        captured = {}

        def failed_last_iterate(fun, x0, **kwargs):
            captured["covariance"] = tr.covariance.copy()
            earlier = x0.copy()
            earlier[0] += .002/np.sqrt(tr.covariance[0, 0])
            captured["earlier_cost"] = fun(earlier)
            last = x0.copy()
            last[0] += .020/np.sqrt(tr.covariance[0, 0])
            # This evaluation also overwrites the model cache. The incumbent
            # must own its delta rather than aliasing that mutable cache.
            return SimpleNamespace(x=last, fun=fun(last), success=False,
                                   status=9, message="Iteration limit reached", nit=30)

        with patch("pose_app.smplh_world_motion_tracker.minimize", side_effect=failed_last_iterate):
            output = tr.step(.03, px[0], px[1])
        diagnostic = output["ground_constraint"]
        self.assertEqual(output["status"], "feasible_unconverged")
        self.assertTrue(diagnostic["constraint_satisfied"])
        self.assertFalse(diagnostic["optimizer_success"])
        self.assertEqual(diagnostic["selected_estimate"], "best_feasible_improving_current_frame_iterate")
        self.assertLess(diagnostic["optimizer_last_minimum_slack_m"], -.009)
        self.assertAlmostEqual(diagnostic["objective"], captured["earlier_cost"])
        self.assertLess(diagnostic["objective"], diagnostic["feasible_start_objective"])
        self.assertNotEqual(diagnostic["objective"], diagnostic["optimizer_last_objective"])
        self.assertAlmostEqual(tr.pose.translation_m[0] - prior.translation_m[0], .002)
        self.assertFalse(output["derivative_valid"].any())
        self.assertTrue(np.isnan(output["velocity_mm_s"]).all())
        self.assertTrue(np.isfinite(output["raw_velocity_mm_s"]).all())
        np.testing.assert_array_equal(tr.sole_anchors, anchors)
        np.testing.assert_allclose(tr.covariance, captured["covariance"], rtol=1e-10, atol=1e-12)

    def test_budget_can_find_first_feasible_pose_without_a_feasible_start(self):
        for feasible_last in (True, False):
            with self.subTest(feasible_last=feasible_last):
                tr = world_tracker(contact=True)
                tr.set_scene_frame(np.eye(3), [0, 0, -1.96], "stage2_feet_static_walker_moving", "updated")
                px = observe(tr)
                tr.step(0., px[0], px[1])
                anchors = tr.sole_anchors.copy()
                tr.pose.translation_m[0] += .03
                tr.last_accepted_pose = tr.pose.copy()
                covariance = tr.covariance.copy()
                correction = np.zeros(tr.n); correction[0] = -.03
                calls = []

                def budget_finds_contact(fun, x0, **kwargs):
                    calls.append(1)
                    if len(calls) == 1:
                        # Bootstrap does not provide a valid start.
                        return SimpleNamespace(x=x0.copy(), fun=fun(x0), success=False,
                                               status=9, message="Iteration limit reached", nit=80)
                    fun(correction)
                    last = correction.copy() if feasible_last else x0.copy()
                    return SimpleNamespace(x=last, fun=fun(last), success=False,
                                           status=9, message="Iteration limit reached", nit=80)

                reasons = np.full((2, 12), "available", dtype=object)
                with patch("pose_app.smplh_world_motion_tracker._minimize_scaled_pose", side_effect=budget_finds_contact):
                    accepted, used, status = tr._constrained_update(px, np.ones((2, 12), bool), reasons)
                self.assertTrue(accepted)
                self.assertEqual(status, "feasible_unconverged")
                diagnostic = tr.contact_diagnostic
                self.assertIsNone(diagnostic["feasible_start_objective"])
                self.assertFalse(diagnostic["feasible_start_available"])
                self.assertEqual(diagnostic["budget_acceptance_reference"], "first_feasible_current_frame_estimate")
                self.assertFalse(diagnostic["optimizer_success"])
                self.assertEqual(diagnostic["selected_estimate"], "best_feasible_current_frame_iterate_without_feasible_start")
                self.assertGreaterEqual(tr._conditions(tr._geometry(tr.model.evaluate(tr.pose, mesh=True)["vertices_m"][0])).min(), -1e-6)
                np.testing.assert_array_equal(tr.sole_anchors, anchors)
                np.testing.assert_allclose(tr.covariance, covariance, rtol=1e-10, atol=1e-12)
                output = tr._output(.03, status, used, reasons, px)
                self.assertFalse(output["derivative_valid"].any())
                self.assertTrue(np.isnan(output["velocity_mm_s"]).all())

    def test_no_feasible_start_does_not_admit_bad_geometry_or_other_solver_failures(self):
        for failure_status, evaluate_feasible in ((9, False), (8, True)):
            with self.subTest(failure_status=failure_status):
                tr = world_tracker(contact=True)
                tr.set_scene_frame(np.eye(3), [0, 0, -1.96], "stage2_feet_static_walker_moving", "updated")
                px = observe(tr); tr.step(0., px[0], px[1])
                tr.pose.translation_m[0] += .03
                tr.last_accepted_pose = tr.pose.copy()
                prior = tr.pose.copy()
                correction = np.zeros(tr.n); correction[0] = -.03
                calls = []

                def failed_solver(fun, x0, **kwargs):
                    calls.append(1)
                    if len(calls) > 1 and evaluate_feasible:
                        fun(correction)
                    last = correction if len(calls) > 1 and evaluate_feasible else x0
                    return SimpleNamespace(x=last.copy(), fun=fun(last), success=False,
                                           status=failure_status, message="test failed solve", nit=80)

                with patch("pose_app.smplh_world_motion_tracker._minimize_scaled_pose", side_effect=failed_solver):
                    accepted, used, status = tr._constrained_update(px, np.ones((2, 12), bool),
                        np.full((2, 12), "available", dtype=object))
                self.assertFalse(accepted)
                self.assertFalse(used.any())
                self.assertEqual(status, "constraint_conflict")
                np.testing.assert_array_equal(tr.pose.translation_m, prior.translation_m)

    def test_known_feasible_start_still_requires_budget_objective_improvement(self):
        tr = world_tracker(contact=True)
        tr.set_scene_frame(np.eye(3), [0, 0, -1.96], "stage2_feet_static_walker_moving", "updated")
        px = observe(tr); tr.step(0., px[0], px[1])

        def unchanged_budget_result(fun, x0, **kwargs):
            return SimpleNamespace(x=x0.copy(), fun=fun(x0), success=False,
                                   status=9, message="Iteration limit reached", nit=80)

        with patch("pose_app.smplh_world_motion_tracker._minimize_scaled_pose", side_effect=unchanged_budget_result):
            accepted, used, status = tr._constrained_update(px, np.ones((2, 12), bool),
                np.full((2, 12), "available", dtype=object))
        self.assertFalse(accepted)
        self.assertFalse(used.any())
        self.assertEqual(status, "constraint_conflict")
        self.assertTrue(tr.contact_diagnostic["feasible_start_available"])
        self.assertEqual(tr.contact_diagnostic["budget_acceptance_reference"], "known_feasible_start_objective")

    def test_position_velocity_acceleration_are_jointly_corrected(self):
        tr = world_tracker()
        px = observe(tr); tr.step(0., px[0], px[1])
        origin = tr.pose.translation_m.copy()
        for i in range(1, 65):
            now = i*.035
            true = MotionPose(origin+[.1*now**2, 0, 0], np.eye(3), np.zeros(1))
            tr.set_scene_frame(np.eye(3), [.01*now, 0, -1.96], "transition", "updated")
            px = observe(tr, true)
            out = tr.step(now, px[0], px[1])
        self.assertAlmostEqual(out["generalized_velocity"][0], .2*now, delta=.04)
        self.assertAlmostEqual(out["generalized_acceleration"][0], .2, delta=.08)


if __name__ == "__main__":
    unittest.main()
