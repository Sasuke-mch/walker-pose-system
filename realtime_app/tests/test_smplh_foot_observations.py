"""Contracts for adding toe pixels without a second state-information update."""
import unittest

import numpy as np
from pose_app.smplh_iterated_motion_tracker import (
    IteratedMAPConfig,
    SMPLHIteratedMotionTracker,
)
from tools.probe_smplh_foot_observations import FootPixelTracker, project

from tests.test_smplh_motion_tracker import AnalyticBodyMap, observation, tracker


class AnalyticToeMap(AnalyticBodyMap):
    def evaluate_batch(self, translations, roots, latents, mesh=False):
        data = super().evaluate_batch(translations, roots, latents, mesh)
        data['coco_m'] = np.concatenate((data['coco_m'], data['coco_m'][:, [1, 2, 3, 4]]), axis=1)
        return data


class FootObservationTests(unittest.TestCase):
    def make_tracker(self, cls=FootPixelTracker):
        return cls(AnalyticToeMap(), tracker().cal,
                   optimizer_config=IteratedMAPConfig(condition_accepted_iterate=True))

    def test_unavailable_foot_measurements_preserve_original_body_and_covariance(self):
        body_only, augmented = self.make_tracker(SMPLHIteratedMotionTracker), self.make_tracker()
        target = body_only.pose.copy()
        target.translation_m += [.03, -.02, .01]
        pixels = observation(body_only, target)
        augmented.extra_pixels = np.full((2, 4, 3), np.nan)
        for t in [0., .03, .06]:
            a, b = [tr.step(t, *pixels) for tr in [body_only, augmented]]
            np.testing.assert_array_equal(a['translation_m'], b['translation_m'])
            np.testing.assert_array_equal(body_only.covariance, augmented.covariance)
            self.assertEqual(b['projected_pixels'].shape, (2, 12, 2))
            self.assertEqual(b['coco_m'].shape, (17, 3))
            self.assertEqual(b['toe_m'].shape, (4, 3))

    def test_extra_toes_reduce_their_pixel_error_with_only_one_posterior(self):
        tr = self.make_tracker()
        body = observation(tr)
        tr.step(0., *body)
        target = tr.pose.copy()
        target.translation_m += [.06, .02, 0.]
        points = tr.model.evaluate(target)['coco_m'][0, 17:]
        toe_pixels = project(points, tr.cal)
        tr.extra_pixels = np.concatenate((toe_pixels, np.ones((2, 4, 1))), axis=-1)
        before = np.linalg.norm(project(tr.model.evaluate(tr.pose)['coco_m'][0, 17:], tr.cal) - toe_pixels)
        result = tr.step(.03, *body)
        after = np.linalg.norm(project(result['toe_m'], tr.cal) - toe_pixels)
        self.assertLess(after, before)
        self.assertEqual(result['iterated_update']['posterior_updates'], 1)
        self.assertEqual(int(tr.foot_used.sum()), 8)
        self.assertFalse(tr.solve_with_feet)
        self.assertEqual(tr.last_point_jacobian.shape, (12, 3, 7))
        self.assertEqual(result['projected_pixels'].shape, (2, 12, 2))

    def test_foot_data_cannot_bypass_original_minimum_body_support_gate(self):
        tr = self.make_tracker()
        body = observation(tr)
        tr.step(0., *body)
        toe_pixels = project(tr.model.evaluate(tr.pose)['coco_m'][0, 17:], tr.cal)
        tr.extra_pixels = np.concatenate((toe_pixels, np.ones((2, 4, 1))), axis=-1)
        missing = np.full((12, 3), np.nan)
        result = tr.step(.03, missing, missing)
        self.assertEqual(result['iterated_update']['status'], 'insufficient_joint_support')
        self.assertEqual(result['iterated_update']['posterior_updates'], 0)
        self.assertFalse(tr.foot_used.any())
        tr.extra_pixels = None
        result = tr.step(.06, *body)
        self.assertEqual(int(tr.foot_used.sum()), 0)
        self.assertEqual(result['projected_pixels'].shape, (2, 12, 2))

    def test_direction_measurement_does_not_fit_toe_midpoint_or_double_count_pair(self):
        a, b = self.make_tracker(), self.make_tracker()
        body = observation(a)
        pixels = project(a.model.evaluate(a.pose)['coco_m'][0, 17:], a.cal)
        for tr, shift in [(a, 0.), (b, 4.)]:
            tr.observation_mode = 'direction'
            tr.step(0., *body)
            tr.extra_pixels = np.concatenate((pixels + shift, np.ones((2, 4, 1))), axis=-1)
            result = tr.step(.03, *body)
            self.assertEqual(int(tr.foot_used.sum()), 4)
            self.assertEqual(result['iterated_update']['posterior_updates'], 1)
            self.assertEqual(result['projected_pixels'].shape, (2, 12, 2))
        np.testing.assert_allclose(a.pose.translation_m, b.pose.translation_m, atol=1e-10)
        np.testing.assert_allclose(a.covariance, b.covariance, atol=1e-10)

    def test_missing_one_toe_rejects_only_its_direction_pair(self):
        tr = self.make_tracker()
        tr.observation_mode = 'direction'
        body = observation(tr)
        tr.step(0., *body)
        pixels = project(tr.model.evaluate(tr.pose)['coco_m'][0, 17:], tr.cal)
        tr.extra_pixels = np.concatenate((pixels, np.ones((2, 4, 1))), axis=-1)
        tr.extra_pixels[0, 1] = np.nan
        result = tr.step(.03, *body)
        self.assertTrue(result['pose_available'])
        self.assertFalse(tr.foot_used[0, 0])
        self.assertEqual(int(tr.foot_used.sum()), 3)
        self.assertEqual(tr.foot_reason[0, 0], 'invalid_paired_toe_direction')


if __name__ == '__main__':
    unittest.main()
