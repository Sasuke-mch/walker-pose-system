"""Checks for the concrete ground/heel geometry contracts, without model assets."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pose_app.foot_ground_reference import SHOE_HEIGHT_M

from pose_app.smplh_ground_constraints import (
    HEEL_VERTEX_IDS, foot_geometry, ground_inequalities, load_sole_vertex_indices,
)


class GroundConstraintTests(unittest.TestCase):
    def setUp(self):
        self.soles = ([10, 11, 12], [20, 21, 22])
        self.vertices = np.zeros((6890, 3))
        self.vertices[list(HEEL_VERTEX_IDS), 2] = [.035, .045]
        self.vertices[:, 2] += SHOE_HEIGHT_M

    def test_heel_is_actual_surface_and_not_projected_to_sole(self):
        geometry = foot_geometry(self.vertices, self.soles)
        np.testing.assert_allclose(geometry["heel_m"][:, 2], np.array([.035, .045])+SHOE_HEIGHT_M)
        np.testing.assert_allclose(geometry["heel_height_above_sole_m"], [.035, .045])
        np.testing.assert_array_equal(geometry["sole_min_m"], [SHOE_HEIGHT_M, SHOE_HEIGHT_M])

    def test_original_surface_loss_matches_closed_form(self):
        self.vertices[self.soles[0], 2] = [-.004, .002, .012]
        self.vertices[self.soles[0], 2] += SHOE_HEIGHT_M
        geometry = foot_geometry(self.vertices, self.soles)
        z = self.vertices[self.soles[0], 2]
        soft = -.005 * np.log(np.exp(-z / .005).mean())
        expected = .010**2 * (np.sqrt(1. + ((soft-SHOE_HEIGHT_M) / .010)**2) - 1.)
        clearance = z-SHOE_HEIGHT_M
        expected += np.maximum(-clearance, 0.).dot(np.maximum(-clearance, 0.)) / len(z)
        self.assertAlmostEqual(geometry["sole_softmin_m"][0], soft)
        self.assertAlmostEqual(geometry["sole_surface_loss_m2"][0], expected)
        self.assertAlmostEqual(geometry["sole_penetration_m"][0], .004)

    def test_batch_stable_softminimum_and_left_right_order(self):
        v = np.stack([self.vertices, self.vertices.copy()])
        v[1, self.soles[0], 2] = SHOE_HEIGHT_M+10.
        v[1, self.soles[1], 2] = SHOE_HEIGHT_M-10.
        geometry = foot_geometry(v, self.soles)
        np.testing.assert_allclose(geometry["sole_softmin_m"], np.array([[0., 0.], [10., -10.]])+SHOE_HEIGHT_M)
        self.assertTrue(np.isfinite(geometry["sole_surface_loss_m2"]).all())

    def test_stage1_permits_swing_but_rejects_penetration(self):
        self.vertices[self.soles[0], 2] = SHOE_HEIGHT_M+.15
        self.vertices[self.soles[1], 2] = SHOE_HEIGHT_M-.006
        geometry = foot_geometry(self.vertices, self.soles)
        values = ground_inequalities(geometry, stage2=False)
        self.assertEqual(values.shape, (2,))
        self.assertGreater(values[0], 0.)
        self.assertLess(values[1], 0.)

    def test_stage2_rejects_floating_sole_even_when_heels_do_not_move(self):
        self.vertices[self.soles[0], 2] = SHOE_HEIGHT_M+.006
        geometry = foot_geometry(self.vertices, self.soles)
        values = ground_inequalities(geometry, stage2=True, heel_anchors_m=geometry["heel_m"])
        self.assertLess(values[2], 0.)
        np.testing.assert_allclose(values[-2:], [.01, .01])

    def test_heel_limit_is_episode_ball_not_per_axis_box(self):
        geometry = foot_geometry(self.vertices, self.soles)
        anchors = geometry["heel_m"].copy()
        geometry["heel_m"][0, :2] += [.008, .008]
        values = ground_inequalities(geometry, stage2=True, heel_anchors_m=anchors)
        self.assertLess(values[-2], 0.)
        self.assertGreater(values[-1], 0.)

    def test_contact_can_be_solved_before_creating_an_anchor(self):
        values = ground_inequalities(foot_geometry(self.vertices, self.soles), stage2=True)
        self.assertEqual(values.shape, (4,))
        self.assertTrue((values >= 0.).all())

    def test_whole_sole_anchor_rejects_rotation_around_a_fixed_heel(self):
        geometry = foot_geometry(self.vertices, self.soles)
        heels, soles = geometry["heel_m"].copy(), geometry["sole_points_m"].copy()
        geometry["sole_points_m"][0, 1, 0] += .011
        values = ground_inequalities(
            geometry, stage2=True, heel_anchors_m=heels, sole_anchors_m=soles,
        )
        self.assertEqual(values.shape, (8,))
        np.testing.assert_allclose(values[-4:-2], [.01, .01])
        self.assertAlmostEqual(values[-2], -.001)
        self.assertAlmostEqual(values[-1], .01)

    def test_batched_sole_anchors_keep_per_foot_maximum(self):
        geometry = foot_geometry(np.stack([self.vertices, self.vertices]), self.soles)
        anchors = geometry["sole_points_m"].copy()
        geometry["sole_points_m"][1, 1, 2, :2] += [.008, .008]
        values = ground_inequalities(geometry, stage2=True, sole_anchors_m=anchors)
        self.assertEqual(values.shape, (2, 6))
        self.assertLess(values[1, -1], 0.)
        self.assertTrue((values[0] >= 0.).all())

    def test_fixed_episode_anchors_broadcast_across_model_jacobian_batch(self):
        single = foot_geometry(self.vertices, self.soles)
        geometry = foot_geometry(np.stack([self.vertices] * 76), self.soles)
        values = ground_inequalities(
            geometry, stage2=True, heel_anchors_m=single["heel_m"],
            sole_anchors_m=single["sole_points_m"],
        )
        self.assertEqual(values.shape, (76, 8))
        self.assertTrue((values >= 0.).all())

    def test_loads_candidates_once_per_vertex(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sets.json"
            path.write_text(json.dumps({"sets": {
                "left_sole_surface_candidate": {"heel": [10, 11], "ball": [11, 12]},
                "right_sole_surface_candidate": {"heel": [20, 21], "toe": [22]},
            }}), encoding="utf-8")
            left, right = load_sole_vertex_indices(path)
        np.testing.assert_array_equal(left, [10, 11, 12])
        np.testing.assert_array_equal(right, [20, 21, 22])

    def test_invalid_geometry_and_inactive_anchors_are_not_silently_accepted(self):
        invalid = self.vertices.copy()
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            foot_geometry(invalid, self.soles)
        with self.assertRaises(ValueError):
            foot_geometry(self.vertices, ([1.5], [2]))
        geometry = foot_geometry(self.vertices, self.soles)
        with self.assertRaises(ValueError):
            ground_inequalities(geometry, stage2=False, heel_anchors_m=geometry["heel_m"])
        with self.assertRaises(ValueError):
            ground_inequalities(geometry, stage2=True, max_heel_displacement_m=0.)
        with self.assertRaises(ValueError):
            ground_inequalities(geometry, stage2=True, heel_anchors_m=0.)
        with self.assertRaises(ValueError):
            ground_inequalities(geometry, stage2=True, sole_anchors_m=0.)


if __name__ == "__main__":
    unittest.main()
