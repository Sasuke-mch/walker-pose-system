"""Check attribution invariants independently of the frozen capture."""
import unittest

import numpy as np
from audit_limb_chain import (
    bend,
    filtered_stage_feet,
    segment_smooth,
    step_parts,
    temporal,
    transform,
)
from scipy.spatial.transform import Rotation


class LimbChainAuditTests(unittest.TestCase):
    def test_constant_body_pose_jump_is_scene_only(self):
        x = np.broadcast_to(np.array([[0., 100., 800.], [50., 0., 400.]]), (4, 2, 3)).copy()
        r = Rotation.from_euler("z", np.array([0, 0, 20, 20])[:, None], degrees=True).as_matrix()
        t = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 50], [0, 0, 50.]])
        world, camera, pose = step_parts(x, r, t)
        np.testing.assert_allclose(camera, 0)
        np.testing.assert_allclose(world, pose)
        self.assertGreater(np.linalg.norm(pose[1]), 50)

    def test_fixed_scene_body_change_is_camera_only(self):
        x = np.arange(36, dtype=float).reshape(4, 3, 3)
        r = np.broadcast_to(Rotation.from_euler("x", 25, degrees=True).as_matrix(), (4, 3, 3))
        t = np.broadcast_to([10., 20., 30.], (4, 3))
        world, camera, pose = step_parts(x, r, t)
        np.testing.assert_allclose(pose, 0)
        np.testing.assert_allclose(world, camera)

    def test_rigid_transform_preserves_residual_and_bend(self):
        rng = np.random.default_rng(31)
        a, b = rng.normal(size=(10, 3, 3)), rng.normal(size=(10, 3, 3))
        r = Rotation.random(10, random_state=rng).as_matrix()
        t = rng.normal(size=(10, 3))
        ga, gb = transform(a, r, t), transform(b, r, t)
        np.testing.assert_allclose(np.linalg.norm(a-b, axis=-1), np.linalg.norm(ga-gb, axis=-1))
        np.testing.assert_allclose(bend(a, (0, 1, 2)), bend(ga, (0, 1, 2)))

    def test_diagnostic_kernel_does_not_fill_missing_or_cross_gap(self):
        x = np.arange(21, dtype=float).reshape(7, 1, 3)
        x[2] += 10
        mask = np.ones((7, 1), bool)
        mask[3] = False
        y = segment_smooth(x, mask)
        self.assertTrue(np.isnan(y[3]).all())
        np.testing.assert_array_equal(y[[2, 4]], x[[2, 4]])
        result = temporal(y[:, 0], mask[:, 0])
        self.assertEqual(result["step"]["n"], 4)
        self.assertEqual(result["second_difference"]["n"], 2)

    def test_actual_foot_median_retains_missing_state_and_resets(self):
        sc = {"accepted": np.ones((6, 17), bool),
              "raw_triangulated_points_mm": np.broadcast_to(np.arange(6)[:, None, None], (6, 17, 3)).astype(float)}
        sc["accepted"][2, 15] = False
        rows = [{"stage": "stage2_feet_static_walker_moving", "pose_audit": {"causal_foot_window_frames": count}}
                for count in [1, 2, 2, 3]]
        rows += [{"stage": "stage1_walker_static_human_moving", "pose_audit": {}},
                 {"stage": "stage2_feet_static_walker_moving", "pose_audit": {"causal_foot_window_frames": 1}}]
        filtered = filtered_stage_feet(sc, rows)
        np.testing.assert_allclose(filtered[1], .5)
        self.assertTrue(np.isnan(filtered[2]).all())
        np.testing.assert_allclose(filtered[3], 1)
        self.assertTrue(np.isnan(filtered[4]).all())
        np.testing.assert_allclose(filtered[5], 5)


if __name__ == "__main__":
    unittest.main()
