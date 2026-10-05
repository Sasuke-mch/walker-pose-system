import unittest

import numpy as np
from experiment_arm_triangulation import evaluate, geometry_reasons
from pose_app.fisheye_camera import StereoFisheyeCalibration, fisheye_project_numpy


class ArmTriangulationExperimentTests(unittest.TestCase):
    def setUp(self):
        k = np.array([[500., 0., 960.], [0., 500., 540.], [0., 0., 1.]])
        self.cal = StereoFisheyeCalibration(k, np.zeros(4), k.copy(), np.zeros(4),
                                           np.eye(3), np.array([-120., 0., 0.]), 1920, 1080)

    def observations(self):
        xyz = np.array([[j * 8. - 60., j * 3., 750. + j * 10.] for j in range(17)])
        left = fisheye_project_numpy(xyz, self.cal.K0, self.cal.D0)
        right = fisheye_project_numpy(xyz + self.cal.T_cam0_to_cam1_mm, self.cal.K1, self.cal.D1)
        return [np.concatenate([p, np.ones((17, 1))], axis=1)[None] for p in (left, right)]

    def test_exact_projection_agrees_and_units_are_mm(self):
        summary, records = evaluate(*self.observations(), self.cal)
        self.assertEqual(summary["total_slots"], 6)
        self.assertTrue(all(r["common_diagnostic_pass"] for r in records))
        self.assertLess(max(r["displacement_mm"] for r in records), 1e-6)
        np.testing.assert_allclose(records[0]["refined_mm"], [-20., 15., 800.], atol=1e-6)

    def test_noisy_pixels_objective_nonincrease(self):
        left, right = self.observations()
        right[0, 5:11, 1] += 8.
        _, rows = evaluate(left, right, self.cal)
        self.assertTrue(all(r["pixel_sse_change"] <= 1e-7 for r in rows))

    def test_invalid_slot_retained(self):
        left, right = self.observations()
        left[0, 9, 0] = -1
        summary, rows = evaluate(left, right, self.cal)
        self.assertEqual(summary["total_slots"], 6)
        row = next(r for r in rows if r["joint"] == "left_wrist")
        self.assertEqual(row["state"], "unavailable")
        self.assertEqual(row["baseline_body_reason"], "out_of_raw_image_bounds")

    def test_gate_boundaries(self):
        self.assertEqual(geometry_reasons(np.array([0., 0., 500.]), [10., 10.], 1., self.cal), [])
        self.assertIn("nonpositive_depth", geometry_reasons(np.array([0., 0., -500.]), [0., 0.], 2., self.cal))
        self.assertIn("reprojection_over_10px", geometry_reasons(np.array([0., 0., 500.]), [10.01, 0.], 2., self.cal))


if __name__ == "__main__":
    unittest.main()
