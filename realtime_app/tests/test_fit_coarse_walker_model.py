from __future__ import annotations

import unittest

import numpy as np

from tools.fit_coarse_walker_model import estimate_dimensions, fit_ground_yaw_translation


class FitCoarseWalkerModelTests(unittest.TestCase):
    def test_front_rail_height_follows_camera_height_with_mounting_offset(self) -> None:
        anchors = {
            "kp_03": np.array([0.0, 0.0, 830.0]),
            "kp_04": np.array([0.0, 0.0, 440.0]),
            "kp_05": np.array([0.0, 0.0, 450.0]),
            "kp_06": np.array([0.0, 0.0, 850.0]),
        }
        dimensions = estimate_dimensions(anchors, camera_height_mm=705.0, camera_center_above_front_rail_mm=30.0)
        self.assertAlmostEqual(dimensions.front_rail_height_mm, 675.0)
        self.assertGreater(dimensions.front_rail_height_mm, dimensions.middle_rail_height_mm)

    def test_planar_fit_recovers_yaw_and_translation_without_tilting_ground(self) -> None:
        source = {
            "kp_01": np.array([2.0, 1.0, 0.0]),
            "kp_02": np.array([-2.0, 1.0, 0.0]),
            "kp_03": np.array([0.0, -1.0, 5.0]),
        }
        angle = np.deg2rad(12.0)
        expected_rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        )
        expected_translation = np.array([10.0, -20.0, 0.0])
        target = {name: expected_rotation @ point + expected_translation for name, point in source.items()}
        rotation, translation, audit = fit_ground_yaw_translation(source, target)
        self.assertTrue(np.allclose(rotation, expected_rotation, atol=1e-10))
        self.assertTrue(np.allclose(translation, expected_translation, atol=1e-10))
        self.assertAlmostEqual(audit["horizontal_residual_max_mm"], 0.0, places=9)
        self.assertTrue(np.allclose(rotation[2], [0.0, 0.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
