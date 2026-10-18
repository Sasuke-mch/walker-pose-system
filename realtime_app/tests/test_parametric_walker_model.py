from __future__ import annotations

import unittest

import numpy as np

from pose_app.parametric_walker_model import (
    WalkerDimensions,
    build_coarse_frame,
    point_segment_distance,
    transform_nodes,
)


class ParametricWalkerModelTests(unittest.TestCase):
    def test_catalog_dimensions_make_rigid_four_foot_frame(self) -> None:
        model = build_coarse_frame(WalkerDimensions(height_mm=840.0))
        nodes = model["nodes_walker_mm"]
        feet = np.asarray([nodes[name] for name in model["foot_nodes"]])
        self.assertTrue(np.allclose(feet[:, 2], 0.0))
        self.assertAlmostEqual(float(np.ptp(feet[:, 0])), 450.0)
        self.assertAlmostEqual(float(np.ptp(feet[:, 1])), 300.0)
        self.assertEqual(len(model["edges"]), 9)
        self.assertGreater(nodes["front_left_rail"][2], nodes["front_left_mid"][2])
        self.assertLess(nodes["front_left_rail"][2], nodes["front_left_top"][2])

    def test_transform_preserves_all_edge_lengths(self) -> None:
        model = build_coarse_frame(WalkerDimensions())
        angle = np.deg2rad(20.0)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        transformed = transform_nodes(model["nodes_walker_mm"], rotation, np.array([10.0, 20.0, 30.0]))
        for start, end in model["edges"]:
            before = np.linalg.norm(np.asarray(model["nodes_walker_mm"][start]) - model["nodes_walker_mm"][end])
            after = np.linalg.norm(np.asarray(transformed[start]) - transformed[end])
            self.assertAlmostEqual(float(before), float(after), places=8)

    def test_point_to_handle_distance_is_bounded_to_segment(self) -> None:
        distance, closest, fraction = point_segment_distance(
            np.array([3.0, 5.0, 0.0]), np.zeros(3), np.array([10.0, 0.0, 0.0])
        )
        self.assertAlmostEqual(distance, 5.0)
        self.assertTrue(np.allclose(closest, [3.0, 0.0, 0.0]))
        self.assertAlmostEqual(fraction, 0.3)

    def test_height_outside_catalog_range_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_coarse_frame(WalkerDimensions(height_mm=1000.0))

    def test_front_rail_must_stay_above_both_side_rails(self) -> None:
        with self.assertRaises(ValueError):
            build_coarse_frame(
                WalkerDimensions(middle_rail_height_mm=500.0, front_rail_height_mm=480.0)
            )


if __name__ == "__main__":
    unittest.main()
