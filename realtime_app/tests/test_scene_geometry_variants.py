import unittest

import cv2
import numpy as np

from pose_app.scene_geometry_variants import (
    angular_error_degrees,
    camera_attached_temporal_consensus,
    estimate_dense_ransac,
    estimate_region_consensus,
    estimate_sparse_tiles,
    estimate_weighted_irls,
    fit_articulated_template,
    fit_rigid_template,
    morphological_skeleton,
    reconstruct_walker_components,
)


class SceneGeometryVariantsTests(unittest.TestCase):
    def planar_fixture(self):
        yy, xx = np.mgrid[-5:6, -7:8]
        points = np.column_stack((xx.ravel() * 30.0, yy.ravel() * 25.0, 900.0 + 0.2 * xx.ravel()))
        pixels = np.column_stack(((xx.ravel() + 7) * 20 + 10, (yy.ravel() + 5) * 20 + 10))
        return points, pixels

    def test_dense_sparse_weighted_and_region_estimators_recover_plane(self):
        points, pixels = self.planar_fixture()
        expected = np.asarray([-0.2 / 30.0, 0.0, 1.0])
        dense = estimate_dense_ransac(points, distance_threshold_mm=2.0, minimum_points=20, seed=3)
        sparse = estimate_sparse_tiles(
            points, pixels, np.ones(len(points)), image_shape=(240, 320), tile_size_px=40,
            maximum_per_tile=2, distance_threshold_mm=2.0, minimum_points=20, seed=3,
        )
        weighted = estimate_weighted_irls(points, np.ones(len(points)), distance_threshold_mm=2.0, minimum_points=20)
        regional = estimate_region_consensus(
            points, pixels, image_shape=(240, 320), distance_threshold_mm=2.0,
            minimum_region_points=10, maximum_region_angle_deg=1.0,
        )
        for estimate in (dense, sparse, weighted, regional):
            self.assertEqual(estimate.status, "candidate")
            self.assertLess(angular_error_degrees(estimate.normal_left_camera, expected), 0.1)
            self.assertGreater(float(estimate.inlier_mask.mean()), 0.95)

    def test_weighted_fit_downweights_outliers(self):
        points, _ = self.planar_fixture()
        outliers = np.asarray([[0.0, 0.0, 200.0], [100.0, 50.0, 1800.0]])
        combined = np.vstack((points, outliers))
        weights = np.r_[np.ones(len(points)), [0.001, 0.001]]
        result = estimate_weighted_irls(combined, weights, minimum_points=20)
        self.assertEqual(result.status, "candidate")
        self.assertLess(angular_error_degrees(result.normal_left_camera, [0.0, 0.0, 1.0]), 1.0)

    def test_region_disagreement_is_rejected(self):
        points, pixels = self.planar_fixture()
        right = pixels[:, 0] >= 160
        points[right, 2] += 2.0 * points[right, 1]
        result = estimate_region_consensus(
            points, pixels, image_shape=(240, 320), minimum_region_points=10,
            maximum_region_angle_deg=5.0,
        )
        self.assertEqual(result.status, "unavailable")
        self.assertIn("regional_planes_disagree", result.reasons)

    def test_skeleton_and_component_tube_reconstruction(self):
        mask = np.zeros((100, 120), dtype=np.uint8)
        cv2.line(mask, (20, 15), (20, 85), 255, 9)
        skeleton = morphological_skeleton(mask)
        self.assertGreater(cv2.countNonZero(skeleton), 20)
        ys, xs = np.nonzero(mask)
        choose = np.arange(0, len(xs), 10)
        pixels = np.column_stack((xs[choose], ys[choose]))
        points = np.column_stack((np.full(len(choose), 100.0), ys[choose] * 5.0, np.full(len(choose), 800.0)))
        result = reconstruct_walker_components(
            points, pixels, mask, minimum_component_pixels=50, minimum_component_points=10,
        )
        self.assertEqual(result["status"], "candidate")
        self.assertEqual(len(result["line_primitives"]), 1)

    def test_template_fit_and_temporal_consensus_do_not_invent_observations(self):
        template = np.asarray([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0]])
        translation = np.asarray([30.0, -20.0, 800.0])
        observed = template + translation
        fit = fit_rigid_template(template, observed, maximum_rms_mm=0.01)
        self.assertEqual(fit["status"], "candidate")
        self.assertLess(fit["rms_mm"], 1e-6)
        consensus = camera_attached_temporal_consensus(
            [observed, observed + np.asarray([2.0, 1.0, 0.0])], voxel_size_mm=20.0,
            minimum_frame_support=2,
        )
        self.assertEqual(consensus["status"], "candidate")
        missing = camera_attached_temporal_consensus([observed], minimum_frame_support=2)
        self.assertEqual(missing["status"], "unavailable")

    def test_articulated_template_allows_segment_rotation_but_checks_joint_gap(self):
        segment = np.asarray([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
        left_observed = segment + np.asarray([10.0, 0.0, 800.0])
        angle = np.deg2rad(10.0)
        rotation = np.asarray([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
        right_observed = (rotation @ segment.T).T + np.asarray([10.0, 0.0, 800.0])
        result = fit_articulated_template(
            {"left": (segment, left_observed), "right": (segment, right_observed)},
            [("left", np.zeros(3), "right", np.zeros(3))],
            maximum_segment_rms_mm=0.01, maximum_joint_gap_mm=1.0,
        )
        self.assertEqual(result["status"], "candidate")


if __name__ == "__main__":
    unittest.main()
