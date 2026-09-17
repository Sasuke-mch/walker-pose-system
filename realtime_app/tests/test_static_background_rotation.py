import unittest

import cv2
import numpy as np

from pose_app.static_background_rotation import (
    RotationTrackCriteria,
    _estimate_rotation_only_ransac,
    estimate_rotation_from_normalized_tracks,
)


class StaticBackgroundRotationTests(unittest.TestCase):
    def test_rotation_only_ransac_recovers_pure_rotation_with_outliers(self):
        rng = np.random.default_rng(8)
        previous = rng.uniform([-0.7, -0.45], [0.7, 0.45], size=(80, 2))
        bearings = np.column_stack((previous, np.ones(len(previous))))
        bearings /= np.linalg.norm(bearings, axis=1, keepdims=True)
        angle = np.deg2rad(2.0)
        expected = np.array([
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ])
        moved = (expected @ bearings.T).T
        current = moved[:, :2] / moved[:, 2:3]
        current[:12] = rng.uniform([-0.8, -0.5], [0.8, 0.5], size=(12, 2))
        estimated, inliers = _estimate_rotation_only_ransac(
            previous, current, RotationTrackCriteria(minimum_tracks=12, minimum_inliers=20)
        )
        self.assertIsNotNone(estimated)
        self.assertGreaterEqual(int(np.count_nonzero(inliers)), 65)
        delta = estimated @ expected.T
        error = np.degrees(np.arccos(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)))
        self.assertLess(error, 0.1)

    def test_recovers_known_rotation_from_normalized_tracks(self):
        rng = np.random.default_rng(4)
        points = np.column_stack((rng.uniform(-1.2, 1.2, 120), rng.uniform(-0.7, 0.7, 120), rng.uniform(3.0, 8.0, 120)))
        rvec = np.deg2rad(np.array([1.0, -1.5, 0.7]))
        rotation, _ = cv2.Rodrigues(rvec)
        translation = np.array([0.12, -0.03, 0.05])
        current = (rotation @ points.T).T + translation
        previous_pixels = points[:, :2] / points[:, 2:3]
        current_pixels = current[:, :2] / current[:, 2:3]
        solved, audit = estimate_rotation_from_normalized_tracks(
            previous_pixels, current_pixels,
            RotationTrackCriteria(essential_threshold_normalized=1e-4, minimum_tracks=20, minimum_inliers=15),
        )
        self.assertEqual(audit["status"], "accepted")
        relative = solved @ rotation.T
        angle = np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
        self.assertLess(angle, 0.2)

    def test_fails_closed_with_too_few_tracks(self):
        solved, audit = estimate_rotation_from_normalized_tracks(
            np.zeros((4, 2)), np.zeros((4, 2)), RotationTrackCriteria(minimum_tracks=10)
        )
        self.assertIsNone(solved)
        self.assertEqual(audit["reason"], "too_few_normalized_tracks")


if __name__ == "__main__":
    unittest.main()
