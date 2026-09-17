import unittest

import cv2
import numpy as np

from pose_app.calibration import StereoCalibration
from pose_app.static_background_stereo_frontend import (
    LocalStereoRectification,
    StereoFrontendCriteria,
    _rectified_triangulation,
    estimate_rectified_pnp_candidate,
    mutual_ratio_matches,
    person_exclusion_feature_mask,
    triangulate_indexed_matches,
    upright_persons_to_raw,
)


class StaticBackgroundStereoFrontendTests(unittest.TestCase):
    def calibration(self):
        return StereoCalibration(
            camera_model="pinhole",
            left_image_size=(640, 480), right_image_size=(640, 480),
            left_K=np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]]),
            left_D=np.zeros(5),
            right_K=np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]]),
            right_D=np.zeros(5),
            R=np.eye(3), T=np.array([-100.0, 0.0, 0.0]), length_unit="mm",
        )

    def test_person_mask_excludes_keypoint_hull(self):
        person = {"keypoints": [[250, 180, 1], [390, 180, 1], [390, 380, 1], [250, 380, 1]]}
        mask = person_exclusion_feature_mask(
            (480, 640), [person], scale=1.0, keypoint_score_threshold=0.25,
            dilation_fraction=0.02,
        )
        self.assertEqual(int(mask[280, 320]), 0)
        self.assertEqual(int(mask[100, 100]), 255)

    def test_upright_keypoints_follow_inverse_runtime_rotations(self):
        person = [{"keypoints": [[100.0, 300.0, 0.9], [200.0, 400.0, 0.8]]}]
        left = upright_persons_to_raw(person, "left", (1080, 1920))[0]["keypoints"]
        right = upright_persons_to_raw(person, "right", (1080, 1920))[0]["keypoints"]
        np.testing.assert_allclose(left[0][:2], [1619.0, 100.0])
        np.testing.assert_allclose(right[0][:2], [300.0, 979.0])

    def test_mutual_ratio_requires_reverse_agreement(self):
        a = np.zeros((3, 32), np.uint8)
        b = np.zeros((3, 32), np.uint8)
        a[1, 0] = 255
        a[2, 1] = 255
        b[1, 0] = 255
        b[2, 1] = 255
        matches = mutual_ratio_matches(a, b, ratio=0.8)
        self.assertEqual({(m.queryIdx, m.trainIdx) for m in matches}, {(0, 0), (1, 1), (2, 2)})

    def test_triangulation_recovers_metric_point(self):
        calibration = self.calibration()
        xyz = np.array([[50.0, 20.0, 1000.0]])
        left = calibration.project_left(xyz)[0]
        right = calibration.project_right(xyz)[0]
        key_left = (cv2.KeyPoint(float(left[0]), float(left[1]), 5),)
        key_right = (cv2.KeyPoint(float(right[0]), float(right[1]), 5),)
        match = cv2.DMatch(0, 0, 0.0)
        result, audit = triangulate_indexed_matches(
            calibration, key_left, key_right, [match], StereoFrontendCriteria()
        )
        self.assertEqual(audit["geometry_inliers"], 1)
        self.assertTrue(np.allclose(result[0], xyz[0], atol=1e-6))

    def rectification(self):
        return LocalStereoRectification(
            np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0)),
            np.eye(3), np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]]),
            np.array([-100.0, 0.0, 0.0]),
        )

    def test_rectified_triangulation_recovers_metric_point(self):
        xyz = np.array([50.0, 20.0, 1000.0])
        left = np.array([345.0, 250.0])
        right = np.array([295.0, 250.0])
        result, audit = _rectified_triangulation(
            (cv2.KeyPoint(*left, 5),), (cv2.KeyPoint(*right, 5),),
            [cv2.DMatch(0, 0, 0.0)], self.rectification(), StereoFrontendCriteria(),
            maximum_vertical_error_px=1.0,
        )
        self.assertEqual(audit["geometry_inliers"], 1)
        self.assertTrue(np.allclose(result[0], xyz, atol=1e-6))

    def test_rectified_pnp_returns_original_left_camera_transform(self):
        object_points = np.array([
            [-200, -100, 1200], [200, -100, 1200], [-200, 150, 1400], [200, 150, 1400],
            [-100, 50, 1800], [250, 80, 2000], [0, -200, 1600], [120, 220, 1700],
        ], dtype=float)
        translation = np.array([15.0, -8.0, 30.0])
        image, _ = cv2.projectPoints(
            object_points, np.zeros(3), translation, self.rectification().virtual_K, None
        )
        links = np.column_stack((object_points, image.reshape(-1, 2)))
        result = estimate_rectified_pnp_candidate(
            links, self.rectification(), reprojection_threshold_px=1.0,
            iterations=100, minimum_inliers=6,
        )
        self.assertEqual(result["status"], "accepted")
        self.assertTrue(np.allclose(result["R_to_from"], np.eye(3), atol=1e-5))
        self.assertTrue(np.allclose(result["q_to_from"], translation, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
