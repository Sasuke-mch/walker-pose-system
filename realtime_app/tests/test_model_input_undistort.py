import unittest

import cv2
import numpy as np

from pose_app.model_undistort import FisheyeModelInput


class FisheyeModelInputTests(unittest.TestCase):
    def setUp(self):
        self.size = (640, 480)
        self.K = np.asarray(
            [[420.0, 0.0, 320.0], [0.0, 420.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.D = np.asarray([-0.03, 0.002, 0.0, 0.0], dtype=np.float64)

    def test_image_preserves_runtime_size(self):
        preprocessor = FisheyeModelInput(self.K, self.D, self.size)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = preprocessor.image(image)
        self.assertEqual(result.shape, image.shape)

    def test_undistorted_points_map_back_to_raw(self):
        preprocessor = FisheyeModelInput(self.K, self.D, self.size)
        points = np.asarray([[120.0, 100.0], [320.0, 240.0], [520.0, 380.0]])
        raw = preprocessor.undistorted_to_raw(points)
        self.assertTrue(np.isfinite(raw).all())
        expected = cv2.fisheye.distortPoints(
            cv2.undistortPoints(points.reshape(-1, 1, 2), self.K, np.zeros((4, 1))),
            self.K,
            self.D.reshape(-1, 1),
        ).reshape(-1, 2)
        np.testing.assert_allclose(raw, expected, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
