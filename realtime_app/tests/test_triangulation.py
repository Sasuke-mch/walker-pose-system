from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pose_app.calibration import StereoCalibration
from pose_app.schema import PersonPose
from pose_app.triangulation import triangulate_matches


class TriangulationTests(unittest.TestCase):
    def calibration(self) -> StereoCalibration:
        K = np.asarray(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return StereoCalibration(
            camera_model="pinhole",
            left_image_size=(640, 480),
            right_image_size=(640, 480),
            left_K=K,
            left_D=np.zeros(5, dtype=np.float64),
            right_K=K.copy(),
            right_D=np.zeros(5, dtype=np.float64),
            R=np.eye(3, dtype=np.float64),
            T=np.asarray([-0.2, 0.0, 0.0], dtype=np.float64),
            length_unit="meter",
        )

    def test_synthetic_triangulation(self):
        calibration = self.calibration()
        xyz = np.asarray(
            [
                [0.01 * (index - 8), 0.015 * ((index % 5) - 2), 2.0 + 0.02 * index]
                for index in range(17)
            ],
            dtype=np.float64,
        )
        left_pixels = calibration.project_left(xyz)
        right_pixels = calibration.project_right(xyz)
        left = PersonPose(
            3,
            [100.0, 50.0, 500.0, 470.0],
            0.95,
            0.90,
            [[float(x), float(y), 0.9] for x, y in left_pixels],
        )
        right = PersonPose(
            7,
            [80.0, 50.0, 480.0, 470.0],
            0.94,
            0.89,
            [[float(x), float(y), 0.9] for x, y in right_pixels],
        )
        persons = triangulate_matches(
            [left],
            [right],
            calibration,
            keypoint_threshold=0.25,
            max_reprojection_error_px=1.0,
        )
        self.assertEqual(len(persons), 1)
        self.assertEqual(persons[0].valid_keypoints, 17)
        reconstructed = np.asarray(
            [item["xyz"] for item in persons[0].keypoints_3d], dtype=np.float64
        )
        self.assertLess(float(np.max(np.abs(reconstructed - xyz))), 1e-6)

    def test_load_and_scale(self):
        raw = {
            "camera_model": "pinhole",
            "length_unit": "millimeter",
            "left": {
                "image_size": [640, 480],
                "K": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                "D": [0, 0, 0, 0, 0],
            },
            "right": {
                "image_size": [640, 480],
                "K": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                "D": [0, 0, 0, 0, 0],
            },
            "stereo": {
                "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "T": [-400, 0, 0],
            },
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "calibration.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            calibration = StereoCalibration.load(path)
        scaled = calibration.for_runtime_sizes((1280, 960), (1280, 960))
        self.assertEqual(scaled.left_K[0, 0], 1000.0)
        self.assertEqual(scaled.left_K[0, 2], 640.0)
        self.assertEqual(scaled.baseline, 400.0)


if __name__ == "__main__":
    unittest.main()
