from pathlib import Path
import sys
import unittest

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_app.calibration import StereoCalibration
from pose_app.geometry_input import MISSING_2D_POINT
from tools.estimate_complete_stereo_3d import (
    JointFrameInput,
    _triangulate_unfiltered,
    estimate_joint_series,
)


class CompleteStereoEstimationTests(unittest.TestCase):
    def calibration(self) -> StereoCalibration:
        matrix = np.asarray(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return StereoCalibration(
            camera_model="pinhole",
            left_image_size=(640, 480),
            right_image_size=(640, 480),
            left_K=matrix,
            left_D=np.zeros(5, dtype=np.float64),
            right_K=matrix.copy(),
            right_D=np.zeros(5, dtype=np.float64),
            R=np.eye(3, dtype=np.float64),
            T=np.asarray([-200.0, 0.0, 0.0], dtype=np.float64),
            length_unit="millimeter",
        )

    def direct(self, calibration: StereoCalibration, xyz: np.ndarray, *, right_y_shift: float = 0.0):
        left = calibration.project_left(xyz.reshape(1, 3))[0]
        right = calibration.project_right(xyz.reshape(1, 3))[0]
        right[1] += right_y_shift
        return _triangulate_unfiltered(
            calibration,
            [float(left[0]), float(left[1]), 0.1],
            [float(right[0]), float(right[1]), 0.1],
        )

    def test_high_reprojection_stereo_pair_remains_stereo_raw(self):
        calibration = self.calibration()
        candidate = self.direct(
            calibration, np.asarray([0.0, 0.0, 2000.0]), right_y_shift=40.0
        )
        self.assertEqual(candidate.state, "stereo_raw")
        self.assertGreater(candidate.reprojection_mean_px, 10.0)
        sample = JointFrameInput(
            pair_id=0,
            file_name="pair_0000.png",
            left_point=[320.0, 240.0, 0.1],
            right_point=[270.0, 280.0, 0.1],
            left_reason=None,
            right_reason=None,
            direct=candidate,
        )
        estimate = estimate_joint_series([sample], calibration)[0]
        self.assertEqual(estimate["estimate_source"], "stereo_raw")
        self.assertTrue(estimate["has_estimate"])

    def test_single_view_uses_interpolated_direct_anchor(self):
        calibration = self.calibration()
        first_xyz = np.asarray([0.0, 0.0, 2000.0])
        last_xyz = np.asarray([100.0, 0.0, 2000.0])
        first = self.direct(calibration, first_xyz)
        last = self.direct(calibration, last_xyz)
        middle_true = np.asarray([50.0, 0.0, 2000.0])
        middle_left = calibration.project_left(middle_true.reshape(1, 3))[0]
        samples = [
            JointFrameInput(0, "pair_0000.png", [320.0, 240.0, 0.1], [270.0, 240.0, 0.1], None, None, first),
            JointFrameInput(1, "pair_0001.png", [float(middle_left[0]), float(middle_left[1]), 0.1], None, None, MISSING_2D_POINT, None),
            JointFrameInput(2, "pair_0002.png", [345.0, 240.0, 0.1], [295.0, 240.0, 0.1], None, None, last),
        ]
        estimate = estimate_joint_series(samples, calibration)[1]
        self.assertEqual(estimate["estimate_source"], "single_view_temporal_interpolated")
        self.assertTrue(estimate["has_estimate"])
        reprojection = calibration.project_left(np.asarray(estimate["xyz"]).reshape(1, 3))[0]
        self.assertLess(float(np.linalg.norm(reprojection - middle_left)), 1e-6)

    def test_missing_without_direct_anchor_stays_unavailable(self):
        calibration = self.calibration()
        sample = JointFrameInput(
            pair_id=0,
            file_name="pair_0000.png",
            left_point=[320.0, 240.0, 0.1],
            right_point=None,
            left_reason=None,
            right_reason=MISSING_2D_POINT,
            direct=None,
        )
        estimate = estimate_joint_series([sample], calibration)[0]
        self.assertEqual(estimate["estimate_source"], "unavailable_no_stereo_anchor")
        self.assertFalse(estimate["has_estimate"])


if __name__ == "__main__":
    unittest.main()
