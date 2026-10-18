import unittest

import cv2
import numpy as np

from pose_app.fisheye_camera import fisheye_project_numpy, left_to_right_numpy, world_to_left_numpy
from pose_app.joint_offset_model import JointOffsetModel, body_basis


class ReprojectionObservationValidityTests(unittest.TestCase):
    def test_numpy_fisheye_projection_matches_opencv(self):
        points = np.asarray([[100.0, -50.0, 900.0], [-250.0, 120.0, 1300.0], [0.0, 0.0, 800.0]])
        K = np.asarray([[520.0, 0.0, 960.0], [0.0, 515.0, 540.0], [0.0, 0.0, 1.0]])
        D = np.asarray([-0.05, 0.008, -0.001, 0.0002])
        expected, _ = cv2.fisheye.projectPoints(
            points.reshape(1, -1, 3), np.zeros(3), np.zeros(3), K, D
        )
        actual = fisheye_project_numpy(points, K, D)
        np.testing.assert_allclose(actual, expected.reshape(-1, 2), atol=1e-9)

    def test_world_and_right_camera_transform_contract(self):
        angle = np.deg2rad(20.0)
        rotation = np.asarray(
            [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
        )
        translation = np.asarray([100.0, 200.0, 300.0])
        point_left = np.asarray([[50.0, -20.0, 900.0]])
        point_world = (rotation @ point_left.T).T + translation
        np.testing.assert_allclose(
            world_to_left_numpy(point_world, rotation, translation), point_left, atol=1e-9
        )

        class Calibration:
            R_cam0_to_cam1 = np.eye(3)
            T_cam0_to_cam1_mm = np.asarray([80.0, 0.0, 0.0])

        np.testing.assert_allclose(
            left_to_right_numpy(point_left, Calibration()), point_left + [80.0, 0.0, 0.0]
        )

    def test_joint_offset_rotates_with_predicted_body(self):
        joints = np.zeros((17, 3), dtype=np.float64)
        joints[5] = [1.0, 2.0, 0.0]
        joints[6] = [-1.0, 2.0, 0.0]
        joints[11] = [0.8, 0.0, 0.0]
        joints[12] = [-0.8, 0.0, 0.0]
        joints[9] = [1.2, 1.0, 0.3]
        offset = np.asarray([20.0, -10.0, 5.0])
        model = JointOffsetModel({9: offset})
        expected = joints[9] - body_basis(joints) @ offset
        np.testing.assert_allclose(model.predict_coco_numpy(joints)[9], expected, atol=1e-12)

        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rotated = (rotation @ joints.T).T
        np.testing.assert_allclose(
            model.predict_coco_numpy(rotated)[9], rotation @ expected, atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
