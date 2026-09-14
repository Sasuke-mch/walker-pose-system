from __future__ import annotations

import unittest

import cv2
import numpy as np

from pose_app.static_ground_reference import (
    StaticGroundReference,
    average_rotations,
    build_reference,
    estimate_board_pose,
    make_charuco_board,
    right_pose_in_left_frame,
)
from tools.offline_visualize_skeleton_on_ground import COCO_EDGES, complete_display_records, transform_row


class StaticGroundReferenceTests(unittest.TestCase):
    def test_board_thickness_moves_floor_below_detected_surface(self) -> None:
        rotation = np.eye(3)
        translation = np.asarray([100.0, 200.0, 1000.0])
        result = build_reference(
            [(rotation, translation), (rotation, translation)],
            source_capture_session="synthetic",
            per_pair_results=[],
            board_spec={
                "dictionary": "DICT_4X4_50", "squares_x": 8, "squares_y": 6,
                "square_mm": 30.0, "marker_mm": 22.0,
            },
            board_thickness_mm=12.0,
        )
        self.assertAlmostEqual(result["detected_board_top_plane_left_camera"]["offset_mm"], 1000.0)
        self.assertAlmostEqual(result["plane_left_camera"]["camera_height_mm"], 1012.0)
        self.assertAlmostEqual(result["repeatability"]["normal_angle_p95_deg"], 0.0)
        reference = StaticGroundReference.from_mapping(result)
        board_top_ground = reference.transform(translation)
        floor_origin_left = translation - 12.0 * reference.plane_normal_left
        self.assertTrue(np.allclose(board_top_ground, [0.0, 0.0, 12.0]))
        self.assertTrue(np.allclose(reference.transform(floor_origin_left), [0.0, 0.0, 0.0]))

    def test_right_pose_is_converted_back_to_left(self) -> None:
        angle = np.deg2rad(12.0)
        rotation_lr = np.asarray([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        translation_lr = np.asarray([250.0, -20.0, 10.0])
        rotation_lb = np.eye(3)
        translation_lb = np.asarray([80.0, 30.0, 900.0])
        rotation_rb = rotation_lr @ rotation_lb
        translation_rb = rotation_lr @ translation_lb + translation_lr
        recovered_rotation, recovered_translation = right_pose_in_left_frame(
            rotation_rb, translation_rb, rotation_lr, translation_lr
        )
        self.assertTrue(np.allclose(recovered_rotation, rotation_lb, atol=1e-10))
        self.assertTrue(np.allclose(recovered_translation, translation_lb, atol=1e-10))

    def test_rotation_average_remains_proper(self) -> None:
        averaged = average_rotations([np.eye(3), np.eye(3)])
        self.assertTrue(np.allclose(averaged.T @ averaged, np.eye(3)))
        self.assertAlmostEqual(float(np.linalg.det(averaged)), 1.0)

    def test_fisheye_pose_recovery_from_synthetic_charuco_points(self) -> None:
        spec = {
            "dictionary": "DICT_4X4_50", "squares_x": 8, "squares_y": 6,
            "square_mm": 30.0, "marker_mm": 22.0,
        }
        board = make_charuco_board(spec)
        object_points = np.asarray(board.getChessboardCorners(), dtype=np.float64)
        ids = np.arange(len(object_points), dtype=np.int32)
        K = np.asarray([[755.0, 0.0, 996.0], [0.0, 755.0, 537.0], [0.0, 0.0, 1.0]])
        D = np.asarray([-0.016, -0.010, 0.005, -0.001])
        rvec = np.asarray([[0.10], [-0.08], [0.03]])
        tvec = np.asarray([[40.0], [-60.0], [950.0]])
        pixels, _ = cv2.fisheye.projectPoints(object_points.reshape(-1, 1, 3), rvec, tvec, K, D)
        result = estimate_board_pose(pixels.reshape(-1, 2), ids, board, K, D, "fisheye")
        self.assertLess(result["reprojection_rmse_px"], 1e-4)
        self.assertTrue(np.allclose(result["translation_camera_from_board_mm"], tvec.reshape(3), atol=1e-3))

    def test_skeleton_point_is_expressed_above_fixed_floor(self) -> None:
        mapping = build_reference(
            [(np.eye(3), np.asarray([0.0, 0.0, 1000.0]))],
            source_capture_session="synthetic",
            per_pair_results=[],
            board_spec={
                "dictionary": "DICT_4X4_50", "squares_x": 8, "squares_y": 6,
                "square_mm": 30.0, "marker_mm": 22.0,
            },
            board_thickness_mm=12.0,
        )
        reference = StaticGroundReference.from_mapping(mapping)
        row = {
            "pair_id": 1,
            "persons_3d": [{"keypoints_3d": [{
                "index": 15, "name": "left_ankle", "valid": True,
                "xyz": [0.0, 0.0, 1000.0], "score": 0.9,
                "reprojection_error_mean_px": 1.0,
            }]}],
        }
        transformed = transform_row(row, reference)
        self.assertEqual(transformed["status"], "rendered")
        self.assertAlmostEqual(transformed["points"][0]["height_above_ground_mm"], 12.0)

    def test_rejected_reference_cannot_be_loaded(self) -> None:
        value = build_reference(
            [(np.eye(3), np.asarray([0.0, 0.0, 1000.0]))],
            source_capture_session="synthetic",
            per_pair_results=[],
            board_spec={
                "dictionary": "DICT_4X4_50", "squares_x": 8, "squares_y": 6,
                "square_mm": 30.0, "marker_mm": 22.0,
            },
            board_thickness_mm=12.0,
        )
        value["status"] = "rejected"
        with self.assertRaisesRegex(ValueError, "measured_static_reference"):
            StaticGroundReference.from_mapping(value)

    def test_coco_topology_connects_head_to_shoulders(self) -> None:
        self.assertIn((3, 5), COCO_EDGES)
        self.assertIn((4, 6), COCO_EDGES)

    def test_display_completion_produces_all_seventeen_points(self) -> None:
        mapping = build_reference(
            [(np.eye(3), np.asarray([0.0, 0.0, 1000.0]))],
            source_capture_session="synthetic",
            per_pair_results=[],
            board_spec={
                "dictionary": "DICT_4X4_50", "squares_x": 8, "squares_y": 6,
                "square_mm": 30.0, "marker_mm": 22.0,
            },
            board_thickness_mm=12.0,
        )
        reference = StaticGroundReference.from_mapping(mapping)

        def row(pair_id: int, include_points: bool) -> dict:
            points = []
            if include_points:
                points = [
                    {
                        "index": index, "name": str(index), "valid": True,
                        "xyz": [float(index), 0.0, 1000.0],
                        "score": 1.0, "reprojection_error_mean_px": 0.0,
                    }
                    for index in range(17)
                ]
            return {
                "pair_id": pair_id,
                "pair_timestamp_sec": pair_id / 30.0,
                "persons_3d": [{"keypoints_3d": points}],
            }

        strict = [row(0, True), row(1, False), row(2, True)]
        force_all = [row(0, True), row(1, False), row(2, True)]
        completed = complete_display_records(strict, force_all, reference)
        self.assertTrue(all(len(record["points"]) == 17 for record in completed))
        self.assertTrue(all(
            point["provenance"] == "linear_temporal_interpolation"
            for point in completed[1]["points"]
        ))


if __name__ == "__main__":
    unittest.main()
