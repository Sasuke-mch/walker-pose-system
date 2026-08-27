from __future__ import annotations

import unittest

import numpy as np

from pose_app.rotation import (
    ROTATION_CHOICES,
    model_image_size,
    model_to_raw_point,
    raw_to_model_point,
    restore_model_result_to_raw,
    rotate_image_for_model,
)
from pose_app.schema import InferenceResult, PersonPose


class RotationNormalizationTests(unittest.TestCase):
    raw_width = 1920
    raw_height = 1080

    def test_point_round_trip_and_image_size_for_all_rotations(self):
        image = np.zeros((self.raw_height, self.raw_width, 3), dtype=np.uint8)
        points = [(0.0, 0.0), (100.25, 200.75), (1919.0, 1079.0)]
        for rotation in ROTATION_CHOICES:
            with self.subTest(rotation=rotation):
                model_width, model_height = model_image_size(
                    self.raw_width, self.raw_height, rotation
                )
                rotated = rotate_image_for_model(image, rotation)
                self.assertEqual(rotated.shape[:2], (model_height, model_width))
                for raw_x, raw_y in points:
                    model_x, model_y = raw_to_model_point(
                        raw_x,
                        raw_y,
                        self.raw_width,
                        self.raw_height,
                        rotation,
                    )
                    restored_x, restored_y = model_to_raw_point(
                        model_x,
                        model_y,
                        self.raw_width,
                        self.raw_height,
                        rotation,
                    )
                    self.assertAlmostEqual(restored_x, raw_x)
                    self.assertAlmostEqual(restored_y, raw_y)

    def test_pixel_rotation_matches_coordinate_mapping(self):
        raw_width, raw_height = 5, 3
        image = np.arange(raw_width * raw_height, dtype=np.uint8).reshape(
            raw_height, raw_width
        )
        for rotation in ROTATION_CHOICES:
            with self.subTest(rotation=rotation):
                rotated = rotate_image_for_model(image, rotation)
                for raw_y in range(raw_height):
                    for raw_x in range(raw_width):
                        model_x, model_y = raw_to_model_point(
                            raw_x, raw_y, raw_width, raw_height, rotation
                        )
                        self.assertEqual(
                            rotated[int(model_y), int(model_x)], image[raw_y, raw_x]
                        )

    def test_restores_bbox_keypoints_and_metadata_for_all_rotations(self):
        raw_bbox = [100.0, 200.0, 400.0, 700.0]
        raw_keypoints = [[120.5, 230.25, 0.9], [350.75, 650.5, 0.8]]
        for rotation in ROTATION_CHOICES:
            with self.subTest(rotation=rotation):
                model_width, model_height = model_image_size(
                    self.raw_width, self.raw_height, rotation
                )
                model_corners = [
                    raw_to_model_point(
                        x, y, self.raw_width, self.raw_height, rotation
                    )
                    for x, y in (
                        (raw_bbox[0], raw_bbox[1]),
                        (raw_bbox[0], raw_bbox[3]),
                        (raw_bbox[2], raw_bbox[1]),
                        (raw_bbox[2], raw_bbox[3]),
                    )
                ]
                model_bbox = [
                    min(point[0] for point in model_corners),
                    min(point[1] for point in model_corners),
                    max(point[0] for point in model_corners),
                    max(point[1] for point in model_corners),
                ]
                model_keypoints = [
                    [
                        *raw_to_model_point(
                            x, y, self.raw_width, self.raw_height, rotation
                        ),
                        score,
                    ]
                    for x, y, score in raw_keypoints
                ]
                result = InferenceResult(
                    source_frame_id=7,
                    source_timestamp_sec=1.25,
                    image_width=model_width,
                    image_height=model_height,
                    model_name="test-pose",
                    model_ms=12.0,
                    roundtrip_ms=15.0,
                    persons=[PersonPose(3, model_bbox, 0.95, 0.9, model_keypoints)],
                    dropped_before=2,
                    stage_times_ms={"pose_ms": 12.0},
                )
                restored = restore_model_result_to_raw(
                    result,
                    raw_width=self.raw_width,
                    raw_height=self.raw_height,
                    rotation=rotation,
                )

                self.assertEqual(
                    (restored.image_width, restored.image_height), (1920, 1080)
                )
                self.assertEqual(restored.source_frame_id, 7)
                self.assertEqual(restored.dropped_before, 2)
                self.assertEqual(restored.stage_times_ms, {"pose_ms": 12.0})
                for actual, expected in zip(restored.persons[0].bbox, raw_bbox):
                    self.assertAlmostEqual(actual, expected)
                for actual, expected in zip(
                    restored.persons[0].keypoints, raw_keypoints
                ):
                    for actual_value, expected_value in zip(actual, expected):
                        self.assertAlmostEqual(actual_value, expected_value)

    def test_rejects_unexpected_model_output_size(self):
        result = InferenceResult(
            source_frame_id=0,
            source_timestamp_sec=0.0,
            image_width=1920,
            image_height=1080,
            model_name="test-pose",
            model_ms=0.0,
            roundtrip_ms=0.0,
            persons=[],
        )
        with self.assertRaisesRegex(RuntimeError, "inconsistent"):
            restore_model_result_to_raw(
                result,
                raw_width=self.raw_width,
                raw_height=self.raw_height,
                rotation="cw90",
            )


if __name__ == "__main__":
    unittest.main()
