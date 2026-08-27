from __future__ import annotations

import logging
import unittest

import numpy as np

from pose_app.local_perspective import LocalPerspectiveModelInput
from pose_app.schema import InferenceResult, PersonPose
from run_stereo import _attempt_local_refinement


class CenterPoseClient:
    """Deterministic local-view model response for coordinate-chain testing."""

    def infer(self, image, frame_id, timestamp, dropped_before=0):
        height, width = image.shape[:2]
        cx, cy = (width - 1.0) * 0.5, (height - 1.0) * 0.5
        keypoints = [[cx + (index % 3 - 1) * 5.0, cy + (index // 3 - 2) * 5.0, 0.9] for index in range(17)]
        return InferenceResult(
            source_frame_id=frame_id,
            source_timestamp_sec=timestamp,
            image_width=width,
            image_height=height,
            model_name="fake",
            model_ms=3.0,
            roundtrip_ms=4.0,
            persons=[PersonPose(99, [cx - 100, cy - 200, cx + 100, cy + 200], 0.9, 0.9, keypoints)],
            dropped_before=dropped_before,
        )


class LocalRefinementPipelineTests(unittest.TestCase):
    def test_rotated_local_prediction_returns_to_raw_fisheye_pixels(self) -> None:
        size = (1280, 720)
        K = np.asarray([[520.0, 0.0, 639.5], [0.0, 520.0, 359.5], [0.0, 0.0, 1.0]])
        preprocessor = LocalPerspectiveModelInput(K, np.zeros((4, 1)), size)
        base_person = PersonPose(
            7,
            [360.0, 180.0, 760.0, 620.0],
            0.8,
            0.8,
            [[560.0, 400.0, 0.8] for _ in range(17)],
        )
        base = InferenceResult(11, 1.25, *size, "base", 10.0, 11.0, [base_person])
        raw = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        for rotation in ("none", "cw90", "ccw90"):
            with self.subTest(rotation=rotation):
                refined = _attempt_local_refinement(
                    client=CenterPoseClient(),
                    raw_image=raw,
                    base_result=base,
                    preprocessor=preprocessor,
                    rotation=rotation,
                    mode="always",
                    min_box_fraction=0.35,
                    keypoint_threshold=0.25,
                    log=logging.getLogger("test"),
                )
                self.assertIsNotNone(refined)
                assert refined is not None
                self.assertEqual(refined.persons[0].person_id, 7)
                mapped_center = np.asarray(refined.persons[0].keypoints[0][:2])
                self.assertLess(np.linalg.norm(mapped_center - np.asarray([560.0, 400.0])), 20.0)
                self.assertTrue(np.all(np.isfinite(np.asarray(refined.persons[0].keypoints))))
                self.assertEqual(refined.stage_times_ms["local_perspective_pose_ms"], 3.0)


if __name__ == "__main__":
    unittest.main()
