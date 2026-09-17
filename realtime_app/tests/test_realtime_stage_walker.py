from __future__ import annotations

import unittest

import cv2
import numpy as np

from pose_app.realtime_stage_walker import (
    STAGE_ONE,
    STAGE_TWO,
    STAGE_TRANSITION,
    StableStructureView,
    StageCriteria,
    TemporalMotionWindow,
    TwoStageRecognizer,
    estimate_background_motion,
    fuse_background_motion,
    keypoint_motion_evidence,
)
from pose_app.schema import PersonPose


def person(upper_dx: float = 0.0, ankle_dx: float = 0.0) -> PersonPose:
    points = []
    for index in range(17):
        x = 90.0 + index * 3.0
        y = 50.0 + index * 4.0
        dx = ankle_dx if index in (15, 16) else upper_dx
        points.append([x + dx, y, 0.95])
    return PersonPose(0, [80.0, 35.0, 160.0, 130.0], 0.9, 0.9, points)


class RealtimeStageWalkerTests(unittest.TestCase):
    def test_temporal_window_detects_slow_consistent_camera_motion(self) -> None:
        criteria = StageCriteria(
            motion_window_frames=5,
            moving_window_displacement_px=1.5,
            moving_window_path_px=1.8,
        )
        window = TemporalMotionWindow(criteria)
        result = None
        for _ in range(5):
            result = window.update(
                {
                    "available": True,
                    "median_motion_px": 0.4,
                    "median_vector_px": [0.4, 0.0],
                }
            )
        assert result is not None
        self.assertTrue(result["motion_active"])
        self.assertFalse(result["static_confident"])
        self.assertAlmostEqual(result["window_displacement_px"], 2.0)

    def test_temporal_window_rejects_oscillating_jitter_as_motion(self) -> None:
        window = TemporalMotionWindow(StageCriteria(motion_window_frames=5))
        result = None
        for dx in (0.4, -0.4, 0.4, -0.4, 0.4):
            result = window.update(
                {"available": True, "median_motion_px": 0.4, "median_vector_px": [dx, 0.0]}
            )
        assert result is not None
        self.assertFalse(result["motion_active"])
    def test_background_fusion_does_not_hide_motion_seen_by_one_reliable_view(self) -> None:
        backgrounds = {
            "left": {
                "available": True, "median_motion_px": 0.32,
                "inlier_count": 80, "inlier_ratio": 0.8,
            },
            "right": {
                "available": True, "median_motion_px": 3.58,
                "inlier_count": 70, "inlier_ratio": 0.7,
            },
        }
        fused, selected = fuse_background_motion(backgrounds)
        self.assertEqual(selected, "right")
        self.assertAlmostEqual(fused["median_motion_px"], 3.58)
        self.assertEqual(fused["reliable_views"], ["left", "right"])

    def test_stage_one_is_confirmed_after_hysteresis(self) -> None:
        recognizer = TwoStageRecognizer(StageCriteria(stage1_confirmation_frames=2))
        background = {"available": True, "median_motion_px": 0.2}
        pose = {"upper_body_median_motion_px": 2.5, "ankle_count": 2}
        recognizer.update(background, pose)
        confirmed, candidate, streak = recognizer.update(background, pose)
        self.assertEqual(candidate, STAGE_ONE)
        self.assertEqual(confirmed, STAGE_ONE)
        self.assertEqual(streak, 2)

    def test_stage_two_requires_two_coherent_ankles(self) -> None:
        recognizer = TwoStageRecognizer(StageCriteria(stage2_confirmation_frames=1))
        background = {"available": True, "median_motion_px": 3.0}
        pose = {
            "upper_body_median_motion_px": 3.0,
            "ankle_count": 2,
            "ankle_median_motion_px": 3.0,
            "ankle_direction_cosine": 0.99,
            "ankle_separation_change_px": 0.1,
            "ankle_background_direction_cosine": 0.98,
            "ankle_background_residual_max_px": 0.7,
            "ankle_compensated_separation_error_px": 0.2,
        }
        confirmed, candidate, _ = recognizer.update(background, pose)
        self.assertEqual(candidate, STAGE_TWO)
        self.assertEqual(confirmed, STAGE_TWO)

    def test_pose_motion_evidence_reports_rigid_ankle_motion(self) -> None:
        evidence = keypoint_motion_evidence(
            person(), person(upper_dx=2.0, ankle_dx=3.0), 0.25, 1.0, np.asarray([3.0, 0.0])
        )
        self.assertEqual(evidence["ankle_count"], 2)
        self.assertAlmostEqual(evidence["ankle_direction_cosine"], 1.0, places=5)
        self.assertAlmostEqual(evidence["ankle_separation_change_px"], 0.0, places=5)
        self.assertAlmostEqual(evidence["ankle_background_direction_cosine"], 1.0, places=5)

    def test_camera_compensation_removes_shared_pose_translation(self) -> None:
        evidence = keypoint_motion_evidence(
            person(),
            person(upper_dx=3.0, ankle_dx=3.0),
            0.25,
            1.0,
            np.asarray([3.0, 0.0]),
            np.asarray([[1.0, 0.0, 3.0], [0.0, 1.0, 0.0]]),
        )
        self.assertAlmostEqual(evidence["upper_body_median_motion_px"], 3.0, places=5)
        self.assertAlmostEqual(evidence["upper_body_compensated_motion_px"], 0.0, places=5)
        self.assertAlmostEqual(evidence["ankle_background_residual_max_px"], 0.0, places=5)
        self.assertAlmostEqual(evidence["ankle_compensated_separation_error_px"], 0.0, places=5)

    def test_stage_two_rejects_feet_that_do_not_follow_background_prediction(self) -> None:
        recognizer = TwoStageRecognizer(StageCriteria(stage2_confirmation_frames=1))
        background = {"available": True, "median_motion_px": 2.0, "motion_active": True}
        pose = {
            "ankle_count": 2,
            "ankle_background_residual_max_px": 5.0,
            "ankle_compensated_separation_error_px": 0.2,
        }
        _, candidate, _ = recognizer.update(background, pose)
        self.assertNotEqual(candidate, STAGE_TWO)

    def test_uncertain_frame_is_displayed_as_transition_without_stale_stage_one(self) -> None:
        recognizer = TwoStageRecognizer(StageCriteria(stage1_confirmation_frames=1))
        stage_one_pose = {"upper_body_compensated_motion_px": 2.0, "ankle_count": 2}
        confirmed, _, _ = recognizer.update(
            {"available": True, "median_motion_px": 0.2, "static_confident": True},
            stage_one_pose,
        )
        self.assertEqual(confirmed, STAGE_ONE)
        confirmed, candidate, _ = recognizer.update(
            {"available": True, "median_motion_px": 0.9, "static_confident": False},
            stage_one_pose,
        )
        self.assertEqual(candidate, STAGE_TRANSITION)
        self.assertEqual(confirmed, STAGE_TRANSITION)

    def test_camera_motion_is_a_hard_exclusion_for_stage_one(self) -> None:
        recognizer = TwoStageRecognizer(StageCriteria(stage1_confirmation_frames=1))
        pose = {"upper_body_compensated_motion_px": 3.0, "ankle_count": 0}
        confirmed, candidate, _ = recognizer.update(
            {
                "available": True,
                "median_motion_px": 2.0,
                "motion_active": True,
                "static_confident": False,
            },
            pose,
        )
        self.assertNotEqual(candidate, STAGE_ONE)
        self.assertEqual(confirmed, STAGE_TRANSITION)

    def test_ready_window_without_phase_is_transition_not_long_warmup(self) -> None:
        recognizer = TwoStageRecognizer(StageCriteria())
        confirmed, _, _ = recognizer.update(
            {
                "available": True,
                "median_motion_px": 0.9,
                "motion_active": False,
                "static_confident": False,
                "window_available": True,
            },
            {},
        )
        self.assertEqual(confirmed, STAGE_TRANSITION)

    def test_background_translation_is_measured_with_person_excluded(self) -> None:
        rng = np.random.default_rng(7)
        previous = rng.integers(0, 255, size=(180, 240), dtype=np.uint8)
        current = cv2.warpAffine(previous, np.float32([[1, 0, 3], [0, 1, 1]]), (240, 180))
        result = estimate_background_motion(previous, current, None, None, 0.25, 1.0, 18)
        self.assertTrue(result["available"])
        self.assertGreater(result["median_motion_px"], 2.0)
        self.assertAlmostEqual(result["median_vector_px"][0], 3.0, delta=0.5)

    def test_background_motion_respects_extra_attached_structure_exclusion(self) -> None:
        rng = np.random.default_rng(11)
        previous = rng.integers(0, 255, size=(180, 240), dtype=np.uint8)
        current = cv2.warpAffine(previous, np.float32([[1, 0, 2], [0, 1, 0]]), (240, 180))
        exclusion = np.zeros_like(previous)
        exclusion[:, :50] = 255
        result = estimate_background_motion(
            previous, current, None, None, 0.25, 1.0, 18, exclusion
        )
        self.assertTrue(result["available"])
        self.assertAlmostEqual(result["median_vector_px"][0], 2.0, delta=0.5)

    def test_structure_accumulator_updates_only_when_background_moves(self) -> None:
        view = StableStructureView((160, 240))
        gray = np.zeros((160, 240), dtype=np.uint8)
        cv2.line(gray, (10, 10), (10, 145), 255, 3)
        valid = np.full_like(gray, 255)
        for _ in range(5):
            mask = view.update(gray, valid, allow_update=False)
        self.assertEqual(int(np.count_nonzero(mask)), 0)
        for _ in range(5):
            mask = view.update(gray, valid, allow_update=True)
        self.assertGreater(int(np.count_nonzero(mask)), 50)


if __name__ == "__main__":
    unittest.main()
