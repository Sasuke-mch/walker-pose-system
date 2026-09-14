import unittest

import numpy as np

from pose_app.gait_interaction_candidates import (
    classify_walking_window,
    foot_ground_candidate,
    hand_handle_proximity_candidate,
    point_segment_distance_mm,
)


class GaitInteractionCandidateTests(unittest.TestCase):
    def test_walking_uses_relative_alternating_ankles(self):
        count = 20
        phase = np.linspace(0.0, 3.0 * np.pi, count)
        left = np.column_stack((80.0 * np.sin(phase), np.zeros(count), np.full(count, 900.0)))
        right = np.column_stack((-80.0 * np.sin(phase), np.zeros(count), np.full(count, 900.0)))
        result = classify_walking_window(np.arange(count) / 30.0, left, right, np.ones(count, dtype=bool))
        self.assertEqual(result["status"], "candidate")
        self.assertEqual(result["state"], "walking_candidate")
        self.assertGreater(result["metrics"]["interankle_range_mm"], 100.0)

    def test_invalid_window_is_unavailable(self):
        points = np.zeros((3, 3))
        result = classify_walking_window(np.arange(3), points, points, np.ones(3, dtype=bool))
        self.assertEqual(result["status"], "unavailable")

    def test_foot_proxy_requires_direct_ground(self):
        unavailable = foot_ground_candidate([0, -1000, 0], [0, 0, 0], [0, 1, 0], 1000, ground_status="propagated")
        self.assertEqual(unavailable["status"], "unavailable")
        candidate = foot_ground_candidate([0, -1000, 0], [0, 0, 0], [0, 1, 0], 1000, ground_status="direct")
        self.assertEqual(candidate["state"], "near_static_ankle_candidate")

    def test_hand_handle_is_soft_proximity_not_contact(self):
        distance, _, fraction = point_segment_distance_mm([50, 20, 0], [0, 0, 0], [100, 0, 0])
        self.assertAlmostEqual(distance, 20.0)
        self.assertAlmostEqual(fraction, 0.5)
        result = hand_handle_proximity_candidate(
            [50, 20, 0], [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0],
            handle_radius_mm=10.0, walker_geometry_status="candidate",
        )
        self.assertGreater(result["proximity_score_0_1"], 0.5)
        self.assertIn("not tactile contact", result["interpretation"])


if __name__ == "__main__":
    unittest.main()
