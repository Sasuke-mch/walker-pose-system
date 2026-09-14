import math
import unittest
import numpy as np

from pose_app.local_plane_propagation import (
    DIRECT, PROPAGATED, UNAVAILABLE, LocalPlane, RelativePose, STATIC_BACKGROUND_DOMAIN,
    parse_relative_pose, plane_agreement, propagate_plane, sequential_propagation,
)


class LocalPlanePropagationTests(unittest.TestCase):
    def pose(self, source: int, target: int, q=(0.0, 0.0, 0.0)):
        return RelativePose(source, target, np.eye(3), np.asarray(q), 10, 8)

    def test_translation_updates_offset_in_target_camera_coordinates(self):
        plane = LocalPlane([0.0, 1.0, 0.0], 1000.0)
        result = propagate_plane(plane, self.pose(0, 1, (0.0, 100.0, 0.0)))
        self.assertTrue(np.allclose(result.normal, [0.0, 1.0, 0.0]))
        self.assertAlmostEqual(result.offset, 900.0)

    def test_rotation_updates_normal(self):
        angle = math.pi / 2.0
        rotation = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(angle), -math.sin(angle)], [0.0, math.sin(angle), math.cos(angle)]])
        plane = LocalPlane([0.0, 1.0, 0.0], 1000.0)
        result = propagate_plane(plane, RelativePose(0, 1, rotation, np.zeros(3), 12, 9))
        self.assertTrue(np.allclose(result.normal, [0.0, 0.0, 1.0], atol=1e-8))

    def test_sequential_state_machine_fails_closed(self):
        direct = [
            {"frame_index": 0, "observation_state": DIRECT, "plane_in_left_camera": {"normal_toward_camera_unit": [0, 1, 0], "offset_mm": 1000}},
            {"frame_index": 1, "observation_state": UNAVAILABLE},
            {"frame_index": 2, "observation_state": UNAVAILABLE},
        ]
        rows = sequential_propagation(direct, {0: self.pose(0, 1)}, max_propagation_frames=1)
        self.assertEqual([row["observation_state"] for row in rows], [DIRECT, PROPAGATED, UNAVAILABLE])
        self.assertEqual(rows[2]["reason"], "maximum_propagation_age_exceeded")

    def test_pose_contract_rejects_ground_dependent_motion(self):
        raw = {"from_frame_index": 0, "to_frame_index": 1, "status": "accepted", "R_to_from": np.eye(3).tolist(), "q_to_from": [0, 0, 0],
               "feature_domain": STATIC_BACKGROUND_DOMAIN, "static_3d_correspondence_count": 9, "ransac_inlier_count": 7, "ground_region_used_for_motion": True}
        self.assertIsNone(parse_relative_pose(raw))

    def test_agreement_uses_canonical_normal_direction(self):
        result = plane_agreement(LocalPlane([0, 1, 0], 1000), LocalPlane([0, -1, 0], -1000))
        self.assertAlmostEqual(result["normal_angle_deg"], 0.0)
        self.assertAlmostEqual(result["camera_plane_distance_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
