import unittest
import numpy as np

from pose_app.local_plane_propagation import STATIC_BACKGROUND_DOMAIN
from pose_app.static_background_vo import StaticCorrespondences, relative_pose_record


class StaticBackgroundVOTests(unittest.TestCase):
    def input(self, *, ground_used=False):
        source = np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100], [50, 50, 50]], dtype=float)
        target = source + np.array([10, -20, 30], dtype=float)
        return {"from_frame_index": 10, "to_frame_index": 11, "feature_domain": STATIC_BACKGROUND_DOMAIN,
                "ground_region_used_for_motion": ground_used, "previous_xyz_left_camera_mm": source.tolist(), "current_xyz_left_camera_mm": target.tolist()}

    def test_ransac_recovers_translation_and_static_contract(self):
        record = relative_pose_record(StaticCorrespondences.from_mapping(self.input()), distance_threshold_mm=0.01,
                                      iterations=100, random_seed=1, minimum_inliers=4)
        self.assertEqual(record["status"], "accepted")
        self.assertTrue(np.allclose(record["R_to_from"], np.eye(3), atol=1e-8))
        self.assertTrue(np.allclose(record["q_to_from"], [10, -20, 30], atol=1e-8))
        self.assertFalse(record["ground_region_used_for_motion"])

    def test_ground_dependent_correspondences_are_rejected_before_ransac(self):
        with self.assertRaisesRegex(ValueError, "ground-fit region"):
            StaticCorrespondences.from_mapping(self.input(ground_used=True))


if __name__ == "__main__":
    unittest.main()
