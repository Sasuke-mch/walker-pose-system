from __future__ import annotations

import unittest

from pose_app.stereo_visualizer import lower_limb_status_lines


class StereoVisualizerTests(unittest.TestCase):
    def test_live_status_overlay_labels_only_downstream_candidate_state(self) -> None:
        lines = lower_limb_status_lines(
            {
                "t1_trajectory": {
                    "frame_status": "accepted_single_person",
                    "direct_observed_joint_names": ["left_hip", "left_knee"],
                },
                "t2_kinematics": {
                    "left_knee_angle_deg": {"available": True, "value": 170.25},
                    "right_knee_angle_deg": {"available": False},
                },
                "t3_coordinates": {"status": "not_configured"},
                "t4_noncontact_candidates": {"newly_confirmed_candidates": [{"candidate_id": "x"}]},
            }
        )
        rendered = "\n".join(lines).lower()
        self.assertIn("direct lower-limb joints=2/6", rendered)
        self.assertIn("left knee=170.2 deg", rendered)
        self.assertIn("right knee=unavailable", rendered)
        self.assertIn("new non-contact candidates=1", rendered)
        self.assertIn("accepted contact events=0", rendered)
        self.assertNotIn("heel", rendered)
        self.assertNotIn("toe-off", rendered)


if __name__ == "__main__":
    unittest.main()
