from __future__ import annotations

import unittest

from tools.export_interactive_walker_viewer import make_frame


class InteractiveWalkerViewerExportTests(unittest.TestCase):
    def test_frame_export_preserves_unsmoothed_points_pose_and_quality(self) -> None:
        dynamic = {
            "pair_id": 7,
            "stage": "stage2_feet_static_walker_moving",
            "rotation_world_from_left_camera": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation_world_from_left_camera_mm": [10, 20, 30],
        }
        strict = {
            "pair_id": 7,
            "persons_3d": [{"keypoints_3d": [{"index": 5, "valid": True, "xyz": [1, 2, 3]}]}],
        }
        visual = {
            "pair_id": 7,
            "persons_3d": [{"keypoints_3d": [{"index": 5, "valid": False, "xyz": [9, 9, 9]}]}],
        }
        interaction = {
            "pair_id": 7,
            "per_keypoint_reprojection_error_mean_px": {"5": 2.345},
            "per_keypoint_provenance": {"5": {"strict_valid": True}},
            "hands": {"left": {"wrist_to_handle_axis_distance_mm": 82.25}, "right": {}},
            "walker_ground": {"maximum_absolute_foot_height_mm": 12.34},
        }
        frame = make_frame(dynamic, strict, visual, interaction)
        self.assertEqual(frame["points_mm"][5], [11.0, 22.0, 33.0])
        self.assertEqual(frame["errors_px"][5], 2.35)
        self.assertTrue(frame["strict"][5])
        self.assertEqual(frame["wrist_handle_mm"]["left"], 82.2)
        self.assertEqual(frame["walker_feet_max_abs_z_mm"], 12.3)


if __name__ == "__main__":
    unittest.main()
