from pathlib import Path
import unittest

import numpy as np

from pose_app.hand_pose_provider import (
    HandObservation,
    MODEL_LOCAL_3D_FRAME,
    build_interwild_command,
    build_wilor_command,
    local_assets,
    make_wilor_observation,
    project_wilor_vertices_to_image,
    validate_raw_fisheye_pixels,
)


class HandPoseProviderTests(unittest.TestCase):
    def test_asset_paths_are_inside_project(self) -> None:
        root = Path(__file__).resolve().parents[2]
        for assets in local_assets().values():
            self.assertTrue(str(assets.source_root).startswith(str(root)))
            self.assertTrue(str(assets.checkpoint).startswith(str(root)))

    def test_commands_use_backend_working_directories(self) -> None:
        wilor, wilor_cwd = build_wilor_command("python", "frames", "out")
        interwild, interwild_cwd = build_interwild_command("python", gpu="0")
        self.assertEqual(wilor[1], "demo.py")
        self.assertEqual(wilor_cwd.name, "WiLoR")
        self.assertEqual(interwild, ["python", "demo.py", "--gpu", "0"])
        self.assertEqual(interwild_cwd.name, "demo")

    def test_observation_defaults_to_unaccepted_local_coordinates(self) -> None:
        observation = HandObservation(backend="wilor", side="left")
        self.assertEqual(observation.coordinate_frame, MODEL_LOCAL_3D_FRAME)
        self.assertIsNone(observation.keypoints_3d_local)

    def test_wilor_projection_uses_original_image_size(self) -> None:
        points = project_wilor_vertices_to_image(
            [[0.0, 0.0, 1.0], [0.1, -0.1, 1.0]],
            [0.0, 0.0, 1.0],
            100.0,
            (640, 480),
        )
        np.testing.assert_allclose(points, [[320.0, 240.0], [325.0, 235.0]])
        validate_raw_fisheye_pixels(points, (640, 480))

    def test_raw_pixel_validation_does_not_clip_crop_errors(self) -> None:
        with self.assertRaises(ValueError):
            validate_raw_fisheye_pixels([[640.0, 10.0]], (640, 480))

    def test_wilor_observation_keeps_pixels_raw_and_3d_unaccepted(self) -> None:
        observation = make_wilor_observation(
            side="right",
            bbox_xyxy=[10.0, 20.0, 100.0, 120.0],
            keypoints_2d=[[30.0, 40.0], [50.0, 60.0]],
            image_size=(640, 480),
            keypoints_3d_local=[[0.0, 0.0, 1.0]],
        )
        self.assertEqual(observation.metadata["pixel_frame"], "raw_fisheye")
        self.assertEqual(observation.coordinate_frame, MODEL_LOCAL_3D_FRAME)


if __name__ == "__main__":
    unittest.main()
