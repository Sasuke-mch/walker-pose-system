from pathlib import Path
import unittest

from pose_app.hand_pose_provider import (
    HandObservation,
    build_interwild_command,
    build_wilor_command,
    local_assets,
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
        self.assertEqual(observation.coordinate_frame, "model_local_unaccepted")
        self.assertIsNone(observation.keypoints_3d_local)


if __name__ == "__main__":
    unittest.main()
