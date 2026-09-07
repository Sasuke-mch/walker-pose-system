from __future__ import annotations

from pathlib import Path
import unittest

from pose_app.config import AppConfig, CameraConfig, DetectorConfig, DockerConfig, ModelConfig, OutputConfig, PMPoseConfig
from pose_app.pipeline_preflight import configured_model_checks, preflight_status, recommended_live_command


def config(root: Path, *, detector_port: int = 18081, pmpose_port: int = 18082) -> AppConfig:
    return AppConfig(
        app_root=root,
        project_root=root,
        docker=DockerConfig("image", 18080, 18080, 1.0, 1.0, "1g", False),
        model=ModelConfig(root, root / "pose.pt", "0", 1, 0.1, 0.1, 1, 0.1, 0.1),
        detector=DetectorConfig("image", root, root / "detector.pt", detector_port, 1, "0", 1, 0.1, 0.1, 1),
        pmpose=PMPoseConfig("image", root, root, pmpose_port, 1, "cpu", "PMPose-b", 0.1, 0.1, "bbox"),
        output=OutputConfig(root, 90, 10.0, 0.2),
        camera=CameraConfig(1920, 1080, 30, "msmf"),
    )


class PipelinePreflightTests(unittest.TestCase):
    def test_detects_missing_model_files_and_duplicate_service_ports(self) -> None:
        root = Path(__file__).resolve().parent
        checks = configured_model_checks(config(root, detector_port=18082, pmpose_port=18082), "pmpose")
        self.assertEqual(preflight_status(checks), "software_blocked")
        self.assertTrue(any(check.name == "pmpose_service_ports" and check.status == "error" for check in checks))
        self.assertTrue(any(check.name == "detector_weight" and check.status == "error" for check in checks))

    def test_recommended_command_keeps_single_subject_and_rotations(self) -> None:
        command = recommended_live_command(
            config_path=Path("config.json"),
            calibration_path=Path("calibration.json"),
            registry_path=Path("registry.json"),
            model="pmpose",
            coordinate_transform_path=None,
            camera_backend="msmf",
        )
        self.assertIn("--enable-lower-limb-pipeline", command)
        self.assertEqual(command[command.index("--left-model-rotation") + 1], "ccw90")
        self.assertEqual(command[command.index("--right-model-rotation") + 1], "cw90")
        self.assertEqual(command[command.index("--stereo-subject-mode") + 1], "single")
        self.assertEqual(command[command.index("--camera-backend") + 1], "msmf")
        self.assertNotIn("--no-json", command)


if __name__ == "__main__":
    unittest.main()
