from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace

from pose_app.camera_registry import CameraRegistryError, resolve_stereo_cameras


def _registry_payload() -> dict:
    return {
        "cam0": {
            "calibration": "calibration/results/cam0_fisheye.json",
            "role": "left",
            "device": {
                "friendly_name": "HD USB Camera",
                "instance_id": "USB\\VID_05A3&PID_9230&MI_00\\6&3405B72D&0&0000",
                "usb_location": "USB(3)",
                "acpi_port": "HS03",
            },
        },
        "cam1": {
            "calibration": "calibration/results/cam1_fisheye.json",
            "role": "right",
            "device": {
                "friendly_name": "HD USB Camera",
                "instance_id": "USB\\VID_05A3&PID_9230&MI_00\\6&214C0688&0&0000",
                "usb_location": "USB(2)",
                "acpi_port": "HS02",
            },
        },
    }


class CameraRegistryTests(unittest.TestCase):
    def _write_registry(self, directory: Path) -> Path:
        path = directory / "camera_registry.json"
        path.write_text(json.dumps(_registry_payload()), encoding="utf-8")
        return path

    def test_resolves_identical_friendly_names_by_full_instance_id(self) -> None:
        with TemporaryDirectory() as temp:
            registry = self._write_registry(Path(temp))
            devices = [
                SimpleNamespace(
                    index=4,
                    name="HD USB Camera",
                    path=(
                        r"@device:pnp:\\?\usb#vid_05a3&pid_9230&mi_00"
                        r"#6&214c0688&0&0000#{6994ad05-93ef-11d0-a3cc-00a0c9223196}"
                    ),
                ),
                SimpleNamespace(
                    index=2,
                    name="HD USB Camera",
                    path=(
                        r"@device:pnp:\\?\usb#vid_05a3&pid_9230&mi_00"
                        r"#6&3405b72d&0&0000#{6994ad05-93ef-11d0-a3cc-00a0c9223196}"
                    ),
                ),
            ]
            resolved = resolve_stereo_cameras(
                registry,
                backend="dshow",
                enumerate_devices=lambda _backend: devices,
            )

        self.assertEqual(resolved.backend, "dshow")
        self.assertEqual(resolved.left.index, 2)
        self.assertEqual(resolved.right.index, 4)

    def test_rejects_missing_or_changed_physical_camera(self) -> None:
        with TemporaryDirectory() as temp:
            registry = self._write_registry(Path(temp))
            devices = [
                SimpleNamespace(
                    index=0,
                    name="HD USB Camera",
                    path=r"@device:pnp:\\?\usb#vid_05a3&pid_9230&mi_00#other#{guid}",
                )
            ]
            with self.assertRaises(CameraRegistryError):
                resolve_stereo_cameras(
                    registry,
                    backend="dshow",
                    enumerate_devices=lambda _backend: devices,
                )


if __name__ == "__main__":
    unittest.main()
