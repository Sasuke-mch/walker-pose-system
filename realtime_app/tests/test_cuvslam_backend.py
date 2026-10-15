"""Verify native build identity and explicit prediction diagnostic controls."""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from walker_tools.scene.cuvslam_backend import native_geometry_contract_version, run


def manifest():
    return {
        "schema_version": 1,
        "calibration": {
            "model": "opencv_fisheye", "R_right_from_left": np.eye(3).tolist(),
            "translation_right_from_left_m": [.1, 0, 0], "image_size": [8, 6],
        },
        "frames": [{"frame_id": i, "pair_id": i, "timestamp_ns": i * 33_333_333}
                   for i in range(3)],
    }


class CuvslamBackendTests(unittest.TestCase):
    def test_geometry_contract_reads_native_symbol_and_keeps_official_unpatched(self):
        patched = SimpleNamespace(CUVSLAMWalkerGeometryContractVersion=Mock(return_value=1))
        with patch("walker_tools.scene.cuvslam_backend.ctypes.CDLL", return_value=patched):
            self.assertEqual(native_geometry_contract_version("/sdk/cuvslam/__init__.py"), 1)
        with patch("walker_tools.scene.cuvslam_backend.ctypes.CDLL", return_value=SimpleNamespace()):
            self.assertEqual(native_geometry_contract_version("/sdk/cuvslam/__init__.py"), 0)

    def test_missing_required_native_patch_preserves_every_failed_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "manifest.json"
            source.write_text(json.dumps(manifest()))
            output = folder / "poses.jsonl"
            fake = SimpleNamespace(__file__="/sdk/cuvslam/__init__.py")
            with patch.dict("sys.modules", cuvslam=fake), \
                    patch("walker_tools.scene.cuvslam_backend.importlib.metadata.version", return_value="17.0.1"), \
                    patch("walker_tools.scene.cuvslam_backend.native_geometry_contract_version", return_value=0), \
                    redirect_stderr(io.StringIO()):
                code = run(source, output, None, True, require_geometry_contract=True)
            self.assertEqual(code, 2)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["frame_id"] for row in rows], [0, 1, 2])
            self.assertTrue(all(row["reason"] == "backend_initialization_failed" for row in rows))
            self.assertTrue(all(row["world_from_left_camera_m"] is None for row in rows))

    def test_no_motion_model_is_applied_and_reported_without_inventing_success(self):
        class Config:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class Tracker:
            OdometryConfig = Config
            MulticameraMode = SimpleNamespace(Precision=1)
            OdometryMode = SimpleNamespace(Multicamera=1)

            def __init__(self, rig, config, slam):
                self.odom = SimpleNamespace(get_primary_cameras=lambda: [0])
                self.config = config

            def track(self, *args, **kwargs):
                raise ValueError("retained diagnostic tracking failure")

        fake = SimpleNamespace(__file__="/sdk/cuvslam/__init__.py", Tracker=Tracker)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "manifest.json"
            source.write_text(json.dumps(manifest()))
            output = folder / "poses.jsonl"
            with patch.dict("sys.modules", cuvslam=fake), \
                    patch("walker_tools.scene.cuvslam_backend.importlib.metadata.version", return_value="17.0.1"), \
                    patch("walker_tools.scene.cuvslam_backend.native_geometry_contract_version", return_value=0), \
                    patch("walker_tools.scene.cuvslam_backend.build_rig", return_value=object()), \
                    patch("walker_tools.scene.cuvslam_backend.load_inputs", return_value=([], [])), \
                    redirect_stdout(io.StringIO()):
                code = run(source, output, None, False, use_motion_model=False, use_gpu=False)
            self.assertEqual(code, 0)
            metadata = json.loads(output.with_suffix(".metadata.json").read_text())
            self.assertIs(metadata["requested_use_motion_model"], False)
            self.assertIs(metadata["native_use_motion_model"], False)
            self.assertIs(metadata["use_gpu"], False)
            self.assertEqual(metadata["implementation_scope"], "native_cpu_diagnostic_not_gpu_runtime_comparison")
            self.assertIs(metadata["native_secondary_mask_filter_before_geometry"], False)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertTrue(all(row["status"] == "unavailable" for row in rows))
            self.assertTrue(all(row["world_from_left_camera_m"] is None for row in rows))


if __name__ == "__main__":
    unittest.main()
