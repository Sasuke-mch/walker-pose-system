"""Rectified rig direction, original-camera output, and conservative mask tests."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from walker_tools.scene.cuvslam_backend import run
from walker_tools.scene.prepare_cuvslam_rectified import conservative_rectified_mask, stereo_rectification


class CuvslamRectificationTests(unittest.TestCase):
    def test_stereo_axes_baseline_and_known_raw_rays(self):
        rotation = cv2.Rodrigues(np.array([.4, .05, 3.05]))[0]
        translation = np.array([-.02, -.27, .1])
        k = np.array([[100., 0, 160], [0, 100., 90], [0, 0, 1.]])
        calibration = {"model": "opencv_fisheye", "image_size": [320, 180],
                       "left_K": k.tolist(), "right_K": k.tolist(),
                       "left_D": [0.] * 4, "right_D": [0.] * 4,
                       "R_right_from_left": rotation.tolist(),
                       "translation_right_from_left_m": translation.tolist()}
        active, details, _ = stereo_rectification(calibration)
        r1 = np.array(details["R_rectified_left_from_original_left"])
        r2 = np.array(details["R_rectified_right_from_original_right"])
        np.testing.assert_allclose(r2 @ rotation @ r1.T, np.eye(3), atol=1e-12)
        self.assertEqual(active["model"], "opencv_pinhole")
        self.assertEqual((active["left_D"], active["right_D"]), ([], []))
        np.testing.assert_allclose(np.linalg.norm(active["translation_right_from_left_m"]),
                                   np.linalg.norm(translation), atol=1e-12)
        points = np.array([[.1, .2, 2.], [-.3, -.1, 3.], [.2, -.1, 5.]])
        rectified_uv = []
        for side, xyz, rectifier in (("left", points, r1),
                                     ("right", points @ rotation.T + translation, r2)):
            raw, _ = cv2.fisheye.projectPoints(xyz[:, None, :], np.zeros(3), np.zeros(3), k, np.zeros(4))
            rect_k = np.array(active[f"{side}_K"])
            pixels = cv2.fisheye.undistortPoints(raw, k, np.zeros(4), R=rectifier, P=rect_k).reshape(-1, 2)
            projected = (xyz @ rectifier.T) @ rect_k.T
            np.testing.assert_allclose(pixels, projected[:, :2] / projected[:, 2:], atol=1e-9)
            rectified_uv.append(pixels)
        np.testing.assert_allclose(rectified_uv[0][:, 1], rectified_uv[1][:, 1], atol=1e-9)

    def test_any_masked_bilinear_source_or_outside_sample_excludes_output(self):
        mask = np.zeros((6, 6), dtype=np.uint8)
        mask[3, 3] = 255
        mx = np.array([[2.2, 1., -.1, 5., np.nan]], dtype=np.float32)
        my = np.array([[2.4, 1., 0., 5., 1.]], dtype=np.float32)
        np.testing.assert_array_equal(conservative_rectified_mask(mask, mx, my), [[255, 0, 255, 255, 255]])

    def test_rectified_pose_and_landmark_export_have_explicit_original_frame(self):
        r1 = cv2.Rodrigues(np.array([0., 0., np.pi / 2]))[0]
        raw_point = np.array([.1, .2, 2.])
        pose = SimpleNamespace(rotation=[0, 0, 0, 1], translation=[1, 2, 3])

        class Config:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class Tracker:
            OdometryConfig = Config
            MulticameraMode = SimpleNamespace(Precision=1)
            OdometryMode = SimpleNamespace(Multicamera=1)

            def __init__(self, rig, config, slam):
                self.odom = SimpleNamespace(get_primary_cameras=lambda: [0],
                                            get_state=lambda: SimpleNamespace(warming_up=False, keyframe=True))

            def track(self, *args, **kwargs):
                estimate = SimpleNamespace(pose=pose, covariance_xyz_rpy=np.eye(6))
                return SimpleNamespace(world_from_rig=estimate), None

            def get_last_observations(self, camera):
                return []

            def get_last_landmarks(self):
                return [SimpleNamespace(id=1, coords=r1 @ raw_point)]

        manifest = {"schema_version": 1, "calibration": {
            "model": "opencv_pinhole", "R_right_from_left": np.eye(3).tolist(),
            "translation_right_from_left_m": [-.1, 0, 0], "image_size": [8, 6],
            "left_D": [], "right_D": [],
        }, "frames": [{"frame_id": 0, "pair_id": 0, "timestamp_ns": 0}],
            "rectification": {"R_rectified_left_from_original_left": r1.tolist()}}
        fake = SimpleNamespace(__file__="/sdk/cuvslam/__init__.py", Tracker=Tracker)
        inputs = ([np.zeros((6, 8), np.uint8)] * 2, [np.zeros((6, 8), np.uint8)] * 2)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "manifest.json"
            source.write_text(json.dumps(manifest))
            output = folder / "pose.jsonl"
            with patch.dict("sys.modules", cuvslam=fake), \
                    patch("walker_tools.scene.cuvslam_backend.importlib.metadata.version", return_value="17.0.1"), \
                    patch("walker_tools.scene.cuvslam_backend.native_geometry_contract_version", return_value=1), \
                    patch("walker_tools.scene.cuvslam_backend.build_rig", return_value=object()), \
                    patch("walker_tools.scene.cuvslam_backend.load_inputs", return_value=inputs), \
                    redirect_stdout(io.StringIO()):
                run(source, output, None, False, use_gpu=False)
            row = json.loads(output.read_text())
            native = np.array(row["world_from_left_camera_m"])
            original = np.array(row["world_from_original_left_camera_m"])
            np.testing.assert_allclose(original[:3, :3], r1, atol=1e-12)
            np.testing.assert_allclose(original @ [*raw_point, 1], native @ [*(r1 @ raw_point), 1])
            np.testing.assert_allclose(row["landmarks"][0]["coords_m_original_left"], raw_point)
            self.assertEqual(row["pose_camera_frame"], "rectified_left_camera")


if __name__ == "__main__":
    unittest.main()
