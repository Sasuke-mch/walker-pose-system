from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


REALTIME_APP_DIR = Path(__file__).resolve().parents[1]
if str(REALTIME_APP_DIR) not in sys.path:
    sys.path.insert(0, str(REALTIME_APP_DIR))

from pose_app.sources import SourceFrame
from pose_app.stereo_camera import validate_camera_frame_image
from pose_app.stereo_sources import StereoSideBySideVideoSource


class _FakeVideoSource:
    def __init__(self, image: np.ndarray) -> None:
        self.path = Path("memory-side-by-side.mp4")
        self.width = int(image.shape[1])
        self.height = int(image.shape[0])
        self.fps = 30.0
        self._frames = [SourceFrame(7, 7 / self.fps, image)]
        self.closed = False

    def read(self):
        return self._frames.pop(0) if self._frames else None

    def close(self) -> None:
        self.closed = True


class ImageIntegrityReplayTests(unittest.TestCase):
    def test_camera_frame_validation_accepts_only_exact_uint8_bgr_raster(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.assertIs(
            validate_camera_frame_image(
                image,
                side="LEFT",
                expected_width=1920,
                expected_height=1080,
            ),
            image,
        )

        with self.assertRaisesRegex(RuntimeError, "resolution changed"):
            validate_camera_frame_image(
                image[:, :1919],
                side="LEFT",
                expected_width=1920,
                expected_height=1080,
            )
        with self.assertRaisesRegex(RuntimeError, "unexpected decoded layout"):
            validate_camera_frame_image(
                image[:, :, 0],
                side="LEFT",
                expected_width=1920,
                expected_height=1080,
            )
        with self.assertRaisesRegex(RuntimeError, "dtype changed"):
            validate_camera_frame_image(
                image.astype(np.float32),
                side="LEFT",
                expected_width=1920,
                expected_height=1080,
            )

    def test_sbs_split_preserves_every_panel_pixel_without_resize(self):
        # Encode all four outer panel corners with unique values.  A crop,
        # resize, or accidental half swap would make this equality test fail.
        image = np.zeros((6, 10, 3), dtype=np.uint8)
        image[:, :5] = (11, 22, 33)
        image[:, 5:] = (44, 55, 66)
        image[0, 0] = (1, 2, 3)
        image[-1, 4] = (4, 5, 6)
        image[0, 5] = (7, 8, 9)
        image[-1, -1] = (10, 11, 12)
        fake = _FakeVideoSource(image)

        with patch("pose_app.stereo_sources.VideoSource", return_value=fake):
            source = StereoSideBySideVideoSource(
                "ignored.mp4", left_panel_width=5
            )
            pair = source.read()

        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertEqual(pair.timestamp_skew_sec, 0.0)
        self.assertEqual(pair.timestamp_type, "side_by_side_video_original_skew_unknown")
        np.testing.assert_array_equal(pair.left.image, image[:, :5])
        np.testing.assert_array_equal(pair.right.image, image[:, 5:])
        self.assertEqual(
            source.integrity_metadata()["left_panel_size"], [5, 6]
        )
        self.assertEqual(
            source.integrity_metadata()["right_panel_size"], [5, 6]
        )
        source.close()
        self.assertTrue(fake.closed)

    def test_sbs_rejects_a_midstream_dimension_change(self):
        image = np.zeros((6, 10, 3), dtype=np.uint8)
        fake = _FakeVideoSource(image)
        fake._frames = [
            SourceFrame(0, 0.0, np.zeros((6, 8, 3), dtype=np.uint8))
        ]
        with patch("pose_app.stereo_sources.VideoSource", return_value=fake):
            source = StereoSideBySideVideoSource(
                "ignored.mp4", left_panel_width=5
            )
            with self.assertRaisesRegex(RuntimeError, "dimensions changed"):
                source.read()

    def test_sbs_sidecar_restores_original_pair_timing(self):
        image = np.zeros((6, 10, 3), dtype=np.uint8)
        fake = _FakeVideoSource(image)
        record = {
            "pair_id": 7,
            "timestamp_skew_ms": 4.25,
            "timestamp_type": "host_read_return_perf_counter_ns",
            "left_frame_id": 31,
            "right_frame_id": 32,
            "left_timestamp_sec": 10.0,
            "right_timestamp_sec": 10.00425,
            "dropped_left": 2,
            "dropped_right": 3,
        }
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / "stereo_results.jsonl"
            metadata.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with patch("pose_app.stereo_sources.VideoSource", return_value=fake):
                source = StereoSideBySideVideoSource(
                    "ignored.mp4",
                    left_panel_width=5,
                    metadata_jsonl=metadata,
                )
                pair = source.read()

        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertEqual(pair.left.frame_id, 31)
        self.assertEqual(pair.right.frame_id, 32)
        self.assertAlmostEqual(pair.timestamp_skew_sec, 0.00425)
        self.assertEqual(pair.timestamp_type, "host_read_return_perf_counter_ns")
        self.assertEqual(pair.dropped_left, 2)
        self.assertEqual(pair.dropped_right, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
