import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pose_app.raw_pair_writer import RawStereoPairWriter


class _FakeVideoWriter:
    def __init__(self, *args):
        self.frames = []
        self.released = False

    def isOpened(self):
        return True

    def write(self, image):
        self.frames.append(image.copy())

    def release(self):
        self.released = True


def _pair(left, right):
    return SimpleNamespace(
        pair_id=3,
        left=SimpleNamespace(frame_id=11, timestamp_sec=1.25, image=left),
        right=SimpleNamespace(frame_id=12, timestamp_sec=1.26, image=right),
        timestamp_skew_sec=0.01,
        dropped_left=2,
        dropped_right=4,
    )


class RawPairWriterTests(unittest.TestCase):
    def test_writes_raw_frames_and_sidecar(self):
        writers = []

        def make_writer(*args):
            writer = _FakeVideoWriter(*args)
            writers.append(writer)
            return writer

        with tempfile.TemporaryDirectory() as directory, patch(
            "pose_app.raw_pair_writer.cv2.VideoWriter_fourcc", return_value=1
        ), patch("pose_app.raw_pair_writer.cv2.VideoWriter", side_effect=make_writer):
            image = np.zeros((4, 6, 3), dtype=np.uint8)
            writer = RawStereoPairWriter(Path(directory), 30.0)
            writer.write(_pair(image, image.copy()))
            summary = writer.close()
            self.assertEqual(summary["frames"], 1)
            self.assertEqual(len(writers), 2)
            self.assertEqual(len(writers[0].frames), 1)
            records = [json.loads(line) for line in (Path(directory) / "raw_pairs.jsonl").read_text().splitlines()]
            self.assertEqual(records[0]["pair_id"], 3)
            self.assertEqual(records[0]["left_shape"], [4, 6, 3])

    def test_rejects_mismatched_dimensions(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "pose_app.raw_pair_writer.cv2.VideoWriter_fourcc", return_value=1
        ), patch("pose_app.raw_pair_writer.cv2.VideoWriter", side_effect=_FakeVideoWriter):
            writer = RawStereoPairWriter(Path(directory), 30.0)
            left = np.zeros((4, 6, 3), dtype=np.uint8)
            right = np.zeros((5, 6, 3), dtype=np.uint8)
            with self.assertRaises(ValueError):
                writer.write(_pair(left, right))
            writer.close()


if __name__ == "__main__":
    unittest.main()
