"""Annotation identity, anatomical naming and upright/raw pixel conversion."""
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from pose_app.smplh_first_frame_labels import apply_first_frame_labels


class FirstFrameLabelTests(unittest.TestCase):
    def make_sources(self, directory):
        root = Path(directory)
        labels = root/"labels"
        labels.mkdir()
        pairs = root/"input"
        for side, folder in (("left", "left_ccw90"), ("right", "right_cw90")):
            image_folder = pairs/folder
            image_folder.mkdir(parents=True)
            (image_folder/"pair_0000.png").write_bytes(cv2.imencode(".png", np.zeros((1920, 1080, 3), np.uint8))[1].tobytes())
            shapes = [{"label": "leftt_hand" if side == "left" else "left_hand", "shape_type": "point", "points": [[200., 300.]]},
                      {"label": "right_foot", "shape_type": "point", "points": [[600., 1600.]]}]
            data = {"imageWidth": 1080, "imageHeight": 1920, "imagePath": str(image_folder/"pair_0000.png"), "shapes": shapes}
            (labels/f"{side}_pair_0000.json").write_text(json.dumps(data), encoding="utf-8")
        return labels, pairs

    def test_explicit_wrist_ankle_alias_and_inverse_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            labels, pairs = self.make_sources(directory)
            left, right = np.zeros((2, 17, 3)), np.zeros((2, 17, 3))
            mask, record = apply_first_frame_labels(labels, pairs, left, right)
            self.assertEqual(record["manual_points"], 4)
            self.assertTrue(mask[0, 4] and mask[1, 11])
            np.testing.assert_array_equal(left[0, 9], [1619., 200., 1.])
            np.testing.assert_array_equal(right[0, 16], [1600., 479., 1.])
            np.testing.assert_array_equal(left[1], np.zeros((17, 3)))

    def test_wrong_image_and_duplicate_joint_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            labels, pairs = self.make_sources(directory)
            path = labels/"left_pair_0000.json"
            data = json.loads(path.read_text())
            data["shapes"].append(data["shapes"][0])
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                apply_first_frame_labels(labels, pairs, np.zeros((1, 17, 3)), np.zeros((1, 17, 3)))
            data["shapes"].pop()
            other = Path(directory)/"other.png"
            other.write_bytes(cv2.imencode(".png", np.ones((1920, 1080, 3), np.uint8))[1].tobytes())
            data["imagePath"] = str(other)
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "not the selected"):
                apply_first_frame_labels(labels, pairs, np.zeros((1, 17, 3)), np.zeros((1, 17, 3)))


if __name__ == "__main__":
    unittest.main()
