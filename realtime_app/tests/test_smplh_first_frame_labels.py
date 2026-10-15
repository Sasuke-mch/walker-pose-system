"""Annotation identity, anatomical naming and upright/raw pixel conversion."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from pose_app.smplh_first_frame_labels import apply_first_frame_labels, apply_first_frame_observations, refine_first_frame_2d


class FirstFrameLabelTests(unittest.TestCase):
    def test_failed_double_polish_restores_shared_model_precision(self):
        import torch
        class FailingVPoser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('value', torch.ones(1))

            def decode(self, latent):
                raise ValueError('deliberate projection failure')
        model = SimpleNamespace(
            device='cpu', initial_pose=SimpleNamespace(latent=np.zeros(32), root_rotation=np.eye(3), translation_m=np.array([0., 0., 2.])),
            betas=torch.zeros((1, 10)), model=torch.nn.Linear(1, 1), vposer=FailingVPoser(),
            fixed_body=torch.eye(3).repeat(21, 1, 1), regressor=torch.zeros((17, 6890)),
            hand_rotations=[torch.eye(3)[None, None].repeat(1, 15, 1, 1) for _ in range(2)])
        cal = SimpleNamespace(image_width=1920, image_height=1080, R_cam0_to_cam1=np.eye(3), T_cam0_to_cam1_mm=np.ones(3))
        pixels = np.tile([500., 300., .75], (2, 12, 1))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'deliberate projection'):
                refine_first_frame_2d(model, cal, pixels, np.zeros((2, 12), bool), Path(directory)/'fit',
                                      steps=0, optimize_shape=False, lbfgs_steps=2, lbfgs_double=True)
        for tensor in (model.model.weight, model.vposer.value, model.fixed_body, model.regressor, *model.hand_rotations):
            self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.equal(model.betas, torch.zeros((1, 10))))

    def test_detector_override_is_pair0_only_and_keeps_confidence(self):
        with tempfile.TemporaryDirectory() as directory:
            _, pairs = self.make_sources(directory)
            values = np.tile([500., 300., .75], (2, 17, 1))
            record = {"pair_id": 0, "coordinate_space": "raw_fisheye_pixels",
                      "keypoints": values.tolist(), "images": {
                          side: str(pairs/folder/"pair_0000.png") for side, folder in
                          (("left", "left_ccw90"), ("right", "right_cw90"))}}
            path = Path(directory)/"observations.json"
            path.write_text(json.dumps(record), encoding='utf-8')
            left, right = np.zeros((3, 17, 3)), np.ones((3, 17, 3))
            apply_first_frame_observations(path, pairs, left, right)
            np.testing.assert_array_equal(left[0], values[0])
            np.testing.assert_array_equal(right[0], values[1])
            self.assertFalse(left[1:].any())
            self.assertTrue(np.all(right[1:] == 1))
            record['images']['right'] = record['images']['left']
            path.write_text(json.dumps(record), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'differs from selected'):
                apply_first_frame_observations(path, pairs, left, right)

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
