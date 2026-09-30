import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pose_app.wilor_mano_prior import (CONVENTION, MIRROR, convert_record,
    encode_pca, read_parameter_view, rotation_pose_loss, validate_rotations)


def record(side="right"):
    return {"side": side, "pixel_frame": "raw_fisheye", "bbox_xyxy": [50, 50, 150, 150],
            "detector_confidence": .8, "mano_parameter_convention": CONVENTION,
            "mano_parameter_metadata": {"joint_order": ["index", "middle", "pinky", "ring", "thumb"],
                "rotation_type": "parent_relative_rotmat", "pose_mean_added_by_layer": False,
                "left_crop_mirrored": side == "left"},
            "mano_parameters": {"hand_pose": np.broadcast_to(np.eye(3), (15,3,3)).tolist(),
                "global_orient": np.eye(3)[None].tolist(), "betas": [0.]*10}}


class NativeManoTests(unittest.TestCase):
    def test_left_mirror_matches_rotated_geometry(self):
        c = record("left")
        r = Rotation.from_rotvec([.3, -.4, .2]).as_matrix()
        c["mano_parameters"]["hand_pose"][0] = r.tolist()
        got = convert_record(c)[0]
        point = np.array([.1, .2, .3])
        np.testing.assert_allclose(got @ (MIRROR @ point), MIRROR @ (r @ point))
        self.assertAlmostEqual(np.linalg.det(got), 1.)

    def test_invalid_reflection_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_rotations(MIRROR[None], 1)

    def test_wrong_mean_convention_is_rejected(self):
        c = record()
        c["mano_parameter_metadata"]["pose_mean_added_by_layer"] = True
        with self.assertRaisesRegex(ValueError, "metadata"):
            convert_record(c)

    def test_truncated_pca_reports_lost_rotation(self):
        r = Rotation.from_rotvec(np.tile([.2,.3,.4], (15,1))).as_matrix()[None]
        _, error = encode_pca(r, np.eye(45)[:12], np.zeros(45), np.ones(12))
        self.assertGreater(error.max(), .5)

    def test_full_pca_roundtrip_includes_mean_and_scale_once(self):
        rng = np.random.default_rng(8)
        c = np.linalg.qr(rng.normal(size=(45,45)))[0]
        mean = rng.normal(size=45) * .15
        scale = np.linspace(.8, 1.2, 45)
        r = Rotation.from_rotvec(rng.normal(size=(30,3))*.3).as_matrix().reshape(2,15,3,3)
        z, error = encode_pca(r, c, mean, scale)
        self.assertLess(error.max(), 1e-12)
        np.testing.assert_allclose(Rotation.from_rotvec((mean+(z/scale)@c).reshape(-1,3)).as_matrix(), r.reshape(-1,3,3), atol=1e-12)

    def test_reader_ignores_projected_2d_and_rejects_legacy(self):
        body = np.zeros((1,17,3)); body[:,9:11] = [100,100,.9]
        good, old = record("right"), record("left")
        good["keypoints_2d_raw_fisheye"] = [[99999,99999]]*21
        del old["mano_parameter_convention"]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"x.jsonl"
            p.write_text(json.dumps({"frame_index":0,"image":"pair_0000.png","records":[good,old]})+'\n')
            _, w, a = read_parameter_view(p,body,(1920,1080))
        self.assertEqual(w[0,0], 0)
        self.assertEqual(w[0,1], .8)
        self.assertTrue(any(x["reason"]=="missing_or_unknown_mano_parameter_convention" for x in a))

    def test_ambiguous_boxes_fail_closed(self):
        body = np.zeros((1,17,3)); body[:,9:11] = [100,100,.9]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x.jsonl"
            p.write_text(json.dumps({"frame_index":0,"image":"pair_0000.png","records":[record(),record()]})+'\n')
            _,w,a=read_parameter_view(p,body,(1920,1080))
        self.assertEqual(w.sum(),0)
        self.assertTrue(any(x["reason"]=="ambiguous_detector_boxes" for x in a))

    def test_pose_loss_gradient_reduces_error_and_masks_rejected_view(self):
        import torch
        pose = torch.full((2,45), .3, requires_grad=True)
        target = torch.eye(3).expand(2,2,15,3,3).clone()
        target[:,1] = torch.tensor(Rotation.from_rotvec([0,0,2]).as_matrix(),dtype=torch.float32)
        weights = torch.tensor([[1.,0.],[1.,0.]])
        opt = torch.optim.Adam([pose],lr=.05)
        initial = float(rotation_pose_loss(pose,target,weights).detach())
        for _ in range(25):
            opt.zero_grad(); loss=rotation_pose_loss(pose,target,weights); loss.backward(); opt.step()
        self.assertTrue(torch.isfinite(pose.grad).all())
        self.assertLess(float(rotation_pose_loss(pose,target,weights).detach()),initial*.1)

    def test_zero_coverage_has_zero_gradient(self):
        import torch
        pose=torch.full((1,45),.3,requires_grad=True)
        target=torch.eye(3).expand(1,2,15,3,3)
        loss=rotation_pose_loss(pose,target,torch.zeros(1,2)); loss.backward()
        self.assertEqual(float(loss.detach()),0)
        self.assertEqual(float(pose.grad.abs().sum()),0)

if __name__ == "__main__":
    unittest.main()
