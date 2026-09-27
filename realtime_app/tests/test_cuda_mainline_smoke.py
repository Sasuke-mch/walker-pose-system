from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import torch

from pose_app.smpl_coco_observation import (
    load_coco17_regressor,
    load_smpl_male,
    regress_coco17_torch,
)


class CudaMainlineSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA PyTorch is unavailable")
        cls.root = Path(__file__).resolve().parents[2]
        cls.smpl_root = cls.root / "models" / "smpl"
        cls.regressor = load_coco17_regressor(cls.smpl_root / "J_regressor_coco.npy")

    def test_smpl_forward_and_coco_regressor_stay_on_cuda(self) -> None:
        model = load_smpl_male(self.smpl_root, device="cuda")
        pose = torch.zeros((1, 72), dtype=torch.float32, device="cuda")
        betas = torch.zeros((1, 10), dtype=torch.float32, device="cuda")
        translation = torch.zeros((1, 3), dtype=torch.float32, device="cuda")
        with torch.no_grad():
            vertices = model(
                global_orient=pose[:, :3],
                body_pose=pose[:, 3:],
                betas=betas,
                transl=translation,
            ).vertices
            joints = regress_coco17_torch(vertices, self.regressor)
        self.assertEqual(tuple(vertices.shape), (1, 6890, 3))
        self.assertEqual(tuple(joints.shape), (1, 17, 3))
        self.assertEqual(vertices.device.type, "cuda")
        self.assertEqual(joints.device.type, "cuda")
        self.assertTrue(np.isfinite(joints.detach().cpu().numpy()).all())


if __name__ == "__main__":
    unittest.main()
