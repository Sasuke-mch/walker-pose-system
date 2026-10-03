from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from pose_app.smpl_coco_observation import (
    COCO17_NAMES,
    SMPL_VERTEX_COUNT,
    inspect_smpl_coco_assets,
    regress_coco17_numpy,
    regress_coco17_torch,
    resolve_smpl_male_model,
    validate_coco17_regressor,
)


def one_hot_regressor() -> np.ndarray:
    matrix = np.zeros((len(COCO17_NAMES), SMPL_VERTEX_COUNT), dtype=np.float64)
    matrix[np.arange(len(COCO17_NAMES)), np.arange(len(COCO17_NAMES))] = 1.0
    return matrix


class SmplCocoObservationTests(unittest.TestCase):
    def test_resolves_official_and_renamed_male_model(self) -> None:
        for name in ("SMPL_MALE.pkl", "basicmodel_m_lbs_10_207_0_v1.0.0.pkl"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                model = Path(folder) / "smpl" / name
                model.parent.mkdir()
                model.write_bytes(b"placeholder")
                self.assertEqual(resolve_smpl_male_model(folder), model.resolve())

    def test_numpy_regressor_selects_expected_vertices(self) -> None:
        vertices = np.arange(SMPL_VERTEX_COUNT * 3, dtype=np.float64).reshape(
            SMPL_VERTEX_COUNT, 3
        )
        result = regress_coco17_numpy(vertices, one_hot_regressor())
        np.testing.assert_array_equal(result, vertices[:17])

    def test_rejects_smplx_topology_and_invalid_weights(self) -> None:
        with self.assertRaisesRegex(ValueError, "6890"):
            regress_coco17_numpy(np.zeros((10475, 3)), one_hot_regressor())
        invalid = one_hot_regressor()
        invalid[0, 0] = -1.0
        with self.assertRaisesRegex(ValueError, "non-negative"):
            validate_coco17_regressor(invalid)

    def test_torch_path_preserves_gradient(self) -> None:
        import torch

        vertices = torch.arange(
            SMPL_VERTEX_COUNT * 3, dtype=torch.float32
        ).reshape(SMPL_VERTEX_COUNT, 3).requires_grad_(True)
        result = regress_coco17_torch(vertices, one_hot_regressor())
        result.sum().backward()
        self.assertEqual(tuple(result.shape), (17, 3))
        self.assertTrue(torch.all(vertices.grad[:17] == 1))
        self.assertTrue(torch.all(vertices.grad[17:] == 0))

    def test_asset_gate_keeps_model_and_regressor_separate(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            regressor = root / "J_regressor_coco.npy"
            np.save(regressor, one_hot_regressor())
            report = inspect_smpl_coco_assets(root, regressor)
            self.assertFalse(report["ready"])
            self.assertEqual(report["regressor_status"], "valid_17x6890")
            self.assertEqual(report["missing"], ["licensed_SMPL_MALE_6890_model"])


if __name__ == "__main__":
    unittest.main()
