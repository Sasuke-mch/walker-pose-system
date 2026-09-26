from __future__ import annotations

"""Fail-closed scene-accepted mask tests for the Stage D temporal loss.

The resolver lives in the frozen-route fit script
``research_records/engineering_validation/G20260924_smpl_vposer_shared_beta_v1/fit_vposer_shared_beta.py``.
Importing that module pulls cv2/SMPL/VPoser weights, so the tests extract the
single pure-numpy function by AST and execute it in isolation. This still
tests the shipped code: any edit to the resolver changes what is executed.
"""

import ast
import unittest
from pathlib import Path

import numpy as np

FIT = (
    Path(__file__).resolve().parents[2]
    / "research_records/engineering_validation/G20260924_smpl_vposer_shared_beta_v1/fit_vposer_shared_beta.py"
)


def load_resolver():
    tree = ast.parse(FIT.read_text(encoding="utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "resolve_scene_frame_mask"
    )
    namespace: dict = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(FIT), "exec"), namespace)
    return namespace["resolve_scene_frame_mask"]


class ResolveSceneFrameMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.resolve = load_resolver()

    def test_normal_joint_field_returns_frame_any(self) -> None:
        full = np.zeros((448, 17), dtype=bool)
        full[60:91, 3] = True
        got = type(self).resolve(full, slice(60, 91), 31, np.ones(31, dtype=bool))
        self.assertEqual(got.shape, (31,))
        self.assertTrue(got.all())

    def test_missing_field_raises_instead_of_all_true(self) -> None:
        with self.assertRaises(ValueError):
            type(self).resolve(None, slice(60, 91), 31, np.ones(31, dtype=bool))

    def test_wrong_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            type(self).resolve(np.zeros((448, 5), dtype=bool), slice(60, 91), 31, np.ones(31, dtype=bool))
        with self.assertRaises(ValueError):
            type(self).resolve(np.zeros((448, 17), dtype=bool), slice(60, 90), 31, np.ones(31, dtype=bool))

    def test_abnormal_values_raise(self) -> None:
        bad = np.full((448, 17), 7)
        with self.assertRaises(ValueError):
            type(self).resolve(bad, slice(60, 91), 31, np.ones(31, dtype=bool))


if __name__ == "__main__":
    unittest.main()
