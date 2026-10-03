from __future__ import annotations

"""Fail-closed contract of the archived Stage D temporal-mask resolver."""

import unittest

import numpy as np

from tests.fixtures.historical_contracts import resolve_scene_frame_mask


def load_resolver():
    return resolve_scene_frame_mask


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
