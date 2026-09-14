from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_floor_semantic_backends import BACKEND_SPECS, floor_class_id, groundnet_unavailable, mask_agreement, parse_args  # noqa: E402


class _Config:
    id2label = {0: "wall", 3: "floor", 12: "person"}


class _Model:
    config = _Config()


class FloorSemanticBackendTests(unittest.TestCase):
    def test_declares_the_four_comparison_backends(self) -> None:
        self.assertEqual(set(BACKEND_SPECS), {"segformer_b0", "mask2former_swin_small", "oneformer_swin_tiny", "groundnet_contract"})

    def test_resolves_floor_class_from_model_labels(self) -> None:
        self.assertEqual(floor_class_id(_Model()), 3)

    def test_mask_agreement_is_symmetric(self) -> None:
        first = np.asarray([[True, False], [True, False]])
        second = np.asarray([[True, True], [False, False]])
        self.assertEqual(mask_agreement(first, second), mask_agreement(second, first))
        self.assertEqual(mask_agreement(first, second)["iou"], 1 / 3)

    def test_groundnet_is_not_silently_converted_to_a_floor_mask(self) -> None:
        result = groundnet_unavailable()
        self.assertEqual(result.status, "unavailable")
        self.assertIsNone(result.mask)
        self.assertIn("monocular_normal_does_not_supply_metric_plane_offset", result.reasons)

    def test_default_batch_size_is_one(self) -> None:
        old_argv = sys.argv
        try:
            sys.argv = ["tool", "--input-dir", "in", "--output-dir", "out"]
            args = parse_args()
            self.assertEqual(args.batch_size, 1)
            self.assertNotIn("groundnet_contract", args.backends)
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
