from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from pose_app.temporal_floor_mask_propagation import (  # noqa: E402
    PropagationThresholds,
    flow_consistency,
    warp_from_previous,
)
from propagate_audited_floor_masks import scheduled_reference_ids  # noqa: E402


class TemporalFloorMaskPropagationTests(unittest.TestCase):
    def test_backward_warp_moves_previous_mask_into_current_pixels(self) -> None:
        source = np.zeros((5, 6), dtype=np.uint8)
        source[2, 2] = 1
        # A current pixel samples the previous pixel one column to its left.
        backward = np.zeros((5, 6, 2), dtype=np.float32)
        backward[..., 0] = -1.0
        warped, in_bounds = warp_from_previous(source, backward, interpolation=0)
        self.assertTrue(in_bounds[2, 3])
        self.assertEqual(warped[2, 3], 1)
        self.assertEqual(warped[2, 2], 0)

    def test_opposite_constant_flows_are_consistent_away_from_border(self) -> None:
        forward = np.zeros((5, 6, 2), dtype=np.float32)
        backward = np.zeros((5, 6, 2), dtype=np.float32)
        forward[..., 0] = 1.0
        backward[..., 0] = -1.0
        residual, valid = flow_consistency(forward, backward)
        self.assertTrue(valid[2, 3])
        self.assertAlmostEqual(float(residual[2, 3]), 0.0, places=5)

    def test_thresholds_reject_invalid_ranges(self) -> None:
        with self.assertRaises(ValueError):
            PropagationThresholds(minimum_candidate_flow_consistency=1.1).validate()

    def test_one_shot_consumes_only_the_initial_manual_reference(self) -> None:
        self.assertEqual(
            scheduled_reference_ids(
                reference_mode="one_shot", initial_seed_pair_id=10,
                reanchor_interval_frames=None, start_pair_id=0, end_pair_id=50,
            ),
            [10],
        )

    def test_periodic_mode_requires_exact_fixed_interval_references(self) -> None:
        self.assertEqual(
            scheduled_reference_ids(
                reference_mode="periodic_reanchor", initial_seed_pair_id=0,
                reanchor_interval_frames=40, start_pair_id=0, end_pair_id=125,
            ),
            [0, 40, 80, 120],
        )


if __name__ == "__main__":
    unittest.main()
