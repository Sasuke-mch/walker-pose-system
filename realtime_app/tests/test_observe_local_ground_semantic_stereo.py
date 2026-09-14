from argparse import Namespace
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from observe_local_ground_semantic_stereo import (  # noqa: E402
    canonical_plane_toward_camera,
    decide_observation,
    local_coverage_fraction,
    mask_seed_upright,
)


def gates() -> Namespace:
    return Namespace(
        minimum_candidates=10,
        minimum_inliers=6,
        minimum_inlier_fraction=0.4,
        minimum_coverage_fraction=0.1,
        maximum_median_residual_mm=25.0,
        maximum_median_reprojection_px=5.0,
    )


class LocalGroundObservationTests(unittest.TestCase):
    def test_plane_sign_is_canonical_for_downstream_signed_support_checks(self):
        plane = canonical_plane_toward_camera(np.asarray([0.0, -2.0, 0.0]), -2000.0)
        self.assertEqual(plane["normal_orientation"], "toward_camera")
        self.assertTrue(np.allclose(plane["normal_left_camera"], [0.0, 1.0, 0.0]))
        self.assertAlmostEqual(plane["offset_mm"], 1000.0)

    def test_direct_requires_audited_external_identity_and_all_geometry_gates(self):
        state, reasons = decide_observation(
            semantic_source="external_binary_mask", semantic_identity_evidence="manually_audited",
            candidate_count=12, inlier_count=8, inlier_fraction=0.67, coverage_fraction=0.2,
            residual_mm=5.0, reprojection_left_px=1.0, reprojection_right_px=1.5, args=gates(),
        )
        self.assertEqual(state, "direct")
        self.assertEqual(reasons, [])

    def test_nonsemantic_lower_region_is_forced_unavailable_even_if_geometry_looks_good(self):
        state, reasons = decide_observation(
            semantic_source="lower_region_unknown_nonsemantic", semantic_identity_evidence="not_available",
            candidate_count=12, inlier_count=8, inlier_fraction=0.67, coverage_fraction=0.2,
            residual_mm=5.0, reprojection_left_px=1.0, reprojection_right_px=1.5, args=gates(),
        )
        self.assertEqual(state, "unavailable")
        self.assertIn("semantic_evidence_missing", reasons)

    def test_empty_mask_has_no_seed_and_collinear_inliers_have_zero_coverage(self):
        self.assertIsNone(mask_seed_upright(np.zeros((10, 10), dtype=np.uint8)))
        self.assertEqual(local_coverage_fraction(np.asarray(((1, 1), (2, 2), (3, 3))), (10, 10)), 0.0)


if __name__ == "__main__":
    unittest.main()
