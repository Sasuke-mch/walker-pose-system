from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from observe_walker_self_occlusion_temporal import (  # noqa: E402
    CandidateMask,
    border_supported_components,
    cross_view_audit,
    make_candidate,
    overlay,
    permitted_structural_pixels,
    temporal_stability,
)


class WalkerSelfOcclusionTemporalTests(unittest.TestCase):
    def test_keeps_only_large_lower_border_supported_component(self):
        stable = np.zeros((20, 20), dtype=bool)
        stable[12:, 2:8] = True
        stable[2:8, 12:18] = True
        mask = border_supported_components(stable, minimum_component_area_px=10, border_anchor_y_fraction=0.35)
        self.assertGreater(int((mask[12:, 2:8] > 0).sum()), 0)
        self.assertEqual(int((mask[2:8, 12:18] > 0).sum()), 0)

    def test_whole_image_stability_is_rejected_as_ambiguous(self):
        result = make_candidate(np.ones((20, 20), dtype=np.float32), 2, minimum_stability_fraction=0.75, minimum_component_area_px=10, maximum_mask_fraction=0.35, border_anchor_y_fraction=0.35)
        self.assertEqual(result.status, "unavailable")
        self.assertIn("stable_region_dominates_image_ambiguous", result.reasons)

    def test_cross_view_requires_both_candidates_and_comparable_area(self):
        left = CandidateMask("candidate", np.zeros((10, 10), dtype=np.uint8), [], {"mask_fraction": 0.1})
        right = CandidateMask("candidate", np.zeros((10, 10), dtype=np.uint8), [], {"mask_fraction": 0.2})
        status, reasons, audit = cross_view_audit(left, right, 4.0)
        self.assertEqual(status, "candidate")
        self.assertEqual(reasons, [])
        self.assertTrue(audit["both_candidate"])
        unavailable = CandidateMask("unavailable", np.zeros((10, 10), dtype=np.uint8), ["x"], {"mask_fraction": 0.0})
        status, reasons, _ = cross_view_audit(left, unavailable, 4.0)
        self.assertEqual(status, "unavailable")
        self.assertIn("candidate_missing_in_one_view", reasons)

    def test_downsampled_flow_is_returned_at_original_image_shape(self):
        images = [np.zeros((80, 160, 3), dtype=np.uint8), np.zeros((80, 160, 3), dtype=np.uint8)]
        stability, counts = temporal_stability(images, flow_max_px=1.5, intensity_difference_max=18.0, flow_width=40)
        self.assertEqual(counts, [1, 1])
        self.assertEqual(stability[0].shape, (80, 160))
        self.assertTrue(np.allclose(stability[0], 1.0))

    def test_temporal_stride_marks_its_two_valid_endpoints(self):
        images = [np.zeros((20, 40, 3), dtype=np.uint8) for _ in range(5)]
        _, counts = temporal_stability(images, flow_max_px=1.5, intensity_difference_max=18.0, temporal_stride=3)
        self.assertEqual(counts, [1, 1, 0, 1, 1])

    def test_overlay_accepts_a_noncontiguous_boolean_selection(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[10:, 3:6] = 255
        rendered = overlay(image, mask, "candidate only")
        self.assertEqual(rendered.shape, image.shape)
        self.assertGreater(int(rendered[15, 4, 1]), 0)

    def test_dark_structure_requires_scene_exclusion_and_excludes_it(self):
        image = np.zeros((12, 12, 3), dtype=np.uint8)
        image[2:10, 2:10] = 255
        excluded = np.zeros((12, 12), dtype=bool)
        excluded[0, 0] = True
        allowed, metrics = permitted_structural_pixels(image, excluded, dark_structure_max=40)
        self.assertFalse(allowed[0, 0])
        self.assertTrue(allowed[11, 11])
        self.assertIsNotNone(metrics["semantic_exclusion_fraction"])
        with self.assertRaises(ValueError):
            permitted_structural_pixels(image, None, dark_structure_max=40)


if __name__ == "__main__":
    unittest.main()
