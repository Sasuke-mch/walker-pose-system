from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_manual_floor_labels import binary_metrics, manual_audit_summary_boundary, rasterize_labelme  # noqa: E402


class ManualFloorLabelAuditTests(unittest.TestCase):
    def test_visible_person_overrides_overlapping_floor(self) -> None:
        sample = {
            "imageWidth": 8, "imageHeight": 8,
            "shapes": [
                {"label": "floor_eligible", "shape_type": "polygon", "points": [[0, 0], [7, 0], [7, 7], [0, 7]]},
                {"label": "person", "shape_type": "polygon", "points": [[2, 2], [5, 2], [5, 5], [2, 5]]},
            ],
        }
        masks = rasterize_labelme(sample)
        self.assertFalse(masks["floor_eligible"][3, 3])
        self.assertTrue(masks["person"][3, 3])

    def test_binary_metrics_report_expected_errors(self) -> None:
        reference = np.asarray([[True, False], [False, False]])
        candidate = np.asarray([[True, True], [False, False]])
        valid = np.ones((2, 2), dtype=bool)
        result = binary_metrics(reference, valid, candidate)
        self.assertEqual(result["true_positive_px"], 1)
        self.assertEqual(result["false_positive_px"], 1)
        self.assertEqual(result["false_negative_px"], 0)
        self.assertEqual(result["iou"], 0.5)

    def test_summary_boundary_reports_actual_labelled_image_count(self) -> None:
        self.assertIn("6 manually labelled left-view images", manual_audit_summary_boundary(6))


if __name__ == "__main__":
    unittest.main()
