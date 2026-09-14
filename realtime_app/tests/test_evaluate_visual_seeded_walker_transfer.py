from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from evaluate_visual_seeded_walker_transfer import binary_metrics, build_template, rasterize_annotation, visible_template  # noqa: E402


class VisualSeededWalkerTransferTests(unittest.TestCase):
    def test_manual_labels_are_exclusive_with_visible_walker_priority(self):
        record = {"person_polygons_xy": [[[0, 0], [7, 0], [7, 7], [0, 7]]], "floor_polygons_xy": [[[0, 0], [9, 0], [9, 9], [0, 9]]], "walker_polylines": [{"points_xy": [[1, 4], [8, 4]], "thickness_px": 1}]}
        labels = rasterize_annotation(record, (10, 10))
        self.assertEqual(int(labels[4, 4]), 2)
        self.assertEqual(int(labels[1, 1]), 1)
        self.assertEqual(int(labels[8, 8]), 3)

    def test_seed_consensus_visibility_and_metrics(self):
        first = np.zeros((4, 4), dtype=np.uint8)
        second = np.zeros((4, 4), dtype=np.uint8)
        third = np.zeros((4, 4), dtype=np.uint8)
        first[1, 1:3] = second[1, 1:3] = third[1, 2:4] = 2
        template = build_template([first, second, third], 2)
        self.assertTrue(np.array_equal(template[1], np.asarray([False, True, True, False])))
        image = np.full((4, 4, 3), 255, dtype=np.uint8)
        image[1, 1:3] = 20
        prediction = visible_template(template, image, 135)
        metrics = binary_metrics(prediction, first == 2)
        self.assertAlmostEqual(metrics["iou"], 1.0)


if __name__ == "__main__":
    unittest.main()
