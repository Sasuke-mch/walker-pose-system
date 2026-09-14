from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from audit_people1_camera_attached_semantics import SceneMasks, exclusive_labels, temporal_edge_template, template_overlap  # noqa: E402


class CameraAttachedSemanticAuditTests(unittest.TestCase):
    def test_persistent_dark_line_produces_template(self):
        images = []
        scenes = []
        for shift in (0, 8, 16, 24):
            image = np.full((80, 100, 3), 220, dtype=np.uint8)
            image[:, 28:33] = 20
            image[55:60, 55 + shift:60 + shift] = 20  # moving edge should not persist
            images.append(image)
            scenes.append(SceneMasks(np.zeros((80, 100), dtype=bool), np.zeros((80, 100), dtype=bool), {}))
        template, metrics = temporal_edge_template(images, scenes, edge_support_min=0.72, dark_template_max=115, maximum_scene_exclusion_fraction=0.3, template_dilate_px=5, minimum_component_area_px=20)
        self.assertTrue(template[:, 30].any())
        self.assertLess(float(template[57, 64]), 0.5)
        self.assertEqual(metrics["sample_count"], 4)

    def test_visible_dark_template_wins_then_person_then_floor(self):
        image = np.full((12, 12, 3), 200, dtype=np.uint8)
        image[2:5, 2:5] = 20
        template = np.zeros((12, 12), dtype=bool)
        template[2:5, 2:5] = True
        person = np.zeros((12, 12), dtype=bool)
        person[2:8, 2:8] = True
        floor = np.ones((12, 12), dtype=bool)
        labels, fractions = exclusive_labels(image, template, SceneMasks(person, floor, {}), 135)
        self.assertTrue(np.all(labels[2:5, 2:5] == 2))
        self.assertEqual(int(labels[6, 6]), 1)
        self.assertEqual(int(labels[10, 10]), 3)
        self.assertAlmostEqual(sum(fractions.values()), 1.0)

    def test_template_overlap_reports_iou_recall_and_precision(self):
        candidate = np.asarray(((1, 1), (1, 0)), dtype=bool)
        reference = np.asarray(((1, 0), (1, 1)), dtype=bool)
        metrics = template_overlap(candidate, reference)
        self.assertAlmostEqual(metrics["iou"], 0.5)
        self.assertAlmostEqual(metrics["reference_coverage_recall"], 2 / 3)
        self.assertAlmostEqual(metrics["candidate_precision_against_reference"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
