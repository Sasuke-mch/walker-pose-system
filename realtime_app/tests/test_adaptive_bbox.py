import unittest

from pose_app.adaptive_bbox import expand_bbox, expand_detections


class AdaptiveBboxTests(unittest.TestCase):
    def test_small_person_box_is_preserved(self):
        box, stats = expand_bbox([100, 100, 300, 500], 1000, 1000)
        self.assertEqual(box, [100.0, 100.0, 300.0, 500.0])
        self.assertFalse(stats["expanded"])


    def test_large_person_box_expands_downward_more_than_sideways(self):
        box, stats = expand_bbox([100, 100, 900, 700], 1000, 1000)
        self.assertTrue(stats["expanded"])
        self.assertLess(box[0], 100)
        self.assertGreater(box[2], 900)
        self.assertLess(box[1], 100)
        self.assertGreater(box[3], 700)
        self.assertLess(100 - box[0], box[3] - 700)
        self.assertTrue(0 <= box[0] < box[2] <= 1000)
        self.assertTrue(0 <= box[1] < box[3] <= 1000)


    def test_non_person_detection_is_unchanged(self):
        output, stats = expand_detections(
            [{"class_name": "cat", "bbox": [10, 10, 900, 900]}], 1000, 1000
        )
        self.assertEqual(output[0]["bbox"], [10, 10, 900, 900])
        self.assertEqual(stats["expanded_detections"], 0)
