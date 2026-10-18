from __future__ import annotations

import unittest

import cv2
import numpy as np

from tools.extract_annotated_walker_evidence import (
    cluster_persistent_axes,
    component_candidates,
    floor_notch_mask,
    polygon_mask,
)


class AnnotatedWalkerEvidenceTests(unittest.TestCase):
    def test_polygon_mask_unions_matching_label_only(self) -> None:
        label = {
            "imageHeight": 40,
            "imageWidth": 50,
            "shapes": [
                {"label": "floor_eligible", "shape_type": "polygon", "points": [[2, 2], [20, 2], [20, 20], [2, 20]]},
                {"label": "person", "shape_type": "polygon", "points": [[30, 2], [40, 2], [40, 20], [30, 20]]},
            ],
        }
        mask = polygon_mask(label, "floor_eligible")
        self.assertEqual(int(mask[10, 10]), 255)
        self.assertEqual(int(mask[10, 35]), 0)

    def test_horizontal_closing_recovers_narrow_floor_slot(self) -> None:
        floor = np.zeros((120, 160), dtype=np.uint8)
        floor[50:115, 10:150] = 255
        floor[50:105, 72:82] = 0
        recovered = floor_notch_mask(floor, 31)
        self.assertGreater(int(np.count_nonzero(recovered[50:105, 72:82])), 400)
        self.assertEqual(int(np.count_nonzero(recovered[:, :5])), 0)

    def test_component_filter_keeps_vertical_slot_and_rejects_wide_gap(self) -> None:
        image = np.full((160, 220, 3), 180, dtype=np.uint8)
        notch = np.zeros((160, 220), dtype=np.uint8)
        notch[50:145, 30:42] = 255
        notch[80:140, 90:190] = 255
        image[50:145, 30:42] = 30
        floor = np.zeros_like(notch)
        floor[45:150, 5:210] = 255
        walker = np.zeros_like(notch)
        person = np.zeros_like(notch)
        records, accepted = component_candidates(
            notch,
            image,
            floor,
            walker,
            person,
            minimum_height=35,
            maximum_width=60,
        )
        self.assertEqual([record["status"] for record in records].count("candidate"), 1)
        self.assertEqual([record["status"] for record in records].count("rejected"), 1)
        self.assertGreater(int(np.count_nonzero(accepted)), 0)

    def test_axis_clustering_merges_repeated_fixed_tube_and_requires_three_frames(self) -> None:
        records = []
        for pair_id, shift in ((0, -2), (40, 1), (80, 0)):
            records.append(
                {
                    "pair_id": pair_id,
                    "side": "left",
                    "persistence_status": "persistent_candidate",
                    "axis_start_upright_px": [100 + shift, 900],
                    "axis_end_upright_px": [140 + shift, 1500],
                    "temporal_support_frames_maximum": 4,
                }
            )
        records.append(
            {
                "pair_id": 120,
                "side": "left",
                "persistence_status": "persistent_candidate",
                "axis_start_upright_px": [700, 900],
                "axis_end_upright_px": [710, 1200],
                "temporal_support_frames_maximum": 3,
            }
        )
        hypotheses = cluster_persistent_axes(records, "left", 1920)
        self.assertEqual(len(hypotheses), 1)
        self.assertEqual(hypotheses[0]["unique_frame_count"], 3)


if __name__ == "__main__":
    unittest.main()
