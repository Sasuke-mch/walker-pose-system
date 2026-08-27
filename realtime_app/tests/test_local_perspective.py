from __future__ import annotations

import unittest

import numpy as np

from pose_app.local_perspective import LocalPerspectiveModelInput


class LocalPerspectiveViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.size = (1280, 720)
        self.K = np.asarray(
            [[520.0, 0.0, 639.5], [0.0, 520.0, 359.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.D = np.zeros((4, 1), dtype=np.float64)
        self.builder = LocalPerspectiveModelInput(
            self.K, self.D, self.size, margin=1.35
        )

    def test_bbox_center_maps_to_virtual_center(self) -> None:
        view = self.builder.build([360.0, 180.0, 760.0, 620.0])
        mapped = view.raw_to_virtual(np.asarray([[560.0, 400.0]]))[0]
        self.assertLess(np.linalg.norm(mapped - np.asarray([view.cx, view.cy])), 1e-7)

    def test_raw_virtual_raw_roundtrip_is_subpixel(self) -> None:
        view = self.builder.build([360.0, 180.0, 760.0, 620.0])
        raw = np.asarray(
            [[420.0, 260.0], [560.0, 400.0], [700.0, 540.0], [480.0, 560.0]],
            dtype=np.float64,
        )
        recovered = view.virtual_to_raw(view.raw_to_virtual(raw))
        self.assertLess(float(np.max(np.abs(recovered - raw))), 1e-6)

    def test_box_corners_fit_inside_expanded_view(self) -> None:
        bbox = [360.0, 180.0, 760.0, 620.0]
        view = self.builder.build(bbox)
        virtual = view.raw_to_virtual(
            np.asarray(
                [[bbox[0], bbox[1]], [bbox[0], bbox[3]], [bbox[2], bbox[1]], [bbox[2], bbox[3]]],
                dtype=np.float64,
            )
        )
        self.assertTrue(np.all(virtual[:, 0] > 0.0))
        self.assertTrue(np.all(virtual[:, 0] < self.size[0] - 1.0))
        self.assertTrue(np.all(virtual[:, 1] > 0.0))
        self.assertTrue(np.all(virtual[:, 1] < self.size[1] - 1.0))

    def test_remap_preserves_expected_shape(self) -> None:
        view = self.builder.build([360.0, 180.0, 760.0, 620.0])
        source = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        # The exact virtual centre lies between output pixels, so use a small
        # constant neighbourhood to test the remap rather than one isolated
        # source pixel whose bilinear weight is necessarily fractional.
        source[397:404, 557:564] = [13, 113, 213]
        image = view.image(source)
        self.assertEqual(image.shape, (self.size[1], self.size[0], 3))
        center = image[int(round(view.cy)), int(round(view.cx))]
        self.assertLess(float(np.max(np.abs(center.astype(float) - source[400, 560]))), 5.0)

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            LocalPerspectiveModelInput(self.K, self.D, self.size, margin=1.0)
        with self.assertRaises(ValueError):
            self.builder.build([1.0, 1.0, 1.5, 1.5])


if __name__ == "__main__":
    unittest.main()
