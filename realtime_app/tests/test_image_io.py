from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from pose_app.image_io import read_image, write_image


class ImagePathTests(unittest.TestCase):
    def test_unicode_path_preserves_color_and_grayscale_pixels(self):
        image = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "中文图像.png"
            self.assertTrue(write_image(path, image))
            np.testing.assert_array_equal(read_image(path), image)
            np.testing.assert_array_equal(read_image(path, cv2.IMREAD_GRAYSCALE),
                                          cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    def test_missing_empty_and_corrupt_unicode_files_remain_unreadable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "不可用.png"
            self.assertIsNone(read_image(path))
            path.write_bytes(b"")
            self.assertIsNone(read_image(path))
            path.write_bytes(b"not an image")
            self.assertIsNone(read_image(path))


if __name__ == "__main__":
    unittest.main()
