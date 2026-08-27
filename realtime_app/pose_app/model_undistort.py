from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np


class FisheyeModelInput:
    """Undistort a fisheye frame for inference and map detections back to raw pixels."""

    def __init__(
        self,
        K: np.ndarray,
        D: np.ndarray,
        image_size: tuple[int, int],
    ) -> None:
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError("image_size must be positive")
        self.K = np.asarray(K, dtype=np.float64)
        self.D = np.asarray(D, dtype=np.float64).reshape(-1, 1)
        if self.K.shape != (3, 3) or self.D.shape != (4, 1):
            raise ValueError("fisheye K must be 3x3 and D must contain 4 coefficients")
        self.image_size = (int(width), int(height))
        identity = np.eye(3, dtype=np.float64)
        self.map_x, self.map_y = cv2.fisheye.initUndistortRectifyMap(
            self.K,
            self.D,
            identity,
            self.K,
            self.image_size,
            cv2.CV_32FC1,
        )

    def image(self, raw_image: np.ndarray) -> np.ndarray:
        expected = (self.image_size[1], self.image_size[0])
        if raw_image.ndim != 3 or raw_image.shape[2] != 3:
            raise ValueError("raw_image must be HxWx3")
        if raw_image.shape[:2] != expected:
            raise ValueError(
                f"raw_image size {raw_image.shape[1::-1]} does not match {self.image_size}"
            )
        return cv2.remap(
            raw_image,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def undistorted_to_raw(self, points: np.ndarray) -> np.ndarray:
        """Map pixels in the undistorted image back to raw fisheye pixels."""

        points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        normalized = cv2.undistortPoints(points, self.K, np.zeros((4, 1), dtype=np.float64))
        raw = cv2.fisheye.distortPoints(normalized, self.K, self.D)
        return raw.reshape(-1, 2)

    def point_mapper(self) -> Callable[[float, float], tuple[float, float]]:
        def mapper(x: float, y: float) -> tuple[float, float]:
            point = self.undistorted_to_raw(np.asarray([[x, y]], dtype=np.float64))[0]
            return float(point[0]), float(point[1])

        return mapper
