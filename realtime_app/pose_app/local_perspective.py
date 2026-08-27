"""Per-person virtual pinhole views for fisheye pose inference.

The actual stereo geometry never leaves the calibrated raw fisheye pixel
system.  A ``LocalPerspectiveView`` is only an invertible model-input view:

raw fisheye pixels -> local pinhole image -> 2D pose model
                                      -> raw fisheye pixels

This is deliberately not a whole-image undistortion.  The virtual camera is
aimed at the current person, so the person stays near the middle of a normal
perspective view even when they are near the edge of the fisheye image.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import cv2
import numpy as np


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length < 1e-12:
        raise ValueError(f"{name} is degenerate")
    return vector / length


def _validate_camera(K: np.ndarray, D: np.ndarray, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be positive")
    matrix = np.asarray(K, dtype=np.float64)
    distortion = np.asarray(D, dtype=np.float64).reshape(-1, 1)
    if matrix.shape != (3, 3) or distortion.shape != (4, 1):
        raise ValueError("fisheye K must be 3x3 and D must contain 4 coefficients")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(distortion)):
        raise ValueError("fisheye parameters must be finite")
    return matrix, distortion


@dataclass(frozen=True)
class LocalPerspectiveView:
    """One reversible virtual pinhole view of a region in a fisheye image."""

    K_fisheye: np.ndarray
    D_fisheye: np.ndarray
    source_size: tuple[int, int]
    output_size: tuple[int, int]
    camera_from_virtual: np.ndarray
    focal_px: float
    map_x: np.ndarray
    map_y: np.ndarray

    @property
    def cx(self) -> float:
        return (self.output_size[0] - 1.0) * 0.5

    @property
    def cy(self) -> float:
        return (self.output_size[1] - 1.0) * 0.5

    def image(self, raw_image: np.ndarray) -> np.ndarray:
        expected = (self.source_size[1], self.source_size[0])
        if raw_image.ndim != 3 or raw_image.shape[2] != 3:
            raise ValueError("raw_image must be HxWx3")
        if raw_image.shape[:2] != expected:
            raise ValueError(
                f"raw_image size {raw_image.shape[1::-1]} does not match {self.source_size}"
            )
        return cv2.remap(
            raw_image,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def virtual_to_raw(self, points: np.ndarray) -> np.ndarray:
        """Map virtual pinhole pixels back to calibrated raw fisheye pixels."""

        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        virtual_rays = np.column_stack(
            [
                (points[:, 0] - self.cx) / self.focal_px,
                (points[:, 1] - self.cy) / self.focal_px,
                np.ones(len(points), dtype=np.float64),
            ]
        )
        camera_rays = virtual_rays @ self.camera_from_virtual.T
        projected, _ = cv2.fisheye.projectPoints(
            camera_rays.reshape(-1, 1, 3),
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            self.K_fisheye,
            self.D_fisheye,
        )
        return projected.reshape(-1, 2)

    def raw_to_virtual(self, points: np.ndarray) -> np.ndarray:
        """Map raw fisheye pixels to virtual pinhole pixels.

        Points behind the virtual camera are rejected rather than silently
        divided by a non-positive depth.
        """

        points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        normalized = cv2.fisheye.undistortPoints(points, self.K_fisheye, self.D_fisheye)
        camera_rays = np.column_stack(
            [normalized.reshape(-1, 2), np.ones(len(points), dtype=np.float64)]
        )
        virtual_rays = camera_rays @ self.camera_from_virtual
        if np.any(virtual_rays[:, 2] <= 1e-9):
            raise ValueError("raw point lies behind the local virtual camera")
        return np.column_stack(
            [
                self.cx + self.focal_px * virtual_rays[:, 0] / virtual_rays[:, 2],
                self.cy + self.focal_px * virtual_rays[:, 1] / virtual_rays[:, 2],
            ]
        )

    def point_mapper(self) -> Callable[[float, float], tuple[float, float]]:
        def mapper(x: float, y: float) -> tuple[float, float]:
            raw = self.virtual_to_raw(np.asarray([[x, y]], dtype=np.float64))[0]
            return float(raw[0]), float(raw[1])

        return mapper


class LocalPerspectiveModelInput:
    """Build person-centred local pinhole views from a calibrated fisheye image."""

    def __init__(
        self,
        K: np.ndarray,
        D: np.ndarray,
        image_size: tuple[int, int],
        *,
        output_size: tuple[int, int] | None = None,
        margin: float = 1.35,
        min_horizontal_fov_deg: float = 35.0,
        max_horizontal_fov_deg: float = 130.0,
    ) -> None:
        self.K, self.D = _validate_camera(K, D, image_size)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.output_size = output_size or self.image_size
        if min(self.output_size) <= 0:
            raise ValueError("output_size must be positive")
        if margin <= 1.0:
            raise ValueError("margin must be greater than 1.0")
        if not 0 < min_horizontal_fov_deg < max_horizontal_fov_deg < 180:
            raise ValueError("virtual horizontal FOV limits must satisfy 0 < min < max < 180")
        self.margin = float(margin)
        self.min_horizontal_fov_deg = float(min_horizontal_fov_deg)
        self.max_horizontal_fov_deg = float(max_horizontal_fov_deg)

    def _rays_from_raw(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        normalized = cv2.fisheye.undistortPoints(points, self.K, self.D).reshape(-1, 2)
        return np.column_stack([normalized, np.ones(len(normalized), dtype=np.float64)])

    @staticmethod
    def _camera_from_virtual(center_ray: np.ndarray) -> np.ndarray:
        """Return a right/down/forward virtual basis in camera coordinates."""

        forward = _unit(center_ray, "center ray")
        camera_down = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        down = camera_down - forward * float(np.dot(camera_down, forward))
        # Only the extreme top/bottom fisheye directions make camera-down
        # unusable.  Use camera-right there; this prevents NaNs in diagnostics.
        if np.linalg.norm(down) < 1e-8:
            fallback = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            down = fallback - forward * float(np.dot(fallback, forward))
        down = _unit(down, "virtual down")
        right = _unit(np.cross(down, forward), "virtual right")
        down = _unit(np.cross(forward, right), "virtual down")
        return np.column_stack([right, down, forward])

    def build(self, bbox_xyxy: list[float] | tuple[float, float, float, float], *, support_points: np.ndarray | None = None) -> LocalPerspectiveView:
        """Create a view that contains the supplied raw-fisheye bounding box.

        The field of view is fitted from all four box corners in the local
        pinhole plane and then expanded by ``margin``.  It is clamped to a
        documented range so a faulty 2D box cannot create a nearly singular
        virtual camera.
        """

        if len(bbox_xyxy) < 4:
            raise ValueError("bbox must contain [x1, y1, x2, y2]")
        width, height = self.image_size
        x1, y1, x2, y2 = (float(value) for value in bbox_xyxy[:4])
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            raise ValueError("bbox must be finite")
        x1, x2 = sorted((max(0.0, min(width - 1.0, x1)), max(0.0, min(width - 1.0, x2))))
        y1, y2 = sorted((max(0.0, min(height - 1.0, y1)), max(0.0, min(height - 1.0, y2))))
        if x2 - x1 < 2.0 or y2 - y1 < 2.0:
            raise ValueError("bbox is too small after clipping to the source image")

        corners = np.asarray([[x1, y1], [x1, y2], [x2, y1], [x2, y2]], dtype=np.float64)
        support = corners
        using_keypoint_support = False
        if support_points is not None:
            candidate_support = np.asarray(support_points, dtype=np.float64).reshape(-1, 2)
            in_bounds = (
                np.isfinite(candidate_support).all(axis=1)
                & (candidate_support[:, 0] >= 0.0)
                & (candidate_support[:, 0] <= width - 1.0)
                & (candidate_support[:, 1] >= 0.0)
                & (candidate_support[:, 1] <= height - 1.0)
            )
            candidate_support = candidate_support[in_bounds]
            if len(candidate_support) >= 4:
                support = candidate_support
                using_keypoint_support = True
        camera_rays = self._rays_from_raw(support)
        # When only a detector box is available, its geometric centre is the
        # explicit target: that point must land at the virtual-image centre.
        # Averaging the four corner rays is not equivalent for fisheye pixels
        # and can visibly displace a centred subject.  With four or more
        # high-confidence pose points, keep the established support-ray aim so
        # that the view follows the observed body rather than an imperfect box.
        if using_keypoint_support:
            center_ray = np.sum(camera_rays, axis=0)
        else:
            bbox_center = np.asarray(
                [[(x1 + x2) * 0.5, (y1 + y2) * 0.5]], dtype=np.float64
            )
            center_ray = self._rays_from_raw(bbox_center)[0]
        camera_from_virtual = self._camera_from_virtual(center_ray)
        local_rays = camera_rays @ camera_from_virtual
        if np.any(local_rays[:, 2] <= 1e-6):
            raise ValueError("subject support spans directions behind the local virtual camera")
        local_x = local_rays[:, 0] / local_rays[:, 2]
        local_y = local_rays[:, 1] / local_rays[:, 2]
        span_x = max(1e-6, float(local_x.max() - local_x.min()))
        span_y = max(1e-6, float(local_y.max() - local_y.min()))
        output_width, output_height = self.output_size
        focal = min(
            output_width / (self.margin * span_x),
            output_height / (self.margin * span_y),
        )
        min_focal = output_width / (2.0 * math.tan(math.radians(self.max_horizontal_fov_deg) * 0.5))
        max_focal = output_width / (2.0 * math.tan(math.radians(self.min_horizontal_fov_deg) * 0.5))
        focal = float(np.clip(focal, min_focal, max_focal))

        xx, yy = np.meshgrid(
            np.arange(output_width, dtype=np.float64),
            np.arange(output_height, dtype=np.float64),
        )
        virtual_rays = np.stack(
            [
                (xx - (output_width - 1.0) * 0.5) / focal,
                (yy - (output_height - 1.0) * 0.5) / focal,
                np.ones_like(xx),
            ],
            axis=-1,
        ).reshape(-1, 3)
        camera_rays = virtual_rays @ camera_from_virtual.T
        projected, _ = cv2.fisheye.projectPoints(
            camera_rays.reshape(-1, 1, 3),
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            self.K,
            self.D,
        )
        raw_pixels = projected.reshape(output_height, output_width, 2)
        return LocalPerspectiveView(
            K_fisheye=self.K,
            D_fisheye=self.D,
            source_size=self.image_size,
            output_size=self.output_size,
            camera_from_virtual=camera_from_virtual,
            focal_px=focal,
            map_x=raw_pixels[..., 0].astype(np.float32),
            map_y=raw_pixels[..., 1].astype(np.float32),
        )
