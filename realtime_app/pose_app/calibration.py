from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


_VALID_MODELS = {"pinhole", "fisheye"}


def _matrix(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name}形状应为{shape}，实际为{array.shape}。")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}包含非有限数值。")
    return array


def _distortion(value: Any, model: str, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if model == "fisheye" and array.size != 4:
        raise ValueError(f"{name}在fisheye模型下必须包含4个系数。")
    if model == "pinhole" and array.size not in {4, 5, 8, 12, 14}:
        raise ValueError(
            f"{name}在pinhole模型下应包含4、5、8、12或14个系数，实际为{array.size}。"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}包含非有限数值。")
    return array


def _scale_intrinsics(
    matrix: np.ndarray,
    calibrated_size: tuple[int, int],
    runtime_size: tuple[int, int],
    camera_name: str,
) -> np.ndarray:
    if calibrated_size == runtime_size:
        return matrix.copy()
    cw, ch = calibrated_size
    rw, rh = runtime_size
    if min(cw, ch, rw, rh) <= 0:
        raise ValueError("图像尺寸必须为正数。")
    sx = rw / cw
    sy = rh / ch
    if abs(sx - sy) > 1e-3:
        raise ValueError(
            f"{camera_name}运行分辨率{runtime_size}与标定分辨率{calibrated_size}宽高缩放比例不一致。"
            "请使用与标定相同的宽高比重新采集或重新标定。"
        )
    scaled = matrix.copy()
    scaled[0, 0] *= sx
    scaled[0, 2] *= sx
    scaled[1, 1] *= sy
    scaled[1, 2] *= sy
    return scaled


@dataclass(frozen=True)
class StereoCalibration:
    """Stereo calibration using OpenCV's left-to-right extrinsic convention.

    R and T satisfy X_right = R @ X_left + T.  Triangulated coordinates are
    therefore expressed in the left-camera coordinate system and use the same
    length unit as T.
    """

    camera_model: str
    left_image_size: tuple[int, int]
    right_image_size: tuple[int, int]
    left_K: np.ndarray
    left_D: np.ndarray
    right_K: np.ndarray
    right_D: np.ndarray
    R: np.ndarray
    T: np.ndarray
    length_unit: str = "meter"

    @classmethod
    def load(cls, path: str | Path) -> "StereoCalibration":
        file_path = Path(path).resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"双目标定文件不存在：{file_path}")
        raw = json.loads(file_path.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            raise ValueError("双目标定文件根节点必须是JSON对象。")
        model = str(raw.get("camera_model", "pinhole")).lower()
        if model not in _VALID_MODELS:
            raise ValueError(f"camera_model只支持{sorted(_VALID_MODELS)}，实际为{model!r}。")

        left = raw.get("left")
        right = raw.get("right")
        stereo = raw.get("stereo")
        if not isinstance(left, dict) or not isinstance(right, dict) or not isinstance(stereo, dict):
            raise ValueError("标定文件必须包含left、right和stereo三个对象。")

        left_size = tuple(int(v) for v in left.get("image_size", []))
        right_size = tuple(int(v) for v in right.get("image_size", []))
        if len(left_size) != 2 or min(left_size) <= 0:
            raise ValueError("left.image_size应为[width, height]。")
        if len(right_size) != 2 or min(right_size) <= 0:
            raise ValueError("right.image_size应为[width, height]。")

        R = _matrix(stereo.get("R"), (3, 3), "stereo.R")
        T = np.asarray(stereo.get("T"), dtype=np.float64).reshape(-1)
        if T.shape != (3,) or not np.all(np.isfinite(T)):
            raise ValueError("stereo.T应为包含3个有限数值的向量。")
        if float(np.linalg.norm(T)) <= 0:
            raise ValueError("stereo.T不能为零向量。")
        det = float(np.linalg.det(R))
        if not np.isclose(det, 1.0, atol=1e-2):
            raise ValueError(f"stereo.R不像有效旋转矩阵，det(R)={det:.6f}。")

        return cls(
            camera_model=model,
            left_image_size=(left_size[0], left_size[1]),
            right_image_size=(right_size[0], right_size[1]),
            left_K=_matrix(left.get("K"), (3, 3), "left.K"),
            left_D=_distortion(left.get("D"), model, "left.D"),
            right_K=_matrix(right.get("K"), (3, 3), "right.K"),
            right_D=_distortion(right.get("D"), model, "right.D"),
            R=R,
            T=T,
            length_unit=str(raw.get("length_unit", "meter")),
        )

    def for_runtime_sizes(
        self,
        left_size: tuple[int, int],
        right_size: tuple[int, int],
    ) -> "StereoCalibration":
        return StereoCalibration(
            camera_model=self.camera_model,
            left_image_size=left_size,
            right_image_size=right_size,
            left_K=_scale_intrinsics(
                self.left_K, self.left_image_size, left_size, "左相机"
            ),
            left_D=self.left_D.copy(),
            right_K=_scale_intrinsics(
                self.right_K, self.right_image_size, right_size, "右相机"
            ),
            right_D=self.right_D.copy(),
            R=self.R.copy(),
            T=self.T.copy(),
            length_unit=self.length_unit,
        )

    @property
    def baseline(self) -> float:
        return float(np.linalg.norm(self.T))

    @property
    def essential_matrix(self) -> np.ndarray:
        tx, ty, tz = self.T
        skew = np.asarray(
            [[0.0, -tz, ty], [tz, 0.0, -tx], [-ty, tx, 0.0]],
            dtype=np.float64,
        )
        return skew @ self.R

    def undistort_normalized(
        self, points: np.ndarray, side: str
    ) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        if side == "left":
            K, D = self.left_K, self.left_D
        elif side == "right":
            K, D = self.right_K, self.right_D
        else:
            raise ValueError("side必须是left或right。")
        if self.camera_model == "fisheye":
            result = cv2.fisheye.undistortPoints(points, K, D.reshape(-1, 1))
        else:
            result = cv2.undistortPoints(points, K, D)
        return result.reshape(-1, 2)

    def project_left(self, xyz_left: np.ndarray) -> np.ndarray:
        return self._project(xyz_left, "left")

    def project_right(self, xyz_left: np.ndarray) -> np.ndarray:
        return self._project(xyz_left, "right")

    def _project(self, xyz_left: np.ndarray, side: str) -> np.ndarray:
        points = np.asarray(xyz_left, dtype=np.float64).reshape(-1, 1, 3)
        if side == "left":
            rotation = np.zeros((3, 1), dtype=np.float64)
            translation = np.zeros((3, 1), dtype=np.float64)
            K, D = self.left_K, self.left_D
        elif side == "right":
            rotation, _ = cv2.Rodrigues(self.R)
            translation = self.T.reshape(3, 1)
            K, D = self.right_K, self.right_D
        else:
            raise ValueError("side必须是left或right。")
        if self.camera_model == "fisheye":
            image_points, _ = cv2.fisheye.projectPoints(
                points, rotation, translation, K, D.reshape(-1, 1)
            )
        else:
            image_points, _ = cv2.projectPoints(
                points, rotation, translation, K, D
            )
        return image_points.reshape(-1, 2)
