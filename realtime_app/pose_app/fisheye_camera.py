"""OpenCV-style fisheye camera model for SMPL-X / skeleton 2D reprojection.

Only the projection direction (3D camera point -> raw pixel) is implemented, in
both numpy (reference) and torch (differentiable) form. The implementation
follows the standard OpenCV fisheye model and was verified against the
per-joint reprojection errors recorded in the frozen people_1 stereo results
(absolute max difference < 1e-9 px on all tested frames).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

import numpy as np

@dataclass(frozen=True)
class StereoFisheyeCalibration:
    K0: np.ndarray
    D0: np.ndarray
    K1: np.ndarray
    D1: np.ndarray
    R_cam0_to_cam1: np.ndarray
    T_cam0_to_cam1_mm: np.ndarray
    image_width: int
    image_height: int

    def left_center_px(self) -> np.ndarray:
        return np.asarray([self.K0[0, 2], self.K0[1, 2]])

    def right_center_px(self) -> np.ndarray:
        return np.asarray([self.K1[0, 2], self.K1[1, 2]])


def load_stereo_fisheye(calibration_dir: str | Path) -> StereoFisheyeCalibration:
    """Load stereo_fisheye.json plus cam0/cam1 intrinsics from a directory."""
    root = Path(calibration_dir).resolve()
    stereo_path = root / "stereo_fisheye.json"
    if not stereo_path.exists():
        raise FileNotFoundError(f"missing stereo calibration: {stereo_path}")
    stereo = json.loads(stereo_path.read_text(encoding="utf-8-sig"))
    cam0_path = root / "cam0_fisheye.json"
    cam1_path = root / "cam1_fisheye.json"
    if not cam0_path.exists() or not cam1_path.exists():
        raise FileNotFoundError("missing cam0/cam1 fisheye intrinsics")
    cam0 = json.loads(cam0_path.read_text(encoding="utf-8-sig"))
    cam1 = json.loads(cam1_path.read_text(encoding="utf-8-sig"))
    image_size = stereo["image_size"]
    return StereoFisheyeCalibration(
        K0=np.asarray(cam0["K"], dtype=np.float64),
        D0=np.asarray(cam0["D"], dtype=np.float64),
        K1=np.asarray(cam1["K"], dtype=np.float64),
        D1=np.asarray(cam1["D"], dtype=np.float64),
        R_cam0_to_cam1=np.asarray(stereo["R_cam0_to_cam1"], dtype=np.float64),
        T_cam0_to_cam1_mm=np.asarray(stereo["T_cam0_to_cam1_mm"], dtype=np.float64),
        image_width=int(image_size[0]),
        image_height=int(image_size[1]),
    )


def fisheye_project_numpy(points_camera_mm: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Project camera-frame points (mm) to raw pixels with the OpenCV fisheye model."""
    pts = np.asarray(points_camera_mm, dtype=np.float64)
    X, Y, Z = pts[..., 0], pts[..., 1], pts[..., 2]
    x = X / Z
    y = Y / Z
    r = np.sqrt(x * x + y * y)
    theta = np.arctan(r)
    k1, k2, k3, k4 = np.asarray(D, dtype=np.float64).reshape(-1)[:4]
    theta_d = theta * (1.0 + k1 * theta ** 2 + k2 * theta ** 4 + k3 * theta ** 6 + k4 * theta ** 8)
    scale = np.where(r > 1e-12, theta_d / np.maximum(r, 1e-12), 1.0)
    u = K[0, 0] * x * scale + K[0, 2]
    v = K[1, 1] * y * scale + K[1, 2]
    return np.stack([u, v], axis=-1)


def fisheye_project_torch(points_camera_m: "torch.Tensor", K: Any, D: Any) -> "torch.Tensor":
    """Differentiable fisheye projection (meters in, pixels out).

    Args:
        points_camera_m: (..., 3) camera-frame points in metres.
        K: 3x3 intrinsics tensor-like (numpy or torch).
        D: 4 distortion coefficients tensor-like.
    Returns:
        (..., 2) pixel coordinates.
    """
    import torch
    dtype = points_camera_m.dtype
    device = points_camera_m.device
    Kt = torch.as_tensor(np.asarray(K, dtype=np.float64), dtype=dtype, device=device)
    Dt = torch.as_tensor(np.asarray(D, dtype=np.float64), dtype=dtype, device=device).reshape(-1)[:4]
    X, Y, Z = points_camera_m[..., 0], points_camera_m[..., 1], points_camera_m[..., 2]
    x = X / Z
    y = Y / Z
    r = torch.sqrt(x * x + y * y)
    theta = torch.atan(r)
    theta_d = theta * (1.0 + Dt[0] * theta ** 2 + Dt[1] * theta ** 4 + Dt[2] * theta ** 6 + Dt[3] * theta ** 8)
    safe_r = torch.clamp(r, min=1e-12)
    scale = torch.where(r > 1e-12, theta_d / safe_r, torch.ones_like(theta_d))
    u = Kt[0, 0] * x * scale + Kt[0, 2]
    v = Kt[1, 1] * y * scale + Kt[1, 2]
    return torch.stack([u, v], dim=-1)


def world_to_cameras_torch_batch(
    points_world_m: "torch.Tensor",
    rotations_world_from_left: Any,
    translations_world_from_left_mm: Any,
    cal: StereoFisheyeCalibration,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Batched world -> left/right camera frame (metres).

    Args:
        points_world_m: (N, ..., 3) world-frame points in metres.
        rotations_world_from_left: (N, 3, 3) or list of N 3x3 matrices.
        translations_world_from_left_mm: (N, 3) or list of N length-3 vectors.
    Returns:
        (left_m, right_m) each (N, ..., 3).
    """
    import torch
    dtype = points_world_m.dtype
    device = points_world_m.device
    R = torch.as_tensor(np.asarray(rotations_world_from_left, dtype=np.float64), dtype=dtype, device=device)
    t = torch.as_tensor(np.asarray(translations_world_from_left_mm, dtype=np.float64), dtype=dtype, device=device) / 1000.0
    diff = points_world_m - t.reshape(-1, *([1] * (points_world_m.dim() - 2)), 3)
    left_m = torch.einsum("nij,n...j->n...i", R.transpose(-1, -2), diff)
    R01 = torch.as_tensor(cal.R_cam0_to_cam1, dtype=dtype, device=device)
    T01 = torch.as_tensor(cal.T_cam0_to_cam1_mm, dtype=dtype, device=device) / 1000.0
    right_m = torch.einsum("ij,n...j->n...i", R01, left_m) + T01
    return left_m, right_m


def world_to_left_numpy(points_world_mm: np.ndarray, rotation_world_from_left: Any, translation_world_from_left_mm: Any) -> np.ndarray:
    """Map world/ground points to the left-camera frame: X_l = R^T (X_w - t)."""
    pts = np.asarray(points_world_mm, dtype=np.float64)
    R = np.asarray(rotation_world_from_left, dtype=np.float64)
    t = np.asarray(translation_world_from_left_mm, dtype=np.float64)
    return (R.T @ (pts - t).T).T


def left_to_right_numpy(points_left_mm: np.ndarray, cal: StereoFisheyeCalibration) -> np.ndarray:
    pts = np.asarray(points_left_mm, dtype=np.float64)
    return (cal.R_cam0_to_cam1 @ pts.T + cal.T_cam0_to_cam1_mm.reshape(3, 1)).T


def pixel_per_mm_at_depth(fx: float, depth_mm: float) -> float:
    """Local linearization scale: pixels per millimetre at a given depth."""
    return fx / max(float(depth_mm), 1e-3)


def world_to_cameras_torch(
    points_world_m: "torch.Tensor",
    rotation_world_from_left: Any,
    translation_world_from_left_mm: Any,
    cal: StereoFisheyeCalibration,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Differentiable world -> left/right camera frame (both in metres).

    Args:
        points_world_m: (..., 3) world-frame points in metres (single frame).
        rotation_world_from_left: 3x3 world<-left rotation for this frame.
        translation_world_from_left_mm: world origin in left-camera mm.
        cal: stereo calibration.
    Returns:
        (left_points_m, right_points_m) each (..., 3).
    """
    import torch
    dtype = points_world_m.dtype
    device = points_world_m.device
    R = torch.as_tensor(np.asarray(rotation_world_from_left, dtype=np.float64), dtype=dtype, device=device)
    t = torch.as_tensor(np.asarray(translation_world_from_left_mm, dtype=np.float64), dtype=dtype, device=device) / 1000.0
    left_m = (R.transpose(-1, -2) @ (points_world_m - t).transpose(-1, -2)).transpose(-1, -2)
    R01 = torch.as_tensor(cal.R_cam0_to_cam1, dtype=dtype, device=device)
    T01 = torch.as_tensor(cal.T_cam0_to_cam1_mm, dtype=dtype, device=device) / 1000.0
    right_m = (R01 @ left_m.transpose(-1, -2) + T01.reshape(3, 1)).transpose(-1, -2)
    return left_m, right_m
