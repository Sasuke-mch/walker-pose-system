"""Differentiable OpenCV equidistant raw-fisheye observations, never GT roots.

Reference: https://docs.opencv.org/4.13.0/db/d58/group__calib3d__fisheye.html
Conservative front-hemisphere and raw image bounds are explicit projection validity.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from realtime_app.learning_flow_mesh.rotations import validate_rotation_matrices


@dataclass(frozen=True)
class Camera:
    instance_id: str
    role: str
    K: torch.Tensor
    D: torch.Tensor
    R: torch.Tensor
    T: torch.Tensor
    image_size: tuple

    def validate(self):
        if self.role not in ("LEFT", "RIGHT") or not self.instance_id:
            raise ValueError("physical camera identity and role required")
        if self.K.shape != (3, 3) or self.D.shape != (4,) or self.T.shape != (3,):
            raise ValueError("camera shape mismatch")
        validate_rotation_matrices(self.R)
        if not all(torch.isfinite(v).all() for v in (self.K, self.D, self.T)):
            raise ValueError("nonfinite camera")
        if self.K[0, 0] <= 0 or self.K[1, 1] <= 0 or min(self.image_size) <= 0:
            raise ValueError("invalid focal/image dimensions")
        if not torch.allclose(self.K[2], self.K.new_tensor([0, 0, 1])):
            raise ValueError("invalid intrinsic homogeneous row")
        return self


def project(points, camera):
    camera.validate()
    xyz = points @ camera.R.to(points).T + camera.T.to(points)
    # r>0 avoids the derivative singularity of norm at the optical axis.
    r = (xyz[..., :2].square().sum(-1) + 1e-24).sqrt()
    theta = torch.atan2(r, xyz[..., 2])
    polynomial = torch.ones_like(theta)
    for i, coefficient in enumerate(camera.D.to(points)):
        polynomial = polynomial + coefficient * theta.pow(2*(i+1))
    distorted = xyz[..., :2] * (theta*polynomial/r)[..., None]
    K = camera.K.to(points)
    uv = torch.stack((K[0, 0]*distorted[..., 0] + K[0, 1]*distorted[..., 1] + K[0, 2],
                      K[1, 1]*distorted[..., 1] + K[1, 2]), -1)
    w, h = camera.image_size
    valid = (torch.isfinite(uv).all(-1) & torch.isfinite(points).all(-1) & (xyz[..., 2] > 1e-8)
             & (uv[..., 0] >= 0) & (uv[..., 0] < w) & (uv[..., 1] >= 0) & (uv[..., 1] < h))
    return uv, valid


@dataclass(frozen=True)
class NoiseModel:
    precision: torch.Tensor
    source: str
    calibrated: bool = False

    def validate(self):
        if self.precision.shape != (2, 17, 2, 2) or not self.source:
            raise ValueError("explicit per-view COCO17 pixel precision/source required")
        if not torch.isfinite(self.precision).all() or not torch.allclose(self.precision, self.precision.transpose(-1, -2)):
            raise ValueError("finite symmetric noise precision required")
        if (torch.linalg.eigvalsh(self.precision) <= 0).any():
            raise ValueError("noise precision must be positive definite")
        return self

    @classmethod
    def proxy(cls, sigma_px=10.0):
        if sigma_px <= 0:
            raise ValueError("positive proxy pixel sigma required")
        return cls(torch.eye(2).expand(2, 17, 2, 2).clone()/sigma_px**2,
                   f"proxy-only fixed isotropic sigma={sigma_px}px; not detection-confidence covariance")


def check_rig(obs, cameras):
    obs.validate()
    if len(cameras) != 2 or tuple(c.instance_id for c in cameras) != obs.camera_ids:
        raise ValueError("PnP camera identities mismatch")
    if tuple(c.role for c in cameras) != ("LEFT", "RIGHT"):
        raise ValueError("calibration role order mismatch")


def observation_errors(points, obs, cameras, noise, *, huber=3.0):
    """K absolute candidate surface points; admitted observation set stays fixed."""
    check_rig(obs, cameras)
    noise.validate()
    if points.ndim != 3 or points.shape[1:] != (17, 3) or huber <= 0:
        raise ValueError("COCO17 surface points and positive Huber threshold required")
    predictions, valid = zip(*(project(points, camera) for camera in cameras))
    uv, projection_valid = torch.stack(predictions, 1), torch.stack(valid, 1)
    admitted = obs.admitted()
    clean_obs = torch.where(admitted[..., None], obs.xy.to(points), 0)
    difference = torch.where(admitted[None, ..., None], uv - clean_obs, 0)
    square = torch.einsum("kvji,vjil,kvjl->kvj", difference, noise.precision.to(points), difference)
    norm = square.clamp_min(0).sqrt()
    robust = torch.where(norm <= huber, .5*square, huber*(norm-.5*huber))
    usable = projection_valid | ~admitted[None]
    per_joint = robust.sum(1) / admitted.sum(0).clamp_min(1)
    per_joint = torch.where(admitted.any(0)[None], per_joint, torch.inf)
    per_joint = torch.where(usable.all(1), per_joint, torch.inf)
    total = robust.sum((1, 2)) / admitted.sum().clamp_min(1)
    total = torch.where(usable.all((1, 2)) & admitted.any(), total, torch.inf)
    return total, per_joint, uv, projection_valid


def load_rig(stereo_path, *, camera_ids, dtype=torch.float64):
    """Resolve existing project calibration and convert extrinsic mm to metres."""
    path = Path(stereo_path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data["model"] != "opencv_fisheye_stereo" or data["transform_convention"] != "X_cam1 = R_cam0_to_cam1 * X_cam0 + T_cam0_to_cam1":
        raise ValueError("unreviewed stereo calibration convention")
    app_root = path.parent.parent.parent
    cameras = []
    for i, role in enumerate(("LEFT", "RIGHT")):
        declared = Path(data[f"cam{i}_intrinsics"])
        local_path = path.parent / declared
        intrinsic_path = local_path if local_path.is_file() else app_root / declared
        intrinsics = json.loads(intrinsic_path.read_text(encoding="utf-8-sig"))
        K = torch.tensor(intrinsics["K"], dtype=dtype)
        D = torch.tensor(intrinsics["D"], dtype=dtype).flatten()
        R = torch.eye(3, dtype=dtype) if i == 0 else torch.tensor(data["R_cam0_to_cam1"], dtype=dtype)
        T = torch.zeros(3, dtype=dtype) if i == 0 else torch.tensor(data["T_cam0_to_cam1_mm"], dtype=dtype)/1000
        cameras.append(Camera(camera_ids[i], role, K, D, R, T, tuple(data["image_size"])).validate())
    return tuple(cameras)
