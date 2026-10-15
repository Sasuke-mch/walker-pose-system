"""SMPL-H error-state motion tracking with raw stereo pixels as the only data term.

The skeleton is always J_coco @ SMPLH(vertices). Translation, SO(3) root and
VPoser coordinates carry velocity/acceleration; shape and non-body articulation
are frozen. No triangulated observations or Stage/contact inputs are consumed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from pose_app.fisheye_camera import StereoFisheyeCalibration, fisheye_project_numpy
from pose_app.smpl_coco_observation import (
    _install_legacy_smpl_pickle_compatibility, load_coco17_regressor,
)
from pose_app.smplx_fitting import load_vposer_explicit


@dataclass
class MotionPose:
    translation_m: np.ndarray
    root_rotation: np.ndarray
    latent: np.ndarray

    def copy(self) -> MotionPose:
        return MotionPose(self.translation_m.copy(), self.root_rotation.copy(), self.latent.copy())


def retract(pose: MotionPose, delta: np.ndarray) -> MotionPose:
    return MotionPose(pose.translation_m + delta[:3],
                      pose.root_rotation @ Rotation.from_rotvec(delta[3:6]).as_matrix(),
                      pose.latent + delta[6:])


def right_jacobian(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    angle = np.linalg.norm(vector)
    if angle < 1e-5:
        return np.eye(3) - 0.5 * skew + skew @ skew / 6.
    return np.eye(3) - (1 - np.cos(angle)) / angle**2 * skew + (angle - np.sin(angle)) / angle**3 * skew @ skew


class SMPLHObservationMap:
    """Actual 6890-vertex model, the installed VPoser and the current COCO regressor."""

    def __init__(self, initial_fit: Path, model_path: Path, regressor_path: Path,
                 vposer_dir: Path, mano_left: Path, mano_right: Path, device: str) -> None:
        import torch
        import smplx
        from smplx.utils import Struct
        self.torch = torch
        self.device = device
        _install_legacy_smpl_pickle_compatibility()
        with np.load(initial_fit, allow_pickle=False) as fitted:
            if fitted["pair_id"].tolist() != [0] or fitted["vposer_latent"].shape != (1, 32):
                raise ValueError("initial fit must contain frame 0 only")
            self.initial_pose = MotionPose(fitted["transl"][0].astype(float),
                                           Rotation.from_rotvec(fitted["global_orient"][0]).as_matrix(),
                                           fitted["vposer_latent"][0].astype(float))
            beta = fitted["betas"].copy()
            hands = [fitted[k][0].reshape(15, 3).copy() for k in ("left_hand_pose", "right_hand_pose")]
            fixed_body = Rotation.from_rotvec(fitted["body_pose"][0].reshape(21, 3)).as_matrix()
        with model_path.open("rb") as stream:
            data = pickle.load(stream, encoding="latin1")
        for side, asset in (("l", mano_left), ("r", mano_right)):
            with asset.open("rb") as stream:
                native = pickle.load(stream, encoding="latin1")
            data["hands_components" + side] = np.asarray(native["hands_components"], np.float32)
            data["hands_mean" + side] = np.asarray(native["hands_mean"], np.float32)
        self.model = smplx.SMPLHLayer(str(model_path), data_struct=Struct(**data), gender="male",
                                    use_pca=False, flat_hand_mean=True).to(device).eval()
        self.vposer, cfg, _ = load_vposer_explicit(vposer_dir, device)
        if int(cfg.model_params.latentD) != 32:
            raise ValueError("expected VPoser32")
        self.dimension = 38
        self.betas = torch.as_tensor(beta, dtype=torch.float32, device=device)
        self.hand_rotations = [torch.as_tensor(Rotation.from_rotvec(h).as_matrix()[None], dtype=torch.float32, device=device) for h in hands]
        # Body indices 9/10 are feet; 11/14 are neck/head. They lack target pixels here.
        self.fixed_body_indices = (9, 10, 11, 14)
        self.fixed_body = torch.as_tensor(fixed_body, dtype=torch.float32, device=device)
        self.regressor = torch.as_tensor(load_coco17_regressor(regressor_path), dtype=torch.float32, device=device)
        self.faces = np.asarray(self.model.faces)
        self.parents = self.model.parents.detach().cpu().numpy()
        for module in (self.model, self.vposer):
            for parameter in module.parameters():
                parameter.requires_grad = False

    def evaluate_batch(self, translations: np.ndarray, roots: np.ndarray, latents: np.ndarray,
                       mesh: bool = False) -> dict:
        torch = self.torch
        count = len(translations)
        with torch.inference_mode():
            z = torch.as_tensor(latents, dtype=torch.float32, device=self.device)
            decoded = self.vposer.decode(z)
            body = decoded["pose_body_matrot"].reshape(count, 21, 3, 3).clone()
            body[:, self.fixed_body_indices] = self.fixed_body[list(self.fixed_body_indices)]
            result = self.model(betas=self.betas.expand(count, -1),
                                transl=torch.as_tensor(translations, dtype=torch.float32, device=self.device),
                                global_orient=torch.as_tensor(roots, dtype=torch.float32, device=self.device),
                                body_pose=body,
                                left_hand_pose=self.hand_rotations[0].expand(count, -1, -1, -1),
                                right_hand_pose=self.hand_rotations[1].expand(count, -1, -1, -1))
            coco = torch.einsum("jv,bvc->bjc", self.regressor, result.vertices)
            output = {"coco_m": coco.cpu().numpy().astype(float),
                      "joints_m": result.joints[:, :52].cpu().numpy().astype(float),
                      "body_rotations": body.cpu().numpy()}
            if mesh:
                output["vertices_m"] = result.vertices.cpu().numpy()
        return output

    def evaluate(self, pose: MotionPose, mesh: bool = False) -> dict:
        return self.evaluate_batch(pose.translation_m[None], pose.root_rotation[None], pose.latent[None], mesh)


@dataclass(frozen=True)
class SMPLHMotionConfig:
    min_score: float = 0.2
    pixel_sigma_px: float = 8.
    huber_px: float = 25.
    max_stereo_epipolar_px: float = 35.
    min_observed_joints: int = 4
    max_prediction_s: float = 0.10
    derivative_warmup_s: float = 0.15
    max_position_std_m: float = 0.10
    max_derivative_reprojection_px: float = 25.
    translation_jerk_density: float = 0.25  # m²/s⁵
    rotation_jerk_density: float = 1.       # rad²/s⁵
    latent_jerk_density: float = 4.         # latent-coordinate²/s⁵
    max_translation_correction_m: float = 0.08
    max_root_correction_rad: float = 0.12
    max_latent_correction_norm: float = 0.40
    latent_prior_sigma: float = 1.5
    max_abs_latent: float = 6.
    derivative_step_s: float = 0.01

    def __post_init__(self) -> None:
        for key in self.__dataclass_fields__:
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if self.min_score > 1 or not 1 <= self.min_observed_joints <= 12:
            raise ValueError("invalid observation thresholds")


class SMPLHMotionTracker:
    """Error-state EKF on model parameters; no free Cartesian point variables."""

    def __init__(self, model: SMPLHObservationMap, cal: StereoFisheyeCalibration,
                 config: SMPLHMotionConfig | None = None) -> None:
        self.model, self.cal, self.config = model, cal, config or SMPLHMotionConfig()
        self.n = model.dimension
        self.pose = model.initial_pose.copy()
        self.velocity = np.zeros(self.n)
        self.acceleration = np.zeros(self.n)
        def grouped(t, r, z):
            return np.r_[np.repeat(t, 3), np.repeat(r, 3), np.repeat(z, self.n - 6)]
        sigma = np.r_[grouped(.10, .35, .5), grouped(.5, 1., 1.5), grouped(1., 2., 3.)]
        self.initial_covariance = np.diag(sigma**2)
        self.covariance = self.initial_covariance.copy()
        self.density = grouped(self.config.translation_jerk_density, self.config.rotation_jerk_density,
                               self.config.latent_jerk_density)
        self.last_time: float | None = None
        self.last_support: float | None = None
        self.last_joint_support = np.full(12, -np.inf)
        self.last_joint_good_fit = np.full(12, -np.inf)
        self.segment_start = 0.
        self.segment = 0
        self.active = True
        self.last_accepted_pose = self.pose.copy()
        self.last_point_jacobian: np.ndarray | None = None
        self.update_scale = 0.
        self.prediction_rejection: dict | None = None

    def _pixels(self, points: np.ndarray) -> np.ndarray | None:
        points = points[:, 5:17]
        other = points @ self.cal.R_cam0_to_cam1.T + self.cal.T_cam0_to_cam1_mm / 1000
        if not np.isfinite(points).all() or min(points[..., 2].min(), other[..., 2].min()) <= 1e-4:
            return None
        return np.stack((fisheye_project_numpy(points, self.cal.K0, self.cal.D0),
                         fisheye_project_numpy(other, self.cal.K1, self.cal.D1)), axis=1)

    def _linearize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        self.last_point_jacobian = None
        offsets = np.r_[np.zeros((1, self.n)), np.eye(self.n) * 1e-3, -np.eye(self.n) * 1e-3]
        data = self.model.evaluate_batch(self.pose.translation_m + offsets[:, :3],
                                        self.pose.root_rotation @ Rotation.from_rotvec(offsets[:, 3:6]).as_matrix(),
                                        self.pose.latent + offsets[:, 6:])
        pixels = self._pixels(data["coco_m"])
        if pixels is None:
            return None
        pixel_jacobian = ((pixels[1:1+self.n] - pixels[1+self.n:]) / 2e-3).reshape(self.n, -1).T
        point_jacobian = np.moveaxis((data["coco_m"][1:1+self.n, 5:17] - data["coco_m"][1+self.n:, 5:17]) / 2e-3, 0, -1)
        if not np.isfinite(pixel_jacobian).all():
            return None
        self.last_point_jacobian = point_jacobian
        return pixels[0], pixel_jacobian, point_jacobian

    def _predict(self, dt: float) -> None:
        n = self.n
        drive = self.velocity * dt + .5 * self.acceleration * dt**2
        F = np.eye(3 * n)
        F[:n, n:2*n] = np.eye(n) * dt
        F[:n, 2*n:] = np.eye(n) * .5 * dt**2
        F[n:2*n, 2*n:] = np.eye(n) * dt
        F[3:6, 3:6] = Rotation.from_rotvec(drive[3:6]).as_matrix().T
        jr = right_jacobian(drive[3:6])
        F[3:6, n+3:n+6] = jr * dt
        F[3:6, 2*n+3:2*n+6] = jr * .5 * dt**2
        block = np.array([[dt**5/20, dt**4/8, dt**3/6], [dt**4/8, dt**3/3, dt**2/2],
                          [dt**3/6, dt**2/2, dt]])
        Q = np.kron(block, np.diag(self.density))
        self.pose = retract(self.pose, drive)
        self.last_point_jacobian = None
        self.velocity += self.acceleration * dt
        self.covariance = F @ self.covariance @ F.T + Q
        self.covariance = (self.covariance + self.covariance.T) / 2

    def _supported_pixels(self, pixels: np.ndarray, epipolar: bool = True) -> tuple[np.ndarray, np.ndarray]:
        reasons = np.full((2, 12), "available", dtype=object)
        finite = np.isfinite(pixels).all(axis=2)
        bounds = (pixels[..., 0] >= 0) & (pixels[..., 0] < self.cal.image_width) & (pixels[..., 1] >= 0) & (pixels[..., 1] < self.cal.image_height)
        score = (pixels[..., 2] >= self.config.min_score) & (pixels[..., 2] <= 1)
        reasons[~finite] = "non_finite"
        reasons[finite & ~bounds] = "out_of_raw_image_bounds"
        reasons[finite & bounds & ~score] = "low_or_invalid_score"
        valid = finite & bounds & score
        if epipolar:
            tx, ty, tz = self.cal.T_cam0_to_cam1_mm
            essential = np.array([[0., -tz, ty], [tz, 0., -tx], [-ty, tx, 0.]]) @ self.cal.R_cam0_to_cam1
            for j in np.flatnonzero(valid.all(axis=0)):
                rays, jacobians = [], []
                for v, K, D in ((0, self.cal.K0, self.cal.D0), (1, self.cal.K1, self.cal.D1)):
                    raw = pixels[v, j, :2]
                    samples = np.r_[raw[None], raw + np.eye(2)*.25, raw - np.eye(2)*.25]
                    normalized = cv2.fisheye.undistortPoints(samples.reshape(-1, 1, 2), K, D).reshape(-1, 2)
                    rays.append(np.r_[normalized[0], 1.])
                    jacobians.append(((normalized[1:3] - normalized[3:])/.5).T)
                e = rays[1] @ essential @ rays[0]
                gradient = np.r_[jacobians[0].T @ (essential.T @ rays[1])[:2],
                                 jacobians[1].T @ (essential @ rays[0])[:2]]
                denominator = np.linalg.norm(gradient)
                distance = abs(e)/denominator if denominator > 1e-12 else np.inf
                if not np.isfinite(distance) or distance > self.config.max_stereo_epipolar_px:
                    valid[:, j] = False
                    reasons[:, j] = "stereo_epipolar_rejected"
        return valid, reasons

    def _update(self, pixels: np.ndarray, valid: np.ndarray, reasons: np.ndarray) -> tuple[bool, np.ndarray, str]:
        self.update_scale = 0.
        used = valid.copy()
        if np.any(used, axis=0).sum() < self.config.min_observed_joints:
            reasons[used] = "insufficient_joint_support"
            return False, np.zeros_like(used), "insufficient_joint_support"
        linearized = self._linearize()
        if linearized is None:
            reasons[used] = "model_projection_invalid"
            return False, np.zeros_like(used), "model_projection_invalid"
        prediction, J, _ = linearized
        n, P = self.n, self.covariance
        selected = np.repeat(used.ravel(), 2)
        # The VPoser regularizer is a pose prior, not another 3D measurement.
        prior_H = np.zeros((n-6, n))
        prior_H[:, 6:] = np.eye(n-6)
        Hq = np.r_[J[selected], prior_H]
        H = np.c_[Hq, np.zeros((len(Hq), 2*n))]
        residual = np.r_[(pixels[..., :2]-prediction).reshape(-1)[selected], -self.pose.latent]
        norms = np.linalg.norm((pixels[..., :2]-prediction)[used], axis=1)
        robust = np.repeat(np.minimum(1., self.config.huber_px/np.maximum(norms, 1e-12)), 2)
        variances = np.r_[self.config.pixel_sigma_px**2/robust,
                          np.repeat(self.config.latent_prior_sigma**2, n-6)]
        S = H @ P @ H.T + np.diag(variances)
        gain = np.linalg.solve(S, H @ P).T
        correction = gain @ residual
        scale = min(1., self.config.max_translation_correction_m/max(np.linalg.norm(correction[:3]), 1e-12),
                    self.config.max_root_correction_rad/max(np.linalg.norm(correction[3:6]), 1e-12),
                    self.config.max_latent_correction_norm/max(np.linalg.norm(correction[6:n]), 1e-12))
        prior_pose = self.pose.copy()
        precision = np.linalg.solve(P[:n, :n], np.eye(n))

        def objective(pose, displacement, projected):
            errors = np.linalg.norm((pixels[..., :2]-projected)[used], axis=1)
            data = np.sum(np.where(errors <= self.config.huber_px, .5*errors**2,
                                   self.config.huber_px*(errors-.5*self.config.huber_px))) / self.config.pixel_sigma_px**2
            return data + .5*displacement @ precision @ displacement + .5*np.sum(pose.latent**2)/self.config.latent_prior_sigma**2

        cost_before = objective(prior_pose, np.zeros(n), prediction)
        for factor in (scale, .5*scale, .25*scale):
            proposal = retract(prior_pose, correction[:n]*factor)
            if np.max(abs(proposal.latent)) > self.config.max_abs_latent:
                continue
            actual = self._pixels(self.model.evaluate(proposal)["coco_m"])
            if actual is None:
                continue
            cost = objective(proposal, correction[:n]*factor, actual[0])
            if np.isfinite(cost) and cost <= cost_before + 1e-6:
                self.pose = proposal
                self.update_scale = factor
                break
        if self.update_scale == 0:
            reasons[used] = "nonlinear_update_rejected"
            return False, np.zeros_like(used), "nonlinear_update_rejected"
        # A bounded correction uses a bounded gain in BOTH the mean and Joseph
        # covariance. Full information after a clipped step caused real drift.
        gain *= self.update_scale
        self.velocity += correction[n:2*n]*self.update_scale
        self.acceleration += correction[2*n:]*self.update_scale
        A = np.eye(3*n) - gain @ H
        posterior = A @ P @ A.T + (gain*variances) @ gain.T
        reset = np.eye(3*n)
        reset[3:6, 3:6] = right_jacobian(correction[3:6]*self.update_scale)
        self.covariance = reset @ posterior @ reset.T
        self.covariance = (self.covariance + self.covariance.T)/2
        self._linearize()  # Position uncertainty is evaluated at the accepted model.
        reasons[used] = "used_2d"
        return True, used, "updated"

    def _output(self, now: float, status: str, used: np.ndarray, reasons: np.ndarray,
                observed_pixels: np.ndarray | None = None) -> dict:
        data = self.model.evaluate(self.pose, mesh=True)
        pixels = self._pixels(data["coco_m"])
        available = self.active and pixels is not None
        if available and self.last_point_jacobian is None:
            self._linearize()
        points = data["coco_m"][0, 5:17]
        eps = self.config.derivative_step_s
        offsets = np.stack((self.velocity * eps + .5*self.acceleration*eps**2,
                            -self.velocity * eps + .5*self.acceleration*eps**2))
        nearby = self.model.evaluate_batch(self.pose.translation_m + offsets[:, :3],
                                          self.pose.root_rotation @ Rotation.from_rotvec(offsets[:, 3:6]).as_matrix(),
                                          self.pose.latent + offsets[:, 6:])["coco_m"][:, 5:17]
        velocity = (nearby[0] - nearby[1]) / (2*eps)
        acceleration = (nearby[0] + nearby[1] - 2*points) / eps**2
        if self.last_point_jacobian is None:
            sigma = np.full(12, np.inf)
        else:
            pcov = self.last_point_jacobian @ self.covariance[:self.n, :self.n] @ np.swapaxes(self.last_point_jacobian, 1, 2)
            sigma = np.sqrt(np.maximum(np.linalg.eigvalsh(pcov).max(axis=1), 0))
        age = now - self.last_joint_support
        pixel_fit_good = np.ones(12, bool)
        if observed_pixels is not None and pixels is not None:
            errors = np.linalg.norm(pixels[0] - observed_pixels[..., :2], axis=2)
            # A rejected current observation still reveals poor projection fit;
            # do not let an empty used mask silently turn it into a good fit.
            quality_views, _ = self._supported_pixels(observed_pixels, epipolar=False)
            worst = np.max(np.where(quality_views, errors, 0.), axis=0)
            pixel_fit_good = worst <= self.config.max_derivative_reprojection_px
            self.last_joint_good_fit[np.any(used, axis=0) & pixel_fit_good] = now
        else:
            self.last_joint_good_fit[np.any(used, axis=0)] = now
        derivative_valid = (available & (age <= self.config.max_prediction_s) & pixel_fit_good
                            & (now-self.last_joint_good_fit <= self.config.max_prediction_s)
                            & (now - self.segment_start >= self.config.derivative_warmup_s)
                            & (sigma <= self.config.max_position_std_m))
        derivative_reason = np.full(12, "available", dtype=object)
        derivative_reason[sigma > self.config.max_position_std_m] = "position_uncertainty"
        derivative_reason[(~pixel_fit_good) | (now-self.last_joint_good_fit > self.config.max_prediction_s)] = "poor_or_stale_pixel_fit"
        derivative_reason[age > self.config.max_prediction_s] = "no_recent_own_pixels"
        if now-self.segment_start < self.config.derivative_warmup_s:
            derivative_reason[:] = "derivative_warmup"
        if not available:
            derivative_reason[:] = "model_unavailable"
        raw_velocity, raw_acceleration = velocity.copy(), acceleration.copy()
        if not available:
            raw_velocity[:] = raw_acceleration[:] = np.nan
        velocity[~derivative_valid] = np.nan
        acceleration[~derivative_valid] = np.nan
        joint_status = np.where(np.any(used, axis=0), "updated", np.where(age <= self.config.max_prediction_s, "predicted", "inferred")).astype(object)
        if status in ("initialized", "reinitialized"):
            joint_status[:] = status
        if not available:
            joint_status[:] = "expired"
            points[:] = np.nan
            data["vertices_m"][:] = np.nan
        return {"status": status, "joint_status": joint_status, "used_views": used.copy(),
                "view_reason": reasons, "segment_id": self.segment, "pose_available": available,
                "position_mm": points * 1000, "velocity_mm_s": velocity * 1000,
                "acceleration_mm_s2": acceleration * 1000, "derivative_valid": derivative_valid,
                "derivative_reason": derivative_reason,
                "raw_velocity_mm_s": raw_velocity*1000, "raw_acceleration_mm_s2": raw_acceleration*1000,
                "position_std_m": sigma, "joint_observation_age_s": age,
                "projected_pixels": None if pixels is None else pixels[0],
                "translation_m": self.pose.translation_m.copy(), "root_rotation": self.pose.root_rotation.copy(),
                "vposer_latent": self.pose.latent.copy(), "generalized_velocity": self.velocity.copy(),
                "generalized_acceleration": self.acceleration.copy(), "covariance_diag": np.diag(self.covariance).copy(),
                "update_scale": self.update_scale,
                "prediction_rejection": self.prediction_rejection,
                "vertices_m": data["vertices_m"][0], "joints_m": data["joints_m"][0],
                "coco_m": data["coco_m"][0], "body_rotations": data["body_rotations"][0]}

    def step(self, now: float, left: np.ndarray, right: np.ndarray) -> dict:
        self.update_scale = 0.
        self.prediction_rejection = None
        pixels = np.asarray([left, right], dtype=float)
        if pixels.shape != (2, 12, 3):
            raise ValueError("expected two views of 12 raw-pixel [x,y,score] points")
        if not np.isfinite(now) or (self.last_time is not None and now <= self.last_time):
            raise ValueError("timestamps must be finite and increase strictly")
        valid, reasons = self._supported_pixels(pixels)
        if self.last_time is None:
            self.last_time = self.last_support = self.segment_start = now
            self.last_joint_support[np.any(valid, axis=0)] = now
            reasons[valid] = "first_frame_initialization_pixels_not_reused"
            return self._output(now, "initialized", np.zeros_like(valid), reasons, pixels)
        restarted = False
        if now - self.last_support > self.config.max_prediction_s:
            self.active = False
        if self.active:
            self._predict(now - self.last_time)
            latent = self.pose.latent
            if not np.isfinite(latent).all() or np.any(abs(latent) > self.config.max_abs_latent):
                # The update's proposal bound alone does not bound q+v*dt+a*dt²/2.
                # Retain the rejected prediction, then use the existing current-
                # pixel recovery path. Clipping would invent a valid pose.
                self.prediction_rejection = {
                    "reason": "latent_prediction_out_of_domain",
                    "predicted_latent": [float(value) if np.isfinite(value) else None for value in latent],
                    "max_abs_latent": self.config.max_abs_latent,
                }
                self.active = False
        if not self.active:
            self.pose = self.last_accepted_pose.copy()
            self.velocity[:] = self.acceleration[:] = 0
            self.covariance = self.initial_covariance.copy()
            self.last_point_jacobian = None
            if np.any(valid, axis=0).sum() < self.config.min_observed_joints:
                self.last_time = now
                return self._output(now, "expired", np.zeros_like(valid), reasons, pixels)
            restarted = True
        accepted, used, status = self._update(pixels, valid, reasons)
        if accepted:
            self.active = True
            self.last_support = now
            self.last_joint_support[np.any(used, axis=0)] = now
            self.last_accepted_pose = self.pose.copy()
            if restarted:
                self.segment += 1
                self.segment_start = now
                status = "reinitialized"
        elif self._pixels(self.model.evaluate(self.pose)["coco_m"]) is None:
            self.active = False
            status = "expired"
        elif self.active:
            status = "predicted"
        else:
            status = "expired"
        self.last_time = now
        return self._output(now, status, used, reasons, pixels)
