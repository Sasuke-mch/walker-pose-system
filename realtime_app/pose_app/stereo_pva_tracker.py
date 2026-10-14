"""Causal stereo pixel-feedback EKF for a joint's [position, velocity, acceleration].

p is the joint position in the current left camera; v = d(p_C(t))/dt and
a = d²(p_C(t))/dt² include camera motion. No world motion or forces are inferred.
Triangulation gates the stereo pair and seeds a new track; only pixels update
an existing track. The model keeps short predictions and explicit track breaks.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pose_app.fisheye_camera import StereoFisheyeCalibration, fisheye_project_numpy


@dataclass(frozen=True)
class StereoPVAConfig:
    min_score: float = 0.20
    max_reprojection_px: float = 25.0
    min_ray_angle_deg: float = 1.0
    pixel_sigma_px: float = 4.0
    innovation_gate_squared: float = 64.0
    jerk_spectral_density_mm2_s5: float = 1e8
    initial_position_sigma_mm: float = 100.0
    initial_velocity_sigma_mm_s: float = 1000.0
    initial_acceleration_sigma_mm_s2: float = 5000.0
    max_prediction_s: float = 0.5

    def __post_init__(self) -> None:
        if not np.isfinite(self.min_score) or not 0 <= self.min_score <= 1:
            raise ValueError("min_score must be in [0,1]")
        for name in self.__dataclass_fields__:
            if name != "min_score" and (not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.min_ray_angle_deg >= 90:
            raise ValueError("min_ray_angle_deg must be below 90")


def constant_acceleration_transition(dt: float) -> np.ndarray:
    """9x9 transition in block order [p_x,y,z, v_x,y,z, a_x,y,z]."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    eye = np.eye(3)
    return np.block([[eye, dt * eye, 0.5 * dt**2 * eye],
                     [np.zeros((3, 3)), eye, dt * eye],
                     [np.zeros((3, 6)), eye]])


def continuous_jerk_covariance(dt: float, spectral_density_mm2_s5: float) -> np.ndarray:
    """Q for da = sqrt(q) dW. q has units mm²/s⁵, not (mm/s³)²."""
    if not np.isfinite([dt, spectral_density_mm2_s5]).all() or min(dt, spectral_density_mm2_s5) <= 0:
        raise ValueError("dt and spectral density must be finite and positive")
    block = np.array([[dt**5 / 20, dt**4 / 8, dt**3 / 6],
                      [dt**4 / 8, dt**3 / 3, dt**2 / 2],
                      [dt**3 / 6, dt**2 / 2, dt]])
    return np.kron(block, np.eye(3)) * spectral_density_mm2_s5


class StereoPVATracker:
    """Independent joint filters; initial p comes from a first-frame-only body fit.

    Missing, geometrically rejected and innovation-rejected observations never
    update the state. A gap longer than max_prediction_s invalidates the track;
    the next valid stereo pair starts a new segment with zero v/a priors.
    Detection scores are a gate, not a calibrated inverse covariance.
    """

    def __init__(self, calibration: StereoFisheyeCalibration, joint_count: int = 12,
                 config: StereoPVAConfig | None = None) -> None:
        if joint_count <= 0:
            raise ValueError("joint_count must be positive")
        self.calibration = calibration
        self.config = config or StereoPVAConfig()
        self.joint_count = int(joint_count)
        self.state = np.full((joint_count, 9), np.nan)
        self.covariance = np.full((joint_count, 9, 9), np.nan)
        self.active = np.zeros(joint_count, dtype=bool)
        self.segment_id = np.full(joint_count, -1, dtype=int)
        self.last_support_s = np.full(joint_count, np.nan)
        self.last_time_s: float | None = None
        self.P0 = np.diag(np.repeat([
            self.config.initial_position_sigma_mm**2,
            self.config.initial_velocity_sigma_mm_s**2,
            self.config.initial_acceleration_sigma_mm_s2**2,
        ], 3))

    def _project(self, point: np.ndarray) -> np.ndarray | None:
        other = self.calibration.R_cam0_to_cam1 @ point + self.calibration.T_cam0_to_cam1_mm
        if not np.isfinite(point).all() or min(point[2], other[2]) <= 1e-6:
            return None
        pixels = np.concatenate((
            fisheye_project_numpy(point[None], self.calibration.K0, self.calibration.D0)[0],
            fisheye_project_numpy(other[None], self.calibration.K1, self.calibration.D1)[0],
        ))
        return pixels if np.isfinite(pixels).all() else None

    def _stereo_observation(self, left: np.ndarray, right: np.ndarray,
                            sl: float, sr: float) -> tuple[str, np.ndarray | None, float, float]:
        if not np.isfinite(np.r_[left, right, sl, sr]).all():
            return "non_finite_observation", None, np.nan, np.nan
        size = np.array([self.calibration.image_width, self.calibration.image_height])
        if np.any(left < 0) or np.any(right < 0) or np.any(left >= size) or np.any(right >= size):
            return "out_of_raw_image_bounds", None, np.nan, np.nan
        if min(sl, sr) < self.config.min_score or max(sl, sr) > 1:
            return "low_or_invalid_score", None, np.nan, np.nan
        cal = self.calibration
        ul = cv2.fisheye.undistortPoints(left.reshape(1, 1, 2), cal.K0, cal.D0).reshape(2)
        ur = cv2.fisheye.undistortPoints(right.reshape(1, 1, 2), cal.K1, cal.D1).reshape(2)
        ray_l = np.r_[ul, 1.0]
        ray_r = cal.R_cam0_to_cam1.T @ np.r_[ur, 1.0]
        cosine = np.dot(ray_l, ray_r) / (np.linalg.norm(ray_l) * np.linalg.norm(ray_r))
        angle = float(np.degrees(np.arccos(np.clip(abs(cosine), 0, 1))))
        if not np.isfinite(angle) or angle < self.config.min_ray_angle_deg:
            return "weak_ray_angle", None, np.nan, angle
        homogeneous = cv2.triangulatePoints(
            np.c_[np.eye(3), np.zeros(3)],
            np.c_[cal.R_cam0_to_cam1, cal.T_cam0_to_cam1_mm],
            ul.reshape(2, 1), ur.reshape(2, 1),
        )[:, 0]
        if not np.isfinite(homogeneous).all() or abs(homogeneous[3]) < 1e-12:
            return "non_finite_triangulation", None, np.nan, angle
        point = homogeneous[:3] / homogeneous[3]
        projected = self._project(point)
        if projected is None:
            return "negative_depth", None, np.nan, angle
        error = float(np.linalg.norm((projected - np.r_[left, right]).reshape(2, 2), axis=1).max())
        if error > self.config.max_reprojection_px:
            return "stereo_reprojection_rejected", None, error, angle
        return "accepted", point, error, angle

    def _initialize(self, joint: int, point: np.ndarray, time_s: float) -> None:
        self.state[joint] = np.r_[point, np.zeros(6)]
        self.covariance[joint] = self.P0
        self.active[joint] = True
        self.segment_id[joint] += 1
        self.last_support_s[joint] = time_s

    def _invalidate(self, joint: int) -> None:
        self.active[joint] = False
        self.state[joint] = np.nan
        self.covariance[joint] = np.nan

    def _update(self, joint: int, measurement: np.ndarray) -> tuple[str, float]:
        point = self.state[joint, :3]
        projected = self._project(point)
        if projected is None:
            return "prediction_out_of_domain", np.nan
        # One-millimetre central differences keep the existing fisheye model.
        H = np.zeros((4, 9))
        for axis in range(3):
            offset = np.eye(3)[axis]
            plus, minus = self._project(point + offset), self._project(point - offset)
            if plus is None or minus is None:
                return "prediction_out_of_domain", np.nan
            H[:, axis] = (plus - minus) / 2
        P = self.covariance[joint]
        R = np.eye(4) * self.config.pixel_sigma_px**2
        innovation = measurement - projected
        S = H @ P @ H.T + R
        solved = np.linalg.solve(S, innovation)
        nis = float(innovation @ solved)
        if not np.isfinite(nis) or nis > self.config.innovation_gate_squared:
            return "innovation_rejected", nis
        K = np.linalg.solve(S, H @ P).T
        candidate = self.state[joint] + K @ innovation
        candidate_pixels = self._project(candidate[:3])
        if candidate_pixels is None:
            return "update_out_of_domain", nis
        nonlinear_residual = measurement - candidate_pixels
        if (np.linalg.norm(nonlinear_residual.reshape(2, 2), axis=1).max() > self.config.max_reprojection_px
                or nonlinear_residual @ nonlinear_residual > innovation @ innovation + 1e-8):
            return "nonlinear_update_rejected", nis
        residual = np.eye(9) - K @ H
        joseph = residual @ P @ residual.T + K @ R @ K.T
        self.state[joint] = candidate
        self.covariance[joint] = (joseph + joseph.T) / 2
        return "updated", nis

    def step(self, timestamp_s: float, left_pixels: np.ndarray, right_pixels: np.ndarray,
             scores_left: np.ndarray, scores_right: np.ndarray,
             initial_points_mm: np.ndarray | None = None) -> dict:
        """Consume a single frame. initial_points_mm is allowed only on frame 0.

        No first-frame pixel update is performed: the body fit already used
        those pixels. Initialization covariance is an engineering prior.
        """
        now = float(timestamp_s)
        if not np.isfinite(now) or (self.last_time_s is not None and now <= self.last_time_s):
            raise ValueError("timestamps must be finite and increase strictly")
        left, right = np.asarray(left_pixels, dtype=float), np.asarray(right_pixels, dtype=float)
        sl, sr = np.asarray(scores_left, dtype=float), np.asarray(scores_right, dtype=float)
        if left.shape != (self.joint_count, 2) or right.shape != left.shape or sl.shape != (self.joint_count,) or sr.shape != sl.shape:
            raise ValueError("pixels must be (joint_count,2), scores (joint_count,)")
        if initial_points_mm is not None:
            initial_points_mm = np.asarray(initial_points_mm, dtype=float)
            if self.last_time_s is not None or initial_points_mm.shape != (self.joint_count, 3):
                raise ValueError("first-frame initialization must be (joint_count,3) and provided once")
        dt = None if self.last_time_s is None else now - self.last_time_s
        F = None if dt is None else constant_acceleration_transition(dt)
        Q = None if dt is None else continuous_jerk_covariance(dt, self.config.jerk_spectral_density_mm2_s5)
        statuses = np.full(self.joint_count, "uninitialized", dtype=object)
        reasons = np.full(self.joint_count, "", dtype=object)
        nis = np.full(self.joint_count, np.nan)
        reprojection = np.full((self.joint_count, 2), np.nan)
        raw_error = np.full(self.joint_count, np.nan)
        ray_angle = np.full(self.joint_count, np.nan)
        for j in range(self.joint_count):
            reason, triangulated, raw_error[j], ray_angle[j] = self._stereo_observation(left[j], right[j], sl[j], sr[j])
            reasons[j] = reason
            if dt is None and initial_points_mm is not None:
                if self._project(initial_points_mm[j]) is not None:
                    self._initialize(j, initial_points_mm[j], now)
                    statuses[j] = "initialized"
                    reasons[j] = "first_frame_body_fit_prior;" + reason
                    if np.isfinite(np.r_[left[j], right[j]]).all():
                        reprojection[j] = np.linalg.norm((self._project(initial_points_mm[j]) - np.r_[left[j], right[j]]).reshape(2, 2), axis=1)
                continue
            expired = self.active[j] and now - self.last_support_s[j] > self.config.max_prediction_s
            if expired:
                self._invalidate(j)
                statuses[j] = "expired"
            if not self.active[j]:
                if triangulated is not None:
                    self._initialize(j, triangulated, now)
                    statuses[j] = "reinitialized" if self.segment_id[j] > 0 else "initialized"
                    reasons[j] = "stereo_restart_zero_v_a_prior"
                elif self.segment_id[j] >= 0:
                    statuses[j] = "expired"
                continue
            self.state[j] = F @ self.state[j]
            P = F @ self.covariance[j] @ F.T + Q
            self.covariance[j] = (P + P.T) / 2
            statuses[j] = "predicted"
            if triangulated is not None:
                update_reason, nis[j] = self._update(j, np.r_[left[j], right[j]])
                reasons[j] = update_reason
                if update_reason == "updated":
                    statuses[j] = "updated"
                    self.last_support_s[j] = now
                elif update_reason == "prediction_out_of_domain":
                    self._invalidate(j)
                    self._initialize(j, triangulated, now)
                    statuses[j] = "reinitialized"
                    reasons[j] = "prediction_out_of_domain;stereo_restart_zero_v_a_prior"
            elif self._project(self.state[j, :3]) is None:
                self._invalidate(j)
                statuses[j] = "expired"
                reasons[j] = "prediction_out_of_domain;" + reason
            if self.active[j]:
                projected = self._project(self.state[j, :3])
                if projected is not None and np.isfinite(np.r_[left[j], right[j]]).all():
                    reprojection[j] = np.linalg.norm((projected - np.r_[left[j], right[j]]).reshape(2, 2), axis=1)
        self.last_time_s = now
        return {
            "timestamp_s": now, "dt_s": dt, "state": self.state.copy(),
            "covariance": self.covariance.copy(), "active": self.active.copy(),
            "segment_id": self.segment_id.copy(), "status": statuses, "reason": reasons,
            "prediction_age_s": np.where(self.active, now - self.last_support_s, np.nan),
            "innovation_squared_mahalanobis": nis, "reprojection_px": reprojection,
            "stereo_reprojection_max_px": raw_error, "ray_angle_deg": ray_angle,
            "coordinate_frame": "current_left_camera",
            "derivative_definition": "v=d(p_C(t))/dt,a=d2(p_C(t))/dt2;includes_camera_motion",
            "world_state_available": False,
        }
