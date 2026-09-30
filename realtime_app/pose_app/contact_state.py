"""Causal contact-state candidates and surface-motion diagnostics.

This module deliberately does not claim physical contact.  It converts
surface geometry and frame-to-frame motion into auditable engineering
candidates for later fitting stages.  Missing or discontinuous ground poses
remain ``unknown`` instead of inheriting a previous contact state.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

import numpy as np


class ContactState(str, Enum):
    UNKNOWN = "unknown"
    SEPARATED = "separated"
    CONTACT_CANDIDATE = "contact_candidate"
    STICKING_CANDIDATE = "sticking_candidate"
    SLIDING_CANDIDATE = "sliding_candidate"


@dataclass(frozen=True)
class ContactConfig:
    """Thresholds in SI units and milliseconds.

    ``enter_distance_m`` and ``exit_distance_m`` intentionally form a
    hysteresis pair.  They are engineering gates, not physical constants.
    """

    enter_distance_m: float = 0.015
    exit_distance_m: float = 0.025
    enter_normal_speed_mps: float = 0.08
    sticking_tangent_speed_mps: float = 0.06
    sliding_tangent_speed_mps: float = 0.12
    enter_duration_ms: float = 80.0
    exit_duration_ms: float = 100.0
    min_quality: float = 0.25
    max_ground_jump_m: float = 0.060

    def __post_init__(self) -> None:
        if not (0.0 < self.enter_distance_m < self.exit_distance_m):
            raise ValueError("enter_distance_m must be positive and below exit_distance_m")
        if self.enter_duration_ms <= 0 or self.exit_duration_ms <= 0:
            raise ValueError("hysteresis durations must be positive")
        if not (0.0 <= self.min_quality <= 1.0):
            raise ValueError("min_quality must be in [0,1]")
        if self.sticking_tangent_speed_mps > self.sliding_tangent_speed_mps:
            raise ValueError("sticking threshold cannot exceed sliding threshold")


@dataclass(frozen=True)
class ContactObservation:
    timestamp_s: float
    distance_m: float
    normal_speed_mps: float
    tangent_speed_mps: float
    quality: float
    ground_continuous: bool = True
    finite: bool = True


@dataclass(frozen=True)
class ContactDecision:
    timestamp_s: float
    state: ContactState
    accepted_for_constraint: bool
    reason: str
    distance_m: float | None
    normal_speed_mps: float | None
    tangent_speed_mps: float | None
    quality: float | None
    source: str = "surface_geometry_plus_temporal_consistency"


def _finite_observation(obs: ContactObservation) -> bool:
    values = (obs.timestamp_s, obs.distance_m, obs.normal_speed_mps,
              obs.tangent_speed_mps, obs.quality)
    return obs.finite and bool(np.isfinite(values).all())


class ContactStateMachine:
    """Causal state machine with time-based enter/exit hysteresis."""

    def __init__(self, config: ContactConfig | None = None) -> None:
        self.config = config or ContactConfig()
        self.state = ContactState.UNKNOWN
        self._near_since: float | None = None
        self._far_since: float | None = None

    def update(self, obs: ContactObservation) -> ContactDecision:
        c = self.config
        if not _finite_observation(obs):
            self.state = ContactState.UNKNOWN
            self._near_since = self._far_since = None
            return ContactDecision(obs.timestamp_s, self.state, False,
                                   "nonfinite_observation", None, None, None, None)
        if obs.quality < c.min_quality:
            self.state = ContactState.UNKNOWN
            self._near_since = self._far_since = None
            return ContactDecision(obs.timestamp_s, self.state, False,
                                   "quality_below_gate", obs.distance_m,
                                   obs.normal_speed_mps, obs.tangent_speed_mps,
                                   obs.quality)
        if not obs.ground_continuous:
            self.state = ContactState.UNKNOWN
            self._near_since = self._far_since = None
            return ContactDecision(obs.timestamp_s, self.state, False,
                                   "ground_transform_discontinuous", obs.distance_m,
                                   obs.normal_speed_mps, obs.tangent_speed_mps,
                                   obs.quality)

        near = (obs.distance_m <= c.enter_distance_m and
                abs(obs.normal_speed_mps) <= c.enter_normal_speed_mps)
        far = obs.distance_m >= c.exit_distance_m
        if near:
            self._near_since = obs.timestamp_s if self._near_since is None else self._near_since
            self._far_since = None
        elif far:
            self._far_since = obs.timestamp_s if self._far_since is None else self._far_since
            self._near_since = None
        else:
            self._near_since = None
            self._far_since = None

        if self.state in (ContactState.UNKNOWN, ContactState.SEPARATED):
            if self._near_since is not None and (obs.timestamp_s - self._near_since) * 1000.0 >= c.enter_duration_ms:
                self.state = self._motion_state(obs)
            elif near:
                self.state = ContactState.CONTACT_CANDIDATE
            else:
                self.state = ContactState.SEPARATED
        else:
            if self._far_since is not None and (obs.timestamp_s - self._far_since) * 1000.0 >= c.exit_duration_ms:
                self.state = ContactState.SEPARATED
            elif self.state in (ContactState.CONTACT_CANDIDATE,
                                ContactState.STICKING_CANDIDATE,
                                ContactState.SLIDING_CANDIDATE):
                self.state = self._motion_state(obs)

        usable = self.state in (ContactState.STICKING_CANDIDATE,
                                ContactState.SLIDING_CANDIDATE)
        reason = "state_candidate" if usable else (
            "within_enter_gate_waiting_hysteresis" if near else "outside_contact_gate")
        return ContactDecision(obs.timestamp_s, self.state, usable, reason,
                               obs.distance_m, obs.normal_speed_mps,
                               obs.tangent_speed_mps, obs.quality)

    def _motion_state(self, obs: ContactObservation) -> ContactState:
        if obs.tangent_speed_mps <= self.config.sticking_tangent_speed_mps:
            return ContactState.STICKING_CANDIDATE
        if obs.tangent_speed_mps >= self.config.sliding_tangent_speed_mps:
            return ContactState.SLIDING_CANDIDATE
        return ContactState.CONTACT_CANDIDATE


def surface_partition_metrics(
    surface_points_ground_m: np.ndarray,
    partitions: Mapping[str, Iterable[int]],
    ground_z_m: float = 0.0,
) -> dict[str, dict[str, float | int]]:
    """Return per-region height, penetration and coverage diagnostics.

    No fitting or interpolation is performed.  Empty or invalid regions are
    explicitly reported as unavailable.
    """
    points = np.asarray(surface_points_ground_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("surface_points_ground_m must have shape [N,3]")
    result: dict[str, dict[str, float | int]] = {}
    for name, raw_indices in partitions.items():
        indices = np.asarray(list(raw_indices), dtype=int)
        valid = indices[(indices >= 0) & (indices < len(points))]
        if len(valid) == 0:
            result[name] = {"available": 0, "valid_points": 0}
            continue
        z = points[valid, 2]
        finite = np.isfinite(z)
        z = z[finite]
        if len(z) == 0:
            result[name] = {"available": 0, "valid_points": 0}
            continue
        result[name] = {
            "available": 1,
            "valid_points": int(len(z)),
            "minimum_height_m": float(np.min(z - ground_z_m)),
            "median_height_m": float(np.median(z - ground_z_m)),
            "penetration_fraction": float(np.mean(z < ground_z_m)),
        }
    return result


__all__ = ["ContactConfig", "ContactDecision", "ContactObservation",
           "ContactState", "ContactStateMachine", "surface_partition_metrics"]
