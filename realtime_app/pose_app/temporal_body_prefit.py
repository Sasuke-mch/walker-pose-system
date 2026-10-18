from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


COCO_EDGES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 1),
    (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
)
STAGE2 = "stage2_feet_static_walker_moving"


@dataclass(frozen=True)
class TemporalBodyPrefitConfig:
    iterations: int = 500
    learning_rate: float = 0.025
    data_weight: float = 18.0
    bone_weight: float = 12.0
    acceleration_weight: float = 4.0
    foot_contact_weight: float = 10.0
    hand_contact_weight: float = 3.0
    low_speed_foot_mm_per_frame: float = 14.0
    minimum_contact_frames: int = 3


@dataclass(frozen=True)
class TemporalBodyPrefitAcceptanceCriteria:
    minimum_acceleration_p95_reduction_fraction: float = 0.30
    maximum_pelvis_net_displacement_change_fraction: float = 0.02
    maximum_fitted_bone_length_cv_median: float = 0.05
    maximum_deviation_median_mm: float = 10.0
    maximum_deviation_p95_mm: float = 40.0
    correction_top_k: int = 30
    minimum_bad_observation_fraction_in_top_corrections: float = 0.80


def observation_weights(errors_px: np.ndarray, strict: np.ndarray, valid: np.ndarray) -> np.ndarray:
    errors = np.asarray(errors_px, dtype=np.float64)
    strict = np.asarray(strict, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    safe = np.where(np.isfinite(errors), errors, 30.0)
    strict_weight = 1.0 / (1.0 + (safe / 5.0) ** 2)
    display_weight = 0.22 / (1.0 + (safe / 10.0) ** 2)
    return np.where(valid, np.where(strict, strict_weight, display_weight), 0.0)


def contiguous_segments(mask: np.ndarray, minimum_length: int) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    segments: list[np.ndarray] = []
    start = None
    for index, active in enumerate(np.r_[mask, False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start >= minimum_length:
                segments.append(np.arange(start, index, dtype=np.int64))
            start = None
    return segments


def fill_missing_linear(points_mm: np.ndarray) -> np.ndarray:
    points = np.asarray(points_mm, dtype=np.float64).copy()
    time = np.arange(len(points), dtype=np.float64)
    for joint in range(points.shape[1]):
        for axis in range(3):
            values = points[:, joint, axis]
            good = np.isfinite(values)
            if not np.any(good):
                raise ValueError(f"joint {joint} axis {axis} has no finite observation")
            values[~good] = np.interp(time[~good], time[good], values[good])
    return points


def estimate_bone_lengths_mm(points_mm: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = []
    for start, end in COCO_EDGES:
        lengths = np.linalg.norm(points_mm[:, start] - points_mm[:, end], axis=1)
        reliable = np.minimum(weights[:, start], weights[:, end]) >= 0.2
        finite = np.isfinite(lengths)
        selected = lengths[reliable & finite]
        if len(selected) < 3:
            selected = lengths[finite]
        values.append(float(np.median(selected)))
    return np.asarray(values, dtype=np.float64)


def _rigid_point(point_left_mm: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    rotation = np.asarray(frame["rotation_world_from_left"], dtype=np.float64)
    translation = np.asarray(frame["translation_world_from_left_mm"], dtype=np.float64)
    return rotation @ np.asarray(point_left_mm, dtype=np.float64) + translation


def build_contact_targets(
    points_mm: np.ndarray,
    frames: list[dict[str, Any]],
    interactions: list[dict[str, Any]],
    walker: dict[str, Any],
    config: TemporalBodyPrefitConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    frame_count = len(frames)
    targets = np.zeros((frame_count, 17, 3), dtype=np.float64)
    weights = np.zeros((frame_count, 17), dtype=np.float64)
    audit: dict[str, Any] = {"foot_segments": {}, "hand_segments": {}}

    stages = np.asarray([frame["stage"] for frame in frames], dtype=object)
    for side, joint in (("left", 15), ("right", 16)):
        velocity = np.r_[np.nan, np.linalg.norm(np.diff(points_mm[:, joint], axis=0), axis=1)]
        height_limit = float(np.nanpercentile(points_mm[:, joint, 2], 35) + 20.0)
        inferred = (velocity <= config.low_speed_foot_mm_per_frame) & (points_mm[:, joint, 2] <= height_limit)
        active = (stages == STAGE2) | inferred
        segments = contiguous_segments(active, config.minimum_contact_frames)
        for segment in segments:
            anchor = np.median(points_mm[segment, joint], axis=0)
            targets[segment, joint] = anchor
            weights[segment, joint] = 1.0
        audit["foot_segments"][side] = [
            {"start": int(segment[0]), "end": int(segment[-1]), "frames": len(segment)}
            for segment in segments
        ]

    nodes = walker["nodes_left_camera_mm"]
    handles = walker["handle_segments"]
    for side, joint in (("left", 9), ("right", 10)):
        start_name, end_name = handles[side]
        valid = np.zeros(frame_count, dtype=bool)
        fractions = np.full(frame_count, np.nan)
        offsets = np.full((frame_count, 3), np.nan)
        handle_ends: list[tuple[np.ndarray, np.ndarray]] = []
        for index, (frame, interaction) in enumerate(zip(frames, interactions)):
            start = _rigid_point(np.asarray(nodes[start_name]), frame)
            end = _rigid_point(np.asarray(nodes[end_name]), frame)
            handle_ends.append((start, end))
            hand = (interaction.get("hands") or {}).get(side) or {}
            fraction = hand.get("closest_segment_fraction")
            valid[index] = bool(
                hand.get("state") == "near_handle_axis_candidate"
                and hand.get("quality_gate_passed")
                and fraction is not None
            )
            if valid[index]:
                fractions[index] = float(fraction)
                closest = start + fractions[index] * (end - start)
                offsets[index] = points_mm[index, joint] - closest
        segments = contiguous_segments(valid, config.minimum_contact_frames)
        for segment in segments:
            fraction = float(np.median(fractions[segment]))
            offset = np.median(offsets[segment], axis=0)
            for index in segment:
                start, end = handle_ends[index]
                targets[index, joint] = start + fraction * (end - start) + offset
                weights[index, joint] = 1.0
        audit["hand_segments"][side] = [
            {"start": int(segment[0]), "end": int(segment[-1]), "frames": len(segment)}
            for segment in segments
        ]
    return targets, weights, audit


def temporal_prefit(
    observations_mm: np.ndarray,
    errors_px: np.ndarray,
    strict: np.ndarray,
    contact_targets_mm: np.ndarray,
    contact_weights: np.ndarray,
    config: TemporalBodyPrefitConfig = TemporalBodyPrefitConfig(),
    device: str = "cpu",
) -> tuple[np.ndarray, dict[str, Any]]:
    # Keep the offline Torch optimizer out of import-time paths. The realtime
    # OpenCV stack can load a different OpenMP runtime on Windows, so merely
    # importing this module must not initialize Torch/libiomp.
    import torch

    observations = np.asarray(observations_mm, dtype=np.float64)
    valid = np.isfinite(observations).all(axis=2)
    initial = fill_missing_linear(observations)
    weights = observation_weights(errors_px, strict, valid)
    bone_lengths = estimate_bone_lengths_mm(initial, weights)

    dtype = torch.float64
    x = torch.nn.Parameter(torch.as_tensor(initial / 1000.0, dtype=dtype, device=device))
    obs = torch.as_tensor(np.nan_to_num(observations / 1000.0), dtype=dtype, device=device)
    obs_weights = torch.as_tensor(weights, dtype=dtype, device=device)
    target = torch.as_tensor(contact_targets_mm / 1000.0, dtype=dtype, device=device)
    target_weights = torch.as_tensor(contact_weights, dtype=dtype, device=device)
    desired_bones = torch.as_tensor(bone_lengths / 1000.0, dtype=dtype, device=device)
    edges = torch.as_tensor(COCO_EDGES, dtype=torch.long, device=device)

    optimizer = torch.optim.Adam([x], lr=config.learning_rate)
    final_terms: dict[str, float] = {}
    for _ in range(config.iterations):
        optimizer.zero_grad()
        residual = torch.linalg.vector_norm(x - obs, dim=2)
        data_loss = torch.sum(obs_weights * residual ** 2) / torch.clamp(obs_weights.sum(), min=1.0)
        edge_vectors = x[:, edges[:, 0]] - x[:, edges[:, 1]]
        edge_lengths = torch.linalg.vector_norm(edge_vectors, dim=2)
        bone_loss = torch.mean((edge_lengths - desired_bones[None]) ** 2)
        acceleration = x[2:] - 2.0 * x[1:-1] + x[:-2]
        acceleration_loss = torch.mean(acceleration * acceleration)
        contact_residual = torch.linalg.vector_norm(x - target, dim=2)
        foot_mask = torch.zeros_like(target_weights)
        foot_mask[:, 15:17] = target_weights[:, 15:17]
        hand_mask = torch.zeros_like(target_weights)
        hand_mask[:, 9:11] = target_weights[:, 9:11]
        foot_loss = torch.sum(foot_mask * contact_residual ** 2) / torch.clamp(foot_mask.sum(), min=1.0)
        hand_loss = torch.sum(hand_mask * contact_residual ** 2) / torch.clamp(hand_mask.sum(), min=1.0)
        loss = (
            config.data_weight * data_loss
            + config.bone_weight * bone_loss
            + config.acceleration_weight * acceleration_loss
            + config.foot_contact_weight * foot_loss
            + config.hand_contact_weight * hand_loss
        )
        loss.backward()
        optimizer.step()
        final_terms = {
            "total": float(loss.detach().cpu()),
            "data": float(data_loss.detach().cpu()),
            "bone": float(bone_loss.detach().cpu()),
            "acceleration": float(acceleration_loss.detach().cpu()),
            "foot_contact": float(foot_loss.detach().cpu()),
            "hand_contact": float(hand_loss.detach().cpu()),
        }
    fitted = x.detach().cpu().numpy() * 1000.0
    return fitted, {"loss_terms": final_terms, "bone_lengths_mm": bone_lengths.tolist()}


def sequence_metrics(raw_mm: np.ndarray, fitted_mm: np.ndarray, contact_targets_mm: np.ndarray, contact_weights: np.ndarray) -> dict[str, Any]:
    raw = fill_missing_linear(raw_mm)
    fitted = np.asarray(fitted_mm, dtype=np.float64)

    def motion(values: np.ndarray) -> dict[str, float]:
        velocity = np.linalg.norm(np.diff(values, axis=0), axis=2)
        acceleration = np.linalg.norm(values[2:] - 2 * values[1:-1] + values[:-2], axis=2)
        return {
            "velocity_median_mm_per_frame": float(np.median(velocity)),
            "velocity_p95_mm_per_frame": float(np.percentile(velocity, 95)),
            "acceleration_median_mm_per_frame2": float(np.median(acceleration)),
            "acceleration_p95_mm_per_frame2": float(np.percentile(acceleration, 95)),
        }

    def bone_cv(values: np.ndarray) -> float:
        lengths = np.stack([np.linalg.norm(values[:, a] - values[:, b], axis=1) for a, b in COCO_EDGES], axis=1)
        return float(np.median(np.std(lengths, axis=0) / np.maximum(np.mean(lengths, axis=0), 1e-6)))

    deviation = np.linalg.norm(fitted - raw, axis=2)
    raw_pelvis = 0.5 * (raw[:, 11] + raw[:, 12])
    fitted_pelvis = 0.5 * (fitted[:, 11] + fitted[:, 12])
    raw_pelvis_net = float(np.linalg.norm(raw_pelvis[-1] - raw_pelvis[0]))
    fitted_pelvis_net = float(np.linalg.norm(fitted_pelvis[-1] - fitted_pelvis[0]))
    active = contact_weights > 0
    contact_before = np.linalg.norm(raw - contact_targets_mm, axis=2)[active]
    contact_after = np.linalg.norm(fitted - contact_targets_mm, axis=2)[active]
    return {
        "raw_motion": motion(raw),
        "fitted_motion": motion(fitted),
        "raw_bone_length_cv_median": bone_cv(raw),
        "fitted_bone_length_cv_median": bone_cv(fitted),
        "fitted_to_raw_deviation_median_mm": float(np.median(deviation)),
        "fitted_to_raw_deviation_p95_mm": float(np.percentile(deviation, 95)),
        "raw_pelvis_net_displacement_mm": raw_pelvis_net,
        "fitted_pelvis_net_displacement_mm": fitted_pelvis_net,
        "pelvis_net_displacement_change_fraction": float(
            abs(fitted_pelvis_net - raw_pelvis_net) / max(raw_pelvis_net, 1e-6)
        ),
        "contact_residual_before_median_mm": float(np.median(contact_before)) if len(contact_before) else None,
        "contact_residual_after_median_mm": float(np.median(contact_after)) if len(contact_after) else None,
    }


def prefit_acceptance_audit(
    raw_mm: np.ndarray,
    fitted_mm: np.ndarray,
    errors_px: np.ndarray,
    strict: np.ndarray,
    metrics: dict[str, Any],
    criteria: TemporalBodyPrefitAcceptanceCriteria = TemporalBodyPrefitAcceptanceCriteria(),
) -> dict[str, Any]:
    """Evaluate bounded engineering gates without treating them as accuracy truth."""
    raw = fill_missing_linear(raw_mm)
    fitted = np.asarray(fitted_mm, dtype=np.float64)
    errors = np.asarray(errors_px, dtype=np.float64)
    strict_mask = np.asarray(strict, dtype=bool)
    deviations = np.linalg.norm(fitted - raw, axis=2)
    flat_order = np.argsort(deviations, axis=None)[::-1]
    top_k = min(criteria.correction_top_k, len(flat_order))
    top_rows = []
    bad_count = 0
    for flat_index in flat_order[:top_k]:
        frame, joint = np.unravel_index(flat_index, deviations.shape)
        error = float(errors[frame, joint]) if np.isfinite(errors[frame, joint]) else None
        is_bad = (not bool(strict_mask[frame, joint])) and (error is None or error > 10.0)
        bad_count += int(is_bad)
        top_rows.append({
            "frame_index": int(frame),
            "joint_index": int(joint),
            "deviation_mm": float(deviations[frame, joint]),
            "strict": bool(strict_mask[frame, joint]),
            "reprojection_error_px": error,
            "non_strict_high_error": bool(is_bad),
        })
    bad_fraction = float(bad_count / max(top_k, 1))
    raw_acceleration = metrics["raw_motion"]["acceleration_p95_mm_per_frame2"]
    fitted_acceleration = metrics["fitted_motion"]["acceleration_p95_mm_per_frame2"]
    acceleration_reduction = float(
        (raw_acceleration - fitted_acceleration) / max(raw_acceleration, 1e-6)
    )
    gates = {
        "acceleration_p95_reduction": {
            "value": acceleration_reduction,
            "required_minimum": criteria.minimum_acceleration_p95_reduction_fraction,
            "passed": acceleration_reduction >= criteria.minimum_acceleration_p95_reduction_fraction,
        },
        "pelvis_net_displacement_change": {
            "value": metrics["pelvis_net_displacement_change_fraction"],
            "required_maximum": criteria.maximum_pelvis_net_displacement_change_fraction,
            "passed": metrics["pelvis_net_displacement_change_fraction"] <= criteria.maximum_pelvis_net_displacement_change_fraction,
        },
        "bone_length_cv_median": {
            "value": metrics["fitted_bone_length_cv_median"],
            "required_maximum": criteria.maximum_fitted_bone_length_cv_median,
            "passed": metrics["fitted_bone_length_cv_median"] <= criteria.maximum_fitted_bone_length_cv_median,
        },
        "deviation_median_mm": {
            "value": metrics["fitted_to_raw_deviation_median_mm"],
            "required_maximum": criteria.maximum_deviation_median_mm,
            "passed": metrics["fitted_to_raw_deviation_median_mm"] <= criteria.maximum_deviation_median_mm,
        },
        "deviation_p95_mm": {
            "value": metrics["fitted_to_raw_deviation_p95_mm"],
            "required_maximum": criteria.maximum_deviation_p95_mm,
            "passed": metrics["fitted_to_raw_deviation_p95_mm"] <= criteria.maximum_deviation_p95_mm,
        },
        "top_corrections_match_bad_observations": {
            "value": bad_fraction,
            "required_minimum": criteria.minimum_bad_observation_fraction_in_top_corrections,
            "passed": bad_fraction >= criteria.minimum_bad_observation_fraction_in_top_corrections,
        },
    }
    return {
        "status": "passed" if all(item["passed"] for item in gates.values()) else "failed",
        "criteria": criteria.__dict__,
        "gates": gates,
        "top_corrections": top_rows,
        "interpretation": (
            "Engineering consistency gates only. Passing does not establish physical 3D accuracy, "
            "contact truth, or SMPL-X mesh correctness."
        ),
    }
