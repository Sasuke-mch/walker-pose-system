"""R3 per-frame data contract and integration policy for SMPL-X fitting routes.

Module-level import must stay free of torch/smplx so that unit tests and the
full-suite runner can import it in any interpreter.

The R3 contract (frozen for G20260919_smplx_r3_integration_v1):

  per R3 frame row::
    points_mm               17x3 world/ground millimetres (NaN = absent)
    availability            17 bool
    output_source           17 str  (e.g. pass_through_base_filter / ik_projected / unavailable)
    is_measurement          17 bool
    is_prediction_only      17 bool
    is_innovation_rejected  17 bool
    is_single_view_inferred 17 bool
    is_unavailable          17 bool
    rejection_reasons       17 lists of str
    per_joint_correction_mm 17 floats (NaN allowed)

``r3_high_quality`` (frozen gate): finite coordinates AND availability AND
is_measurement AND NOT is_prediction_only AND NOT is_innovation_rejected AND
NOT is_single_view_inferred AND NOT is_unavailable AND
per_joint_correction_mm <= 40.  Every other joint receives weight exactly 0 in
the C-R3-soft term and is excluded from the R3 initialization; old values are
never filled in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

R3_REQUIRED_FIELDS = (
    "points_mm",
    "availability",
    "output_source",
    "is_measurement",
    "is_prediction_only",
    "is_innovation_rejected",
    "is_single_view_inferred",
    "is_unavailable",
    "rejection_reasons",
    "per_joint_correction_mm",
)

JOINT_COUNT = 17
MAX_HIGH_QUALITY_CORRECTION_MM = 40.0

# Frozen route plan for the four comparison routes.
ROUTE_C_PREFIT = "C-prefit"
ROUTE_C_R3_INIT = "C-R3-init"
ROUTE_B_R3_INIT = "B-R3-init"
ROUTE_C_R3_SOFT = "C-R3-soft"
ALL_ROUTES = (ROUTE_C_PREFIT, ROUTE_C_R3_INIT, ROUTE_B_R3_INIT, ROUTE_C_R3_SOFT)

# C-R3-soft small-weight grid. Scanning larger weights to force depth onto R3
# is explicitly forbidden by the protocol.
R3_SOFT_WEIGHT_GRID = (0.10, 0.25, 0.50, 1.00, 2.00)

# R3 initialization pre-optimization budget (frozen before running; timed
# separately and included in the end-to-end time).
R3_INIT_ITERATIONS = 60
R3_INIT_LEARNING_RATE = 0.02
R3_SOFT_SCALE_M = 0.03  # Geman-McClure scale shared with the prefit soft 3D term


class R3ContractError(ValueError):
    """Raised when an R3 per-frame row violates the frozen data contract."""


@dataclass(frozen=True)
class R3PerFrame:
    frame_index: int
    points_mm: np.ndarray  # (17, 3) float64
    availability: np.ndarray  # (17,) bool
    output_source: tuple[str, ...]
    is_measurement: np.ndarray  # (17,) bool
    is_prediction_only: np.ndarray  # (17,) bool
    is_innovation_rejected: np.ndarray  # (17,) bool
    is_single_view_inferred: np.ndarray  # (17,) bool
    is_unavailable: np.ndarray  # (17,) bool
    rejection_reasons: tuple[tuple[str, ...], ...]  # 17 tuples of str
    per_joint_correction_mm: np.ndarray  # (17,) float64
    meta: dict[str, Any] = field(default_factory=dict)


def _require(row: dict[str, Any], frame_index: Any) -> None:
    missing = [name for name in R3_REQUIRED_FIELDS if name not in row]
    if missing:
        raise R3ContractError(
            f"R3 frame {frame_index}: missing required field(s) {missing}; "
            "the row is rejected and no old value may be filled in"
        )


def parse_r3_per_frame(row: dict[str, Any]) -> R3PerFrame:
    """Strictly parse one R3 row; raises R3ContractError on any violation."""
    frame_index = row.get("frame_index")
    _require(row, frame_index)
    points = np.asarray(row["points_mm"], dtype=np.float64)
    if points.shape != (JOINT_COUNT, 3):
        raise R3ContractError(
            f"R3 frame {frame_index}: points_mm has shape {points.shape}, expected (17, 3)"
        )
    vector_fields = {
        "availability": bool,
        "is_measurement": bool,
        "is_prediction_only": bool,
        "is_innovation_rejected": bool,
        "is_single_view_inferred": bool,
        "is_unavailable": bool,
    }
    parsed: dict[str, np.ndarray] = {}
    for name, dtype in vector_fields.items():
        value = np.asarray(row[name], dtype=dtype)
        if value.shape != (JOINT_COUNT,):
            raise R3ContractError(
                f"R3 frame {frame_index}: {name} has shape {value.shape}, expected (17,)"
            )
        parsed[name] = value
    sources = tuple(str(item) for item in row["output_source"])
    if len(sources) != JOINT_COUNT:
        raise R3ContractError(
            f"R3 frame {frame_index}: output_source has length {len(sources)}, expected 17"
        )
    reasons = row["rejection_reasons"]
    if not isinstance(reasons, (list, tuple)) or len(reasons) != JOINT_COUNT:
        raise R3ContractError(
            f"R3 frame {frame_index}: rejection_reasons has length {len(reasons)}, expected 17"
        )
    reason_tuples = tuple(tuple(str(item) for item in joint) for joint in reasons)
    corrections = np.asarray(row["per_joint_correction_mm"], dtype=np.float64)
    if corrections.shape != (JOINT_COUNT,):
        raise R3ContractError(
            f"R3 frame {frame_index}: per_joint_correction_mm has shape "
            f"{corrections.shape}, expected (17,)"
        )
    meta = row.get("meta") or {}
    if not isinstance(meta, dict):
        raise R3ContractError(f"R3 frame {frame_index}: meta must be a dict")
    return R3PerFrame(
        frame_index=int(frame_index),
        points_mm=points,
        availability=parsed["availability"],
        output_source=sources,
        is_measurement=parsed["is_measurement"],
        is_prediction_only=parsed["is_prediction_only"],
        is_innovation_rejected=parsed["is_innovation_rejected"],
        is_single_view_inferred=parsed["is_single_view_inferred"],
        is_unavailable=parsed["is_unavailable"],
        rejection_reasons=reason_tuples,
        per_joint_correction_mm=corrections,
        meta=meta,
    )


def load_r3_jsonl(path: str | Path) -> dict[int, R3PerFrame]:
    """Load the whole R3 per-frame JSONL into {frame_index: R3PerFrame}."""
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    records: dict[int, R3PerFrame] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json_loads(line)
        except ValueError as error:  # malformed JSON is a contract violation
            raise R3ContractError(f"R3 jsonl line {line_number}: invalid JSON: {error}") from error
        record = parse_r3_per_frame(row)
        records[record.frame_index] = record
    return records


def json_loads(line: str) -> dict[str, Any]:
    import json
    return json.loads(line)


def r3_high_quality(frame: R3PerFrame, joint: int) -> bool:
    """Frozen per-(frame, joint) R3 quality gate."""
    if not (0 <= joint < JOINT_COUNT):
        return False
    point = frame.points_mm[joint]
    if not bool(np.all(np.isfinite(point))):
        return False
    correction = frame.per_joint_correction_mm[joint]
    if not np.isfinite(correction) or correction > MAX_HIGH_QUALITY_CORRECTION_MM:
        return False
    return bool(
        frame.availability[joint]
        and frame.is_measurement[joint]
        and not frame.is_prediction_only[joint]
        and not frame.is_innovation_rejected[joint]
        and not frame.is_single_view_inferred[joint]
        and not frame.is_unavailable[joint]
    )


def r3_quality_mask(frame: R3PerFrame) -> np.ndarray:
    """(17,) bool mask of high-quality joints for one frame."""
    return np.asarray([r3_high_quality(frame, joint) for joint in range(JOINT_COUNT)], dtype=bool)


def r3_rejection_reasons(frame: R3PerFrame, joint: int) -> list[str]:
    """Why a joint fails the r3_high_quality gate (empty when it passes)."""
    if r3_high_quality(frame, joint):
        return []
    reasons: list[str] = []
    point = frame.points_mm[joint]
    if not bool(np.all(np.isfinite(point))):
        reasons.append("point_non_finite")
    if not bool(frame.availability[joint]):
        reasons.append("availability_false")
    if not bool(frame.is_measurement[joint]):
        reasons.append("not_measurement")
    if bool(frame.is_prediction_only[joint]):
        reasons.append("prediction_only")
    if bool(frame.is_innovation_rejected[joint]):
        reasons.append("innovation_rejected")
    if bool(frame.is_single_view_inferred[joint]):
        reasons.append("single_view_inferred")
    if bool(frame.is_unavailable[joint]):
        reasons.append("unavailable")
    correction = frame.per_joint_correction_mm[joint]
    if np.isfinite(correction) and correction > MAX_HIGH_QUALITY_CORRECTION_MM:
        reasons.append("correction_exceeds_40mm")
    return reasons


def build_r3_soft_weights(
    records_by_frame: dict[int, R3PerFrame],
    window_indices: np.ndarray,
    body_coco_positions: np.ndarray,
    hold_out: set[int],
) -> tuple[np.ndarray, dict[str, int]]:
    """Per-(frame, body-joint) C-R3-soft weights.

    body_coco_positions maps positions 0..11 to COCO joint indices (12 body
    joints).  Weight is 1.0 only for r3_high_quality joints not held out;
    everything else is exactly 0.0 (no old-value filling).
    """
    weights = np.zeros((len(window_indices), len(body_coco_positions)), dtype=np.float64)
    audit = {"high_quality_count": 0, "zero_weight_count": 0, "missing_frame_count": 0}
    for local, frame_index in enumerate(window_indices):
        record = records_by_frame.get(int(frame_index))
        if record is None:
            audit["missing_frame_count"] += 1
            audit["zero_weight_count"] += len(body_coco_positions)
            continue
        for position, coco_joint in enumerate(body_coco_positions):
            if int(coco_joint) in hold_out:
                audit["zero_weight_count"] += 1
                continue
            if r3_high_quality(record, int(coco_joint)):
                weights[local, position] = 1.0
                audit["high_quality_count"] += 1
            else:
                audit["zero_weight_count"] += 1
    return weights, audit


def r3_targets_m(
    records_by_frame: dict[int, R3PerFrame],
    window_indices: np.ndarray,
    body_coco_positions: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """(N, J, 3) R3 target in metres; NaN replaced by 0 where the weight is 0.

    Contract guarantee: wherever ``weights > 0`` the target is finite by
    construction (r3_high_quality requires finite coordinates), so NaN can
    never enter the loss through a weighted entry.
    """
    targets = np.zeros((len(window_indices), len(body_coco_positions), 3), dtype=np.float64)
    for local, frame_index in enumerate(window_indices):
        record = records_by_frame.get(int(frame_index))
        if record is None:
            continue
        for position, coco_joint in enumerate(body_coco_positions):
            if weights[local, position] > 0.0:
                targets[local, position] = record.points_mm[int(coco_joint)] / 1000.0
    return targets


def r3_soft_loss_torch(
    predicted_coco_body_m: Any,
    target_m: Any,
    weights: Any,
    scale_m: float = R3_SOFT_SCALE_M,
) -> Any:
    """Robust soft 3D R3 term normalised by the number of valid joints.

    NaN safety: residual is explicitly zeroed wherever the weight is 0, so a
    NaN sitting in a zero-weighted target entry cannot propagate.
    """
    import torch

    safe_target = torch.nan_to_num(target_m, nan=0.0)
    residual = (predicted_coco_body_m - safe_target) * weights.unsqueeze(-1)
    squared = (residual * residual).sum(dim=-1)
    robust = squared / (squared + scale_m * scale_m)
    denominator = torch.clamp(weights.sum(), min=1.0)
    return torch.sum(weights * robust) / denominator


def r3_init_loss_torch(
    predicted_coco_body_m: Any,
    target_m: Any,
    weights: Any,
    scale_m: float = R3_SOFT_SCALE_M,
) -> Any:
    """Identical robust fit used by the short R3-initialization pre-optimisation."""
    return r3_soft_loss_torch(predicted_coco_body_m, target_m, weights, scale_m=scale_m)
