"""T4: conservative event and parameter candidates from direct T2 signals."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any

import numpy as np


KINEMATIC_SIGNALS: tuple[str, ...] = (
    "left_knee_angle_deg",
    "right_knee_angle_deg",
    "ankle_separation_mm",
)


def _metric_value(record: Mapping[str, Any], signal: str) -> float | None:
    metric = record.get("metrics", {}).get(signal)
    if not isinstance(metric, Mapping) or not metric.get("available"):
        return None
    try:
        value = float(metric.get("value"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _metric_unit(record: Mapping[str, Any], signal: str) -> str:
    metric = record.get("metrics", {}).get(signal)
    if isinstance(metric, Mapping) and isinstance(metric.get("unit"), str):
        return metric["unit"]
    raise ValueError(f"missing unit for {signal}")


def _validate_records(records: list[Mapping[str, Any]]) -> None:
    if not records:
        raise ValueError("kinematics stream is empty")
    previous_pair_id: int | None = None
    for record in records:
        if record.get("coordinate_frame") != "left_camera":
            raise ValueError("T4 currently consumes T2 left_camera records")
        if record.get("length_unit") != "millimeter":
            raise ValueError("T4 currently consumes millimeter T2 records")
        pair_id = int(record["pair_id"])
        if previous_pair_id is not None and pair_id <= previous_pair_id:
            raise ValueError("pair_id values must be strictly increasing")
        previous_pair_id = pair_id


def _strict_turning_points(records: list[Mapping[str, Any]], signal: str) -> list[dict[str, Any]]:
    """Find a raw three-frame turning point without smoothing across a gap."""

    output: list[dict[str, Any]] = []
    for before, current, after in zip(records, records[1:], records[2:]):
        pair_ids = (int(before["pair_id"]), int(current["pair_id"]), int(after["pair_id"]))
        if pair_ids[1] != pair_ids[0] + 1 or pair_ids[2] != pair_ids[1] + 1:
            continue
        values = (_metric_value(before, signal), _metric_value(current, signal), _metric_value(after, signal))
        if any(value is None for value in values):
            continue
        first, middle, last = values
        if middle > first and middle > last:
            turning_point = "strict_local_max"
        elif middle < first and middle < last:
            turning_point = "strict_local_min"
        else:
            continue
        timestamp = current.get("pair_timestamp_sec")
        if timestamp is not None:
            timestamp = float(timestamp)
            if not math.isfinite(timestamp):
                timestamp = None
        output.append(
            {
                "candidate_id": f"{signal}:{turning_point}:pair_{pair_ids[1]}",
                "candidate_type": "kinematic_turning_point_candidate",
                "signal": signal,
                "turning_point": turning_point,
                "pair_id": pair_ids[1],
                "pair_timestamp_sec": timestamp,
                "value": middle,
                "unit": _metric_unit(current, signal),
                "support_pair_ids": list(pair_ids),
                "source_observation_policy": "direct_accepted_stereo_only",
                "interpretation": (
                    "Raw kinematic turning point only; not heel strike, toe-off, "
                    "ground contact, support, swing, step, or gait-cycle label."
                ),
            }
        )
    return output


def derive_gait_candidates(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Create a deliberately non-contact T4 contract from T2 records."""

    source = list(records)
    _validate_records(source)
    events = [event for signal in KINEMATIC_SIGNALS for event in _strict_turning_points(source, signal)]
    events.sort(key=lambda event: (event["pair_id"], event["signal"], event["turning_point"]))

    signal_summary: dict[str, dict[str, Any]] = {}
    for signal in KINEMATIC_SIGNALS:
        values = [_metric_value(record, signal) for record in source]
        available = [value for value in values if value is not None]
        signal_events = [event for event in events if event["signal"] == signal]
        signal_summary[signal] = {
            "available_frames": len(available),
            "unavailable_frames": len(source) - len(available),
            "unit": _metric_unit(next(record for record in source if signal in record["metrics"]), signal),
            "minimum_direct_observation": float(np.min(available)) if available else None,
            "maximum_direct_observation": float(np.max(available)) if available else None,
            "kinematic_turning_point_candidates": len(signal_events),
            "interpretation": "Observed signal range and raw turning points; not a gait parameter.",
        }

    unavailable_parameters = {
        "gait_cycle_duration_sec": "no_valid_contact_event_definition",
        "cadence_steps_per_min": "no_valid_contact_event_definition",
        "step_length_mm": "target_coordinate_frame_not_measured_locked",
        "step_width_mm": "target_coordinate_frame_not_measured_locked",
        "foot_ground_height_mm": "target_coordinate_frame_not_measured_locked",
    }
    parameter_candidates = {
        name: {
            "available": False,
            "value": None,
            "unit": "second" if name.endswith("_sec") else ("steps_per_min" if name.endswith("per_min") else "millimeter"),
            "reason": reason,
            "interpretation": "Unavailable by design in the current T4 contract.",
        }
        for name, reason in unavailable_parameters.items()
    }
    return {
        "records": source,
        "events": events,
        "summary": {
            "frames": len(source),
            "first_pair_id": int(source[0]["pair_id"]),
            "last_pair_id": int(source[-1]["pair_id"]),
            "event_candidates": len(events),
            "accepted_contact_events": 0,
            "signal_summary": signal_summary,
            "gait_parameter_candidates": parameter_candidates,
            "interpretation": (
                "Raw direct-observation kinematic turning-point candidates only. "
                "No smoothing, interpolation, contact inference, step/stride calculation, "
                "or accuracy claim is made."
            ),
        },
    }
