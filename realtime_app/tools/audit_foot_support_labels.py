#!/usr/bin/env python3
"""Audit current-run foot support labels without fitting or filling invalid data.

The input is the current run's motion JSON. No historical contact target,
walker pose, or fitted parameter file is read. The output is an audit artifact
that can be used as the only contact-label input for a later foot-only window.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ANKLES = {"left": 15, "right": 16}
STAGE1 = {"stage1", "stage_1", "Stage 1"}
STAGE2 = {"stage2", "stage_2", "Stage 2"}
TRANSITION = {"transition", "warmup", "warming_up", "unknown", "unavailable"}


def _stage(frame: dict) -> str:
    value = str(frame.get("stage", "unknown"))
    lowered = value.lower().replace(" ", "_")
    if lowered in {"stage_1", "stage1"}:
        return "stage1"
    if lowered in {"stage_2", "stage2"}:
        return "stage2"
    return "transition"


def _points(frame: dict) -> np.ndarray:
    values = frame.get("points_ground_mm", frame.get("points_mm"))
    if isinstance(values, dict):
        values = [values.get(str(index)) for index in range(17)]
    result = np.full((17, 3), np.nan, dtype=np.float64)
    for index, value in enumerate(values or []):
        if index < 17 and value is not None and len(value) == 3:
            result[index] = np.asarray(value, dtype=np.float64)
    return result


def _valid(frame: dict, points: np.ndarray, joint: int) -> bool:
    if not np.isfinite(points[joint]).all():
        return False
    strict = frame.get("strict")
    if isinstance(strict, list) and joint < len(strict) and not bool(strict[joint]):
        return False
    return True


def _segments(mask: np.ndarray, minimum: int = 3) -> np.ndarray:
    result = np.zeros_like(mask, dtype=bool)
    start = None
    for index, active in enumerate(np.r_[mask, False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start >= minimum:
                result[start:index] = True
            start = None
    return result


def _score(speed: float, height: float, speed_limit: float, height_limit: float) -> float:
    if not np.isfinite(speed) or not np.isfinite(height):
        return 0.0
    velocity_score = max(0.0, 1.0 - speed / max(speed_limit, 1e-6))
    height_score = max(0.0, 1.0 - max(height - height_limit, 0.0) / max(abs(height_limit), 1.0))
    return float(np.clip(velocity_score * height_score, 0.0, 1.0))


def build_audit(motion: dict, minimum_support_frames: int = 3) -> tuple[list[dict], dict]:
    frames = motion["frames"]
    points = np.stack([_points(frame) for frame in frames])
    stages = np.asarray([_stage(frame) for frame in frames], dtype=object)
    speeds = np.full((len(frames), 2), np.nan, dtype=np.float64)
    for side_index, joint in enumerate(ANKLES.values()):
        deltas = np.linalg.norm(np.diff(points[:, joint], axis=0), axis=1)
        valid_delta = np.isfinite(points[:-1, joint]).all(axis=1) & np.isfinite(points[1:, joint]).all(axis=1)
        speeds[1:, side_index] = np.where(valid_delta, deltas, np.nan)

    limits: dict[str, dict[str, float]] = {}
    for side_index, (side, joint) in enumerate(ANKLES.items()):
        stage1 = stages == "stage1"
        valid = np.isfinite(points[:, joint]).all(axis=1) & stage1
        speed_limit = 25.0
        if np.any(np.isfinite(speeds[stage1, side_index])):
            speed_limit = max(15.0, float(np.nanpercentile(speeds[stage1, side_index], 35)))
        heights = points[valid, joint, 2]
        height_limit = float(np.nanpercentile(heights, 35) + 20.0) if len(heights) else np.nan
        raw_support = valid & np.isfinite(speeds[:, side_index]) & (speeds[:, side_index] <= speed_limit)
        raw_support &= points[:, joint, 2] <= height_limit
        support = _segments(raw_support, minimum_support_frames)
        limits[side] = {
            "speed_limit_mm_per_frame": speed_limit,
            "height_limit_mm": height_limit,
            "raw_support_count": int(raw_support.sum()),
            "support_count": int(support.sum()),
        }
        limits[side]["support_mask"] = support

    rows: list[dict] = []
    for index, frame in enumerate(frames):
        stage = stages[index]
        left_valid = _valid(frame, points[index], 15)
        right_valid = _valid(frame, points[index], 16)
        stage2_candidate = stage == "stage2" and left_valid and right_valid
        foot_rows: dict[str, dict] = {}
        for side_index, (side, joint) in enumerate(ANKLES.items()):
            valid = _valid(frame, points[index], joint)
            speed = float(speeds[index, side_index]) if np.isfinite(speeds[index, side_index]) else None
            height = float(points[index, joint, 2]) if valid else None
            if not valid:
                label = "invalid"
            elif stage != "stage1":
                label = "stage2_contact_candidate" if stage2_candidate else "ambiguous"
            elif bool(limits[side]["support_mask"][index]):
                label = "support"
            elif speed is not None and speed > limits[side]["speed_limit_mm_per_frame"]:
                label = "swing"
            else:
                label = "ambiguous"
            weight = 0.0
            if label == "support":
                weight = _score(speed, height, limits[side]["speed_limit_mm_per_frame"], limits[side]["height_limit_mm"])
            elif label == "stage2_contact_candidate":
                weight = 0.25 * _score(speed, height, 35.0, float(np.nanpercentile(points[:, joint, 2], 35) + 25.0))
            foot_rows[side] = {
                "label": label,
                "velocity_mm_per_frame": speed,
                "height_mm": height,
                "three_d_valid": valid,
                "contact_weight": float(max(weight, 0.0)),
                "ground_position_mm": points[index, joint].tolist() if valid else None,
            }
        rows.append({
            "frame_index": index,
            "pair_id": frame.get("pair"),
            "stage": stage,
            "left": foot_rows["left"],
            "right": foot_rows["right"],
            "stage2_both_feet_candidate": bool(stage2_candidate),
            "contact_enabled": stage in {"stage1", "stage2"} and not stage == "transition",
        })

    summary = {
        "schema_version": "foot_support_audit_v1",
        "status": "audit_only",
        "input_contract": "current_run_motion_only; no_old_contact_targets; no_interpolation",
        "frame_count": len(rows),
        "stage_counts": {stage: int(np.sum(stages == stage)) for stage in ("stage1", "stage2", "transition")},
        "label_counts": {
            side: {label: int(sum(row[side]["label"] == label for row in rows))
                   for label in ("support", "swing", "ambiguous", "invalid", "stage2_contact_candidate")}
            for side in ANKLES
        },
        "thresholds": {
            side: {key: value for key, value in data.items() if key != "support_mask"}
            for side, data in limits.items()
        },
        "stage2_both_feet_candidate_count": int(sum(row["stage2_both_feet_candidate"] for row in rows)),
        "transition_contact_enabled_count": int(sum(row["stage"] == "transition" and row["contact_enabled"] for row in rows)),
    }
    return rows, summary


def choose_window(rows: list[dict], candidates: list[tuple[int, int]]) -> dict:
    evaluations = []
    for start, end in candidates:
        selected = [row for row in rows if start <= row["frame_index"] <= end]
        stage1 = sum(row["stage"] == "stage1" for row in selected)
        stage2 = sum(row["stage"] == "stage2" for row in selected)
        transitions = sum(row["stage"] == "transition" for row in selected)
        supports = sum(row[side]["label"] == "support" for row in selected for side in ANKLES)
        both = sum(row["stage2_both_feet_candidate"] for row in selected)
        valid_ankles = sum(row[side]["three_d_valid"] for row in selected for side in ANKLES)
        complete = len(selected) == end - start + 1
        passed = complete and stage1 > 0 and stage2 > 0 and transitions > 0 and valid_ankles > 0 and (supports >= 2 or both >= 3)
        evaluations.append({"start": start, "end": end, "stage1_frames": stage1, "stage2_frames": stage2,
                            "transition_frames": transitions, "support_labels": supports,
                            "stage2_both_feet_candidates": both, "valid_ankle_observations": valid_ankles,
                            "complete": complete, "eligible": passed})
    chosen = next((item for item in evaluations if item["eligible"]), None)
    return {"candidates": evaluations, "selected": chosen,
            "selection_status": "selected" if chosen else "blocked_no_window_meets_protocol"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", action="append", default=[], metavar="START:END")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refuse to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    motion = json.loads(args.motion_json.read_text(encoding="utf-8-sig"))
    rows, summary = build_audit(motion)
    candidates = []
    for value in args.window or ["60:90"]:
        start, end = (int(part) for part in value.split(":", 1))
        candidates.append((start, end))
    selection = choose_window(rows, candidates)
    payload = {**summary, "input_motion_json": str(args.motion_json.resolve()), "window_selection": selection}
    (out / "contact_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "foot_support_labels.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    print(json.dumps({"summary": summary, "window_selection": selection}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
