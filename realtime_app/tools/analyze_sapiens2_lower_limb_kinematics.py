#!/usr/bin/env python3
"""Extract non-contact lower-limb kinematic signals from observed 3-D runs.

The signals use only within-leg or between-foot relative geometry: shank to
forefoot angle and bilateral forefoot separation.  They are invariant to camera
translation but are not ground-contact labels.  Missing observations remain
missing; local extrema are reported only as kinematic turning points.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-sequence", type=Path, required=True)
    parser.add_argument("--filtered-sequence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or [int(row["pair_id"]) for row in rows] != list(range(len(rows))):
        raise RuntimeError(f"{path}: expected contiguous ordered pair IDs")
    return rows


def point(candidate: dict, filtered: dict, name: str) -> np.ndarray | None:
    source_group = candidate["sapiens2_hka"] if name in candidate["sapiens2_hka"] else candidate["sapiens2_distal_foot"]
    source = source_group[name]
    if not source.get("valid") or source.get("xyz_left_camera_mm") is None:
        return None
    derived = filtered["points"][name]["derived_filtered_xyz_left_camera_mm"]
    return np.asarray(derived, dtype=np.float64) if derived is not None else None


def angle_deg(a: np.ndarray, b: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-10:
        return None
    cosine = float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def finite_segments(values: list[float | None]) -> list[tuple[int, int]]:
    output = []
    start = None
    for index, value in enumerate(values + [None]):
        if value is not None and start is None:
            start = index
        elif value is None and start is not None:
            output.append((start, index - 1))
            start = None
    return output


def turning_points(values: list[float | None]) -> list[dict]:
    output = []
    for start, end in finite_segments(values):
        for index in range(start + 1, end):
            before, current, after = values[index - 1], values[index], values[index + 1]
            if current is None or before is None or after is None:
                continue
            if current > before and current >= after:
                output.append({"pair_id": index, "type": "local_max", "value": current})
            elif current < before and current <= after:
                output.append({"pair_id": index, "type": "local_min", "value": current})
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def render(rows: list[dict], output: Path, fps: float) -> None:
    import matplotlib.pyplot as plt

    time = [row["pair_id"] / fps for row in rows]
    left = [row["left_shank_forefoot_angle_deg"] for row in rows]
    right = [row["right_shank_forefoot_angle_deg"] for row in rows]
    separation = [row["bilateral_forefoot_separation_mm"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    axes[0].plot(time, left, color="#4c78a8", label="left shank–forefoot")
    axes[0].plot(time, right, color="#f58518", label="right shank–forefoot")
    axes[0].set_ylabel("angle (deg)")
    axes[0].set_title("Observed lower-limb relative-geometry signals")
    axes[0].legend(); axes[0].grid(alpha=0.25)
    axes[1].plot(time, separation, color="#54a24b", label="forefoot separation")
    axes[1].set_xlabel("time (s; 30 fps frame index)")
    axes[1].set_ylabel("distance (mm)")
    axes[1].legend(); axes[1].grid(alpha=0.25)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    candidate = load_jsonl(args.candidate_sequence.resolve())
    filtered = load_jsonl(args.filtered_sequence.resolve())
    if len(candidate) != len(filtered) or [row["file_name"] for row in candidate] != [row["file_name"] for row in filtered]:
        raise RuntimeError("Candidate and filtered sequences do not match")
    rows: list[dict] = []
    for record, derived in zip(candidate, filtered):
        left_knee, left_ankle = point(record, derived, "left_knee"), point(record, derived, "left_ankle")
        right_knee, right_ankle = point(record, derived, "right_knee"), point(record, derived, "right_ankle")
        left_big, left_small = point(record, derived, "left_big_toe"), point(record, derived, "left_small_toe")
        right_big, right_small = point(record, derived, "right_big_toe"), point(record, derived, "right_small_toe")
        left_fore = (left_big + left_small) / 2.0 if left_big is not None and left_small is not None else None
        right_fore = (right_big + right_small) / 2.0 if right_big is not None and right_small is not None else None
        left_angle = angle_deg(left_knee - left_ankle, left_fore - left_ankle) if left_knee is not None and left_ankle is not None and left_fore is not None else None
        right_angle = angle_deg(right_knee - right_ankle, right_fore - right_ankle) if right_knee is not None and right_ankle is not None and right_fore is not None else None
        separation = float(np.linalg.norm(left_fore - right_fore)) if left_fore is not None and right_fore is not None else None
        rows.append({
            "pair_id": int(record["pair_id"]), "file_name": record["file_name"],
            "time_sec": int(record["pair_id"]) / args.fps,
            "left_shank_forefoot_angle_deg": left_angle,
            "right_shank_forefoot_angle_deg": right_angle,
            "bilateral_forefoot_separation_mm": separation,
        })
    summary = []
    for name in ("left_shank_forefoot_angle_deg", "right_shank_forefoot_angle_deg", "bilateral_forefoot_separation_mm"):
        values = [float(row[name]) for row in rows if row[name] is not None]
        segments = finite_segments([row[name] for row in rows])
        summary.append({
            "signal": name,
            "available_frames": len(values),
            "median": float(np.median(values)) if values else None,
            "p10": float(np.percentile(values, 10)) if values else None,
            "p90": float(np.percentile(values, 90)) if values else None,
            "longest_contiguous_run_frames": max((end - start + 1 for start, end in segments), default=0),
            "longest_contiguous_run_sec": max((end - start + 1 for start, end in segments), default=0) / args.fps,
        })
    event_rows = []
    for name in ("left_shank_forefoot_angle_deg", "right_shank_forefoot_angle_deg"):
        for event in turning_points([row[name] for row in rows]):
            event_rows.append({"signal": name, "time_sec": event["pair_id"] / args.fps, **event})
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "per_frame_relative_kinematics.csv", rows)
    write_csv(args.output_dir / "kinematic_signal_summary.csv", summary)
    if event_rows:
        write_csv(args.output_dir / "kinematic_turning_points.csv", event_rows)
    render(rows, args.output_dir / "relative_kinematic_signals.png", args.fps)
    metadata = {
        "candidate_sequence": str(args.candidate_sequence.resolve()),
        "derived_short_window_sequence": str(args.filtered_sequence.resolve()),
        "fps": args.fps,
        "coordinate_frame": "left_camera",
        "length_unit": "mm",
        "interpretation": "Signals are relative lower-limb geometry only. They do not use a ground plane and local extrema are not labeled heel strike, toe-off, support, swing, or contact.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "README.md").write_text("# 下肢相对运动学信号\n\n本输出使用已观察到的膝、踝和前足点，计算小腿—前足夹角与双足前足间距。它们不依赖地面，适合检查连续下肢信息是否具有可读的动作变化；没有把任何局部极值解释为足地接触、支撑相或摆动相。\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "frames": len(rows), "turning_points": len(event_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
