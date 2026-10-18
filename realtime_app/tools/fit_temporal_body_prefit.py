#!/usr/bin/env python3
"""Fit a contact-aware fixed-bone temporal body scaffold before optional SMPL-X."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from pose_app.temporal_body_prefit import (  # noqa: E402
    TemporalBodyPrefitConfig,
    build_contact_targets,
    prefit_acceptance_audit,
    sequence_metrics,
    temporal_prefit,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-json", type=Path, required=True)
    parser.add_argument("--interaction-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)

    motion = json.loads(args.motion_json.read_text(encoding="utf-8-sig"))
    frames = motion["frames"]
    interactions = read_jsonl(args.interaction_jsonl)
    if len(frames) != len(interactions):
        raise ValueError("motion and interaction streams must align one-to-one")
    points = np.asarray([
        [point if point is not None else [np.nan, np.nan, np.nan] for point in frame["points_mm"]]
        for frame in frames
    ], dtype=np.float64)
    errors = np.asarray([
        [value if value is not None else np.nan for value in frame["errors_px"]]
        for frame in frames
    ], dtype=np.float64)
    strict = np.asarray([frame["strict"] for frame in frames], dtype=bool)
    config = TemporalBodyPrefitConfig(iterations=args.iterations)
    targets, contact_weights, contact_audit = build_contact_targets(
        points, frames, interactions, motion["walker"], config
    )
    started = time.perf_counter()
    fitted, optimizer_audit = temporal_prefit(
        points, errors, strict, targets, contact_weights, config=config, device=args.device
    )
    elapsed = time.perf_counter() - started
    metrics = sequence_metrics(points, fitted, targets, contact_weights)
    acceptance = prefit_acceptance_audit(points, fitted, errors, strict, metrics)

    records = []
    for index, (frame, fitted_points) in enumerate(zip(frames, fitted)):
        records.append({
            "pair_id": int(frame["pair"]),
            "stage": frame["stage"],
            "points_ground_mm": {
                str(joint): [round(float(value), 4) for value in point]
                for joint, point in enumerate(fitted_points)
            },
            "foot_contact_regularizer_active": {
                "left": bool(contact_weights[index, 15] > 0),
                "right": bool(contact_weights[index, 16] > 0),
            },
            "hand_contact_regularizer_active": {
                "left": bool(contact_weights[index, 9] > 0),
                "right": bool(contact_weights[index, 10] > 0),
            },
        })
    record_path = output / "temporal_body_prefit.jsonl"
    record_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
    )
    summary = {
        "schema_version": "contact_aware_temporal_body_prefit_v1",
        "status": "articulated_scaffold_prefit_not_smplx_mesh",
        "frame_count": len(frames),
        "device": args.device,
        "elapsed_seconds": elapsed,
        "config": config.__dict__,
        "metrics": metrics,
        "acceptance_audit": acceptance,
        "contact_audit": contact_audit,
        "optimizer_audit": optimizer_audit,
        "output_jsonl": str(record_path),
        "interpretation": (
            "Offline fixed-bone articulated prefit for continuity and soft contact regularization. "
            "It is an initialization/control for future licensed SMPL-X fitting, not a body mesh or accuracy truth."
        ),
        "contact_boundary": (
            "Stage-2 foot anchors are part of the camera-pose assumption and are not independent evidence. "
            "Hand constraints use visual wrist-to-coarse-handle proximity, not tactile or load sensing."
        ),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
