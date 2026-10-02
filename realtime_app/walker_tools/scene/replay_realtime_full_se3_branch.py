#!/usr/bin/env python3
"""Replay the exact causal full-SE(3) live branch over saved stereo frames."""

from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
_tool_prepare_imports()

import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import cv2
import numpy as np


APP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP_ROOT))

from pose_app.calibration import StereoCalibration  # noqa: E402
from pose_app.dynamic_ground_live import RealtimeDynamicGroundWriter  # noqa: E402
from pose_app.static_background_rotation import RealtimeStereoRotationTracker  # noqa: E402


STAGE2 = "stage2_feet_static_walker_moving"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def pose_result(person_rows: list[dict], side: str, upright_size: tuple[int, int]) -> SimpleNamespace:
    persons = [SimpleNamespace(
        keypoints=person.get("keypoints", []),
        pose_score=float(person.get("pose_score", 0.0)),
        bbox_score=float(person.get("bbox_score", 0.0)),
    ) for person in person_rows]
    return SimpleNamespace(persons=persons)


def stage_name(row: dict) -> str:
    value = row.get("stage") or {}
    return str(value.get("operational") or value.get("confirmed") or "unknown")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--stereo-jsonl", type=Path, required=True)
    parser.add_argument("--stage-jsonl", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--ground-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--processing-width", type=int, default=480)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    stereo, stages = rows(args.stereo_jsonl), rows(args.stage_jsonl)
    if len(stereo) != len(stages):
        raise ValueError("stereo and stage row counts differ")
    calibration = StereoCalibration.load(args.calibration)
    tracker = RealtimeStereoRotationTracker(calibration, processing_width=args.processing_width)
    writer = RealtimeDynamicGroundWriter(
        output / "realtime_dynamic_ground_pose.jsonl", args.ground_reference, pose_mode="full_se3",
        stage1_landed_support_reanchor=True,
    )
    rotation_records = []
    branch_ms = []
    rotation_statuses: Counter[str] = Counter()
    for payload, stage_row in zip(stereo, stages):
        pair_id = int(payload["pair_id"])
        if pair_id != int(stage_row["pair_id"]):
            raise ValueError(f"pair mismatch at {pair_id}")
        name = str(payload.get("file_name", f"pair_{pair_id:04d}.png"))
        left_upright = cv2.imread(str(args.input_dir / "left_ccw90" / name), cv2.IMREAD_COLOR)
        right_upright = cv2.imread(str(args.input_dir / "right_cw90" / name), cv2.IMREAD_COLOR)
        if left_upright is None or right_upright is None:
            raise FileNotFoundError(name)
        upright_size = (left_upright.shape[1], left_upright.shape[0])
        left = cv2.rotate(left_upright, cv2.ROTATE_90_CLOCKWISE)
        right = cv2.rotate(right_upright, cv2.ROTATE_90_COUNTERCLOCKWISE)
        left_result = pose_result(payload["left"].get("persons", []), "left", upright_size)
        right_result = pose_result(payload["right"].get("persons", []), "right", upright_size)
        stage = stage_name(stage_row)
        started = time.perf_counter()
        rotation = tracker.update(
            left, right, left_result, right_result, estimate=stage in {STAGE2, "transition"},
        )
        pose = writer.consume(payload, stage_row, rotation)
        branch_ms.append((time.perf_counter() - started) * 1000.0)
        rotation_statuses[rotation["status"]] += 1
        rotation_records.append({
            "pair_id": pair_id, "stage": stage, "rotation": rotation,
            "pose_status": pose["pose_audit"].get("status"), "branch_processing_ms": branch_ms[-1],
        })
    writer_summary = writer.close(completed=True)
    rotation_path = output / "rotation_and_pose_status.jsonl"
    rotation_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rotation_records), encoding="utf-8"
    )
    values = np.asarray(branch_ms)
    summary = {
        "schema_version": "realtime_full_se3_branch_replay_v2",
        "frame_count": len(stereo), "rotation_status_counts": dict(rotation_statuses),
        "branch_processing_ms": {
            "mean": float(values.mean()), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "maximum": float(values.max()),
            "scope": "image resize and masks + stage2 stereo rotation consensus when requested + full-SE3 pose fusion; image decode excluded",
        },
        "writer": writer_summary,
        "production_eligible": False,
        "boundary": "saved-frame software replay; no physical live camera timing or external trajectory truth",
        "rotation_records": str(rotation_path.resolve()),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
