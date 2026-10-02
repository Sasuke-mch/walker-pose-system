from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
from walker_tools._compat import APP_ROOT as _tool_app_root
_tool_prepare_imports()

import argparse
import json
from pathlib import Path
import sys

import cv2


APP_ROOT = _tool_app_root

from pose_app.calibration import StereoCalibration
from pose_app.realtime_stage_walker import RealtimeStageWalkerWriter
from pose_app.schema import InferenceResult, PersonPose
from pose_app.sources import SourceFrame
from pose_app.stereo_sources import StereoFramePair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay saved PMPose results through the real-time two-stage/walker branch."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--stereo-results", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--processing-width", type=int, default=480)
    parser.add_argument("--reconstruction-interval", type=int, default=10)
    return parser.parse_args()


def inference(payload: dict, side: str) -> InferenceResult:
    value = payload[side]
    return InferenceResult(
        source_frame_id=int(value["source_frame_id"]),
        source_timestamp_sec=float(value["source_timestamp_sec"]),
        image_width=int(value["image_width"]),
        image_height=int(value["image_height"]),
        model_name=str(value["model_name"]),
        model_ms=float(value.get("model_ms", 0.0)),
        roundtrip_ms=float(value.get("roundtrip_ms", 0.0)),
        persons=[PersonPose.from_dict(item) for item in value.get("persons", [])],
        dropped_before=int(value.get("dropped_before", 0)),
        stage_times_ms=dict(value.get("stage_times_ms", {})),
    )


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = StereoCalibration.load(args.calibration)
    writer = RealtimeStageWalkerWriter(
        output_dir / "realtime_stage_walker.jsonl",
        calibration,
        processing_width=args.processing_width,
        reconstruction_interval=args.reconstruction_interval,
    )
    processed = 0
    with Path(args.stereo_results).resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            name = payload.get("file_name", f"pair_{int(payload['pair_id']):04d}.png")
            left_upright = cv2.imread(str(input_dir / "left_ccw90" / name), cv2.IMREAD_COLOR)
            right_upright = cv2.imread(str(input_dir / "right_cw90" / name), cv2.IMREAD_COLOR)
            if left_upright is None or right_upright is None:
                raise FileNotFoundError(f"Missing replay image pair: {name}")
            left_raw = cv2.rotate(left_upright, cv2.ROTATE_90_CLOCKWISE)
            right_raw = cv2.rotate(right_upright, cv2.ROTATE_90_COUNTERCLOCKWISE)
            timestamp = float(payload.get("pair_timestamp_sec", processed / 30.0))
            pair = StereoFramePair(
                pair_id=int(payload["pair_id"]),
                left=SourceFrame(int(payload.get("left_frame_id", processed)), timestamp, left_raw),
                right=SourceFrame(int(payload.get("right_frame_id", processed)), timestamp, right_raw),
                timestamp_skew_sec=float(payload.get("timestamp_skew_ms", 0.0)) / 1000.0,
                timestamp_type=str(payload.get("timestamp_type", "replay")),
            )
            writer.consume(pair, inference(payload, "left"), inference(payload, "right"))
            processed += 1
            if args.max_pairs is not None and processed >= args.max_pairs:
                break
    summary = writer.close(completed=True)
    summary.update(
        {
            "input_dir": str(input_dir),
            "stereo_results": str(Path(args.stereo_results).resolve()),
            "calibration": str(Path(args.calibration).resolve()),
            "image_orientation": "upright inputs inverse-rotated to raw camera pixels",
            "purpose": "offline replay of the same runtime branch; no inference rerun",
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
