"""Create a balanced, traceable 60-pair input set from the new bright capture.

The output contains both raw fisheye PNGs and model-oriented copies.  Rotation
is only an orientation normalization for 2-D models; geometry always uses the
raw fisheye images and inverse-mapped keypoints.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROTATIONS = {
    "none": None,
    "cw90": cv2.ROTATE_90_CLOCKWISE,
    "ccw90": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180": cv2.ROTATE_180,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare balanced matched pairs from far/mid/near bright stereo captures."
    )
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pairs-per-condition", type=int, default=20)
    parser.add_argument("--max-pair-delta-ms", type=float, default=15.0)
    parser.add_argument("--left-rotation", choices=ROTATIONS, default="cw90")
    parser.add_argument("--right-rotation", choices=ROTATIONS, default="ccw90")
    return parser.parse_args()


def rotate(image: np.ndarray, mode: str) -> np.ndarray:
    operation = ROTATIONS[mode]
    return image if operation is None else cv2.rotate(image, operation)


def find_session(condition_dir: Path) -> Path:
    sessions = [path for path in condition_dir.iterdir() if path.is_dir()]
    if len(sessions) != 1:
        raise RuntimeError(
            f"Expected exactly one session under {condition_dir}; found {len(sessions)}."
        )
    return sessions[0]


def choose_evenly(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if len(rows) < count:
        raise RuntimeError(f"Need {count} eligible pairs, only found {len(rows)}.")
    indices = np.linspace(0, len(rows) - 1, count, dtype=int)
    if len(set(int(value) for value in indices)) != count:
        raise RuntimeError("Uniform selection created duplicate pair positions.")
    return [rows[int(index)] for index in indices]


def read_frame(capture: cv2.VideoCapture, frame_id: int, side: str) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, image = capture.read()
    if not ok or image is None:
        raise RuntimeError(f"Cannot read {side} source frame {frame_id}.")
    return image


def load_video_frame_positions(session: Path, side: str) -> dict[int, int]:
    """Map camera frame_id to its zero-based position in the recorded AVI.

    Capture frame IDs intentionally survive the pre-flight warm-up, while the
    AVI begins only at the formal start signal.  The sidecar CSV is therefore
    the authoritative bridge from a stereo-pair camera frame ID to an AVI
    seek position.
    """

    csv_path = session / f"{side}_frames.csv"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    positions = {int(row["frame_id"]): index for index, row in enumerate(rows)}
    if len(positions) != len(rows):
        raise RuntimeError(f"Duplicate {side} frame_id values in {csv_path}.")
    return positions


def require_capture_mapping(metadata: dict) -> None:
    mapping = metadata.get("calibration_camera_mapping", {})
    left = mapping.get("left", {})
    right = mapping.get("right", {})
    if left.get("logical_camera") != "cam0" or right.get("logical_camera") != "cam1":
        raise RuntimeError("Capture metadata does not preserve cam0=LEFT and cam1=RIGHT.")
    if mapping.get("selection_mode") != "physical_registry":
        raise RuntimeError("Capture was not resolved by the physical camera registry.")
    if metadata.get("capture_start_control", {}).get("pre_roll_recorded") is not False:
        raise RuntimeError("Capture metadata does not certify exclusion of pre-roll frames.")


def main() -> int:
    args = parse_args()
    args.capture_root = args.capture_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.pairs_per_condition <= 0:
        raise ValueError("--pairs-per-condition must be positive.")
    if args.max_pair_delta_ms <= 0:
        raise ValueError("--max-pair-delta-ms must be positive.")
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")

    conditions = ("far_3m", "mid_2m", "near_1p3m")
    sessions = {name: find_session(args.capture_root / name) for name in conditions}

    raw_left_dir = args.output_dir / "raw_fisheye" / "left"
    raw_right_dir = args.output_dir / "raw_fisheye" / "right"
    model_left_dir = args.output_dir / f"left_{args.left_rotation}"
    model_right_dir = args.output_dir / f"right_{args.right_rotation}"
    for directory in (raw_left_dir, raw_right_dir, model_left_dir, model_right_dir):
        directory.mkdir(parents=True)

    manifest: list[dict[str, object]] = []
    global_index = 0
    source_summary: dict[str, object] = {}

    for condition in conditions:
        session = sessions[condition]
        metadata = json.loads((session / "metadata.json").read_text(encoding="utf-8-sig"))
        require_capture_mapping(metadata)
        with (session / "stereo_pairs.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        eligible = [
            row for row in rows if float(row["abs_host_delta_ms"]) <= args.max_pair_delta_ms
        ]
        selected = choose_evenly(eligible, args.pairs_per_condition)
        left_positions = load_video_frame_positions(session, "left")
        right_positions = load_video_frame_positions(session, "right")

        left_capture = cv2.VideoCapture(str(session / "left_capture.avi"))
        right_capture = cv2.VideoCapture(str(session / "right_capture.avi"))
        if not left_capture.isOpened() or not right_capture.isOpened():
            raise RuntimeError(f"Cannot open both AVI files for {condition}.")
        try:
            for condition_index, row in enumerate(selected):
                left_id = int(row["left_frame_id"])
                right_id = int(row["right_frame_id"])
                if left_id not in left_positions or right_id not in right_positions:
                    raise RuntimeError(
                        f"Pair references a frame not recorded after the start gate: "
                        f"left={left_id in left_positions}, right={right_id in right_positions}."
                    )
                left_video_index = left_positions[left_id]
                right_video_index = right_positions[right_id]
                left_raw = read_frame(left_capture, left_video_index, "left")
                right_raw = read_frame(right_capture, right_video_index, "right")
                if left_raw.shape != right_raw.shape:
                    raise RuntimeError(
                        f"Raw shape mismatch at {condition}/{condition_index}: "
                        f"left={left_raw.shape}, right={right_raw.shape}."
                    )

                name = f"pair_{global_index:03d}.png"
                for path, image in (
                    (raw_left_dir / name, left_raw),
                    (raw_right_dir / name, right_raw),
                    (model_left_dir / name, rotate(left_raw, args.left_rotation)),
                    (model_right_dir / name, rotate(right_raw, args.right_rotation)),
                ):
                    if not cv2.imwrite(str(path), image):
                        raise RuntimeError(f"Cannot write {path}.")

                manifest.append(
                    {
                        "global_index": global_index,
                        "file_name": name,
                        "condition": condition,
                        "condition_index": condition_index,
                        "source_session_dir": str(session),
                        "source_pair_id": int(row["pair_id"]),
                        "left_frame_id": left_id,
                        "right_frame_id": right_id,
                        "left_video_frame_index": left_video_index,
                        "right_video_frame_index": right_video_index,
                        "signed_host_delta_ms_right_minus_left": float(
                            row["signed_host_delta_ms_right_minus_left"]
                        ),
                        "abs_host_delta_ms": float(row["abs_host_delta_ms"]),
                        "raw_width": int(left_raw.shape[1]),
                        "raw_height": int(left_raw.shape[0]),
                    }
                )
                global_index += 1
        finally:
            left_capture.release()
            right_capture.release()

        source_summary[condition] = {
            "session_dir": str(session),
            "recorded_pairs": len(rows),
            "eligible_pairs": len(eligible),
            "selected_pairs": len(selected),
        }

    with (args.output_dir / "selection_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    summary = {
        "capture_root": str(args.capture_root),
        "conditions": source_summary,
        "total_selected_pairs": len(manifest),
        "max_pair_delta_ms": args.max_pair_delta_ms,
        "raw_input": {
            "left_dir": str(raw_left_dir),
            "right_dir": str(raw_right_dir),
            "camera_model": "original_fisheye",
        },
        "model_input": {
            "left_dir": str(model_left_dir),
            "right_dir": str(model_right_dir),
            "left_rotation": args.left_rotation,
            "right_rotation": args.right_rotation,
            "note": "Rotation is orientation-only; 2-D predictions must be inverse-mapped to raw fisheye pixels.",
        },
        "selection_manifest": str(args.output_dir / "selection_manifest.csv"),
    }
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
