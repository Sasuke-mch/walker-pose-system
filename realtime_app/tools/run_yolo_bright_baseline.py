"""Run the direct raw-fisheye YOLO baseline on the balanced bright 60-pair set."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(APP_ROOT))

from pose_app.rotation import ROTATION_CHOICES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-selection", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=12)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def instances_from_view(view: dict) -> list[dict]:
    output = []
    for person_id, (bbox, score, points, point_scores) in enumerate(
        zip(
            view["boxes_xyxy"],
            view["box_scores"],
            view["keypoints_xy"],
            view["keypoint_scores"],
        )
    ):
        if len(points) < 17 or len(point_scores) < 17:
            continue
        output.append(
            {
                "person_id": person_id,
                "bbox_xyxy": [float(value) for value in bbox[:4]],
                "bbox_score": float(score),
                "keypoints": [
                    [float(x), float(y), float(score)]
                    for (x, y), score in zip(points[:17], point_scores[:17])
                ],
            }
        )
    return output


def run_side(
    image_dir: Path,
    side: str,
    run_dir: Path,
    runner: Path,
    weights: Path,
    device: str,
    batch_size: int,
) -> tuple[dict, dict]:
    images = sorted(image_dir.glob("*.png"))
    if len(images) != 60:
        raise RuntimeError(f"Expected 60 {side} model images, found {len(images)}.")
    views: list[dict] = []
    runtime: dict | None = None
    for offset in range(0, len(images), batch_size):
        chunk = images[offset : offset + batch_size]
        output = run_dir / f"{side}_isolated_chunk_{offset // batch_size:03d}.json"
        subprocess.run(
            [
                sys.executable,
                str(runner),
                "--weights", str(weights),
                "--output", str(output),
                "--device", str(device),
                "--imgsz", "1280",
                "--conf", "0.05",
                "--iou", "0.7",
                "--max-det", "300",
                *[str(path) for path in chunk],
            ],
            check=True,
        )
        payload = json.loads(output.read_text(encoding="utf-8"))
        if len(payload.get("views", [])) != len(chunk):
            raise RuntimeError(f"{output.name} did not return one view per image.")
        views.extend(payload["views"])
        runtime = payload

    frames = [
        {"file_name": path.name, "frame_index": index, "instances": instances_from_view(view)}
        for index, (path, view) in enumerate(zip(images, views))
    ]
    predictions = {"frames": frames}
    output_path = run_dir / f"{side}_predictions.json"
    output_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    return predictions, runtime or {}


def condition_summaries(results: Path, manifest: list[dict]) -> dict[str, dict]:
    condition_by_name = {row["file_name"]: row["condition"] for row in manifest}
    groups: dict[str, list[dict]] = defaultdict(list)
    for line in results.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            groups[condition_by_name[record["file_name"]]].append(record)

    summaries: dict[str, dict] = {}
    for condition, records in groups.items():
        right_ankle_valid = 0
        reasons: Counter[str] = Counter()
        errors: list[float] = []
        for record in records:
            people = record["persons_3d"]
            if not people:
                reasons["no_stereo_person"] += 1
                continue
            ankle = next((point for point in people[0]["keypoints_3d"] if point["index"] == 16), None)
            if ankle is None:
                reasons["missing_right_ankle"] += 1
                continue
            if ankle["valid"]:
                right_ankle_valid += 1
            else:
                reasons[str(ankle["reason"])] += 1
            if ankle.get("reprojection_error_mean_px") is not None:
                errors.append(float(ankle["reprojection_error_mean_px"]))
        summaries[condition] = {
            "pairs": len(records),
            "right_ankle_valid_3d": right_ankle_valid,
            "right_ankle_valid_rate": right_ankle_valid / len(records),
            "right_ankle_rejections": dict(reasons),
            "right_ankle_mean_reprojection_error_px_when_available": (
                float(np.mean(errors)) if errors else None
            ),
        }
    return summaries


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    input_root = args.input_selection.resolve()
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {run_dir}")
    weights = PROJECT_ROOT / "models" / "yolo26" / "yolo26x-pose.pt"
    runner = Path(__file__).with_name("run_yolo_pose_isolated.py")
    evaluator = Path(__file__).with_name("evaluate_offline_stereo_predictions.py")
    calibration = APP_ROOT / "calibration" / "results" / "stereo_fisheye.json"
    for path in (weights, runner, evaluator, calibration, input_root / "dataset_summary.json"):
        if not path.is_file():
            raise FileNotFoundError(path)

    dataset = json.loads((input_root / "dataset_summary.json").read_text(encoding="utf-8"))
    left_rotation = dataset["model_input"]["left_rotation"]
    right_rotation = dataset["model_input"]["right_rotation"]
    if left_rotation not in ROTATION_CHOICES or right_rotation not in ROTATION_CHOICES:
        raise RuntimeError("Invalid model rotations in dataset summary.")
    with (input_root / "selection_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if len(manifest) != 60:
        raise RuntimeError("Expected exactly 60 selected pairs.")

    started = time.perf_counter()
    run_dir.mkdir(parents=True)
    _, left_runtime = run_side(
        input_root / f"left_{left_rotation}", "left", run_dir, runner, weights, args.device, args.batch_size
    )
    _, right_runtime = run_side(
        input_root / f"right_{right_rotation}", "right", run_dir, runner, weights, args.device, args.batch_size
    )
    geometry_dir = run_dir / "stereo_geometry"
    subprocess.run(
        [
            sys.executable, str(evaluator), "--model", "yolo26x_pose",
            "--left-json", str(run_dir / "left_predictions.json"),
            "--right-json", str(run_dir / "right_predictions.json"),
            "--calibration", str(calibration), "--output-dir", str(geometry_dir),
            "--left-model-rotation", left_rotation, "--right-model-rotation", right_rotation,
            "--keypoint-threshold", "0.25", "--max-association-cost", "0.05",
            "--max-reprojection-error-px", "10.0",
        ],
        check=True,
    )
    result_path = geometry_dir / "offline_stereo_results.jsonl"
    summary = {
        "experiment_id": "E20260829-M1_yolo26x_pose_raw_fisheye_host_isolated",
        "input_selection": str(input_root),
        "input_capture": dataset["capture_root"],
        "model": "yolo26x_pose",
        "execution_scope": "isolated local process; not Docker sequence_pipeline",
        "weights": str(weights),
        "weights_sha256": sha256(weights),
        "settings": {
            "imgsz": 1280, "candidate_conf": 0.05, "iou": 0.7, "max_det": 300,
            "left_rotation": left_rotation, "right_rotation": right_rotation,
            "geometry_keypoint_threshold": 0.25, "max_association_cost": 0.05,
            "max_reprojection_error_px": 10.0,
        },
        "input_geometry": "original_fisheye; rotations are inverse-mapped before triangulation",
        "runtime": {
            "seconds": time.perf_counter() - started,
            "left": {key: left_runtime.get(key) for key in ("torch_version", "cuda_available", "execution_scope")},
            "right": {key: right_runtime.get(key) for key in ("torch_version", "cuda_available", "execution_scope")},
        },
        "condition_summaries": condition_summaries(result_path, manifest),
        "accuracy_boundary": "2-D predictions and stereo self-consistency only; no manual 2-D labels or 3-D ground truth.",
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
