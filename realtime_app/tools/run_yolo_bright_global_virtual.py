"""Repeat M1 YOLO pose screening with only a global virtual-pinhole input added.

Raw-fisheye M1 remains the control.  This U1-style condition uses the same
weights and detection/geometry thresholds, maps every prediction back into
raw fisheye pixel coordinates, then triangulates with the original fisheye
calibration.  It must not be confused with a new calibrated camera.
"""
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

import cv2
import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(APP_ROOT))

from pose_app.calibration import StereoCalibration  # noqa: E402
from pose_app.model_undistort import FisheyeModelInput  # noqa: E402
from pose_app.rotation import model_to_raw_point, rotate_image_for_model  # noqa: E402


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


def load_calibration(path: Path) -> StereoCalibration:
    cwd = Path.cwd()
    try:
        # New-format calibration has relative intrinsic references.
        import os
        os.chdir(APP_ROOT)
        return StereoCalibration.load(path).for_runtime_sizes((1920, 1080), (1920, 1080))
    finally:
        os.chdir(cwd)


def transformed_bbox(bbox: list[float], mapper, rotation: str, raw_size: tuple[int, int]) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    # A nonlinear fisheye inverse is not represented by four corners alone.
    points = []
    for x in np.linspace(x1, x2, 5):
        for y in np.linspace(y1, y2, 5):
            ux, uy = model_to_raw_point(x, y, *raw_size, rotation)
            points.append(mapper(ux, uy))
    xs, ys = zip(*points)
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def map_view_to_raw(view: dict, mapper, rotation: str, raw_size: tuple[int, int]) -> list[dict]:
    result = []
    for person_id, (bbox, score, points, scores) in enumerate(zip(
        view["boxes_xyxy"], view["box_scores"], view["keypoints_xy"], view["keypoint_scores"]
    )):
        if len(points) < 17 or len(scores) < 17:
            continue
        raw_points = []
        for (x, y), point_score in zip(points[:17], scores[:17]):
            ux, uy = model_to_raw_point(float(x), float(y), *raw_size, rotation)
            rx, ry = mapper(ux, uy)
            raw_points.append([float(rx), float(ry), float(point_score)])
        result.append({
            "person_id": person_id,
            "bbox_xyxy": transformed_bbox(bbox, mapper, rotation, raw_size),
            "bbox_score": float(score),
            "keypoints": raw_points,
        })
    return result


def prepare_side(selection: Path, run_dir: Path, side: str, rotation: str, preprocessor: FisheyeModelInput) -> Path:
    raw_dir = selection / "raw_fisheye" / side
    output_dir = run_dir / "virtual_model_input" / side
    output_dir.mkdir(parents=True)
    for raw_path in sorted(raw_dir.glob("*.png")):
        raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if raw is None:
            raise RuntimeError(f"Could not read {raw_path}")
        undistorted = preprocessor.image(raw)
        model_image = rotate_image_for_model(undistorted, rotation)
        if not cv2.imwrite(str(output_dir / raw_path.name), model_image):
            raise RuntimeError(f"Could not write virtual input for {raw_path.name}")
    if len(list(output_dir.glob("*.png"))) != 60:
        raise RuntimeError(f"Expected 60 virtual images for {side}")
    return output_dir


def run_side(image_dir: Path, side: str, run_dir: Path, runner: Path, weights: Path,
             device: str, batch_size: int, mapper, rotation: str) -> dict:
    images = sorted(image_dir.glob("*.png"))
    views: list[dict] = []
    runtime: dict | None = None
    for offset in range(0, len(images), batch_size):
        chunk = images[offset: offset + batch_size]
        output = run_dir / f"{side}_isolated_chunk_{offset // batch_size:03d}.json"
        subprocess.run([
            sys.executable, str(runner), "--weights", str(weights), "--output", str(output),
            "--device", str(device), "--imgsz", "1280", "--conf", "0.05", "--iou", "0.7",
            "--max-det", "300", *[str(path) for path in chunk],
        ], check=True)
        payload = json.loads(output.read_text(encoding="utf-8"))
        if len(payload.get("views", [])) != len(chunk):
            raise RuntimeError(f"{output.name} did not return one view per image.")
        views.extend(payload["views"])
        runtime = payload
    predictions = {"frames": [
        {"file_name": path.name, "frame_index": index,
         "instances": map_view_to_raw(view, mapper, rotation, (1920, 1080))}
        for index, (path, view) in enumerate(zip(images, views))
    ]}
    (run_dir / f"{side}_predictions_raw_fisheye.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return runtime or {}


def condition_summaries(results: Path, manifest: list[dict]) -> dict[str, dict]:
    condition_by_name = {row["file_name"]: row["condition"] for row in manifest}
    groups: dict[str, list[dict]] = defaultdict(list)
    for line in results.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            groups[condition_by_name[record["file_name"]]].append(record)
    summaries: dict[str, dict] = {}
    for condition, records in groups.items():
        valid, reasons, errors = 0, Counter(), []
        for record in records:
            if not record["persons_3d"]:
                reasons["no_stereo_person"] += 1
                continue
            ankle = next((p for p in record["persons_3d"][0]["keypoints_3d"] if p["index"] == 16), None)
            if ankle is None:
                reasons["missing_right_ankle"] += 1
                continue
            if ankle["valid"]:
                valid += 1
            else:
                reasons[str(ankle["reason"])] += 1
            if ankle.get("reprojection_error_mean_px") is not None:
                errors.append(float(ankle["reprojection_error_mean_px"]))
        summaries[condition] = {
            "pairs": len(records), "right_ankle_valid_3d": valid,
            "right_ankle_valid_rate": valid / len(records),
            "right_ankle_rejections": dict(reasons),
            "right_ankle_mean_reprojection_error_px_when_available": float(np.mean(errors)) if errors else None,
        }
    return summaries


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    selection, run_dir = args.input_selection.resolve(), args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {run_dir}")
    weights = PROJECT_ROOT / "models" / "yolo26" / "yolo26x-pose.pt"
    runner = Path(__file__).with_name("run_yolo_pose_isolated.py")
    evaluator = Path(__file__).with_name("evaluate_offline_stereo_predictions.py")
    calibration_path = APP_ROOT / "calibration" / "results" / "stereo_fisheye.json"
    dataset = json.loads((selection / "dataset_summary.json").read_text(encoding="utf-8"))
    with (selection / "selection_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if len(manifest) != 60:
        raise RuntimeError("Expected exactly 60 selected pairs")
    for path in (weights, runner, evaluator, calibration_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    calibration = load_calibration(calibration_path)
    left_rotation, right_rotation = dataset["model_input"]["left_rotation"], dataset["model_input"]["right_rotation"]
    run_dir.mkdir(parents=True)
    started = time.perf_counter()
    left_input = prepare_side(selection, run_dir, "left", left_rotation,
                              FisheyeModelInput(calibration.left_K, calibration.left_D, (1920, 1080)))
    right_input = prepare_side(selection, run_dir, "right", right_rotation,
                               FisheyeModelInput(calibration.right_K, calibration.right_D, (1920, 1080)))
    left_runtime = run_side(left_input, "left", run_dir, runner, weights, args.device, args.batch_size,
                            FisheyeModelInput(calibration.left_K, calibration.left_D, (1920, 1080)).point_mapper(), left_rotation)
    right_runtime = run_side(right_input, "right", run_dir, runner, weights, args.device, args.batch_size,
                             FisheyeModelInput(calibration.right_K, calibration.right_D, (1920, 1080)).point_mapper(), right_rotation)
    geometry_dir = run_dir / "stereo_geometry"
    subprocess.run([
        sys.executable, str(evaluator), "--model", "yolo26x_pose",
        "--left-json", str(run_dir / "left_predictions_raw_fisheye.json"),
        "--right-json", str(run_dir / "right_predictions_raw_fisheye.json"),
        "--calibration", str(calibration_path), "--output-dir", str(geometry_dir),
        # Coordinates have already been restored to raw fisheye pixels.
        "--left-model-rotation", "none", "--right-model-rotation", "none",
        "--keypoint-threshold", "0.25", "--max-association-cost", "0.05", "--max-reprojection-error-px", "10.0",
    ], check=True)
    result_path = geometry_dir / "offline_stereo_results.jsonl"
    summary = {
        "experiment_id": "E20260829-U1_yolo26x_pose_global_virtual_pinhole_host_isolated",
        "input_selection": str(selection), "input_capture": dataset["capture_root"],
        "model": "yolo26x_pose", "weights": str(weights), "weights_sha256": sha256(weights),
        "execution_scope": "isolated local process; not Docker sequence_pipeline",
        "settings": {"imgsz": 1280, "candidate_conf": 0.05, "iou": 0.7, "max_det": 300,
                     "left_rotation": left_rotation, "right_rotation": right_rotation,
                     "geometry_keypoint_threshold": 0.25, "max_association_cost": 0.05,
                     "max_reprojection_error_px": 10.0},
        "input_geometry": "global fisheye-to-pinhole remap for model input only; all detections inverse-mapped to raw fisheye pixels before original-calibration triangulation",
        "runtime": {"seconds": time.perf_counter() - started,
                    "left": {key: left_runtime.get(key) for key in ("torch_version", "cuda_available", "execution_scope")},
                    "right": {key: right_runtime.get(key) for key in ("torch_version", "cuda_available", "execution_scope")}},
        "condition_summaries": condition_summaries(result_path, manifest),
        "accuracy_boundary": "A direct M1 control comparison of model input projection and stereo self-consistency only; no manual 2-D labels or 3-D ground truth.",
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
