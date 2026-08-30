"""Replay the B0 YOLO/geometry configuration on sparse far/mid/near clips.

The direct isolated YOLO process avoids this host's OpenMP conflict.  Because
it is not the archived B0 Docker image, this is an engineering diagnostic, not
an official baseline reproduction or an accuracy experiment.
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
CAPTURE_ROOT = PROJECT_ROOT / "research_records" / "raw_captures" / "R20260826-01_far_to_near_domain_capture"
CALIBRATION = APP_ROOT / "calibration" / "results" / "stereo_fisheye.json"
WEIGHTS = PROJECT_ROOT / "models" / "yolo26" / "yolo26x-pose.pt"
ISOLATED_RUNNER = Path(__file__).with_name("run_yolo_pose_isolated.py")
EVALUATOR = Path(__file__).with_name("evaluate_offline_stereo_predictions.py")
SEQUENCES = {
    "far_static": CAPTURE_ROOT / "far_static" / "20260826_195305_083",
    "mid_static": CAPTURE_ROOT / "mid_static" / "20260826_195344_142",
    "near_static": CAPTURE_ROOT / "near_static" / "20260826_195421_326",
}
DEFAULT_RUN_DIR = PROJECT_ROOT / "research_records" / "engineering_validation" / "V20260829-F6_static_distance_b0_config_diagnostic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--pairs-per-condition", type=int, default=12)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def select_pairs(folder: Path, count: int) -> list[dict]:
    rows = [row for row in read_csv_rows(folder / "stereo_pairs.csv") if float(row["abs_host_delta_ms"]) <= 25.0]
    if len(rows) < count:
        raise RuntimeError(f"{folder.name} contains only {len(rows)} pairs within 25 ms")
    # Exclude the first and last raw pair, then select deterministically across
    # the retained sequence instead of cherry-picking frames by visual result.
    positions = np.linspace(1, len(rows) - 2, count).round().astype(int)
    return [rows[int(position)] for position in positions]


def read_selected_frames(video_path: Path, required_indices: set[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    frames, index = {}, 0
    try:
        while required_indices:
            ok, frame = capture.read()
            if not ok:
                break
            if index in required_indices:
                frames[index] = frame.copy()
                required_indices.remove(index)
            index += 1
    finally:
        capture.release()
    if required_indices:
        raise RuntimeError(f"Missing frames in {video_path}: {sorted(required_indices)}")
    return frames


def instances_from_view(view: dict) -> list[dict]:
    output = []
    for person_id, (bbox, score, points, keypoint_scores) in enumerate(
        zip(view["boxes_xyxy"], view["box_scores"], view["keypoints_xy"], view["keypoint_scores"])
    ):
        points_array = np.asarray(points, dtype=np.float64)
        scores_array = np.asarray(keypoint_scores, dtype=np.float64)
        if points_array.shape[0] < 17 or scores_array.shape[0] < 17:
            continue
        output.append(
            {
                "person_id": person_id,
                "bbox_xyxy": [float(value) for value in bbox[:4]],
                "bbox_score": float(score),
                "keypoints": np.column_stack([points_array[:17, :2], scores_array[:17]]).astype(float).tolist(),
            }
        )
    return output


def condition_summary(results_path: Path) -> dict:
    records = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line]
    matched = 0
    ankle_valid, reasons, ankle_reprojection = 0, Counter(), []
    for record in records:
        people = record["persons_3d"]
        matched += bool(people)
        if not people:
            reasons["no_stereo_person"] += 1
            continue
        ankle = next((point for point in people[0]["keypoints_3d"] if point["index"] == 16), None)
        if ankle is None:
            reasons["no_right_ankle_record"] += 1
        elif ankle["valid"]:
            ankle_valid += 1
            ankle_reprojection.append(float(ankle["reprojection_error_mean_px"]))
        else:
            reasons[str(ankle["reason"])] += 1
            if ankle.get("reprojection_error_mean_px") is not None:
                ankle_reprojection.append(float(ankle["reprojection_error_mean_px"]))
    return {
        "processed_pairs": len(records),
        "matched_stereo_person_pairs": matched,
        "right_ankle_valid_3d_count": ankle_valid,
        "right_ankle_valid_3d_rate": ankle_valid / len(records) if records else None,
        "right_ankle_rejection_reasons": dict(reasons),
        "right_ankle_mean_reprojection_error_px_with_stereo_person": float(np.mean(ankle_reprojection)) if ankle_reprojection else None,
    }


def main() -> int:
    args = parse_args()
    if args.pairs_per_condition <= 0 or args.batch_size <= 0:
        raise ValueError("pairs-per-condition and batch-size must be positive")
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing output: {run_dir}")
    if args.resume and (run_dir / "run_metadata.json").exists():
        raise FileExistsError(f"Refusing to resume completed output: {run_dir}")
    if not (WEIGHTS.is_file() and ISOLATED_RUNNER.is_file() and EVALUATOR.is_file() and CALIBRATION.is_file()):
        raise FileNotFoundError("YOLO weights, isolated runner, evaluator, or calibration is missing")
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    input_dir = run_dir / "model_inputs"
    input_dir.mkdir(exist_ok=args.resume)
    selected_by_condition = {name: select_pairs(folder, args.pairs_per_condition) for name, folder in SEQUENCES.items()}
    input_order, raw_prediction_rows = [], {"left": [], "right": []}
    for condition, pairs in selected_by_condition.items():
        folder = SEQUENCES[condition]
        left_frames = read_selected_frames(folder / "left_capture.avi", {int(row["left_frame_id"]) for row in pairs})
        right_frames = read_selected_frames(folder / "right_capture.avi", {int(row["right_frame_id"]) for row in pairs})
        for row in pairs:
            pair_id = int(row["pair_id"])
            common_name = f"{condition}_sourcepair_{pair_id:03d}"
            for camera, rotation, frames, frame_key in (
                ("left", cv2.ROTATE_90_CLOCKWISE, left_frames, "left_frame_id"),
                ("right", cv2.ROTATE_90_COUNTERCLOCKWISE, right_frames, "right_frame_id"),
            ):
                image = cv2.rotate(frames[int(row[frame_key])], rotation)
                image_path = input_dir / f"{common_name}_{camera}.jpg"
                if not image_path.is_file() and not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"Could not write {image_path}")
                input_order.append({"condition": condition, "camera": camera, "name": common_name, "frame_id": int(row[frame_key]), "path": image_path})
    started = time.perf_counter()
    merged_views = []
    isolated_runtime = None
    for start_index in range(0, len(input_order), args.batch_size):
        chunk = input_order[start_index : start_index + args.batch_size]
        chunk_path = run_dir / f"isolated_yolo_prediction_chunk_{start_index // args.batch_size:03d}.json"
        if not chunk_path.is_file():
            subprocess.run(
                [
                    sys.executable, str(ISOLATED_RUNNER), "--weights", str(WEIGHTS), "--output", str(chunk_path),
                    "--device", str(args.device), "--imgsz", "1280", "--conf", "0.01", "--iou", "0.7", "--max-det", "300",
                    *[str(item["path"]) for item in chunk],
                ],
                check=True,
            )
        chunk_result = json.loads(chunk_path.read_text(encoding="utf-8"))
        if len(chunk_result.get("views", [])) != len(chunk):
            raise RuntimeError(f"YOLO chunk {chunk_path.name} result count does not match its inputs")
        merged_views.extend(chunk_result["views"])
        isolated_runtime = chunk_result
    isolated = {"views": merged_views}
    if len(isolated["views"]) != len(input_order):
        raise RuntimeError("Isolated YOLO result count does not match model inputs")
    isolated_output = run_dir / "isolated_yolo_prediction.json"
    isolated_output.write_text(json.dumps({"views": merged_views}, ensure_ascii=False, indent=2), encoding="utf-8")
    for source, prediction in zip(input_order, isolated["views"]):
        raw_prediction_rows[source["camera"]].append(
            {
                "file_name": source["name"],
                "frame_index": source["frame_id"],
                "instances": instances_from_view(prediction),
            }
        )
    # The replay utility pairs by file name.  Keep every source pair in the
    # same deterministic far/mid/near order for both camera JSON files.
    for camera in ("left", "right"):
        raw_prediction_rows[camera].sort(key=lambda value: value["file_name"])
        (run_dir / f"{camera}_predictions.json").write_text(
            json.dumps({"frames": raw_prediction_rows[camera]}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    geometry_dir = run_dir / "geometry_replay"
    subprocess.run(
        [
            sys.executable, str(EVALUATOR), "--model", "yolo26x_pose",
            "--left-json", str(run_dir / "left_predictions.json"), "--right-json", str(run_dir / "right_predictions.json"),
            "--calibration", str(CALIBRATION), "--output-dir", str(geometry_dir),
            "--left-model-rotation", "cw90", "--right-model-rotation", "ccw90",
            "--keypoint-threshold", "0.25", "--max-association-cost", "0.05", "--max-reprojection-error-px", "10.0",
        ],
        check=True,
    )
    result_records = [json.loads(line) for line in (geometry_dir / "offline_stereo_results.jsonl").read_text(encoding="utf-8").splitlines() if line]
    grouped_paths: dict[str, Path] = {}
    for condition in SEQUENCES:
        subset_path = run_dir / f"{condition}_geometry_results.jsonl"
        subset = [record for record in result_records if record["file_name"].startswith(f"{condition}_")]
        subset_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in subset), encoding="utf-8")
        grouped_paths[condition] = subset_path
    summaries = {condition: condition_summary(path) for condition, path in grouped_paths.items()}
    selection_rows = []
    for condition, rows in selected_by_condition.items():
        for row in rows:
            selection_rows.append({"condition": condition, **row})
    with (run_dir / "selected_stereo_pairs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["condition", "pair_id", "left_frame_id", "right_frame_id", "abs_host_delta_ms", "signed_host_delta_ms_right_minus_left"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selection_rows)
    metadata = {
        "classification": "engineering_validation",
        "validation_id": "V20260829-F6_static_distance_b0_config_diagnostic",
        "input_capture_id": "R20260826-01_far_to_near_domain_capture",
        "input_sequences": {name: str(folder) for name, folder in SEQUENCES.items()},
        "reference_run": "E20260827-B0_replay_baseline",
        "unique_variable": "static distance condition (far/mid/near); same YOLO weights, rotations, and offline geometry thresholds",
        "execution_difference": "YOLO ran in isolated local processes (fixed 12-image batches), not the archived B0 Docker image; do not call this an official B0 reproduction.",
        "weights_sha256": sha256(WEIGHTS),
        "settings": {"left_rotation": "cw90", "right_rotation": "ccw90", "imgsz": 1280, "candidate_conf": 0.01, "iou": 0.7, "max_det": 300, "keypoint_score_threshold": 0.2, "geometry_keypoint_threshold": 0.25, "max_association_cost": 0.05, "max_reprojection_error_px": 10.0},
        "pairs_per_condition": args.pairs_per_condition,
        "yolo_batch_size": args.batch_size,
        "isolated_runtime": {"torch_version": isolated_runtime["torch_version"], "cuda_available": isolated_runtime["cuda_available"]},
        "total_pairs": sum(value["processed_pairs"] for value in summaries.values()),
        "runtime_sec": time.perf_counter() - started,
        "success_criterion": "Each condition has traceable synchronized inputs, raw predictions, and geometry replay output.",
        "condition_summaries": summaries,
        "caveat": "No human 2-D labels or 3-D ground truth. Valid point counts and reprojection self-consistency are not accuracy measurements.",
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    run_dir.joinpath("EXPERIMENT.md").write_text(
        "# V20260829-F6 — 静态距离段的 B0 配置工程重放\n\n"
        "- 分类：`engineering_validation`。\n"
        "- 输入：far/mid/near 各固定 12 对、主机时间差不超过 25 ms 的静态双目帧。\n"
        "- 对照配置：B0 的 YOLO26x-pose 权重、左右旋转和离线几何阈值。\n"
        "- 执行差异：本机隔离 YOLO 进程代替 B0 Docker 镜像，因此不能称为正式 B0 复现。\n"
        "- 成功标准：每段均保存原始输入、原始预测、几何重放和右踝拒绝原因。\n\n"
        "## 结论边界\n\n"
        "没有人工 2D 标注或 3D 真值。任何有效点数量或重投影统计均只反映模型—标定的内部一致性，不能证明准确度。\n",
        encoding="utf-8",
    )
    command = "python .\\tools\\replay_static_distance_diagnostic.py"
    if args.resume:
        command += " --resume"
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
