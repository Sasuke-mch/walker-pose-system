"""Run the L1-style per-person local camera condition on the fixed 60 pairs.

The source image remains the calibrated raw fisheye image.  A baseline YOLO
box defines one invertible, person-centred local *virtual pinhole* view per
image; local YOLO output is mapped back to raw fisheye pixels before scoring.
This is a single-variable test of local camera input, not a foot-ROI model.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(APP_ROOT))

from pose_app.calibration import StereoCalibration
from pose_app.local_perspective import LocalPerspectiveModelInput
from pose_app.rotation import model_image_size, model_to_raw_point, rotate_image_for_model
from tools.evaluate_offline_stereo_predictions import _records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-selection", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--left-baseline", required=True, type=Path)
    parser.add_argument("--right-baseline", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--left-rotation", default="cw90")
    parser.add_argument("--right-rotation", default="ccw90")
    parser.add_argument("--margin", type=float, default=1.35)
    parser.add_argument("--weights", default=PROJECT_ROOT / "models" / "yolo26" / "yolo26x-pose.pt", type=Path)
    parser.add_argument("--isolated-runner", default=Path(__file__).with_name("run_yolo_pose_isolated.py"), type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=12)
    return parser.parse_args()


def raw_bbox_and_support(person, rotation: str) -> tuple[list[float], np.ndarray]:
    x1, y1, x2, y2 = [float(value) for value in person.bbox[:4]]
    corners = np.asarray([[x1, y1], [x1, y2], [x2, y1], [x2, y2]], dtype=np.float64)
    raw_corners = np.asarray([model_to_raw_point(x, y, 1920, 1080, rotation) for x, y in corners])
    raw_bbox = [float(raw_corners[:, 0].min()), float(raw_corners[:, 1].min()), float(raw_corners[:, 0].max()), float(raw_corners[:, 1].max())]
    support = []
    for x, y, score in person.keypoints:
        if float(score) >= 0.25:
            support.append(model_to_raw_point(float(x), float(y), 1920, 1080, rotation))
    return raw_bbox, np.asarray(support, dtype=np.float64)


def central_index(view: dict, rotated_size: tuple[int, int]) -> int | None:
    if not view["boxes_xyxy"]:
        return None
    center = np.asarray([(rotated_size[0] - 1.0) * 0.5, (rotated_size[1] - 1.0) * 0.5])
    diagonal = float(np.hypot(*rotated_size))
    choices = []
    for index, (box, score) in enumerate(zip(view["boxes_xyxy"], view["box_scores"])):
        x1, y1, x2, y2 = [float(value) for value in box[:4]]
        contained = x1 <= center[0] <= x2 and y1 <= center[1] <= y2
        distance = float(np.linalg.norm(np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5]) - center) / diagonal)
        choices.append((0 if contained else 1, distance - 0.02 * float(score), index))
    return min(choices)[2]


def map_local_result(view: dict, local_view, rotation: str) -> dict:
    rotated_size = model_image_size(*local_view.output_size, rotation)
    index = central_index(view, rotated_size)
    if index is None:
        return {"instances": [], "status": "no_person_in_local_view"}
    local_points = np.asarray(view["keypoints_xy"][index][:17], dtype=np.float64)
    local_points = np.asarray([model_to_raw_point(x, y, *local_view.output_size, rotation) for x, y in local_points])
    raw_points = local_view.virtual_to_raw(local_points)
    local_box = np.asarray(view["boxes_xyxy"][index][:4], dtype=np.float64).reshape(2, 2)
    rotated_corners = np.asarray([[local_box[0, 0], local_box[0, 1]], [local_box[0, 0], local_box[1, 1]], [local_box[1, 0], local_box[0, 1]], [local_box[1, 0], local_box[1, 1]]])
    virtual_corners = np.asarray([model_to_raw_point(x, y, *local_view.output_size, rotation) for x, y in rotated_corners])
    raw_corners = local_view.virtual_to_raw(virtual_corners)
    scores = [float(value) for value in view["keypoint_scores"][index][:17]]
    return {
        "instances": [{
            "person_id": 0,
            "bbox_xyxy": [float(raw_corners[:, 0].min()), float(raw_corners[:, 1].min()), float(raw_corners[:, 0].max()), float(raw_corners[:, 1].max())],
            "bbox_score": float(view["box_scores"][index]),
            "keypoints": [[float(x), float(y), score] for (x, y), score in zip(raw_points, scores)],
        }],
        "status": "central_local_person_selected",
    }


def run_side(args, calibration, side: str, baseline_path: Path, rotation: str, output: Path) -> tuple[dict, list[dict]]:
    records = _records(baseline_path.resolve(), "yolo26x_pose", "raw_keypoint")
    by_name = {record["name"]: record for record in records}
    image_dir = args.input_selection.resolve() / "raw_fisheye" / side
    images = sorted(image_dir.glob("*.png"))
    if len(images) != 60 or set(path.name for path in images) != set(by_name):
        raise RuntimeError(f"{side}: fixed selection and baseline predictions do not match")
    k, d = (calibration.left_K, calibration.left_D) if side == "left" else (calibration.right_K, calibration.right_D)
    builder = LocalPerspectiveModelInput(k, d, (1920, 1080), margin=args.margin)
    model_dir = output / "local_model_input" / side
    model_dir.mkdir(parents=True)
    prepared, audit, local_views = [], {}, {}
    for image_path in images:
        record = by_name[image_path.name]
        if not record["persons"]:
            audit[image_path.name] = {"status": "no_baseline_person"}
            continue
        target = max(record["persons"], key=lambda person: person.bbox_score)
        raw_bbox, support = raw_bbox_and_support(target, rotation)
        raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if raw is None:
            raise RuntimeError(f"Cannot read {image_path}")
        try:
            local_view = builder.build(raw_bbox, support_points=support)
            model_image = rotate_image_for_model(local_view.image(raw), rotation)
        except Exception as exc:
            audit[image_path.name] = {"status": "local_view_build_failed", "reason": str(exc)}
            continue
        destination = model_dir / image_path.name
        if not cv2.imwrite(str(destination), model_image):
            raise RuntimeError(f"Could not write {destination}")
        local_views[image_path.name] = local_view
        prepared.append(destination)
        audit[image_path.name] = {"status": "local_view_ready", "baseline_bbox_score": float(target.bbox_score), "raw_bbox_xyxy": raw_bbox, "support_point_count": int(len(support)), "focal_px": float(local_view.focal_px)}
    predictions: dict[str, dict] = {}
    for start in range(0, len(prepared), args.batch_size):
        batch = prepared[start:start + args.batch_size]
        chunk = output / f"{side}_local_chunk_{start // args.batch_size:03d}.json"
        command = [sys.executable, str(args.isolated_runner.resolve()), "--weights", str(args.weights.resolve()), "--output", str(chunk), "--device", str(args.device), "--imgsz", "1280", "--conf", "0.05", "--iou", "0.7", "--max-det", "300", *[str(path) for path in batch]]
        subprocess.run(command, check=True)
        payload = json.loads(chunk.read_text(encoding="utf-8"))
        if len(payload.get("views", [])) != len(batch):
            raise RuntimeError(f"{chunk} has an unexpected image count")
        for path, result in zip(batch, payload["views"]):
            predictions[path.name] = map_local_result(result, local_views[path.name], rotation)
    frames = []
    for index, image_path in enumerate(images):
        item = predictions.get(image_path.name, {"instances": [], "status": audit[image_path.name]["status"]})
        audit[image_path.name]["prediction_status"] = item["status"]
        frames.append({"file_name": image_path.name, "frame_index": index, "instances": item["instances"]})
    return {"frames": frames}, [{"file_name": name, **value} for name, value in sorted(audit.items())]


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if args.margin <= 1.0 or args.batch_size <= 0:
        raise ValueError("margin must exceed 1 and batch size must be positive")
    if not args.weights.is_file() or not args.isolated_runner.is_file():
        raise FileNotFoundError("YOLO weights or isolated runner is missing")
    calibration = StereoCalibration.load(args.calibration.resolve()).for_runtime_sizes((1920, 1080), (1920, 1080))
    output.mkdir(parents=True)
    left, left_audit = run_side(args, calibration, "left", args.left_baseline, args.left_rotation, output)
    right, right_audit = run_side(args, calibration, "right", args.right_baseline, args.right_rotation, output)
    for side, payload in (("left", left), ("right", right)):
        (output / f"{side}_predictions_raw_fisheye.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "local_view_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = sorted({key for row in left_audit + right_audit for key in row})
        writer = csv.DictWriter(handle, fieldnames=["side", *fields])
        writer.writeheader()
        for side, rows in (("left", left_audit), ("right", right_audit)):
            writer.writerows([{"side": side, **row} for row in rows])
    metadata = {
        "experiment_id": "E20260829-L2_yolo26x_local_perspective_raw_fisheye_60pair",
        "input_selection": str(args.input_selection.resolve()),
        "camera_input": "calibrated raw fisheye; local virtual pinhole is model input only",
        "baseline_box_source": "saved M1 YOLO26x predictions; highest bbox-score person per image",
        "model_input_rotation": {"left": args.left_rotation, "right": args.right_rotation},
        "output_coordinate_space": "original 1920x1080 raw fisheye pixels after inverse local-view and rotation mapping",
        "margin": args.margin,
        "interpretation_boundary": "This assesses a local virtual-camera input condition for YOLO, not a foot-specialist detector and not a raw-fisheye crop test. Baseline boxes seed the local view, so detector independence is not evaluated.",
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "left": len(left["frames"]), "right": len(right["frames"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
