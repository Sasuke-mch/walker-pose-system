"""Check whether the existing full-body YOLO pose model is valid on foot ROIs.

This intentionally evaluates model scope, not accuracy.  A detector trained
for complete people may hallucinate a whole-body skeleton when only legs and
feet are present.  Such output must not be used as a foot-specialist result.
"""
from __future__ import annotations

import argparse
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
from pose_app.local_perspective import LocalPerspectiveModelInput, LocalPerspectiveView  # noqa: E402


PAIR_ID = 182
ROI_OUTPUT_SIZE = (960, 720)
RAW_CAPTURE = (
    PROJECT_ROOT
    / "research_records"
    / "raw_captures"
    / "R20260826-01_far_to_near_domain_capture"
    / "approach_far_to_near"
    / "20260826_195459_568"
)
B0_JSONL = (
    PROJECT_ROOT
    / "research_records"
    / "official_runs"
    / "E20260827-B0_replay_baseline"
    / "stereo_results.jsonl"
)
CALIBRATION = APP_ROOT / "calibration" / "results" / "stereo_fisheye.json"
WEIGHTS = PROJECT_ROOT / "models" / "yolo26" / "yolo26x-pose.pt"
F2_INPUT_DIR = (
    PROJECT_ROOT
    / "research_records"
    / "engineering_validation"
    / "V20260828-F2_DA3_local_foot_pair182"
)
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "research_records"
    / "engineering_validation"
    / "V20260828-F3_yolo_fullbody_on_foot_roi_pair182"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--isolated-runner",
        type=Path,
        default=Path(__file__).with_name("run_yolo_pose_isolated.py"),
    )
    return parser.parse_args()


def read_pair_record() -> dict:
    with B0_JSONL.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("pair_id") == PAIR_ID:
                return record
    raise RuntimeError(f"B0 pair_id={PAIR_ID} is unavailable")


def extract_frame(video_path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, image = capture.read()
    finally:
        capture.release()
    if not ok or image is None:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return image


def lower_leg_points(person: dict) -> dict[str, np.ndarray]:
    keypoints = person["keypoints"]
    knee = np.asarray(keypoints[14][:2], dtype=np.float64)
    ankle = np.asarray(keypoints[16][:2], dtype=np.float64)
    support = ankle + 0.5 * (ankle - knee)
    return {"right_knee": knee, "right_ankle": ankle, "ground_support_proxy": support}


def roi_bbox(points: dict[str, np.ndarray], image_size: tuple[int, int]) -> list[float]:
    width, height = image_size
    values = np.asarray(list(points.values()), dtype=np.float64)
    lower = np.maximum(values.min(axis=0) - 90.0, [0.0, 0.0])
    upper = np.minimum(values.max(axis=0) + 90.0, [width - 1.0, height - 1.0])
    return [float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1])]


def make_view(calibration: StereoCalibration, camera: str, image: np.ndarray, points: dict[str, np.ndarray]) -> LocalPerspectiveView:
    k = calibration.left_K if camera == "left" else calibration.right_K
    d = calibration.left_D if camera == "left" else calibration.right_D
    return LocalPerspectiveModelInput(
        k,
        d,
        (image.shape[1], image.shape[0]),
        output_size=ROI_OUTPUT_SIZE,
        margin=1.20,
        min_horizontal_fov_deg=25.0,
        max_horizontal_fov_deg=120.0,
    ).build(roi_bbox(points, (image.shape[1], image.shape[0])))


def draw_raw_audit(
    raw: np.ndarray,
    reference: dict[str, np.ndarray],
    candidate_raw: np.ndarray | None,
    candidate_scores: np.ndarray | None,
) -> np.ndarray:
    image = raw.copy()
    for name, point in reference.items():
        colour = (255, 0, 255) if name != "right_ankle" else (0, 255, 0)
        xy = tuple(np.rint(point).astype(int))
        cv2.circle(image, xy, 9, colour, -1)
        cv2.putText(image, f"B0_{name}", (xy[0] + 8, xy[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
    if candidate_raw is not None and candidate_scores is not None:
        for index, label in ((14, "roi_right_knee"), (16, "roi_right_ankle")):
            xy = tuple(np.rint(candidate_raw[index]).astype(int))
            score = float(candidate_scores[index])
            cv2.circle(image, xy, 9, (0, 220, 255), 2)
            cv2.putText(image, f"{label} {score:.2f}", (xy[0] + 8, xy[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
    return image


def result_for_camera(
    result: dict,
    local_view: LocalPerspectiveView,
    raw: np.ndarray,
    reference: dict[str, np.ndarray],
) -> tuple[dict, np.ndarray]:
    empty = {
        "person_count": 0,
        "selected_person": None,
        "result": "no_person_detected",
        "caveat": "No conclusion about foot quality; full-body model has no valid person output on this crop.",
    }
    if not result["boxes_xyxy"]:
        return empty, draw_raw_audit(raw, reference, None, None)
    boxes = np.asarray(result["boxes_xyxy"], dtype=np.float64)
    box_scores = np.asarray(result["box_scores"], dtype=np.float64)
    points = np.asarray(result["keypoints_xy"], dtype=np.float64)
    scores = np.asarray(result["keypoint_scores"], dtype=np.float64)
    selected = int(np.argmax(box_scores))
    local_keypoints = np.asarray(points[selected][:17], dtype=np.float64)
    raw_keypoints = local_view.virtual_to_raw(local_keypoints)
    keypoint_scores = np.asarray(scores[selected][:17], dtype=np.float64)
    b0_knee, b0_ankle = reference["right_knee"], reference["right_ankle"]
    disagreement = {
        "right_knee_px": float(np.linalg.norm(raw_keypoints[14] - b0_knee)),
        "right_ankle_px": float(np.linalg.norm(raw_keypoints[16] - b0_ankle)),
    }
    selected_payload = {
        "bbox_xyxy_local": boxes[selected].astype(float).tolist(),
        "bbox_score": float(box_scores[selected]),
        "keypoints_local_xy_score": np.column_stack([local_keypoints, keypoint_scores]).astype(float).tolist(),
        "right_leg_raw_xy_score": {
            "right_knee": [float(raw_keypoints[14, 0]), float(raw_keypoints[14, 1]), float(keypoint_scores[14])],
            "right_ankle": [float(raw_keypoints[16, 0]), float(raw_keypoints[16, 1]), float(keypoint_scores[16])],
        },
        "disagreement_vs_B0_raw_px": disagreement,
    }
    automatic_scope_flag = bool(keypoint_scores[14] >= 0.2 and keypoint_scores[16] >= 0.2)
    payload = {
        "person_count": int(len(boxes)),
        "selected_person": selected_payload,
        "result": "candidate_requires_visual_anatomical_review" if automatic_scope_flag else "leg_keypoints_below_existing_score_threshold",
        "automatic_scope_flag": automatic_scope_flag,
        "caveat": (
            "Distances to B0 are disagreements between two model outputs, not 2-D error. "
            "A full-body model on a foot crop remains invalid until visual anatomical review and independent foot labels approve it."
        ),
    }
    return payload, draw_raw_audit(raw, reference, raw_keypoints, keypoint_scores)


def draw_local_audit(
    input_path: Path,
    result: dict,
) -> np.ndarray:
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read local input: {input_path}")
    if not result["boxes_xyxy"]:
        return image
    boxes = np.asarray(result["boxes_xyxy"], dtype=np.float64)
    scores = np.asarray(result["box_scores"], dtype=np.float64)
    points = np.asarray(result["keypoints_xy"], dtype=np.float64)
    selected = int(np.argmax(scores))
    x1, y1, x2, y2 = np.rint(boxes[selected]).astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 255), 2)
    cv2.putText(image, f"YOLO full-body {scores[selected]:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
    for index, point in enumerate(points[selected][:17]):
        xy = tuple(np.rint(point).astype(int))
        cv2.circle(image, xy, 3, (0, 180, 255), -1)
        if index in {14, 16}:
            cv2.putText(image, f"{index}", (xy[0] + 5, xy[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 2)
    return image


def write_experiment_md(run_dir: Path, metadata: dict) -> None:
    results = metadata["views"]
    result_text = ", ".join(f"{camera}: {value['result']}" for camera, value in results.items())
    run_dir.joinpath("EXPERIMENT.md").write_text(
        "# V20260828-F3 — 全身 YOLO26x-pose 在脚部局部图上的适用性核查\n\n"
        "- 分类：`engineering_validation`。\n"
        "- 输入：F1/F2 固定的 pair 182 左右脚部局部图。\n"
        "- 参考：B0 全图 `yolo26x-pose` 的右膝/踝原始像点。\n"
        "- 唯一变量：仅把全身 YOLO26x-pose 的输入改成脚部局部图；不改变权重、阈值、左右关联或 3D 三角化。\n"
        "- 成功标准：两个视图都得到同一人的、可经人工复核为解剖合理的膝/踝点。\n\n"
        f"## 自动结果\n\n{result_text}\n\n"
        "## 结论边界\n\n"
        "该模型的训练目标是全身姿态；即使产生 COCO-17 输出，也可能是裁剪条件下的幻觉。"
        "与 B0 的像素距离只是两个模型输出之间的不一致，不是人工 2D 误差。"
        "没有独立脚部标注和人工解剖复核时，不得把任何输出称为脚部细化成功。\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {run_dir}")
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Missing YOLO weights: {WEIGHTS}")
    for name in ("left_input.png", "right_input.png"):
        if not (F2_INPUT_DIR / name).is_file():
            raise FileNotFoundError(f"F2 input is unavailable: {F2_INPUT_DIR / name}")
    run_dir.mkdir(parents=True)
    record = read_pair_record()
    calibration = StereoCalibration.load(CALIBRATION)
    raw_images = {
        "left": extract_frame(RAW_CAPTURE / "left_capture.avi", int(record["left_frame_id"])),
        "right": extract_frame(RAW_CAPTURE / "right_capture.avi", int(record["right_frame_id"])),
    }
    local_views, references = {}, {}
    for camera in ("left", "right"):
        reference = lower_leg_points(record[camera]["persons"][0])
        references[camera] = reference
        local_views[camera] = make_view(calibration, camera, raw_images[camera], reference)
    runner = args.isolated_runner.resolve()
    if not runner.is_file():
        raise FileNotFoundError(f"Missing isolated YOLO runner: {runner}")
    started = time.perf_counter()
    isolated_json = run_dir / "isolated_yolo_prediction.json"
    subprocess.run(
        [
            sys.executable,
            str(runner),
            "--weights",
            str(WEIGHTS),
            "--output",
            str(isolated_json),
            "--device",
            str(args.device),
            "--imgsz",
            "1280",
            "--conf",
            "0.01",
            "--iou",
            "0.7",
            "--max-det",
            "300",
            str(F2_INPUT_DIR / "left_input.png"),
            str(F2_INPUT_DIR / "right_input.png"),
        ],
        check=True,
    )
    elapsed_sec = time.perf_counter() - started
    model_results = json.loads(isolated_json.read_text(encoding="utf-8"))
    if len(model_results.get("views", [])) != 2:
        raise RuntimeError("Isolated YOLO runner did not return one record per input view")
    views = {}
    for camera, result in zip(("left", "right"), model_results["views"]):
        payload, raw_overlay = result_for_camera(result, local_views[camera], raw_images[camera], references[camera])
        cv2.imwrite(str(run_dir / f"{camera}_raw_overlay.png"), raw_overlay)
        local_annotated = draw_local_audit(F2_INPUT_DIR / f"{camera}_input.png", result)
        cv2.imwrite(str(run_dir / f"{camera}_local_yolo_output.png"), local_annotated)
        views[camera] = payload
    metadata = {
        "classification": "engineering_validation",
        "validation_id": "V20260828-F3_yolo_fullbody_on_foot_roi_pair182",
        "input_capture_id": "R20260826-01_far_to_near_domain_capture",
        "input_pair_id": PAIR_ID,
        "reference_run": "E20260827-B0_replay_baseline",
        "unique_variable": "existing yolo26x-pose full-body model receives F1 local foot ROI instead of full image",
        "weights": str(WEIGHTS),
        "inference_settings": {"device": str(args.device), "imgsz": 1280, "conf": 0.01, "iou": 0.7, "max_det": 300, "keypoint_score_threshold": 0.2},
        "execution_scope": model_results["execution_scope"],
        "isolated_runtime": {"torch_version": model_results["torch_version"], "cuda_available": model_results["cuda_available"]},
        "runtime_sec": elapsed_sec,
        "success_criterion": "both images yield a same-person knee/ankle candidate that passes independent visual anatomical review",
        "views": views,
        "caveat": "This is a scope negative-control, not a foot-specialist accuracy experiment and not a 3-D reconstruction run.",
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_experiment_md(run_dir, metadata)
    (run_dir / "command.txt").write_text("python .\\tools\\validate_yolo_fullbody_on_foot_roi.py\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
