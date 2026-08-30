"""Run DA3 on one calibrated local lower-leg/foot virtual stereo pair.

This is deliberately a one-pair engineering validation.  It does not compare
against 3-D ground truth, invoke a new pose detector, or claim that a DA3
depth image repairs the B0 right-ankle rejection.  Its sole purpose is to
check whether an F1-style, invertible local pinhole input yields finite DA3
depth/confidence samples at the selected ankle and adjacent ground region.
"""
from __future__ import annotations

import argparse
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
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "research_records"
    / "engineering_validation"
    / "V20260828-F2_DA3_local_foot_pair182"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--da3-root", type=Path, required=True, help="Depth-Anything-3 checkout")
    parser.add_argument("--model-dir", type=Path, required=True, help="Local DA3-LARGE-1.1 model directory")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
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
    # COCO-17 right knee/ankle are 14/16.  The third point only targets a
    # nearby foot/ground region; it is not an unobserved anatomical landmark.
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
    if np.any(upper - lower < 2.0):
        raise RuntimeError("Foot ROI collapsed after image clipping")
    return [float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1])]


def build_local_view(
    calibration: StereoCalibration,
    camera: str,
    image: np.ndarray,
    points: dict[str, np.ndarray],
) -> LocalPerspectiveView:
    k = calibration.left_K if camera == "left" else calibration.right_K
    d = calibration.left_D if camera == "left" else calibration.right_D
    builder = LocalPerspectiveModelInput(
        k,
        d,
        (image.shape[1], image.shape[0]),
        output_size=ROI_OUTPUT_SIZE,
        margin=1.20,
        min_horizontal_fov_deg=25.0,
        max_horizontal_fov_deg=120.0,
    )
    return builder.build(roi_bbox(points, (image.shape[1], image.shape[0])))


def virtual_intrinsics(view: LocalPerspectiveView) -> np.ndarray:
    return np.asarray(
        [[view.focal_px, 0.0, view.cx], [0.0, view.focal_px, view.cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def virtual_w2c(view: LocalPerspectiveView, calibration: StereoCalibration, camera: str) -> np.ndarray:
    """Map the left-camera world frame into this virtual pinhole camera.

    ``camera_from_virtual`` maps local virtual rays into their original
    fisheye-camera rays.  Its transpose is therefore the required original-
    camera-to-virtual rotation.  The right fisheye camera follows the existing
    B0 convention: X_right = R @ X_left + T.
    """
    camera_to_virtual = view.camera_from_virtual.T
    base_rotation = np.eye(3, dtype=np.float64) if camera == "left" else calibration.R
    base_translation = np.zeros(3, dtype=np.float64) if camera == "left" else calibration.T
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = (camera_to_virtual @ base_rotation).astype(np.float32)
    w2c[:3, 3] = (camera_to_virtual @ base_translation).astype(np.float32)
    return w2c


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def repo_commit(da3_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={da3_root.as_posix()}", "rev-parse", "HEAD"],
            cwd=da3_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sample_prediction(
    prediction: object,
    virtual_points: list[dict[str, np.ndarray]],
    source_size: tuple[int, int],
) -> dict[str, dict[str, dict[str, float | bool | list[int]]]]:
    depth = np.asarray(prediction.depth)
    confidence = np.asarray(prediction.conf)
    source_width, source_height = source_size
    samples: dict[str, dict[str, dict[str, float | bool | list[int]]]] = {}
    for index, (camera, points) in enumerate(zip(("left", "right"), virtual_points)):
        height, width = depth[index].shape
        camera_samples: dict[str, dict[str, float | bool | list[int]]] = {}
        for name, point in points.items():
            # upper_bound_resize is aspect-preserving for the fixed 4:3 local
            # inputs.  Sample the corresponding integer map coordinate.
            x = int(np.clip(np.rint(point[0] * (width - 1) / (source_width - 1)), 0, width - 1))
            y = int(np.clip(np.rint(point[1] * (height - 1) / (source_height - 1)), 0, height - 1))
            depth_value = float(depth[index, y, x])
            confidence_value = float(confidence[index, y, x])
            camera_samples[name] = {
                "prediction_px": [x, y],
                "depth": depth_value,
                "confidence": confidence_value,
                "finite_depth_and_confidence": bool(np.isfinite(depth_value) and np.isfinite(confidence_value)),
            }
        samples[camera] = camera_samples
    return samples


def write_experiment_md(run_dir: Path, metadata: dict) -> None:
    checks = metadata["sampled_output"]
    status = "通过" if metadata["all_samples_finite"] else "失败"
    run_dir.joinpath("EXPERIMENT.md").write_text(
        "# V20260828-F2 — DA3 局部脚部视图推理\n\n"
        "- 分类：`engineering_validation`，不是正式精度实验。\n"
        "- 输入：F1 的固定 pair 182 局部针孔左右视图；原始帧为左/右 242/241。\n"
        "- 参考：B0。该 pair 的右踝在高 2D 分数下仍因 11.547 px 平均重投影误差被拒。\n"
        "- 唯一变量：DA3 的输入从既有 D0/D1 全图换成可逆脚部局部针孔图，并向 DA3 提供对应局部内参与相对位姿。\n"
        "- 成功标准：DA3 前向完成，两个视图的右踝和地面支撑代理采样到有限深度与置信度。\n\n"
        f"## 结果\n\n{status}。逐点数值见 `run_metadata.json` 的 `sampled_output`。\n\n"
        "## 结论边界\n\n"
        "DA3 会为局部输入输出连续相对深度，但两视图的相似尺度对齐对仅两台相机是退化问题。"
        "本验证不产生米制深度、脚部真实误差、跨视图几何一致性或 B0 修复的结论。ROI 由 B0 记录的 2D 点选定，"
        "因此它也不能替代独立的脚部检测评估。\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    da3_root = args.da3_root.resolve()
    model_dir = args.model_dir.resolve()
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {run_dir}")
    if not (da3_root / "src" / "depth_anything_3" / "api.py").is_file():
        raise FileNotFoundError(f"Invalid DA3 checkout: {da3_root}")
    weights = model_dir / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Missing DA3 model weights: {weights}")
    sys.path.insert(0, str(da3_root / "src"))
    import torch
    from depth_anything_3.api import DepthAnything3

    run_dir.mkdir(parents=True)
    calibration = StereoCalibration.load(CALIBRATION)
    record = read_pair_record()
    raw_images = {
        "left": extract_frame(RAW_CAPTURE / "left_capture.avi", int(record["left_frame_id"])),
        "right": extract_frame(RAW_CAPTURE / "right_capture.avi", int(record["right_frame_id"])),
    }
    local_views: dict[str, LocalPerspectiveView] = {}
    local_points: dict[str, dict[str, np.ndarray]] = {}
    image_paths: list[str] = []
    for camera in ("left", "right"):
        person = record[camera]["persons"][0]
        raw_points = lower_leg_points(person)
        view = build_local_view(calibration, camera, raw_images[camera], raw_points)
        image = view.image(raw_images[camera])
        image_path = run_dir / f"{camera}_input.png"
        if not cv2.imwrite(str(image_path), image):
            raise RuntimeError(f"Could not save {image_path}")
        local_views[camera] = view
        local_points[camera] = {
            name: point for name, point in zip(raw_points, view.raw_to_virtual(np.asarray(list(raw_points.values()))))
        }
        image_paths.append(str(image_path))

    intrinsics = np.stack([virtual_intrinsics(local_views[camera]) for camera in ("left", "right")])
    extrinsics = np.stack([virtual_w2c(local_views[camera], calibration, camera) for camera in ("left", "right")])
    np.save(run_dir / "input_intrinsics.npy", intrinsics)
    np.save(run_dir / "input_extrinsics_w2c.npy", extrinsics)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    model = DepthAnything3.from_pretrained(str(model_dir)).to(device).eval()
    prediction = model.inference(
        image_paths,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        # The model normalizes poses before inference.  With only two cameras,
        # similarity-scale alignment is rank-degenerate; avoid converting an
        # arbitrary alignment factor into a fake physical depth scale.
        align_to_input_ext_scale=False,
        process_res=504,
        process_res_method="upper_bound_resize",
        export_dir=str(run_dir),
        export_format="mini_npz-depth_vis",
    )
    runtime_sec = time.perf_counter() - started
    samples = sample_prediction(
        prediction,
        [local_points["left"], local_points["right"]],
        ROI_OUTPUT_SIZE,
    )
    all_samples_finite = all(
        sample["finite_depth_and_confidence"]
        for camera in samples.values()
        for sample in camera.values()
    )
    metadata = {
        "classification": "engineering_validation",
        "validation_id": "V20260828-F2_DA3_local_foot_pair182",
        "input_capture_id": "R20260826-01_far_to_near_domain_capture",
        "input_pair_id": PAIR_ID,
        "input_frames": {"left": int(record["left_frame_id"]), "right": int(record["right_frame_id"])},
        "reference_run": "E20260827-B0_replay_baseline",
        "comparison_conditions": ["DA3_pair182_D0_raw_fisheye_large", "DA3_pair182_D1_upright_large"],
        "unique_variable": "F1 calibrated local lower-leg/foot virtual pinhole pair as DA3 input",
        "model": {
            "directory": str(model_dir),
            "weights_sha256": sha256(weights),
            "da3_repo_commit": repo_commit(da3_root),
            "device": str(device),
        },
        "local_camera_conditions": {
            "input_size_px": list(ROI_OUTPUT_SIZE),
            "intrinsics": intrinsics.tolist(),
            "extrinsics_w2c": extrinsics.tolist(),
            "focal_px": {camera: float(local_views[camera].focal_px) for camera in ("left", "right")},
        },
        "success_criterion": "DA3 completes and all ankle/ground-support samples have finite depth and confidence.",
        "prediction": {
            "depth_shape": list(np.asarray(prediction.depth).shape),
            "confidence_shape": list(np.asarray(prediction.conf).shape),
            "runtime_sec": runtime_sec,
        },
        "virtual_target_points_px": {
            camera: {name: point.tolist() for name, point in points.items()}
            for camera, points in local_points.items()
        },
        "sampled_output": samples,
        "all_samples_finite": all_samples_finite,
        "caveat": (
            "DA3 local-input observability validation only. Two-view similarity-scale alignment is rank-degenerate; "
            "relative depth values are not metric and no cross-view accuracy, 3-D ground truth, or B0 repair is claimed."
        ),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_experiment_md(run_dir, metadata)
    (run_dir / "command.txt").write_text(
        "python .\\tools\\run_da3_foot_roi_pair182.py --da3-root <Depth-Anything-3> --model-dir <DA3-LARGE-1.1>\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0 if all_samples_finite else 2


if __name__ == "__main__":
    raise SystemExit(main())
