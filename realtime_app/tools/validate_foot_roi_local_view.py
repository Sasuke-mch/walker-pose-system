"""Validate calibrated lower-leg/foot local virtual views on one B0 pair.

This is an engineering validation, not a new pose/3-D accuracy experiment.
It verifies that an invertible fisheye-to-pinhole view can retain the selected
lower-leg, ankle, and immediately-ahead ground support in *both* cameras.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(APP_ROOT))

from pose_app.calibration import StereoCalibration  # noqa: E402
from pose_app.local_perspective import LocalPerspectiveModelInput  # noqa: E402


PAIR_ID = 182
SIDE = "right"
ROI_OUTPUT_SIZE = (960, 720)
BOUNDARY_PX = 24.0
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
RUN_DIR = (
    PROJECT_ROOT
    / "research_records"
    / "engineering_validation"
    / "V20260828-F1_foot_roi_local_view_coverage"
)


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


def select_primary_person(record: dict, camera: str) -> dict:
    people = record[camera]["persons"]
    if not people:
        raise RuntimeError(f"B0 pair {PAIR_ID} has no {camera} person")
    return people[0]


def named_lower_leg_points(person: dict) -> dict[str, np.ndarray]:
    # COCO-17 right knee/ankle are indices 14/16.  The third point extends the
    # knee-to-ankle ray, representing the immediately-ahead foot/ground support
    # region rather than an unobserved anatomical point.
    keypoints = person["keypoints"]
    knee = np.asarray(keypoints[14][:2], dtype=np.float64)
    ankle = np.asarray(keypoints[16][:2], dtype=np.float64)
    support = ankle + 0.5 * (ankle - knee)
    return {"right_knee": knee, "right_ankle": ankle, "ground_support_proxy": support}


def roi_bbox(points: dict[str, np.ndarray], image_size: tuple[int, int]) -> list[float]:
    width, height = image_size
    values = np.asarray(list(points.values()), dtype=np.float64)
    # A fixed 90 px raw-fisheye guard keeps local texture around the lower leg;
    # the perspective builder supplies the documented 1.20 optical margin.
    lower = values.min(axis=0) - 90.0
    upper = values.max(axis=0) + 90.0
    lower = np.maximum(lower, [0.0, 0.0])
    upper = np.minimum(upper, [width - 1.0, height - 1.0])
    if np.any(upper - lower < 2.0):
        raise RuntimeError("Foot ROI collapsed after image clipping")
    return [float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1])]


def draw_raw(image: np.ndarray, bbox: list[float], points: dict[str, np.ndarray]) -> np.ndarray:
    view = image.copy()
    cv2.rectangle(view, tuple(np.rint(bbox[:2]).astype(int)), tuple(np.rint(bbox[2:]).astype(int)), (0, 230, 255), 3)
    for name, point in points.items():
        color = (0, 255, 0) if name == "right_ankle" else (255, 0, 255)
        xy = tuple(np.rint(point).astype(int))
        cv2.circle(view, xy, 8, color, -1)
        cv2.putText(view, name, (xy[0] + 10, xy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return view


def draw_virtual(image: np.ndarray, virtual_points: dict[str, np.ndarray]) -> np.ndarray:
    view = image.copy()
    for name, point in virtual_points.items():
        color = (0, 255, 0) if name == "right_ankle" else (255, 0, 255)
        xy = tuple(np.rint(point).astype(int))
        cv2.circle(view, xy, 8, color, -1)
        cv2.putText(view, name, (xy[0] + 10, xy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return view


def validate_view(
    calibration: StereoCalibration,
    camera: str,
    image: np.ndarray,
    points: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict]:
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
    bbox = roi_bbox(points, (image.shape[1], image.shape[0]))
    local = builder.build(bbox)
    virtual_points = {
        name: value for name, value in zip(points, local.raw_to_virtual(np.asarray(list(points.values()))))
    }
    width, height = local.output_size
    coverage = {
        name: bool(
            BOUNDARY_PX <= point[0] <= width - 1 - BOUNDARY_PX
            and BOUNDARY_PX <= point[1] <= height - 1 - BOUNDARY_PX
        )
        for name, point in virtual_points.items()
    }
    payload = {
        "camera": camera,
        "raw_roi_bbox_xyxy": bbox,
        "virtual_output_size": list(local.output_size),
        "virtual_focal_px": local.focal_px,
        "raw_points_px": {name: value.tolist() for name, value in points.items()},
        "virtual_points_px": {name: value.tolist() for name, value in virtual_points.items()},
        "coverage_boundary_px": BOUNDARY_PX,
        "coverage": coverage,
        "all_required_points_covered": all(coverage.values()),
    }
    return draw_raw(image, bbox, points), draw_virtual(local.image(image), virtual_points), payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing validation directory: {run_dir}")
    run_dir.mkdir(parents=True)
    record = read_pair_record()
    calibration = StereoCalibration.load(CALIBRATION)
    left = extract_frame(RAW_CAPTURE / "left_capture.avi", int(record["left_frame_id"]))
    right = extract_frame(RAW_CAPTURE / "right_capture.avi", int(record["right_frame_id"]))
    outputs = {}
    tiles = []
    for camera, image in (("left", left), ("right", right)):
        points = named_lower_leg_points(select_primary_person(record, camera))
        raw_overlay, roi_overlay, payload = validate_view(calibration, camera, image, points)
        cv2.imwrite(str(run_dir / f"{camera}_raw_overlay.png"), raw_overlay)
        cv2.imwrite(str(run_dir / f"{camera}_foot_roi.png"), roi_overlay)
        # Make a same-height visual audit pair without altering source outputs.
        raw_scaled = cv2.resize(raw_overlay, (ROI_OUTPUT_SIZE[0], ROI_OUTPUT_SIZE[1]))
        tiles.append(np.hstack([raw_scaled, roi_overlay]))
        outputs[camera] = payload
    preview = np.vstack(tiles)
    cv2.imwrite(str(run_dir / "foot_roi_coverage_preview.png"), preview)
    manifest = {
        "classification": "engineering_validation",
        "validation_id": "V20260828-F1_foot_roi_local_view_coverage",
        "input_capture_id": "R20260826-01_far_to_near_domain_capture",
        "input_pair_id": PAIR_ID,
        "reference_run": "E20260827-B0_replay_baseline",
        "unique_variable": "calibrated lower-leg/foot local virtual pinhole view",
        "success_criterion": "right knee, right ankle, and ground-support proxy are all inside both virtual views with a 24 px boundary",
        "result": {camera: payload["all_required_points_covered"] for camera, payload in outputs.items()},
        "all_views_pass": all(payload["all_required_points_covered"] for payload in outputs.values()),
        "caveat": "Coverage validation only. No 2-D re-inference, stereo association, triangulation, or 3-D accuracy comparison was run.",
        "views": outputs,
    }
    (run_dir / "coverage_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "command.txt").write_text("python .\\tools\\validate_foot_roi_local_view.py\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["all_views_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
