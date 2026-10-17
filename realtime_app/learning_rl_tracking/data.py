"""Read the existing 448 pixel/time/conditional-scene contract once."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from pose_app.fisheye_camera import load_stereo_fisheye, fisheye_project_numpy
from pose_app.body_observations import load_ordered_predictions as _load_predictions
from pose_app.replay_pair_timing import load_capture_pair_times as _read_capture_times

JOINTS = np.arange(5, 17)
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "research_records/engineering_validation"


@dataclass
class TrackingData:
    pixels: np.ndarray
    valid: np.ndarray
    time: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    stage: np.ndarray
    scene_status: np.ndarray
    calibration: object
    capture_ids: np.ndarray
    skew_ms: np.ndarray
    initial_fit: Path

    def project(self, points_world: np.ndarray, index: int) -> np.ndarray:
        camera = (points_world - self.translation[index]) @ self.rotation[index]
        cal = self.calibration
        right = camera @ cal.R_cam0_to_cam1.T + cal.T_cam0_to_cam1_mm / 1000
        with np.errstate(divide="ignore", invalid="ignore"):
            pixels = np.stack(
                [
                    fisheye_project_numpy(camera, cal.K0, cal.D0),
                    fisheye_project_numpy(right, cal.K1, cal.D1),
                ]
            )
        domain = (camera[:, 2] > 0) & (right[:, 2] > 0)
        pixels[:, ~domain] = np.nan
        return pixels


def load_real(root: Path = ROOT) -> TrackingData:
    base = root / "research_records/engineering_validation"
    people = base / "V20260908_people0_1_2_pmpose_c3_chain/people_1"
    predictions = np.stack(
        [
            _load_predictions(
                people / f"c3_predictions/{side}/pmpose/raw_predictions.json", side
            )
            for side in ("left", "right")
        ],
        axis=1,
    )[:, :, JOINTS]
    pixels, score = predictions[..., :2], predictions[..., 2]
    valid = (
        np.isfinite(predictions).all(axis=-1)
        & (score >= 0.2)
        & (score <= 1)
        & (pixels[..., 0] >= 0)
        & (pixels[..., 0] < 1920)
        & (pixels[..., 1] >= 0)
        & (pixels[..., 1] < 1080)
    )
    time, skew, capture_ids, _ = _read_capture_times(
        people / "input_448pairs", len(pixels)
    )
    scene_dir = base / "G20261013_reliability448_view_v1/view/scene"
    with np.load(scene_dir / "scene_transforms.npz", allow_pickle=True) as scene:
        rotation = scene["rotation_ground_from_left"].astype(float)
        translation = scene["translation_ground_from_left_mm"].astype(float) / 1000
        stage = scene["stage"].astype(str)
    rows = [
        json.loads(line)
        for line in (scene_dir / "dynamic_ground_pose.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    status = np.asarray([row["pose_audit"]["status"] for row in rows])
    if not (len(rows) == len(rotation) == len(time) == 448):
        raise ValueError("expected complete same-source 448 frames")
    if any(row["pair_id"] != i for i, row in enumerate(rows)):
        raise ValueError("scene IDs must match observation IDs")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError(
            "conditional scene has unavailable transforms; do not fill them"
        )
    return TrackingData(
        pixels,
        valid,
        time,
        rotation,
        translation,
        stage,
        status,
        load_stereo_fisheye(root / "realtime_app/calibration/results"),
        capture_ids,
        skew,
        base / "G20261014_stereo_pva_baseline448_v2/first_frame_fit/result.npz",
    )
