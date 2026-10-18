#!/usr/bin/env python3
"""Run and audit a fixed window of independent male-SMPL COCO fits.

Each frame is fit from clean zero state by fit_smpl_coco_single_frame.py. This
driver only orchestrates independent runs and aggregates diagnostics; it does
not pass parameters between frames or add temporal, 3-D, shape, or contact
losses.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


COCO17_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)
RIGHT_KNEE_INDEX = 14


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def rotation_jump_magnitudes(rotvecs: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    rotations = Rotation.from_rotvec(rotvecs)
    relative = rotations[:-1].inv() * rotations[1:]
    return relative.magnitude()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stereo-jsonl", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--regressor", type=Path, required=True)
    parser.add_argument("--left-image-dir", type=Path, required=True)
    parser.add_argument("--right-image-dir", type=Path, required=True)
    parser.add_argument("--start-pair", type=int, required=True)
    parser.add_argument("--end-pair", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--minimum-score", type=float, default=0.20)
    parser.add_argument("--camera-iterations", type=int, default=250)
    parser.add_argument("--body-iterations", type=int, default=800)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    frame_root = output / "frames"
    frame_root.mkdir()
    script = Path(__file__).with_name("fit_smpl_coco_single_frame.py")
    pair_ids = list(range(args.start_pair, args.end_pair + 1))

    def run_pair(pair_id: int) -> dict:
        frame_output = frame_root / f"frame_{pair_id:04d}"
        command = [
            sys.executable, "-u", str(script),
            "--stereo-jsonl", str(args.stereo_jsonl.resolve()),
            "--calibration-dir", str(args.calibration_dir.resolve()),
            "--model-path", str(args.model_path.resolve()),
            "--regressor", str(args.regressor.resolve()),
            "--left-upright-image", str((args.left_image_dir / f"pair_{pair_id:04d}.png").resolve()),
            "--right-upright-image", str((args.right_image_dir / f"pair_{pair_id:04d}.png").resolve()),
            "--pair-id", str(pair_id),
            "--output-dir", str(frame_output),
            "--minimum-score", str(args.minimum_score),
            "--camera-iterations", str(args.camera_iterations),
            "--body-iterations", str(args.body_iterations),
            "--device", args.device,
            "--skip-overlays",
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        status = {
            "pair_id": pair_id,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
        if completed.returncode == 0:
            status["metrics_path"] = str(frame_output / "metrics.json")
        return status

    run_status: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run_pair, pair_id): pair_id for pair_id in pair_ids}
        for future in as_completed(futures):
            status = future.result()
            run_status.append(status)
            print(json.dumps({"finished_pair": status["pair_id"], "returncode": status["returncode"]}))
    run_status.sort(key=lambda item: item["pair_id"])
    failures = [item for item in run_status if item["returncode"] != 0]

    reports = []
    results = []
    for status in run_status:
        if status["returncode"] != 0:
            continue
        frame_dir = frame_root / f"frame_{status['pair_id']:04d}"
        reports.append(json.loads((frame_dir / "metrics.json").read_text(encoding="utf-8")))
        results.append(np.load(frame_dir / "result.npz"))

    left_errors: list[float] = []
    right_errors: list[float] = []
    jointwise = {
        name: {"left_px": [], "right_px": [], "diagnostic_left_px": [], "diagnostic_right_px": []}
        for name in COCO17_NAMES
    }
    for report in reports:
        for name in COCO17_NAMES:
            item = report["final"]["jointwise"][name]
            jointwise[name]["diagnostic_left_px"].append(item["left_px"])
            jointwise[name]["diagnostic_right_px"].append(item["right_px"])
            if item["left_supervised"]:
                left_errors.append(item["left_px"])
                jointwise[name]["left_px"].append(item["left_px"])
            if item["right_supervised"]:
                right_errors.append(item["right_px"])
                jointwise[name]["right_px"].append(item["right_px"])

    per_joint = {}
    for name, values in jointwise.items():
        combined = values["left_px"] + values["right_px"]
        per_joint[name] = {
            "supervised_samples": len(combined),
            "combined_p95_px": percentile(combined, 95),
            "left_p95_px": percentile(values["left_px"], 95),
            "right_p95_px": percentile(values["right_px"], 95),
            "diagnostic_left_p95_px": percentile(values["diagnostic_left_px"], 95),
            "diagnostic_right_p95_px": percentile(values["diagnostic_right_px"], 95),
        }

    temporal = {}
    boundary = {}
    if results:
        roots = np.concatenate([item["root_orient"] for item in results], axis=0)
        translations = np.concatenate([item["translation_m"] for item in results], axis=0)
        poses = np.concatenate([item["body_pose"] for item in results], axis=0)
        coco = np.stack([item["coco_left_m"] for item in results])
        root_jumps = rotation_jump_magnitudes(roots)
        translation_jumps = np.linalg.norm(np.diff(translations, axis=0), axis=1) * 1000.0
        accelerations = np.linalg.norm(np.diff(coco, n=2, axis=0), axis=2) * 1000.0
        temporal = {
            "root_orientation_jump_p95_rad": percentile(root_jumps.tolist(), 95),
            "root_orientation_jump_max_rad": float(root_jumps.max()) if len(root_jumps) else None,
            "translation_jump_p95_mm": percentile(translation_jumps.tolist(), 95),
            "translation_jump_max_mm": float(translation_jumps.max()) if len(translation_jumps) else None,
            "supervised_coco_acceleration_p95_mm_per_frame2": percentile(
                accelerations[:, [i for i in range(17) if i != RIGHT_KNEE_INDEX]].reshape(-1).tolist(), 95
            ) if len(accelerations) else None,
            "right_knee_acceleration_p95_mm_per_frame2": percentile(
                accelerations[:, RIGHT_KNEE_INDEX].tolist(), 95
            ) if len(accelerations) else None,
        }
        boundary = {
            "pose_components_at_abs_2_5": int(np.sum(np.abs(poses) >= 2.499)),
            "frames_with_pose_component_at_abs_2_5": [
                reports[index]["pair_id"] for index in np.flatnonzero(np.any(np.abs(poses) >= 2.499, axis=1))
            ],
            "maximum_abs_pose_component_rad": float(np.max(np.abs(poses))),
        }

    summary = {
        "schema_version": "pmpose_smpl_coco_independent_window_v1",
        "window": [args.start_pair, args.end_pair],
        "requested_frames": len(pair_ids),
        "successful_frames": len(reports),
        "failed_frames": failures,
        "fit_contract": {
            "model": "official_male_SMPL_6890",
            "observation_model": "Pose2Mesh_COCO17_17x6890",
            "initialization": "clean_zero_per_frame_with_2d_ray_depth_candidates",
            "pose": "three_axis_except_frozen_ankle_foot_wrist_hand_rotations",
            "betas": "fixed_zero",
            "losses": ["stereo_fisheye_2d_huber", "minimal_l2_pose_prior"],
            "excluded": ["stored_3d", "temporal", "contact", "old_fit_state"],
            "right_knee_supervision_weight": 0,
        },
        "reprojection": {
            "combined_median_px": percentile(left_errors + right_errors, 50),
            "combined_p95_px": percentile(left_errors + right_errors, 95),
            "left_median_px": percentile(left_errors, 50),
            "left_p95_px": percentile(left_errors, 95),
            "right_median_px": percentile(right_errors, 50),
            "right_p95_px": percentile(right_errors, 95),
            "per_joint": per_joint,
        },
        "temporal_diagnostic_only": temporal,
        "parameter_boundary": boundary,
        "negative_depth_points_left_total": int(sum(r["final"]["negative_depth_points_left"] for r in reports)),
        "negative_depth_points_right_total": int(sum(r["final"]["negative_depth_points_right"] for r in reports)),
        "conclusion_boundary": (
            "All fits are independent. Temporal statistics diagnose frame-to-frame branch stability; "
            "they are not temporal loss results or true 2-D/3-D accuracy."
        ),
    }
    (output / "run_status.json").write_text(
        json.dumps(run_status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
