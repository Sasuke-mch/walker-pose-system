"""Stage-independent cuVSLAM stereo runner for the common scene manifest.

Run in the isolated Linux cuVSLAM environment. Manifest images are unrectified
OpenCV-fisheye pixels; mask PNG values != 0 mean EXCLUDED. Translation is in
metres. Rig origin is the left optical centre, with OpenCV x-right/y-down/z-forward.
The emitted world is a visual map gauge, not an independently certified ground.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np


def quaternion_xyzw(rotation: np.ndarray) -> list[float]:
    """Convert a proper rotation matrix, including rotations near pi."""
    r = np.asarray(rotation, dtype=np.float64)
    values, vectors = np.linalg.eigh(np.array([
        [r[0, 0]-r[1, 1]-r[2, 2], r[1, 0]+r[0, 1], r[2, 0]+r[0, 2], r[2, 1]-r[1, 2]],
        [r[1, 0]+r[0, 1], r[1, 1]-r[0, 0]-r[2, 2], r[2, 1]+r[1, 2], r[0, 2]-r[2, 0]],
        [r[2, 0]+r[0, 2], r[2, 1]+r[1, 2], r[2, 2]-r[0, 0]-r[1, 1], r[1, 0]-r[0, 1]],
        [r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1], np.trace(r)],
    ]) / 3.0)
    q = vectors[:, np.argmax(values)]
    return (q if q[3] >= 0 else -q).tolist()


def matrix_from_pose(pose: Any) -> np.ndarray:
    """cuVSLAM Pose quaternion order is x,y,z,w; translation is metres."""
    q = np.asarray(pose.rotation, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("cuVSLAM returned invalid quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    result = np.eye(4)
    result[:3, :3] = [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]
    result[:3, 3] = np.asarray(pose.translation, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("cuVSLAM returned non-finite pose")
    return result


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("Expected common scene manifest schema_version=1")
    calibration = manifest["calibration"]
    if calibration["model"] != "opencv_fisheye":
        raise ValueError("This backend requires OpenCV fisheye calibration")
    rotation = np.asarray(calibration["R_right_from_left"], dtype=np.float64)
    translation = np.asarray(calibration["translation_right_from_left_m"], dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("Invalid stereo rotation")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("Stereo rotation is not SO(3)")
    if translation.shape != (3,) or not np.isfinite(translation).all() or np.linalg.norm(translation) <= 1e-6:
        raise ValueError("Stereo translation must be finite, nonzero, in metres")
    timestamps = [int(frame["timestamp_ns"]) for frame in manifest["frames"]]
    if not timestamps or any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("Input timestamps must be nonempty and strictly increasing")


def build_rig(cuvslam: Any, calibration: dict[str, Any]) -> Any:
    rotation = np.asarray(calibration["R_right_from_left"], dtype=np.float64)
    translation = np.asarray(calibration["translation_right_from_left_m"], dtype=np.float64)
    # X_right = R_right_from_left X_left + t; rig_from_right is its inverse.
    right_to_left_r = rotation.T
    right_to_left_t = -rotation.T @ translation
    cameras = []
    for side, pose in (
        ("left", cuvslam.Pose([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0])),
        ("right", cuvslam.Pose(quaternion_xyzw(right_to_left_r), right_to_left_t.tolist())),
    ):
        k = np.asarray(calibration[f"{side}_K"], dtype=np.float64)
        d = np.asarray(calibration[f"{side}_D"], dtype=np.float64).reshape(-1)
        if k.shape != (3, 3) or d.shape != (4,) or not np.isfinite(k).all() or not np.isfinite(d).all():
            raise ValueError(f"Invalid {side} fisheye intrinsics")
        cameras.append(cuvslam.Camera(
            size=calibration["image_size"],
            focal=[float(k[0, 0]), float(k[1, 1])],
            principal=[float(k[0, 2]), float(k[1, 2])],
            distortion=cuvslam.Distortion(cuvslam.Distortion.Model.Fisheye, d.tolist()),
            rig_from_camera=pose,
        ))
    return cuvslam.Rig(cameras=cameras)


def load_inputs(frame: dict[str, Any], size: list[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    images, masks = [], []
    for side in ("left", "right"):
        image = cv2.imread(frame[f"{side}_path"], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(frame[f"{side}_mask_path"], cv2.IMREAD_GRAYSCALE)
        expected = (int(size[1]), int(size[0]))
        if image is None or mask is None or image.shape != expected or mask.shape != expected:
            raise ValueError(f"Missing or wrongly sized {side} image/mask at frame {frame['frame_id']}")
        images.append(np.ascontiguousarray(image))
        masks.append(np.ascontiguousarray((mask != 0).astype(np.uint8) * 255))
    return images, masks


def base_record(frame: dict[str, Any], timestamp_source: str) -> dict[str, Any]:
    return {
        "backend": "cuvslam", "frame_id": frame["frame_id"], "pair_id": frame["pair_id"],
        "timestamp_ns": frame["timestamp_ns"], "timestamp_source": timestamp_source,
        "status": "unavailable", "world_from_left_camera_m": None,
        "native_pose_returned": False, "native_world_from_left_camera_proposal_m": None,
        "map_world_from_left_camera_m": None, "track_ms": None,
        "landmark_count": None, "finite_landmark_count": None,
        "observation_count": None, "native_warming_up": None,
        "world_pose_certified": False, "metric_pose_usable": False,
        "metric_pose_usable_reason": "independent_metric_quality_gate_not_passed",
        "stage_used": False, "confidence_probability": None,
    }


def run(manifest_path: Path, output_path: Path, max_frames: int | None, use_slam: bool) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    validate_manifest(manifest)
    frames = manifest["frames"][:max_frames] if max_frames is not None else manifest["frames"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output_path}")
    timestamp_source = manifest.get("timestamp_source", "manifest_nominal_not_exposure_verified")
    metadata: dict[str, Any] = {
        "backend": "cuvslam", "manifest_path": str(manifest_path), "frame_count": len(frames),
        "use_slam": use_slam, "multicamera_mode": "Precision", "use_gpu": True,
        "rig_origin": "left_optical_center", "image_coordinates": "raw_opencv_fisheye",
        "mask_nonzero_excluded": True, "mask_effectiveness_certified": False,
        "timestamp_source": timestamp_source,
        "world_pose_certified": False, "stage_used": False,
        "track_timing": "native_tracker_call_including_cpu_to_gpu_transfer_and_bundle_adjustment",
        "frame_total_timing": "image_and_mask_load_plus_tracking_and_diagnostics_excluding_json_write",
        "confidence_interpretation": "native_covariance_is_uncalibrated_solver_approximation; observation_count_is_not_probability",
        "native_pose_availability_is_not_metric_support": True,
        "tracked_status_scope": "native_pose_plus_nonempty_last_frame_3d_export; not_metric_quality_acceptance",
        "landmark_count_scope": "last_frame_exported_tracks3d; not_total_slam_map_size_or_solver_inlier_count",
        "observation_count_scope": "last_frame_exported_tracks2d_per_camera; not_pose_solver_inlier_count",
        "visual_update_vs_prediction_flag_available": False,
        "output_segment_scope": "conservative_break_after_missing_exported_support; not_native_map_reset_evidence",
    }
    initialized = time.perf_counter()
    try:
        import cuvslam
        metadata["cuvslam_version"] = importlib.metadata.version("cuvslam")
        rig = build_rig(cuvslam, manifest["calibration"])
        config = cuvslam.Tracker.OdometryConfig(
            use_gpu=True, multicam_mode=cuvslam.Tracker.MulticameraMode.Precision,
            odometry_mode=cuvslam.Tracker.OdometryMode.Multicamera,
            rectified_stereo_camera=False, async_sba=False,
            enable_observations_export=True, enable_landmarks_export=True,
        )
        metadata["native_use_motion_model"] = bool(config.use_motion_model)
        slam_config = cuvslam.Tracker.SlamConfig(sync_mode=True) if use_slam else None
        tracker = cuvslam.Tracker(rig, config, slam_config)
        # The installed wheel may select a subset despite the requested mode;
        # report the native rig decision instead of assuming both are primary.
        metadata["native_primary_cameras"] = list(tracker.odom.get_primary_cameras())
        metadata["initialization_ms"] = (time.perf_counter()-initialized)*1000.0
    except Exception as error:
        metadata["initialization_error"] = f"{type(error).__name__}: {error}"
        with output_path.open("w", encoding="utf-8") as stream:
            for frame in frames:
                row = base_record(frame, timestamp_source)
                row["reason"] = "backend_initialization_failed"
                row["error"] = metadata["initialization_error"]
                stream.write(json.dumps(row, allow_nan=False)+"\n")
        output_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(metadata["initialization_error"], file=sys.stderr)
        return 2

    started = time.perf_counter()
    tracked_count = 0
    segment_id = 0
    was_lost = False
    with output_path.open("w", encoding="utf-8") as stream:
        for frame in frames:
            frame_start = time.perf_counter()
            row = base_record(frame, timestamp_source)
            try:
                images, masks = load_inputs(frame, manifest["calibration"]["image_size"])
                row["excluded_pixel_fraction"] = [float(np.count_nonzero(m)/m.size) for m in masks]
                track_start = time.perf_counter()
                estimate, map_pose = tracker.track(int(frame["timestamp_ns"]), images, masks=masks)
                row["track_ms"] = (time.perf_counter()-track_start)*1000.0
                if estimate.world_from_rig is None:
                    row["status"] = "lost"
                    row["reason"] = "native_odometry_pose_unavailable"
                    was_lost = True
                else:
                    row["native_pose_returned"] = True
                    row["native_world_from_left_camera_proposal_m"] = matrix_from_pose(estimate.world_from_rig.pose).tolist()
                    covariance = np.asarray(estimate.world_from_rig.covariance_xyz_rpy, dtype=np.float64)
                    row["covariance_xyz_rpy_uncalibrated"] = covariance.reshape(6, 6).tolist() if covariance.size == 36 and np.isfinite(covariance).all() else None
                row["map_world_from_left_camera_m"] = matrix_from_pose(map_pose).tolist() if map_pose is not None else None
                # Preserve the native proposal separately. The primary pose
                # channel requires exported 3D support; missing diagnostics keep
                # that support unknown. These fields are not extra pose factors.
                try:
                    observations = [tracker.get_last_observations(i) for i in (0, 1)]
                    row["observation_count"] = [len(o) for o in observations]
                    landmarks = tracker.get_last_landmarks()
                    row["landmark_count"] = len(landmarks)
                    row["landmarks"] = []
                    for point in landmarks:
                        coords = np.asarray(point.coords, dtype=float)
                        finite = coords.shape == (3,) and np.isfinite(coords).all()
                        row["landmarks"].append({"id": int(point.id), "coords_m_native": coords.tolist() if finite else None, "finite": bool(finite)})
                    row["finite_landmark_count"] = sum(point["finite"] for point in row["landmarks"])
                    row["has_exported_3d_landmarks"] = row["finite_landmark_count"] > 0
                    # v17.0.1 GetLastLandmarks exports stat.tracks3d in the
                    # current rig frame. GetFinalLandmarks separately multiplies
                    # by the world pose. The Python coords doc calls it world;
                    # retain the native values and never consume them as world.
                    row["landmark_coordinate_frame"] = "current_rig_per_v17_source_and_reprojection; not_consumed_as_world"
                    row["observations"] = [[{"id": int(o.id), "u": float(o.u), "v": float(o.v)} for o in view] for view in observations]
                    # Subpixel/pyramid selection can straddle a one-pixel mask
                    # boundary. Preserve the actual overlap for common evaluation.
                    row["masked_observation_count_rounded"] = [
                        sum(int(mask[int(round(o.v)), int(round(o.u))] != 0) for o in view
                            if 0 <= round(o.v) < mask.shape[0] and 0 <= round(o.u) < mask.shape[1])
                        for view, mask in zip(observations, masks)
                    ]
                    state = tracker.odom.get_state()
                    row["native_warming_up"] = bool(state.warming_up) if state is not None else None
                    row["native_keyframe"] = bool(state.keyframe) if state is not None else None
                except Exception as error:
                    row["diagnostics_error"] = f"{type(error).__name__}: {error}"
                if row["native_pose_returned"]:
                    support = row["finite_landmark_count"]
                    if row["native_warming_up"] is True:
                        row["status"] = "initializing"
                        row["reason"] = "native_warming_up_with_pose_proposal"
                        was_lost = True
                    elif support is not None and support > 0 and row["native_warming_up"] is False:
                        if was_lost and tracked_count > 0:
                            segment_id += 1
                        was_lost = False
                        row["status"] = "tracked"
                        row["world_from_left_camera_m"] = row["native_world_from_left_camera_proposal_m"]
                        row["reason"] = "native_visual_pose_and_3d_support_available_not_external_accuracy_certified"
                        tracked_count += 1
                    elif support == 0:
                        row["status"] = "initializing" if tracked_count == 0 else "lost"
                        row["reason"] = "native_pose_proposal_without_any_exported_3d_landmarks"
                        was_lost = True
                    else:
                        row["status"] = "unavailable"
                        row["reason"] = "native_pose_proposal_metric_support_unknown"
                        was_lost = True
                row["segment_id"] = segment_id
            except Exception as error:
                row["status"] = "unavailable"
                row["world_from_left_camera_m"] = None
                row["reason"] = "frame_input_or_tracking_error"
                row["error"] = f"{type(error).__name__}: {error}"
                was_lost = True
            row["frame_total_ms"] = (time.perf_counter()-frame_start)*1000.0
            stream.write(json.dumps(row, allow_nan=False)+"\n")
            stream.flush()
    metadata.update(tracked_count=tracked_count, run_wall_s=time.perf_counter()-started)
    output_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "tracked": tracked_count, "frames": len(frames), "run_wall_s": metadata["run_wall_s"]}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--slam", action="store_true", help="Enable synchronous native SLAM; keep its map pose separate from VO")
    arguments = parser.parse_args()
    if arguments.max_frames is not None and arguments.max_frames <= 0:
        parser.error("--max-frames must be positive")
    return run(arguments.manifest, arguments.output, arguments.max_frames, arguments.slam)


if __name__ == "__main__":
    raise SystemExit(main())
