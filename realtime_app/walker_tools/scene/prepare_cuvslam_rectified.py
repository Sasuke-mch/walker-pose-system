"""Prepare a cuVSLAM-only pinhole stereo diagnostic from the frozen raw manifest.

No calibration fitting, Stage input, temporal filling, or image mask blacking is
used. Every remapped image pixel is allowed only if all four source samples used
by bilinear interpolation are in bounds and allowed by the original mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from walker_tools.scene.cuvslam_backend import validate_manifest


def stereo_rectification(calibration: dict) -> tuple[dict, dict, list[tuple[np.ndarray, np.ndarray]]]:
    if calibration["model"] != "opencv_fisheye":
        raise ValueError("Rectification requires the frozen raw OpenCV fisheye calibration")
    size = tuple(calibration["image_size"])
    k1, k2 = [np.asarray(calibration[f"{side}_K"], dtype=float) for side in ("left", "right")]
    d1, d2 = [np.asarray(calibration[f"{side}_D"], dtype=float) for side in ("left", "right")]
    rotation = np.asarray(calibration["R_right_from_left"], dtype=float)
    translation = np.asarray(calibration["translation_right_from_left_m"], dtype=float)
    r1, r2, p1, p2, _ = cv2.fisheye.stereoRectify(
        k1, d1, k2, d2, size, rotation, translation, cv2.CALIB_ZERO_DISPARITY,
        newImageSize=size, balance=0.0, fov_scale=1.0,
    )
    rectified_rotation = r2 @ rotation @ r1.T
    rectified_translation = r2 @ translation
    if not np.allclose(rectified_rotation, np.eye(3), atol=1e-8):
        raise ValueError("Rectification failed to align stereo camera axes")
    if not np.isclose(np.linalg.norm(rectified_translation), np.linalg.norm(translation), atol=1e-10):
        raise ValueError("Rectification changed the metric baseline")
    if not np.allclose(p2[:3, 3], p2[:3, :3] @ rectified_translation, atol=1e-7):
        raise ValueError("Rectified right projection baseline direction disagrees with the rig")
    maps = [cv2.fisheye.initUndistortRectifyMap(k, d, r, p[:3, :3], size, cv2.CV_32FC1)
            for k, d, r, p in ((k1, d1, r1, p1), (k2, d2, r2, p2))]
    active = {
        "model": "opencv_pinhole", "image_size": list(size),
        "left_K": p1[:3, :3].tolist(), "right_K": p2[:3, :3].tolist(),
        "left_D": [], "right_D": [], "R_right_from_left": rectified_rotation.tolist(),
        "translation_right_from_left_m": rectified_translation.tolist(),
        "left_camera_axes": "rectified_left_camera", "translation_unit": "m",
    }
    details = {
        "method": "opencv_fisheye_stereoRectify", "balance": 0.0, "fov_scale": 1.0,
        "R_rectified_left_from_original_left": r1.tolist(),
        "R_rectified_right_from_original_right": r2.tolist(),
        "P_left_rectified": p1.tolist(), "P_right_rectified": p2.tolist(),
        "source_calibration": calibration,
        "image_interpolation": "bilinear",
        "mask_rule": "all_four_bilinear_source_pixels_must_be_in_bounds_and_allowed; no_mask_interpolation",
        "raw_pose_conversion": "world_from_original_left = world_from_rectified_left @ R_rectified_left_from_original_left",
        "raw_landmark_conversion": "original_left_xyz = R_rectified_left_from_original_left.T @ current_rectified_rig_xyz",
        "shared_prepare_modified": False, "stage_used": False,
    }
    return active, details, maps


def conservative_rectified_mask(mask: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    finite = np.isfinite(map_x) & np.isfinite(map_y)
    x0 = np.floor(np.where(finite, map_x, -2)).astype(np.int32)
    y0 = np.floor(np.where(finite, map_y, -2)).astype(np.int32)
    valid = finite & (x0 >= 0) & (y0 >= 0) & (x0 + 1 < mask.shape[1]) & (y0 + 1 < mask.shape[0])
    allowed = np.zeros(map_x.shape, dtype=bool)
    y, x = y0[valid], x0[valid]
    allowed[valid] = ((mask[y, x] == 0) & (mask[y, x + 1] == 0) &
                      (mask[y + 1, x] == 0) & (mask[y + 1, x + 1] == 0))
    return np.where(allowed, 0, 255).astype(np.uint8)


def prepare(manifest_path: Path, output_dir: Path, max_frames: int | None = None) -> Path:
    source = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    validate_manifest(source)
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        source["frames"] = source["frames"][:max_frames]
    calibration, details, maps = stereo_rectification(source["calibration"])
    output_dir.mkdir(parents=True, exist_ok=False)
    result = dict(source)
    result.update(calibration=calibration, rectification=details,
                  source_raw_manifest=str(manifest_path.resolve()), image_coordinates="rectified_pinhole")
    result["frames"] = []
    mask_fractions = []
    for side in ("left", "right"):
        for kind in ("image", "mask"):
            (output_dir / side / kind).mkdir(parents=True)
    for frame in source["frames"]:
        row = dict(frame)
        fractions = []
        for side, (map_x, map_y) in zip(("left", "right"), maps):
            image = cv2.imread(frame[f"{side}_path"], cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(frame[f"{side}_mask_path"], cv2.IMREAD_GRAYSCALE)
            expected = tuple(reversed(source["calibration"]["image_size"]))
            if image is None or mask is None or image.shape != expected or mask.shape != expected:
                raise ValueError(f"Missing/wrongly sized original {side} frame {frame['frame_id']}")
            rectified_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            rectified_mask = conservative_rectified_mask(mask, map_x, map_y)
            image_path = output_dir / side / "image" / f"{frame['frame_id']:04d}.png"
            mask_path = output_dir / side / "mask" / f"{frame['frame_id']:04d}.png"
            for path, pixels in ((image_path, rectified_image), (mask_path, rectified_mask)):
                if not cv2.imwrite(str(path), pixels, [cv2.IMWRITE_PNG_COMPRESSION, 2]):
                    raise OSError(f"Cannot save {path}")
            row[f"source_raw_{side}_path"] = frame[f"{side}_path"]
            row[f"source_raw_{side}_mask_path"] = frame[f"{side}_mask_path"]
            row[f"{side}_path"] = str(image_path.resolve())
            row[f"{side}_mask_path"] = str(mask_path.resolve())
            # Raw test-mask pixels are not rectified observations. The input's
            # failed holdout remains documented but is not silently reused.
            row.pop(f"{side}_test_mask_path", None)
            fractions.append(float(np.count_nonzero(rectified_mask) / rectified_mask.size))
        result["frames"].append(row)
        mask_fractions.append(fractions)
    details["excluded_pixel_fraction_per_frame"] = mask_fractions
    target = output_dir / "manifest.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(target), "frames": len(result["frames"]),
                      "excluded_fraction_median": np.median(mask_fractions, axis=0).tolist()}))
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    prepare(args.manifest, args.output_dir, args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
