#!/usr/bin/env python3
"""Read-only, per-camera ChArUco intrinsic validation with a fixed stereo pose.

Each image estimates only the board pose using its camera's existing fisheye
intrinsics.  The tool then measures raw-pixel reprojection residuals separately
for cam0 and cam1.  When both sides pass detection, their independently fitted
board poses are also compared through the supplied fixed stereo extrinsics.

This is an independent-image self-consistency check for fixed intrinsics; it
does not refit K/D, modify calibration, establish human-pose accuracy, or turn
host pairing timestamps into exposure synchronization evidence.
"""

from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
_tool_prepare_imports()

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


BOARD_SIZE = (8, 6)
SQUARE_MM = 30.0
MARKER_MM = 22.0
RADIAL_BINS = (0.0, 0.25, 0.5, 0.75, 1.0, math.inf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--cam0-calib", required=True, type=Path)
    parser.add_argument("--cam1-calib", required=True, type=Path)
    parser.add_argument("--stereo-calib", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-corners", type=int, default=8)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    raw = load_json(path)
    K = np.asarray(raw["K"], dtype=np.float64).reshape(3, 3)
    D = np.asarray(raw["D"], dtype=np.float64).reshape(4, 1)
    size = tuple(int(value) for value in raw["image_size"])
    if len(size) != 2 or min(size) <= 0:
        raise ValueError(f"Invalid image_size: {path}")
    return K, D, (size[0], size[1])


def load_stereo(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = load_json(path)
    if "R_cam0_to_cam1" in raw:
        R = raw["R_cam0_to_cam1"]
        T = raw["T_cam0_to_cam1_mm"]
    else:
        stereo = raw.get("stereo")
        if not isinstance(stereo, dict):
            raise ValueError(f"No stereo extrinsics in {path}")
        R, T = stereo["R"], stereo["T"]
    return np.asarray(R, dtype=np.float64).reshape(3, 3), np.asarray(T, dtype=np.float64).reshape(3, 1)


def make_detector() -> tuple[Any, np.ndarray]:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(BOARD_SIZE, SQUARE_MM, MARKER_MM, dictionary)
    return cv2.aruco.CharucoDetector(board), np.asarray(board.getChessboardCorners(), dtype=np.float64).reshape(-1, 3)


def detect(detector: Any, image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    corners, ids, _, _ = detector.detectBoard(image)
    if corners is None or ids is None:
        return None
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    if len(corners) != len(ids) or not len(ids):
        return None
    return corners, ids


def estimate_pose_and_errors(
    observed: np.ndarray, ids: np.ndarray, board_xyz: np.ndarray, K: np.ndarray, D: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    object_points = board_xyz[ids].reshape(-1, 1, 3)
    normalized = cv2.fisheye.undistortPoints(observed.reshape(-1, 1, 2), K, D)
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        normalized,
        np.eye(3, dtype=np.float64),
        np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None
    reprojected, _ = cv2.fisheye.projectPoints(object_points, rvec, tvec, K, D)
    reprojected = reprojected.reshape(-1, 2)
    errors = np.linalg.norm(reprojected - observed, axis=1)
    normalized_xy = normalized.reshape(-1, 2)
    normalized_radius = np.linalg.norm(normalized_xy, axis=1)
    rotation, _ = cv2.Rodrigues(rvec)
    return rotation, tvec.reshape(3, 1), reprojected, errors, normalized_radius


def numeric(values: list[float]) -> dict[str, int | float | None]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def bin_label(radius: float) -> str:
    for lower, upper in zip(RADIAL_BINS[:-1], RADIAL_BINS[1:]):
        if lower <= radius < upper:
            return f"[{lower:.2f},{upper:.2f})"
    return f"[{RADIAL_BINS[-2]:.2f},inf)"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.min_corners < 4:
        raise ValueError("--min-corners must be at least 4")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    session = args.session.resolve()
    detector, board_xyz = make_detector()
    K0, D0, size0 = load_intrinsics(args.cam0_calib.resolve())
    K1, D1, size1 = load_intrinsics(args.cam1_calib.resolve())
    R01, T01 = load_stereo(args.stereo_calib.resolve())
    configs = {"cam0": (K0, D0, size0), "cam1": (K1, D1, size1)}

    names0 = {path.stem.removesuffix("_cam0") for path in (session / "cam0").glob("pair_*_cam0.png")}
    names1 = {path.stem.removesuffix("_cam1") for path in (session / "cam1").glob("pair_*_cam1.png")}
    names = sorted(names0 | names1)
    if not names:
        raise ValueError(f"No ChArUco images under {session}")

    corner_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    all_errors: dict[str, list[float]] = defaultdict(list)
    radial_errors: dict[tuple[str, str], list[float]] = defaultdict(list)
    detected_poses: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)

    for pair_name in names:
        for camera in ("cam0", "cam1"):
            path = session / camera / f"{pair_name}_{camera}.png"
            K, D, expected_size = configs[camera]
            base = {"pair": pair_name, "camera": camera, "image_path": str(path)}
            if not path.is_file():
                image_rows.append({**base, "status": "missing_image", "detected_corners": 0, "reprojection_median_px": None, "reprojection_p95_px": None, "reprojection_max_px": None, "max_normalized_radius": None})
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or (image.shape[1], image.shape[0]) != expected_size:
                image_rows.append({**base, "status": "invalid_image_size", "detected_corners": 0, "reprojection_median_px": None, "reprojection_p95_px": None, "reprojection_max_px": None, "max_normalized_radius": None})
                continue
            found = detect(detector, image)
            if found is None:
                image_rows.append({**base, "status": "detection_failed", "detected_corners": 0, "reprojection_median_px": None, "reprojection_p95_px": None, "reprojection_max_px": None, "max_normalized_radius": None})
                continue
            observed, ids = found
            if len(ids) < args.min_corners:
                image_rows.append({**base, "status": "too_few_corners", "detected_corners": int(len(ids)), "reprojection_median_px": None, "reprojection_p95_px": None, "reprojection_max_px": None, "max_normalized_radius": None})
                continue
            solved = estimate_pose_and_errors(observed, ids, board_xyz, K, D)
            if solved is None:
                image_rows.append({**base, "status": "pnp_failed", "detected_corners": int(len(ids)), "reprojection_median_px": None, "reprojection_p95_px": None, "reprojection_max_px": None, "max_normalized_radius": None})
                continue
            rotation, translation, reprojected, errors, radii = solved
            detected_poses[pair_name][camera] = (rotation, translation)
            all_errors[camera].extend(float(value) for value in errors)
            image_rows.append({**base, "status": "accepted", "detected_corners": int(len(ids)), "reprojection_median_px": float(np.median(errors)), "reprojection_p95_px": float(np.percentile(errors, 95)), "reprojection_max_px": float(np.max(errors)), "max_normalized_radius": float(np.max(radii))})
            for corner_id, observed_xy, projected_xy, error, radius in zip(ids, observed, reprojected, errors, radii):
                label = bin_label(float(radius))
                radial_errors[(camera, label)].append(float(error))
                corner_rows.append({**base, "corner_id": int(corner_id), "observed_x_px": float(observed_xy[0]), "observed_y_px": float(observed_xy[1]), "projected_x_px": float(projected_xy[0]), "projected_y_px": float(projected_xy[1]), "reprojection_error_px": float(error), "normalized_radius": float(radius), "normalized_radius_bin": label})

        poses = detected_poses[pair_name]
        if "cam0" not in poses or "cam1" not in poses:
            pair_rows.append({"pair": pair_name, "status": "missing_single_camera_pose", "external_rotation_disagreement_deg": None, "external_translation_disagreement_mm": None})
            continue
        R0, T0 = poses["cam0"]
        R1, T1 = poses["cam1"]
        R1_expected = R01 @ R0
        T1_expected = R01 @ T0 + T01
        pair_rows.append({"pair": pair_name, "status": "accepted", "external_rotation_disagreement_deg": rotation_angle_deg(R1 @ R1_expected.T), "external_translation_disagreement_mm": float(np.linalg.norm(T1 - T1_expected))})

    if not corner_rows:
        raise RuntimeError("No valid per-camera intrinsic observations")
    output.mkdir(parents=True)
    write_csv(output / "per_corner_intrinsic_residual.csv", corner_rows)
    write_csv(output / "per_image_intrinsic_residual.csv", image_rows)
    write_csv(output / "per_pair_fixed_extrinsic_pose_agreement.csv", pair_rows)
    camera_summary = {camera: numeric(all_errors[camera]) for camera in ("cam0", "cam1")}
    radial_summary = [
        {"camera": camera, "normalized_radius_bin": label, **numeric(values)}
        for (camera, label), values in sorted(radial_errors.items())
    ]
    write_csv(output / "per_camera_radial_residual_summary.csv", radial_summary)
    accepted_pairs = [row for row in pair_rows if row["status"] == "accepted"]
    result = {
        "scope": "Fixed intrinsics, per-image board-pose estimation, raw-pixel reprojection audit, and fixed-extrinsic cross-camera board-pose agreement.",
        "session": str(session),
        "inputs": {"cam0_intrinsics": str(args.cam0_calib.resolve()), "cam1_intrinsics": str(args.cam1_calib.resolve()), "fixed_stereo_extrinsics": str(args.stereo_calib.resolve())},
        "board": {"dictionary": "DICT_4X4_50", "squares_x": BOARD_SIZE[0], "squares_y": BOARD_SIZE[1], "square_mm": SQUARE_MM, "marker_mm": MARKER_MM},
        "min_corners": args.min_corners,
        "camera_reprojection_px": camera_summary,
        "fixed_extrinsic_cross_camera_pose_agreement": {
            "accepted_pairs": len(accepted_pairs),
            "rotation_disagreement_deg": numeric([float(row["external_rotation_disagreement_deg"]) for row in accepted_pairs]),
            "translation_disagreement_mm": numeric([float(row["external_translation_disagreement_mm"]) for row in accepted_pairs]),
        },
        "interpretation": "This tests fixed intrinsics on images outside their original calibration fit and reports compatibility with the fixed stereo extrinsics. It does not refit intrinsics/extrinsics, provide human-pose accuracy, or prove physical ground-coordinate accuracy.",
    }
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "cam0_median_px": camera_summary["cam0"]["median"], "cam1_median_px": camera_summary["cam1"]["median"], "cross_camera_pairs": len(accepted_pairs)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
