"""Audit DA3 raw-fisheye outputs at Sapiens2 pseudo-labelled lower-body points.

This intentionally measures only what the saved DA3 arrays support: finite
outputs, relative depth/confidence values at approximate rescaled keypoint
locations, and DA3's *own* two-view pseudo-geometry consistency.  Raw fisheye
input is not a calibrated pinhole camera, and DA3 did not persist an exact
preprocessing transform, so the audit cannot establish metric geometry.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PAIR_TO_DISTANCE = {"pair_010.png": "far_3m", "pair_030.png": "mid_2m", "pair_050.png": "near_1p3m"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--da3-root", required=True, type=Path)
    parser.add_argument("--reference-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_reference(path: Path) -> dict[tuple[str, str, str], dict]:
    result = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["file_name"] in PAIR_TO_DISTANCE and row["reference_usable"] == "true":
                result[(row["file_name"], row["side"], row["joint_subject_anatomy"])] = row
    return result


def sample_index(x_raw: float, y_raw: float, width: int, height: int) -> tuple[int, int]:
    # The original-to-export transform was not persisted by DA3.  This direct
    # resize mapping is an explicit approximate sampling convention.
    x = int(np.clip(round(x_raw * (width - 1) / (1920 - 1)), 0, width - 1))
    y = int(np.clip(round(y_raw * (height - 1) / (1080 - 1)), 0, height - 1))
    return x, y


def point_world(x: int, y: int, depth: float, K: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    ray = np.linalg.solve(K, np.asarray([float(x), float(y), 1.0]))
    camera = ray * float(depth)
    R, t = w2c[:, :3], w2c[:, 3]
    return R.T @ (camera - t)


def camera_center(w2c: np.ndarray) -> np.ndarray:
    R, t = w2c[:, :3], w2c[:, 3]
    return -R.T @ t


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    reference = load_reference(args.reference_csv.resolve())
    if not reference:
        raise RuntimeError("No usable Sapiens2 references for DA3 pair_010/030/050")
    rows, consistency = [], []
    for file_name, distance in PAIR_TO_DISTANCE.items():
        archive = np.load(args.da3_root.resolve() / distance / "exports" / "mini_npz" / "results.npz")
        depth, conf, intrinsics, extrinsics = (archive[key] for key in ("depth", "conf", "intrinsics", "extrinsics"))
        if depth.shape != conf.shape or depth.shape[0] != 2:
            raise RuntimeError(f"Unexpected DA3 shape for {distance}: {depth.shape}")
        height, width = depth.shape[1:]
        by_side_joint = {}
        for side_index, side in enumerate(("left", "right")):
            for joint in ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"):
                ref = reference.get((file_name, side, joint))
                if ref is None:
                    continue
                x, y = sample_index(float(ref["x_px_raw_fisheye"]), float(ref["y_px_raw_fisheye"]), width, height)
                d, c = float(depth[side_index, y, x]), float(conf[side_index, y, x])
                world = point_world(x, y, d, intrinsics[side_index], extrinsics[side_index])
                entry = {
                    "distance_condition": distance, "file_name": file_name, "camera_side": side, "joint": joint,
                    "raw_x_px": float(ref["x_px_raw_fisheye"]), "raw_y_px": float(ref["y_px_raw_fisheye"]),
                    "sapiens2_confidence": float(ref["sapiens2_confidence"]), "da3_x_px_approx": x, "da3_y_px_approx": y,
                    "da3_depth_relative": d, "da3_confidence_raw": c,
                    "depth_finite": bool(np.isfinite(d)), "confidence_finite": bool(np.isfinite(c)),
                }
                rows.append(entry)
                by_side_joint[(side, joint)] = (world, entry)
        baseline = float(np.linalg.norm(camera_center(extrinsics[0]) - camera_center(extrinsics[1])))
        for joint in ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"):
            left, right = by_side_joint.get(("left", joint)), by_side_joint.get(("right", joint))
            if left is None or right is None:
                continue
            discrepancy = float(np.linalg.norm(left[0] - right[0]))
            consistency.append({
                "distance_condition": distance, "file_name": file_name, "joint": joint,
                "da3_left_right_pseudo3d_discrepancy": discrepancy,
                "da3_predicted_camera_baseline": baseline,
                "discrepancy_over_predicted_baseline": discrepancy / baseline if baseline > 1e-12 else None,
            })
    output.mkdir(parents=True)
    for file_name, data in (("da3_at_sapiens2_reference_points.csv", rows), ("da3_pseudo3d_crossview_consistency.csv", consistency)):
        with (output / file_name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    summary = {}
    for distance in PAIR_TO_DISTANCE.values():
        local = [row for row in rows if row["distance_condition"] == distance]
        cross = [row for row in consistency if row["distance_condition"] == distance]
        summary[distance] = {
            "usable_reference_samples": len(local),
            "finite_depth_fraction": float(np.mean([row["depth_finite"] for row in local])),
            "finite_confidence_fraction": float(np.mean([row["confidence_finite"] for row in local])),
            "relative_depth_median": float(np.median([row["da3_depth_relative"] for row in local])),
            "confidence_median_raw": float(np.median([row["da3_confidence_raw"] for row in local])),
            "crossview_same_joint_count": len(cross),
            "pseudo3d_discrepancy_over_predicted_baseline_median": None if not cross else float(np.median([row["discrepancy_over_predicted_baseline"] for row in cross])),
            "pseudo3d_discrepancy_over_predicted_baseline_max": None if not cross else float(np.max([row["discrepancy_over_predicted_baseline"] for row in cross])),
        }
    metadata = {
        "reference": str(args.reference_csv.resolve()),
        "da3_root": str(args.da3_root.resolve()),
        "sample_coordinate_rule": "approximate direct resize from raw 1920x1080 to DA3 504x280; exact DA3 preprocessing transform was not persisted",
        "geometry_rule": "backproject each DA3 point with DA3-predicted pinhole intrinsics/extrinsics, then compare left/right same-joint pseudo-3D positions",
        "interpretation_boundary": "DA3 received raw fisheye images without calibration. Its depth/confidence and pseudo-geometry are diagnostic model outputs, not metric calibrated stereo or an independent keypoint truth source.",
        "summary": summary,
    }
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
