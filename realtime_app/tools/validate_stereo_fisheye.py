from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


BOARD_SQUARES = (8, 6)
SQUARE_MM = 30.0
MARKER_MM = 22.0


def load_json(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )


def load_intrinsics(path: Path):
    data = load_json(path)

    K = np.asarray(
        data["K"],
        dtype=np.float64,
    ).reshape(3, 3)

    D = np.asarray(
        data["D"],
        dtype=np.float64,
    ).reshape(4, 1)

    return K, D


def make_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50
    )

    board = cv2.aruco.CharucoBoard(
        BOARD_SQUARES,
        SQUARE_MM,
        MARKER_MM,
        dictionary,
    )

    detector = cv2.aruco.CharucoDetector(board)

    return detector


def detect(detector, image):
    corners, ids, _, _ = detector.detectBoard(image)

    if corners is None or ids is None:
        return None, None

    corners = np.asarray(
        corners,
        dtype=np.float64,
    ).reshape(-1, 2)

    ids = np.asarray(
        ids,
        dtype=np.int32,
    ).reshape(-1)

    return corners, ids


def common_points(c0, ids0, c1, ids1):
    map0 = {
        int(i): np.asarray(p, dtype=np.float64)
        for i, p in zip(ids0, c0)
    }

    map1 = {
        int(i): np.asarray(p, dtype=np.float64)
        for i, p in zip(ids1, c1)
    }

    ids = sorted(
        set(map0).intersection(map1)
    )

    if not ids:
        return None

    p0 = np.asarray(
        [map0[i] for i in ids],
        dtype=np.float64,
    ).reshape(-1, 1, 2)

    p1 = np.asarray(
        [map1[i] for i in ids],
        dtype=np.float64,
    ).reshape(-1, 1, 2)

    return ids, p0, p1


def pixels_to_unit_rays(points, K, D):
    normalized = cv2.fisheye.undistortPoints(
        points,
        K,
        D,
    ).reshape(-1, 2)

    rays = np.column_stack(
        [
            normalized[:, 0],
            normalized[:, 1],
            np.ones(len(normalized)),
        ]
    )

    good = np.all(
        np.isfinite(rays),
        axis=1,
    )

    rays = rays[good]

    norms = np.linalg.norm(
        rays,
        axis=1,
        keepdims=True,
    )

    good_norm = (
        norms[:, 0] > 1e-12
    )

    rays = rays[good_norm]
    norms = norms[good_norm]

    rays /= norms

    return rays, good, good_norm


def epipolar_angular_errors(
    rays0,
    rays1,
    R,
    T,
):
    """
    Evaluate epipolar consistency directly in ray space.

    Coordinate convention:
        X_cam1 = R * X_cam0 + T

    CAM0 ray expressed in CAM1:
        a = R * ray0

    CAM0 center expressed in CAM1:
        T

    The epipolar plane in CAM1 is spanned by:
        T and a

    Its normal:
        n = T x a

    A correct CAM1 ray b lies in this plane.
    """

    a = (
        R @ rays0.T
    ).T

    b = rays1

    Tvec = T.reshape(3)

    normals = np.cross(
        np.broadcast_to(
            Tvec,
            a.shape,
        ),
        a,
    )

    normal_norm = np.linalg.norm(
        normals,
        axis=1,
    )

    valid = (
        normal_norm > 1e-12
    )

    normals = normals[valid]
    b_valid = b[valid]

    denom = np.linalg.norm(
        normals,
        axis=1,
    )

    sine_error = np.abs(
        np.sum(
            normals * b_valid,
            axis=1,
        )
    ) / denom

    sine_error = np.clip(
        sine_error,
        0.0,
        1.0,
    )

    angle_rad = np.arcsin(
        sine_error
    )

    return angle_rad, valid


def triangulation_ray_gap(
    rays0,
    rays1,
    R,
    T,
):
    """
    Find the closest points between the two
    3D viewing rays.

    Returns:
        ray gap in mm
        CAM0 ray depth parameter
        CAM1 ray depth parameter
    """

    a_all = (
        R @ rays0.T
    ).T

    b_all = rays1

    Tvec = T.reshape(3)

    gaps = []
    depth0 = []
    depth1 = []

    for a, b in zip(
        a_all,
        b_all,
    ):
        A = np.column_stack(
            [a, -b]
        )

        solution, _, _, _ = (
            np.linalg.lstsq(
                A,
                -Tvec,
                rcond=None,
            )
        )

        s = float(solution[0])
        t = float(solution[1])

        point0 = (
            Tvec + s * a
        )

        point1 = (
            t * b
        )

        gap = float(
            np.linalg.norm(
                point0 - point1
            )
        )

        gaps.append(gap)
        depth0.append(s)
        depth1.append(t)

    return (
        np.asarray(
            gaps,
            dtype=np.float64,
        ),
        np.asarray(
            depth0,
            dtype=np.float64,
        ),
        np.asarray(
            depth1,
            dtype=np.float64,
        ),
    )


def stats(values):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]

    if x.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }

    return {
        "count":
            int(x.size),

        "mean":
            float(np.mean(x)),

        "median":
            float(np.median(x)),

        "p95":
            float(
                np.percentile(
                    x,
                    95,
                )
            ),

        "max":
            float(np.max(x)),
    }


def print_stats(title, data, unit):
    print(title)

    if data["count"] == 0:
        print("  no valid data")
        return

    print(
        f"  count  : {data['count']}"
    )
    print(
        f"  mean   : {data['mean']:.6f} {unit}"
    )
    print(
        f"  median : {data['median']:.6f} {unit}"
    )
    print(
        f"  P95    : {data['p95']:.6f} {unit}"
    )
    print(
        f"  max    : {data['max']:.6f} {unit}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate fisheye stereo extrinsics "
            "in spherical/ray space."
        )
    )

    parser.add_argument(
        "--session",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--cam0-calib",
        type=Path,
        default=Path(
            "calibration/results/cam0_fisheye.json"
        ),
    )

    parser.add_argument(
        "--cam1-calib",
        type=Path,
        default=Path(
            "calibration/results/cam1_fisheye.json"
        ),
    )

    parser.add_argument(
        "--stereo-calib",
        type=Path,
        default=Path(
            "calibration/results/stereo_fisheye.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "calibration/results/"
            "stereo_validation.json"
        ),
    )

    parser.add_argument(
        "--min-common-corners",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    K0, D0 = load_intrinsics(
        args.cam0_calib
    )

    K1, D1 = load_intrinsics(
        args.cam1_calib
    )

    stereo = load_json(
        args.stereo_calib
    )

    R = np.asarray(
        stereo["R_cam0_to_cam1"],
        dtype=np.float64,
    ).reshape(3, 3)

    T = np.asarray(
        stereo["T_cam0_to_cam1_mm"],
        dtype=np.float64,
    ).reshape(3, 1)

    baseline_mm = float(
        np.linalg.norm(T)
    )

    used_calibration_pairs = {
        item["pair"]
        for item in stereo.get(
            "used_pairs",
            [],
        )
    }

    detector = make_detector()

    cam0_dir = (
        args.session / "cam0"
    )

    cam1_dir = (
        args.session / "cam1"
    )

    files0 = sorted(
        cam0_dir.glob(
            "pair_*_cam0.png"
        )
    )

    all_angles_mrad = []
    all_angles_deg = []
    all_gaps_mm = []

    calibration_angles_mrad = []
    calibration_gaps_mm = []

    holdout_angles_mrad = []
    holdout_gaps_mm = []

    all_positive = 0
    all_depth_count = 0

    per_pair = []
    rejected = []

    for path0 in files0:

        pair_name = (
            path0.stem.removesuffix(
                "_cam0"
            )
        )

        path1 = (
            cam1_dir
            / f"{pair_name}_cam1.png"
        )

        if not path1.exists():
            rejected.append(
                {
                    "pair": pair_name,
                    "reason": "missing_cam1",
                }
            )
            continue

        img0 = cv2.imread(
            str(path0)
        )

        img1 = cv2.imread(
            str(path1)
        )

        if img0 is None or img1 is None:
            rejected.append(
                {
                    "pair": pair_name,
                    "reason":
                        "image_read_failed",
                }
            )
            continue

        c0, ids0 = detect(
            detector,
            img0,
        )

        c1, ids1 = detect(
            detector,
            img1,
        )

        if (
            ids0 is None
            or ids1 is None
        ):
            rejected.append(
                {
                    "pair": pair_name,
                    "reason":
                        "charuco_detection_failed",
                }
            )
            continue

        common = common_points(
            c0,
            ids0,
            c1,
            ids1,
        )

        if common is None:
            rejected.append(
                {
                    "pair": pair_name,
                    "reason":
                        "no_common_points",
                }
            )
            continue

        common_ids, pts0, pts1 = common

        if (
            len(common_ids)
            < args.min_common_corners
        ):
            rejected.append(
                {
                    "pair": pair_name,
                    "reason":
                        "too_few_common_points",
                    "common":
                        len(common_ids),
                }
            )
            continue

        rays0, finite0, norm0 = (
            pixels_to_unit_rays(
                pts0,
                K0,
                D0,
            )
        )

        rays1, finite1, norm1 = (
            pixels_to_unit_rays(
                pts1,
                K1,
                D1,
            )
        )

        # Normally every ChArUco point should
        # produce a finite normalized ray.
        # Reject a pair if this is not true,
        # instead of silently mismatching IDs.
        if (
            len(rays0)
            != len(common_ids)
            or len(rays1)
            != len(common_ids)
        ):
            rejected.append(
                {
                    "pair": pair_name,
                    "reason":
                        "non_finite_undistorted_ray",
                }
            )
            continue

        angle_rad, epi_valid = (
            epipolar_angular_errors(
                rays0,
                rays1,
                R,
                T,
            )
        )

        if not np.all(epi_valid):
            rays0_valid = rays0[
                epi_valid
            ]
            rays1_valid = rays1[
                epi_valid
            ]
        else:
            rays0_valid = rays0
            rays1_valid = rays1

        if len(angle_rad) == 0:
            rejected.append(
                {
                    "pair": pair_name,
                    "reason":
                        "degenerate_epipolar_plane",
                }
            )
            continue

        angles_mrad = (
            angle_rad * 1000.0
        )

        angles_deg = np.degrees(
            angle_rad
        )

        gaps_mm, depth0, depth1 = (
            triangulation_ray_gap(
                rays0_valid,
                rays1_valid,
                R,
                T,
            )
        )

        positive = (
            (depth0 > 0.0)
            & (depth1 > 0.0)
        )

        positive_count = int(
            np.count_nonzero(
                positive
            )
        )

        all_positive += (
            positive_count
        )

        all_depth_count += (
            len(depth0)
        )

        is_calibration_pair = (
            pair_name
            in used_calibration_pairs
        )

        split = (
            "calibration"
            if is_calibration_pair
            else "holdout"
        )

        all_angles_mrad.extend(
            angles_mrad.tolist()
        )

        all_angles_deg.extend(
            angles_deg.tolist()
        )

        all_gaps_mm.extend(
            gaps_mm.tolist()
        )

        if is_calibration_pair:
            calibration_angles_mrad.extend(
                angles_mrad.tolist()
            )

            calibration_gaps_mm.extend(
                gaps_mm.tolist()
            )

        else:
            holdout_angles_mrad.extend(
                angles_mrad.tolist()
            )

            holdout_gaps_mm.extend(
                gaps_mm.tolist()
            )

        pair_angle = stats(
            angles_mrad
        )

        pair_gap = stats(
            gaps_mm
        )

        per_pair.append(
            {
                "pair":
                    pair_name,

                "split":
                    split,

                "common_corners":
                    len(common_ids),

                "angular_error_mrad":
                    pair_angle,

                "ray_gap_mm":
                    pair_gap,

                "positive_depth_ratio":
                    positive_count
                    / len(depth0),
            }
        )

    optical_axis0 = np.asarray(
        [0.0, 0.0, 1.0],
        dtype=np.float64,
    )

    optical_axis0_in_cam1 = (
        R @ optical_axis0
    )

    optical_axis1 = np.asarray(
        [0.0, 0.0, 1.0],
        dtype=np.float64,
    )

    cos_angle = float(
        np.dot(
            optical_axis0_in_cam1,
            optical_axis1,
        )
        / (
            np.linalg.norm(
                optical_axis0_in_cam1
            )
            * np.linalg.norm(
                optical_axis1
            )
        )
    )

    cos_angle = np.clip(
        cos_angle,
        -1.0,
        1.0,
    )

    optical_axis_angle_deg = float(
        np.degrees(
            np.arccos(cos_angle)
        )
    )

    all_angle_stats = stats(
        all_angles_mrad
    )

    all_angle_deg_stats = stats(
        all_angles_deg
    )

    all_gap_stats = stats(
        all_gaps_mm
    )

    calib_angle_stats = stats(
        calibration_angles_mrad
    )

    calib_gap_stats = stats(
        calibration_gaps_mm
    )

    holdout_angle_stats = stats(
        holdout_angles_mrad
    )

    holdout_gap_stats = stats(
        holdout_gaps_mm
    )

    positive_depth_ratio = (
        all_positive
        / all_depth_count
        if all_depth_count
        else None
    )

    worst_pairs = sorted(
        per_pair,
        key=lambda x:
            x[
                "angular_error_mrad"
            ]["median"],
        reverse=True,
    )[:10]

    print()
    print(
        "=== Stereo physical geometry ==="
    )

    print(
        f"Baseline: "
        f"{baseline_mm:.3f} mm"
    )

    print(
        f"Optical-axis angle: "
        f"{optical_axis_angle_deg:.3f} deg"
    )

    print()
    print(
        "=== All common ChArUco points ==="
    )

    print_stats(
        "Angular epipolar error:",
        all_angle_stats,
        "mrad",
    )

    print()

    print_stats(
        "Closest-ray gap:",
        all_gap_stats,
        "mm",
    )

    if positive_depth_ratio is not None:
        print()
        print(
            "Positive-depth ratio: "
            f"{positive_depth_ratio:.6f}"
        )

    print()
    print(
        "=== Calibration-pair subset ==="
    )

    print_stats(
        "Angular epipolar error:",
        calib_angle_stats,
        "mrad",
    )

    print_stats(
        "Closest-ray gap:",
        calib_gap_stats,
        "mm",
    )

    print()
    print(
        "=== Holdout subset ==="
    )

    print(
        "(pairs not used by the "
        "25-point stereo calibration)"
    )

    print_stats(
        "Angular epipolar error:",
        holdout_angle_stats,
        "mrad",
    )

    print_stats(
        "Closest-ray gap:",
        holdout_gap_stats,
        "mm",
    )

    print()
    print(
        "=== Worst 10 pairs by "
        "median angular error ==="
    )

    for item in worst_pairs:
        print(
            f"{item['pair']:>10s}  "
            f"{item['split']:<11s} "
            f"n={item['common_corners']:2d}  "
            f"median="
            f"{item['angular_error_mrad']['median']:.4f} mrad  "
            f"P95="
            f"{item['angular_error_mrad']['p95']:.4f} mrad  "
            f"gap_med="
            f"{item['ray_gap_mm']['median']:.3f} mm"
        )

    result = {
        "opencv_version":
            cv2.__version__,

        "transform_convention":
            (
                "X_cam1 = "
                "R_cam0_to_cam1 * "
                "X_cam0 + "
                "T_cam0_to_cam1"
            ),

        "baseline_mm":
            baseline_mm,

        "optical_axis_angle_deg":
            optical_axis_angle_deg,

        "input_pairs":
            len(files0),

        "validated_pairs":
            len(per_pair),

        "rejected_pairs":
            rejected,

        "all_points": {
            "angular_error_mrad":
                all_angle_stats,

            "angular_error_deg":
                all_angle_deg_stats,

            "ray_gap_mm":
                all_gap_stats,

            "positive_depth_ratio":
                positive_depth_ratio,
        },

        "calibration_subset": {
            "angular_error_mrad":
                calib_angle_stats,

            "ray_gap_mm":
                calib_gap_stats,
        },

        "holdout_subset": {
            "note":
                (
                    "Pairs not used by the "
                    "fixed-25-point stereo "
                    "calibration. This is a "
                    "diagnostic holdout, not "
                    "a fully independent dataset."
                ),

            "angular_error_mrad":
                holdout_angle_stats,

            "ray_gap_mm":
                holdout_gap_stats,
        },

        "worst_10_pairs":
            worst_pairs,

        "per_pair":
            per_pair,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Saved:")
    print(args.output)


if __name__ == "__main__":
    main()
