from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


BOARD_SQUARES = (8, 6)
SQUARE_MM = 30.0
MARKER_MM = 22.0
DICT_NAME = "DICT_4X4_50"


def load_intrinsics(path: Path):
    data = json.loads(
        path.read_text(encoding="utf-8-sig")
    )

    K = np.asarray(
        data["K"],
        dtype=np.float64,
    ).reshape(3, 3)

    D = np.asarray(
        data["D"],
        dtype=np.float64,
    ).reshape(4, 1)

    image_size = tuple(
        int(v) for v in data["image_size"]
    )

    if len(image_size) != 2:
        raise ValueError(
            f"Invalid image_size in {path}: "
            f"{data['image_size']!r}"
        )

    return data, K, D, image_size


def make_board():
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

    return board, detector


def detect_charuco(detector, image):
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

    if len(corners) != len(ids):
        raise RuntimeError(
            "ChArUco corner/ID count mismatch."
        )

    return corners, ids


def build_id_map(corners, ids):
    return {
        int(i): np.asarray(p, dtype=np.float64)
        for i, p in zip(ids, corners)
    }


def select_spread_ids(
    common_ids,
    board_points,
    target_count,
):
    """
    Select exactly target_count ChArUco IDs
    with broad spatial coverage.

    Uses farthest-point sampling in board XY.
    """

    ids = np.asarray(
        common_ids,
        dtype=np.int32,
    )

    if len(ids) < target_count:
        return None

    if len(ids) == target_count:
        return sorted(int(i) for i in ids)

    xy = np.asarray(
        [
            board_points[int(i), :2]
            for i in ids
        ],
        dtype=np.float64,
    )

    center = xy.mean(axis=0)

    first = int(
        np.argmax(
            np.linalg.norm(
                xy - center,
                axis=1,
            )
        )
    )

    selected_indices = [first]

    while len(selected_indices) < target_count:

        selected_xy = xy[selected_indices]

        min_dist = np.full(
            len(ids),
            np.inf,
            dtype=np.float64,
        )

        for p in selected_xy:
            d = np.linalg.norm(
                xy - p,
                axis=1,
            )

            min_dist = np.minimum(
                min_dist,
                d,
            )

        min_dist[selected_indices] = -1.0

        next_index = int(
            np.argmax(min_dist)
        )

        selected_indices.append(next_index)

    selected_ids = [
        int(ids[i])
        for i in selected_indices
    ]

    return sorted(selected_ids)


def collect_common_points(
    corners0,
    ids0,
    corners1,
    ids1,
    board_points,
    fixed_points_per_pair,
):

    map0 = build_id_map(
        corners0,
        ids0,
    )

    map1 = build_id_map(
        corners1,
        ids1,
    )

    common_ids = sorted(
        set(map0).intersection(map1)
    )

    if len(common_ids) < fixed_points_per_pair:
        return None

    selected_ids = select_spread_ids(
        common_ids,
        board_points,
        fixed_points_per_pair,
    )

    if selected_ids is None:
        return None

    selected_obj = np.asarray(
        [
            board_points[i]
            for i in selected_ids
        ],
        dtype=np.float64,
    ).reshape(-1, 1, 3)

    selected_img0 = np.asarray(
        [
            map0[i]
            for i in selected_ids
        ],
        dtype=np.float64,
    ).reshape(-1, 1, 2)

    selected_img1 = np.asarray(
        [
            map1[i]
            for i in selected_ids
        ],
        dtype=np.float64,
    ).reshape(-1, 1, 2)

    # All common points are retained for
    # later epipolar-error evaluation.
    all_img0 = np.asarray(
        [
            map0[i]
            for i in common_ids
        ],
        dtype=np.float64,
    ).reshape(-1, 1, 2)

    all_img1 = np.asarray(
        [
            map1[i]
            for i in common_ids
        ],
        dtype=np.float64,
    ).reshape(-1, 1, 2)

    return {
        "common_ids": common_ids,
        "selected_ids": selected_ids,
        "selected_obj": selected_obj,
        "selected_img0": selected_img0,
        "selected_img1": selected_img1,
        "all_img0": all_img0,
        "all_img1": all_img1,
    }


def as_dense_fisheye_array(
    items,
    n_views,
    n_points,
    channels,
):
    """
    Convert per-view point arrays into a
    fixed-size dense OpenCV fisheye array.

    Object points:
        (N, 1, P, 3)

    Image points:
        (N, 1, P, 2)
    """

    arr = np.asarray(
        [
            x.reshape(n_points, channels)
            for x in items
        ],
        dtype=np.float64,
    )

    arr = arr.reshape(
        n_views,
        1,
        n_points,
        channels,
    )

    return np.ascontiguousarray(arr)


def stats_dict(values):

    x = np.asarray(
        values,
        dtype=np.float64,
    )

    if x.size == 0:
        return {
            "mean_px": None,
            "median_px": None,
            "p95_px": None,
            "max_px": None,
        }

    return {
        "mean_px": float(np.mean(x)),
        "median_px": float(np.median(x)),
        "p95_px": float(
            np.percentile(x, 95)
        ),
        "max_px": float(np.max(x)),
    }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Stereo extrinsic calibration for "
            "two OpenCV-fisheye cameras using "
            "paired ChArUco images."
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
            "calibration/results/"
            "cam0_fisheye.json"
        ),
    )

    parser.add_argument(
        "--cam1-calib",
        type=Path,
        default=Path(
            "calibration/results/"
            "cam1_fisheye.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "calibration/results/"
            "stereo_fisheye.json"
        ),
    )

    parser.add_argument(
        "--min-common-corners",
        type=int,
        default=12,
        help=(
            "General quality threshold. "
            "Effective threshold is max("
            "min-common-corners, "
            "fixed-points-per-pair)."
        ),
    )

    parser.add_argument(
        "--fixed-points-per-pair",
        type=int,
        default=25,
        help=(
            "Use exactly this many spatially "
            "spread common ChArUco points "
            "from each stereo pair."
        ),
    )

    args = parser.parse_args()

    if args.min_common_corners < 4:
        raise ValueError(
            "--min-common-corners must be >= 4"
        )

    if args.fixed_points_per_pair < 4:
        raise ValueError(
            "--fixed-points-per-pair must be >= 4"
        )

    # 8x6 ChArUco board has:
    # (8-1)*(6-1) = 35 ChArUco corners.
    if args.fixed_points_per_pair > 35:
        raise ValueError(
            "--fixed-points-per-pair cannot "
            "exceed 35."
        )

    required_common = max(
        args.min_common_corners,
        args.fixed_points_per_pair,
    )

    _, K0, D0, size0 = load_intrinsics(
        args.cam0_calib
    )

    _, K1, D1, size1 = load_intrinsics(
        args.cam1_calib
    )

    if size0 != size1:
        raise RuntimeError(
            f"CAM0/CAM1 image sizes differ: "
            f"{size0} vs {size1}"
        )

    image_size = size0

    board, detector = make_board()

    board_points = np.asarray(
        board.getChessboardCorners(),
        dtype=np.float64,
    ).reshape(-1, 3)

    cam0_dir = args.session / "cam0"
    cam1_dir = args.session / "cam1"

    files0 = sorted(
        cam0_dir.glob(
            "pair_*_cam0.png"
        )
    )

    if not files0:
        raise RuntimeError(
            f"No CAM0 images found in: "
            f"{cam0_dir}"
        )

    selected_object_points = []
    selected_image_points0 = []
    selected_image_points1 = []

    validation_image_points0 = []
    validation_image_points1 = []

    used_pairs = []
    rejected_pairs = []

    for path0 in files0:

        pair_name = path0.stem.removesuffix(
            "_cam0"
        )

        path1 = (
            cam1_dir
            / f"{pair_name}_cam1.png"
        )

        if not path1.exists():
            rejected_pairs.append(
                {
                    "pair": pair_name,
                    "reason": "missing_cam1",
                }
            )
            continue

        img0 = cv2.imread(
            str(path0),
            cv2.IMREAD_COLOR,
        )

        img1 = cv2.imread(
            str(path1),
            cv2.IMREAD_COLOR,
        )

        if img0 is None or img1 is None:
            rejected_pairs.append(
                {
                    "pair": pair_name,
                    "reason":
                        "image_read_failed",
                }
            )
            continue

        size_img0 = (
            img0.shape[1],
            img0.shape[0],
        )

        size_img1 = (
            img1.shape[1],
            img1.shape[0],
        )

        if (
            size_img0 != image_size
            or size_img1 != image_size
        ):
            raise RuntimeError(
                f"{pair_name}: expected "
                f"{image_size}, got "
                f"CAM0={size_img0}, "
                f"CAM1={size_img1}"
            )

        corners0, ids0 = detect_charuco(
            detector,
            img0,
        )

        corners1, ids1 = detect_charuco(
            detector,
            img1,
        )

        if ids0 is None or ids1 is None:
            rejected_pairs.append(
                {
                    "pair": pair_name,
                    "reason":
                        "charuco_detection_failed",
                }
            )
            continue

        map0 = build_id_map(
            corners0,
            ids0,
        )

        map1 = build_id_map(
            corners1,
            ids1,
        )

        common_count = len(
            set(map0).intersection(map1)
        )

        if common_count < required_common:
            rejected_pairs.append(
                {
                    "pair": pair_name,
                    "reason":
                        "too_few_common_corners",
                    "common_corners":
                        common_count,
                    "required_common_corners":
                        required_common,
                }
            )
            continue

        match = collect_common_points(
            corners0,
            ids0,
            corners1,
            ids1,
            board_points,
            args.fixed_points_per_pair,
        )

        if match is None:
            rejected_pairs.append(
                {
                    "pair": pair_name,
                    "reason":
                        "fixed_point_selection_failed",
                    "common_corners":
                        common_count,
                }
            )
            continue

        selected_object_points.append(
            match["selected_obj"]
        )

        selected_image_points0.append(
            match["selected_img0"]
        )

        selected_image_points1.append(
            match["selected_img1"]
        )

        validation_image_points0.append(
            match["all_img0"]
        )

        validation_image_points1.append(
            match["all_img1"]
        )

        used_pairs.append(
            {
                "pair": pair_name,
                "common_corners":
                    len(
                        match["common_ids"]
                    ),
                "used_corners":
                    len(
                        match["selected_ids"]
                    ),
                "selected_ids":
                    match["selected_ids"],
            }
        )

    print()
    print("=== Stereo dataset ===")
    print(
        f"Input pairs: "
        f"{len(files0)}"
    )
    print(
        f"Usable pairs: "
        f"{len(used_pairs)}"
    )
    print(
        f"Rejected pairs: "
        f"{len(rejected_pairs)}"
    )
    print(
        f"Required common corners: "
        f"{required_common}"
    )
    print(
        f"Fixed points per usable pair: "
        f"{args.fixed_points_per_pair}"
    )

    if len(used_pairs) < 15:
        raise RuntimeError(
            f"Only {len(used_pairs)} "
            f"usable pairs remain; "
            f"need at least 15."
        )

    n_views = len(used_pairs)
    n_points = (
        args.fixed_points_per_pair
    )

    object_points_dense = (
        as_dense_fisheye_array(
            selected_object_points,
            n_views,
            n_points,
            3,
        )
    )

    image_points0_dense = (
        as_dense_fisheye_array(
            selected_image_points0,
            n_views,
            n_points,
            2,
        )
    )

    image_points1_dense = (
        as_dense_fisheye_array(
            selected_image_points1,
            n_views,
            n_points,
            2,
        )
    )

    print()
    print(
        "=== OpenCV input arrays ==="
    )

    print(
        f"object_points: "
        f"{object_points_dense.shape} "
        f"{object_points_dense.dtype}"
    )

    print(
        f"cam0_points  : "
        f"{image_points0_dense.shape} "
        f"{image_points0_dense.dtype}"
    )

    print(
        f"cam1_points  : "
        f"{image_points1_dense.shape} "
        f"{image_points1_dense.dtype}"
    )

    K0_work = K0.copy()
    D0_work = D0.copy()

    K1_work = K1.copy()
    D1_work = D1.copy()

    flags = (
        cv2.fisheye.CALIB_FIX_INTRINSIC
        | cv2.fisheye.CALIB_CHECK_COND
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        200,
        1e-9,
    )

    result = (
        cv2.fisheye.stereoCalibrate(
            object_points_dense,
            image_points0_dense,
            image_points1_dense,
            K0_work,
            D0_work,
            K1_work,
            D1_work,
            image_size,
            flags=flags,
            criteria=criteria,
        )
    )

    rms = float(result[0])

    K0_out = np.asarray(
        result[1],
        dtype=np.float64,
    )

    D0_out = np.asarray(
        result[2],
        dtype=np.float64,
    )

    K1_out = np.asarray(
        result[3],
        dtype=np.float64,
    )

    D1_out = np.asarray(
        result[4],
        dtype=np.float64,
    )

    R = np.asarray(
        result[5],
        dtype=np.float64,
    ).reshape(3, 3)

    T = np.asarray(
        result[6],
        dtype=np.float64,
    ).reshape(3, 1)

    baseline_mm = float(
        np.linalg.norm(T)
    )

    det_R = float(
        np.linalg.det(R)
    )

    orthogonality_error = float(
        np.linalg.norm(
            R.T @ R - np.eye(3),
            ord="fro",
        )
    )

    rvec, _ = cv2.Rodrigues(R)

    rotation_angle_deg = float(
        np.linalg.norm(rvec)
        * 180.0
        / np.pi
    )

    intrinsic_change = {
        "K0_max_abs":
            float(
                np.max(
                    np.abs(
                        K0_out - K0
                    )
                )
            ),
        "D0_max_abs":
            float(
                np.max(
                    np.abs(
                        D0_out - D0
                    )
                )
            ),
        "K1_max_abs":
            float(
                np.max(
                    np.abs(
                        K1_out - K1
                    )
                )
            ),
        "D1_max_abs":
            float(
                np.max(
                    np.abs(
                        D1_out - D1
                    )
                )
            ),
    }

    print()
    print(
        "=== Stereo calibration ==="
    )

    print(
        f"RMS: {rms:.6f} px"
    )

    print()
    print(
        "R (CAM0 -> CAM1):"
    )

    print(
        np.array2string(
            R,
            precision=9,
            suppress_small=False,
        )
    )

    print()
    print(
        "T (CAM0 -> CAM1) [mm]:"
    )

    print(
        np.array2string(
            T.reshape(-1),
            precision=6,
            suppress_small=False,
        )
    )

    print(
        f"Baseline: "
        f"{baseline_mm:.3f} mm"
    )

    print(
        f"Rotation magnitude: "
        f"{rotation_angle_deg:.3f} deg"
    )

    print(
        f"det(R): "
        f"{det_R:.9f}"
    )

    print(
        f"R orthogonality error: "
        f"{orthogonality_error:.3e}"
    )

    print(
        "Max intrinsic change under "
        "CALIB_FIX_INTRINSIC: "
        f"{max(intrinsic_change.values()):.3e}"
    )

    (
        R0_rect,
        R1_rect,
        P0,
        P1,
        Q,
    ) = cv2.fisheye.stereoRectify(
        K0,
        D0,
        K1,
        D1,
        image_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        newImageSize=image_size,
        balance=0.0,
        fov_scale=1.0,
    )

    all_vertical_errors = []
    per_pair_metrics = []

    for info, pts0, pts1 in zip(
        used_pairs,
        validation_image_points0,
        validation_image_points1,
    ):

        rect0 = (
            cv2.fisheye.undistortPoints(
                pts0,
                K0,
                D0,
                R=R0_rect,
                P=P0,
            )
            .reshape(-1, 2)
        )

        rect1 = (
            cv2.fisheye.undistortPoints(
                pts1,
                K1,
                D1,
                R=R1_rect,
                P=P1,
            )
            .reshape(-1, 2)
        )

        dy = np.abs(
            rect0[:, 1]
            - rect1[:, 1]
        )

        all_vertical_errors.extend(
            dy.tolist()
        )

        pair_stats = stats_dict(dy)

        pair_stats.update(
            {
                "pair":
                    info["pair"],
                "common_corners":
                    info[
                        "common_corners"
                    ],
            }
        )

        per_pair_metrics.append(
            pair_stats
        )

    epipolar = stats_dict(
        all_vertical_errors
    )

    print()
    print(
        "=== Rectified epipolar "
        "vertical error ==="
    )

    print(
        f"mean   : "
        f"{epipolar['mean_px']:.6f} px"
    )

    print(
        f"median : "
        f"{epipolar['median_px']:.6f} px"
    )

    print(
        f"P95    : "
        f"{epipolar['p95_px']:.6f} px"
    )

    print(
        f"max    : "
        f"{epipolar['max_px']:.6f} px"
    )

    output = {
        "opencv_version":
            cv2.__version__,

        "model":
            "opencv_fisheye_stereo",

        "transform_convention":
            (
                "X_cam1 = "
                "R_cam0_to_cam1 * "
                "X_cam0 + "
                "T_cam0_to_cam1"
            ),

        "board_units":
            "mm",

        "image_size":
            list(image_size),

        "board": {
            "type":
                "ChArUco",
            "dictionary":
                DICT_NAME,
            "squares_x":
                BOARD_SQUARES[0],
            "squares_y":
                BOARD_SQUARES[1],
            "square_mm":
                SQUARE_MM,
            "marker_mm":
                MARKER_MM,
            "charuco_corner_count":
                int(
                    len(board_points)
                ),
        },

        "cam0_intrinsics":
            str(args.cam0_calib),

        "cam1_intrinsics":
            str(args.cam1_calib),

        "session":
            str(args.session),

        "input_pairs":
            len(files0),

        "usable_pairs":
            len(used_pairs),

        "rejected_pair_count":
            len(rejected_pairs),

        "required_common_corners":
            required_common,

        "fixed_points_per_pair":
            n_points,

        "opencv_input_shapes": {
            "object_points":
                list(
                    object_points_dense.shape
                ),
            "cam0_points":
                list(
                    image_points0_dense.shape
                ),
            "cam1_points":
                list(
                    image_points1_dense.shape
                ),
        },

        "stereo_rms_px":
            rms,

        "R_cam0_to_cam1":
            R.tolist(),

        "T_cam0_to_cam1_mm":
            T.reshape(-1).tolist(),

        "baseline_mm":
            baseline_mm,

        "rotation_angle_deg":
            rotation_angle_deg,

        "R_determinant":
            det_R,

        "R_orthogonality_error":
            orthogonality_error,

        "intrinsic_change_under_fix_intrinsic":
            intrinsic_change,

        "R0_rectification":
            np.asarray(
                R0_rect
            ).tolist(),

        "R1_rectification":
            np.asarray(
                R1_rect
            ).tolist(),

        "P0":
            np.asarray(P0).tolist(),

        "P1":
            np.asarray(P1).tolist(),

        "Q":
            np.asarray(Q).tolist(),

        "epipolar_vertical_error_all_common_points":
            epipolar,

        "used_pairs":
            used_pairs,

        "rejected_pairs":
            rejected_pairs,

        "per_pair_epipolar_metrics":
            per_pair_metrics,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            output,
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
