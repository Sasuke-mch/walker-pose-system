from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_intrinsics(path: Path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))

    K = np.asarray(data["K"], dtype=np.float64)
    D = np.asarray(data["D"], dtype=np.float64).reshape(4, 1)

    size = tuple(int(x) for x in data["image_size"])

    return data, K, D, size


def make_board():
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50
    )

    board = cv2.aruco.CharucoBoard(
        (8, 6),
        30.0,
        22.0,
        dictionary,
    )

    detector = cv2.aruco.CharucoDetector(board)

    return board, detector


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


def common_points(
    corners0,
    ids0,
    corners1,
    ids1,
    board_points,
):
    map0 = {
        int(i): p
        for i, p in zip(ids0, corners0)
    }

    map1 = {
        int(i): p
        for i, p in zip(ids1, corners1)
    }

    common_ids = sorted(
        set(map0.keys()).intersection(map1.keys())
    )

    if not common_ids:
        return None

    obj = np.asarray(
        [board_points[i] for i in common_ids],
        dtype=np.float64,
    ).reshape(-1, 1, 3)

    img0 = np.asarray(
        [map0[i] for i in common_ids],
        dtype=np.float64,
    ).reshape(-1, 1, 2)

    img1 = np.asarray(
        [map1[i] for i in common_ids],
        dtype=np.float64,
    ).reshape(-1, 1, 2)

    return common_ids, obj, img0, img1


def percentile(values, p):
    if not values:
        return None

    return float(
        np.percentile(
            np.asarray(values, dtype=np.float64),
            p,
        )
    )


def main():
    parser = argparse.ArgumentParser()

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
        "--output",
        type=Path,
        default=Path(
            "calibration/results/stereo_fisheye.json"
        ),
    )

    parser.add_argument(
        "--min-common-corners",
        type=int,
        default=12,
    )

    args = parser.parse_args()

    cam0_data, K0, D0, size0 = load_intrinsics(
        args.cam0_calib
    )

    cam1_data, K1, D1, size1 = load_intrinsics(
        args.cam1_calib
    )

    if size0 != size1:
        raise RuntimeError(
            f"Image sizes differ: {size0} vs {size1}"
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
        cam0_dir.glob("pair_*_cam0.png")
    )

    if not files0:
        raise RuntimeError(
            f"No cam0 images found in {cam0_dir}"
        )

    object_points = []
    image_points0 = []
    image_points1 = []

    used_pairs = []
    rejected_pairs = []

    for path0 in files0:
        pair_name = path0.stem.replace(
            "_cam0",
            "",
        )

        path1 = (
            cam1_dir /
            f"{pair_name}_cam1.png"
        )

        if not path1.exists():
            rejected_pairs.append(
                {
                    "pair": pair_name,
                    "reason": "missing_cam1",
                }
            )
            continue

        img0 = cv2.imread(str(path0))
        img1 = cv2.imread(str(path1))

        if img0 is None or img1 is None:
            rejected_pairs.append(
                {
                    "pair": pair_name,
                    "reason": "read_failed",
                }
            )
            continue

        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]

        if (w0, h0) != image_size:
            raise RuntimeError(
                f"{path0.name} has wrong size"
            )

        if (w1, h1) != image_size:
            raise RuntimeError(
                f"{path1.name} has wrong size"
            )

        corners0, ids0 = detect(
            detector,
            img0,
        )

        corners1, ids1 = detect(
            detector,
            img1,
        )

        if ids0 is None or ids1 is None:
            rejected_pairs.append(
                {
                    "pair": pair_name,
                    "reason": "detection_failed",
                }
            )
            continue

        common = common_points(
            corners0,
            ids0,
            corners1,
            ids1,
            board_points,
        )

        if common is None:
            rejected_pairs.append(
                {
                    "pair": pair_name,
                    "reason": "no_common_ids",
                }
            )
            continue

        common_ids, obj, pts0, pts1 = common

        if len(common_ids) < args.min_common_corners:
            rejected_pairs.append(
                {
                    "pair": pair_name,
                    "reason": "too_few_common_corners",
                    "common": len(common_ids),
                }
            )
            continue

        object_points.append(obj)
        image_points0.append(pts0)
        image_points1.append(pts1)

        used_pairs.append(
            {
                "pair": pair_name,
                "common_corners": len(common_ids),
            }
        )

    print("")
    print("=== Stereo dataset ===")
    print(f"Input pairs: {len(files0)}")
    print(f"Usable pairs: {len(used_pairs)}")
    print(f"Rejected pairs: {len(rejected_pairs)}")

    if len(used_pairs) < 15:
        raise RuntimeError(
            "Too few usable stereo pairs."
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

    result = cv2.fisheye.stereoCalibrate(
        object_points,
        image_points0,
        image_points1,
        K0_work,
        D0_work,
        K1_work,
        D1_work,
        image_size,
        flags=flags,
        criteria=criteria,
    )

    rms = float(result[0])

    R = np.asarray(
        result[5],
        dtype=np.float64,
    )

    T = np.asarray(
        result[6],
        dtype=np.float64,
    ).reshape(3, 1)

    baseline_mm = float(
        np.linalg.norm(T)
    )

    print("")
    print("=== Stereo calibration ===")
    print(f"RMS: {rms:.6f} px")

    print("")
    print("R:")
    print(
        np.array2string(
            R,
            precision=9,
            suppress_small=False,
        )
    )

    print("")
    print("T [mm]:")
    print(
        np.array2string(
            T.reshape(-1),
            precision=6,
            suppress_small=False,
        )
    )

    print(
        f"Baseline: {baseline_mm:.3f} mm"
    )

    R0, R1, P0, P1, Q = (
        cv2.fisheye.stereoRectify(
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
    )

    all_vertical_errors = []
    pair_metrics = []

    for info, pts0, pts1 in zip(
        used_pairs,
        image_points0,
        image_points1,
    ):
        rect0 = cv2.fisheye.undistortPoints(
            pts0,
            K0,
            D0,
            R=R0,
            P=P0,
        ).reshape(-1, 2)

        rect1 = cv2.fisheye.undistortPoints(
            pts1,
            K1,
            D1,
            R=R1,
            P=P1,
        ).reshape(-1, 2)

        dy = np.abs(
            rect0[:, 1] - rect1[:, 1]
        )

        all_vertical_errors.extend(
            dy.tolist()
        )

        pair_metrics.append(
            {
                "pair": info["pair"],
                "common_corners": info[
                    "common_corners"
                ],
                "vertical_error_mean_px":
                    float(np.mean(dy)),
                "vertical_error_median_px":
                    float(np.median(dy)),
                "vertical_error_p95_px":
                    float(np.percentile(dy, 95)),
                "vertical_error_max_px":
                    float(np.max(dy)),
            }
        )

    epi = {
        "mean_px": float(
            np.mean(all_vertical_errors)
        ),
        "median_px": float(
            np.median(all_vertical_errors)
        ),
        "p95_px": percentile(
            all_vertical_errors,
            95,
        ),
        "max_px": float(
            np.max(all_vertical_errors)
        ),
    }

    print("")
    print("=== Rectified epipolar vertical error ===")
    print(
        f"mean   : {epi['mean_px']:.6f} px"
    )
    print(
        f"median : {epi['median_px']:.6f} px"
    )
    print(
        f"P95    : {epi['p95_px']:.6f} px"
    )
    print(
        f"max    : {epi['max_px']:.6f} px"
    )

    output = {
        "model": "opencv_fisheye_stereo",
        "image_size": list(image_size),

        "cam0_intrinsics":
            str(args.cam0_calib),

        "cam1_intrinsics":
            str(args.cam1_calib),

        "session":
            str(args.session),

        "usable_pairs":
            len(used_pairs),

        "rejected_pairs":
            rejected_pairs,

        "stereo_rms_px":
            rms,

        "R_cam0_to_cam1":
            R.tolist(),

        "T_cam0_to_cam1_mm":
            T.reshape(-1).tolist(),

        "baseline_mm":
            baseline_mm,

        "R0_rectification":
            R0.tolist(),

        "R1_rectification":
            R1.tolist(),

        "P0":
            P0.tolist(),

        "P1":
            P1.tolist(),

        "Q":
            Q.tolist(),

        "epipolar_vertical_error":
            epi,

        "per_pair_epipolar_metrics":
            pair_metrics,
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

    print("")
    print("Saved:")
    print(args.output)


if __name__ == "__main__":
    main()