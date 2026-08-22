import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def make_board(square_mm: float, marker_mm: float):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((8, 6), square_mm, marker_mm, dictionary)
    detector = cv2.aruco.CharucoDetector(board)
    return board, detector


def detect_dataset(input_dir: Path, board, detector, min_corners: int):
    records = []
    image_size = None

    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        h, w = img.shape[:2]
        if image_size is None:
            image_size = (w, h)
        elif image_size != (w, h):
            raise RuntimeError(
                f"Mixed image sizes: expected {image_size}, got {(w, h)} in {path.name}"
            )

        cc, ci, mc, mi = detector.detectBoard(img)
        if ci is None or cc is None or len(ci) < min_corners:
            continue

        # OpenCV 4.13: match detected ChArUco ids to board 3D points.
        obj_pts, img_pts = board.matchImagePoints(cc, ci)
        obj_pts = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 1, 3)
        img_pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 1, 2)

        pts2 = img_pts.reshape(-1, 2)
        centroid = pts2.mean(axis=0)
        center = np.array([w / 2.0, h / 2.0], dtype=np.float64)
        denom = np.hypot(w / 2.0, h / 2.0)
        max_radius = float(np.max(np.linalg.norm(pts2 - center, axis=1)) / denom)

        records.append(
            {
                "path": path,
                "object_points": obj_pts,
                "image_points": img_pts,
                "n_corners": len(ci),
                "centroid_x_norm": float(centroid[0] / w),
                "centroid_y_norm": float(centroid[1] / h),
                "max_radius_norm": max_radius,
            }
        )

    if image_size is None:
        raise RuntimeError(f"No readable images in {input_dir}")
    if len(records) < 15:
        raise RuntimeError(f"Only {len(records)} usable views; need more calibration images.")

    return records, image_size


def spatial_split(records, val_ratio=0.20, seed=20260819):
    """Choose validation views from each coarse 3x4 image region."""
    rng = np.random.default_rng(seed)
    groups = {}
    for i, r in enumerate(records):
        col = min(3, max(0, int(r["centroid_x_norm"] * 4)))
        row = min(2, max(0, int(r["centroid_y_norm"] * 3)))
        groups.setdefault((row, col), []).append(i)

    val_idx = set()
    for key, inds in groups.items():
        inds = list(inds)
        rng.shuffle(inds)
        n_val = max(1, int(round(len(inds) * val_ratio))) if len(inds) >= 2 else 0
        val_idx.update(inds[:n_val])

    # Bring total near requested ratio without destroying regional coverage.
    target = max(1, int(round(len(records) * val_ratio)))
    remaining = [i for i in range(len(records)) if i not in val_idx]
    rng.shuffle(remaining)
    while len(val_idx) < target and remaining:
        val_idx.add(remaining.pop())

    train = [r for i, r in enumerate(records) if i not in val_idx]
    val = [r for i, r in enumerate(records) if i in val_idx]
    return train, val


def point_errors(obs, pred):
    obs = np.asarray(obs, dtype=np.float64).reshape(-1, 2)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 2)
    return np.linalg.norm(obs - pred, axis=1)


def summarize_errors(all_errors, per_view_rmse):
    e = np.concatenate(all_errors) if all_errors else np.array([], dtype=np.float64)
    p = np.asarray(per_view_rmse, dtype=np.float64)
    if len(e) == 0:
        return {}
    return {
        "point_mean_px": float(np.mean(e)),
        "point_median_px": float(np.median(e)),
        "point_p95_px": float(np.percentile(e, 95)),
        "point_max_px": float(np.max(e)),
        "view_rmse_mean_px": float(np.mean(p)),
        "view_rmse_median_px": float(np.median(p)),
        "view_rmse_p95_px": float(np.percentile(p, 95)),
        "view_rmse_max_px": float(np.max(p)),
    }


def calibrate_pinhole(train, image_size):
    obj = [r["object_points"].astype(np.float32) for r in train]
    img = [r["image_points"].astype(np.float32) for r in train]

    flags = cv2.CALIB_RATIONAL_MODEL
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj, img, image_size, None, None, flags=flags
    )
    return rms, K, D, rvecs, tvecs


def calibrate_fisheye(train, image_size):
    obj = [r["object_points"].astype(np.float64) for r in train]
    img = [r["image_points"].astype(np.float64) for r in train]

    K = np.zeros((3, 3), dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_CHECK_COND
        | cv2.fisheye.CALIB_FIX_SKEW
    )
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        200,
        1e-9,
    )
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        obj, img, image_size, K, D, None, None, flags, criteria
    )
    return rms, K, D, rvecs, tvecs


def eval_train_pinhole(train, K, D, rvecs, tvecs):
    all_e, view_rmse = [], []
    for r, rv, tv in zip(train, rvecs, tvecs):
        pred, _ = cv2.projectPoints(r["object_points"], rv, tv, K, D)
        e = point_errors(r["image_points"], pred)
        all_e.append(e)
        view_rmse.append(float(np.sqrt(np.mean(e ** 2))))
    return summarize_errors(all_e, view_rmse)


def eval_train_fisheye(train, K, D, rvecs, tvecs):
    all_e, view_rmse = [], []
    for r, rv, tv in zip(train, rvecs, tvecs):
        pred, _ = cv2.fisheye.projectPoints(r["object_points"], rv, tv, K, D)
        e = point_errors(r["image_points"], pred)
        all_e.append(e)
        view_rmse.append(float(np.sqrt(np.mean(e ** 2))))
    return summarize_errors(all_e, view_rmse)


def eval_val_pinhole(val, K, D):
    all_e, view_rmse, failures = [], [], []
    for r in val:
        ok, rv, tv = cv2.solvePnP(
            r["object_points"], r["image_points"], K, D,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            failures.append(r["path"].name)
            continue
        pred, _ = cv2.projectPoints(r["object_points"], rv, tv, K, D)
        e = point_errors(r["image_points"], pred)
        all_e.append(e)
        view_rmse.append(float(np.sqrt(np.mean(e ** 2))))
    return summarize_errors(all_e, view_rmse), failures


def eval_val_fisheye(val, K, D):
    all_e, view_rmse, failures = [], [], []
    eye = np.eye(3, dtype=np.float64)
    zero = np.zeros((4, 1), dtype=np.float64)

    for r in val:
        # Convert fisheye-distorted pixels to normalized perspective coordinates.
        und = cv2.fisheye.undistortPoints(r["image_points"], K, D)
        ok, rv, tv = cv2.solvePnP(
            r["object_points"], und, eye, None,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            failures.append(r["path"].name)
            continue
        pred, _ = cv2.fisheye.projectPoints(r["object_points"], rv, tv, K, D)
        e = point_errors(r["image_points"], pred)
        all_e.append(e)
        view_rmse.append(float(np.sqrt(np.mean(e ** 2))))
    return summarize_errors(all_e, view_rmse), failures


def ray_from_pixel_pinhole(pixel, K, D):
    p = np.array(pixel, dtype=np.float64).reshape(1, 1, 2)
    und = cv2.undistortPoints(p, K, D).reshape(2)
    ray = np.array([und[0], und[1], 1.0], dtype=np.float64)
    ray /= np.linalg.norm(ray)
    return ray


def ray_from_pixel_fisheye(pixel, K, D):
    p = np.array(pixel, dtype=np.float64).reshape(1, 1, 2)
    und = cv2.fisheye.undistortPoints(p, K, D).reshape(2)
    ray = np.array([und[0], und[1], 1.0], dtype=np.float64)
    ray /= np.linalg.norm(ray)
    return ray


def angle_deg(a, b):
    c = float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))
    return math.degrees(math.acos(c))


def estimate_fov(image_size, K, D, model):
    w, h = image_size
    cx, cy = float(K[0, 2]), float(K[1, 2])
    ray_fn = ray_from_pixel_fisheye if model == "fisheye" else ray_from_pixel_pinhole

    # Use principal-point row/column, clipped to valid pixel coordinates.
    y = float(np.clip(cy, 0, h - 1))
    x = float(np.clip(cx, 0, w - 1))

    left = ray_fn((0.0, y), K, D)
    right = ray_fn((w - 1.0, y), K, D)
    top = ray_fn((x, 0.0), K, D)
    bottom = ray_fn((x, h - 1.0), K, D)
    optical = np.array([0.0, 0.0, 1.0])

    return {
        "HFOV_deg": float(angle_deg(left, right)),
        "VFOV_deg": float(angle_deg(top, bottom)),
        "left_half_deg": float(angle_deg(left, optical)),
        "right_half_deg": float(angle_deg(right, optical)),
        "top_half_deg": float(angle_deg(top, optical)),
        "bottom_half_deg": float(angle_deg(bottom, optical)),
        "note": (
            "Model-derived raw-image centerline FOV. For extreme >180-degree diagonal coverage, "
            "do not infer diagonal FOV from this rectilinear ray representation."
        ),
    }


def arr(x):
    return np.asarray(x).tolist()


def print_model(name, rms, K, D, train_metrics, val_metrics, fov):
    print("")
    print(f"=== {name} ===")
    print(f"Calibration RMS: {rms:.6f} px")
    print("K:")
    print(np.array2string(K, precision=6, suppress_small=False))
    print("D:")
    print(np.array2string(D.reshape(-1), precision=8, suppress_small=False))
    print("TRAIN:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.6f}")
    print("HELD-OUT VALIDATION:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.6f}")
    print("MODEL-DERIVED RAW FOV:")
    print(f"  HFOV: {fov['HFOV_deg']:.3f} deg "
          f"(left {fov['left_half_deg']:.3f}, right {fov['right_half_deg']:.3f})")
    print(f"  VFOV: {fov['VFOV_deg']:.3f} deg "
          f"(top {fov['top_half_deg']:.3f}, bottom {fov['bottom_half_deg']:.3f})")


def main():
    p = argparse.ArgumentParser(description="Compare pinhole-rational and fisheye ChArUco calibration")
    p.add_argument("--input", default="calibration/captures/cam0_intrinsic")
    p.add_argument("--output", default="calibration/results/cam0_model_compare")
    p.add_argument("--square-mm", type=float, default=30.0)
    p.add_argument("--marker-mm", type=float, default=22.0)
    p.add_argument("--min-corners", type=int, default=12)
    p.add_argument("--val-ratio", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=20260819)
    args = p.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    board, detector = make_board(args.square_mm, args.marker_mm)
    records, image_size = detect_dataset(
        input_dir, board, detector, args.min_corners
    )
    train, val = spatial_split(records, args.val_ratio, args.seed)

    print("=== Dataset ===")
    print(f"Usable views: {len(records)}")
    print(f"Train views: {len(train)}")
    print(f"Validation views: {len(val)}")
    print(f"Image size: {image_size[0]}x{image_size[1]}")
    print(f"Board: 8x6 squares, square={args.square_mm:.6f} mm, marker={args.marker_mm:.6f} mm")

    split = {
        "train": [r["path"].name for r in train],
        "validation": [r["path"].name for r in val],
    }
    (output_dir / "split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Pinhole + rational distortion model
    p_rms, p_K, p_D, p_rv, p_tv = calibrate_pinhole(train, image_size)
    p_train = eval_train_pinhole(train, p_K, p_D, p_rv, p_tv)
    p_val, p_fail = eval_val_pinhole(val, p_K, p_D)
    p_fov = estimate_fov(image_size, p_K, p_D, "pinhole")
    print_model("PINHOLE + RATIONAL DISTORTION", p_rms, p_K, p_D, p_train, p_val, p_fov)

    results = {
        "opencv_version": cv2.__version__,
        "image_size": list(image_size),
        "board": {
            "type": "ChArUco",
            "dictionary": "DICT_4X4_50",
            "squares_x": 8,
            "squares_y": 6,
            "square_mm": args.square_mm,
            "marker_mm": args.marker_mm,
        },
        "dataset": {
            "usable_views": len(records),
            "train_views": len(train),
            "validation_views": len(val),
            "min_corners": args.min_corners,
            "seed": args.seed,
        },
        "pinhole_rational": {
            "calibration_rms_px": float(p_rms),
            "K": arr(p_K),
            "D": arr(p_D.reshape(-1)),
            "train_metrics": p_train,
            "validation_metrics": p_val,
            "validation_pose_failures": p_fail,
            "raw_fov": p_fov,
        },
    }

    # Fisheye model
    try:
        f_rms, f_K, f_D, f_rv, f_tv = calibrate_fisheye(train, image_size)
        f_train = eval_train_fisheye(train, f_K, f_D, f_rv, f_tv)
        f_val, f_fail = eval_val_fisheye(val, f_K, f_D)
        f_fov = estimate_fov(image_size, f_K, f_D, "fisheye")
        print_model("OPENCV FISHEYE", f_rms, f_K, f_D, f_train, f_val, f_fov)

        results["fisheye"] = {
            "calibration_rms_px": float(f_rms),
            "K": arr(f_K),
            "D": arr(f_D.reshape(-1)),
            "train_metrics": f_train,
            "validation_metrics": f_val,
            "validation_pose_failures": f_fail,
            "raw_fov": f_fov,
        }
    except cv2.error as e:
        print("")
        print("=== OPENCV FISHEYE FAILED ===")
        print(str(e))
        results["fisheye"] = {"failed": True, "error": str(e)}

    result_path = output_dir / "model_compare.json"
    result_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("")
    print("Saved:")
    print(" ", output_dir / "split.json")
    print(" ", result_path)
    print("")
    print("IMPORTANT: Do not choose a model from calibration RMS alone.")
    print("Send model_compare.json or the full terminal output for interpretation.")


if __name__ == "__main__":
    main()
