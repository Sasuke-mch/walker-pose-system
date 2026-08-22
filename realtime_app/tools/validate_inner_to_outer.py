import argparse
import csv
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


def detect_view(path: Path, board, detector):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None

    cc, ci, _, _ = detector.detectBoard(img)
    if cc is None or ci is None or len(ci) < 8:
        return None

    obj, img_pts = board.matchImagePoints(cc, ci)
    obj = np.asarray(obj, dtype=np.float64).reshape(-1, 1, 3)
    img_pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 1, 2)
    ids = np.asarray(ci, dtype=np.int32).reshape(-1)

    h, w = img.shape[:2]
    return obj, img_pts, ids, w, h


def point_errors(obs, pred):
    obs = np.asarray(obs, dtype=np.float64).reshape(-1, 2)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 2)
    return np.linalg.norm(obs - pred, axis=1)


def solve_pose_pinhole(obj, img_pts, K, D):
    ok, rvec, tvec = cv2.solvePnP(
        obj,
        img_pts,
        K,
        D,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    return rvec, tvec


def solve_pose_fisheye(obj, img_pts, K, D):
    # Robust Python path for OpenCV 4.x:
    # 1) remove fisheye distortion to normalized coordinates
    # 2) solve PnP with identity intrinsics
    und = cv2.fisheye.undistortPoints(img_pts, K, D)
    I = np.eye(3, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        obj,
        und,
        I,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    return rvec, tvec


def project_pinhole(obj, rvec, tvec, K, D):
    pred, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    return pred


def project_fisheye(obj, rvec, tvec, K, D):
    pred, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, K, D)
    return pred


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0:
        return None
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def fmt(s):
    if s is None:
        return "n=0"
    return (
        f"n={s['n']:4d}  mean={s['mean']:.4f}  "
        f"median={s['median']:.4f}  p95={s['p95']:.4f}  max={s['max']:.4f}"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Strict inner-to-outer validation for pinhole-rational vs fisheye."
    )
    ap.add_argument("--input", default="calibration/captures/cam0_intrinsic")
    ap.add_argument("--compare-dir", default="calibration/results/cam0_model_compare")
    ap.add_argument("--output", default="calibration/reports/cam0_inner_outer")
    ap.add_argument(
        "--inner-radius",
        type=float,
        default=0.55,
        help="Use only points with normalized image radius <= this value to estimate pose.",
    )
    ap.add_argument(
        "--outer-radius",
        type=float,
        default=0.65,
        help="Evaluate only points with normalized image radius >= this value.",
    )
    ap.add_argument("--min-inner", type=int, default=6)
    ap.add_argument("--min-outer", type=int, default=3)
    args = ap.parse_args()

    input_dir = Path(args.input)
    compare_dir = Path(args.compare_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    compare = json.loads(
        (compare_dir / "model_compare.json").read_text(encoding="utf-8")
    )
    split = json.loads(
        (compare_dir / "split.json").read_text(encoding="utf-8")
    )

    board_cfg = compare["board"]
    board, detector = make_board(
        float(board_cfg["square_mm"]),
        float(board_cfg["marker_mm"]),
    )

    pK = np.asarray(compare["pinhole_rational"]["K"], dtype=np.float64)
    pD = np.asarray(compare["pinhole_rational"]["D"], dtype=np.float64).reshape(-1, 1)

    fish = compare.get("fisheye", {})
    if fish.get("failed"):
        raise RuntimeError("Previous fisheye calibration failed; cannot run comparison.")
    fK = np.asarray(fish["K"], dtype=np.float64)
    fD = np.asarray(fish["D"], dtype=np.float64).reshape(-1, 1)

    val_names = split["validation"]

    point_rows = []
    view_rows = []
    skipped = []

    for name in val_names:
        path = input_dir / name
        det = detect_view(path, board, detector)
        if det is None:
            skipped.append((name, "detection_failed"))
            continue

        obj, img_pts, ids, w, h = det
        pts = img_pts.reshape(-1, 2)

        cx_img, cy_img = w / 2.0, h / 2.0
        half_diag = math.hypot(cx_img, cy_img)
        radius = np.linalg.norm(
            pts - np.array([cx_img, cy_img], dtype=np.float64),
            axis=1,
        ) / half_diag

        inner_mask = radius <= args.inner_radius
        outer_mask = radius >= args.outer_radius

        n_inner = int(np.sum(inner_mask))
        n_outer = int(np.sum(outer_mask))

        if n_inner < args.min_inner:
            skipped.append((name, f"too_few_inner:{n_inner}"))
            continue
        if n_outer < args.min_outer:
            skipped.append((name, f"too_few_outer:{n_outer}"))
            continue

        obj_inner = obj[inner_mask]
        img_inner = img_pts[inner_mask]
        obj_outer = obj[outer_mask]
        img_outer = img_pts[outer_mask]
        ids_outer = ids[outer_mask]
        rad_outer = radius[outer_mask]

        pp = solve_pose_pinhole(obj_inner, img_inner, pK, pD)
        fp = solve_pose_fisheye(obj_inner, img_inner, fK, fD)
        if pp is None or fp is None:
            skipped.append((name, "pose_failed"))
            continue

        p_rv, p_tv = pp
        f_rv, f_tv = fp

        p_pred = project_pinhole(obj_outer, p_rv, p_tv, pK, pD)
        f_pred = project_fisheye(obj_outer, f_rv, f_tv, fK, fD)

        p_err = point_errors(img_outer, p_pred)
        f_err = point_errors(img_outer, f_pred)

        # Normalized horizontal/vertical distance from image center for diagnostics.
        outer_xy = img_outer.reshape(-1, 2)
        x_edge = np.abs(outer_xy[:, 0] - cx_img) / cx_img
        y_edge = np.abs(outer_xy[:, 1] - cy_img) / cy_img

        for i in range(n_outer):
            point_rows.append(
                {
                    "file": name,
                    "charuco_id": int(ids_outer[i]),
                    "radius_norm": float(rad_outer[i]),
                    "x_edge_norm": float(x_edge[i]),
                    "y_edge_norm": float(y_edge[i]),
                    "x_px": float(outer_xy[i, 0]),
                    "y_px": float(outer_xy[i, 1]),
                    "pinhole_error_px": float(p_err[i]),
                    "fisheye_error_px": float(f_err[i]),
                    "fisheye_minus_pinhole_px": float(f_err[i] - p_err[i]),
                }
            )

        p_s = stats(p_err)
        f_s = stats(f_err)
        view_rows.append(
            {
                "file": name,
                "inner_points_used_for_pose": n_inner,
                "outer_points_tested": n_outer,
                "outer_radius_min": float(np.min(rad_outer)),
                "outer_radius_max": float(np.max(rad_outer)),
                "pinhole_outer_median_px": p_s["median"],
                "pinhole_outer_p95_px": p_s["p95"],
                "fisheye_outer_median_px": f_s["median"],
                "fisheye_outer_p95_px": f_s["p95"],
                "median_winner": (
                    "pinhole"
                    if p_s["median"] < f_s["median"]
                    else "fisheye"
                ),
            }
        )

    if point_rows:
        points_csv = output_dir / "outer_prediction_points.csv"
        with points_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(point_rows[0].keys()))
            writer.writeheader()
            writer.writerows(point_rows)
    else:
        points_csv = output_dir / "outer_prediction_points.csv"

    if view_rows:
        views_csv = output_dir / "outer_prediction_views.csv"
        with views_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(view_rows[0].keys()))
            writer.writeheader()
            writer.writerows(view_rows)
    else:
        views_csv = output_dir / "outer_prediction_views.csv"

    p_all = stats([r["pinhole_error_px"] for r in point_rows])
    f_all = stats([r["fisheye_error_px"] for r in point_rows])

    very_outer = [r for r in point_rows if r["radius_norm"] >= 0.75]
    p_vo = stats([r["pinhole_error_px"] for r in very_outer])
    f_vo = stats([r["fisheye_error_px"] for r in very_outer])

    far_x = [r for r in point_rows if r["x_edge_norm"] >= 0.75]
    p_fx = stats([r["pinhole_error_px"] for r in far_x])
    f_fx = stats([r["fisheye_error_px"] for r in far_x])

    far_y = [r for r in point_rows if r["y_edge_norm"] >= 0.75]
    p_fy = stats([r["pinhole_error_px"] for r in far_y])
    f_fy = stats([r["fisheye_error_px"] for r in far_y])

    lines = []
    lines.append("=== Cam0 INNER -> OUTER prediction validation ===")
    lines.append(f"OpenCV: {cv2.__version__}")
    lines.append(f"Validation views available: {len(val_names)}")
    lines.append(
        f"Pose uses ONLY points with radius <= {args.inner_radius:.2f}; "
        f"evaluation uses ONLY points with radius >= {args.outer_radius:.2f}."
    )
    lines.append(
        f"Minimum per usable view: inner >= {args.min_inner}, outer >= {args.min_outer}"
    )
    lines.append("")
    lines.append(f"Usable mixed inner/outer views: {len(view_rows)}")
    lines.append(f"Outer points evaluated: {len(point_rows)}")
    lines.append(f"Skipped views: {len(skipped)}")
    for name, reason in skipped:
        lines.append(f"  {name}: {reason}")

    lines.append("")
    lines.append("ALL OUTER TEST POINTS")
    lines.append("Pinhole : " + fmt(p_all))
    lines.append("Fisheye : " + fmt(f_all))

    lines.append("")
    lines.append("VERY OUTER RADIAL POINTS (r >= 0.75)")
    lines.append("Pinhole : " + fmt(p_vo))
    lines.append("Fisheye : " + fmt(f_vo))

    lines.append("")
    lines.append("FAR HORIZONTAL POINTS (|x-center| / half-width >= 0.75)")
    lines.append("Pinhole : " + fmt(p_fx))
    lines.append("Fisheye : " + fmt(f_fx))

    lines.append("")
    lines.append("FAR VERTICAL POINTS (|y-center| / half-height >= 0.75)")
    lines.append("Pinhole : " + fmt(p_fy))
    lines.append("Fisheye : " + fmt(f_fy))

    if view_rows:
        p_wins = sum(r["median_winner"] == "pinhole" for r in view_rows)
        f_wins = sum(r["median_winner"] == "fisheye" for r in view_rows)
        lines.append("")
        lines.append("PER-VIEW MEDIAN WIN COUNT")
        lines.append(f"Pinhole wins: {p_wins}")
        lines.append(f"Fisheye wins: {f_wins}")

    lines.append("")
    lines.append("DECISION READINESS")
    if len(view_rows) < 5 or len(point_rows) < 30:
        lines.append(
            "NOT READY: existing held-out views do not contain enough images that span "
            "both inner and outer regions. Capture a dedicated validation set."
        )
    else:
        lines.append(
            "Enough mixed inner/outer data exists for a useful comparison; interpret "
            "outer median/P95 and the very-outer subsets."
        )

    if len(very_outer) < 20:
        lines.append(
            "WARNING: fewer than 20 points at r >= 0.75; extreme radial extrapolation "
            "is still weakly tested."
        )
    if len(far_x) < 20:
        lines.append(
            "WARNING: fewer than 20 far-horizontal points; HFOV model choice remains weakly tested."
        )
    if len(far_y) < 20:
        lines.append(
            "WARNING: fewer than 20 far-vertical points; VFOV model choice remains weakly tested."
        )

    lines.append("")
    lines.append(
        "Interpretation: this test is stricter than ordinary held-out reprojection error "
        "because outer points are NOT used to estimate that view's board pose."
    )

    summary = "\n".join(lines)
    summary_path = output_dir / "summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    print(summary)
    print("")
    print("Saved:")
    print(" ", summary_path)
    if point_rows:
        print(" ", points_csv)
    if view_rows:
        print(" ", views_csv)


if __name__ == "__main__":
    main()
