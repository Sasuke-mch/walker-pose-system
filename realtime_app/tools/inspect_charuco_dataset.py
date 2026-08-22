import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def make_board():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(
        (8, 6),      # squaresX, squaresY
        30.0,        # square length (relative units are enough for detection)
        22.0,        # marker length
        dictionary,
    )
    detector = cv2.aruco.CharucoDetector(board)
    return board, detector


def laplacian_sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def normalized_radius(points, width, height):
    """Maximum detected-corner radius from image center, normalized so corners are ~1."""
    if points is None or len(points) == 0:
        return 0.0
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    cx, cy = width / 2.0, height / 2.0
    denom = math.hypot(cx, cy)
    r = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    return float(r.max() / denom) if denom > 0 else 0.0


def bbox_area_ratio(points, width, height):
    if points is None or len(points) == 0:
        return 0.0
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    area = max(0.0, float(x1 - x0)) * max(0.0, float(y1 - y0))
    return area / float(width * height)


def centroid(points):
    if points is None or len(points) == 0:
        return None
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    c = pts.mean(axis=0)
    return float(c[0]), float(c[1])


def classify(corners, sharpness):
    # Conservative first-pass rules; final keep/reject decision is made after geometry review.
    if corners < 8:
        return "REJECT_TOO_FEW_CORNERS"
    if corners < 12:
        return "WEAK_FEW_CORNERS"
    if sharpness < 60:
        return "WEAK_BLUR"
    return "OK"


def main():
    p = argparse.ArgumentParser(description="Inspect ChArUco intrinsic-calibration dataset")
    p.add_argument(
        "--input",
        default="calibration/captures/cam0_intrinsic",
        help="Folder containing calibration images",
    )
    p.add_argument(
        "--output",
        default="calibration/reports/cam0_intrinsic_inspection",
        help="Folder for CSV, overlays and summary",
    )
    p.add_argument("--save-overlays", action="store_true", help="Save detection overlay images")
    args = p.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    overlay_dir = output_dir / "overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS])
    if not files:
        raise SystemExit(f"No images found in: {input_dir}")

    board, detector = make_board()

    rows = []
    grid_rows, grid_cols = 3, 4
    dataset_grid = np.zeros((grid_rows, grid_cols), dtype=np.int32)
    detected_corner_grid = np.zeros((grid_rows, grid_cols), dtype=np.int32)

    width0 = height0 = None
    shape_mismatch = []

    for idx, path in enumerate(files, start=1):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            rows.append({
                "file": path.name,
                "width": "",
                "height": "",
                "markers": 0,
                "charuco_corners": 0,
                "sharpness": 0.0,
                "bbox_area_ratio": 0.0,
                "max_radius_norm": 0.0,
                "centroid_x_norm": "",
                "centroid_y_norm": "",
                "status": "REJECT_READ_FAILED",
            })
            continue

        h, w = img.shape[:2]
        if width0 is None:
            width0, height0 = w, h
        elif (w, h) != (width0, height0):
            shape_mismatch.append((path.name, w, h))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharp = laplacian_sharpness(gray)

        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(img)

        n_markers = 0 if marker_ids is None else len(marker_ids)
        n_corners = 0 if charuco_ids is None else len(charuco_ids)

        area_ratio = bbox_area_ratio(charuco_corners, w, h)
        radius_norm = normalized_radius(charuco_corners, w, h)
        c = centroid(charuco_corners)

        cxn = cyn = ""
        if c is not None:
            cxn = c[0] / w
            cyn = c[1] / h
            gc = min(grid_cols - 1, max(0, int(cxn * grid_cols)))
            gr = min(grid_rows - 1, max(0, int(cyn * grid_rows)))
            dataset_grid[gr, gc] += 1

            pts = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2)
            for px, py in pts:
                gc2 = min(grid_cols - 1, max(0, int((px / w) * grid_cols)))
                gr2 = min(grid_rows - 1, max(0, int((py / h) * grid_rows)))
                detected_corner_grid[gr2, gc2] += 1

        status = classify(n_corners, sharp)

        rows.append({
            "file": path.name,
            "width": w,
            "height": h,
            "markers": n_markers,
            "charuco_corners": n_corners,
            "sharpness": round(sharp, 2),
            "bbox_area_ratio": round(area_ratio, 4),
            "max_radius_norm": round(radius_norm, 4),
            "centroid_x_norm": "" if cxn == "" else round(cxn, 4),
            "centroid_y_norm": "" if cyn == "" else round(cyn, 4),
            "status": status,
        })

        if args.save_overlays:
            vis = img.copy()
            if marker_ids is not None and len(marker_ids):
                cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
            if charuco_ids is not None and len(charuco_ids):
                cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
            cv2.putText(
                vis,
                f"corners={n_corners} sharp={sharp:.1f} status={status}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0) if status == "OK" else (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(overlay_dir / path.name), vis)

    csv_path = output_dir / "inspection.csv"
    fields = [
        "file", "width", "height", "markers", "charuco_corners", "sharpness",
        "bbox_area_ratio", "max_radius_norm", "centroid_x_norm", "centroid_y_norm", "status"
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    readable = [r for r in rows if isinstance(r["charuco_corners"], int)]
    n_total = len(rows)
    n_ok = sum(r["status"] == "OK" for r in rows)
    n_weak = sum(str(r["status"]).startswith("WEAK") for r in rows)
    n_reject = sum(str(r["status"]).startswith("REJECT") for r in rows)

    corner_counts = np.array([r["charuco_corners"] for r in rows], dtype=float)
    sharpnesses = np.array([float(r["sharpness"]) for r in rows], dtype=float)
    radii = np.array([float(r["max_radius_norm"]) for r in rows], dtype=float)

    summary_lines = []
    summary_lines.append("=== ChArUco Dataset Inspection ===")
    summary_lines.append(f"Input: {input_dir}")
    summary_lines.append(f"Images: {n_total}")
    summary_lines.append(f"Reference image size: {width0}x{height0}")
    summary_lines.append(f"OK / WEAK / REJECT: {n_ok} / {n_weak} / {n_reject}")
    summary_lines.append("")
    summary_lines.append(
        f"ChArUco corners: median={np.median(corner_counts):.1f}, "
        f"min={corner_counts.min():.0f}, max={corner_counts.max():.0f}"
    )
    summary_lines.append(
        f"Sharpness: median={np.median(sharpnesses):.1f}, "
        f"min={sharpnesses.min():.1f}, max={sharpnesses.max():.1f}"
    )
    summary_lines.append(
        f"Max normalized radius: median={np.median(radii):.3f}, max={radii.max():.3f}"
    )
    summary_lines.append("")
    summary_lines.append("Board-centroid coverage grid (3 rows x 4 cols):")
    for row in dataset_grid:
        summary_lines.append("  " + " ".join(f"{int(v):3d}" for v in row))
    summary_lines.append("")
    summary_lines.append("Detected-corner coverage grid (3 rows x 4 cols):")
    for row in detected_corner_grid:
        summary_lines.append("  " + " ".join(f"{int(v):4d}" for v in row))

    empty_centroid_cells = np.argwhere(dataset_grid == 0)
    summary_lines.append("")
    if len(empty_centroid_cells):
        cells = [f"(row={r+1}, col={c+1})" for r, c in empty_centroid_cells]
        summary_lines.append("WARNING: no board centroid samples in grid cells: " + ", ".join(cells))
    else:
        summary_lines.append("Board centroids cover all 12 coarse image regions.")

    if radii.max() < 0.75:
        summary_lines.append(
            "WARNING: detected corners do not reach far enough toward the image edge "
            "(max normalized radius < 0.75). Add edge/corner views."
        )
    else:
        summary_lines.append("Edge coverage looks plausible (max normalized radius >= 0.75).")

    if shape_mismatch:
        summary_lines.append("")
        summary_lines.append("WARNING: image-size mismatches detected:")
        for name, w, h in shape_mismatch:
            summary_lines.append(f"  {name}: {w}x{h}")

    weak_names = [r["file"] for r in rows if r["status"] != "OK"]
    if weak_names:
        summary_lines.append("")
        summary_lines.append("Files needing review:")
        for name in weak_names:
            summary_lines.append("  " + name)

    summary = "\n".join(summary_lines)
    summary_path = output_dir / "summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    print(summary)
    print("")
    print("Saved:")
    print(" ", csv_path)
    print(" ", summary_path)
    if args.save_overlays:
        print(" ", overlay_dir)


if __name__ == "__main__":
    main()
