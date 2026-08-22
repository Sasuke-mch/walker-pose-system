import argparse
import json
from pathlib import Path

import cv2
import numpy as np


BOARD_SIZE = (8, 6)
SQUARE_MM = 30.0
MARKER_MM = 22.0


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_intrinsics(path):
    d = load_json(path)
    K = np.asarray(d["K"], np.float64).reshape(3, 3)
    D = np.asarray(d["D"], np.float64).reshape(4, 1)
    return K, D


def make_board():
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50
    )

    board = cv2.aruco.CharucoBoard(
        BOARD_SIZE,
        SQUARE_MM,
        MARKER_MM,
        dictionary,
    )

    detector = cv2.aruco.CharucoDetector(board)

    xyz = np.asarray(
        board.getChessboardCorners(),
        np.float64,
    ).reshape(-1, 3)

    return detector, xyz


def detect(detector, image):
    corners, ids, _, _ = detector.detectBoard(image)

    if corners is None or ids is None:
        return None, None

    return (
        np.asarray(corners, np.float64).reshape(-1, 2),
        np.asarray(ids, np.int32).reshape(-1),
    )


def get_common(c0, id0, c1, id1):
    m0 = {int(i): p for i, p in zip(id0, c0)}
    m1 = {int(i): p for i, p in zip(id1, c1)}

    ids = sorted(set(m0) & set(m1))

    if not ids:
        return None

    p0 = np.asarray([m0[i] for i in ids], np.float64).reshape(-1, 1, 2)
    p1 = np.asarray([m1[i] for i in ids], np.float64).reshape(-1, 1, 2)

    return ids, p0, p1


def rays(points, K, D):
    p = cv2.fisheye.undistortPoints(
        points,
        K,
        D,
    ).reshape(-1, 2)

    r = np.column_stack(
        [p[:, 0], p[:, 1], np.ones(len(p))]
    )

    r /= np.linalg.norm(r, axis=1, keepdims=True)

    return r


def triangulate(r0s, r1s, R, T):
    T = T.reshape(3)

    out = []
    gaps = []

    for r0, r1 in zip(r0s, r1s):
        a = R @ r0
        b = r1

        A = np.column_stack([a, -b])

        st, _, _, _ = np.linalg.lstsq(
            A,
            -T,
            rcond=None,
        )

        s, t = st

        q0 = T + s * a
        q1 = t * b

        midpoint_cam1 = (q0 + q1) / 2.0

        midpoint_cam0 = R.T @ (
            midpoint_cam1 - T
        )

        out.append(midpoint_cam0)
        gaps.append(np.linalg.norm(q0 - q1))

    return np.asarray(out), np.asarray(gaps)


def board_edges(board_xyz):
    edges = []

    for i in range(len(board_xyz)):
        for j in range(i + 1, len(board_xyz)):
            d = np.linalg.norm(
                board_xyz[i] - board_xyz[j]
            )

            if abs(d - SQUARE_MM) < 1e-5:
                edges.append((i, j))

    return edges


def plane_residual(points):
    c = points.mean(axis=0)
    q = points - c

    _, _, vh = np.linalg.svd(q)

    normal = vh[-1]

    return np.abs(q @ normal)


def stat(x):
    x = np.asarray(x, np.float64)

    return {
        "count": len(x),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def show(name, s, unit):
    print(name)
    print(f"  count  : {s['count']}")
    print(f"  mean   : {s['mean']:.6f} {unit}")
    print(f"  median : {s['median']:.6f} {unit}")
    print(f"  P95    : {s['p95']:.6f} {unit}")
    print(f"  max    : {s['max']:.6f} {unit}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--session", required=True)
    ap.add_argument("--cam0-calib", required=True)
    ap.add_argument("--cam1-calib", required=True)
    ap.add_argument("--stereo-calib", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-common-corners", type=int, default=8)

    args = ap.parse_args()

    K0, D0 = load_intrinsics(args.cam0_calib)
    K1, D1 = load_intrinsics(args.cam1_calib)

    stereo = load_json(args.stereo_calib)

    R = np.asarray(
        stereo["R_cam0_to_cam1"],
        np.float64,
    ).reshape(3, 3)

    T = np.asarray(
        stereo["T_cam0_to_cam1_mm"],
        np.float64,
    ).reshape(3, 1)

    detector, board_xyz = make_board()
    edges = board_edges(board_xyz)

    session = Path(args.session)
    cam0_dir = session / "cam0"
    cam1_dir = session / "cam1"

    edge_lengths = []
    edge_errors = []
    plane_errors = []
    ray_gaps = []

    pair_results = []
    rejected = []

    files = sorted(cam0_dir.glob("pair_*_cam0.png"))

    for f0 in files:
        pair = f0.stem.removesuffix("_cam0")
        f1 = cam1_dir / f"{pair}_cam1.png"

        if not f1.exists():
            rejected.append((pair, "missing_cam1"))
            continue

        im0 = cv2.imread(str(f0))
        im1 = cv2.imread(str(f1))

        c0, id0 = detect(detector, im0)
        c1, id1 = detect(detector, im1)

        if id0 is None or id1 is None:
            rejected.append((pair, "detection_failed"))
            continue

        common = get_common(c0, id0, c1, id1)

        if common is None:
            rejected.append((pair, "no_common_ids"))
            continue

        ids, p0, p1 = common

        if len(ids) < args.min_common_corners:
            rejected.append((pair, "too_few_common"))
            continue

        r0 = rays(p0, K0, D0)
        r1 = rays(p1, K1, D1)

        xyz, gaps = triangulate(
            r0,
            r1,
            R,
            T,
        )

        point_map = {
            int(i): p
            for i, p in zip(ids, xyz)
        }

        local_lengths = []

        for a, b in edges:
            if a in point_map and b in point_map:
                length = np.linalg.norm(
                    point_map[a] - point_map[b]
                )

                local_lengths.append(length)
                edge_lengths.append(length)
                edge_errors.append(
                    abs(length - SQUARE_MM)
                )

        pres = plane_residual(xyz)

        plane_errors.extend(pres.tolist())
        ray_gaps.extend(gaps.tolist())

        pair_results.append(
            {
                "pair": pair,
                "common_corners": len(ids),
                "edge_error_median_mm": (
                    float(
                        np.median(
                            np.abs(
                                np.asarray(local_lengths)
                                - SQUARE_MM
                            )
                        )
                    )
                    if local_lengths
                    else None
                ),
                "plane_median_mm":
                    float(np.median(pres)),
                "ray_gap_median_mm":
                    float(np.median(gaps)),
            }
        )

    print()
    print("=== Triangulation dataset ===")
    print(f"Input pairs     : {len(files)}")
    print(f"Validated pairs : {len(pair_results)}")
    print(f"Rejected pairs  : {len(rejected)}")
    print(f"Known edge      : {SQUARE_MM:.3f} mm")

    print()

    show(
        "Adjacent edge length:",
        stat(edge_lengths),
        "mm",
    )

    print()

    show(
        "Adjacent edge absolute error:",
        stat(edge_errors),
        "mm",
    )

    print()

    show(
        "Plane residual:",
        stat(plane_errors),
        "mm",
    )

    print()

    show(
        "Closest-ray gap:",
        stat(ray_gaps),
        "mm",
    )

    print()

    worst = sorted(
        pair_results,
        key=lambda x:
            x["edge_error_median_mm"]
            if x["edge_error_median_mm"] is not None
            else -1,
        reverse=True,
    )[:10]

    print("=== Worst 10 pairs ===")

    for x in worst:
        print(
            f"{x['pair']:>10s}  "
            f"n={x['common_corners']:2d}  "
            f"edge_med={x['edge_error_median_mm']:.3f} mm  "
            f"plane_med={x['plane_median_mm']:.3f} mm  "
            f"gap_med={x['ray_gap_median_mm']:.3f} mm"
        )

    result = {
        "input_pairs": len(files),
        "validated_pairs": len(pair_results),
        "rejected_pairs": rejected,
        "known_edge_mm": SQUARE_MM,
        "edge_length_mm": stat(edge_lengths),
        "edge_absolute_error_mm": stat(edge_errors),
        "plane_residual_mm": stat(plane_errors),
        "ray_gap_mm": stat(ray_gaps),
        "worst_10_pairs": worst,
        "per_pair": pair_results,
    }

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Saved:")
    print(output)


if __name__ == "__main__":
    main()
