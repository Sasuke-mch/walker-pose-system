from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

import cv2
import numpy as np


REALTIME_APP_DIR = Path(__file__).resolve().parents[1]
if str(REALTIME_APP_DIR) not in sys.path:
    sys.path.insert(0, str(REALTIME_APP_DIR))

from pose_app.camera_registry import ResolvedStereoCameras, resolve_stereo_cameras
from pose_app.stereo_camera import StereoCameraConfig, StereoCameraSource


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


def detect_charuco(detector, image):
    charuco_corners, charuco_ids, marker_corners, marker_ids = (
        detector.detectBoard(image)
    )

    ids = set()
    if charuco_ids is not None:
        ids = set(
            int(x)
            for x in np.asarray(charuco_ids).reshape(-1)
        )

    return {
        "charuco_corners": charuco_corners,
        "charuco_ids": charuco_ids,
        "marker_corners": marker_corners,
        "marker_ids": marker_ids,
        "ids": ids,
    }


def draw_detection(image, detection):
    vis = image.copy()

    marker_ids = detection["marker_ids"]
    marker_corners = detection["marker_corners"]
    charuco_ids = detection["charuco_ids"]
    charuco_corners = detection["charuco_corners"]

    if marker_ids is not None and len(marker_ids):
        cv2.aruco.drawDetectedMarkers(
            vis,
            marker_corners,
            marker_ids,
        )

    if charuco_ids is not None and len(charuco_ids):
        cv2.aruco.drawDetectedCornersCharuco(
            vis,
            charuco_corners,
            charuco_ids,
        )

    return vis


def put_text(image, text, y, color):
    cv2.putText(
        image,
        text,
        (15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        color,
        2,
        cv2.LINE_AA,
    )


def resolve_camera_selection(
    args: argparse.Namespace,
) -> tuple[int, int, str, ResolvedStereoCameras | None, str]:
    """Resolve calibrated cam0/LEFT and cam1/RIGHT before opening devices."""

    indexed_mode = args.left_camera is not None or args.right_camera is not None
    if indexed_mode:
        if args.camera_registry is not None:
            raise ValueError("--camera-registry cannot be combined with --left-camera/--right-camera.")
        if args.left_camera is None or args.right_camera is None:
            raise ValueError("Manual camera mode requires both --left-camera and --right-camera.")
        if args.left_camera == args.right_camera:
            raise ValueError("LEFT and RIGHT camera indices must be different.")
        return args.left_camera, args.right_camera, args.backend, None, "manual_index"

    registry_path = args.camera_registry or (REALTIME_APP_DIR / "camera_registry.json")
    resolved = resolve_stereo_cameras(registry_path, backend=args.backend)
    return (
        resolved.left.index,
        resolved.right.index,
        resolved.backend,
        resolved,
        "physical_registry",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Capture paired stereo ChArUco images for fisheye extrinsic calibration."
    )

    parser.add_argument(
        "--camera-registry",
        type=Path,
        help=(
            "Physical-camera registry. Normal calibration capture resolves cam0/LEFT "
            "and cam1/RIGHT by PnP device identity."
        ),
    )
    parser.add_argument(
        "--left-camera",
        type=int,
        help="Unsafe manual OpenCV index for calibrated cam0/LEFT. Supply both only for diagnosis.",
    )
    parser.add_argument(
        "--right-camera",
        type=int,
        help="Unsafe manual OpenCV index for calibrated cam1/RIGHT. Supply both only for diagnosis.",
    )

    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.0)

    parser.add_argument(
        "--backend",
        choices=["msmf", "dshow", "auto"],
        default="auto",
    )

    parser.add_argument(
        "--max-pair-delta-ms",
        type=float,
        default=25.0,
    )

    parser.add_argument(
        "--min-common-corners",
        type=int,
        default=12,
        help="Minimum number of common ChArUco IDs required before saving.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=REALTIME_APP_DIR
        / "calibration"
        / "captures"
        / "stereo_extrinsic",
    )

    args = parser.parse_args()

    left_camera, right_camera, selected_backend, resolved_cameras, selection_mode = (
        resolve_camera_selection(args)
    )

    board, detector = make_board()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = args.output_root / f"session_{stamp}"

    cam0_dir = session_dir / "cam0"
    cam1_dir = session_dir / "cam1"

    cam0_dir.mkdir(parents=True, exist_ok=False)
    cam1_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "created_local": datetime.now().isoformat(timespec="seconds"),
        "mapping": {
            "left": {
                "logical_camera": "cam0",
                "opencv_index": left_camera,
                "calibration": "calibration/results/cam0_fisheye.json",
            },
            "right": {
                "logical_camera": "cam1",
                "opencv_index": right_camera,
                "calibration": "calibration/results/cam1_fisheye.json",
            },
        },
        "capture": {
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "backend": selected_backend,
            "max_pair_delta_ms": args.max_pair_delta_ms,
        },
        "camera_identity_resolution": {
            "selection_mode": selection_mode,
            "camera_registry_resolution": (
                resolved_cameras.to_dict() if resolved_cameras is not None else None
            ),
            "manual_index_warning": (
                "Manual indices are runtime enumeration values and were not physically verified."
                if selection_mode == "manual_index"
                else None
            ),
        },
        "board": {
            "type": "ChArUco",
            "dictionary": "DICT_4X4_50",
            "squares_x": 8,
            "squares_y": 6,
            "square_mm": 30.0,
            "marker_mm": 22.0,
        },
        "timestamp_warning": (
            "Host timestamps are recorded after VideoCapture.read() returns. "
            "They are not sensor exposure timestamps."
        ),
    }

    (session_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    config = StereoCameraConfig(
        left_id=left_camera,
        right_id=right_camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        backend=selected_backend,
        max_pair_delta_ms=args.max_pair_delta_ms,
        queue_size=8,
    )

    source = StereoCameraSource(config)

    csv_path = session_dir / "pairs.csv"

    saved_count = 0

    print("")
    print("=== Stereo ChArUco Capture ===")
    print(f"LEFT  = CAM0 = OpenCV index {left_camera}")
    print(f"RIGHT = CAM1 = OpenCV index {right_camera}")
    print(f"Selection = {selection_mode}; backend = {selected_backend}")
    print(f"Output: {session_dir}")
    print("")
    print("S = save one stereo pair")
    print("Q / ESC = quit")
    print("")
    print(
        "IMPORTANT: keep the ChArUco board stationary "
        "for about 0.5-1 s before pressing S."
    )
    print("")

    try:
        source.start()

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:

            writer = csv.writer(csv_file)

            writer.writerow(
                [
                    "saved_pair_id",
                    "source_pair_id",
                    "cam0_frame_id",
                    "cam1_frame_id",
                    "cam0_host_timestamp_ns",
                    "cam1_host_timestamp_ns",
                    "signed_host_delta_ms_right_minus_left",
                    "abs_host_delta_ms",
                    "cam0_charuco_corners",
                    "cam1_charuco_corners",
                    "common_charuco_corners",
                    "common_ids",
                    "cam0_file",
                    "cam1_file",
                ]
            )

            while True:
                pair = source.read(timeout_sec=1.0)

                if pair is None:
                    continue

                raw0 = pair.left.image
                raw1 = pair.right.image

                det0 = detect_charuco(detector, raw0)
                det1 = detect_charuco(detector, raw1)

                common_ids = sorted(
                    det0["ids"].intersection(det1["ids"])
                )

                n0 = len(det0["ids"])
                n1 = len(det1["ids"])
                n_common = len(common_ids)

                ready = n_common >= args.min_common_corners

                vis0 = draw_detection(raw0, det0)
                vis1 = draw_detection(raw1, det1)

                scale = min(
                    700.0 / args.width,
                    500.0 / args.height,
                    1.0,
                )

                if scale != 1.0:
                    vis0 = cv2.resize(
                        vis0,
                        (
                            int(args.width * scale),
                            int(args.height * scale),
                        ),
                    )

                    vis1 = cv2.resize(
                        vis1,
                        (
                            int(args.width * scale),
                            int(args.height * scale),
                        ),
                    )

                green = (0, 255, 0)
                orange = (0, 165, 255)
                white = (255, 255, 255)

                put_text(
                    vis0,
                    f"LEFT = CAM0 = index {left_camera}",
                    30,
                    green,
                )

                put_text(
                    vis0,
                    f"corners={n0}",
                    60,
                    green if n0 >= args.min_common_corners else orange,
                )

                put_text(
                    vis1,
                    f"RIGHT = CAM1 = index {right_camera}",
                    30,
                    green,
                )

                put_text(
                    vis1,
                    f"corners={n1}",
                    60,
                    green if n1 >= args.min_common_corners else orange,
                )

                combined = np.hstack([vis0, vis1])

                status_color = green if ready else orange

                put_text(
                    combined,
                    (
                        f"common={n_common}  "
                        f"|dt_host|={pair.abs_host_delta_ms:.2f} ms  "
                        f"saved={saved_count}"
                    ),
                    combined.shape[0] - 45,
                    status_color,
                )

                put_text(
                    combined,
                    "READY - press S"
                    if ready
                    else f"NOT READY - need >= {args.min_common_corners} common corners",
                    combined.shape[0] - 15,
                    status_color,
                )

                cv2.imshow(
                    "Stereo ChArUco Extrinsic Capture",
                    combined,
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    break

                if key in (ord("s"), ord("S")):
                    if not ready:
                        print(
                            f"Not saved: only {n_common} common corners "
                            f"(need >= {args.min_common_corners})."
                        )
                        continue

                    name0 = (
                        f"pair_{saved_count:03d}_cam0.png"
                    )

                    name1 = (
                        f"pair_{saved_count:03d}_cam1.png"
                    )

                    path0 = cam0_dir / name0
                    path1 = cam1_dir / name1

                    ok0 = cv2.imwrite(str(path0), raw0)
                    ok1 = cv2.imwrite(str(path1), raw1)

                    if not ok0 or not ok1:
                        raise RuntimeError(
                            f"Failed to save stereo pair {saved_count}."
                        )

                    writer.writerow(
                        [
                            saved_count,
                            pair.pair_id,
                            pair.left.frame_id,
                            pair.right.frame_id,
                            pair.left.host_return_timestamp_ns,
                            pair.right.host_return_timestamp_ns,
                            f"{pair.signed_host_delta_ms:.6f}",
                            f"{pair.abs_host_delta_ms:.6f}",
                            n0,
                            n1,
                            n_common,
                            " ".join(str(x) for x in common_ids),
                            str(path0.relative_to(session_dir)),
                            str(path1.relative_to(session_dir)),
                        ]
                    )

                    csv_file.flush()

                    print(
                        f"Saved pair {saved_count:03d}: "
                        f"common={n_common}, "
                        f"|dt_host|={pair.abs_host_delta_ms:.2f} ms"
                    )

                    saved_count += 1

    except KeyboardInterrupt:
        print("Stopped by Ctrl+C.")

    finally:
        source.close()
        cv2.destroyAllWindows()

    print("")
    print("=== Finished ===")
    print(f"Saved stereo pairs: {saved_count}")
    print(f"Session: {session_dir}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
