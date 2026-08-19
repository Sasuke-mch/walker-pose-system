from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np


REALTIME_APP_DIR = Path(__file__).resolve().parents[1]
if str(REALTIME_APP_DIR) not in sys.path:
    sys.path.insert(0, str(REALTIME_APP_DIR))


def _backend_api(name: str) -> int:
    name = name.lower()
    if name == "msmf":
        return cv2.CAP_MSMF
    if name == "dshow":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open several Windows camera indices simultaneously so physical cameras can be identified."
    )
    parser.add_argument("--ids", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--backend", choices=["msmf", "dshow", "auto"], default="msmf")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--display-width", type=int, default=1600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    captures: dict[int, cv2.VideoCapture] = {}
    api = _backend_api(args.backend)

    try:
        for camera_id in args.ids:
            cap = cv2.VideoCapture(camera_id, api)
            if not cap.isOpened():
                cap.release()
                print(f"[camera {camera_id}] cannot open")
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            cap.set(cv2.CAP_PROP_FPS, args.fps)
            captures[camera_id] = cap

            try:
                backend_name = cap.getBackendName()
            except Exception:
                backend_name = args.backend.upper()
            print(
                f"[camera {camera_id}] "
                f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
                f"reported_fps={cap.get(cv2.CAP_PROP_FPS):.2f} "
                f"backend={backend_name}"
            )

        if not captures:
            print("No camera could be opened.")
            return 2

        print("Cover one physical lens at a time and read the CAMERA ID label. Press Q/ESC to quit.")

        while True:
            panels: list[np.ndarray] = []
            for camera_id, cap in captures.items():
                ok, frame = cap.read()
                if not ok or frame is None:
                    frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                    cv2.putText(
                        frame,
                        "READ FAILED",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2,
                    )
                frame = cv2.resize(frame, (args.width, args.height))
                cv2.putText(
                    frame,
                    f"CAMERA {camera_id}",
                    (20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
                panels.append(frame)

            combined = np.hstack(panels)
            if args.display_width > 0 and combined.shape[1] > args.display_width:
                scale = args.display_width / combined.shape[1]
                combined = cv2.resize(
                    combined,
                    (args.display_width, max(1, round(combined.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )

            cv2.imshow("Camera Probe", combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

        return 0
    finally:
        for cap in captures.values():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
