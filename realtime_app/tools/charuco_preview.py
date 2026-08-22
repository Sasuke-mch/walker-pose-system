import argparse
import time
from pathlib import Path
import cv2

def make_board():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((8, 6), 30.0, 22.0, dictionary)
    return board

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=1)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--save-dir", default="outputs/charuco_preview")
    a = p.parse_args()

    board = make_board()
    detector = cv2.aruco.CharucoDetector(board)

    cap = cv2.VideoCapture(a.camera, cv2.CAP_MSMF)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {a.camera} with MSMF")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    cap.set(cv2.CAP_PROP_FPS, a.fps)

    print("Opened:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
          "x", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
          "reported FPS =", cap.get(cv2.CAP_PROP_FPS))
    print("q = quit, s = save RAW frame")

    save_dir = Path(a.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    last_t = time.perf_counter()
    fps_ema = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            raw = frame.copy()
            vis = frame.copy()

            cc, ci, mc, mi = detector.detectBoard(frame)

            markers = 0 if mi is None else len(mi)
            corners = 0 if ci is None else len(ci)

            if mi is not None and len(mi):
                cv2.aruco.drawDetectedMarkers(vis, mc, mi)
            if ci is not None and len(ci):
                cv2.aruco.drawDetectedCornersCharuco(vis, cc, ci)

            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            if dt > 0:
                inst = 1.0 / dt
                fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst

            text = f"markers={markers} charuco_corners={corners}"
            if fps_ema is not None:
                text += f" preview_fps={fps_ema:.1f}"
            cv2.putText(vis, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0,255,0) if corners >= 10 else (0,165,255), 2)

            scale = min(1400 / vis.shape[1], 800 / vis.shape[0], 1.0)
            show = vis if scale == 1 else cv2.resize(
                vis, (int(vis.shape[1]*scale), int(vis.shape[0]*scale))
            )
            cv2.imshow("ChArUco Preview", show)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("s"):
                path = save_dir / f"charuco_raw_{time.strftime('%Y%m%d_%H%M%S')}_{saved:03d}.png"
                cv2.imwrite(str(path), raw)
                print("Saved:", path)
                saved += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
