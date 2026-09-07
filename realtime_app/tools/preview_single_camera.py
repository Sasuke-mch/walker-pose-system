"""Display one registered camera in a screen-fitting window; never record media."""

import argparse
import ctypes
from pathlib import Path
import sys
import time
import tkinter as tk

import cv2
import numpy as np
from cv2_enumerate_cameras import enumerate_cameras

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_app.camera_registry import load_camera_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera-registry', type=Path, required=True)
    parser.add_argument('--side', choices=['left', 'right'], default='right')
    args = parser.parse_args()
    registered = load_camera_registry(args.camera_registry)[args.side == 'right']
    token = lambda value: ''.join(c for c in value.casefold() if c.isalnum())
    matches = [d for d in enumerate_cameras(cv2.CAP_DSHOW)
               if token(registered.instance_id) in token(d.path)]
    if len(matches) != 1:
        raise RuntimeError(f'Expected one {args.side} camera; found {len(matches)}')
    if sys.platform == 'win32':
        ctypes.windll.user32.SetProcessDPIAware()
    root = tk.Tk()
    root.withdraw()
    screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    cap = cv2.VideoCapture(matches[0].index, cv2.CAP_DSHOW)
    title = f'{args.side.upper()} camera - full frame - NO RECORDING'
    try:
        if not cap.isOpened():
            raise RuntimeError('Camera could not open')
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(title, 30, 40)
        first = True
        last_success = time.monotonic()
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if time.monotonic() - last_success > 10:
                    raise RuntimeError('No camera frame received within 10 seconds')
                time.sleep(0.02)
                continue
            last_success = time.monotonic()
            height, width = frame.shape[:2]
            if (width, height) != (1920, 1080):
                raise RuntimeError(f'Unexpected decoded size: {width}x{height}')
            scale = min(1.0, 1200 / width, (screen_w - 120) / width,
                        (screen_h - 200) / height)
            dw, dh = round(width * scale), round(height * scale)
            # Resize the entire decoded image, never slice/crop it. Text and border
            # are outside the image so the first/last source columns remain visible.
            canvas = np.zeros((dh + 90, dw + 24, 3), dtype=np.uint8)
            canvas[54:54 + dh, 12:12 + dw] = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
            cv2.rectangle(canvas, (10, 52), (13 + dw, 55 + dh), (0, 255, 0), 1)
            lines = [f'{args.side.upper()} / {registered.key} | RAW {width}x{height} | MJPG | NO RECORDING',
                     f'Full width x=0..{width-1}; full height y=0..{height-1} | Q / ESC: close']
            for i, line in enumerate(lines):
                cv2.putText(canvas, line, (12, 20 + i * 23), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, 'LEFT EDGE', (12, dh + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.putText(canvas, 'RIGHT EDGE', (dw - 86, dh + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.imshow(title, canvas)
            key = cv2.waitKey(1) & 0xff
            if first:
                print(f'DISPLAYED side={args.side} index={matches[0].index} decoded={width}x{height} '
                      f'screen={screen_w}x{screen_h} image={dw}x{dh} canvas={canvas.shape[1]}x{canvas.shape[0]} '
                      f'driver_zoom={cap.get(cv2.CAP_PROP_ZOOM)} no_crop=True no_media_saved=True', flush=True)
                first = False
            if key in (27, ord('q'), ord('Q')) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
