from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import queue
import sys
import threading
import time

import cv2
import numpy as np


REALTIME_APP_DIR = Path(__file__).resolve().parents[1]
if str(REALTIME_APP_DIR) not in sys.path:
    sys.path.insert(0, str(REALTIME_APP_DIR))

from pose_app.stereo_camera import CameraFrame, StereoCameraConfig, StereoCameraSource


class FrameRecorder:
    """Asynchronously write one camera stream plus per-frame timestamps.

    The camera capture thread only performs a non-blocking queue put. Encoding
    and disk I/O happen here, so recording cannot intentionally block camera
    acquisition. If this recorder cannot keep up, queue_drops is incremented
    and reported explicitly in summary.json.
    """

    def __init__(
        self,
        run_dir: Path,
        side: str,
        width: int,
        height: int,
        fps: float,
        queue_size: int = 12,
    ) -> None:
        self.side = side
        self.path = run_dir / f"{side.lower()}_capture.avi"
        self.csv_path = run_dir / f"{side.lower()}_frames.csv"
        self.width = width
        self.height = height
        self.fps = fps

        self.queue: queue.Queue[CameraFrame] = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.thread: threading.Thread | None = None

        self.error: BaseException | None = None
        self.written = 0
        self.queue_drops = 0

    def start(self, timeout_sec: float = 5.0) -> None:
        self.thread = threading.Thread(
            target=self._loop,
            name=f"{self.side.lower()}-frame-writer",
            daemon=True,
        )
        self.thread.start()

        # Do not start cameras until the writer is actually ready. Otherwise
        # early frames can fill the recorder queue during VideoWriter startup.
        if not self.ready_event.wait(timeout=timeout_sec):
            raise RuntimeError(
                f"{self.side} frame recorder did not become ready within {timeout_sec:.1f}s."
            )
        if self.error is not None:
            raise RuntimeError(f"{self.side} frame recorder failed to start.") from self.error

    def submit(self, frame: CameraFrame) -> None:
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            self.queue_drops += 1

    def _loop(self) -> None:
        writer = cv2.VideoWriter(
            str(self.path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            float(self.fps),
            (self.width, self.height),
        )

        if not writer.isOpened():
            self.error = RuntimeError(f"Cannot open video writer: {self.path}")
            self.ready_event.set()
            return

        try:
            with self.csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(
                    [
                        "frame_id",
                        "host_return_timestamp_ns",
                        "read_duration_ms",
                    ]
                )
                csv_file.flush()
                self.ready_event.set()

                while not self.stop_event.is_set() or not self.queue.empty():
                    try:
                        frame = self.queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    try:
                        image = frame.image
                        if image.shape[1] != self.width or image.shape[0] != self.height:
                            raise RuntimeError(
                                f"{self.side} frame size changed unexpectedly: "
                                f"{image.shape[1]}x{image.shape[0]}"
                            )

                        writer.write(image)
                        csv_writer.writerow(
                            [
                                frame.frame_id,
                                frame.host_return_timestamp_ns,
                                f"{frame.read_duration_ms:.6f}",
                            ]
                        )
                        self.written += 1
                    finally:
                        self.queue.task_done()

        except BaseException as exc:
            self.error = exc
            self.ready_event.set()
        finally:
            writer.release()

    def stop(self, timeout_sec: float = 15.0) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=timeout_sec)
            if self.thread.is_alive() and self.error is None:
                self.error = RuntimeError(
                    f"{self.side} frame recorder did not stop within {timeout_sec:.1f}s."
                )


class RecorderHub:
    def __init__(self, left: FrameRecorder, right: FrameRecorder) -> None:
        self.left = left
        self.right = right

    def listener(self, side: str, frame: CameraFrame) -> None:
        if side == "LEFT":
            self.left.submit(frame)
        elif side == "RIGHT":
            self.right.submit(frame)
        else:
            raise ValueError(f"Unknown camera side: {side!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture dual HF868 streams and timestamp-matched stereo pairs."
    )
    parser.add_argument("--left-camera", type=int, default=1)
    parser.add_argument("--right-camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--backend", choices=["msmf", "dshow", "auto"], default="msmf")
    parser.add_argument("--max-pair-delta-ms", type=float, default=25.0)
    parser.add_argument("--camera-queue-size", type=int, default=8)
    parser.add_argument("--recorder-queue-size", type=int, default=12)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Acquisition seconds AFTER camera startup. 0 means run until Q/ESC/Ctrl+C.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REALTIME_APP_DIR / "outputs" / "stereo_capture",
    )
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--no-raw-video", action="store_true")
    return parser.parse_args()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def main() -> int:
    args = parse_args()
    if args.duration < 0:
        raise ValueError("--duration must be >= 0")

    run_dir = args.output_root.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    config = StereoCameraConfig(
        left_id=args.left_camera,
        right_id=args.right_camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        backend=args.backend,
        max_pair_delta_ms=args.max_pair_delta_ms,
        queue_size=args.camera_queue_size,
    )
    config.validate()

    left_recorder = FrameRecorder(
        run_dir, "LEFT", args.width, args.height, args.fps, args.recorder_queue_size
    )
    right_recorder = FrameRecorder(
        run_dir, "RIGHT", args.width, args.height, args.fps, args.recorder_queue_size
    )
    recorder_hub = RecorderHub(left_recorder, right_recorder)

    listener = None if args.no_raw_video else recorder_hub.listener
    source = StereoCameraSource(config, frame_listener=listener)

    pair_csv_path = run_dir / "stereo_pairs.csv"
    metadata_path = run_dir / "metadata.json"
    summary_path = run_dir / "summary.json"

    pair_count = 0
    abs_deltas: list[float] = []

    process_started_wall = time.time()
    process_started_perf = time.perf_counter()
    capture_started_wall: float | None = None
    capture_started_perf: float | None = None
    capture_ended_perf: float | None = None
    setup_sec: float | None = None
    shutdown_sec: float | None = None

    recorders_started = False
    source_started = False
    preview_enabled = not args.no_preview
    run_error: str | None = None

    try:
        if not args.no_raw_video:
            left_recorder.start()
            try:
                right_recorder.start()
            except Exception:
                left_recorder.stop()
                raise
            recorders_started = True

        # Camera open/configuration can take several seconds under MSMF. The
        # acquisition duration MUST NOT start before this returns.
        source.start()
        source_started = True

        capture_started_perf = time.perf_counter()
        capture_started_wall = time.time()
        setup_sec = capture_started_perf - process_started_perf

        metadata = {
            "created_local": datetime.now().isoformat(timespec="seconds"),
            "camera_config": asdict(config),
            "timestamp_type": StereoCameraSource.TIMESTAMP_TYPE,
            "timestamp_warning": (
                "host_return_timestamp_ns is recorded after VideoCapture.read() returns; "
                "it is not a sensor exposure timestamp or hardware synchronization error."
            ),
            "pairing": "online one-to-one nearest-time with one-frame lookahead",
            "duration_definition": (
                "--duration starts after both cameras have been opened/configured and capture threads started."
            ),
            "left_camera_info": asdict(source.left_info),
            "right_camera_info": asdict(source.right_info),
            "frame_recording": {
                "enabled": not args.no_raw_video,
                "codec": "MJPG/AVI" if not args.no_raw_video else None,
                "note": (
                    "Videos contain decoded BGR frames re-encoded as MJPG/AVI; they are not sensor RAW "
                    "or the original UVC bitstream. Per-frame host timestamps are stored in the CSV files. "
                    "Recorder queue drops are reported explicitly."
                    if not args.no_raw_video
                    else None
                ),
            },
        }
        write_json(metadata_path, metadata)

        with pair_csv_path.open("w", newline="", encoding="utf-8") as pair_file:
            pair_writer = csv.writer(pair_file)
            pair_writer.writerow(
                [
                    "pair_id",
                    "left_frame_id",
                    "right_frame_id",
                    "left_host_return_timestamp_ns",
                    "right_host_return_timestamp_ns",
                    "signed_host_delta_ms_right_minus_left",
                    "abs_host_delta_ms",
                    "left_read_duration_ms",
                    "right_read_duration_ms",
                ]
            )

            print(f"Output: {run_dir}")
            print(
                f"LEFT={args.left_camera}, RIGHT={args.right_camera}, "
                f"{args.width}x{args.height}@{args.fps:g}, backend={args.backend}"
            )
            print(f"max_pair_delta_ms={args.max_pair_delta_ms:g}")
            if args.duration > 0:
                print(f"capture duration={args.duration:g}s (camera startup is NOT included)")
            print("Press Q/ESC to stop. Ctrl+C also works.")

            while True:
                now = time.perf_counter()
                capture_elapsed = now - capture_started_perf

                if args.duration > 0:
                    remaining = args.duration - capture_elapsed
                    if remaining <= 0:
                        break
                    read_timeout = min(0.25, remaining)
                else:
                    read_timeout = 0.25

                pair = source.read(timeout_sec=read_timeout)
                if pair is None:
                    continue

                pair_writer.writerow(
                    [
                        pair.pair_id,
                        pair.left.frame_id,
                        pair.right.frame_id,
                        pair.left.host_return_timestamp_ns,
                        pair.right.host_return_timestamp_ns,
                        f"{pair.signed_host_delta_ms:.6f}",
                        f"{pair.abs_host_delta_ms:.6f}",
                        f"{pair.left.read_duration_ms:.6f}",
                        f"{pair.right.read_duration_ms:.6f}",
                    ]
                )
                pair_count += 1
                abs_deltas.append(pair.abs_host_delta_ms)

                if preview_enabled:
                    left_show = cv2.resize(pair.left.image, (640, 360))
                    right_show = cv2.resize(pair.right.image, (640, 360))

                    cv2.putText(
                        left_show,
                        f"LEFT frame={pair.left.frame_id}",
                        (15, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        right_show,
                        f"RIGHT frame={pair.right.frame_id}",
                        (15, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    combined = np.hstack([left_show, right_show])
                    cv2.putText(
                        combined,
                        f"pair={pair.pair_id}  host |dt|={pair.abs_host_delta_ms:.2f} ms",
                        (365, 345),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow("Stereo Capture", combined)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        break

                if pair_count % 150 == 0:
                    stats = source.stats()
                    recent = np.asarray(abs_deltas[-150:], dtype=np.float64)
                    print(
                        f"pairs={pair_count}  "
                        f"recent median|dt|={np.median(recent):.2f} ms  "
                        f"captureFPS L/R={stats.left_measured_fps or 0:.2f}/"
                        f"{stats.right_measured_fps or 0:.2f}  "
                        f"drops(match) L/R={stats.left_match_drops}/{stats.right_match_drops}  "
                        f"drops(overflow) L/R={stats.left_overflow_drops}/{stats.right_overflow_drops}"
                    )

        capture_ended_perf = time.perf_counter()

    except KeyboardInterrupt:
        print("Stopped by Ctrl+C.")
        capture_ended_perf = time.perf_counter()
    except Exception as exc:
        import traceback

        run_error = repr(exc)
        traceback.print_exc()
        capture_ended_perf = time.perf_counter()
    finally:
        if capture_ended_perf is None and capture_started_perf is not None:
            capture_ended_perf = time.perf_counter()

        if source_started:
            source.close()
        cv2.destroyAllWindows()

        shutdown_started_perf = time.perf_counter()
        if recorders_started:
            left_recorder.stop()
            right_recorder.stop()
        shutdown_sec = time.perf_counter() - shutdown_started_perf

    process_ended_perf = time.perf_counter()
    total_program_sec = process_ended_perf - process_started_perf

    if capture_started_perf is not None and capture_ended_perf is not None:
        capture_loop_sec = max(1e-9, capture_ended_perf - capture_started_perf)
    else:
        capture_loop_sec = None

    stats = source.stats()
    abs_values = np.asarray(abs_deltas, dtype=np.float64)

    pair_utilization = None
    min_captured = min(stats.left_captured, stats.right_captured)
    if min_captured > 0:
        pair_utilization = stats.stereo_pairs / min_captured

    summary = {
        "run_dir": str(run_dir),
        "run_error": run_error,
        "process_started_wall_time_unix": process_started_wall,
        "capture_started_wall_time_unix": capture_started_wall,
        "timing": {
            "requested_capture_duration_sec": args.duration if args.duration > 0 else None,
            "setup_sec": setup_sec,
            "capture_loop_sec": capture_loop_sec,
            "recorder_shutdown_sec": shutdown_sec,
            "total_program_sec": total_program_sec,
        },
        "pairs_written": pair_count,
        "effective_pair_fps": (
            _safe_ratio(pair_count, capture_loop_sec) if capture_loop_sec is not None else None
        ),
        "pair_utilization_fraction": pair_utilization,
        "source_stats": stats.to_dict(),
        "pair_delta_ms": {
            "median": float(np.median(abs_values)) if abs_values.size else None,
            "mean": float(np.mean(abs_values)) if abs_values.size else None,
            "p95": float(np.percentile(abs_values, 95)) if abs_values.size else None,
            "p99": float(np.percentile(abs_values, 99)) if abs_values.size else None,
            "max": float(np.max(abs_values)) if abs_values.size else None,
        },
        "frame_recorders": {
            "enabled": not args.no_raw_video,
            "left_written": left_recorder.written if not args.no_raw_video else 0,
            "right_written": right_recorder.written if not args.no_raw_video else 0,
            "left_recorder_queue_drops": left_recorder.queue_drops if not args.no_raw_video else 0,
            "right_recorder_queue_drops": right_recorder.queue_drops if not args.no_raw_video else 0,
            "left_error": repr(left_recorder.error) if left_recorder.error else None,
            "right_error": repr(right_recorder.error) if right_recorder.error else None,
        },
        "files": {
            "metadata": str(metadata_path),
            "summary": str(summary_path),
            "stereo_pairs_csv": str(pair_csv_path),
            "left_capture_video": str(left_recorder.path) if not args.no_raw_video else None,
            "right_capture_video": str(right_recorder.path) if not args.no_raw_video else None,
            "left_frames_csv": str(left_recorder.csv_path) if not args.no_raw_video else None,
            "right_frames_csv": str(right_recorder.csv_path) if not args.no_raw_video else None,
        },
    }
    write_json(summary_path, summary)

    print("\n=== FINAL ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if run_error is not None:
        return 2
    if not args.no_raw_video and (left_recorder.error or right_recorder.error):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
