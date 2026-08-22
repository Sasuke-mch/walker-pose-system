from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import cv2

from pose_app.calibration import StereoCalibration
from pose_app.config import load_config
from pose_app.docker_service import DockerPoseService
from pose_app.http_client import PMPosePipelineClient, PoseServiceClient
from pose_app.stereo_output import StereoOutputWriter
from pose_app.stereo_sources import StereoCameraSource, StereoVideoSource
from pose_app.stereo_visualizer import draw_stereo
from pose_app.triangulation import triangulate_matches


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="双摄像头二维姿态估计、近邻时间配对与三角化基线"
    )
    parser.add_argument("--config", default=str(root / "config.json"))
    parser.add_argument("--calibration", required=True, help="双目标定JSON文件")
    parser.add_argument(
        "--model", choices=["yolo26x_pose", "pmpose"], default="yolo26x_pose"
    )

    parser.add_argument("--left-camera", type=int)
    parser.add_argument("--right-camera", type=int)
    parser.add_argument("--left-video")
    parser.add_argument("--right-video")
    parser.add_argument("--left-start-frame", type=int, default=0)
    parser.add_argument("--right-start-frame", type=int, default=0)
    parser.add_argument("--loop", action="store_true")

    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--keypoint-threshold", type=float, default=0.25)
    parser.add_argument("--max-association-cost", type=float, default=0.05)
    parser.add_argument("--max-reprojection-error-px", type=float, default=10.0)
    parser.add_argument(
        "--warn-skew-ms",
        type=float,
        default=40.0,
        help="仅记录警告，不丢弃帧对",
    )
    parser.add_argument("--camera-queue-size", type=int, default=8)
    parser.add_argument("--display-width", type=int, default=1920)
    parser.add_argument("--output-fps", type=float)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-json", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--connect-only", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    camera_mode = args.left_camera is not None or args.right_camera is not None
    video_mode = args.left_video is not None or args.right_video is not None
    if camera_mode == video_mode:
        parser.error("必须且只能选择一组双目输入：左右摄像头，或左右视频。")
    if camera_mode and (args.left_camera is None or args.right_camera is None):
        parser.error("摄像头模式必须同时提供--left-camera和--right-camera。")
    if video_mode and (not args.left_video or not args.right_video):
        parser.error("视频模式必须同时提供--left-video和--right-video。")
    if args.max_pairs is not None and args.max_pairs <= 0:
        parser.error("--max-pairs必须大于0。")
    if args.keypoint_threshold < 0 or args.keypoint_threshold > 1:
        parser.error("--keypoint-threshold必须在0到1之间。")
    if args.max_reprojection_error_px <= 0:
        parser.error("--max-reprojection-error-px必须大于0。")
    return args


def connect_client(args: argparse.Namespace, config):
    if args.model == "yolo26x_pose":
        client = PoseServiceClient(
            f"http://127.0.0.1:{config.docker.host_port}",
            config.docker.request_timeout_sec,
            config.output.jpeg_quality,
        )
    else:
        client = PMPosePipelineClient(
            f"http://127.0.0.1:{config.detector.host_port}",
            f"http://127.0.0.1:{config.pmpose.host_port}",
            config.docker.request_timeout_sec,
            config.output.jpeg_quality,
        )
    client.health()
    return client


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    calibration_source = StereoCalibration.load(args.calibration)
    source = None
    service = None
    writer = None
    preview_opened = False

    stamp = time.strftime("%Y%m%d_%H%M%S")
    source_label = "camera" if args.left_camera is not None else "video"
    run_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else config.output.root / f"{stamp}_{args.model}_stereo_{source_label}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(run_dir / "runtime.log", encoding="utf-8"),
        ],
        force=True,
    )
    log = logging.getLogger("stereo-main")

    try:
        if args.left_camera is not None:
            source = StereoCameraSource(
                args.left_camera,
                args.right_camera,
                config.camera,
                queue_size=args.camera_queue_size,
            )
        else:
            source = StereoVideoSource(
                args.left_video,
                args.right_video,
                args.left_start_frame,
                args.right_start_frame,
                args.loop,
            )

        calibration = calibration_source.for_runtime_sizes(
            (source.left_width, source.left_height),
            (source.right_width, source.right_height),
        )
        log.info(
            "双目标定已加载：model=%s baseline=%.6f %s left=%sx%s right=%sx%s",
            calibration.camera_model,
            calibration.baseline,
            calibration.length_unit,
            source.left_width,
            source.left_height,
            source.right_width,
            source.right_height,
        )

        if args.connect_only:
            client = connect_client(args, config)
        else:
            service = DockerPoseService(config, run_dir, args.model)
            client = service.start()

        output_fps = args.output_fps or min(
            config.output.realtime_output_fps, source.fps
        )
        writer = StereoOutputWriter(
            run_dir,
            save_video=not args.no_video,
            save_json=not args.no_json,
            output_fps=output_fps,
            source_name=source.name,
            calibration=calibration,
        )

        processed = 0
        while True:
            pair = source.read()
            if pair is None:
                break
            skew_ms = pair.timestamp_skew_sec * 1000.0
            if skew_ms > args.warn_skew_ms:
                log.warning(
                    "帧对%d主机接收时间差较大：%.3f ms；仍继续三角化并保留该数值。",
                    pair.pair_id,
                    skew_ms,
                )

            left_result = client.infer(
                pair.left.image,
                pair.left.frame_id,
                pair.left.timestamp_sec,
                pair.dropped_left,
            )
            right_result = client.infer(
                pair.right.image,
                pair.right.frame_id,
                pair.right.timestamp_sec,
                pair.dropped_right,
            )
            persons_3d = triangulate_matches(
                left_result.persons,
                right_result.persons,
                calibration,
                keypoint_threshold=args.keypoint_threshold,
                max_association_cost=args.max_association_cost,
                max_reprojection_error_px=args.max_reprojection_error_px,
            )
            processed += 1
            annotated = draw_stereo(
                pair,
                left_result,
                right_result,
                persons_3d,
                threshold=config.output.draw_keypoint_threshold,
                processed=processed,
                display_width=args.display_width,
            )
            writer.write(
                annotated, pair, left_result, right_result, persons_3d
            )

            if not args.headless:
                preview_opened = True
                cv2.imshow("Walker stereo pose + triangulation", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key == ord(" "):
                    while True:
                        key = cv2.waitKey(50) & 0xFF
                        if key in (27, ord("q"), ord("Q")):
                            break
                        if key == ord(" "):
                            key = -1
                            break
                    if key in (27, ord("q"), ord("Q")):
                        break

            if args.max_pairs is not None and processed >= args.max_pairs:
                break

        summary = writer.close()
        writer = None
        summary["requested_model"] = args.model
        summary["calibration_file"] = str(Path(args.calibration).resolve())
        (run_dir / "stereo_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        log.info("双目处理完成：%s", summary)
        print(f"\n输出目录：{run_dir}")
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception:
        log.exception("双目三角化程序运行失败")
        return 1
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                log.exception("关闭双目输出失败")
        if source is not None:
            try:
                source.close()
            except Exception:
                log.exception("关闭双目输入失败")
        if service is not None:
            service.stop()
        if preview_opened:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
