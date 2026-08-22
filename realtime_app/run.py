from __future__ import annotations
import argparse
import logging
from pathlib import Path
import sys
import time
from pose_app.config import load_config
from pose_app.docker_service import DockerPoseService
from pose_app.http_client import PMPosePipelineClient, PoseServiceClient
from pose_app.output import OutputWriter
from pose_app.runner import RunOptions, Runner
from pose_app.sources import CameraSource, ImageDirectorySource, VideoSource


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="YOLO26x-pose与PMPose视频实时测试程序")
    parser.add_argument("--config", default=str(root / "config.json"))
    parser.add_argument("--model", choices=["yolo26x_pose", "pmpose"], default="yolo26x_pose")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="本地视频路径")
    source.add_argument("--images", help="按文件名排序的图片序列目录")
    source.add_argument("--camera", type=int, help="摄像头编号，例如0")
    parser.add_argument("--source-fps", type=float, default=25.0, help="图片序列帧率")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, help="只处理前N个输出帧，适合小迭代")
    parser.add_argument("--simulate-realtime", action="store_true", help="按原始帧率读取，只处理最新帧")
    parser.add_argument("--loop", action="store_true", help="视频或图片序列循环播放")
    parser.add_argument("--headless", action="store_true", help="不打开预览窗口")
    parser.add_argument("--no-video", action="store_true", help="不保存可视化视频")
    parser.add_argument("--no-json", action="store_true", help="不保存逐帧JSONL")
    parser.add_argument("--output-dir", help="指定本次输出目录")
    parser.add_argument("--output-fps", type=float, help="覆盖输出视频帧率")
    parser.add_argument("--connect-only", action="store_true", help="连接已手动启动的服务")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def connect_client(args, config):
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
    if args.video:
        source = VideoSource(args.video, args.start_frame, args.loop)
        source_type = "video"
    elif args.images:
        source = ImageDirectorySource(args.images, args.source_fps, args.start_frame, args.loop)
        source_type = "images"
    else:
        source = CameraSource(args.camera, config.camera)
        source_type = "camera"
    mode = "realtime" if (args.simulate_realtime or source.is_live) else "sequential"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else config.output.root / f"{stamp}_{args.model}_{source_type}_{mode}"
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
    log = logging.getLogger("main")
    service = None
    try:
        if args.connect_only:
            client = connect_client(args, config)
        else:
            service = DockerPoseService(config, run_dir, args.model)
            client = service.start()
        output_fps = args.output_fps or (
            config.output.realtime_output_fps if mode == "realtime" else source.fps
        )
        writer = OutputWriter(
            run_dir,
            not args.no_video,
            not args.no_json,
            output_fps,
            source.name,
            mode,
        )
        options = RunOptions(
            mode == "realtime",
            args.max_frames,
            not args.headless,
            args.loop,
            output_fps,
            config.output.draw_keypoint_threshold,
        )
        summary = Runner(source, client, writer, options).run()
        summary["requested_model"] = args.model
        (run_dir / "summary.json").write_text(
            __import__("json").dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("处理完成：%s", summary)
        print(f"\n输出目录：{run_dir}")
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception:
        log.exception("程序运行失败")
        return 1
    finally:
        if service:
            service.stop()


if __name__ == "__main__":
    raise SystemExit(main())
