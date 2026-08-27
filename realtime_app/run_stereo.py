from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import cv2

from pose_app.calibration import StereoCalibration
from pose_app.camera_registry import (
    ResolvedStereoCameras,
    backend_candidates,
    resolve_stereo_cameras,
)
from pose_app.config import load_config
from pose_app.docker_service import DockerPoseService
from pose_app.http_client import PMPosePipelineClient, PoseServiceClient
from pose_app.local_perspective import LocalPerspectiveModelInput
from pose_app.model_undistort import FisheyeModelInput
from pose_app.rotation import (
    ROTATION_CHOICES,
    restore_model_result_to_raw,
    rotate_image_for_model,
)
from pose_app.raw_pair_writer import RawStereoPairWriter
from pose_app.stereo_output import StereoOutputWriter
from pose_app.stereo_sources import (
    StereoCameraInput,
    StereoCaptureReplaySource,
    StereoSideBySideVideoSource,
    StereoVideoSource,
)
from pose_app.stereo_visualizer import draw_stereo
from pose_app.triangulation import triangulate_matches
from pose_app.schema import InferenceResult, PersonPose


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
    parser.add_argument(
        "--left-model-rotation",
        choices=ROTATION_CHOICES,
        default="none",
        help=(
            "Rotation applied only before left-image 2D inference. "
            "Predicted boxes/keypoints are inverse-mapped to raw camera pixels "
            "before stereo geometry."
        ),
    )
    parser.add_argument(
        "--right-model-rotation",
        choices=ROTATION_CHOICES,
        default="none",
        help=(
            "Rotation applied only before right-image 2D inference. "
            "Predicted boxes/keypoints are inverse-mapped to raw camera pixels "
            "before stereo geometry."
        ),
    )

    parser.add_argument("--left-camera", type=int)
    parser.add_argument("--right-camera", type=int)
    parser.add_argument(
        "--camera-registry",
        help=(
            "Windows physical-camera registry JSON. When supplied, cam0 is "
            "resolved as LEFT and cam1 as RIGHT by exact PnP device identity."
        ),
    )
    parser.add_argument(
        "--camera-backend",
        choices=["auto", "dshow", "msmf"],
        default="auto",
        help="Backend used after physical-camera resolution (default: auto=msmf then dshow).",
    )
    parser.add_argument("--left-video")
    parser.add_argument("--right-video")
    parser.add_argument("--left-start-frame", type=int, default=0)
    parser.add_argument("--right-start-frame", type=int, default=0)
    parser.add_argument(
        "--stereo-capture-dir",
        help=(
            "一次tools/capture_stereo.py采集的输出目录。严格读取其中的"
            "stereo_pairs.csv、left/right_frames.csv及left/right_capture.avi，"
            "按真实已配对frame_id回放；不能与其他输入模式混用。"
        ),
    )
    parser.add_argument(
        "--stereo-sbs-video",
        help=(
            "One full-resolution side-by-side stereo video. Each decoded frame is "
            "split directly into left/right panels without resize or re-encoding."
        ),
    )
    parser.add_argument(
        "--sbs-left-panel-width",
        type=int,
        help=(
            "Exact left-panel width in --stereo-sbs-video. Defaults to half of an "
            "even input width."
        ),
    )
    parser.add_argument(
        "--sbs-metadata-jsonl",
        help=(
            "Optional stereo_results.jsonl from the recorded side-by-side run. "
            "It restores the original left/right frame IDs and host-time skew; "
            "without it, that historical skew is marked unknown."
        ),
    )
    parser.add_argument("--sbs-start-frame", type=int, default=0)
    parser.add_argument("--loop", action="store_true")

    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--keypoint-threshold", type=float, default=0.25)
    parser.add_argument("--max-association-cost", type=float, default=0.05)
    parser.add_argument("--max-reprojection-error-px", type=float, default=10.0)
    parser.add_argument(
        "--stereo-subject-mode",
        choices=["single", "multi"],
        default="single",
        help=(
            "Expected physical walker users. single retains only the best geometric "
            "left/right match, so overlapping 2D duplicates cannot be counted as "
            "extra 3D people; use multi only after separate validation."
        ),
    )
    parser.add_argument(
        "--max-pair-delta-ms",
        type=float,
        default=25.0,
        help="摄像头在线配对允许的最大主机侧时间差；仅摄像头模式生效",
    )
    parser.add_argument(
        "--warn-skew-ms",
        type=float,
        default=15.0,
        help="主机侧帧时间差超过该值时记录警告；不等同于曝光同步误差",
    )
    parser.add_argument("--camera-queue-size", type=int, default=8)
    parser.add_argument("--display-width", type=int, default=1920)
    parser.add_argument("--output-fps", type=float)
    parser.add_argument(
        "--model-input-undistort",
        action="store_true",
        help=(
            "Undistort fisheye frames before 2D inference, then map detections "
            "back to raw fisheye pixels for stereo geometry."
        ),
    )
    parser.add_argument(
        "--model-input-local-perspective",
        choices=["off", "auto", "always"],
        default="off",
        help=(
            "Per-person local virtual pinhole re-inference for fisheye images. "
            "auto only attempts it for a large detected person; always attempts "
            "it for the highest-confidence detected person on each side. "
            "All selected keypoints are mapped back to raw fisheye pixels before "
            "the existing stereo geometry."
        ),
    )
    parser.add_argument(
        "--local-perspective-min-box-fraction",
        type=float,
        default=0.35,
        help=(
            "In auto mode, trigger local re-inference when the long side of the "
            "raw-fisheye person box is at least this fraction of the long image side."
        ),
    )
    parser.add_argument(
        "--local-perspective-margin",
        type=float,
        default=1.35,
        help="Virtual-view box expansion factor; must be greater than 1.0.",
    )
    parser.add_argument(
        "--save-raw-pairs",
        action="store_true",
        help="Save pre-inference raw left/right camera images as separate replay videos plus raw_pairs.jsonl.",
    )
    parser.add_argument(
        "--raw-pair-output-dir",
        help="Directory for --save-raw-pairs (default: <output-dir>/raw_pairs).",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-json", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--connect-only",
        action="store_true",
        help="Use an already-running pose inference service; this is not a camera-only test.",
    )
    parser.add_argument(
        "--camera-probe-only",
        action="store_true",
        help="Open and pair cameras without starting or contacting an inference service.",
    )
    parser.add_argument(
        "--probe-pairs",
        type=int,
        default=60,
        help="Number of paired frames required by --camera-probe-only (default: 60).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    indexed_camera_mode = args.left_camera is not None or args.right_camera is not None
    registry_camera_mode = args.camera_registry is not None
    camera_mode = indexed_camera_mode or registry_camera_mode
    paired_video_mode = args.left_video is not None or args.right_video is not None
    capture_replay_mode = args.stereo_capture_dir is not None
    sbs_video_mode = args.stereo_sbs_video is not None
    input_mode_count = (
        int(camera_mode)
        + int(paired_video_mode)
        + int(capture_replay_mode)
        + int(sbs_video_mode)
    )
    if input_mode_count != 1:
        parser.error(
            "必须且只能选择一组双目输入：左右摄像头、左右独立视频、"
            "一组capture_stereo采集目录，或一个左右拼接视频。"
        )
    if indexed_camera_mode and (args.left_camera is None or args.right_camera is None):
        parser.error("摄像头模式必须同时提供--left-camera和--right-camera。")
    if registry_camera_mode and indexed_camera_mode:
        parser.error("--camera-registry不能与--left-camera/--right-camera同时使用。")
    if paired_video_mode and (not args.left_video or not args.right_video):
        parser.error("视频模式必须同时提供--left-video和--right-video。")
    if capture_replay_mode and (args.left_start_frame or args.right_start_frame):
        parser.error(
            "--stereo-capture-dir不能与--left-start-frame/--right-start-frame一起使用；"
            "真实配对回放不允许跳过任一侧的历史帧。"
        )
    if capture_replay_mode and args.loop:
        parser.error("--stereo-capture-dir不支持--loop；真实配对记录只能完整回放一次。")
    if args.max_pairs is not None and args.max_pairs <= 0:
        parser.error("--max-pairs必须大于0。")
    if args.keypoint_threshold < 0 or args.keypoint_threshold > 1:
        parser.error("--keypoint-threshold必须在0到1之间。")
    if args.max_reprojection_error_px <= 0:
        parser.error("--max-reprojection-error-px必须大于0。")
    if not 0 < args.local_perspective_min_box_fraction <= 1:
        parser.error("--local-perspective-min-box-fraction必须在(0, 1]内。")
    if args.local_perspective_margin <= 1.0:
        parser.error("--local-perspective-margin必须大于1.0。")
    if args.model_input_undistort and args.model_input_local_perspective != "off":
        parser.error(
            "--model-input-undistort与--model-input-local-perspective不能同时使用；"
            "两种投影的串联没有经过验证。"
        )
    if args.max_pair_delta_ms <= 0:
        parser.error("--max-pair-delta-ms必须大于0。")
    if args.warn_skew_ms < 0:
        parser.error("--warn-skew-ms不能小于0。")
    if args.camera_queue_size < 2:
        parser.error("--camera-queue-size必须至少为2。")
    if args.camera_probe_only and not camera_mode:
        parser.error("--camera-probe-only只能用于实时双摄像头输入。")
    if args.camera_probe_only and args.connect_only:
        parser.error("--camera-probe-only不能与--connect-only同时使用。")
    if args.probe_pairs <= 0:
        parser.error("--probe-pairs必须大于0。")
    if args.sbs_left_panel_width is not None and args.sbs_left_panel_width <= 0:
        parser.error("--sbs-left-panel-width必须大于0。")
    if args.sbs_start_frame < 0:
        parser.error("--sbs-start-frame不能小于0。")
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


def _open_live_camera_source(args: argparse.Namespace, config, log):
    """Open live cameras, resolving physical identity before any capture starts.

    Registry mode tries one common backend for both cameras at a time.  It is
    not enough for a backend merely to enumerate the two devices: both must
    also accept the requested capture settings before the mapping is used.
    """

    if args.camera_registry is None:
        source = StereoCameraInput(
            args.left_camera,
            args.right_camera,
            config.camera,
            queue_size=args.camera_queue_size,
            max_pair_delta_ms=args.max_pair_delta_ms,
        )
        return source, None

    failures: list[str] = []
    for backend in backend_candidates(args.camera_backend):
        source = None
        try:
            resolved = resolve_stereo_cameras(args.camera_registry, backend=backend)
            source = StereoCameraInput(
                resolved.left.index,
                resolved.right.index,
                config.camera,
                queue_size=args.camera_queue_size,
                max_pair_delta_ms=args.max_pair_delta_ms,
                backend=resolved.backend,
            )
            log.info(
                "相机注册表解析成功：backend=%s left(cam0)=%s right(cam1)=%s",
                resolved.backend,
                resolved.left.index,
                resolved.right.index,
            )
            log.debug("相机注册表详情：%s", resolved.to_dict())
            return source, resolved
        except Exception as exc:
            if source is not None:
                source.close()
            message = f"{backend}: {exc}"
            failures.append(message)
            log.warning("相机后端预检失败，继续尝试下一候选：%s", message)

    raise RuntimeError(
        "无法以任何候选后端打开通过注册表验证的双摄像头。" + " | ".join(failures)
    )


def _run_camera_probe(
    source: StereoCameraInput,
    probe_pairs: int,
    run_dir: Path,
    log: logging.Logger,
    resolved: ResolvedStereoCameras | None,
) -> None:
    """Capture paired frames only; this intentionally never starts inference."""

    observed_skews_ms: list[float] = []
    for _ in range(probe_pairs):
        pair = source.read(timeout_sec=5.0)
        if pair is None:
            raise RuntimeError(
                "Timed out while waiting for a paired frame during camera probe."
            )
        observed_skews_ms.append(pair.timestamp_skew_sec * 1000.0)

    stats = source.stats()
    summary = {
        "mode": "camera_probe_only",
        "requested_pairs": probe_pairs,
        "received_pairs": len(observed_skews_ms),
        "mean_abs_host_delta_ms": sum(observed_skews_ms) / len(observed_skews_ms),
        "max_abs_host_delta_ms": max(observed_skews_ms),
        "camera_source_stats": stats,
        "camera_registry_resolution": resolved.to_dict() if resolved else None,
    }
    (run_dir / "camera_probe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    log.info("双相机探测通过：%s", summary)


def _eligible_person_index(
    result: InferenceResult,
    image_size: tuple[int, int],
    mode: str,
    min_box_fraction: float,
) -> int | None:
    """Return one safe single-subject target; do not silently refine everyone."""

    if not result.persons:
        return None
    long_image_side = float(max(image_size))
    candidates: list[tuple[float, int]] = []
    for index, person in enumerate(result.persons):
        x1, y1, x2, y2 = person.bbox
        long_box_side = max(abs(float(x2) - float(x1)), abs(float(y2) - float(y1)))
        if mode == "auto" and long_box_side / long_image_side < min_box_fraction:
            continue
        candidates.append((float(person.bbox_score), index))
    if not candidates:
        return None
    # One target per side is intentional: the current study is a single walker
    # user.  Refining multiple people would multiply inference cost and make
    # per-person geometric selection ambiguous.
    return max(candidates)[1]


def _central_person_index(result: InferenceResult) -> int | None:
    """Select the pose belonging to the centered local view, not a bystander."""

    if not result.persons:
        return None
    center_x = (result.image_width - 1.0) * 0.5
    center_y = (result.image_height - 1.0) * 0.5
    diagonal = max(1.0, (result.image_width ** 2 + result.image_height ** 2) ** 0.5)
    ranked: list[tuple[float, float, int]] = []
    for index, person in enumerate(result.persons):
        x1, y1, x2, y2 = person.bbox
        box_center_x, box_center_y = (float(x1) + float(x2)) * 0.5, (float(y1) + float(y2)) * 0.5
        distance = ((box_center_x - center_x) ** 2 + (box_center_y - center_y) ** 2) ** 0.5 / diagonal
        contains_center = float(x1) <= center_x <= float(x2) and float(y1) <= center_y <= float(y2)
        # Center containment dominates confidence; a local crop is constructed
        # specifically so that its intended subject passes through this point.
        ranked.append((0.0 if contains_center else 1.0, distance - 0.02 * float(person.pose_score), index))
    return min(ranked)[2]


def _replace_person(result: InferenceResult, index: int, replacement: PersonPose, local_result: InferenceResult) -> InferenceResult:
    persons = list(result.persons)
    original = persons[index]
    persons[index] = PersonPose(
        person_id=original.person_id,
        bbox=list(replacement.bbox),
        bbox_score=float(replacement.bbox_score),
        pose_score=float(replacement.pose_score),
        keypoints=[list(point) for point in replacement.keypoints],
    )
    stages = dict(result.stage_times_ms)
    stages["local_perspective_pose_ms"] = float(local_result.model_ms)
    stages["local_perspective_roundtrip_ms"] = float(local_result.roundtrip_ms)
    return InferenceResult(
        source_frame_id=result.source_frame_id,
        source_timestamp_sec=result.source_timestamp_sec,
        image_width=result.image_width,
        image_height=result.image_height,
        model_name=result.model_name,
        model_ms=float(result.model_ms) + float(local_result.model_ms),
        roundtrip_ms=float(result.roundtrip_ms) + float(local_result.roundtrip_ms),
        persons=persons,
        dropped_before=result.dropped_before,
        stage_times_ms=stages,
    )


def _attempt_local_refinement(
    *,
    client,
    raw_image,
    base_result: InferenceResult,
    preprocessor: LocalPerspectiveModelInput,
    rotation: str,
    mode: str,
    min_box_fraction: float,
    keypoint_threshold: float,
    log: logging.Logger,
) -> InferenceResult | None:
    target_index = _eligible_person_index(
        base_result, (raw_image.shape[1], raw_image.shape[0]), mode, min_box_fraction
    )
    if target_index is None:
        return None
    target = base_result.persons[target_index]
    try:
        support_points = [
            [float(point[0]), float(point[1])]
            for point in target.keypoints
            if len(point) >= 3 and float(point[2]) >= keypoint_threshold
        ]
        view = preprocessor.build(target.bbox, support_points=support_points)
        model_image = rotate_image_for_model(view.image(raw_image), rotation)
        local_model_result = client.infer(
            model_image,
            base_result.source_frame_id,
            base_result.source_timestamp_sec,
            base_result.dropped_before,
        )
        selected_index = _central_person_index(local_model_result)
        if selected_index is None:
            log.debug("局部虚拟视角未检测到居中人体：frame=%s", base_result.source_frame_id)
            return None
        local_raw_result = restore_model_result_to_raw(
            local_model_result,
            raw_width=view.output_size[0],
            raw_height=view.output_size[1],
            rotation=rotation,
            undistorted_to_raw=view.point_mapper(),
        )
        return _replace_person(
            base_result,
            target_index,
            local_raw_result.persons[selected_index],
            local_model_result,
        )
    except Exception as exc:
        # The baseline output remains valid.  A local-view failure is recorded
        # but must never terminate a data-collection run.
        log.warning("局部虚拟视角回退到基线：frame=%s reason=%s", base_result.source_frame_id, exc)
        return None


def _primary_geometry_rank(persons_3d) -> tuple[int, frozenset[int], float]:
    """Return primary valid-count, lower-body joint set, and geometry error.

    The overall valid-joint count preserves the original general quality gate.
    The actual hip/knee/ankle index set is additionally retained so a local
    view cannot exchange an already valid gait joint for a different one while
    keeping the same count.
    """

    if not persons_3d:
        return 0, frozenset(), float("inf")
    primary = persons_3d[0]
    valid_indices = frozenset(
        int(point["index"])
        for point in primary.keypoints_3d
        if bool(point["valid"])
    )
    lower_body_indices = frozenset(
        index for index in valid_indices if index in {11, 12, 13, 14, 15, 16}
    )
    reprojection = primary.mean_reprojection_error_px
    return (
        len(valid_indices),
        lower_body_indices,
        float(reprojection) if reprojection is not None else float("inf"),
    )


def _geometry_sort_key(persons_3d) -> tuple[int, int, float]:
    valid_count, lower_body_indices, reprojection = _primary_geometry_rank(persons_3d)
    return len(lower_body_indices), valid_count, -reprojection


def _strictly_better_geometry(baseline, candidate) -> bool:
    """Select a local result only when it Pareto-improves the primary subject."""

    base_valid, base_lower, base_error = _primary_geometry_rank(baseline)
    candidate_valid, candidate_lower, candidate_error = _primary_geometry_rank(candidate)
    if base_valid == 0:
        return candidate_valid >= 4
    if candidate_valid < base_valid:
        return False
    # This is stricter than comparing lower-body counts: it prevents an
    # ankle<->knee substitution even when their numbers are equal.
    if not base_lower.issubset(candidate_lower):
        return False
    if len(candidate_lower) > len(base_lower):
        return True
    if candidate_valid > base_valid:
        return True
    return candidate_error + 0.25 < base_error


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    calibration_source = StereoCalibration.load(args.calibration)
    source = None
    service = None
    writer = None
    raw_writer = None
    preview_opened = False
    local_perspective_attempts = 0
    local_perspective_selected = 0

    stamp = time.strftime("%Y%m%d_%H%M%S")
    camera_mode = args.camera_registry is not None or args.left_camera is not None
    source_label = (
        "camera"
        if camera_mode
        else "capture_replay"
        if args.stereo_capture_dir is not None
        else "sbs_video"
        if args.stereo_sbs_video is not None
        else "video"
    )
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
        resolved_cameras = None
        if camera_mode:
            source, resolved_cameras = _open_live_camera_source(args, config, log)
        elif args.stereo_capture_dir is not None:
            source = StereoCaptureReplaySource(args.stereo_capture_dir)
        elif args.stereo_sbs_video is not None:
            source = StereoSideBySideVideoSource(
                args.stereo_sbs_video,
                left_panel_width=args.sbs_left_panel_width,
                metadata_jsonl=args.sbs_metadata_jsonl,
                start_frame=args.sbs_start_frame,
                loop=args.loop,
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
        left_model_preprocessor = None
        right_model_preprocessor = None
        left_local_preprocessor = None
        right_local_preprocessor = None
        if args.model_input_undistort:
            if calibration.camera_model != "fisheye":
                raise RuntimeError(
                    "--model-input-undistort currently requires fisheye calibration."
                )
            left_model_preprocessor = FisheyeModelInput(
                calibration.left_K,
                calibration.left_D,
                (source.left_width, source.left_height),
            )
            right_model_preprocessor = FisheyeModelInput(
                calibration.right_K,
                calibration.right_D,
                (source.right_width, source.right_height),
            )
        if args.model_input_local_perspective != "off":
            if calibration.camera_model != "fisheye":
                raise RuntimeError(
                    "--model-input-local-perspective currently requires fisheye calibration."
                )
            left_local_preprocessor = LocalPerspectiveModelInput(
                calibration.left_K,
                calibration.left_D,
                (source.left_width, source.left_height),
                margin=args.local_perspective_margin,
            )
            right_local_preprocessor = LocalPerspectiveModelInput(
                calibration.right_K,
                calibration.right_D,
                (source.right_width, source.right_height),
                margin=args.local_perspective_margin,
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
        log.info(
            "二维模型输入方向归一化：left=%s right=%s；"
            "模型输出将在三角化前反变换回原始相机像素坐标。",
            args.left_model_rotation,
            args.right_model_rotation,
        )
        if args.model_input_undistort:
            log.info(
                "模型输入鱼眼去畸变已启用：推理前去畸变，关键点反变换回原始鱼眼像素。"
            )
        if args.model_input_local_perspective != "off":
            log.info(
                "局部虚拟透视已启用：mode=%s min_box_fraction=%.3f margin=%.3f；"
                "只在基线检测到的最高置信度单人上二次推理，最终按双目几何严格择优。",
                args.model_input_local_perspective,
                args.local_perspective_min_box_fraction,
                args.local_perspective_margin,
            )
        if args.stereo_sbs_video is not None:
            log.info(
                "离线SBS重放：每个已解码帧按x=%d直接分成left=%dx%d、right=%dx%d；"
                "不缩放、不重新编码。",
                source.left_width,
                source.left_width,
                source.left_height,
                source.right_width,
                source.right_height,
            )
        if args.stereo_capture_dir is not None:
            log.info(
                "离线真实配对重放：从stereo_pairs.csv恢复%d对历史帧；"
                "left/right AVI不会按相同帧号硬配。",
                len(source.pairs),
            )

        if args.camera_probe_only:
            assert isinstance(source, StereoCameraInput)
            _run_camera_probe(
                source,
                args.probe_pairs,
                run_dir,
                log,
                resolved_cameras,
            )
            print(f"\n相机探测输出目录：{run_dir}")
            return 0

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
        if args.save_raw_pairs:
            raw_output_dir = (
                Path(args.raw_pair_output_dir).resolve()
                if args.raw_pair_output_dir
                else run_dir / "raw_pairs"
            )
            raw_writer = RawStereoPairWriter(raw_output_dir, source.fps)
            log.info(
                "将保存推理前原始左右帧：left/right MP4 + raw_pairs.jsonl，目录=%s",
                raw_output_dir,
            )

        processed = 0
        while True:
            pair = source.read()
            if pair is None:
                break
            skew_ms = pair.timestamp_skew_sec * 1000.0
            if args.warn_skew_ms > 0 and skew_ms > args.warn_skew_ms:
                log.warning(
                    "帧对%d主机侧read-return时间差较大：%.3f ms；"
                    "该量不是两相机曝光同步误差。",
                    pair.pair_id,
                    skew_ms,
                )
            if raw_writer is not None:
                raw_writer.write(pair)

            left_inference_source = (
                left_model_preprocessor.image(pair.left.image)
                if left_model_preprocessor is not None
                else pair.left.image
            )
            right_inference_source = (
                right_model_preprocessor.image(pair.right.image)
                if right_model_preprocessor is not None
                else pair.right.image
            )
            left_model_image = rotate_image_for_model(
                left_inference_source, args.left_model_rotation
            )
            right_model_image = rotate_image_for_model(
                right_inference_source, args.right_model_rotation
            )
            left_model_result = client.infer(
                left_model_image,
                pair.left.frame_id,
                pair.left.timestamp_sec,
                pair.dropped_left,
            )
            right_model_result = client.infer(
                right_model_image,
                pair.right.frame_id,
                pair.right.timestamp_sec,
                pair.dropped_right,
            )
            left_result = restore_model_result_to_raw(
                left_model_result,
                raw_width=pair.left.image.shape[1],
                raw_height=pair.left.image.shape[0],
                rotation=args.left_model_rotation,
                undistorted_to_raw=(
                    left_model_preprocessor.point_mapper()
                    if left_model_preprocessor is not None
                    else None
                ),
            )
            right_result = restore_model_result_to_raw(
                right_model_result,
                raw_width=pair.right.image.shape[1],
                raw_height=pair.right.image.shape[0],
                rotation=args.right_model_rotation,
                undistorted_to_raw=(
                    right_model_preprocessor.point_mapper()
                    if right_model_preprocessor is not None
                    else None
                ),
            )
            persons_3d = triangulate_matches(
                left_result.persons,
                right_result.persons,
                calibration,
                keypoint_threshold=args.keypoint_threshold,
                max_association_cost=args.max_association_cost,
                max_reprojection_error_px=args.max_reprojection_error_px,
                max_matches=1 if args.stereo_subject_mode == "single" else None,
            )
            if left_local_preprocessor is not None and right_local_preprocessor is not None:
                left_local = _attempt_local_refinement(
                    client=client,
                    raw_image=pair.left.image,
                    base_result=left_result,
                    preprocessor=left_local_preprocessor,
                    rotation=args.left_model_rotation,
                    mode=args.model_input_local_perspective,
                    min_box_fraction=args.local_perspective_min_box_fraction,
                    keypoint_threshold=args.keypoint_threshold,
                    log=log,
                )
                right_local = _attempt_local_refinement(
                    client=client,
                    raw_image=pair.right.image,
                    base_result=right_result,
                    preprocessor=right_local_preprocessor,
                    rotation=args.right_model_rotation,
                    mode=args.model_input_local_perspective,
                    min_box_fraction=args.local_perspective_min_box_fraction,
                    keypoint_threshold=args.keypoint_threshold,
                    log=log,
                )
                local_perspective_attempts += int(left_local is not None) + int(right_local is not None)
                candidates = [(left_result, right_result, persons_3d)]
                if left_local is not None:
                    candidates.append((
                        left_local,
                        right_result,
                        triangulate_matches(
                            left_local.persons, right_result.persons, calibration,
                            keypoint_threshold=args.keypoint_threshold,
                            max_association_cost=args.max_association_cost,
                            max_reprojection_error_px=args.max_reprojection_error_px,
                            max_matches=1 if args.stereo_subject_mode == "single" else None,
                        ),
                    ))
                if right_local is not None:
                    candidates.append((
                        left_result,
                        right_local,
                        triangulate_matches(
                            left_result.persons, right_local.persons, calibration,
                            keypoint_threshold=args.keypoint_threshold,
                            max_association_cost=args.max_association_cost,
                            max_reprojection_error_px=args.max_reprojection_error_px,
                            max_matches=1 if args.stereo_subject_mode == "single" else None,
                        ),
                    ))
                if left_local is not None and right_local is not None:
                    candidates.append((
                        left_local,
                        right_local,
                        triangulate_matches(
                            left_local.persons, right_local.persons, calibration,
                            keypoint_threshold=args.keypoint_threshold,
                            max_association_cost=args.max_association_cost,
                            max_reprojection_error_px=args.max_reprojection_error_px,
                            max_matches=1 if args.stereo_subject_mode == "single" else None,
                        ),
                    ))
                baseline_geometry = persons_3d
                best_left, best_right, best_geometry = max(
                    candidates,
                    key=lambda value: _geometry_sort_key(value[2]),
                )
                if _strictly_better_geometry(baseline_geometry, best_geometry):
                    left_result, right_result, persons_3d = best_left, best_right, best_geometry
                    local_perspective_selected += 1
                    left_result.stage_times_ms["local_perspective_selected"] = 1.0
                    right_result.stage_times_ms["local_perspective_selected"] = 1.0
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
        if raw_writer is not None:
            summary["raw_pair_recording"] = raw_writer.close()
            raw_writer = None
        summary["requested_model"] = args.model
        summary["calibration_file"] = str(Path(args.calibration).resolve())
        summary["model_input_rotation"] = {
            "left": args.left_model_rotation,
            "right": args.right_model_rotation,
            "result_coordinate_space": "raw_camera_pixels_after_inverse_rotation",
        }
        summary["model_input_undistort"] = bool(args.model_input_undistort)
        summary["stereo_subject_mode"] = args.stereo_subject_mode
        summary["model_input_local_perspective"] = {
            "mode": args.model_input_local_perspective,
            "min_box_fraction": args.local_perspective_min_box_fraction,
            "margin": args.local_perspective_margin,
            "side_refinement_attempts": local_perspective_attempts,
            "pairwise_geometric_selections": local_perspective_selected,
            "selection_rule": (
                "score only the primary geometric person; retain at least the baseline "
                "valid-joint count and every baseline-valid hip/knee/ankle (set inclusion), then prefer additional "
                "hip/knee/ankle points, additional total points, or reprojection error "
                "improvement greater than 0.25 px"
            ),
        }
        summary["max_pair_delta_ms"] = (
            args.max_pair_delta_ms if camera_mode else None
        )
        if camera_mode and hasattr(source, "stats"):
            summary["camera_source_stats"] = source.stats()
        if hasattr(source, "integrity_metadata"):
            summary["input_integrity"] = source.integrity_metadata()
        if resolved_cameras is not None:
            summary["camera_registry_resolution"] = resolved_cameras.to_dict()
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
        if raw_writer is not None:
            try:
                raw_writer.close()
            except Exception:
                log.exception("关闭原始双目帧录制失败")
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
