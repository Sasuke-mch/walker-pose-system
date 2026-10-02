from __future__ import annotations
import base64
import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import cv2
import numpy as np
from .schema import InferenceResult, PersonPose
from .person_roi import RULE, foot_inclusive_box


class ServiceError(RuntimeError):
    pass


def _json_request(url: str, method: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, method=method, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise ServiceError(f"模型服务HTTP {exc.code}：{details}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise ServiceError(f"无法连接模型服务 {url}：{exc}") from exc
    try:
        data = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ServiceError(f"模型服务返回了无效JSON：{raw[:300]!r}") from exc
    if not isinstance(data, dict):
        raise ServiceError("模型服务返回值不是JSON对象。")
    if data.get("ok") is False:
        raise ServiceError(str(data.get("error", "模型服务推理失败。")))
    return data


def encode_jpeg(image: np.ndarray, quality: int) -> str:
    ok, encoded = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise ServiceError("OpenCV无法编码当前帧。")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _persons(response: dict[str, Any]) -> list[PersonPose]:
    raw = response.get("persons", [])
    if not isinstance(raw, list):
        return []
    return [PersonPose.from_dict(item) for item in raw if isinstance(item, dict)]


@dataclass
class PoseServiceClient:
    base_url: str
    timeout: float
    jpeg_quality: int

    def health(self) -> dict[str, Any]:
        return _json_request(f"{self.base_url}/health", "GET", None, min(10.0, self.timeout))

    def infer(
        self,
        image: np.ndarray,
        source_frame_id: int,
        source_timestamp_sec: float,
        dropped_before: int = 0,
    ) -> InferenceResult:
        height, width = image.shape[:2]
        payload = {
            "source_frame_id": source_frame_id,
            "source_timestamp_sec": source_timestamp_sec,
            "image_base64": encode_jpeg(image, self.jpeg_quality),
        }
        started = time.perf_counter()
        response = _json_request(f"{self.base_url}/infer", "POST", payload, self.timeout)
        roundtrip_ms = (time.perf_counter() - started) * 1000.0
        model_ms = float(response.get("model_ms", 0.0))
        return InferenceResult(
            source_frame_id=source_frame_id,
            source_timestamp_sec=source_timestamp_sec,
            image_width=int(response.get("image_width", width)),
            image_height=int(response.get("image_height", height)),
            model_name=str(response.get("model", "YOLO26x-pose")),
            model_ms=model_ms,
            roundtrip_ms=roundtrip_ms,
            persons=_persons(response),
            dropped_before=dropped_before,
            stage_times_ms={"pose_ms": model_ms},
        )


@dataclass
class PMPosePipelineClient:
    detector_url: str
    pose_url: str
    timeout: float
    jpeg_quality: int
    adaptive_box_enabled: bool = True
    adaptive_box_threshold: float = RULE["threshold"]
    adaptive_box_power: float = RULE["power"]
    adaptive_box_frames: int = 0
    adaptive_box_expanded_detections: int = 0

    def health(self) -> dict[str, Any]:
        detector = _json_request(
            f"{self.detector_url}/health", "GET", None, min(10.0, self.timeout)
        )
        pose = _json_request(
            f"{self.pose_url}/health", "GET", None, min(10.0, self.timeout)
        )
        return {"status": "ready", "detector": detector, "pose": pose}

    def infer(
        self,
        image: np.ndarray,
        source_frame_id: int,
        source_timestamp_sec: float,
        dropped_before: int = 0,
    ) -> InferenceResult:
        height, width = image.shape[:2]
        encoded = encode_jpeg(image, self.jpeg_quality)
        common = {
            "source_frame_id": source_frame_id,
            "source_timestamp_sec": source_timestamp_sec,
            "image_base64": encoded,
        }
        started = time.perf_counter()
        detector_response = _json_request(
            f"{self.detector_url}/infer", "POST", common, self.timeout
        )
        detections = detector_response.get("detections", [])
        if not isinstance(detections, list):
            detections = []
        adaptive_stats = {"expanded_detections": 0}
        if self.adaptive_box_enabled:
            # Reuse the original audited offline rule without reimplementing it.
            expanded_detections = []
            for detection in detections:
                original = list(detection["bbox"])
                roi, stats = foot_inclusive_box(
                    original, width, height,
                    self.adaptive_box_threshold, self.adaptive_box_power,
                )
                expanded_detections.append(dict(
                    detection, bbox=roi, adaptive_original_bbox=original,
                    adaptive_box_stats=stats,
                ))
                adaptive_stats["expanded_detections"] += int(stats["growth"] > 0)
            detections = expanded_detections
            self.adaptive_box_frames += 1
            self.adaptive_box_expanded_detections += int(
                adaptive_stats["expanded_detections"]
            )
        pose_payload = dict(common)
        pose_payload["detections"] = detections
        pose_response = _json_request(
            f"{self.pose_url}/infer", "POST", pose_payload, self.timeout
        )
        roundtrip_ms = (time.perf_counter() - started) * 1000.0
        detector_ms = float(detector_response.get("model_ms", 0.0))
        pose_ms = float(pose_response.get("model_ms", 0.0))
        return InferenceResult(
            source_frame_id=source_frame_id,
            source_timestamp_sec=source_timestamp_sec,
            image_width=int(pose_response.get("image_width", width)),
            image_height=int(pose_response.get("image_height", height)),
            model_name=str(pose_response.get("model", "YOLO26x + PMPose-b")),
            model_ms=detector_ms + pose_ms,
            roundtrip_ms=roundtrip_ms,
            persons=_persons(pose_response),
            dropped_before=dropped_before,
            stage_times_ms={
                "detector_ms": detector_ms,
                "pose_ms": pose_ms,
                "adaptive_box_expanded_detections": float(
                    adaptive_stats["expanded_detections"]
                ),
            },
        )

    def adaptive_box_summary(self) -> dict[str, Any]:
        return {
            "enabled": self.adaptive_box_enabled,
            "implementation": "tools.build_continuous_foot_inclusive_roi.foot_inclusive_box",
            "threshold": self.adaptive_box_threshold,
            "power": self.adaptive_box_power,
            "frames": self.adaptive_box_frames,
            "expanded_detections": self.adaptive_box_expanded_detections,
        }
