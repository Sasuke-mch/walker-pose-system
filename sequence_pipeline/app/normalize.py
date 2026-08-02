from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import read_json, write_json

COCO17 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def _frames(payload: Any) -> list[dict]:
    if isinstance(payload, dict):
        for key in ("frames", "images", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
    if isinstance(payload, list):
        return payload
    return []


def _as_instances(frame: dict) -> list[dict]:
    for key in ("instances", "predictions", "poses", "outputs"):
        value = frame.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    for key in ("result", "output", "prediction"):
        value = frame.get(key)
        if isinstance(value, dict):
            # Batched arrays from PMPose-like outputs.
            kpts = value.get("keypoints")
            if isinstance(kpts, list) and kpts and isinstance(kpts[0], list) and kpts[0] and isinstance(kpts[0][0], list):
                boxes = value.get("bboxes") or value.get("bboxes_xyxy") or frame.get("bboxes") or []
                scores = value.get("bbox_scores") or frame.get("bbox_scores") or []
                presence = value.get("presence") or []
                visibility = value.get("visibility") or []
                result = []
                for i, kp in enumerate(kpts):
                    result.append({
                        "keypoints": kp,
                        "bbox": boxes[i] if i < len(boxes) else None,
                        "bbox_score": scores[i] if i < len(scores) else None,
                        "presence": presence[i] if i < len(presence) else None,
                        "visibility": visibility[i] if i < len(visibility) else None,
                    })
                return result
            if "keypoints" in value:
                return [value]
    if "keypoints" in frame:
        return [frame]
    return []


def _points(instance: dict) -> tuple[list, list]:
    kp = instance.get("keypoints") or instance.get("keypoints_xy") or []
    scores = instance.get("keypoint_scores") or instance.get("scores") or instance.get("presence") or []
    if not isinstance(kp, list):
        return [], []
    return kp, scores if isinstance(scores, list) else []


def normalize_raw(raw_path: Path, common_path: Path, model_key: str, detector: str | None, input_dir: Path) -> tuple[bool, str]:
    try:
        payload = read_json(raw_path)
        source_frames = _frames(payload)
        if not source_frames:
            return False, "No frames/images list found in raw JSON."

        out_frames = []
        total_instances = 0
        for frame_index, frame in enumerate(source_frames):
            if not isinstance(frame, dict):
                continue
            file_name = frame.get("file_name") or frame.get("image_name") or frame.get("name") or f"frame_{frame_index:06d}"
            common_instances = []
            for person_id, inst in enumerate(_as_instances(frame)):
                kp, scores = _points(inst)
                if not kp:
                    continue
                names = COCO17 if len(kp) == 17 else [f"keypoint_{i}" for i in range(len(kp))]
                out_kp = []
                for i, point in enumerate(kp):
                    if not isinstance(point, (list, tuple)) or len(point) < 2:
                        continue
                    score = None
                    if len(point) >= 3:
                        score = point[2]
                    elif i < len(scores):
                        score = scores[i]
                    out_kp.append({"name": names[i], "x": float(point[0]), "y": float(point[1]), "score": float(score) if score is not None else 1.0})
                bbox = inst.get("bbox_xyxy") or inst.get("bbox") or frame.get("bbox") or [0.0, 0.0, 0.0, 0.0]
                common_instances.append({
                    "person_id": person_id,
                    "bbox_xyxy": [float(v) for v in list(bbox)[:4]],
                    "bbox_score": inst.get("bbox_score"),
                    "keypoints": out_kp,
                    "extras": {k: inst[k] for k in ("presence", "visibility") if inst.get(k) is not None},
                })
            total_instances += len(common_instances)
            out_frames.append({
                "frame_index": frame_index,
                "file_name": file_name,
                "timestamp": None,
                "camera_id": None,
                "width": frame.get("width"),
                "height": frame.get("height"),
                "instances": common_instances,
            })

        write_json(common_path, {
            "schema_version": "1.0",
            "model": {"name": model_key, "detector": detector, "keypoint_format": "COCO17" if any(len(x["keypoints"]) == 17 for f in out_frames for x in f["instances"]) else "model_native"},
            "sequence": {"input_dir": str(input_dir), "num_frames": len(out_frames), "fps": None},
            "frames": out_frames,
        })
        return True, f"Normalized {len(out_frames)} frames and {total_instances} instances."
    except Exception as exc:
        return False, str(exc)
