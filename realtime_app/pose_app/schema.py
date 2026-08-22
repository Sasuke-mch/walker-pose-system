from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import math


def finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def bbox4(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        raise ValueError(f"人体框格式错误：{value!r}")
    x1, y1, x2, y2 = [finite(v) for v in value[:4]]
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def keypoints17(value: Any) -> list[list[float]]:
    result: list[list[float]] = []
    if isinstance(value, list):
        for item in value[:17]:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                score = finite(item[2], 1.0) if len(item) >= 3 else 1.0
                result.append([finite(item[0]), finite(item[1]), score])
            else:
                result.append([0.0, 0.0, 0.0])
    while len(result) < 17:
        result.append([0.0, 0.0, 0.0])
    return result


@dataclass
class PersonPose:
    person_id: int
    bbox: list[float]
    bbox_score: float
    pose_score: float
    keypoints: list[list[float]]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersonPose":
        return cls(
            person_id=int(data.get("person_id", 0)),
            bbox=bbox4(data.get("bbox")),
            bbox_score=finite(data.get("bbox_score", 1.0)),
            pose_score=finite(data.get("pose_score", 0.0)),
            keypoints=keypoints17(data.get("keypoints", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "bbox": self.bbox,
            "bbox_score": self.bbox_score,
            "pose_score": self.pose_score,
            "keypoints": self.keypoints,
        }


@dataclass
class InferenceResult:
    source_frame_id: int
    source_timestamp_sec: float
    image_width: int
    image_height: int
    model_name: str
    model_ms: float
    roundtrip_ms: float
    persons: list[PersonPose]
    dropped_before: int = 0
    stage_times_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_frame_id": self.source_frame_id,
            "source_timestamp_sec": self.source_timestamp_sec,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "model_name": self.model_name,
            "model_ms": self.model_ms,
            "roundtrip_ms": self.roundtrip_ms,
            "stage_times_ms": self.stage_times_ms,
            "persons": [person.to_dict() for person in self.persons],
            "dropped_before": self.dropped_before,
        }
