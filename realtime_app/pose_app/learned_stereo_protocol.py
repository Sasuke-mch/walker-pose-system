"""Process-isolated protocol between the CPU benchmark and a learned stereo worker.

Why this module exists
----------------------

The ground benchmark is a pure NumPy/OpenCV/geometry program, while IGEV-Stereo
and DynamicStereo need ``torch``.  On this host those two worlds cannot share one
interpreter: the Anaconda MKL NumPy and ``torch`` each load their own
``libiomp5md.dll``, and the second OpenMP initialisation aborts the process (OMP
Error #15) as soon as NumPy linear algebra runs after ``import torch``.  The
benchmark plane fitters call ``numpy.linalg.eigh``/``svd`` on every frame.

The two processes therefore exchange *files*:

* the benchmark process rectifies the stereo pair exactly as before, writes the
  rectified images, and writes a :class:`LearnedStereoRequest`;
* the worker process (its own conda environment, its own ``torch``) reads the
  request, runs one matcher, writes a float32 disparity ``.npy`` plus a validity
  mask, and writes a :class:`LearnedStereoResult`;
* the benchmark process reads the disparity and continues with the frozen
  filtering, triangulation, plane fitting and cross-region gate.

Contract for every disparity this module accepts
-----------------------------------------------

* ``float32``, exactly two dimensions, exactly the SGBM runtime size;
* unit is *rectified local image pixels*, the same unit as the frozen SGBM
  disparity, with the frozen sign convention: a positive disparity ``d`` at
  left-view pixel ``x`` means the partner in the right view is at ``x - d``;
* an invalid pixel is ``NaN``.  Zero is never used as "no match";
* padding may only be appended on the right/bottom edge and must be cropped back
  before the array is handed over.  No resizing is allowed, because resizing
  would also rescale the virtual focal length and the pixel disparity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np


__all__ = [
    "MatcherName",
    "LearnedStereoRequest",
    "LearnedStereoResult",
    "MULTIPLE_OF",
    "learned_stereo_request_to_json",
    "learned_stereo_request_from_json",
    "learned_stereo_result_to_json",
    "learned_stereo_result_from_json",
    "write_request",
    "read_request",
    "write_result",
    "read_result",
    "validate_disparity",
    "padded_size",
    "pad_bottom_right",
    "crop_to_original",
    "invalid_to_nan",
    "flip_disparity_for_reverse_field",
    "temporal_window_metadata",
    "RESULT_STATUSES",
]


MatcherName = Literal["igev", "dynamicstereo"]
RESULT_STATUSES = ("ok", "unavailable", "failed")

# Learned stereo networks need feature maps whose size is a multiple of 32.
MULTIPLE_OF = 32

#: frozen SGBM search range in the rectified local view, reported for context
SGBM_NUM_DISPARITIES = 160


@dataclass(frozen=True)
class LearnedStereoRequest:
    matcher_name: MatcherName
    left_image_paths: tuple[str, ...]
    right_image_paths: tuple[str, ...]
    output_disparity_path: str
    output_metadata_path: str
    runtime_width: int
    runtime_height: int
    target_index: int
    warmup: bool


@dataclass(frozen=True)
class LearnedStereoResult:
    matcher_name: MatcherName
    disparity_path: str
    valid_mask_path: str
    original_width: int
    original_height: int
    padded_width: int
    padded_height: int
    target_index: int
    temporal_window_frames: int
    future_lookahead_frames: int
    gpu_forward_ms: float | None
    cpu_worker_wall_ms: float | None
    peak_gpu_memory_mb: float | None
    status: Literal["ok", "unavailable", "failed"]
    reasons: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def temporal_window_metadata(
    matcher_name: MatcherName, window_frames: int, target_index: int
) -> dict[str, Any]:
    """The offline/online bookkeeping every result must carry.

    IGEV is a two-view matcher: one frame in, no lookahead, real-time compatible.
    DynamicStereo consumes a five-frame window and only the first frame is the
    reported target, so four frames of the future are consumed and the result is
    *not* real-time compatible.  That number must never be presented as a
    single-frame latency.
    """
    if window_frames < 1:
        raise ValueError("window_frames must be positive")
    if not 0 <= target_index < window_frames:
        raise ValueError("target_index must index the window")
    if matcher_name == "igev" and window_frames != 1:
        raise ValueError("igev is a two-view matcher and must receive exactly one frame pair")
    lookahead = max(0, int(window_frames) - 1 - int(target_index))
    return {
        "temporal_window_frames": int(window_frames),
        "target_position_in_window": int(target_index),
        "future_lookahead_frames": int(lookahead),
        "realtime_compatible": bool(window_frames == 1 and target_index == 0),
    }


def learned_stereo_request_to_json(request: LearnedStereoRequest) -> dict[str, Any]:
    payload = asdict(request)
    payload["left_image_paths"] = list(request.left_image_paths)
    payload["right_image_paths"] = list(request.right_image_paths)
    return _json_ready(payload)


def learned_stereo_request_from_json(payload: dict[str, Any]) -> LearnedStereoRequest:
    required = (
        "matcher_name", "left_image_paths", "right_image_paths", "output_disparity_path",
        "output_metadata_path", "runtime_width", "runtime_height", "target_index", "warmup",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"request JSON is missing {missing}")
    matcher = payload["matcher_name"]
    if matcher not in ("igev", "dynamicstereo"):
        raise ValueError(f"unsupported matcher_name: {matcher!r}")
    left = tuple(str(item) for item in payload["left_image_paths"])
    right = tuple(str(item) for item in payload["right_image_paths"])
    if len(left) != len(right) or not left:
        raise ValueError("left/right image path lists must be non-empty and equally long")
    window = len(left)
    target = int(payload["target_index"])
    if not 0 <= target < window:
        raise ValueError("target_index must index one of the supplied frames")
    if matcher == "igev" and window != 1:
        raise ValueError("igev is a two-view matcher and must receive exactly one frame pair")
    return LearnedStereoRequest(
        matcher_name=matcher,
        left_image_paths=left,
        right_image_paths=right,
        output_disparity_path=str(payload["output_disparity_path"]),
        output_metadata_path=str(payload["output_metadata_path"]),
        runtime_width=int(payload["runtime_width"]),
        runtime_height=int(payload["runtime_height"]),
        target_index=target,
        warmup=bool(payload["warmup"]),
    )


def learned_stereo_result_to_json(result: LearnedStereoResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["reasons"] = list(result.reasons)
    return _json_ready(payload)


def learned_stereo_result_from_json(payload: dict[str, Any]) -> LearnedStereoResult:
    required = (
        "matcher_name", "disparity_path", "valid_mask_path", "original_width", "original_height",
        "padded_width", "padded_height", "target_index", "temporal_window_frames",
        "future_lookahead_frames", "gpu_forward_ms", "cpu_worker_wall_ms", "peak_gpu_memory_mb",
        "status", "reasons",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"result JSON is missing {missing}")
    status = payload["status"]
    if status not in RESULT_STATUSES:
        raise ValueError(f"unsupported status: {status!r}")

    def optional_float(key: str) -> float | None:
        value = payload[key]
        return None if value is None else float(value)

    return LearnedStereoResult(
        matcher_name=payload["matcher_name"],
        disparity_path=str(payload["disparity_path"]),
        valid_mask_path=str(payload["valid_mask_path"]),
        original_width=int(payload["original_width"]),
        original_height=int(payload["original_height"]),
        padded_width=int(payload["padded_width"]),
        padded_height=int(payload["padded_height"]),
        target_index=int(payload["target_index"]),
        temporal_window_frames=int(payload["temporal_window_frames"]),
        future_lookahead_frames=int(payload["future_lookahead_frames"]),
        gpu_forward_ms=optional_float("gpu_forward_ms"),
        cpu_worker_wall_ms=optional_float("cpu_worker_wall_ms"),
        peak_gpu_memory_mb=optional_float("peak_gpu_memory_mb"),
        status=status,
        reasons=tuple(str(item) for item in payload["reasons"]),
        metadata=dict(payload.get("metadata") or {}),
    )


def write_request(path: Path, request: LearnedStereoRequest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(learned_stereo_request_to_json(request), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_request(path: Path) -> LearnedStereoRequest:
    return learned_stereo_request_from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def write_result(path: Path, result: LearnedStereoResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(learned_stereo_result_to_json(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_result(path: Path) -> LearnedStereoResult:
    return learned_stereo_result_from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_disparity(disparity: np.ndarray, *, width: int, height: int) -> np.ndarray:
    """Validate the disparity contract and return a contiguous float32 copy.

    Rejects anything that is not a two-dimensional ``float32`` array of exactly
    ``(height, width)``.  ``NaN`` is preserved as the invalid marker; any other
    non-finite value is converted to ``NaN`` so that it can never be mistaken for
    a valid match in a downstream comparison.
    """
    array = np.asarray(disparity)
    if array.dtype != np.float32:
        raise ValueError(f"disparity must be float32, got {array.dtype}")
    if array.ndim != 2:
        raise ValueError(f"disparity must be 2-D, got shape {array.shape}")
    if array.shape != (int(height), int(width)):
        raise ValueError(
            f"disparity shape {array.shape} does not match the required ({int(height)}, {int(width)})"
        )
    array = array.astype(np.float32, copy=True)
    non_finite = ~np.isfinite(array)
    if non_finite.any():
        array[non_finite] = np.float32("nan")
    return np.ascontiguousarray(array, dtype=np.float32)


def padded_size(width: int, height: int, multiple: int = MULTIPLE_OF) -> tuple[int, int]:
    """Smallest ``(padded_width, padded_height)`` that is a multiple on both axes."""
    if multiple < 1:
        raise ValueError("multiple must be positive")
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    padded_width = ((int(width) + multiple - 1) // multiple) * multiple
    padded_height = ((int(height) + multiple - 1) // multiple) * multiple
    return int(padded_width), int(padded_height)


def pad_bottom_right(image: np.ndarray, multiple: int = MULTIPLE_OF) -> np.ndarray:
    """Zero-pad only the right and bottom edges up to a multiple of ``multiple``."""
    array = np.asarray(image)
    if array.ndim < 2:
        raise ValueError("image must have at least two dimensions")
    height, width = int(array.shape[0]), int(array.shape[1])
    padded_width, padded_height = padded_size(width, height, multiple)
    pad_width = padded_width - width
    pad_height = padded_height - height
    if pad_width == 0 and pad_height == 0:
        return array.copy()
    pad_spec = [(0, pad_height), (0, pad_width)] + [(0, 0)] * (array.ndim - 2)
    return np.pad(array, pad_spec, mode="constant", constant_values=0)


def crop_to_original(array: np.ndarray, width: int, height: int) -> np.ndarray:
    """Undo :func:`pad_bottom_right`: take the top-left ``(height, width)`` window."""
    array = np.asarray(array)
    if array.ndim < 2:
        raise ValueError("array must have at least two dimensions")
    if array.shape[0] < height or array.shape[1] < width:
        raise ValueError(
            f"cannot crop shape {array.shape} to ({height}, {width}): the window does not fit"
        )
    return array[:height, :width].copy()


def invalid_to_nan(disparity: np.ndarray, *, minimum_valid: float = 1.0) -> np.ndarray:
    """Replace every non-match with ``NaN`` using one documented rule.

    A learned matcher has no built-in "no match" symbol, so the frozen pipeline's
    own validity rule (``disparity > 1.0``) is applied here at the boundary.
    Zero, negative and non-finite values all become ``NaN``; they are never left
    as 0, which the downstream gates would otherwise treat as "checked and
    rejected" rather than "no measurement".
    """
    array = np.asarray(disparity, dtype=np.float32).copy()
    invalid = ~np.isfinite(array) | ~(array > float(minimum_valid))
    array[invalid] = np.float32("nan")
    return array


def flip_disparity_for_reverse_field(disparity: np.ndarray) -> np.ndarray:
    """Turn a flipped-pair disparity into the frozen rightward reverse field.

    The frozen corrected reverse field is defined on the *right* view: at right
    pixel ``x`` it holds the positive offset to its partner, which lies to its
    right.  A left-reference matcher cannot be asked for that directly, so the
    pair is mirrored before inference and the result is mirrored back here.
    Mirroring maps the standard leftward search onto the required rightward one.
    """
    array = np.asarray(disparity)
    if array.ndim != 2:
        raise ValueError("disparity must be 2-D")
    return np.ascontiguousarray(np.fliplr(array), dtype=np.float32)
