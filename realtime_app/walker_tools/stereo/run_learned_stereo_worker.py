#!/usr/bin/env python3
"""Run one learned stereo matcher in its own process and environment.

Usage (from the repository root, with the matcher's own interpreter)::

    <model env python> realtime_app/tools/run_learned_stereo_worker.py \
        --request <absolute request.json> --result <absolute result.json>

The worker is intentionally tiny and one-way: read the request, load the
already-rectified images, run one forward pass of the requested official model,
write ``disparity.npy`` (float32, NaN for invalid) and ``valid_mask.npy``, write
the result JSON, exit.  It never touches the benchmark's geometry: it does not
import ``scene_geometry_variants``, ``benchmark_ground_walker_reconstruction``,
any plane fitter, and it performs no NumPy linear algebra -- so the Anaconda MKL
NumPy / torch OpenMP conflict of the benchmark process cannot arise here, and no
gap fitting is possible in this process by construction.

Official models and weights only; nothing is trained, fine-tuned or structurally
modified:

* IGEV-Stereo -- https://github.com/gangweiX/IGEV (MIT), weights from the
  official Google Drive folder linked by that README, ``sceneflow.pth``.
* DynamicStereo -- https://github.com/facebookresearch/dynamic_stereo
  (Attribution-NonCommercial 4.0), weights
  ``https://dl.fbaipublicfiles.com/dynamic_replica_v1/dynamic_stereo_sf.pth``.

Disparity sign convention
-------------------------

The frozen SGBM field uses the standard left-reference convention: a positive
disparity ``d`` at left pixel ``x`` matches right pixel ``x - d``.  Evidence from
the official code:

* IGEV ``core/geometry.py::build_gwc_volume`` fills ``volume[d, x]`` from
  ``left[x]`` and ``right[x - d]``, so its output already means "partner is to the
  left" -- factor ``+1``.
* DynamicStereo ``models/core/corr.py::CorrBlock1D.__call__`` samples the
  correlation at ``coords + flow``, so a positive flow means "partner is to the
  right" -- the opposite -- factor ``-1``.

That expectation is *verified* here with a photometric test on the actual
rectified pair (patch ZNCC of the left view against the right view sampled at
``x - d`` versus ``x + d``).  If the test is decisive and contradicts the
documented factor, the request fails loudly instead of silently producing a
mirrored depth map.  Both scores are stored in the result metadata.
"""

from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
_tool_prepare_imports()

import argparse
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any

import numpy as np


TOOLS_ROOT = (Path(__file__).resolve().parents[2] / "tools")
PROJECT_ROOT = TOOLS_ROOT.parents[1]
REALTIME_ROOT = PROJECT_ROOT / "realtime_app"
for candidate in (str(REALTIME_ROOT), str(TOOLS_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from pose_app.learned_stereo_protocol import (  # noqa: E402
    LearnedStereoRequest,
    LearnedStereoResult,
    SGBM_NUM_DISPARITIES,
    crop_to_original,
    invalid_to_nan,
    pad_bottom_right,
    padded_size,
    read_request,
    temporal_window_metadata,
    write_result,
)


DEFAULT_IGEV_REPO = PROJECT_ROOT / "external_models" / "IGEV" / "IGEV-Stereo"
DEFAULT_IGEV_WEIGHTS = PROJECT_ROOT / "external_models" / "weights" / "igev" / "sceneflow" / "sceneflow.pth"
DEFAULT_DS_REPO = PROJECT_ROOT / "external_models" / "dynamic_stereo"
DEFAULT_DS_WEIGHTS = (
    PROJECT_ROOT / "external_models" / "weights" / "dynamic_stereo" / "dynamic_stereo_sf.pth"
)

IGEV_REPO_URL = "https://github.com/gangweiX/IGEV"
IGEV_WEIGHTS_URL = "https://drive.google.com/drive/folders/1SsMHRyN7808jDViMN1sKz1Nx-71JxUuz (official README link, sceneflow/sceneflow.pth)"
DS_REPO_URL = "https://github.com/facebookresearch/dynamic_stereo"
DS_WEIGHTS_URL = "https://dl.fbaipublicfiles.com/dynamic_replica_v1/dynamic_stereo_sf.pth"

# Documented output-sign factor relative to the frozen convention (see module docstring).
EXPECTED_SIGN_FACTOR = {"igev": 1.0, "dynamicstereo": -1.0}
SIGN_CHECK_MINIMUM_PIXELS = 2000
SIGN_CHECK_DECISIVE_MARGIN = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--igev-repo", type=Path, default=DEFAULT_IGEV_REPO)
    parser.add_argument("--igev-weights", type=Path, default=DEFAULT_IGEV_WEIGHTS)
    parser.add_argument("--igev-iters", type=int, default=32)
    parser.add_argument("--dynamicstereo-repo", type=Path, default=DEFAULT_DS_REPO)
    parser.add_argument("--dynamicstereo-weights", type=Path, default=DEFAULT_DS_WEIGHTS)
    parser.add_argument("--dynamicstereo-iters", type=int, default=20)
    return parser.parse_args()


def load_rgb(path: str) -> np.ndarray:
    """Load an image as uint8 HxWx3 RGB without importing any geometry library."""
    from PIL import Image

    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.uint8)


def to_tensor_batch(images: list[np.ndarray], padded_width: int, padded_height: int) -> Any:
    """Stack HxWx3 uint8 RGB frames into a padded float tensor (T, 3, H, W)."""
    import torch

    frames = [pad_bottom_right(image, 32) for image in images]
    for frame in frames:
        if frame.shape[0] != padded_height or frame.shape[1] != padded_width:
            raise RuntimeError(
                f"padded frame shape {frame.shape[:2]} != ({padded_height}, {padded_width})"
            )
    array = np.stack(frames, axis=0).astype(np.float32)
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()
    return tensor


def patch_zncc_sign_check(
    left_rgb: np.ndarray, right_rgb: np.ndarray, disparity: np.ndarray, factor: float
) -> dict[str, Any]:
    """Median patch ZNCC of the left view against the right view at ``x - factor*d``.

    Purely photometric and deterministic: it uses no plane fitting and no NumPy
    linear algebra.  Local mean subtraction cancels the additive brightness bias
    and the per-patch normalisation cancels the multiplicative one, both of which
    were measured to be present in this calibration.
    """
    height, width = disparity.shape
    step = 6
    radius = 3
    ys, xs = np.mgrid[radius : height - radius : step, radius : width - radius : step]
    ys = ys.ravel()
    xs = xs.ravel()
    d = disparity[ys, xs]
    finite = np.isfinite(d) & (np.abs(d) > 3.0)
    ys, xs, d = ys[finite], xs[finite], d[finite]
    if len(ys) == 0:
        return {"pixels": 0, "median_zncc": None}

    offsets = np.arange(-radius, radius + 1)
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    dy = dy.ravel()
    dx = dx.ravel()
    left_patches = left_rgb[ys[:, None] + dy[None, :], xs[:, None] + dx[None, :]].astype(np.float64)
    partner_x = np.rint(xs.astype(np.float64) - factor * d.astype(np.float64)).astype(np.int64)
    in_bounds = (partner_x - radius >= 0) & (partner_x + radius < width)
    if not in_bounds.any():
        return {"pixels": 0, "median_zncc": None}
    left_patches = left_patches[in_bounds]
    ys = ys[in_bounds]
    partner_x = partner_x[in_bounds]
    right_patches = right_rgb[ys[:, None] + dy[None, :], partner_x[:, None] + dx[None, :]].astype(np.float64)

    left_centered = left_patches - left_patches.mean(axis=1, keepdims=True)
    right_centered = right_patches - right_patches.mean(axis=1, keepdims=True)
    numerator = (left_centered * right_centered).sum(axis=1)
    denominator = np.sqrt((left_centered ** 2).sum(axis=1) * (right_centered ** 2).sum(axis=1))
    usable = denominator > 1e-6
    if not usable.any():
        return {"pixels": 0, "median_zncc": None}
    zncc = numerator[usable] / denominator[usable]
    return {
        "pixels": int(usable.sum()),
        "median_zncc": float(np.median(zncc)),
    }


def verify_sign_convention(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    raw_disparity: np.ndarray,
    expected_factor: float,
) -> tuple[float, dict[str, Any], list[str]]:
    """Return the factor to apply plus the recorded evidence and any failure reasons."""
    forward = patch_zncc_sign_check(left_rgb, right_rgb, raw_disparity, 1.0)
    mirrored = patch_zncc_sign_check(left_rgb, right_rgb, raw_disparity, -1.0)
    evidence = {
        "zncc_factor_plus_1": forward["median_zncc"],
        "zncc_factor_minus_1": mirrored["median_zncc"],
        "zncc_pixels": min(int(forward["pixels"]), int(mirrored["pixels"])),
        "expected_factor_from_official_code": expected_factor,
    }
    reasons: list[str] = []
    if min(int(forward["pixels"]), int(mirrored["pixels"])) < SIGN_CHECK_MINIMUM_PIXELS:
        evidence["photometric_verification"] = "inconclusive_too_few_pixels"
        return expected_factor, evidence, reasons
    if forward["median_zncc"] is None or mirrored["median_zncc"] is None:
        evidence["photometric_verification"] = "inconclusive_empty_patches"
        return expected_factor, evidence, reasons
    margin = float(forward["median_zncc"]) - float(mirrored["median_zncc"])
    evidence["zncc_margin_factor_plus_minus_minus"] = margin
    measured_factor = 1.0 if margin > 0 else -1.0
    if abs(margin) < SIGN_CHECK_DECISIVE_MARGIN:
        evidence["photometric_verification"] = "inconclusive_small_margin"
        return expected_factor, evidence, reasons
    evidence["photometric_verification"] = "decisive"
    evidence["measured_factor"] = measured_factor
    if measured_factor != expected_factor:
        reasons.append("sign_convention_conflicts_with_documented_official_convention")
    return measured_factor, evidence, reasons


def build_igev(args: argparse.Namespace, device: str) -> Any:
    import torch

    repo = Path(args.igev_repo)
    if not (repo / "core" / "igev_stereo.py").exists():
        raise FileNotFoundError(f"IGEV-Stereo sources not found under {repo}")
    for candidate in (str(repo), str(repo / "core")):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    from igev_stereo import IGEVStereo  # noqa: PLC0415

    model_args = SimpleNamespace(
        max_disp=192,
        mixed_precision=False,
        precision_dtype="float32",
        corr_levels=2,
        corr_radius=4,
        n_downsample=2,
        n_gru_layers=3,
        hidden_dims=[128, 128, 128],
    )
    model = IGEVStereo(model_args)
    state = torch.load(str(args.igev_weights), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"IGEV checkpoint is missing {len(missing)} tensors, e.g. {missing[:3]}")
    model.to(device)
    model.eval()
    return (model, model_args, {"unexpected_checkpoint_keys": len(unexpected)})


def build_dynamicstereo(args: argparse.Namespace, device: str) -> Any:
    import torch

    repo = Path(args.dynamicstereo_repo)
    if not (repo / "models" / "core" / "dynamic_stereo.py").exists():
        raise FileNotFoundError(f"DynamicStereo sources not found under {repo}")
    # ``dynamic_stereo.models...`` is an absolute package import, so the *parent*
    # of the repository directory must be importable (the official README adds it
    # to PYTHONPATH in exactly the same way).
    if str(repo.parent) not in sys.path:
        sys.path.insert(0, str(repo.parent))
    from dynamic_stereo.models.core.dynamic_stereo import DynamicStereo  # noqa: PLC0415

    model = DynamicStereo(
        mixed_precision=False,
        num_frames=5,
        attention_type="self_stereo_temporal_update_time_update_space",
        use_3d_update_block=True,
        different_update_blocks=True,
    )
    state = torch.load(str(args.dynamicstereo_weights), map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state:
        state = {"module." + key: value for key, value in state["state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(
            f"DynamicStereo checkpoint is missing {len(missing)} tensors, e.g. {missing[:3]}"
        )
    model.to(device)
    model.eval()
    return (model, None, {"unexpected_checkpoint_keys": len(unexpected)})


def run_request(args: argparse.Namespace, request: LearnedStereoRequest) -> LearnedStereoResult:
    import torch

    window_info = temporal_window_metadata(
        request.matcher_name, len(request.left_image_paths), int(request.target_index)
    )

    def failed(reasons: list[str], metadata: dict[str, Any] | None = None) -> LearnedStereoResult:
        return LearnedStereoResult(
            matcher_name=request.matcher_name,
            disparity_path=str(request.output_disparity_path),
            valid_mask_path=str(Path(request.output_disparity_path).with_name("valid_mask.npy")),
            original_width=int(request.runtime_width),
            original_height=int(request.runtime_height),
            padded_width=0,
            padded_height=0,
            target_index=int(request.target_index),
            temporal_window_frames=window_info["temporal_window_frames"],
            future_lookahead_frames=window_info["future_lookahead_frames"],
            gpu_forward_ms=None,
            cpu_worker_wall_ms=None,
            peak_gpu_memory_mb=None,
            status="failed",
            reasons=tuple(reasons),
            metadata=dict(metadata or {}),
        )

    window = len(request.left_image_paths)
    if request.matcher_name == "dynamicstereo" and window != 5:
        return failed(["dynamicstereo_requires_exactly_five_window_frames"])
    if not torch.cuda.is_available():
        return failed(
            ["cuda_not_available_in_worker_environment"],
            {"torch_version": torch.__version__, "cuda_build": torch.version.cuda},
        )

    device = args.device if torch.cuda.is_available() else "cpu"
    model_load_start = time.perf_counter()
    if request.matcher_name == "igev":
        model, model_args, load_notes = build_igev(args, device)
        weights_path, repo_url, weights_url, iters = args.igev_weights, IGEV_REPO_URL, IGEV_WEIGHTS_URL, args.igev_iters
        model_settings = {"max_disp": model_args.max_disp, "mixed_precision": False, "precision_dtype": "float32"}
    else:
        model, _, load_notes = build_dynamicstereo(args, device)
        weights_path, repo_url, weights_url, iters = (
            args.dynamicstereo_weights, DS_REPO_URL, DS_WEIGHTS_URL, args.dynamicstereo_iters,
        )
        model_settings = {
            "num_frames": 5,
            "attention_type": "self_stereo_temporal_update_time_update_space",
            "use_3d_update_block": True,
            "different_update_blocks": True,
            "mixed_precision": False,
        }
    model_load_ms = (time.perf_counter() - model_load_start) * 1000.0

    padded_width, padded_height = padded_size(request.runtime_width, request.runtime_height, 32)

    # The measured interval starts here, at the first read of the model input, and
    # therefore excludes model construction, checkpoint loading and CUDA warm-up.
    wall_start = time.perf_counter()
    left_rgb = [load_rgb(path) for path in request.left_image_paths]
    right_rgb = [load_rgb(path) for path in request.right_image_paths]
    if any(image.shape[0] != request.runtime_height or image.shape[1] != request.runtime_width
           for image in left_rgb + right_rgb):
        return failed(["rectified_input_size_does_not_match_the_runtime_size"])
    if len(left_rgb) != window or len(right_rgb) != window:
        return failed(["left_right_window_length_mismatch"])
    images_read_ms = (time.perf_counter() - wall_start) * 1000.0

    left_tensor = to_tensor_batch(left_rgb, padded_width, padded_height).to(device)
    right_tensor = to_tensor_batch(right_rgb, padded_width, padded_height).to(device)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    with torch.no_grad():
        if request.matcher_name == "igev":
            prediction = model(left_tensor, right_tensor, iters=iters, test_mode=True)
        else:
            prediction = model(
                left_tensor[None], right_tensor[None], iters=iters, test_mode=True
            )
    end_event.record()
    torch.cuda.synchronize()
    gpu_forward_ms = float(start_event.elapsed_time(end_event))
    peak_gpu_memory_mb = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))

    if request.matcher_name == "igev":
        raw = prediction[0, 0].detach().float().cpu().numpy()
    else:
        raw = prediction[int(request.target_index), 0, 0].detach().float().cpu().numpy()
    raw = crop_to_original(raw, request.runtime_width, request.runtime_height)

    sign_factor, sign_evidence, sign_reasons = verify_sign_convention(
        left_rgb[request.target_index],
        right_rgb[request.target_index],
        raw,
        EXPECTED_SIGN_FACTOR[request.matcher_name],
    )
    if sign_reasons:
        metadata = {
            "weight_source": str(weights_path),
            "weights_url": weights_url,
            "model_repo": repo_url,
            **sign_evidence,
        }
        return failed(sign_reasons, metadata)

    signed = (raw.astype(np.float32) * np.float32(sign_factor)).astype(np.float32)
    disparity = invalid_to_nan(signed, minimum_valid=1.0)
    valid_mask = np.isfinite(disparity).astype(np.uint8)

    disparity_path = Path(request.output_disparity_path)
    disparity_path.parent.mkdir(parents=True, exist_ok=True)
    valid_mask_path = disparity_path.with_name(disparity_path.stem + "_valid_mask.npy")
    # The measured interval ends here, immediately before the disparity is written.
    cpu_worker_wall_ms = (time.perf_counter() - wall_start) * 1000.0
    np.save(disparity_path, disparity)
    np.save(valid_mask_path, valid_mask * 255)

    finite = np.isfinite(disparity)
    metadata = {
        "model_repo": repo_url,
        "weights_url": weights_url,
        "weight_source": str(weights_path),
        "weight_size_bytes": int(Path(weights_path).stat().st_size),
        "model_settings": model_settings,
        "iterations": int(iters),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "input_height": int(request.runtime_height),
        "input_width": int(request.runtime_width),
        "padding": {
            "mode": "constant_zero_bottom_right_only",
            "padded_height": int(padded_height),
            "padded_width": int(padded_width),
            "pad_bottom_px": int(padded_height - request.runtime_height),
            "pad_right_px": int(padded_width - request.runtime_width),
        },
        "resize_performed": False,
        "temporal_window_frames": int(window_info["temporal_window_frames"]),
        "target_position_in_window": int(window_info["target_position_in_window"]),
        "future_lookahead_frames": int(window_info["future_lookahead_frames"]),
        "realtime_compatible": bool(window_info["realtime_compatible"]),
        "warmup_request": bool(request.warmup),
        "model_load_ms": float(model_load_ms),
        "image_read_ms": float(images_read_ms),
        "sign_factor_applied": float(sign_factor),
        **sign_evidence,
        "valid_fraction": float(valid_mask.mean()),
        "valid_pixel_count": int(finite.sum()),
        "disparity_min_px": None if not finite.any() else float(disparity[finite].min()),
        "disparity_median_px": None if not finite.any() else float(np.median(disparity[finite])),
        "disparity_max_px": None if not finite.any() else float(disparity[finite].max()),
        "fraction_beyond_frozen_sgbm_search_range": (
            None if not finite.any()
            else float((disparity[finite] > float(SGBM_NUM_DISPARITIES - 1)).mean())
        ),
        "sgbm_search_range_px": [0, int(SGBM_NUM_DISPARITIES - 1)],
        **load_notes,
    }
    return LearnedStereoResult(
        matcher_name=request.matcher_name,
        disparity_path=str(disparity_path),
        valid_mask_path=str(valid_mask_path),
        original_width=int(request.runtime_width),
        original_height=int(request.runtime_height),
        padded_width=int(padded_width),
        padded_height=int(padded_height),
        target_index=int(request.target_index),
        temporal_window_frames=int(window_info["temporal_window_frames"]),
        future_lookahead_frames=int(window_info["future_lookahead_frames"]),
        gpu_forward_ms=gpu_forward_ms,
        cpu_worker_wall_ms=cpu_worker_wall_ms,
        peak_gpu_memory_mb=peak_gpu_memory_mb,
        status="ok",
        reasons=(),
        metadata=metadata,
    )


def main() -> int:
    args = parse_args()
    request = read_request(args.request)
    result_path = Path(args.result)
    try:
        result = run_request(args, request)
    except Exception as error:  # noqa: BLE001 - a structured failure beats a missing file
        buffer = io.StringIO()
        traceback.print_exc(file=buffer)
        result = LearnedStereoResult(
            matcher_name=request.matcher_name,
            disparity_path=str(request.output_disparity_path),
            valid_mask_path=str(Path(request.output_disparity_path).with_name("valid_mask.npy")),
            original_width=int(request.runtime_width),
            original_height=int(request.runtime_height),
            padded_width=0,
            padded_height=0,
            target_index=int(request.target_index),
            temporal_window_frames=len(request.left_image_paths),
            future_lookahead_frames=max(0, len(request.left_image_paths) - 1 - int(request.target_index)),
            gpu_forward_ms=None,
            cpu_worker_wall_ms=None,
            peak_gpu_memory_mb=None,
            status="failed",
            reasons=(f"worker_exception:{type(error).__name__}:{error}",),
            metadata={"traceback_tail": buffer.getvalue().splitlines()[-12:]},
        )
    write_result(result_path, result)
    declared = Path(request.output_metadata_path)
    if declared.resolve() != result_path.resolve():
        try:
            write_result(declared, result)
        except Exception:  # noqa: BLE001 - the --result file is authoritative
            pass
    print(json.dumps({"status": result.status, "reasons": list(result.reasons)}, ensure_ascii=False))
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
