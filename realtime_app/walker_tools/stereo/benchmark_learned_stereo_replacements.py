#!/usr/bin/env python3
"""Compare IGEV-Stereo and DynamicStereo against the frozen SGBM ground pipeline.

Only the "left/right images -> disparity" step changes.  Everything downstream is
inherited from task-01 and is called, not re-implemented:

* the mask-directed local fisheye rectification, the semantic masks, the
  calibrated triangulation rule, the depth/photometric/left-right gates, the
  three plane fitters and the shared cross-region gate all come from the same
  frozen code path (``observe_local_ground_semantic_stereo``,
  ``observe_local_ground_semantic_stereo_lr_direction_control`` and
  ``benchmark_ground_walker_reconstruction`` imported as a module);
* the SGBM comparison rows are *read* from the archived task-01 output; SGBM is
  never re-run here.

The learned matchers run in separate processes in their own conda environments
(``realtime_app/.model_envs/...``) because ``torch`` and this host's Anaconda MKL
NumPy cannot share one interpreter.  This process never imports torch.

Timing rules
------------

Each combination reports the module timing required by the protocol plus a
single comparable estimate::

    estimated_direct_pipeline_ms = mask_load_ms + rectification_ms
        + stereo_gpu_forward_ms + stereo_gpu_reverse_forward_ms
        + candidate_filter_ms + triangulation_ms + plane_fit_ms

The reverse field is included because the frozen chain needs it for the
left/right consistency gate, and the SGBM reference row already contains both of
its disparity passes inside ``rectification_and_disparity_ms``.  Cross-source
``total_cpu_wall_ms`` from task-01 is deliberately not used for ordering.
"""

from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
from walker_tools._compat import LEGACY_TOOLS as _tool_legacy_tools
_tool_prepare_imports()

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np


TOOLS_ROOT = _tool_legacy_tools
PROJECT_ROOT = TOOLS_ROOT.parents[1]
REALTIME_ROOT = PROJECT_ROOT / "realtime_app"

from pose_app.benchmark_timing import TimingCollector  # noqa: E402
from pose_app.calibration import StereoCalibration  # noqa: E402
from pose_app.learned_stereo_protocol import (  # noqa: E402
    LearnedStereoRequest,
    flip_disparity_for_reverse_field,
    read_result,
    validate_disparity,
    write_request,
)
import walker_tools.scene.benchmark_ground_walker_reconstruction as frozen  # noqa: E402
import walker_tools.scene.observe_local_ground_semantic_stereo as strict  # noqa: E402
import walker_tools.scene.observe_local_ground_semantic_stereo_lr_direction_control as corrected  # noqa: E402


SCHEMA_VERSION = "learned_stereo_replacement_benchmark_v1"
LEARNED_MATCHERS = ("igev", "dynamicstereo")
ALL_MATCHERS = ("sgbm",) + LEARNED_MATCHERS
MAIN_METHODS = frozen.MAIN_METHODS
REGION_METHOD = frozen.REGION_METHOD
CONFIDENCE_BOUNDARY = frozen.CONFIDENCE_BOUNDARY
SEMANTIC_SOURCES = ("manual_floor", "mask2former_floor")
SGBM_BASELINE_DIR_NAME = "G20260912_modular_ground_benchmark_v1"

# Extra (documented) timing keys next to the nine required ones.
EXTRA_TIMING_KEYS = (
    "stereo_gpu_reverse_forward_ms",
    "stereo_cpu_worker_reverse_wall_ms",
    "evidence_ms",
    "stereo_io_write_ms",
    "window_rectification_ms",
    "estimated_direct_pipeline_with_worker_wall_ms",
)
REQUIRED_TIMING_KEYS = (
    "mask_load_ms",
    "rectification_ms",
    "stereo_gpu_forward_ms",
    "stereo_cpu_worker_wall_ms",
    "stereo_transfer_ms",
    "candidate_filter_ms",
    "triangulation_ms",
    "plane_fit_ms",
    "region_consensus_ms",
    "estimated_direct_pipeline_ms",
)

#: DynamicStereo's published usage: five frames, target is the first one.
DYNAMIC_WINDOW_FRAMES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--left-dir", type=Path, required=True)
    parser.add_argument("--right-dir", type=Path, required=True)
    parser.add_argument("--left-label-dir", type=Path, required=True)
    parser.add_argument("--right-label-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--frozen-baseline-dir", type=Path, required=True)
    parser.add_argument("--sgbm-baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workspace-dir", type=Path)
    parser.add_argument("--auto-floor-left-dir", type=Path)
    parser.add_argument("--auto-floor-right-dir", type=Path)
    parser.add_argument("--auto-floor-name", default="mask2former_floor")
    parser.add_argument("--igev-python", type=Path, default=REALTIME_ROOT / ".model_envs" / "igev" / "python.exe")
    parser.add_argument("--dynamicstereo-python", type=Path,
                        default=REALTIME_ROOT / ".model_envs" / "dynamicstereo" / "python.exe")
    parser.add_argument("--igev-repo", type=Path, default=PROJECT_ROOT / "external_models" / "IGEV" / "IGEV-Stereo")
    parser.add_argument("--igev-weights", type=Path,
                        default=PROJECT_ROOT / "external_models" / "weights" / "igev" / "sceneflow" / "sceneflow.pth")
    parser.add_argument("--dynamicstereo-repo", type=Path, default=PROJECT_ROOT / "external_models" / "dynamic_stereo")
    parser.add_argument("--dynamicstereo-weights", type=Path,
                        default=PROJECT_ROOT / "external_models" / "weights" / "dynamic_stereo" / "dynamic_stereo_sf.pth")
    parser.add_argument("--matchers", default="igev,dynamicstereo")
    parser.add_argument("--maximum-pairs", type=int, default=0)
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="reuse already completed worker results inside an existing output directory",
    )
    parser.add_argument("--visualization-pairs", default="0,160,320")
    return parser.parse_args()


class RunLog:
    """Append-only per-matcher run log written into the experiment directory.

    The learned matchers are two separate model runs; each gets its own log file
    so the stdout of a run can be inspected per model even when both are executed
    by one command.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8", newline="\n")

    def write(self, message: str) -> None:
        self._handle.write(message + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


# --------------------------------------------------------------------------------------
# frozen-input readers
# --------------------------------------------------------------------------------------

def load_sgbm_baseline(baseline_dir: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Read the archived task-01 SGBM result. Read-only by construction."""
    summary = json.loads((baseline_dir / "summary.json").read_text(encoding="utf-8"))
    records: dict[int, dict[str, Any]] = {}
    with (baseline_dir / "frame_records.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                records[int(row["pair_id"])] = row
    return summary, records


def sgbm_combination_rows(
    summary: dict[str, Any], records: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    """The six SGBM rows (2 semantic sources x 3 plane fitters), read not recomputed."""
    rows: list[dict[str, Any]] = []
    for pair_id in sorted(records):
        for source in records[pair_id]["floor_sources"]:
            for method in MAIN_METHODS:
                entry = source["methods"][method]
                timing_ms = entry["timing_ms"]
                estimated = (
                    timing_ms["mask_load_ms"]
                    + timing_ms["rectification_and_disparity_ms"]
                    + timing_ms["stereo_candidate_filter_ms"]
                    + timing_ms["triangulation_ms"]
                    + timing_ms["plane_fit_ms"]
                )
                rows.append({
                    "pair_id": int(pair_id),
                    "warmup_frame": bool(records[pair_id].get("warmup_frame", False)),
                    "matcher": "sgbm",
                    "plane_method": method,
                    "semantic_source": source["source_name"],
                    "semantic_evidence": entry["evidence"]["semantic_evidence"],
                    "stereo_evidence": entry["evidence"]["stereo_evidence"],
                    "plane_evidence": entry["evidence"]["plane_evidence"],
                    "cross_region_evidence": entry["evidence"]["cross_region_evidence"],
                    "timing": {
                        "mask_load_ms": timing_ms["mask_load_ms"],
                        "rectification_ms": timing_ms["rectification_and_disparity_ms"],
                        "stereo_gpu_forward_ms": None,
                        "stereo_cpu_worker_wall_ms": None,
                        "stereo_transfer_ms": None,
                        "candidate_filter_ms": timing_ms["stereo_candidate_filter_ms"],
                        "triangulation_ms": timing_ms["triangulation_ms"],
                        "plane_fit_ms": timing_ms["plane_fit_ms"],
                        "region_consensus_ms": timing_ms["region_consensus_ms"],
                        "estimated_direct_pipeline_ms": estimated,
                    },
                    "confidence_boundary": CONFIDENCE_BOUNDARY,
                    "timing_note": (
                        "sgbm row read from the archived task-01 output; both SGBM disparity passes "
                        "(forward and the corrected reverse field) are inside rectification_ms, and no "
                        "separate GPU/worker timing exists because SGBM runs in this process"
                    ),
                })
    return rows


# --------------------------------------------------------------------------------------
# rectified local views (verbatim rectification block of the frozen tool)
# --------------------------------------------------------------------------------------

def build_rectified_views(
    left_upright: np.ndarray,
    right_upright: np.ndarray,
    left_mask_upright: np.ndarray,
    right_mask_upright: np.ndarray,
    calibration: StereoCalibration,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Mask-directed rectification exactly as the frozen tool performs it.

    The lines below are the rectification half of
    ``benchmark_ground_walker_reconstruction.stereo_candidates`` copied verbatim;
    the tool's own equivalence check (``verify_rectification_equivalence.py``)
    confirms bit equality against that function.
    """
    runtime_size = (int(parameters["runtime_width"]), int(parameters["runtime_height"]))
    left_seed = strict.mask_seed_upright(left_mask_upright)
    right_seed = strict.mask_seed_upright(right_mask_upright)
    if left_seed is None or right_seed is None:
        return {"status": "unavailable", "reasons": ["empty_semantic_mask"]}
    left_raw, right_raw = strict.inverse_upright(left_upright, right_upright)
    raw_size = (left_raw.shape[1], left_raw.shape[0])
    scale = np.asarray((runtime_size[0] / raw_size[0], runtime_size[1] / raw_size[1]))
    upright_size = (left_upright.shape[1], left_upright.shape[0])
    left_seed_raw = strict.upright_point_to_raw(left_seed, "left", upright_size) * scale
    right_seed_raw = strict.upright_point_to_raw(right_seed, "right", upright_size) * scale
    rectification = strict.make_mask_directed_rectification(
        calibration, left_seed_raw, right_seed_raw, runtime_size, float(parameters["virtual_focal_px"])
    )
    left_raw = cv2.resize(left_raw, runtime_size, interpolation=cv2.INTER_AREA)
    right_raw = cv2.resize(right_raw, runtime_size, interpolation=cv2.INTER_AREA)
    left_mask_raw = cv2.resize(
        strict.rotate_mask_to_raw(left_mask_upright, "left"), runtime_size, interpolation=cv2.INTER_NEAREST
    )
    right_mask_raw = cv2.resize(
        strict.rotate_mask_to_raw(right_mask_upright, "right"), runtime_size, interpolation=cv2.INTER_NEAREST
    )
    left_local = cv2.remap(left_raw, rectification.left_map_x, rectification.left_map_y, cv2.INTER_LINEAR)
    right_local = cv2.remap(right_raw, rectification.right_map_x, rectification.right_map_y, cv2.INTER_LINEAR)
    left_mask_local = cv2.remap(
        left_mask_raw, rectification.left_map_x, rectification.left_map_y, cv2.INTER_NEAREST
    )
    right_mask_local = cv2.remap(
        right_mask_raw, rectification.right_map_x, rectification.right_map_y, cv2.INTER_NEAREST
    )
    return {
        "status": "candidate",
        "reasons": [],
        "rectification": rectification,
        "left_raw": left_raw,
        "right_raw": right_raw,
        "left_local": left_local,
        "right_local": right_local,
        "left_mask_local": left_mask_local,
        "right_mask_local": right_mask_local,
        "left_mask_for_matching": left_mask_local,
        "right_mask_for_matching": right_mask_local,
    }


def rectify_extra_frame(
    left_upright: np.ndarray, right_upright: np.ndarray, rectification: Any, parameters: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Rectify one extra temporal-window frame in the *target* frame's virtual camera.

    The mask-directed rectification defines the virtual camera from the target
    frame's masks.  Applying that same rectification to the window frames keeps a
    single consistent virtual camera across the window, which is what a temporal
    stereo model assumes.  Only the target frame's rectified pair is the one the
    frozen pipeline uses downstream.
    """
    runtime_size = (int(parameters["runtime_width"]), int(parameters["runtime_height"]))
    left_raw, right_raw = strict.inverse_upright(left_upright, right_upright)
    left_raw = cv2.resize(left_raw, runtime_size, interpolation=cv2.INTER_AREA)
    right_raw = cv2.resize(right_raw, runtime_size, interpolation=cv2.INTER_AREA)
    left_local = cv2.remap(left_raw, rectification.left_map_x, rectification.left_map_y, cv2.INTER_LINEAR)
    right_local = cv2.remap(right_raw, rectification.right_map_x, rectification.right_map_y, cv2.INTER_LINEAR)
    return left_local, right_local


# --------------------------------------------------------------------------------------
# worker exchange
# --------------------------------------------------------------------------------------

def write_image(path: Path, image: np.ndarray) -> None:
    """Write a PNG through an encoded buffer.

    ``cv2.imwrite`` refuses non-ASCII paths on Windows (it fails silently and
    returns False), so the encoded buffer is written by Python instead.  The
    archive path is ASCII, but a scratch or temporary workspace need not be.
    """
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"could not encode PNG for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.tobytes())


def write_exchange_images(
    directory: Path, images: dict[str, np.ndarray]
) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name, image in images.items():
        path = directory / f"{name}.png"
        write_image(path, image)
        written[name] = str(path.resolve())
    return written


def run_worker(
    python: Path,
    worker_script: Path,
    request_path: Path,
    result_path: Path,
    log_path: Path,
    extra_args: list[str],
) -> dict[str, Any]:
    command = [
        str(python), str(worker_script),
        "--request", str(request_path.resolve()),
        "--result", str(result_path.resolve()),
        *extra_args,
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True)
    wall_ms = (time.perf_counter() - started) * 1000.0
    log_path.write_text(
        "command: " + json.dumps(command, ensure_ascii=False, indent=2) + "\n"
        f"returncode: {completed.returncode}\n"
        "----- stdout -----\n" + (completed.stdout or "") + "\n"
        "----- stderr -----\n" + (completed.stderr or "") + "\n",
        encoding="utf-8",
    )
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "process_wall_ms": float(wall_ms),
        "log_path": str(log_path.resolve()),
    }


def learned_candidate_from_disparity(
    disparity_forward: np.ndarray,
    disparity_reverse_flipped: np.ndarray,
    views: dict[str, Any],
    parameters: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    """Reproduce the frozen candidate stage with a learned disparity field.

    The grids, the boolean gates and the triangulation call are the frozen ones;
    only the two disparity fields are supplied by the worker instead of SGBM.
    Returns the candidate bundle plus the measured split between the shared
    filter grids and the frozen triangulation call.
    """
    left_local = views["left_local"]
    right_local = views["right_local"]
    left_mask = views["left_mask_for_matching"]
    right_mask = views["right_mask_for_matching"]
    rectification = views["rectification"]
    forward = disparity_forward
    reverse = flip_disparity_for_reverse_field(disparity_reverse_flipped)
    filter_started = time.perf_counter()
    height, width = forward.shape
    yy, xx = np.mgrid[0:height, 0:width]
    safe_forward = np.where(np.isfinite(forward), forward, np.float32(0.0))
    partner_x_grid = np.rint(xx.astype(np.float32) - safe_forward).astype(np.int32)
    partner_clipped = np.clip(partner_x_grid, 0, width - 1)
    in_bounds_grid = (partner_x_grid >= 0) & (partner_x_grid < width)
    left_semantic = left_mask > 0
    forward_valid = left_semantic & (forward > 1.0) & in_bounds_grid
    right_semantic = right_mask[yy, partner_clipped] > 0
    right_mask_valid = forward_valid & right_semantic
    reverse_at_partner = reverse[yy, partner_clipped]
    reverse_valid = right_mask_valid & (reverse_at_partner > 1.0)
    lr_valid = reverse_valid & (np.abs(forward - reverse_at_partner) <= float(parameters["lr_consistency_px"]))
    filter_ms = (time.perf_counter() - filter_started) * 1000.0
    triangulation_started = time.perf_counter()
    with np.errstate(invalid="ignore"):
        candidate = corrected.reconstruct(
            left_local, right_local, left_mask, right_mask, rectification,
            forward, reverse, int(parameters["num_disparities"]), float(parameters["lr_consistency_px"]),
        )
    triangulation_ms = (time.perf_counter() - triangulation_started) * 1000.0
    left_pixels = candidate["left_pixels"]
    right_pixels = candidate["right_pixels"]
    # Deterministic texture/photometric candidate score, copied verbatim from the
    # frozen candidate stage so the sparse and IRLS fitters receive identical input.
    scoring_started = time.perf_counter()
    left_gray = cv2.cvtColor(left_local, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_local, cv2.COLOR_BGR2GRAY)
    gradient = cv2.magnitude(
        cv2.Sobel(left_gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(left_gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    if len(left_pixels):
        texture = gradient[left_pixels[:, 1], left_pixels[:, 0]].astype(np.float64)
        texture /= max(float(np.percentile(texture, 90)), 1.0)
        texture = np.clip(texture, 0.0, 1.0)
        photo = np.abs(
            left_gray[left_pixels[:, 1], left_pixels[:, 0]].astype(np.float64)
            - right_gray[right_pixels[:, 1], right_pixels[:, 0]].astype(np.float64)
        )
        confidence = (0.1 + 0.9 * texture) * np.exp(-photo / 20.0)
    else:
        photo = np.empty(0, dtype=np.float64)
        confidence = np.empty(0, dtype=np.float64)
    filter_ms += (time.perf_counter() - scoring_started) * 1000.0
    return {
        "status": "candidate",
        "reasons": [],
        **candidate,
        "confidence": confidence,
        "photometric_difference": photo,
        "left_local": left_local,
        "right_local": right_local,
        "left_mask_local": views["left_mask_local"],
        "right_mask_local": views["right_mask_local"],
        "left_mask_for_matching": left_mask,
        "right_mask_for_matching": right_mask,
        "mask_dilation_px": 0,
        "rectification": rectification,
        "matching_funnel": {
            "left_mask_pixels": int(left_semantic.sum()),
            "forward_valid": int(forward_valid.sum()),
            "right_mask_valid": int(right_mask_valid.sum()),
            "reverse_valid": int(reverse_valid.sum()),
            "lr_consistent": int(lr_valid.sum()),
            "final_candidates": int(len(candidate["points"])),
        },
        "disparity_field_statistics": {
            "finite_fraction": float(np.isfinite(forward).mean()),
            "median_px": None if not np.isfinite(forward).any() else float(np.median(forward[np.isfinite(forward)])),
            "fraction_beyond_frozen_sgbm_search_range": (
                None if not np.isfinite(forward).any()
                else float((forward[np.isfinite(forward)] > float(parameters["num_disparities"]) - 1).mean())
            ),
        },
    }, {"candidate_filter_ms": filter_ms, "triangulation_ms": triangulation_ms}


def aggregate_evidence_blocks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def median(values: list[float]) -> float | None:
        present = [float(value) for value in values if value is not None]
        return float(statistics.median(present)) if present else None

    semantic = [row["semantic_evidence"] for row in rows]
    stereo = [row["stereo_evidence"] for row in rows]
    plane = [row["plane_evidence"] for row in rows]
    cross = [row["cross_region_evidence"] for row in rows]
    return {
        "semantic_source": rows[0]["semantic_source"],
        "semantic_evidence": {
            "source": semantic[0].get("source"),
            "semantic_identity_evidence": semantic[0].get("semantic_identity_evidence"),
            "mask_status": semantic[0].get("mask_status"),
            "manual_audit_available": bool(semantic[0].get("manual_audit_available")),
            "frames": len(rows),
            "median_mask_area_fraction": median([item.get("mask_area_fraction") for item in semantic]),
        },
        "stereo_evidence": {
            "frames_with_stereo_measurement": int(sum(
                1 for item in stereo if item.get("candidate_pixels") is not None
            )),
            "median_candidate_pixels": median([item.get("candidate_pixels") for item in stereo]),
            "median_forward_disparity_valid_fraction": median(
                [item.get("forward_disparity_valid_fraction") for item in stereo]
            ),
            "median_right_mask_correspondence_fraction": median(
                [item.get("right_mask_correspondence_fraction") for item in stereo]
            ),
            "median_lr_consistency_fraction": median([item.get("lr_consistency_fraction") for item in stereo]),
            "median_positive_depth_fraction": median([item.get("positive_depth_fraction") for item in stereo]),
            "median_reprojection_px": median([item.get("median_reprojection_px") for item in stereo]),
            "median_p95_reprojection_px": median([item.get("p95_reprojection_px") for item in stereo]),
        },
        "plane_evidence": {
            "frames": len(rows),
            "plane_candidate_frames": int(sum(item.get("status") == "candidate" for item in plane)),
            "plane_unavailable_frames": int(sum(item.get("status") == "unavailable" for item in plane)),
            "median_inlier_fraction": median([item.get("inlier_fraction") for item in plane]),
            "median_residual_mm": median([item.get("median_residual_mm") for item in plane]),
            "median_coverage_fraction": median([item.get("coverage_fraction") for item in plane]),
        },
        "cross_region_evidence": {
            "pass_frames": int(sum(item.get("status") == "pass" for item in cross)),
            "fail_frames": int(sum(item.get("status") == "fail" for item in cross)),
            "unavailable_frames": int(sum(item.get("status") == "unavailable" for item in cross)),
            "median_region_count": median([item.get("region_count") for item in cross]),
            "median_normal_spread_deg": median([item.get("normal_spread_deg") for item in cross]),
            "median_offset_spread_mm": median([item.get("offset_spread_mm") for item in cross]),
        },
        "confidence_boundary": CONFIDENCE_BOUNDARY,
    }


def timing_summary_for(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    collector = TimingCollector(gpu_synchronize=False)
    for row in rows:
        for key, value in row["timing"].items():
            if value is None:
                continue
            collector.record_ms(key, float(value))
    return collector.summary()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    matchers = tuple(item.strip() for item in args.matchers.split(",") if item.strip())
    unknown = [item for item in matchers if item not in LEARNED_MATCHERS]
    if unknown:
        raise ValueError(f"unsupported matchers: {unknown}")
    parameters, _ = corrected.load_frozen(args.frozen_baseline_dir)
    runtime_size = (int(parameters["runtime_width"]), int(parameters["runtime_height"]))
    calibration = StereoCalibration.load(args.calibration).for_runtime_sizes(runtime_size, runtime_size)
    names = frozen.selected_names(args)
    if not names:
        raise RuntimeError("no paired LabelMe annotations found")
    pair_ids = [frozen.pair_id(Path(name)) for name in names]
    sgbm_summary, sgbm_records = load_sgbm_baseline(args.sgbm_baseline_dir)
    missing_sgbm = [pid for pid in pair_ids if pid not in sgbm_records]
    if missing_sgbm:
        raise RuntimeError(f"archived SGBM baseline is missing pairs {missing_sgbm}")

    workspace = args.workspace_dir or (args.output_dir / "stereo_io")
    worker_script = TOOLS_ROOT / "run_learned_stereo_worker.py"
    python_for = {"igev": args.igev_python, "dynamicstereo": args.dynamicstereo_python}
    worker_extra_args = {
        "igev": ["--igev-repo", str(args.igev_repo), "--igev-weights", str(args.igev_weights)],
        "dynamicstereo": [
            "--dynamicstereo-repo", str(args.dynamicstereo_repo),
            "--dynamicstereo-weights", str(args.dynamicstereo_weights),
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "visualizations").mkdir(exist_ok=True)
    run_logs = {matcher: RunLog(args.output_dir / f"run_stdout_{matcher}.txt") for matcher in matchers}
    for matcher, log in run_logs.items():
        log.write(f"# learned matcher run log: {matcher}")
        log.write(f"# worker interpreter: {python_for[matcher]}")
        log.write(f"# output directory: {args.output_dir.resolve()}")
        log.write(f"# pairs: {pair_ids}")

    records: list[dict[str, Any]] = []
    worker_runs: list[dict[str, Any]] = []
    frame_reports: list[dict[str, Any]] = []
    visualization_targets = {
        int(item) for item in args.visualization_pairs.split(",") if item.strip()
    } if not args.no_visualization else set()

    for index, name in enumerate(names):
        pair_id = frozen.pair_id(Path(name))
        frame_start = time.perf_counter()
        left = cv2.imread(str(args.left_dir / name), cv2.IMREAD_COLOR)
        right = cv2.imread(str(args.right_dir / name), cv2.IMREAD_COLOR)
        if left is None or right is None or left.shape[:2] != right.shape[:2]:
            raise RuntimeError(f"cannot read equal-size stereo pair {name}")
        mask_start = time.perf_counter()
        left_masks = frozen.load_manual_masks(args.left_label_dir / Path(name).with_suffix(".json"), left.shape[:2])
        right_masks = frozen.load_manual_masks(args.right_label_dir / Path(name).with_suffix(".json"), right.shape[:2])
        manual_mask_area = float((left_masks["floor_eligible"] > 0).mean())
        manual_mask_load_ms = (time.perf_counter() - mask_start) * 1000.0
        auto_left = auto_right = None
        auto_mask_area = None
        auto_mask_load_ms = None
        if args.auto_floor_left_dir is not None and args.auto_floor_right_dir is not None:
            auto_start = time.perf_counter()
            auto_left = frozen.load_binary(args.auto_floor_left_dir / name, left.shape[:2])
            auto_right = frozen.load_binary(args.auto_floor_right_dir / name, right.shape[:2])
            auto_mask_area = float((auto_left > 0).mean())
            auto_mask_load_ms = (time.perf_counter() - auto_start) * 1000.0

        source_specs = [
            {
                "source_name": "manual_floor",
                "identity_evidence": "manually_audited",
                "left_mask": left_masks["floor_eligible"],
                "right_mask": right_masks["floor_eligible"],
                "mask_area_fraction": manual_mask_area,
                "mask_load_ms": manual_mask_load_ms,
            }
        ]
        if auto_left is not None and auto_right is not None:
            source_specs.append({
                "source_name": args.auto_floor_name,
                "identity_evidence": "provided_unvalidated",
                "left_mask": auto_left,
                "right_mask": auto_right,
                "mask_area_fraction": auto_mask_area,
                "mask_load_ms": auto_mask_load_ms,
            })

        # Per source: frozen rectification of the target frame.
        rectified: dict[str, Any] = {}
        rectification_ms: dict[str, float] = {}
        for spec in source_specs:
            started = time.perf_counter()
            views = build_rectified_views(
                left, right, spec["left_mask"], spec["right_mask"], calibration, parameters
            )
            rectification_ms[spec["source_name"]] = (time.perf_counter() - started) * 1000.0
            rectified[spec["source_name"]] = views

        for matcher in matchers:
            window = DYNAMIC_WINDOW_FRAMES if matcher == "dynamicstereo" else 1
            window_ids = [pair_id + offset for offset in range(window)]
            window_names = [f"pair_{value:04d}.png" for value in window_ids]
            window_images: list[tuple[np.ndarray, np.ndarray]] = []
            extra_available = True
            for window_name in window_names:
                extra_left = cv2.imread(str(args.left_dir / window_name), cv2.IMREAD_COLOR)
                extra_right = cv2.imread(str(args.right_dir / window_name), cv2.IMREAD_COLOR)
                if extra_left is None or extra_right is None:
                    extra_available = False
                    break
                window_images.append((extra_left, extra_right))
            if not extra_available:
                reason = f"temporal_window_frames_missing:{window_names}"
                for spec in source_specs:
                    for method in MAIN_METHODS:
                        records.append(unavailable_record(
                            pair_id, name, index, matcher, method, spec, reason, parameters
                        ))
                continue

            for spec in source_specs:
                source_name = spec["source_name"]
                views = rectified[source_name]
                if views["status"] != "candidate":
                    for method in MAIN_METHODS:
                        records.append(unavailable_record(
                            pair_id, name, index, matcher, method, spec, "empty_semantic_mask", parameters
                        ))
                    continue
                case_dir = workspace / matcher / source_name / f"pair_{pair_id:04d}"
                window_start = time.perf_counter()
                rectified_window: list[tuple[np.ndarray, np.ndarray]] = []
                for window_index, (extra_left, extra_right) in enumerate(window_images):
                    if window_index == 0:
                        # The target frame must be the frozen rectification itself.
                        left_local = views["left_local"]
                        right_local = views["right_local"]
                    else:
                        left_local, right_local = rectify_extra_frame(
                            extra_left, extra_right, views["rectification"], parameters
                        )
                    rectified_window.append((left_local, right_local))
                images: dict[str, np.ndarray] = {}
                for window_index, (left_local, right_local) in enumerate(rectified_window):
                    images[f"w{window_index}_left"] = left_local
                    images[f"w{window_index}_right"] = right_local
                    images[f"w{window_index}_left_flip"] = np.fliplr(left_local).copy()
                    images[f"w{window_index}_right_flip"] = np.fliplr(right_local).copy()
                io_start = time.perf_counter()
                window_rectification_ms = (io_start - window_start) * 1000.0
                written = write_exchange_images(case_dir, images)
                stereo_io_write_ms = (time.perf_counter() - io_start) * 1000.0

                forward_left = [written[f"w{i}_left"] for i in range(window)]
                forward_right = [written[f"w{i}_right"] for i in range(window)]
                reverse_left = [written[f"w{i}_right_flip"] for i in range(window)]
                reverse_right = [written[f"w{i}_left_flip"] for i in range(window)]

                forward_request_path = case_dir / "request_forward.json"
                reverse_request_path = case_dir / "request_reverse.json"
                forward_result_path = case_dir / "result_forward.json"
                reverse_result_path = case_dir / "result_reverse.json"
                disparity_forward_path = case_dir / "disparity_forward.npy"
                disparity_reverse_path = case_dir / "disparity_reverse_flipped.npy"
                warmup = bool(index == 0)
                write_request(forward_request_path, LearnedStereoRequest(
                    matcher_name=matcher,
                    left_image_paths=tuple(forward_left),
                    right_image_paths=tuple(forward_right),
                    output_disparity_path=str(disparity_forward_path),
                    output_metadata_path=str(case_dir / "result_forward.json"),
                    runtime_width=runtime_size[0],
                    runtime_height=runtime_size[1],
                    target_index=0,
                    warmup=warmup,
                ))
                write_request(reverse_request_path, LearnedStereoRequest(
                    matcher_name=matcher,
                    left_image_paths=tuple(reverse_left),
                    right_image_paths=tuple(reverse_right),
                    output_disparity_path=str(disparity_reverse_path),
                    output_metadata_path=str(case_dir / "result_reverse.json"),
                    runtime_width=runtime_size[0],
                    runtime_height=runtime_size[1],
                    target_index=0,
                    warmup=warmup,
                ))
                reuse = False
                if args.resume and forward_result_path.exists() and reverse_result_path.exists():
                    try:
                        cached_forward = read_result(forward_result_path)
                        cached_reverse = read_result(reverse_result_path)
                    except Exception:  # noqa: BLE001 - a broken cache entry is simply re-run
                        reuse = False
                    else:
                        reuse = bool(
                            cached_forward.status == "ok" and cached_reverse.status == "ok"
                            and disparity_forward_path.exists() and disparity_reverse_path.exists()
                        )
                        if reuse:
                            run_logs[matcher].write(
                                f"pair={pair_id:04d} source={source_name} reused completed worker results "
                                f"(gpu_forward_ms={cached_forward.gpu_forward_ms}, "
                                f"gpu_reverse_ms={cached_reverse.gpu_forward_ms})"
                            )
                if not reuse:
                    forward_run = run_worker(
                        python_for[matcher], worker_script, forward_request_path, forward_result_path,
                        case_dir / "worker_forward.log", worker_extra_args[matcher],
                    )
                    reverse_run = run_worker(
                        python_for[matcher], worker_script, reverse_request_path, reverse_result_path,
                        case_dir / "worker_reverse.log", worker_extra_args[matcher],
                    )
                    log = run_logs[matcher]
                    log.write(
                        f"pair={pair_id:04d} source={source_name} worker forward rc={forward_run['returncode']} "
                        f"process_wall_ms={forward_run['process_wall_ms']:.1f} | reverse rc={reverse_run['returncode']} "
                        f"process_wall_ms={reverse_run['process_wall_ms']:.1f} | warmup_request={warmup}"
                    )
                    log.write(f"pair={pair_id:04d} source={source_name} worker_forward_log={forward_run['log_path']}")
                    log.write(f"pair={pair_id:04d} source={source_name} worker_reverse_log={reverse_run['log_path']}")
                else:
                    forward_run = {"reused": True, "log_path": str((case_dir / "worker_forward.log"))}
                    reverse_run = {"reused": True, "log_path": str((case_dir / "worker_reverse.log"))}
                worker_runs.append({
                    "pair_id": pair_id, "matcher": matcher, "semantic_source": source_name,
                    "warmup": warmup, "forward": forward_run, "reverse": reverse_run,
                })
                forward_result = read_result(forward_result_path) if forward_result_path.exists() else None
                reverse_result = read_result(reverse_result_path) if reverse_result_path.exists() else None
                failure_reasons: list[str] = []
                if forward_result is None or forward_result.status != "ok":
                    failure_reasons.append(
                        "forward_pass_failed:" + ("no_result_file" if forward_result is None
                                                  else ",".join(forward_result.reasons) or forward_result.status)
                    )
                if reverse_result is None or reverse_result.status != "ok":
                    failure_reasons.append(
                        "reverse_pass_failed:" + ("no_result_file" if reverse_result is None
                                                  else ",".join(reverse_result.reasons) or reverse_result.status)
                    )
                if failure_reasons:
                    for method in MAIN_METHODS:
                        records.append(unavailable_record(
                            pair_id, name, index, matcher, method, spec, ";".join(failure_reasons), parameters,
                            extra=worker_failure_details(forward_result, reverse_result),
                        ))
                    continue

                transfer_start = time.perf_counter()
                disparity_forward = validate_disparity(
                    np.load(disparity_forward_path), width=runtime_size[0], height=runtime_size[1]
                )
                disparity_reverse = validate_disparity(
                    np.load(disparity_reverse_path), width=runtime_size[0], height=runtime_size[1]
                )
                stereo_transfer_ms = (time.perf_counter() - transfer_start) * 1000.0

                candidate, candidate_timing = learned_candidate_from_disparity(
                    disparity_forward, disparity_reverse, views, parameters
                )
                candidate_filter_ms = candidate_timing["candidate_filter_ms"]
                triangulation_ms = candidate_timing["triangulation_ms"]

                method_result = frozen.floor_methods(
                    candidate, parameters, 20260912 + index,
                    calibration=calibration,
                    source_name=source_name,
                    identity_evidence=spec["identity_evidence"],
                    mask_area_fraction=spec["mask_area_fraction"],
                    module_times={},
                    module_reasons={},
                )
                common_timing = {
                    "mask_load_ms": spec["mask_load_ms"],
                    "rectification_ms": rectification_ms[source_name],
                    "stereo_gpu_forward_ms": forward_result.gpu_forward_ms,
                    "stereo_cpu_worker_wall_ms": forward_result.cpu_worker_wall_ms,
                    "stereo_transfer_ms": stereo_transfer_ms,
                    "candidate_filter_ms": candidate_filter_ms,
                    "triangulation_ms": triangulation_ms,
                    "stereo_gpu_reverse_forward_ms": reverse_result.gpu_forward_ms,
                    "stereo_cpu_worker_reverse_wall_ms": reverse_result.cpu_worker_wall_ms,
                    "stereo_io_write_ms": stereo_io_write_ms,
                    "window_rectification_ms": window_rectification_ms,
                }
                for method in MAIN_METHODS:
                    timing = dict(common_timing)
                    timing["plane_fit_ms"] = method_result["timing"][method]["plane_fit_ms"]
                    timing["region_consensus_ms"] = method_result["timing"][method]["region_consensus_ms"]
                    timing["evidence_ms"] = method_result["timing"][method]["evidence_ms"]
                    timing["estimated_direct_pipeline_ms"] = estimate_pipeline(timing, use_gpu=True)
                    timing["estimated_direct_pipeline_with_worker_wall_ms"] = estimate_pipeline(timing, use_gpu=False)
                    records.append({
                        "schema_version": SCHEMA_VERSION,
                        "pair_id": int(pair_id),
                        "frame_id": name,
                        "warmup_request": warmup,
                        "matcher": matcher,
                        "plane_method": method,
                        "semantic_source": source_name,
                        "status": "ok",
                        "temporal_window_frames": int(window),
                        "target_position_in_window": 0,
                        "future_lookahead_frames": int(window - 1),
                        "realtime_compatible": bool(window == 1),
                        "semantic_evidence": method_result["evidence"][method]["semantic_evidence"],
                        "stereo_evidence": method_result["evidence"][method]["stereo_evidence"],
                        "plane_evidence": method_result["evidence"][method]["plane_evidence"],
                        "cross_region_evidence": method_result["evidence"][method]["cross_region_evidence"],
                        "timing": timing,
                        "confidence_boundary": CONFIDENCE_BOUNDARY,
                    })
                frame_reports.append({
                    "pair_id": int(pair_id), "matcher": matcher, "semantic_source": source_name,
                    "warmup_request": warmup,
                    "stereo_candidate_count": int(len(candidate["points"])),
                    "matching_funnel": candidate["matching_funnel"],
                    "disparity_field_statistics": candidate["disparity_field_statistics"],
                    "forward_metadata": forward_result.metadata,
                    "reverse_metadata": reverse_result.metadata,
                    "process_wall_ms": {
                        "forward": forward_run.get("process_wall_ms"),
                        "reverse": reverse_run.get("process_wall_ms"),
                        "reused_from_earlier_run": bool(reuse),
                    },
                })
                if pair_id in visualization_targets:
                    write_visualization(
                        args.output_dir / "visualizations" / f"{matcher}_{source_name}_{name}",
                        views["left_local"], disparity_forward, matcher, pair_id,
                        float(np.isfinite(disparity_forward).mean()),
                        forward_result.gpu_forward_ms, forward_result.cpu_worker_wall_ms,
                    )

    # ---------------------------------------------------------------- per-matcher logs
    for matcher, log in run_logs.items():
        rows = [row for row in records if row["matcher"] == matcher]
        ok_rows = [row for row in rows if row.get("status") == "ok"]
        failed_rows = [row for row in rows if row.get("status") != "ok"]
        log.write(
            f"# matcher={matcher} records={len(rows)} ok={len(ok_rows)} unavailable={len(failed_rows)}"
        )
        for source_name in SEMANTIC_SOURCES:
            for method in MAIN_METHODS:
                subset = [
                    row for row in ok_rows
                    if row["semantic_source"] == source_name and row["plane_method"] == method
                ]
                candidates = sum(1 for row in subset if row["plane_evidence"]["status"] == "candidate")
                gates = sum(1 for row in subset if row["cross_region_evidence"]["status"] == "pass")
                log.write(
                    f"# matcher={matcher} source={source_name} method={method} frames={len(subset)} "
                    f"plane_candidates={candidates} cross_region_pass={gates}"
                )
        for row in failed_rows:
            log.write(
                f"# matcher={matcher} pair={row['pair_id']:04d} source={row['semantic_source']} "
                f"method={row['plane_method']} status={row.get('status')} reasons={row.get('reasons')}"
            )
        log.close()

    # ---------------------------------------------------------------- aggregation
    sgbm_rows = sgbm_combination_rows(sgbm_summary, sgbm_records)
    combinations: list[dict[str, Any]] = []
    for matcher in ALL_MATCHERS:
        for source_name in SEMANTIC_SOURCES:
            for method in MAIN_METHODS:
                if matcher == "sgbm":
                    rows = [
                        row for row in sgbm_rows
                        if row["semantic_source"] == source_name and row["plane_method"] == method
                    ]
                else:
                    rows = [
                        row for row in records
                        if row["matcher"] == matcher and row["semantic_source"] == source_name
                        and row["plane_method"] == method
                    ]
                if not rows:
                    continue
                measured = [row for row in rows if not row.get("warmup_frame", row.get("warmup_request", False))]
                combo = aggregate_evidence_blocks(rows)
                combo["matcher"] = matcher
                combo["plane_method"] = method
                combo["semantic_source"] = source_name
                combo["timing"] = {
                    "frames_in_percentiles": len(measured),
                    "warmup_excluded": len(rows) - len(measured),
                    **timing_summary_for(measured),
                }
                if matcher == "sgbm":
                    combo["timing_note"] = (
                        "read from the archived task-01 output; SGBM's own inference is inside "
                        "rectification_ms and there is no GPU/worker split"
                    )
                combos_all = [row for row in rows if not row.get("warmup_frame", row.get("warmup_request", False))]
                combo["timing_on_task01_comparable_frames"] = {
                    "frames_in_percentiles": len([
                        row for row in combos_all
                        if int(row["pair_id"]) not in task01_warmup_pair_ids()
                    ]),
                    **timing_summary_for([
                        row for row in combos_all if int(row["pair_id"]) not in task01_warmup_pair_ids()
                    ]),
                }
                combinations.append(combo)

    unavailable_rows: list[dict[str, Any]] = []
    for matcher in matchers:
        for source_name in SEMANTIC_SOURCES:
            for method in MAIN_METHODS:
                rows = [
                    row for row in records
                    if row["matcher"] == matcher and row["semantic_source"] == source_name
                    and row["plane_method"] == method and row.get("status") == "unavailable"
                ]
                if rows:
                    unavailable_rows.append({
                        "matcher": matcher, "semantic_source": source_name, "plane_method": method,
                        "frames": len(rows),
                        "reasons": sorted({reason for row in rows for reason in row.get("reasons", [])}),
                    })

    summary = {
        "schema_version": SCHEMA_VERSION,
        "pair_ids": [int(value) for value in pair_ids],
        "matchers": list(ALL_MATCHERS),
        "learned_matchers_run": list(matchers),
        "plane_methods_ranked": list(MAIN_METHODS),
        "shared_gate_method": REGION_METHOD,
        "semantic_sources": list(SEMANTIC_SOURCES),
        "sgbm_source": {
            "archive_dir": str(args.sgbm_baseline_dir.resolve()),
            "note": "SGBM rows are read from the archived task-01 output; SGBM is never re-run here",
        },
        "combinations": combinations,
        "unavailable_combinations": unavailable_rows,
        "dynamicstereo_offline_window": {
            "temporal_window_frames": DYNAMIC_WINDOW_FRAMES,
            "target_position_in_window": 0,
            "future_lookahead_frames": DYNAMIC_WINDOW_FRAMES - 1,
            "realtime_compatible": False,
        },
        "warmup_protocol": {
            "warmup_requests_per_matcher": 2,
            "warmup_request_definition": (
                "the forward and the reverse request of the first target frame (pair_0000) of each matcher"
            ),
            "excluded_from_percentiles": True,
            "task01_comparable_frame_set": "the same 10 frames task-01 timed (pair_0000 and pair_0040 excluded)",
        },
        "timing_formula": {
            "estimated_direct_pipeline_ms": (
                "mask_load_ms + rectification_ms + stereo_gpu_forward_ms + stereo_gpu_reverse_forward_ms "
                "+ candidate_filter_ms + triangulation_ms + plane_fit_ms"
            ),
            "why_reverse_pass_included": (
                "the frozen left/right consistency gate needs the rightward reverse field, and the SGBM "
                "reference row already contains both of its disparity passes inside rectification_ms"
            ),
            "excluded_from_estimate": (
                "process-isolated image exchange (.png write/read), model construction and checkpoint loading, "
                "and the offline 5-frame window rectification"
            ),
            "forbidden_ordering_key": "task-01 total_cpu_wall_ms is not comparable across sources and is not used",
        },
        "interpretation_boundary": (
            "Internal semantic, stereo, plane and cross-region evidence only. Semantic masks condition 2-D "
            "identity, not 3-D truth; a fitted plane with a small residual is not the physical floor. No "
            "physical ground accuracy, foot contact, support phase, step length or clinical claim is made. "
            "DynamicStereo is an offline 5-frame lookahead comparison and must not be read as a real-time "
            "single-frame latency."
        ),
    }
    with (args.output_dir / "frame_records.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "stereo_frame_reports.json").write_text(
        json.dumps({"items": frame_reports}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "worker_runs.json").write_text(
        json.dumps({"items": worker_runs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    timing_summary = {
        "schema_version": SCHEMA_VERSION + "_timing_v1",
        "warmup_protocol": summary["warmup_protocol"],
        "timing_formula": summary["timing_formula"],
        "timing_by_matcher_source_and_method": {
            f"{combo['matcher']}|{combo['semantic_source']}|{combo['plane_method']}": combo["timing"]
            for combo in combinations
        },
        "timing_on_task01_comparable_frames": {
            f"{combo['matcher']}|{combo['semantic_source']}|{combo['plane_method']}":
                combo["timing_on_task01_comparable_frames"]
            for combo in combinations
        },
        "per_frame_timing": [
            {
                "pair_id": row["pair_id"], "matcher": row["matcher"], "semantic_source": row["semantic_source"],
                "plane_method": row["plane_method"], "warmup_request": row["warmup_request"],
                "timing": row["timing"],
            }
            for row in records
        ],
        "interpretation_boundary": summary["interpretation_boundary"],
    }
    (args.output_dir / "timing_summary.json").write_text(
        json.dumps(timing_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "left_dir": str(args.left_dir.resolve()),
            "right_dir": str(args.right_dir.resolve()),
            "left_label_dir": str(args.left_label_dir.resolve()),
            "right_label_dir": str(args.right_label_dir.resolve()),
            "calibration": str(args.calibration.resolve()),
            "frozen_baseline_dir": str(args.frozen_baseline_dir.resolve()),
            "sgbm_baseline_dir": str(args.sgbm_baseline_dir.resolve()),
            "auto_floor_left_dir": None if args.auto_floor_left_dir is None else str(args.auto_floor_left_dir.resolve()),
            "auto_floor_right_dir": None if args.auto_floor_right_dir is None else str(args.auto_floor_right_dir.resolve()),
            "workspace_dir": str(workspace.resolve()),
        },
        "frozen_parameters": parameters,
        "pair_names": names,
        "matchers": list(ALL_MATCHERS),
        "plane_methods": list(MAIN_METHODS) + [REGION_METHOD],
        "worker": {
            "script": str(worker_script.resolve()),
            "igev_python": str(args.igev_python),
            "dynamicstereo_python": str(args.dynamicstereo_python),
            "igev_repo": str(args.igev_repo),
            "igev_weights": str(args.igev_weights),
            "dynamicstereo_repo": str(args.dynamicstereo_repo),
            "dynamicstereo_weights": str(args.dynamicstereo_weights),
        },
        "temporal_window": summary["dynamicstereo_offline_window"],
        "timing_formula": summary["timing_formula"],
        "notes": [
            "only the left/right -> disparity step is replaced; masks, calibration, rectification, "
            "triangulation, plane fitters and the cross-region gate are the frozen task-01 code path",
            "no training, fine-tuning or architecture change; official repositories and official checkpoints only",
            "the benchmark process never imports torch; the matchers run in separate conda environments",
            "the reverse field is obtained by mirroring the pair and un-mirroring the result, which turns the "
            "leftward search a left-reference matcher implements into the rightward search this calibration needs",
            "no physical ground truth exists in this round",
        ],
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "records": len(records),
        "combinations": len(combinations),
        "unavailable_combinations": unavailable_rows,
    }, ensure_ascii=False))
    return 0


def task01_warmup_pair_ids() -> set[int]:
    return {0, 40}


def estimate_pipeline(timing: dict[str, Any], *, use_gpu: bool) -> float | None:
    """The one documented comparable estimate for a matcher/method combination."""
    keys = ["mask_load_ms", "rectification_ms", "candidate_filter_ms", "triangulation_ms", "plane_fit_ms"]
    values = [timing.get(key) for key in keys]
    if use_gpu:
        values.append(timing.get("stereo_gpu_forward_ms"))
        values.append(timing.get("stereo_gpu_reverse_forward_ms"))
    else:
        values.append(timing.get("stereo_cpu_worker_wall_ms"))
        values.append(timing.get("stereo_cpu_worker_reverse_wall_ms"))
    if any(value is None for value in values):
        return None
    return float(sum(values))


def unavailable_record(
    pair_id: int,
    name: str,
    index: int,
    matcher: str,
    method: str,
    spec: dict[str, Any],
    reason: str,
    parameters: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timing = {key: None for key in REQUIRED_TIMING_KEYS + EXTRA_TIMING_KEYS}
    timing["mask_load_ms"] = spec.get("mask_load_ms")
    return {
        "schema_version": SCHEMA_VERSION,
        "pair_id": int(pair_id),
        "frame_id": name,
        "warmup_request": bool(index == 0),
        "matcher": matcher,
        "plane_method": method,
        "semantic_source": spec["source_name"],
        "temporal_window_frames": (DYNAMIC_WINDOW_FRAMES if matcher == "dynamicstereo" else 1),
        "target_position_in_window": 0,
        "future_lookahead_frames": (DYNAMIC_WINDOW_FRAMES - 1 if matcher == "dynamicstereo" else 0),
        "realtime_compatible": bool(matcher != "dynamicstereo"),
        "status": "unavailable",
        "reasons": [reason],
        "semantic_evidence": frozen.semantic_evidence_block(
            spec["source_name"], spec["identity_evidence"], spec.get("mask_area_fraction")
        ),
        "stereo_evidence": {
            "candidate_pixels": None,
            "forward_disparity_valid_fraction": None,
            "right_mask_correspondence_fraction": None,
            "lr_consistency_fraction": None,
            "positive_depth_fraction": None,
            "median_reprojection_px": None,
            "p95_reprojection_px": None,
            "reason": reason,
        },
        "plane_evidence": {
            "status": "unavailable", "inlier_count": None, "inlier_fraction": None,
            "median_residual_mm": None, "coverage_fraction": None,
            "normal_left_camera": None, "offset_mm": None, "reasons": [reason],
        },
        "cross_region_evidence": {
            "status": "unavailable", "region_count": None, "normal_spread_deg": None,
            "offset_spread_mm": None, "reasons": [reason],
            "applies_to": "shared_stereo_candidate_set_gate",
        },
        "timing": timing,
        "timing_null_reasons": {
            key: "stereo_candidate_stage_unavailable" for key, value in timing.items() if value is None
        },
        "confidence_boundary": CONFIDENCE_BOUNDARY,
        **(extra or {}),
    }


def worker_failure_details(forward: Any, reverse: Any) -> dict[str, Any]:
    return {
        "worker_failure": {
            "forward": None if forward is None else {
                "status": forward.status, "reasons": list(forward.reasons), "metadata": forward.metadata,
            },
            "reverse": None if reverse is None else {
                "status": reverse.status, "reasons": list(reverse.reasons), "metadata": reverse.metadata,
            },
        }
    }


def write_visualization(
    path: Path,
    left_local: np.ndarray,
    disparity: np.ndarray,
    matcher: str,
    pair_id: int,
    valid_fraction: float,
    gpu_forward_ms: float | None,
    cpu_worker_wall_ms: float | None,
) -> None:
    finite = np.isfinite(disparity)
    normalized = np.zeros(disparity.shape, dtype=np.uint8)
    if finite.any():
        values = disparity[finite]
        low, high = float(np.percentile(values, 2)), float(np.percentile(values, 98))
        if high - low < 1e-6:
            high = low + 1e-6
        scaled = np.clip((disparity - low) / (high - low), 0.0, 1.0)
        normalized = np.where(finite, (scaled * 255.0), 0).astype(np.uint8)
    heat = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(left_local, 0.45, heat, 0.55, 0.0)
    overlay[~finite] = (0.35 * left_local[~finite]).astype(np.uint8)
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 64), (255, 255, 255), -1)
    gpu_text = "n/a" if gpu_forward_ms is None else f"{gpu_forward_ms:.1f} ms"
    wall_text = "n/a" if cpu_worker_wall_ms is None else f"{cpu_worker_wall_ms:.1f} ms"
    cv2.putText(
        overlay,
        f"{matcher} | target pair_{pair_id:04d} | valid disparity={valid_fraction * 100:.1f}% | "
        f"gpu forward={gpu_text} | worker wall={wall_text}",
        (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        "INTERNAL CANDIDATE ONLY - not physical ground truth",
        (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 200), 1, cv2.LINE_AA,
    )
    write_image(path, overlay)


if __name__ == "__main__":
    raise SystemExit(main())
