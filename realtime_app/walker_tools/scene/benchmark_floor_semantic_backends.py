#!/usr/bin/env python3
"""Compare image-space floor-candidate backends on one upright camera view.

The script deliberately separates a *floor candidate mask* from a directly
observed metric ground plane.  It is an offline visual comparison tool: its
outputs never enter pose, association, calibration, triangulation, contact, or
local-ground state code.

Backends
--------
``segformer_b0``
    Cached ADE20K SegFormer baseline.
``mask2former_swin_small`` / ``oneformer_swin_tiny``
    ADE20K semantic models exposed through the same binary-floor interface.
    They require their public model files to be cached first, unless
    ``--allow-download`` is explicitly supplied.
``groundnet_contract``
    A non-inference compatibility record for GroundNet.  GroundNet's intended
    result is a monocular plane normal/horizon estimate; its auxiliary ground
    segmentation is not a validated floor mask for this fisheye stereo setup.
    No made-up mask is emitted when no compatible checkpoint and adaptation
    protocol have been supplied.

Agreement between model masks is diagnostic only.  It is not segmentation
accuracy, identity evidence, or a permission to fit/accept a physical ground
plane.
"""

from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
_tool_prepare_imports()

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


SCHEMA_VERSION = "floor_semantic_backend_comparison_v1"
BACKEND_SPECS: dict[str, dict[str, str]] = {
    "segformer_b0": {
        "model_id": "nvidia/segformer-b0-finetuned-ade-512-512",
        "kind": "segformer",
    },
    "mask2former_swin_small": {
        "model_id": "facebook/mask2former-swin-small-ade-semantic",
        "kind": "mask2former",
    },
    "oneformer_swin_tiny": {
        "model_id": "shi-labs/oneformer_ade20k_swin_tiny",
        "kind": "oneformer",
    },
    "groundnet_contract": {
        "model_id": "GroundNet_MM2019",
        "kind": "groundnet_contract",
    },
}
FLOOR_COLOR_BGR = (35, 185, 35)


@dataclass(frozen=True)
class BackendResult:
    status: str
    mask: np.ndarray | None
    metrics: dict[str, Any]
    reasons: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="one upright view containing pair_*.png")
    parser.add_argument("--output-dir", required=True, type=Path, help="new comparison output directory")
    parser.add_argument("--backends", nargs="+", choices=tuple(BACKEND_SPECS), default=("segformer_b0", "mask2former_swin_small", "oneformer_swin_tiny"))
    parser.add_argument("--pair-ids", nargs="*", type=int, help="explicit pair ids; otherwise stride sampling is used")
    parser.add_argument("--sample-stride", type=int, default=40, help="use every Nth frame when --pair-ids is omitted")
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1, help="inference images per forward pass; defaults to 1 for bounded GPU memory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-download", action="store_true", help="allow transformers to fetch missing public model files")
    return parser.parse_args()


def select_paths(input_dir: Path, pair_ids: list[int] | None, sample_stride: int, max_frames: int) -> list[Path]:
    if sample_stride < 1 or max_frames < 1:
        raise ValueError("--sample-stride and --max-frames must be positive")
    all_paths = sorted(input_dir.glob("pair_*.png"))
    if not all_paths:
        raise RuntimeError(f"no pair_*.png files in {input_dir}")
    by_id = {int(path.stem.removeprefix("pair_")): path for path in all_paths}
    if pair_ids is not None and len(pair_ids):
        missing = [value for value in pair_ids if value not in by_id]
        if missing:
            raise ValueError(f"requested pair ids missing from input: {missing}")
        return [by_id[value] for value in pair_ids]
    return all_paths[::sample_stride][:max_frames]


def floor_class_id(model: Any) -> int:
    matches = [int(index) for index, label in model.config.id2label.items() if str(label).strip().casefold() == "floor"]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one ADE20K 'floor' class, found {matches}")
    return matches[0]


def resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return requested


def load_semantic_backend(name: str, requested_device: str, allow_download: bool) -> tuple[Callable[[list[np.ndarray]], list[np.ndarray]], dict[str, Any]]:
    """Return a batch predictor whose outputs are integer semantic label maps."""
    spec = BACKEND_SPECS[name]
    if spec["kind"] == "groundnet_contract":
        raise ValueError("groundnet_contract has no semantic image predictor")
    import torch

    device = resolve_device(torch, requested_device)
    local_files_only = not allow_download
    if spec["kind"] == "segformer":
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        processor = SegformerImageProcessor.from_pretrained(spec["model_id"], local_files_only=local_files_only)
        model = SegformerForSemanticSegmentation.from_pretrained(spec["model_id"], local_files_only=local_files_only).to(device).eval()

        def predict(images: list[np.ndarray]) -> list[np.ndarray]:
            rgb = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
            inputs = {key: value.to(device) for key, value in processor(images=rgb, return_tensors="pt").items()}
            with torch.no_grad():
                output = model(**inputs)
            return [item.detach().cpu().numpy() for item in processor.post_process_semantic_segmentation(output, target_sizes=[item.shape[:2] for item in images])]

    elif spec["kind"] == "mask2former":
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        processor = AutoImageProcessor.from_pretrained(spec["model_id"], local_files_only=local_files_only)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(spec["model_id"], local_files_only=local_files_only).to(device).eval()

        def predict(images: list[np.ndarray]) -> list[np.ndarray]:
            rgb = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
            inputs = {key: value.to(device) for key, value in processor(images=rgb, return_tensors="pt").items()}
            with torch.no_grad():
                output = model(**inputs)
            return [item.detach().cpu().numpy() for item in processor.post_process_semantic_segmentation(output, target_sizes=[item.shape[:2] for item in images])]

    elif spec["kind"] == "oneformer":
        from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

        processor = OneFormerProcessor.from_pretrained(spec["model_id"], local_files_only=local_files_only)
        model = OneFormerForUniversalSegmentation.from_pretrained(spec["model_id"], local_files_only=local_files_only).to(device).eval()

        def predict(images: list[np.ndarray]) -> list[np.ndarray]:
            rgb = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
            inputs = {key: value.to(device) for key, value in processor(images=rgb, task_inputs=["semantic"] * len(rgb), return_tensors="pt").items()}
            with torch.no_grad():
                output = model(**inputs)
            return [item.detach().cpu().numpy() for item in processor.post_process_semantic_segmentation(output, target_sizes=[item.shape[:2] for item in images])]

    else:  # pragma: no cover - construction guards this branch
        raise AssertionError(f"unsupported backend kind: {spec['kind']}")
    return predict, {"model_id": spec["model_id"], "backend_kind": spec["kind"], "device": device, "floor_class_id": floor_class_id(model), "local_files_only": local_files_only}


def render_overlay(image: np.ndarray, mask: np.ndarray, text: str) -> np.ndarray:
    color = np.zeros_like(image)
    color[mask] = FLOOR_COLOR_BGR
    output = cv2.addWeighted(image, 0.62, color, 0.38, 0.0)
    cv2.rectangle(output, (0, 0), (min(1100, output.shape[1]), 30), (255, 255, 255), -1)
    cv2.putText(output, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
    return output


def mask_agreement(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    if first.shape != second.shape:
        raise ValueError("cannot compare masks with different image shapes")
    intersection = int(np.logical_and(first, second).sum())
    union = int(np.logical_or(first, second).sum())
    return {"intersection_px": intersection, "union_px": union, "iou": float(intersection / union) if union else 1.0}


def groundnet_unavailable() -> BackendResult:
    return BackendResult(
        "unavailable",
        None,
        {
            "model_intended_output": "monocular_ground_plane_normal_and_horizon; auxiliary_ground_segmentation",
            "project_compatibility": "not_configured_for_fisheye_metric_plane_or_floor_mask_acceptance",
        },
        [
            "no_project-adapted_groundnet_checkpoint",
            "paper_assumes_pinhole_camera_model",
            "paper_evaluates_outdoor_KITTI_and_ApolloScape",
            "monocular_normal_does_not_supply_metric_plane_offset",
        ],
    )


def run_backend(name: str, images: list[np.ndarray], requested_device: str, allow_download: bool, batch_size: int) -> tuple[list[BackendResult], dict[str, Any]]:
    if name == "groundnet_contract":
        return [groundnet_unavailable() for _ in images], {"backend": name, **groundnet_unavailable().metrics}
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    start = time.perf_counter()
    try:
        predict, metadata = load_semantic_backend(name, requested_device, allow_download)
        predictions: list[np.ndarray] = []
        for start_index in range(0, len(images), batch_size):
            predictions.extend(predict(images[start_index:start_index + batch_size]))
        elapsed = time.perf_counter() - start
        floor_id = int(metadata["floor_class_id"])
        results = [BackendResult("candidate", array == floor_id, {"floor_fraction": float((array == floor_id).mean()), "inference_seconds_per_image_including_load": elapsed / len(images)}, []) for array in predictions]
        return results, {"backend": name, **metadata, "elapsed_seconds_including_load": elapsed}
    except Exception as error:  # missing cache and optional backends must not block other methods
        return [BackendResult("unavailable", None, {}, [f"backend_initialization_or_inference_failed:{type(error).__name__}", str(error)]) for _ in images], {"backend": name, "model_id": BACKEND_SPECS[name]["model_id"], "status": "unavailable", "error_type": type(error).__name__, "error": str(error), "allow_download": allow_download}
    finally:
        # The comparison is sequential.  Retaining a previous universal model
        # would turn the next backend's failure into a host-memory artifact.
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    paths = select_paths(args.input_dir, args.pair_ids, args.sample_stride, args.max_frames)
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    if any(image is None for image in images):
        raise RuntimeError("cannot read one or more selected images")
    if len({image.shape[:2] for image in images}) != 1:
        raise RuntimeError("comparison requires equally sized images")
    args.output_dir.mkdir(parents=True)
    all_results: dict[str, list[BackendResult]] = {}
    backend_metadata: dict[str, dict[str, Any]] = {}
    for name in args.backends:
        results, metadata = run_backend(name, images, args.device, args.allow_download, args.batch_size)
        all_results[name], backend_metadata[name] = results, metadata
        folder = args.output_dir / name
        folder.mkdir()
        mask_folder = folder / "floor_masks"
        mask_folder.mkdir()
        for path, image, result in zip(paths, images, results):
            row = {"schema_version": SCHEMA_VERSION, "pair_id": int(path.stem.removeprefix("pair_")), "status": result.status, "metrics": result.metrics, "reasons": result.reasons, "interpretation": "image-space floor candidate only; not ground identity or metric geometry"}
            (folder / f"{path.stem}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if result.mask is not None:
                cv2.imwrite(str(folder / f"{path.stem}_floor_candidate.png"), result.mask.astype(np.uint8) * 255)
                # This is intentionally only an interface-compatible export.
                # ``observe_local_ground_semantic_stereo`` still receives the
                # provenance as unvalidated and therefore cannot accept a
                # physical plane from it.
                cv2.imwrite(str(mask_folder / path.name), result.mask.astype(np.uint8) * 255)
                cv2.imwrite(str(folder / f"{path.stem}_overlay.png"), render_overlay(image, result.mask, f"{name}; green=floor candidate; {path.name}"))
    agreement: list[dict[str, Any]] = []
    candidate_names = [name for name, results in all_results.items() if all(item.mask is not None for item in results)]
    for index, path in enumerate(paths):
        for first_index, first_name in enumerate(candidate_names):
            for second_name in candidate_names[first_index + 1:]:
                agreement.append({"pair_id": int(path.stem.removeprefix("pair_")), "first": first_name, "second": second_name, **mask_agreement(all_results[first_name][index].mask, all_results[second_name][index].mask)})
    payload = {"schema_version": SCHEMA_VERSION, "input_dir": str(args.input_dir.resolve()), "pair_ids": [int(path.stem.removeprefix("pair_")) for path in paths], "backend_metadata": backend_metadata, "pairwise_candidate_mask_agreement": agreement, "interpretation_boundary": "Model-mask agreement is not ground-mask accuracy. No backend result is passed to stereo, plane fitting, calibration, contact, or gait."}
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "command.txt").write_text(" ".join(str(item) for item in ["python", Path(__file__).name, *__import__("sys").argv[1:]]) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "pair_count": len(paths), "available_candidate_backends": candidate_names, "unavailable_backends": [name for name in all_results if name not in candidate_names]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
