"""Run DA3 directly on representative raw-fisheye stereo pairs from M1.

This is deliberately not a calibrated stereo reconstruction: no rotation,
undistortion, intrinsics or extrinsics are passed to DA3.  Each distance group
is inferred independently, so DA3 never mistakes frames from different times
for one multiview scene.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-selection", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--da3-root", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument(
        "--condition-index", type=int, default=10,
        help="Zero-based representative pair within each 20-pair distance condition.",
    )
    return parser.parse_args()


def finite_summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"finite_count": 0, "finite_fraction": 0.0, "median": None}
    return {
        "finite_count": int(finite.size),
        "finite_fraction": float(finite.size / values.size),
        "median": float(np.median(finite)),
    }


def main() -> int:
    args = parse_args()
    selection = args.input_selection.resolve()
    run_dir = args.run_dir.resolve()
    da3_root = args.da3_root.resolve()
    model_dir = args.model_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {run_dir}")
    for path in (selection / "selection_manifest.csv", selection / "dataset_summary.json", model_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    source_dir = da3_root / "src"
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    sys.path.insert(0, str(source_dir))
    from depth_anything_3.api import DepthAnything3

    with (selection / "selection_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["condition"], []).append(row)
    expected = {"far_3m", "mid_2m", "near_1p3m"}
    if set(groups) != expected or any(len(items) != 20 for items in groups.values()):
        raise RuntimeError("M1 selection must contain exactly 20 pairs per far/mid/near condition.")
    if not 0 <= args.condition_index < 20:
        raise ValueError("--condition-index must be in [0, 19].")

    run_dir.mkdir(parents=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    model = DepthAnything3.from_pretrained(str(model_dir)).to(device)
    condition_outputs: dict[str, dict] = {}
    for condition in ("far_3m", "mid_2m", "near_1p3m"):
        row = sorted(groups[condition], key=lambda item: int(item["condition_index"]))[args.condition_index]
        left = selection / "raw_fisheye" / "left" / row["file_name"]
        right = selection / "raw_fisheye" / "right" / row["file_name"]
        if not left.is_file() or not right.is_file():
            raise FileNotFoundError(f"Missing raw pair for {condition}: {left}, {right}")
        condition_dir = run_dir / condition
        condition_started = time.perf_counter()
        prediction = model.inference(
            [str(left), str(right)],
            process_res=504,
            process_res_method="upper_bound_resize",
            export_dir=str(condition_dir),
            export_format="mini_npz-depth_vis",
        )
        depth = np.asarray(prediction.depth)
        confidence = np.asarray(prediction.conf)
        condition_outputs[condition] = {
            "selected_manifest_row": row,
            "raw_fisheye_images": [str(left), str(right)],
            "runtime_seconds": time.perf_counter() - condition_started,
            "depth_shape": list(depth.shape),
            "confidence_shape": list(confidence.shape),
            "depth_finite": finite_summary(depth),
            "confidence_finite": finite_summary(confidence),
            "export_directory": str(condition_dir),
        }

    dataset = json.loads((selection / "dataset_summary.json").read_text(encoding="utf-8"))
    metadata = {
        "experiment_id": "E20260829-D3_DA3_large_raw_fisheye_far_mid_near",
        "input_selection": str(selection),
        "input_capture": dataset["capture_root"],
        "model": "DA3-LARGE-1.1",
        "model_directory": str(model_dir),
        "execution_device": device,
        "condition_index": args.condition_index,
        "conditions": condition_outputs,
        "runtime_seconds_total": time.perf_counter() - started,
        "input_geometry": "original raw fisheye pixels; no rotation, undistortion, virtual camera, intrinsics or extrinsics supplied",
        "grouping_rule": "each synchronized left/right pair is inferred separately; distance conditions are not mixed into one multiview group",
        "interpretation_boundary": "DA3 output is a qualitative relative-depth and model-confidence diagnostic. Finite values or confidence values do not establish metric depth, calibrated stereo consistency, keypoint accuracy, or a repair of failed triangulation.",
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
