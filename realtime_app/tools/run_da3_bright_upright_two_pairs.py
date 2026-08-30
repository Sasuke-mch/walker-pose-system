"""Run DA3 on upright versions of selected raw-fisheye stereo pairs.

This is a display-orientation experiment only.  It rotates the existing raw
fisheye frames before DA3 inference (left: counter-clockwise 90 degrees;
right: clockwise 90 degrees), while still supplying no camera calibration to
DA3.  The outputs must not be interpreted as calibrated stereo geometry.
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


CONDITIONS = ("far_3m", "mid_2m", "near_1p3m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-selection", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--da3-root", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument(
        "--experiment-id", default="E20260829-D5_DA3_upright_2pairs_per_distance",
        help="Identifier recorded in run_metadata.json.",
    )
    parser.add_argument(
        "--condition-indices", default="5,15",
        help="Comma-separated zero-based pair indices within every 20-pair distance group.",
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


def parse_indices(raw: str) -> list[int]:
    indices = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not indices or len(set(indices)) != len(indices) or any(index < 0 or index >= 20 for index in indices):
        raise ValueError("--condition-indices must contain one or more distinct values in [0, 19].")
    return sorted(indices)


def main() -> int:
    args = parse_args()
    selection = args.input_selection.resolve()
    run_dir = args.run_dir.resolve()
    da3_root = args.da3_root.resolve()
    model_dir = args.model_dir.resolve()
    pair_indices = parse_indices(args.condition_indices)
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {run_dir}")
    for path in (selection / "selection_manifest.csv", selection / "dataset_summary.json", model_dir, da3_root / "src"):
        if not path.exists():
            raise FileNotFoundError(path)

    with (selection / "selection_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["condition"], []).append(row)
    if set(groups) != set(CONDITIONS) or any(len(items) != 20 for items in groups.values()):
        raise RuntimeError("Input selection must contain exactly 20 pairs in each far/mid/near condition.")

    for condition in CONDITIONS:
        for index in pair_indices:
            row = sorted(groups[condition], key=lambda item: int(item["condition_index"]))[index]
            left = selection / "left_ccw90" / row["file_name"]
            right = selection / "right_cw90" / row["file_name"]
            if not left.is_file() or not right.is_file():
                raise FileNotFoundError(f"Missing upright pair for {condition} index {index}: {left}, {right}")

    sys.path.insert(0, str(da3_root / "src"))
    from depth_anything_3.api import DepthAnything3

    run_dir.mkdir(parents=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    model = DepthAnything3.from_pretrained(str(model_dir)).to(device)
    outputs: dict[str, list[dict]] = {condition: [] for condition in CONDITIONS}

    for condition in CONDITIONS:
        ordered = sorted(groups[condition], key=lambda item: int(item["condition_index"]))
        for index in pair_indices:
            row = ordered[index]
            left = selection / "left_ccw90" / row["file_name"]
            right = selection / "right_cw90" / row["file_name"]
            pair_dir = run_dir / condition / row["file_name"].removesuffix(".png")
            pair_started = time.perf_counter()
            prediction = model.inference(
                [str(left), str(right)],
                process_res=504,
                process_res_method="upper_bound_resize",
                export_dir=str(pair_dir),
                export_format="mini_npz-depth_vis",
            )
            depth = np.asarray(prediction.depth)
            confidence = np.asarray(prediction.conf)
            outputs[condition].append({
                "selected_manifest_row": row,
                "upright_images": [str(left), str(right)],
                "input_rotation": {"left": "ccw90", "right": "cw90"},
                "runtime_seconds": time.perf_counter() - pair_started,
                "depth_shape": list(depth.shape),
                "confidence_shape": list(confidence.shape),
                "depth_finite": finite_summary(depth),
                "confidence_finite": finite_summary(confidence),
                "export_directory": str(pair_dir),
            })

    dataset = json.loads((selection / "dataset_summary.json").read_text(encoding="utf-8"))
    metadata = {
        "experiment_id": args.experiment_id,
        "input_selection": str(selection),
        "input_capture": dataset["capture_root"],
        "model": "DA3-LARGE-1.1",
        "model_directory": str(model_dir),
        "execution_device": device,
        "condition_indices": pair_indices,
        "pairs_per_distance": len(pair_indices),
        "conditions": outputs,
        "runtime_seconds_total": time.perf_counter() - started,
        "input_geometry": "raw fisheye frames rotated upright for model input (left ccw90; right cw90); no undistortion or calibration supplied",
        "interpretation_boundary": "DA3 output remains qualitative relative depth and model confidence. Rotation improves viewing orientation only; it does not make DA3 depth metric, calibrated, or a proof of correct cross-view correspondence.",
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
