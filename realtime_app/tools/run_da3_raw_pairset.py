"""Run DA3 independently on every selected raw-fisheye stereo pair.

The outputs are deliberately diagnostic only: each synchronized left/right
pair is inferred independently with no calibration supplied to DA3.  The
script exists so a later ROI experiment never reuses one representative depth
map for different frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-selection", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--da3-root", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    return parser.parse_args()


def finite_summary(values: np.ndarray) -> dict[str, float | int]:
    finite = np.isfinite(values)
    count = int(finite.sum())
    return {
        "finite_count": count,
        "finite_fraction": float(count / values.size) if values.size else 0.0,
        "median": float(np.median(values[finite])) if count else float("nan"),
    }


def load_selection(root: Path) -> dict[str, list[dict[str, str]]]:
    manifest = root / "selection_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Selection manifest not found: {manifest}")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            condition = row.get("condition")
            file_name = row.get("file_name")
            if not condition or not file_name:
                raise ValueError("Selection manifest requires condition and file_name columns")
            groups[condition].append(row)
    expected = {"far_3m", "mid_2m", "near_1p3m"}
    if set(groups) != expected or any(len(rows) != 20 for rows in groups.values()):
        raise RuntimeError("Expected exactly 20 selected pairs in each far/mid/near condition")
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["condition_index"]))
    return dict(groups)


def main() -> int:
    args = parse_args()
    selection = args.input_selection.resolve()
    run_dir = args.run_dir.resolve()
    da3_root = args.da3_root.resolve()
    model_dir = args.model_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {run_dir}")
    for path in (selection / "raw_fisheye" / "left", selection / "raw_fisheye" / "right", model_dir):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    groups = load_selection(selection)
    from depth_anything_3.api import DepthAnything3
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir.mkdir(parents=True)
    started = time.perf_counter()
    model = DepthAnything3.from_pretrained(str(model_dir)).to(device)
    pair_outputs: list[dict[str, Any]] = []

    for condition in ("far_3m", "mid_2m", "near_1p3m"):
        for row in groups[condition]:
            file_name = row["file_name"]
            left = selection / "raw_fisheye" / "left" / file_name
            right = selection / "raw_fisheye" / "right" / file_name
            if not left.is_file() or not right.is_file():
                raise FileNotFoundError(f"Missing raw stereo pair: {left} / {right}")
            pair_dir = run_dir / condition / file_name.removesuffix(".png")
            pair_started = time.perf_counter()
            prediction = model.inference(
                [str(left), str(right)],
                process_res_method="upper_bound_resize",
                export_dir=str(pair_dir),
                export_format="mini_npz-depth_vis",
            )
            depth = np.asarray(prediction.depth)
            confidence = np.asarray(prediction.conf)
            pair_outputs.append(
                {
                    "condition": condition,
                    "selected_manifest_row": row,
                    "raw_fisheye_images": [str(left), str(right)],
                    "runtime_seconds": time.perf_counter() - pair_started,
                    "depth_shape": list(depth.shape),
                    "confidence_shape": list(confidence.shape),
                    "depth_finite": finite_summary(depth),
                    "confidence_finite": finite_summary(confidence),
                    "export_directory": str(pair_dir),
                }
            )
            print(f"{condition} {file_name}: {pair_outputs[-1]['runtime_seconds']:.3f}s", flush=True)

    metadata = {
        "classification": "engineering_validation",
        "experiment_id": "V20260901-DA3-RAW60",
        "input_selection": str(selection),
        "model": "DA3-LARGE-1.1",
        "model_directory": str(model_dir),
        "execution_device": device,
        "input_geometry": "original raw fisheye pixels; no rotation, undistortion, intrinsics, or extrinsics supplied",
        "grouping_rule": "each synchronized left/right pair is inferred independently",
        "interpretation_boundary": "Relative depth is used only to derive an image-space ROI candidate. It is not metric depth, calibrated stereo geometry, a keypoint label, or a triangulation substitute.",
        "pairs": pair_outputs,
        "runtime_seconds_total": time.perf_counter() - started,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Completed {len(pair_outputs)} synchronized pairs: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
