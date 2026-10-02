#!/usr/bin/env python3
"""Evaluate propagated floor-mask candidates only on manual labels not used as references."""

from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
_tool_prepare_imports()

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REALTIME_ROOT = Path(__file__).resolve().parents[2]
if str(REALTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(REALTIME_ROOT))

from walker_tools.scene.audit_manual_floor_labels import binary_metrics, load_labelme, rasterize_labelme, render_error_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-output-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heldout-pair-ids", type=int, nargs="+", help="optional exact held-out subset; must exclude manual reference frames")
    return parser.parse_args()


def read_records(path: Path) -> dict[int, dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {int(record["pair_id"]): record for record in records}


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    summary = json.loads((args.temporal_output_dir / "summary.json").read_text(encoding="utf-8"))
    references = {int(pair_id) for pair_id in summary["manual_reference_pair_ids"]}
    records = read_records(args.temporal_output_dir / "records.jsonl")
    labelled = {int(path.stem.removeprefix("pair_")) for path in args.label_dir.glob("pair_*.json")}
    heldout = sorted(labelled & set(records) - references) if args.heldout_pair_ids is None else sorted(set(args.heldout_pair_ids))
    if not heldout:
        raise ValueError("no labelled non-reference held-out frames available")
    if set(heldout) & references:
        raise ValueError("manual reference frames cannot be evaluated as held-out")
    if not set(heldout) <= labelled & set(records):
        raise ValueError("every requested held-out frame needs both a label and propagation record")
    args.output_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for pair_id in heldout:
        stem = f"pair_{pair_id:04d}"
        image = cv2.imread(str(args.image_dir / f"{stem}.png"), cv2.IMREAD_COLOR)
        candidate = cv2.imread(str(args.temporal_output_dir / "candidate_masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or candidate is None:
            raise RuntimeError(f"cannot read image or propagated candidate for {stem}")
        masks = rasterize_labelme(load_labelme(args.label_dir / f"{stem}.json"))
        if image.shape[:2] != masks["floor_eligible"].shape or candidate.shape != masks["floor_eligible"].shape:
            raise ValueError(f"shape mismatch for {stem}")
        metrics = binary_metrics(masks["floor_eligible"], masks["valid_evaluation"], candidate > 0)
        record = records[pair_id]
        row = {"pair_id": pair_id, "propagation_status": record["status"], "seed_pair_id": record.get("seed_pair_id"), **metrics}
        rows.append(row)
        overlay = render_error_overlay(image, masks["floor_eligible"], masks["valid_evaluation"], candidate > 0, f"{stem} temporal propagation; green=TP red=FP magenta=FN")
        cv2.imwrite(str(args.output_dir / f"{stem}_error_overlay.png"), overlay)
    keys = ("precision", "recall", "iou")
    result = {
        "schema_version": "temporal_floor_propagation_heldout_audit_v1",
        "temporal_output_dir": str(args.temporal_output_dir.resolve()),
        "manual_reference_pair_ids_excluded": sorted(references),
        "heldout_pair_ids": heldout,
        "heldout_macro_metrics": {key: float(np.mean([row[key] for row in rows])) for key in keys},
        "rows": rows,
        "interpretation_boundary": "Held-out image-space mask agreement only. It does not validate stereo correspondence, ground-plane geometry, contact, support, gait, or future unseen sequences.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
