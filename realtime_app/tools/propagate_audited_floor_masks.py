#!/usr/bin/env python3
"""Propagate manually audited floor-mask seeds through an upright video sequence.

The tool never runs stereo or estimates a plane.  It supports two explicit
reference-information modes: a one-shot manual seed followed only by
propagation, or periodic manual re-anchors at a declared frame interval.  It
transports a mask one adjacent frame at a time with bidirectional optical-flow
quality gates.  Each non-manual output remains a candidate and must be checked
against independent held-out labels before it is used for any later research
decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REALTIME_ROOT = Path(__file__).resolve().parents[1]
if str(REALTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(REALTIME_ROOT))

from audit_manual_floor_labels import load_labelme, rasterize_labelme
from pose_app.temporal_floor_mask_propagation import PropagationThresholds, propagate_adjacent


SCHEMA_VERSION = "audited_floor_mask_temporal_propagation_v1"
ACCEPTED_PREVIOUS_STATUSES = {"seed", "manual_reanchor", "candidate"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="upright pair_XXXX.png sequence for one view")
    parser.add_argument("--label-dir", type=Path, required=True, help="LabelMe JSON files for manually audited reference frames")
    parser.add_argument("--reference-mode", choices=("one_shot", "periodic_reanchor"), required=True)
    parser.add_argument("--initial-seed-pair-id", type=int, required=True, help="first manually audited floor-mask frame")
    parser.add_argument("--reanchor-interval-frames", type=int, help="required only for periodic_reanchor; exact number of frames between manual references")
    parser.add_argument("--start-pair-id", type=int, required=True)
    parser.add_argument("--end-pair-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="new output directory")
    parser.add_argument("--maximum-forward-backward-error-px", type=float, default=1.5)
    parser.add_argument("--minimum-candidate-flow-consistency", type=float, default=0.70)
    parser.add_argument("--maximum-candidate-photometric-median", type=float, default=35.0)
    parser.add_argument("--minimum-candidate-pixels", type=int, default=200)
    return parser.parse_args()


def image_path(input_dir: Path, pair_id: int) -> Path:
    return input_dir / f"pair_{pair_id:04d}.png"


def scheduled_reference_ids(
    *,
    reference_mode: str,
    initial_seed_pair_id: int,
    reanchor_interval_frames: int | None,
    start_pair_id: int,
    end_pair_id: int,
) -> list[int]:
    """Return the only frames permitted to consume accurate manual information."""
    if not start_pair_id <= initial_seed_pair_id <= end_pair_id:
        raise ValueError("initial seed pair ID must lie within the explicit output interval")
    if reference_mode == "one_shot":
        if reanchor_interval_frames is not None:
            raise ValueError("one_shot does not accept reanchor_interval_frames")
        return [initial_seed_pair_id]
    if reference_mode != "periodic_reanchor":
        raise ValueError(f"unsupported reference mode: {reference_mode}")
    if reanchor_interval_frames is None or reanchor_interval_frames < 1:
        raise ValueError("periodic_reanchor requires a positive reanchor_interval_frames")
    return list(range(initial_seed_pair_id, end_pair_id + 1, reanchor_interval_frames))


def manual_floor_seed(label_dir: Path, pair_id: int, shape: tuple[int, int]) -> np.ndarray:
    label_path = label_dir / f"pair_{pair_id:04d}.json"
    if not label_path.is_file():
        raise FileNotFoundError(f"manual seed label missing: {label_path}")
    labels = rasterize_labelme(load_labelme(label_path))
    mask = labels["floor_eligible"]
    if mask.shape != shape:
        raise ValueError(f"{label_path}: label shape {mask.shape} differs from image shape {shape}")
    if not mask.any():
        raise ValueError(f"{label_path}: floor_eligible seed is empty")
    return mask


def render_overlay(image: np.ndarray, mask: np.ndarray, status: str) -> np.ndarray:
    color = np.zeros_like(image)
    color[mask] = (0, 180, 0) if status in {"seed", "manual_reanchor", "candidate"} else (0, 120, 230)
    output = cv2.addWeighted(image, 0.62, color, 0.38, 0.0)
    cv2.rectangle(output, (0, 0), (min(900, output.shape[1]), 32), (255, 255, 255), -1)
    cv2.putText(output, f"temporal floor-mask {status}; image-space candidate only", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 0, 0), 1, cv2.LINE_AA)
    return output


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    if args.end_pair_id < args.start_pair_id:
        raise ValueError("end pair ID must be >= start pair ID")
    reference_ids = scheduled_reference_ids(
        reference_mode=args.reference_mode,
        initial_seed_pair_id=args.initial_seed_pair_id,
        reanchor_interval_frames=args.reanchor_interval_frames,
        start_pair_id=args.start_pair_id,
        end_pair_id=args.end_pair_id,
    )
    references = set(reference_ids)
    thresholds = PropagationThresholds(
        maximum_forward_backward_error_px=args.maximum_forward_backward_error_px,
        minimum_candidate_flow_consistency=args.minimum_candidate_flow_consistency,
        maximum_candidate_photometric_median=args.maximum_candidate_photometric_median,
        minimum_candidate_pixels=args.minimum_candidate_pixels,
    )
    thresholds.validate()
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "candidate_masks").mkdir()
    (args.output_dir / "overlays").mkdir()
    previous_image: np.ndarray | None = None
    previous_mask: np.ndarray | None = None
    previous_status = "before_first_seed"
    active_seed_pair_id: int | None = None
    records: list[dict[str, Any]] = []
    for pair_id in range(args.start_pair_id, args.end_pair_id + 1):
        path = image_path(args.input_dir, pair_id)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot read required adjacent frame: {path}")
        shape = image.shape[:2]
        if pair_id in references:
            mask = manual_floor_seed(args.label_dir, pair_id, shape)
            active_seed_pair_id = pair_id
            status = "seed" if pair_id == args.initial_seed_pair_id else "manual_reanchor"
            record: dict[str, Any] = {
                "pair_id": pair_id,
                "status": status,
                "source_pair_id": pair_id,
                "seed_pair_id": pair_id,
                "reference_information": "manual_accurate_floor_label",
                "reasons": [],
                "propagated_floor_pixel_count": int(mask.sum()),
                "interpretation": "manually audited image-space seed; not stereo, geometry, contact, or gait evidence",
            }
        elif previous_image is None or previous_mask is None or previous_status not in ACCEPTED_PREVIOUS_STATUSES:
            mask = np.zeros(shape, dtype=bool)
            record = {
                "pair_id": pair_id,
                "status": "unavailable",
                "source_pair_id": None,
                "seed_pair_id": active_seed_pair_id,
                "reference_information": "none",
                "reasons": ["no_accepted_adjacent_seed_or_candidate"],
                "propagated_floor_pixel_count": 0,
                "interpretation": "no image-space floor candidate is emitted; not a ground absence claim",
            }
        else:
            mask, step = propagate_adjacent(previous_image, image, previous_mask, thresholds)
            record = {
                "pair_id": pair_id,
                "source_pair_id": pair_id - 1,
                "seed_pair_id": active_seed_pair_id,
                "reference_information": "none",
                **step,
            }
        cv2.imwrite(str(args.output_dir / "candidate_masks" / path.name), mask.astype(np.uint8) * 255)
        cv2.imwrite(str(args.output_dir / "overlays" / path.name), render_overlay(image, mask, record["status"]))
        records.append(record)
        previous_image, previous_mask, previous_status = image, mask, record["status"]
    counts = {status: sum(record["status"] == status for record in records) for status in ("seed", "manual_reanchor", "candidate", "unavailable")}
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "input_dir": str(args.input_dir.resolve()),
        "label_dir": str(args.label_dir.resolve()),
        "reference_mode": args.reference_mode,
        "initial_seed_pair_id": args.initial_seed_pair_id,
        "manual_reference_pair_ids": reference_ids,
        "pair_interval": [args.start_pair_id, args.end_pair_id],
        "status_counts": counts,
        "parameters": {
            "reanchor_interval_frames": args.reanchor_interval_frames,
            "maximum_forward_backward_error_px": thresholds.maximum_forward_backward_error_px,
            "minimum_candidate_flow_consistency": thresholds.minimum_candidate_flow_consistency,
            "maximum_candidate_photometric_median": thresholds.maximum_candidate_photometric_median,
            "minimum_candidate_pixels": thresholds.minimum_candidate_pixels,
        },
        "interpretation_boundary": "Only listed manual reference frames consume accurate floor-mask information. All other masks are optical-flow propagated image-space candidates, not independently manually audited masks, stereo matches, metric ground planes, contact, support, or gait outputs.",
    }
    (args.output_dir / "records.jsonl").write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
