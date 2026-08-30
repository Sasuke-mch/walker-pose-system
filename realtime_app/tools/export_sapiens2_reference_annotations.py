"""Export a transparent Sapiens2-assisted single-view reference-label table.

The export intentionally calls these labels *references* / pseudo-labels.  They
are suitable for a consistent engineering baseline and for triaging uncertain
frames, not for an independent accuracy claim about the same Sapiens2 model.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from pose_app.rotation import model_to_raw_point
from tools.evaluate_offline_stereo_predictions import _records


JOINTS = {
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-selection", required=True, type=Path)
    parser.add_argument("--left-predictions", required=True, type=Path)
    parser.add_argument("--right-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--left-rotation", default="ccw90")
    parser.add_argument("--right-rotation", default="cw90")
    parser.add_argument("--usable-score", type=float, default=0.25)
    parser.add_argument("--high-score", type=float, default=0.50)
    parser.add_argument(
        "--reference-source",
        default="Sapiens2-0.4B_AI_assisted",
        help="Traceable label describing the exact Sapiens2 inference condition.",
    )
    return parser.parse_args()


def select_target(persons):
    if not persons:
        return None, True, "no_person"
    ordered = sorted(persons, key=lambda person: person.bbox_score, reverse=True)
    primary = ordered[0]
    ambiguous = (
        len(ordered) > 1
        and primary.bbox_score > 0
        and ordered[1].bbox_score / primary.bbox_score >= 0.70
    )
    return primary, ambiguous, "highest_detector_bbox_score"


def reference_status(score: float, usable_score: float, high_score: float) -> tuple[int, bool, str]:
    if score >= high_score:
        return 2, True, "high_confidence_reference"
    if score >= usable_score:
        return 1, True, "usable_low_confidence_reference"
    return 0, False, "below_usable_score"


def main() -> int:
    args = parse_args()
    selection = args.input_selection.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    with (selection / "selection_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if len(manifest) != 60:
        raise RuntimeError("Expected the fixed 60-pair selection")
    condition_by_name = {row["file_name"]: row["condition"] for row in manifest}
    paths = {
        "left": (args.left_predictions.resolve(), args.left_rotation),
        "right": (args.right_predictions.resolve(), args.right_rotation),
    }
    output.mkdir(parents=True)
    annotation_rows: list[dict] = []
    review_rows: list[dict] = []
    confidence: dict[str, list[float]] = defaultdict(list)
    for side, (path, rotation) in paths.items():
        records = _records(path, "sapiens2", "raw_keypoint")
        if len(records) != 60 or {record["name"] for record in records} != set(condition_by_name):
            raise RuntimeError(f"{side}: Sapiens2 prediction names do not match the selection")
        for record in records:
            target, ambiguous, method = select_target(record["persons"])
            image_id = f"{side}_{record['name'].removesuffix('.png')}"
            review_rows.append(
                {
                    "image_id": image_id,
                    "file_name": record["name"],
                    "side": side,
                    "condition": condition_by_name[record["name"]],
                    "candidate_instance_count": len(record["persons"]),
                    "selected_person_id": "" if target is None else target.person_id,
                    "selected_bbox_score": "" if target is None else f"{target.bbox_score:.8f}",
                    "selected_pose_score": "" if target is None else f"{target.pose_score:.8f}",
                    "selection_method": method,
                    "needs_manual_target_review": str(ambiguous).lower(),
                }
            )
            for index, joint in JOINTS.items():
                if target is None or len(target.keypoints) <= index:
                    score, x, y = 0.0, "", ""
                    visibility, usable, status = 0, False, "no_target_person"
                else:
                    x_model, y_model, score = [float(value) for value in target.keypoints[index]]
                    x_raw, y_raw = model_to_raw_point(
                        x_model, y_model, 1920, 1080, rotation
                    )
                    visibility, usable, status = reference_status(
                        score, args.usable_score, args.high_score
                    )
                    x = f"{x_raw:.6f}" if usable else ""
                    y = f"{y_raw:.6f}" if usable else ""
                    confidence[f"{side}/{condition_by_name[record['name']]}/{joint}"].append(score)
                annotation_rows.append(
                    {
                        "image_id": image_id,
                        "file_name": record["name"],
                        "side": side,
                        "condition": condition_by_name[record["name"]],
                        "joint_subject_anatomy": joint,
                        "keypoint_index_coco17": index,
                        "x_px_raw_fisheye": x,
                        "y_px_raw_fisheye": y,
                        "reference_visibility": visibility,
                        "reference_usable": str(usable).lower(),
                        "sapiens2_confidence": f"{score:.8f}",
                        "reference_status": status,
                        "reference_source": args.reference_source,
                    }
                )
    with (output / "sapiens2_ai_reference_keypoints.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(annotation_rows[0]))
        writer.writeheader()
        writer.writerows(annotation_rows)
    with (output / "sapiens2_target_selection_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)

    confidence_summary = {
        key: {"count": len(values), "mean": float(np.mean(values)), "median": float(np.median(values))}
        for key, values in sorted(confidence.items())
    }
    metadata = {
        "experiment_id": "E20260829-A1_Sapiens2_AI_assisted_single_view_reference",
        "input_selection": str(selection),
        "input_orientation": {"left": args.left_rotation, "right": args.right_rotation},
        "reference_source": args.reference_source,
        "coordinate_space": "original 1920x1080 raw fisheye pixels after inverse rotation",
        "images": 120,
        "reference_joints": list(JOINTS.values()),
        "target_selection": "highest detector bbox score; rows marked needs_manual_target_review=true require later review",
        "confidence_thresholds": {"usable": args.usable_score, "high": args.high_score},
        "interpretation_boundary": "AI-assisted Sapiens2 pseudo-labels are an operational reference only. They must not be used to report Sapiens2 accuracy or to call agreement with Sapiens2 an independent model-accuracy comparison.",
        "confidence_summary": confidence_summary,
    }
    (output / "reference_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
