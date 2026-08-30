"""Evaluate saved single-view COCO-17 predictions against Sapiens2 pseudo-labels.

This is deliberately an agreement analysis.  The supplied Sapiens2 points are
not independent ground truth, so no output from this tool may be called model
accuracy.  Candidate selection is reference-guided to separate pose agreement
from the unrelated multi-person detector-association problem; the selected
candidate and every per-point decision are written for audit.
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


JOINTS = {11: "left_hip", 12: "right_hip", 13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--reference-selection-review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--usable-score", type=float, default=0.25)
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        metavar="NAME|MODEL|LEFT_JSON|RIGHT_JSON|LEFT_ROT|RIGHT_ROT|SPACE",
        help="SPACE is model (apply inverse rotation) or raw (already original fisheye).",
    )
    return parser.parse_args()


def parse_condition(value: str) -> dict:
    fields = value.split("|")
    if len(fields) != 7:
        raise ValueError("Each --condition must have seven | separated fields.")
    name, model, left, right, left_rotation, right_rotation, space = fields
    if model not in {"yolo26x_pose", "pmpose", "probpose", "bboxmaskpose", "sapiens2"}:
        raise ValueError(f"Unsupported saved-prediction model: {model}")
    if space not in {"model", "raw"}:
        raise ValueError("Prediction coordinate SPACE must be model or raw.")
    return {"name": name, "model": model, "left": Path(left), "right": Path(right),
            "left_rotation": left_rotation, "right_rotation": right_rotation, "space": space}


def load_reference(path: Path) -> tuple[dict, dict]:
    by_image: dict[str, dict[str, dict]] = defaultdict(dict)
    image_metadata: dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = row["image_id"]
            image_metadata[image_id] = {key: row[key] for key in ("file_name", "side", "condition")}
            by_image[image_id][row["joint_subject_anatomy"]] = row
    if len(by_image) != 120:
        raise RuntimeError(f"Expected 120 reference images, found {len(by_image)}")
    return by_image, image_metadata


def load_ambiguous_images(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["image_id"] for row in rows if row["needs_manual_target_review"].lower() == "true"}


def raw_points(person, rotation: str, space: str) -> dict[str, tuple[float, float, float]]:
    output = {}
    for index, joint in JOINTS.items():
        if len(person.keypoints) <= index:
            continue
        x, y, score = [float(value) for value in person.keypoints[index]]
        if space == "model":
            x, y = model_to_raw_point(x, y, 1920, 1080, rotation)
        output[joint] = (x, y, score)
    return output


def select_reference_guided_candidate(persons, reference, rotation, space, usable_score):
    """Pick the candidate with most common usable joints, then lowest median error.

    This is an oracle identity match.  It is intentionally not a deploy-time
    person-selection algorithm and is reported as such in the output metadata.
    """
    choices = []
    for person in persons:
        points = raw_points(person, rotation, space)
        errors = []
        for joint, ref in reference.items():
            if ref["reference_usable"] != "true" or joint not in points:
                continue
            x, y, score = points[joint]
            if score < usable_score:
                continue
            errors.append(float(np.hypot(x - float(ref["x_px_raw_fisheye"]), y - float(ref["y_px_raw_fisheye"]))))
        # Descending common-joint count, ascending median distance, descending detector score.
        choices.append((len(errors), float(np.median(errors)) if errors else float("inf"), -float(person.bbox_score), person, points))
    if not choices:
        return None, {}, "no_model_person", 0, None
    choices.sort(key=lambda item: (-item[0], item[1], item[2]))
    common, median, _, person, points = choices[0]
    if common == 0:
        return person, points, "no_common_usable_joint", common, None
    return person, points, "reference_guided_max_common_then_min_median_error", common, median


def pck(errors: list[float], threshold: float) -> float | None:
    return None if not errors else float(np.mean(np.asarray(errors) <= threshold))


def summarize(rows: list[dict], name: str, subset: str) -> list[dict]:
    filtered = [row for row in rows if row["reference_subset"] == subset]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in filtered:
        groups[(row["distance_condition"], row["camera_side"], row["joint"])].append(row)
    output = []
    for (condition, side, joint), items in sorted(groups.items()):
        reference_usable = [item for item in items if item["reference_usable"]]
        compared = [item for item in reference_usable if item["model_usable"]]
        errors = [item["error_px"] for item in compared]
        scores = [item["model_confidence"] for item in compared]
        output.append({
            "condition_name": name,
            "reference_subset": subset,
            "distance_condition": condition,
            "camera_side": side,
            "joint": joint,
            "reference_usable_points": len(reference_usable),
            "model_usable_points": len(compared),
            "relative_coverage": len(compared) / len(reference_usable) if reference_usable else None,
            "mean_relative_error_px": float(np.mean(errors)) if errors else None,
            "median_relative_error_px": float(np.median(errors)) if errors else None,
            "p90_relative_error_px": float(np.percentile(errors, 90)) if errors else None,
            "relative_pck_25px": pck(errors, 25.0),
            "relative_pck_50px": pck(errors, 50.0),
            "mean_model_confidence_when_compared": float(np.mean(scores)) if scores else None,
        })
    return output


def main() -> int:
    args = parse_args()
    if not 0 <= args.usable_score <= 1:
        raise ValueError("--usable-score must be in [0, 1]")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    reference, image_metadata = load_reference(args.reference_csv.resolve())
    ambiguous_images = load_ambiguous_images(args.reference_selection_review.resolve())
    all_rows, all_summaries, manifest = [], [], []
    for definition in [parse_condition(value) for value in args.condition]:
        if definition["model"] == "sapiens2":
            raise ValueError("Do not evaluate Sapiens2 against labels exported from that same run.")
        manifest.append({key: str(value) if isinstance(value, Path) else value for key, value in definition.items()})
        records_by_side = {}
        for side, path, rotation in (("left", definition["left"], definition["left_rotation"]), ("right", definition["right"], definition["right_rotation"])):
            records = _records(path.resolve(), definition["model"], "raw_keypoint")
            records_by_side[side] = {record["name"]: (record, rotation) for record in records}
        condition_rows = []
        for image_id, joints in sorted(reference.items()):
            meta = image_metadata[image_id]
            side, file_name = meta["side"], meta["file_name"]
            if file_name not in records_by_side[side]:
                raise RuntimeError(f"{definition['name']}: missing {side}/{file_name}")
            record, rotation = records_by_side[side][file_name]
            selected, points, selection_method, common, median = select_reference_guided_candidate(
                record["persons"], joints, rotation, definition["space"], args.usable_score
            )
            for joint, ref in joints.items():
                ref_usable = ref["reference_usable"] == "true"
                point = points.get(joint)
                model_usable = bool(point is not None and point[2] >= args.usable_score)
                error = None
                if ref_usable and model_usable:
                    error = float(np.hypot(point[0] - float(ref["x_px_raw_fisheye"]), point[1] - float(ref["y_px_raw_fisheye"])))
                row = {
                    "condition_name": definition["name"], "model": definition["model"], "coordinate_space": definition["space"],
                    "image_id": image_id, "file_name": file_name, "camera_side": side, "distance_condition": meta["condition"],
                    "joint": joint, "reference_usable": ref_usable, "reference_confidence": float(ref["sapiens2_confidence"]),
                    "model_person_count": len(record["persons"]), "selected_person_id": "" if selected is None else selected.person_id,
                    "selected_bbox_score": "" if selected is None else selected.bbox_score, "selection_method": selection_method,
                    "selection_common_usable_joints": common, "selection_median_relative_error_px": median,
                    "model_usable": model_usable, "model_confidence": "" if point is None else point[2],
                    "model_x_px_raw_fisheye": "" if point is None else point[0], "model_y_px_raw_fisheye": "" if point is None else point[1],
                    "error_px": error,
                }
                # Report both full data and the subset without reference target ambiguity.
                for subset in ("all_reference_images", "reference_target_unambiguous_only"):
                    if subset == "reference_target_unambiguous_only" and image_id in ambiguous_images:
                        continue
                    copied = dict(row)
                    copied["reference_subset"] = subset
                    condition_rows.append(copied)
        # Ambiguity is injected after loading because the point table deliberately has no detector fields.
        all_rows.extend(condition_rows)
        all_summaries.extend(summarize(condition_rows, definition["name"], "all_reference_images"))
        all_summaries.extend(summarize(condition_rows, definition["name"], "reference_target_unambiguous_only"))

    output.mkdir(parents=True)
    with (output / "per_keypoint_relative_error_all_reference.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = list(all_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([row for row in all_rows if row["reference_subset"] == "all_reference_images"])
    with (output / "summary_by_model_distance_side_joint.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_summaries[0]))
        writer.writeheader()
        writer.writerows(all_summaries)
    metadata = {
        "reference_csv": str(args.reference_csv.resolve()),
        "reference_selection_review": str(args.reference_selection_review.resolve()),
        "reference_target_ambiguous_images": len(ambiguous_images),
        "conditions": manifest,
        "scoring": {
            "usable_score": args.usable_score,
            "coordinate_space": "original 1920x1080 raw fisheye pixels",
            "candidate_selection": "reference-guided max shared usable joints then lowest median point distance; oracle analysis only, not deployment association",
            "metrics": "relative error and relative PCK measure agreement with Sapiens2 pseudo-labels, not accuracy",
        },
        "limitations": [
            "Sapiens2 is the label source and is excluded from scoring to avoid circular zero error.",
            "The current reference source and several compared runs use the historical CW/CCW model-input condition; results cannot restore the retired stereo/rotation experiment as a deployable conclusion.",
            "Both all-image and reference-target-unambiguous summaries are emitted; the latter excludes images with ambiguous Sapiens2 target selection.",
        ],
    }
    (output / "evaluation_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "per_keypoint_rows": sum(row["reference_subset"] == "all_reference_images" for row in all_rows), "summary_rows": len(all_summaries)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
