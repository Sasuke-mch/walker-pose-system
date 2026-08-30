"""Re-summarize saved 2D agreement after targeted visual pseudo-label review.

The input table contains agreement with Sapiens2 pseudo-labels.  This tool does
not turn those labels into ground truth.  It clears only target identity for
images explicitly reviewed as correct and excludes images whose reviewed lower-
body reference is missing.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-keypoint-csv", type=Path, required=True)
    parser.add_argument("--review-results-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def metric_rows(rows: list[dict[str, str]], group_fields: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, items in sorted(groups.items()):
        reference = [item for item in items if as_bool(item["reference_usable"])]
        compared = [item for item in reference if as_bool(item["model_usable"])]
        errors = np.asarray([float(item["error_px"]) for item in compared], dtype=np.float64)
        record = {field: value for field, value in zip(group_fields, key)}
        record.update({
            "reference_usable_points": len(reference),
            "model_usable_points": len(compared),
            "relative_coverage": len(compared) / len(reference) if reference else None,
            "mean_relative_error_px": float(np.mean(errors)) if errors.size else None,
            "median_relative_error_px": float(np.median(errors)) if errors.size else None,
            "p90_relative_error_px": float(np.percentile(errors, 90)) if errors.size else None,
            "relative_pck_25px": float(np.mean(errors <= 25.0)) if errors.size else None,
            "relative_pck_50px": float(np.mean(errors <= 50.0)) if errors.size else None,
        })
        output.append(record)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    rows = read_csv(args.per_keypoint_csv.resolve())
    reviews = {row["image_id"]: row for row in read_csv(args.review_results_csv.resolve())}
    rejected = {image_id for image_id, row in reviews.items() if row["lower_body_reference_status"] == "rejected_missing"}
    cleared_targets = {image_id for image_id, row in reviews.items() if row["target_selection_visual_status"] == "correct"}

    included = []
    for row in rows:
        if row["image_id"] in rejected:
            continue
        copied = dict(row)
        copied["reviewed_target_identity"] = row["image_id"] in cleared_targets
        copied["reviewed_reference_status"] = reviews.get(row["image_id"], {}).get("lower_body_reference_status", "not_targeted_for_visual_review")
        copied["evaluation_subset"] = "target_reviewed_and_missing_reference_excluded"
        included.append(copied)

    summary_overall = metric_rows(included, ("condition_name", "model"))
    summary_distance = metric_rows(included, ("condition_name", "model", "distance_condition"))
    summary_detail = metric_rows(included, ("condition_name", "model", "distance_condition", "camera_side", "joint"))

    output.mkdir(parents=True)
    write_csv(output / "per_keypoint_reviewed_subset.csv", included)
    write_csv(output / "summary_overall.csv", summary_overall)
    write_csv(output / "summary_by_distance.csv", summary_distance)
    write_csv(output / "summary_by_distance_side_joint.csv", summary_detail)
    metadata = {
        "source_per_keypoint_csv": str(args.per_keypoint_csv.resolve()),
        "visual_review_results_csv": str(args.review_results_csv.resolve()),
        "source_images": len({row["image_id"] for row in rows}),
        "included_images": len({row["image_id"] for row in included}),
        "visually_reviewed_images": len(reviews),
        "target_identity_cleared_images": len(cleared_targets),
        "excluded_missing_reference_images": sorted(rejected),
        "interpretation": "Agreement with Sapiens2 pseudo-labels after targeted identity review; not absolute 2D accuracy.",
        "selection_note": "All 12 previously target-ambiguous images were visually reviewed and selected the intended subject. Only reviewed missing-reference images are excluded.",
    }
    (output / "evaluation_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "included_rows": len(included), "summary_overall_rows": len(summary_overall)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
