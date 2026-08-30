"""Compare two Sapiens2-308 foot pseudo-label exports on identical images."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


KEY_FIELDS = ("image_id", "joint_subject_anatomy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-a", type=Path, required=True)
    parser.add_argument("--condition-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-a", default="condition_a")
    parser.add_argument("--label-b", default="condition_b")
    return parser.parse_args()


def load(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {tuple(row[field] for field in KEY_FIELDS): row for row in rows}
    if len(indexed) != len(rows):
        raise RuntimeError(f"Duplicate image/joint rows in {path}")
    return indexed


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], group_fields: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[field]) for field in group_fields)].append(row)
    output = []
    for key, items in sorted(groups.items()):
        common = [item for item in items if item["both_usable"]]
        errors = np.asarray([item["raw_pixel_difference"] for item in common], dtype=np.float64)
        record = {field: value for field, value in zip(group_fields, key)}
        record.update({
            "candidate_points": len(items),
            "condition_a_usable_points": sum(item["condition_a_usable"] for item in items),
            "condition_b_usable_points": sum(item["condition_b_usable"] for item in items),
            "both_usable_points": len(common),
            "both_usable_rate": len(common) / len(items),
            "median_raw_pixel_difference": float(np.median(errors)) if errors.size else None,
            "mean_raw_pixel_difference": float(np.mean(errors)) if errors.size else None,
            "p90_raw_pixel_difference": float(np.percentile(errors, 90)) if errors.size else None,
        })
        output.append(record)
    return output


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    condition_a = load(args.condition_a.resolve())
    condition_b = load(args.condition_b.resolve())
    if set(condition_a) != set(condition_b):
        raise RuntimeError("The two exports do not contain identical image/joint keys")

    rows = []
    for key in sorted(condition_a):
        a, b = condition_a[key], condition_b[key]
        a_usable = a["reference_usable"].lower() == "true"
        b_usable = b["reference_usable"].lower() == "true"
        difference = None
        if a_usable and b_usable:
            difference = float(np.hypot(
                float(a["x_px_raw_fisheye"]) - float(b["x_px_raw_fisheye"]),
                float(a["y_px_raw_fisheye"]) - float(b["y_px_raw_fisheye"]),
            ))
        rows.append({
            "image_id": a["image_id"],
            "file_name": a["file_name"],
            "side": a["side"],
            "condition": a["condition"],
            "joint": a["joint_subject_anatomy"],
            "condition_a_usable": a_usable,
            "condition_a_confidence": float(a["sapiens2_confidence"]),
            "condition_b_usable": b_usable,
            "condition_b_confidence": float(b["sapiens2_confidence"]),
            "both_usable": a_usable and b_usable,
            "raw_pixel_difference": difference,
        })

    overall = summarize(rows, tuple())
    by_distance = summarize(rows, ("condition",))
    detail = summarize(rows, ("condition", "side", "joint"))
    output.mkdir(parents=True)
    write_csv(output / "per_point_orientation_comparison.csv", rows)
    write_csv(output / "summary_overall.csv", overall)
    write_csv(output / "summary_by_distance.csv", by_distance)
    write_csv(output / "summary_by_distance_side_joint.csv", detail)
    metadata = {
        "experiment_id": "E20260830-A8_Sapiens2_foot_orientation_matched_control",
        "condition_a": {"label": args.label_a, "path": str(args.condition_a.resolve())},
        "condition_b": {"label": args.label_b, "path": str(args.condition_b.resolve())},
        "comparison": "Same image and same Sapiens2-308 anatomical point after both outputs are restored to original fisheye pixels.",
        "interpretation": "Pixel difference measures sensitivity to model-input rotation, not error against ground truth.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "points": len(rows), "overall": overall[0]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
