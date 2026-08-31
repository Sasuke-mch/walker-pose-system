#!/usr/bin/env python3
"""Paired C0/C1 agreement summary for a frozen detector-box expansion test.

It reports Sapiens2-relative agreement only.  Coverage is calculated over the
same usable pseudo-label points; error is reported both for each condition's
available points and, separately, for their paired common points.  The latter
prevents coverage changes from being mistaken for error improvement.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _bool(v: str) -> bool:
    return v.strip().lower() == "true"


def _metrics(items: list[dict[str, str]], c0: str, c1: str) -> dict[str, float | int | None]:
    refs = [x for x in items if _bool(x["reference_usable"])]
    left = { (x["image_id"], x["joint"]): x for x in refs if x["condition_name"] == c0 }
    right = { (x["image_id"], x["joint"]): x for x in refs if x["condition_name"] == c1 }
    keys = sorted(set(left) | set(right))
    if not keys:
        return {}
    base = [left[k] for k in keys if k in left and _bool(left[k]["model_usable"])]
    expanded = [right[k] for k in keys if k in right and _bool(right[k]["model_usable"])]
    common = [(left[k], right[k]) for k in keys if k in left and k in right and _bool(left[k]["model_usable"]) and _bool(right[k]["model_usable"])]
    base_err = np.asarray([float(x["error_px"]) for x in base], dtype=float)
    exp_err = np.asarray([float(x["error_px"]) for x in expanded], dtype=float)
    common0 = np.asarray([float(a["error_px"]) for a, _ in common], dtype=float)
    common1 = np.asarray([float(b["error_px"]) for _, b in common], dtype=float)
    denom = len(keys)

    def mean(a: np.ndarray) -> float | None: return float(a.mean()) if a.size else None
    def pck(a: np.ndarray) -> float | None: return float((a <= 25).mean()) if a.size else None
    def delta(a: float | None, b: float | None) -> float | None: return None if a is None or b is None else b - a
    def pct(a: float | None, b: float | None) -> float | None: return None if a in (None, 0) or b is None else (b - a) / a * 100.0

    base_cov, exp_cov = len(base) / denom, len(expanded) / denom
    base_mean, exp_mean = mean(base_err), mean(exp_err)
    b_pck, e_pck = pck(base_err), pck(exp_err)
    cm0, cm1 = mean(common0), mean(common1)
    cp0, cp1 = pck(common0), pck(common1)
    return {
        "reference_usable_points": denom,
        "c0_usable_points": len(base), "c1_usable_points": len(expanded),
        "c0_relative_coverage": base_cov, "c1_relative_coverage": exp_cov,
        "coverage_change_percentage_points": (exp_cov - base_cov) * 100.0,
        "coverage_relative_change_percent": pct(base_cov, exp_cov),
        "c0_mean_relative_error_px": base_mean, "c1_mean_relative_error_px": exp_mean,
        "mean_error_change_px_c1_minus_c0": delta(base_mean, exp_mean),
        "mean_error_relative_change_percent": pct(base_mean, exp_mean),
        "c0_relative_pck25": b_pck, "c1_relative_pck25": e_pck,
        "pck25_change_percentage_points": None if b_pck is None or e_pck is None else (e_pck - b_pck) * 100.0,
        "paired_common_points": len(common),
        "paired_c0_mean_relative_error_px": cm0, "paired_c1_mean_relative_error_px": cm1,
        "paired_mean_error_change_px_c1_minus_c0": delta(cm0, cm1),
        "paired_mean_error_relative_change_percent": pct(cm0, cm1),
        "paired_c0_relative_pck25": cp0, "paired_c1_relative_pck25": cp1,
        "paired_pck25_change_percentage_points": None if cp0 is None or cp1 is None else (cp1 - cp0) * 100.0,
        "became_usable": sum(k in right and _bool(right[k]["model_usable"]) and (k not in left or not _bool(left[k]["model_usable"])) for k in keys),
        "became_unusable": sum(k in left and _bool(left[k]["model_usable"]) and (k not in right or not _bool(right[k]["model_usable"])) for k in keys),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reviewed-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--c0", default="C0_current_yolo_top1")
    p.add_argument("--c1", default="C1_current_yolo_expanded")
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    with args.reviewed_csv.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [x for x in rows if x["condition_name"] in {args.c0, args.c1}]
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"],)].append(row)
        grouped[(row["model"], row["distance_condition"])].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        record = {"model": key[0], "distance_condition": key[1] if len(key) > 1 else "all"}
        record.update(_metrics(values, args.c0, args.c1))
        output.append(record)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "c0_vs_c1_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(output[0])); w.writeheader(); w.writerows(output)
    metadata = {
        "source": str(args.reviewed_csv.resolve()), "c0": args.c0, "c1": args.c1,
        "interpretation": "Sapiens2-relative agreement, not absolute accuracy. A negative error change means lower pixel deviation.",
        "critical_test": "A coverage-only gain is insufficient evidence of a box-limited system. Inspect paired common-point error and PCK alongside coverage.",
    }
    (args.output_dir / "comparison_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(output), "output": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
