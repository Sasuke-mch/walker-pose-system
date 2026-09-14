#!/usr/bin/env python3
"""Correct a single-view semantic latency addition to a stereo-pair accounting.
This is a read-only accounting correction.  It groups existing Mask2Former
single-image timing records by (pair_id, repeat_index), requires exactly one
left and one right record in every group, and creates a new versioned report.
It never runs inference or changes any upstream geometry result.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence
SCHEMA_VERSION = "mask2former_stereo_pair_latency_correction_v2"
EXPERIMENT_ID = "G20260913_mask2former_stereo_pair_latency_correction_v2"
REQUIRED_VIEWS = ("left", "right")
REQUIRED_RECORD_FIELDS = ("pair_id", "repeat_index", "view", "full_online_semantic_ms")
PAIR_BOUNDARY = (
    "A stereo-pair semantic time is the sum of one measured left-image and one measured right-image "
    "Mask2Former pass from the same pair_id and repeat_index. It is still an estimated module input "
    "to a larger chain, not a measured end-to-end latency or a physical ground-accuracy result."
)
class CorrectionError(RuntimeError):
    """Raised when frozen input accounting cannot be corrected unambiguously."""
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-inference-records", type=Path, required=True)
    parser.add_argument("--source-matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser
def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"missing JSON input: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing JSONL input: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                raise CorrectionError(f"blank JSONL line at {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorrectionError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise CorrectionError(f"JSONL row at {path}:{line_number} is not an object")
            rows.append(row)
    if not rows:
        raise CorrectionError(f"no records in {path}")
    return rows
def finite_nonnegative(value: Any, field: str, context: str) -> float:
    if isinstance(value, bool):
        raise CorrectionError(f"{context}: {field} must be numeric, got boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CorrectionError(f"{context}: {field} is not numeric: {value!r}") from exc
    if not math.isfinite(number) or number < 0.0:
        raise CorrectionError(f"{context}: {field} must be finite and nonnegative, got {number!r}")
    return number
def linear_percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise CorrectionError("cannot calculate percentile from an empty value list")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"invalid percentile fraction: {fraction}")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)
def summarize_ms(values: Sequence[float]) -> dict[str, float | int]:
    numeric = sorted(float(value) for value in values)
    if not numeric:
        raise CorrectionError("cannot summarize zero timing values")
    return {
        "count": len(numeric),
        "mean_ms": float(statistics.fmean(numeric)),
        "median_ms": linear_percentile(numeric, 0.50),
        "p90_ms": linear_percentile(numeric, 0.90),
        "p95_ms": linear_percentile(numeric, 0.95),
        "min_ms": float(numeric[0]),
        "max_ms": float(numeric[-1]),
    }
def make_stereo_pair_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for index, record in enumerate(records):
        context = f"record index {index}"
        missing = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
        if missing:
            raise CorrectionError(f"{context}: missing required fields: {missing}")
        if isinstance(record["pair_id"], bool) or isinstance(record["repeat_index"], bool):
            raise CorrectionError(f"{context}: pair_id and repeat_index must be integer-like values")
        pair_id = int(record["pair_id"])
        repeat_index = int(record["repeat_index"])
        if pair_id < 0 or repeat_index < 0:
            raise CorrectionError(f"{context}: pair_id and repeat_index must be nonnegative")
        if record["view"] not in REQUIRED_VIEWS:
            raise CorrectionError(f"{context}: invalid view {record['view']!r}")
        finite_nonnegative(record["full_online_semantic_ms"], "full_online_semantic_ms", context)
        key = (pair_id, repeat_index)
        views = grouped.setdefault(key, {})
        view = str(record["view"])
        if view in views:
            raise CorrectionError(f"duplicate {view} record for pair_id={pair_id}, repeat_index={repeat_index}")
        views[view] = record
    paired: list[dict[str, Any]] = []
    for pair_id, repeat_index in sorted(grouped):
        views = grouped[(pair_id, repeat_index)]
        observed = sorted(views)
        if observed != list(REQUIRED_VIEWS):
            raise CorrectionError(
                f"pair_id={pair_id}, repeat_index={repeat_index}: expected exactly {REQUIRED_VIEWS}, got {observed}"
            )
        left_ms = finite_nonnegative(
            views["left"]["full_online_semantic_ms"], "full_online_semantic_ms", f"pair_id={pair_id} left"
        )
        right_ms = finite_nonnegative(
            views["right"]["full_online_semantic_ms"], "full_online_semantic_ms", f"pair_id={pair_id} right"
        )
        paired.append(
            {
                "schema_version": SCHEMA_VERSION,
                "pair_id": pair_id,
                "repeat_index": repeat_index,
                "left_full_online_semantic_ms": left_ms,
                "right_full_online_semantic_ms": right_ms,
                "stereo_pair_full_online_semantic_ms": left_ms + right_ms,
                "pairing_rule": "exactly one left plus one right record with identical pair_id and repeat_index",
                "interpretation_boundary": PAIR_BOUNDARY,
            }
        )
    if len(paired) * 2 != len(records):
        raise CorrectionError("single-view records are not exhausted by exact left/right stereo pairing")
    return paired
def validate_source_matrix(matrix: Any) -> list[dict[str, Any]]:
    if not isinstance(matrix, dict):
        raise CorrectionError("source matrix must be a JSON object")
    rows = matrix.get("rows")
    if not isinstance(rows, list) or len(rows) != 27:
        raise CorrectionError(f"source matrix must contain exactly 27 rows, got {None if not isinstance(rows, list) else len(rows)}")
    if matrix.get("row_count") != 27:
        raise CorrectionError(f"source matrix row_count must equal 27, got {matrix.get('row_count')!r}")
    identifiers = [row.get("combination_id") for row in rows if isinstance(row, dict)]
    if len(identifiers) != 27 or len(set(identifiers)) != 27 or any(not isinstance(value, str) for value in identifiers):
        raise CorrectionError("source matrix has missing or duplicate combination_id values")
    for row in rows:
        if "estimated_end_to_end_latency_ms" not in row or "estimated_end_to_end_latency_p95_ms" not in row:
            raise CorrectionError(f"row {row.get('combination_id')!r} lacks base estimated latency fields")
        if "semantic_online_p50_ms" not in row or "semantic_online_p95_ms" not in row:
            raise CorrectionError(f"row {row.get('combination_id')!r} is not the v1 single-view semantic matrix")
    return rows
def corrected_latency(base: Any, semantic_pair_ms: float) -> float | None:
    if base is None:
        return None
    return finite_nonnegative(base, "base estimated latency", "source matrix row") + semantic_pair_ms
def correct_matrix(source_matrix: dict[str, Any], pair_summary: dict[str, float | int]) -> dict[str, Any]:
    validate_source_matrix(source_matrix)
    corrected = copy.deepcopy(source_matrix)
    corrected["schema_version"] = SCHEMA_VERSION + "_combination_matrix"
    corrected["experiment_id"] = EXPERIMENT_ID
    corrected["correction_kind"] = "replace_single_view_semantic_addition_with_exact_paired_left_plus_right_addition"
    corrected["source_matrix_schema_version"] = source_matrix.get("schema_version")
    corrected["single_view_semantic_latency_ms"] = {
        "prior_p50_ms": source_matrix.get("semantic_online_p50_ms"),
        "prior_p95_ms": source_matrix.get("semantic_online_p95_ms"),
        "note": "Retained only as the historical single-image measurement; never use it as a stereo-pair semantic total.",
    }
    corrected["stereo_pair_semantic_latency_ms"] = dict(pair_summary)
    corrected["measured_end_to_end_latency_available"] = False
    corrected["interpretation_boundary"] = PAIR_BOUNDARY
    for row in corrected["rows"]:
        row["single_view_semantic_latency_ms"] = {
            "historical_p50_ms": source_matrix.get("semantic_online_p50_ms"),
            "historical_p95_ms": source_matrix.get("semantic_online_p95_ms"),
        }
        row["stereo_pair_semantic_latency_ms"] = {
            "p50_ms": pair_summary["median_ms"],
            "p95_ms": pair_summary["p95_ms"],
            "count": pair_summary["count"],
            "pairing_rule": "left + right with identical pair_id and repeat_index",
        }
        row["estimated_module_sum_with_stereo_pair_semantic_p50_ms"] = corrected_latency(
            row["estimated_end_to_end_latency_ms"], float(pair_summary["median_ms"])
        )
        row["estimated_module_sum_with_stereo_pair_semantic_p95_ms"] = corrected_latency(
            row["estimated_end_to_end_latency_p95_ms"], float(pair_summary["p95_ms"])
        )
        row["latency_kind"] = "estimated_module_sum_with_stereo_pair_semantic"
        row["measured_end_to_end_latency_ms"] = None
        row["stereo_pair_latency_boundary"] = PAIR_BOUNDARY
    return corrected
def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    columns = [
        "combination_id", "matcher", "plane_method", "temporal_mode", "measurement_scope",
        "estimated_end_to_end_latency_ms", "estimated_end_to_end_latency_p95_ms",
        "historical_single_view_semantic_p50_ms", "historical_single_view_semantic_p95_ms",
        "stereo_pair_semantic_p50_ms", "stereo_pair_semantic_p95_ms",
        "estimated_module_sum_with_stereo_pair_semantic_p50_ms",
        "estimated_module_sum_with_stereo_pair_semantic_p95_ms",
        "measured_end_to_end_latency_ms", "realtime_compatible", "vo_module_status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "combination_id": row["combination_id"],
                    "matcher": row["matcher"],
                    "plane_method": row["plane_method"],
                    "temporal_mode": row["temporal_mode"],
                    "measurement_scope": row["measurement_scope"],
                    "estimated_end_to_end_latency_ms": row["estimated_end_to_end_latency_ms"],
                    "estimated_end_to_end_latency_p95_ms": row["estimated_end_to_end_latency_p95_ms"],
                    "historical_single_view_semantic_p50_ms": row["single_view_semantic_latency_ms"]["historical_p50_ms"],
                    "historical_single_view_semantic_p95_ms": row["single_view_semantic_latency_ms"]["historical_p95_ms"],
                    "stereo_pair_semantic_p50_ms": row["stereo_pair_semantic_latency_ms"]["p50_ms"],
                    "stereo_pair_semantic_p95_ms": row["stereo_pair_semantic_latency_ms"]["p95_ms"],
                    "estimated_module_sum_with_stereo_pair_semantic_p50_ms": row[
                        "estimated_module_sum_with_stereo_pair_semantic_p50_ms"
                    ],
                    "estimated_module_sum_with_stereo_pair_semantic_p95_ms": row[
                        "estimated_module_sum_with_stereo_pair_semantic_p95_ms"
                    ],
                    "measured_end_to_end_latency_ms": row["measured_end_to_end_latency_ms"],
                    "realtime_compatible": row["realtime_compatible"],
                    "vo_module_status": row["vo_module_status"],
                }
            )
def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")
    records = read_jsonl(args.per_inference_records)
    pairs = make_stereo_pair_records(records)
    pair_summary = summarize_ms([row["stereo_pair_full_online_semantic_ms"] for row in pairs])
    source_matrix = read_json(args.source_matrix)
    corrected_matrix = correct_matrix(source_matrix, pair_summary)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output_dir / "stereo_pair_semantic_latency_records.jsonl", pairs)
    write_json(args.output_dir / "corrected_combination_matrix_27.json", corrected_matrix)
    write_csv(args.output_dir / "corrected_combination_matrix_27.csv", corrected_matrix["rows"])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_readonly_accounting_correction",
        "input_single_view_record_count": len(records),
        "stereo_pair_record_count": len(pairs),
        "stereo_pair_semantic_latency_ms": pair_summary,
        "source_matrix": str(args.source_matrix),
        "source_matrix_row_count": len(corrected_matrix["rows"]),
        "corrected_matrix": str(args.output_dir / "corrected_combination_matrix_27.json"),
        "measured_end_to_end_latency_ms": None,
        "interpretation_boundary": PAIR_BOUNDARY,
        "not_run": ["Mask2Former inference", "SGBM", "IGEV", "DynamicStereo", "plane fitting", "optical flow", "VO"],
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary
def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    stats = summary["stereo_pair_semantic_latency_ms"]
    print(f"status={summary['status']}")
    print(f"single_view_records={summary['input_single_view_record_count']}")
    print(f"stereo_pair_records={summary['stereo_pair_record_count']}")
    print(f"stereo_pair_p50_ms={stats['median_ms']:.6f}")
    print(f"stereo_pair_p95_ms={stats['p95_ms']:.6f}")
    print(f"corrected_matrix_rows={summary['source_matrix_row_count']}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
