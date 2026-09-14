from __future__ import annotations
import json
import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for candidate in (str(ROOT), str(TOOLS)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
import correct_mask2former_stereo_pair_latency as tool  # noqa: E402
def record(pair_id: int, repeat_index: int, view: str, milliseconds: float) -> dict:
    return {
        "pair_id": pair_id,
        "repeat_index": repeat_index,
        "view": view,
        "full_online_semantic_ms": milliseconds,
    }
def matrix() -> dict:
    rows = []
    for index in range(27):
        rows.append(
            {
                "combination_id": f"combo_{index:02d}",
                "matcher": "sgbm",
                "plane_method": "dense_ransac",
                "temporal_mode": "direct_current_frame",
                "measurement_scope": "direct_chain",
                "estimated_end_to_end_latency_ms": None if index == 0 else float(index),
                "estimated_end_to_end_latency_p95_ms": None if index == 0 else float(index + 10),
                "semantic_online_p50_ms": 10.0,
                "semantic_online_p95_ms": 12.0,
                "measured_end_to_end_latency_ms": None,
                "realtime_compatible": False,
                "vo_module_status": "unavailable",
            }
        )
    return {"schema_version": "historical_v1", "row_count": 27, "rows": rows, "semantic_online_p50_ms": 10.0, "semantic_online_p95_ms": 12.0}
class StereoPairLatencyTests(unittest.TestCase):
    def test_exact_pairing_and_sum(self) -> None:
        records = [
            record(1, 0, "right", 20.0), record(0, 0, "left", 10.0),
            record(1, 0, "left", 11.0), record(0, 0, "right", 12.0),
        ]
        paired = tool.make_stereo_pair_records(records)
        self.assertEqual([(row["pair_id"], row["repeat_index"]) for row in paired], [(0, 0), (1, 0)])
        self.assertEqual(paired[0]["stereo_pair_full_online_semantic_ms"], 22.0)
        self.assertEqual(paired[1]["stereo_pair_full_online_semantic_ms"], 31.0)
    def test_missing_or_duplicate_view_fails(self) -> None:
        with self.assertRaises(tool.CorrectionError):
            tool.make_stereo_pair_records([record(0, 0, "left", 1.0)])
        with self.assertRaises(tool.CorrectionError):
            tool.make_stereo_pair_records([
                record(0, 0, "left", 1.0), record(0, 0, "left", 2.0), record(0, 0, "right", 3.0)
            ])
    def test_linear_percentile_and_summary(self) -> None:
        summary = tool.summarize_ms([10.0, 20.0, 30.0, 40.0])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["median_ms"], 25.0)
        self.assertAlmostEqual(summary["p95_ms"], 38.5)
    def test_corrected_matrix_uses_pair_not_single_view_latency(self) -> None:
        paired_summary = tool.summarize_ms([100.0, 120.0, 140.0])
        corrected = tool.correct_matrix(matrix(), paired_summary)
        self.assertEqual(corrected["row_count"], 27)
        self.assertEqual(len(corrected["rows"]), 27)
        self.assertEqual(corrected["rows"][0]["estimated_module_sum_with_stereo_pair_semantic_p50_ms"], None)
        self.assertEqual(corrected["rows"][1]["estimated_module_sum_with_stereo_pair_semantic_p50_ms"], 121.0)
        self.assertEqual(corrected["rows"][1]["estimated_module_sum_with_stereo_pair_semantic_p95_ms"], 149.0)
        self.assertEqual(corrected["rows"][1]["measured_end_to_end_latency_ms"], None)
        self.assertEqual(corrected["rows"][1]["semantic_online_p50_ms"], 10.0)
    def test_invalid_source_matrix_fails(self) -> None:
        with self.assertRaises(tool.CorrectionError):
            tool.validate_source_matrix({"row_count": 0, "rows": []})
if __name__ == "__main__":
    unittest.main(verbosity=2)
