from __future__ import annotations

import json
from pathlib import Path
import sys
import subprocess
import unittest

import numpy as np


REALTIME_ROOT = Path(__file__).resolve().parents[1]
if str(REALTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(REALTIME_ROOT))

from pose_app.learned_stereo_protocol import (  # noqa: E402
    LearnedStereoRequest,
    LearnedStereoResult,
    crop_to_original,
    flip_disparity_for_reverse_field,
    invalid_to_nan,
    learned_stereo_request_from_json,
    learned_stereo_request_to_json,
    learned_stereo_result_from_json,
    learned_stereo_result_to_json,
    pad_bottom_right,
    padded_size,
    read_request,
    read_result,
    temporal_window_metadata,
    validate_disparity,
    write_request,
    write_result,
)


PROJECT_ROOT = REALTIME_ROOT.parent
SGBM_BASELINE = (
    PROJECT_ROOT / "research_records" / "engineering_validation"
    / "G20260912_modular_ground_benchmark_v1"
)
TOOLS_ROOT = REALTIME_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))


def sample_request(matcher: str = "igev", window: int = 1) -> LearnedStereoRequest:
    return LearnedStereoRequest(
        matcher_name=matcher,  # type: ignore[arg-type]
        left_image_paths=tuple(f"C:/tmp/left_{index}.png" for index in range(window)),
        right_image_paths=tuple(f"C:/tmp/right_{index}.png" for index in range(window)),
        output_disparity_path="C:/tmp/out/disparity.npy",
        output_metadata_path="C:/tmp/out/result.json",
        runtime_width=960,
        runtime_height=540,
        target_index=0,
        warmup=False,
    )


def sample_result() -> LearnedStereoResult:
    return LearnedStereoResult(
        matcher_name="dynamicstereo",
        disparity_path="C:/tmp/out/disparity.npy",
        valid_mask_path="C:/tmp/out/valid_mask.npy",
        original_width=960,
        original_height=540,
        padded_width=960,
        padded_height=544,
        target_index=0,
        temporal_window_frames=5,
        future_lookahead_frames=4,
        gpu_forward_ms=123.5,
        cpu_worker_wall_ms=456.25,
        peak_gpu_memory_mb=789.0,
        status="ok",
        reasons=(),
        metadata={"gpu_name": "NVIDIA GeForce RTX 5070 Ti Laptop GPU", "weights_url": "https://example/weights.pth"},
    )


class RequestRoundTripTests(unittest.TestCase):
    def test_request_json_round_trip(self):
        request = sample_request("dynamicstereo", window=5)
        payload = learned_stereo_request_to_json(request)
        restored = learned_stereo_request_from_json(json.loads(json.dumps(payload)))
        self.assertEqual(restored, request)
        self.assertEqual(restored.left_image_paths, request.left_image_paths)

    def test_request_file_round_trip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            request = sample_request("igev")
            write_request(path, request)
            self.assertEqual(read_request(path), request)

    def test_result_json_round_trip_preserves_metadata_and_floats(self):
        result = sample_result()
        payload = learned_stereo_result_to_json(result)
        restored = learned_stereo_result_from_json(json.loads(json.dumps(payload)))
        self.assertEqual(restored.matcher_name, result.matcher_name)
        self.assertEqual(restored.status, "ok")
        self.assertEqual(restored.reasons, ())
        self.assertAlmostEqual(restored.gpu_forward_ms or 0.0, 123.5, places=6)
        self.assertAlmostEqual(restored.cpu_worker_wall_ms or 0.0, 456.25, places=6)
        self.assertAlmostEqual(restored.peak_gpu_memory_mb or 0.0, 789.0, places=6)
        self.assertEqual(restored.metadata["gpu_name"], "NVIDIA GeForce RTX 5070 Ti Laptop GPU")

    def test_result_file_round_trip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            result = sample_result()
            write_result(path, result)
            restored = read_result(path)
            self.assertEqual(restored.disparity_path, result.disparity_path)
            self.assertEqual(restored.temporal_window_frames, 5)
            self.assertEqual(restored.future_lookahead_frames, 4)

    def test_request_rejects_unknown_matcher_and_bad_window(self):
        payload = learned_stereo_request_to_json(sample_request("igev"))
        payload["matcher_name"] = "sgbm"
        with self.assertRaises(ValueError):
            learned_stereo_request_from_json(payload)
        # igev is a two-view matcher: a five-frame window must be refused.
        with self.assertRaises(ValueError):
            learned_stereo_request_from_json(learned_stereo_request_to_json(sample_request("igev", window=5)))
        # target_index must index the window.
        payload = learned_stereo_request_to_json(sample_request("dynamicstereo", window=5))
        payload["target_index"] = 5
        with self.assertRaises(ValueError):
            learned_stereo_request_from_json(payload)


class DisparityContractTests(unittest.TestCase):
    def test_float32_2d_disparity_is_accepted_and_preserved(self):
        array = np.full((12, 20), 7.5, dtype=np.float32)
        checked = validate_disparity(array, width=20, height=12)
        self.assertEqual(checked.dtype, np.float32)
        self.assertEqual(checked.shape, (12, 20))
        np.testing.assert_allclose(checked, array)

    def test_non_float32_disparity_is_rejected(self):
        for dtype in (np.float64, np.float16, np.int32, np.uint8):
            with self.subTest(dtype=dtype):
                with self.assertRaises(ValueError):
                    validate_disparity(np.zeros((6, 8), dtype=dtype), width=8, height=6)

    def test_three_dimensional_disparity_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_disparity(np.zeros((6, 8, 1), dtype=np.float32), width=8, height=6)

    def test_wrong_size_disparity_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_disparity(np.zeros((7, 8), dtype=np.float32), width=8, height=6)
        with self.assertRaises(ValueError):
            validate_disparity(np.zeros((6, 9), dtype=np.float32), width=8, height=6)

    def test_nan_invalid_values_are_preserved(self):
        array = np.full((4, 5), 12.0, dtype=np.float32)
        array[1, 2] = np.nan
        array[3, 0] = np.nan
        checked = validate_disparity(array, width=5, height=4)
        self.assertTrue(np.isnan(checked[1, 2]))
        self.assertTrue(np.isnan(checked[3, 0]))
        self.assertEqual(int(np.isnan(checked).sum()), 2)

    def test_non_finite_non_nan_values_become_nan_and_never_zero(self):
        array = np.zeros((2, 3), dtype=np.float32)
        array[0, 0] = np.inf
        array[0, 1] = -np.inf
        array[1, 1] = 0.0
        checked = validate_disparity(array, width=3, height=2)
        self.assertTrue(np.isnan(checked[0, 0]))
        self.assertTrue(np.isnan(checked[0, 1]))
        # A zero that was already present in the file is not silently promoted to
        # a valid disparity by the validator; the worker marks it invalid.
        self.assertEqual(float(checked[1, 1]), 0.0)
        self.assertFalse(bool(np.isfinite(checked[1, 1]) and checked[1, 1] > 1.0))

    def test_invalid_to_nan_turns_zero_and_negative_into_nan(self):
        array = np.asarray([[0.0, -3.0, 0.5, 2.0, np.nan, 200.0]], dtype=np.float32)
        converted = invalid_to_nan(array, minimum_valid=1.0)
        self.assertTrue(np.isnan(converted[0, 0]))
        self.assertTrue(np.isnan(converted[0, 1]))
        self.assertTrue(np.isnan(converted[0, 2]))
        self.assertTrue(np.isnan(converted[0, 4]))
        self.assertAlmostEqual(float(converted[0, 3]), 2.0, places=6)
        self.assertAlmostEqual(float(converted[0, 5]), 200.0, places=6)
        self.assertEqual(int(np.isnan(converted).sum()), 4)


class PaddingTests(unittest.TestCase):
    def test_padded_size_rounds_up_to_a_multiple_of_32(self):
        self.assertEqual(padded_size(960, 540, 32), (960, 544))
        self.assertEqual(padded_size(960, 544, 32), (960, 544))
        self.assertEqual(padded_size(100, 33, 32), (128, 64))

    def test_padding_is_right_and_bottom_only_and_content_is_preserved(self):
        image = np.arange(6 * 4 * 3, dtype=np.uint8).reshape(6, 4, 3)
        padded = pad_bottom_right(image, 4)
        self.assertEqual(padded.shape, (8, 4, 3))
        np.testing.assert_array_equal(padded[:6, :4], image)
        self.assertEqual(int(padded[6:, :, :].sum()), 0)
        self.assertEqual(int(padded[:, 4:, :].sum()), 0)
        wide = np.ones((4, 6, 3), dtype=np.uint8)
        padded_wide = pad_bottom_right(wide, 4)
        self.assertEqual(padded_wide.shape, (4, 8, 3))
        np.testing.assert_array_equal(padded_wide[:, :6], wide)
        self.assertEqual(int(padded_wide[:, 6:].sum()), 0)

    def test_padding_then_cropping_returns_the_original_rectangle(self):
        image = np.random.default_rng(0).integers(0, 255, size=(540, 960, 3)).astype(np.uint8)
        padded = pad_bottom_right(image, 32)
        self.assertEqual(padded.shape[:2], (544, 960))
        cropped = crop_to_original(padded, 960, 540)
        self.assertEqual(cropped.shape[:2], (540, 960))
        np.testing.assert_array_equal(cropped, image)

    def test_crop_to_original_refuses_a_window_that_does_not_fit(self):
        with self.assertRaises(ValueError):
            crop_to_original(np.zeros((10, 10), dtype=np.float32), 20, 10)

    def test_disparity_padding_round_trip_keeps_values_and_nan(self):
        disparity = np.full((540, 960), 42.0, dtype=np.float32)
        disparity[0, 0] = np.nan
        padded = pad_bottom_right(disparity, 32)
        self.assertEqual(padded.shape, (544, 960))
        restored = crop_to_original(padded, 960, 540)
        self.assertTrue(np.isnan(restored[0, 0]))
        self.assertEqual(float(restored[5, 5]), 42.0)


class TemporalWindowTests(unittest.TestCase):
    def test_dynamicstereo_five_frame_window_metadata(self):
        metadata = temporal_window_metadata("dynamicstereo", 5, 0)
        self.assertEqual(metadata["temporal_window_frames"], 5)
        self.assertEqual(metadata["target_position_in_window"], 0)
        self.assertEqual(metadata["future_lookahead_frames"], 4)
        self.assertFalse(metadata["realtime_compatible"])

    def test_igev_single_frame_is_realtime_metadata(self):
        metadata = temporal_window_metadata("igev", 1, 0)
        self.assertEqual(metadata["temporal_window_frames"], 1)
        self.assertEqual(metadata["future_lookahead_frames"], 0)
        self.assertTrue(metadata["realtime_compatible"])

    def test_lookahead_never_becomes_negative(self):
        metadata = temporal_window_metadata("dynamicstereo", 5, 4)
        self.assertEqual(metadata["future_lookahead_frames"], 0)
        self.assertFalse(metadata["realtime_compatible"])

    def test_invalid_windows_are_rejected(self):
        with self.assertRaises(ValueError):
            temporal_window_metadata("igev", 5, 0)
        with self.assertRaises(ValueError):
            temporal_window_metadata("dynamicstereo", 5, 5)
        with self.assertRaises(ValueError):
            temporal_window_metadata("dynamicstereo", 0, 0)

    def test_five_frame_request_carries_the_dynamicstereo_window(self):
        request = learned_stereo_request_from_json(
            learned_stereo_request_to_json(sample_request("dynamicstereo", window=5))
        )
        metadata = temporal_window_metadata(
            request.matcher_name, len(request.left_image_paths), request.target_index
        )
        self.assertEqual(len(request.left_image_paths), 5)
        self.assertEqual(len(request.right_image_paths), 5)
        self.assertEqual(metadata["future_lookahead_frames"], 4)
        self.assertFalse(metadata["realtime_compatible"])


class ReverseFieldTests(unittest.TestCase):
    def test_mirroring_maps_a_leftward_search_onto_the_rightward_field(self):
        # A left-reference disparity of 10 px at column 3 means: the partner of
        # flipped-view pixel 3 is at column -7 -> outside, so use a wider fixture.
        disparity = np.zeros((1, 11), dtype=np.float32)
        disparity[0, 8] = 8.0
        reversed_field = flip_disparity_for_reverse_field(disparity)
        # After un-mirroring, right pixel 2 (= 10 - 8) holds the 8 px offset and it
        # now means "partner 8 px to the right", which is column 10 in the left view.
        self.assertAlmostEqual(float(reversed_field[0, 2]), 8.0, places=6)
        self.assertEqual(int(np.count_nonzero(reversed_field)), 1)

    def test_nan_survives_the_mirror(self):
        disparity = np.full((2, 4), 5.0, dtype=np.float32)
        disparity[1, 0] = np.nan
        mirrored = flip_disparity_for_reverse_field(disparity)
        self.assertTrue(np.isnan(mirrored[1, 3]))
        self.assertEqual(int(np.isnan(mirrored).sum()), 1)

    def test_three_dimensional_input_is_rejected(self):
        with self.assertRaises(ValueError):
            flip_disparity_for_reverse_field(np.zeros((2, 3, 1), dtype=np.float32))


class BenchmarkIsolationTests(unittest.TestCase):
    def test_benchmark_tool_does_not_import_torch(self):
        # Isolation is a property of this tool, independent of pytest order.
        code = (
            f"import sys; sys.path[:0] = {[str(REALTIME_ROOT), str(TOOLS_ROOT)]!r}; "
            "import benchmark_learned_stereo_replacements as benchmark; "
            "assert hasattr(benchmark, 'main'); "
            "assert 'torch' not in sys.modules, 'benchmark imported torch'"
        )
        result = subprocess.run([sys.executable, "-B", "-c", code],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_benchmark_and_worker_sources_avoid_torch_in_the_main_process(self):
        import ast

        from walker_tools.catalog import command_path
        benchmark_source = command_path("benchmark_learned_stereo_replacements.py").read_text(encoding="utf-8")
        self.assertNotIn("import torch", benchmark_source)
        self.assertNotIn("from torch", benchmark_source)
        # The worker is the only place torch may appear, and never at module scope:
        # a module-scope import would break the structured failure path and would
        # make the failure look like a crash instead of a recorded reason.
        worker_source = command_path("run_learned_stereo_worker.py").read_text(encoding="utf-8")
        module_level_imports = [
            line for line in worker_source.splitlines()
            if line.startswith("import torch") or line.startswith("from torch")
        ]
        self.assertEqual(module_level_imports, [])
        tree = ast.parse(worker_source)
        module_scope_torch = [
            node for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and any(
                "torch" in (alias.name if isinstance(node, ast.Import) else (node.module or ""))
                for alias in node.names
            )
        ]
        self.assertEqual(module_scope_torch, [])

    def test_worker_does_not_import_plane_fitting_modules_or_linear_algebra(self):
        import ast

        from walker_tools.catalog import command_path
        worker_source = command_path("run_learned_stereo_worker.py").read_text(encoding="utf-8")
        tree = ast.parse(worker_source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        for forbidden in ("scene_geometry_variants", "benchmark_ground_walker_reconstruction"):
            self.assertFalse(
                any(forbidden in name for name in imported),
                f"the worker must not import {forbidden}; imports were {sorted(imported)}",
            )
        linear_algebra_calls = [
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in ("eigh", "svd", "lstsq", "inv", "solve")
        ]
        self.assertEqual(linear_algebra_calls, [], "the worker must not run NumPy linear algebra")


class SgbmBaselineIntegrityTests(unittest.TestCase):
    def test_reading_the_task01_baseline_leaves_it_untouched(self):
        if not (SGBM_BASELINE / "frame_records.jsonl").exists():
            self.skipTest("archived task-01 benchmark output is not present on this machine")
        import benchmark_learned_stereo_replacements as benchmark

        def snapshot() -> dict[str, tuple[int, int]]:
            return {
                path.name: (path.stat().st_size, path.stat().st_mtime_ns)
                for path in sorted(SGBM_BASELINE.glob("*"))
                if path.is_file()
            }

        before = snapshot()
        summary, records = benchmark.load_sgbm_baseline(SGBM_BASELINE)
        rows = benchmark.sgbm_combination_rows(summary, records)
        after = snapshot()
        self.assertEqual(before, after, "reading the archived SGBM baseline must not modify it")
        self.assertEqual(len(records), 12)
        self.assertEqual(len(rows), 12 * 2 * 3)
        self.assertEqual({row["matcher"] for row in rows}, {"sgbm"})
        self.assertEqual({row["semantic_source"] for row in rows}, {"manual_floor", "mask2former_floor"})
        self.assertEqual({row["plane_method"] for row in rows}, set(benchmark.MAIN_METHODS))

    def test_sgbm_rows_keep_none_for_learned_only_timing_fields(self):
        if not (SGBM_BASELINE / "summary.json").exists():
            self.skipTest("archived task-01 benchmark output is not present on this machine")
        import benchmark_learned_stereo_replacements as benchmark

        summary, records = benchmark.load_sgbm_baseline(SGBM_BASELINE)
        rows = benchmark.sgbm_combination_rows(summary, records)
        for row in rows:
            self.assertIsNone(row["timing"]["stereo_gpu_forward_ms"])
            self.assertIsNone(row["timing"]["stereo_cpu_worker_wall_ms"])
            self.assertIsNotNone(row["timing"]["estimated_direct_pipeline_ms"])
            self.assertEqual(row["confidence_boundary"], benchmark.CONFIDENCE_BOUNDARY)


if __name__ == "__main__":
    unittest.main()
