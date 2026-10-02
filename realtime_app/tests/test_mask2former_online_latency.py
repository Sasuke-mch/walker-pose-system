"""Unit tests for the Mask2Former online latency and cached-mask parity tool.

Every test here is synthetic or purely arithmetic: no camera image, no model, no
calibration, no GPU and no upstream experiment output is read or written.  The
one integration-style test replaces the model forward with a fake stage runner so
that the reported number can only come from the tool's own stage arithmetic and
mask comparison.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import uuid
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for _candidate in (str(ROOT), str(ROOT / "tools")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import benchmark_mask2former_online_latency as benchmark  # noqa: E402


HEIGHT, WIDTH = 8, 6
PAIR_IDS = (0, 40)
# The forbidden overclaiming phrases are written in this file only so the guard
# test can prove the tool never emits them.
FORBIDDEN_PHRASES = ("真实地面精度提高", "真实三维精度提高", "接触识别成功", "步态识别成功")


@contextlib.contextmanager
def ascii_tempdir():
    """A temporary scratch directory with an ASCII path under the repository root.

    OpenCV's image codecs use the narrow-character Windows API, so a path with
    non-ASCII characters can break imread/imwrite.  ``tempfile.mkdtemp`` cannot
    be used here because on this host a directory created by it cannot receive
    new children, so the directory is created with a plain unique name instead.
    """
    directory = ROOT.parent / f"tmp_m2f_online_test_{uuid.uuid4().hex[:10]}"
    directory.mkdir()
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cv2.imencode(".png", array)[1].tobytes())


def synthetic_image(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(40, 200, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:4, :] = 30
    return image


def synthetic_mask(kind: str = "lower_half") -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    if kind == "lower_half":
        mask[HEIGHT // 2 :, :] = 255
    elif kind == "one_pixel_different":
        mask[HEIGHT // 2 :, :] = 255
        mask[0, 0] = 255
    elif kind == "empty":
        pass
    else:  # pragma: no cover - guarded by the callers
        raise AssertionError(f"unknown synthetic mask kind {kind}")
    return mask


def labelme_payload(width: int = WIDTH, height: int = HEIGHT) -> dict:
    """LabelMe payload with all five labels, including an overlapping ignore area."""
    return {
        "version": "5.4.1",
        "imageWidth": width,
        "imageHeight": height,
        "shapes": [
            {"label": "floor_eligible", "shape_type": "polygon", "points": [[0, height // 2], [width, height // 2], [width, height], [0, height]]},
            {"label": "walker", "shape_type": "polygon", "points": [[0, height - 2], [2, height - 2], [2, height], [0, height]]},
            {"label": "person", "shape_type": "polygon", "points": [[width - 2, height - 2], [width, height - 2], [width, height], [width - 2, height]]},
            {"label": "static_other", "shape_type": "polygon", "points": [[0, 0], [1, 0], [1, 1], [0, 1]]},
            {"label": "ignore_uncertain", "shape_type": "polygon", "points": [[3, 3], [4, 3], [4, 4], [3, 4]]},
        ],
    }


def build_inputs(
    root: Path,
    *,
    with_right_cache: bool = True,
    cache_shape: tuple[int, int] | None = None,
    pair_ids=PAIR_IDS,
) -> dict[str, Path]:
    """Create one upright view pair tree with images, cached masks and manual labels."""
    left_images = root / "left_images"
    right_images = root / "right_images"
    left_cache = root / "left_cache"
    right_cache = root / "right_cache"
    left_labels = root / "left_labels"
    right_labels = root / "right_labels"
    for directory in (left_images, right_images, left_cache, right_cache, left_labels, right_labels):
        directory.mkdir(parents=True, exist_ok=True)
    cache_height, cache_width = cache_shape or (HEIGHT, WIDTH)
    for pair_id in pair_ids:
        name = benchmark.pair_file_name(pair_id)
        write_png(left_images / name, synthetic_image(seed=10 + pair_id))
        write_png(right_images / name, synthetic_image(seed=20 + pair_id))
        cached = np.zeros((cache_height, cache_width), dtype=np.uint8)
        cached[cache_height // 2 :, :] = 255
        write_png(left_cache / name, cached)
        if with_right_cache:
            write_png(right_cache / name, cached)
        payload = labelme_payload()
        (left_labels / benchmark.label_file_name(pair_id)).write_text(json.dumps(payload), encoding="utf-8")
        (right_labels / benchmark.label_file_name(pair_id)).write_text(json.dumps(payload), encoding="utf-8")
    return {
        "left_images": left_images,
        "right_images": right_images,
        "left_cache": left_cache,
        "right_cache": right_cache,
        "left_labels": left_labels,
        "right_labels": right_labels,
    }


def resolve(dirs: dict[str, Path], pair_ids=PAIR_IDS) -> list[dict]:
    return benchmark.resolve_pair_inputs(
        pair_ids,
        dirs["left_images"],
        dirs["right_images"],
        dirs["left_cache"],
        dirs["right_cache"],
        dirs["left_labels"],
        dirs["right_labels"],
    )


def fake_stage_runner(recorded: list[dict], base_gpu_ms: float = 90.0):
    """Stand-in for the real forward that keeps every stage field non-trivial."""

    def runner(image_path, cached_mask_path, runtime, torch, device, measure_cache_read=True, measure_mask_write=False):
        cached = cv2.imread(str(cached_mask_path), cv2.IMREAD_GRAYSCALE)
        if cached is None:
            raise benchmark.OnlineLatencyError(f"cannot read cached candidate mask: {cached_mask_path}")
        candidate = synthetic_mask("lower_half")
        stage = {
            "image_decode_ms": 11.0,
            "bgr_to_rgb_ms": 1.0,
            "processor_cpu_ms": 22.0,
            "host_to_device_ms": 2.0,
            "gpu_forward_ms": base_gpu_ms,
            "postprocess_ms": 13.0,
            "floor_mask_extract_ms": 4.0,
        }
        record = dict(stage)
        record[benchmark.FULL_ONLINE_FIELD] = benchmark.full_online_semantic_ms(stage)
        record["cache_mask_read_ms"] = 3.0 if measure_cache_read else None
        record["candidate_png_write_ms"] = None
        record["candidate_mask"] = candidate
        record["cached_mask_bool"] = cached > 0
        record["image_shape_hw"] = [HEIGHT, WIDTH]
        recorded.append({"image": str(image_path), "stage": stage})
        return record

    return runner


def run_main(monkeypatch_target: str, fake, argv: list[str]) -> int:
    # These tests exercise file/timing contracts, never real model availability.
    runtime = {"floor_class_id": 3, "model_load_ms": 1.0,
               "processor_size": {"height": HEIGHT, "width": WIDTH}, "num_labels": 150}
    fake_torch = SimpleNamespace(
        __version__="test", version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(cudnn=SimpleNamespace(is_available=lambda: False)),
    )
    with patch.object(benchmark, monkeypatch_target, fake), \
         patch.object(benchmark, "load_model", return_value=runtime), \
         patch.dict(sys.modules, {"torch": fake_torch,
                                  "transformers": SimpleNamespace(__version__="test")}):
        return benchmark.main(argv)


def full_argv(dirs: dict[str, Path], output_dir: Path, pair_ids=PAIR_IDS, repeats: int = 1, warmup: int = 1) -> list[str]:
    return [
        "--left-image-dir", str(dirs["left_images"]),
        "--right-image-dir", str(dirs["right_images"]),
        "--left-cache-dir", str(dirs["left_cache"]),
        "--right-cache-dir", str(dirs["right_cache"]),
        "--left-label-dir", str(dirs["left_labels"]),
        "--right-label-dir", str(dirs["right_labels"]),
        "--output-dir", str(output_dir),
        "--pair-ids", *[str(value) for value in pair_ids],
        "--device", "cpu",
        "--warmup-count", str(warmup),
        "--measured-repeats", str(repeats),
        "--batch-size", "1",
        "--local-files-only",
    ]


def synthetic_matrix_row(combination_id: str, p50, p95) -> dict:
    return {
        "combination_id": combination_id,
        "matcher": "sgbm",
        "plane_method": "dense_ransac",
        "temporal_mode": "direct_current_frame",
        "realtime_compatible": False,
        "estimated_end_to_end_latency_ms": p50,
        "estimated_end_to_end_latency_p95_ms": p95,
        "measured_end_to_end_latency_ms": None,
        "semantic_model": "mask2former_swin_small",
    }


def synthetic_matrix(rows: list[dict]) -> dict:
    return {
        "schema_version": "temporal_ground_composition_v1",
        "row_count": len(rows),
        "semantic_model": "mask2former_swin_small",
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# 1. batch-size contract
# --------------------------------------------------------------------------- #
class BatchSizeContractTests(unittest.TestCase):
    def test_batch_size_one_is_accepted(self) -> None:
        self.assertEqual(benchmark.validate_batch_size(1), 1)

    def test_other_batch_sizes_are_rejected(self) -> None:
        for value in (0, 2, 4, -1):
            with self.subTest(batch_size=value):
                with self.assertRaises(ValueError):
                    benchmark.validate_batch_size(value)

    def test_argparse_contract_defaults(self) -> None:
        args = benchmark.parse_args(
            [
                "--left-image-dir", "l", "--right-image-dir", "r",
                "--left-cache-dir", "lc", "--right-cache-dir", "rc",
                "--left-label-dir", "ll", "--right-label-dir", "rl",
                "--output-dir", "o", "--pair-ids", "0",
            ]
        )
        self.assertEqual(args.batch_size, 1)
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.warmup_count, 3)
        self.assertEqual(args.measured_repeats, 5)
        self.assertTrue(args.local_files_only)

    def test_run_parameter_validation_rejects_other_batch_size(self) -> None:
        args = benchmark.parse_args(
            [
                "--left-image-dir", "l", "--right-image-dir", "r",
                "--left-cache-dir", "lc", "--right-cache-dir", "rc",
                "--left-label-dir", "ll", "--right-label-dir", "rl",
                "--output-dir", "o", "--pair-ids", "0", "--batch-size", "2",
            ]
        )
        with self.assertRaises(ValueError):
            benchmark.validate_run_parameters(args)

    def test_no_argument_enables_a_download(self) -> None:
        parser = benchmark.build_parser()
        option_strings = {option for action in parser._actions for option in action.option_strings}
        self.assertNotIn("--allow-download", option_strings)
        self.assertIn("--local-files-only", option_strings)


# --------------------------------------------------------------------------- #
# 2. output directory contract
# --------------------------------------------------------------------------- #
class OutputDirectoryContractTests(unittest.TestCase):
    def test_existing_output_directory_is_rejected(self) -> None:
        with ascii_tempdir() as temporary:
            existing = temporary / "already_there"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                benchmark.ensure_output_dir_absent(existing)

    def test_main_refuses_to_overwrite_an_existing_output_directory(self) -> None:
        with ascii_tempdir() as temporary:
            root = temporary
            dirs = build_inputs(root)
            output_dir = root / "existing_output"
            output_dir.mkdir()
            (output_dir / "keep_me.txt").write_text("untouched", encoding="utf-8")
            with patch.object(benchmark, "load_model", side_effect=AssertionError("model must not load")) as loader:
                with self.assertRaises(FileExistsError):
                    benchmark.main(full_argv(dirs, output_dir))
                loader.assert_not_called()
            self.assertEqual((output_dir / "keep_me.txt").read_text(encoding="utf-8"), "untouched")
            self.assertEqual(sorted(item.name for item in output_dir.iterdir()), ["keep_me.txt"])

    def test_absent_output_directory_is_accepted(self) -> None:
        with ascii_tempdir() as temporary:
            benchmark.ensure_output_dir_absent(temporary / "fresh")


# --------------------------------------------------------------------------- #
# 3. input contract
# --------------------------------------------------------------------------- #
class InputContractTests(unittest.TestCase):
    def test_missing_image_is_rejected(self) -> None:
        with ascii_tempdir() as temporary:
            dirs = build_inputs(temporary)
            (dirs["right_images"] / benchmark.pair_file_name(PAIR_IDS[1])).unlink()
            with self.assertRaises(FileNotFoundError) as context:
                resolve(dirs)
            self.assertIn("right_image", str(context.exception))

    def test_missing_cached_mask_is_rejected(self) -> None:
        with ascii_tempdir() as temporary:
            dirs = build_inputs(temporary, with_right_cache=False)
            with self.assertRaises(FileNotFoundError) as context:
                resolve(dirs)
            self.assertIn("right_cache", str(context.exception))

    def test_missing_manual_label_is_rejected(self) -> None:
        with ascii_tempdir() as temporary:
            dirs = build_inputs(temporary)
            (dirs["left_labels"] / benchmark.label_file_name(PAIR_IDS[0])).unlink()
            with self.assertRaises(FileNotFoundError) as context:
                resolve(dirs)
            self.assertIn("manual label", str(context.exception))

    def test_cached_mask_size_mismatch_is_rejected(self) -> None:
        with ascii_tempdir() as temporary:
            dirs = build_inputs(temporary, cache_shape=(HEIGHT + 3, WIDTH + 3))
            with self.assertRaises(benchmark.OnlineLatencyError) as context:
                resolve(dirs)
            self.assertIn("does not match input image size", str(context.exception))

    def test_valid_inputs_resolve_all_four_required_paths(self) -> None:
        with ascii_tempdir() as temporary:
            dirs = build_inputs(temporary)
            entries = resolve(dirs)
            self.assertEqual([entry["pair_id"] for entry in entries], list(PAIR_IDS))
            for entry in entries:
                for key in ("left_image", "right_image", "left_cache", "right_cache"):
                    self.assertTrue(entry[key].is_file(), key)

    def test_cuda_is_refused_when_unavailable(self) -> None:
        class FakeTorch:
            class cuda:  # noqa: N801 - mirrors the torch namespace for the check
                @staticmethod
                def is_available() -> bool:
                    return False

        with self.assertRaises(benchmark.OnlineLatencyError):
            benchmark.resolve_device(FakeTorch, "cuda")
        self.assertEqual(benchmark.resolve_device(FakeTorch, "cpu"), "cpu")


# --------------------------------------------------------------------------- #
# 4. stage arithmetic
# --------------------------------------------------------------------------- #
class StageArithmeticTests(unittest.TestCase):
    def test_full_online_semantic_ms_is_the_stage_sum(self) -> None:
        stage = {
            "image_decode_ms": 19.4553,
            "bgr_to_rgb_ms": 12.4152,
            "processor_cpu_ms": 21.3643,
            "host_to_device_ms": 1.6323,
            "gpu_forward_ms": 1461.3216,
            "postprocess_ms": 54.9997,
            "floor_mask_extract_ms": 6.5327,
        }
        expected = sum(stage.values())
        self.assertAlmostEqual(benchmark.full_online_semantic_ms(stage), expected, places=9)

    def test_io_diagnostics_are_not_part_of_the_module_time(self) -> None:
        self.assertNotIn("cache_mask_read_ms", benchmark.STAGE_FIELDS)
        self.assertNotIn("candidate_png_write_ms", benchmark.STAGE_FIELDS)
        stage = {field: 1.0 for field in benchmark.STAGE_FIELDS}
        self.assertAlmostEqual(benchmark.full_online_semantic_ms(stage), float(len(benchmark.STAGE_FIELDS)))
        self.assertEqual(set(benchmark.IO_DIAGNOSTIC_FIELDS), {"cache_mask_read_ms", "candidate_png_write_ms"})

    def test_stage_fields_are_exactly_the_declared_seven(self) -> None:
        self.assertEqual(
            benchmark.STAGE_FIELDS,
            (
                "image_decode_ms",
                "bgr_to_rgb_ms",
                "processor_cpu_ms",
                "host_to_device_ms",
                "gpu_forward_ms",
                "postprocess_ms",
                "floor_mask_extract_ms",
            ),
        )

    def test_timing_summary_reports_stage_percentiles_per_view(self) -> None:
        records = []
        for view, offset in (("left", 0.0), ("right", 5.0)):
            for index in range(6):
                stage = {field: 1.0 + offset + index for field in benchmark.STAGE_FIELDS}
                record = {"view": view, **stage, benchmark.FULL_ONLINE_FIELD: benchmark.full_online_semantic_ms(stage)}
                records.append(record)
        timing = benchmark.timing_summary_from_records(records)
        self.assertEqual(timing["main_measurement_record_count"], 12)
        self.assertEqual(timing["all_views"][benchmark.FULL_ONLINE_FIELD]["count"], 12)
        self.assertEqual(timing["left"][benchmark.FULL_ONLINE_FIELD]["count"], 6)
        self.assertEqual(timing["right"][benchmark.FULL_ONLINE_FIELD]["count"], 6)
        self.assertGreater(
            timing["right"][benchmark.FULL_ONLINE_FIELD]["median_ms"],
            timing["left"][benchmark.FULL_ONLINE_FIELD]["median_ms"],
        )
        for field in benchmark.STAGE_FIELDS:
            block = timing["all_views"][field]
            for key in ("count", "mean_ms", "median_ms", "p90_ms", "p95_ms", "min_ms", "max_ms"):
                self.assertIn(key, block)


# --------------------------------------------------------------------------- #
# 5 / 6. pixel parity
# --------------------------------------------------------------------------- #
class ParityTests(unittest.TestCase):
    def test_identical_masks_are_an_exact_match(self) -> None:
        mask = synthetic_mask("lower_half").astype(bool)
        parity = benchmark.mask_comparison(mask, mask.copy())
        self.assertEqual(parity["different_pixel_count"], 0)
        self.assertEqual(parity["different_pixel_fraction"], 0.0)
        self.assertTrue(parity["exact_match"])
        self.assertEqual(parity["equal_pixel_count"], mask.size)
        self.assertEqual(parity["iou_vs_cached"], 1.0)
        self.assertEqual(parity["parity_status"], benchmark.PARITY_EXACT)

    def test_one_different_pixel_is_a_mismatch_and_is_not_silently_dropped(self) -> None:
        candidate = synthetic_mask("lower_half").astype(bool)
        cached = candidate.copy()
        cached[0, 0] = True
        parity = benchmark.mask_comparison(candidate, cached)
        self.assertEqual(parity["different_pixel_count"], 1)
        self.assertFalse(parity["exact_match"])
        self.assertAlmostEqual(parity["different_pixel_fraction"], 1.0 / float(candidate.size))
        self.assertEqual(parity["parity_status"], benchmark.PARITY_MISMATCH)
        self.assertLess(parity["iou_vs_cached"], 1.0)

    def test_two_empty_masks_are_an_exact_match_with_iou_one(self) -> None:
        empty = synthetic_mask("empty").astype(bool)
        parity = benchmark.mask_comparison(empty, empty.copy())
        self.assertTrue(parity["exact_match"])
        self.assertEqual(parity["iou_vs_cached"], 1.0)
        self.assertEqual(parity["union_pixel_count"], 0)

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaises(benchmark.OnlineLatencyError):
            benchmark.mask_comparison(np.zeros((4, 4), dtype=bool), np.zeros((5, 5), dtype=bool))

    def test_overlay_is_renderable_even_when_the_masks_agree(self) -> None:
        image = synthetic_image()
        mask = synthetic_mask("lower_half")
        parity = benchmark.mask_comparison(mask.astype(bool), mask.astype(bool))
        overlay = benchmark.render_parity_overlay(image, mask.astype(bool), mask.astype(bool), parity, "unit test")
        self.assertEqual(overlay.ndim, 3)
        self.assertGreater(overlay.shape[0], HEIGHT)
        self.assertGreater(overlay.shape[1], WIDTH)


# --------------------------------------------------------------------------- #
# 7 / 8. latency augmentation arithmetic
# --------------------------------------------------------------------------- #
class LatencyAugmentationTests(unittest.TestCase):
    def test_null_old_latency_stays_null(self) -> None:
        matrix = synthetic_matrix([synthetic_matrix_row("a__vo_propagated_plane", None, None)])
        augmented = benchmark.augment_matrix_with_semantic_latency(matrix, 191.5, 260.25, benchmark.PARITY_ALL_EXACT)
        row = augmented["rows"][0]
        self.assertIsNone(row["estimated_end_to_end_including_semantic_p50_ms"])
        self.assertIsNone(row["estimated_end_to_end_including_semantic_p95_ms"])
        self.assertEqual(row["semantic_online_p50_ms"], 191.5)
        self.assertEqual(row["semantic_online_p95_ms"], 260.25)
        self.assertEqual(row["latency_kind"], "estimated_module_sum")

    def test_numeric_old_latency_is_summed_exactly(self) -> None:
        matrix = synthetic_matrix([synthetic_matrix_row("b__direct_current_frame", 745.0684000014007, 785.7495500053119)])
        augmented = benchmark.augment_matrix_with_semantic_latency(matrix, 191.5, 260.25, benchmark.PARITY_ALL_EXACT)
        row = augmented["rows"][0]
        self.assertEqual(row["estimated_end_to_end_including_semantic_p50_ms"], 745.0684000014007 + 191.5)
        self.assertEqual(row["estimated_end_to_end_including_semantic_p95_ms"], 785.7495500053119 + 260.25)
        self.assertEqual(row["estimated_end_to_end_latency_ms"], 745.0684000014007)

    def test_one_null_and_one_numeric_in_the_same_matrix(self) -> None:
        matrix = synthetic_matrix(
            [
                synthetic_matrix_row("null_row", None, None),
                synthetic_matrix_row("numeric_row", 100.0, 200.0),
            ]
        )
        augmented = benchmark.augment_matrix_with_semantic_latency(matrix, 10.0, 20.0, benchmark.PARITY_MISMATCH_PRESENT)
        self.assertIsNone(augmented["rows"][0]["estimated_end_to_end_including_semantic_p50_ms"])
        self.assertEqual(augmented["rows"][1]["estimated_end_to_end_including_semantic_p50_ms"], 110.0)
        self.assertEqual(augmented["rows"][1]["estimated_end_to_end_including_semantic_p95_ms"], 220.0)

    def test_a_null_semantic_value_produces_null_totals(self) -> None:
        matrix = synthetic_matrix([synthetic_matrix_row("numeric_row", 100.0, 200.0)])
        augmented = benchmark.augment_matrix_with_semantic_latency(matrix, None, None, benchmark.PARITY_MISMATCH_PRESENT)
        self.assertIsNone(augmented["rows"][0]["estimated_end_to_end_including_semantic_p50_ms"])

    def test_boundary_text_follows_the_parity_status(self) -> None:
        matrix = synthetic_matrix([synthetic_matrix_row("row", 1.0, 2.0)])
        exact = benchmark.augment_matrix_with_semantic_latency(matrix, 1.0, 2.0, benchmark.PARITY_ALL_EXACT)
        mismatch = benchmark.augment_matrix_with_semantic_latency(matrix, 1.0, 2.0, benchmark.PARITY_MISMATCH_PRESENT)
        self.assertEqual(exact["rows"][0]["semantic_latency_boundary"], benchmark.BOUNDARY_EXACT)
        self.assertEqual(mismatch["rows"][0]["semantic_latency_boundary"], benchmark.BOUNDARY_MISMATCH)
        self.assertIn("does not exactly reproduce", mismatch["rows"][0]["semantic_latency_boundary"])
        self.assertIn("not a measured end-to-end latency", exact["rows"][0]["semantic_latency_boundary"])


# --------------------------------------------------------------------------- #
# 9 / 10 / 11. the real 27-row matrix
# --------------------------------------------------------------------------- #
class RealMatrixAugmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matrix_path = Path(benchmark.summary_source_matrix_path())
        if not cls.matrix_path.is_file():
            raise unittest.SkipTest(f"source matrix is unavailable: {cls.matrix_path}")
        cls.source = json.loads(cls.matrix_path.read_text(encoding="utf-8"))
        cls.augmented = benchmark.augment_matrix_with_semantic_latency(
            cls.source, 191.5, 260.25, benchmark.PARITY_ALL_EXACT
        )

    def test_source_matrix_is_the_frozen_27_row_matrix(self) -> None:
        self.assertEqual(len(self.source["rows"]), 27)
        self.assertEqual(len({row["combination_id"] for row in self.source["rows"]}), 27)

    def test_augmented_matrix_keeps_27_rows_and_unique_combination_ids(self) -> None:
        self.assertEqual(len(self.augmented["rows"]), 27)
        ids = [row["combination_id"] for row in self.augmented["rows"]]
        self.assertEqual(len(set(ids)), 27)
        self.assertEqual([row["combination_id"] for row in self.source["rows"]], ids)

    def test_every_original_field_is_preserved(self) -> None:
        source_by_id = {row["combination_id"]: row for row in self.source["rows"]}
        for row in self.augmented["rows"]:
            original = source_by_id[row["combination_id"]]
            for key, value in original.items():
                self.assertEqual(row[key], value, f"{row['combination_id']}:{key}")

    def test_no_row_gains_a_measured_end_to_end_latency(self) -> None:
        for row in self.augmented["rows"]:
            self.assertIsNone(row["measured_end_to_end_latency_ms"], row["combination_id"])

    def test_every_row_stays_realtime_incompatible(self) -> None:
        for row in self.augmented["rows"]:
            self.assertIs(row["realtime_compatible"], False, row["combination_id"])
        self.assertEqual(sum(1 for row in self.augmented["rows"] if row["realtime_compatible"]), 0)

    def test_every_row_uses_the_estimated_module_sum_kind(self) -> None:
        for row in self.augmented["rows"]:
            self.assertEqual(row["latency_kind"], "estimated_module_sum", row["combination_id"])
            self.assertEqual(row["semantic_latency_source"], benchmark.EXPERIMENT_ID)

    def test_numeric_rows_add_and_null_rows_stay_null(self) -> None:
        numeric = 0
        nulls = 0
        for row in self.augmented["rows"]:
            base = row["estimated_end_to_end_latency_ms"]
            if base is None:
                nulls += 1
                self.assertIsNone(row["estimated_end_to_end_including_semantic_p50_ms"], row["combination_id"])
            else:
                numeric += 1
                self.assertEqual(row["estimated_end_to_end_including_semantic_p50_ms"], base + 191.5, row["combination_id"])
        self.assertEqual(numeric + nulls, 27)
        self.assertGreater(numeric, 0)
        self.assertGreater(nulls, 0)

    def test_csv_projection_keeps_the_source_columns(self) -> None:
        rows = benchmark.matrix_rows_to_csv_rows(self.augmented)
        self.assertEqual(len(rows), 27)
        for column in (
            "combination_id",
            "matcher",
            "plane_method",
            "temporal_mode",
            "full_chain_measured",
            "measurement_scope",
            "estimated_end_to_end_latency_ms",
            "measured_end_to_end_latency_ms",
            "realtime_compatible",
            "cross_region_pass_frames",
            "median_inlier_fraction",
            "median_residual_mm",
            "flow_module_status",
            "vo_module_status",
        ):
            self.assertIn(column, benchmark.MATRIX_CSV_HEADER)
        for row in rows:
            for column in benchmark.MATRIX_CSV_HEADER:
                self.assertIn(column, row, column)
            # The source CSV's measured column stays empty and the new column is
            # the only place the "measured end-to-end" null is mirrored.
            self.assertIsNone(row["measured_end_to_end_latency_ms"])
            self.assertIsNone(row["measured_end_to_end_latency_ms_including_semantic"])
            self.assertEqual(row["latency_kind"], "estimated_module_sum")

    def test_csv_text_is_parseable_and_has_one_header(self) -> None:
        text = benchmark.matrix_csv_text(self.augmented)
        lines = text.strip().splitlines()
        self.assertEqual(lines[0], ",".join(benchmark.MATRIX_CSV_HEADER))
        self.assertEqual(len(lines), 28)
        for line in lines[1:]:
            self.assertEqual(len(line.split(",")), len(benchmark.MATRIX_CSV_HEADER))


# --------------------------------------------------------------------------- #
# 12. text guard
# --------------------------------------------------------------------------- #
class TextGuardTests(unittest.TestCase):
    def test_forbidden_overclaiming_phrases_are_absent_from_every_emitted_string(self) -> None:
        emitted = [
            benchmark.__doc__ or "",
            benchmark.INTERPRETATION_BOUNDARY,
            benchmark.BOUNDARY_EXACT,
            benchmark.BOUNDARY_MISMATCH,
        ]
        for text in emitted:
            for phrase in FORBIDDEN_PHRASES:
                self.assertNotIn(phrase, text, f"unexpected overclaiming phrase {phrase}")
        # The guard list itself is the only place those phrases may appear.
        self.assertEqual(set(FORBIDDEN_PHRASES), set(benchmark.TEXT_FORBIDDEN_PHRASES))

    def test_documented_boundaries_are_explicit(self) -> None:
        self.assertIn("not a measured end-to-end latency or physical", benchmark.BOUNDARY_EXACT)
        self.assertIn("automatic 2-D image-space floor candidate", benchmark.INTERPRETATION_BOUNDARY)
        self.assertIn("not a measured end-to-end latency", benchmark.INTERPRETATION_BOUNDARY)
        self.assertIn("Estimated module sum only", benchmark.BOUNDARY_EXACT)
        self.assertIn("Timing was measured", benchmark.BOUNDARY_MISMATCH)

    def test_assert_no_forbidden_phrases_rejects_an_overclaim(self) -> None:
        benchmark.assert_no_forbidden_phrases("This is a module time only.")
        for phrase in FORBIDDEN_PHRASES:
            with self.subTest(phrase=phrase):
                with self.assertRaises(benchmark.OnlineLatencyError):
                    benchmark.assert_no_forbidden_phrases(f"结论：{phrase}。")

    def test_manual_2d_audit_wording_is_used_and_not_accuracy_wording(self) -> None:
        with ascii_tempdir() as temporary:
            dirs = build_inputs(temporary)
            entries = resolve(dirs)
            for entry in entries:
                entry["candidate"] = {
                    "left": synthetic_mask("lower_half").astype(bool),
                    "right": synthetic_mask("lower_half").astype(bool),
                }
            audit = benchmark.semantic_audit("left", entries, lambda entry, view: entry["candidate"][view])
            self.assertEqual(audit["metric_name"], "manual 2-D mask consistency audit")
            self.assertIn("not semantic accuracy", audit["interpretation_boundary"])
            self.assertEqual(audit["labelled_image_count"], len(PAIR_IDS))
            self.assertEqual(audit["pair_ids"], list(PAIR_IDS))
            for key in ("precision", "recall", "iou", "walker_to_floor_fraction"):
                self.assertIn(key, audit)
                self.assertIsNotNone(audit[key], key)
            # The manual walker polygon overlaps the candidate's floor area, so the
            # leak fraction is computable and bounded on this synthetic input.
            self.assertGreaterEqual(audit["walker_to_floor_fraction"], 0.0)
            self.assertLessEqual(audit["walker_to_floor_fraction"], 1.0)

    def test_priority_order_matches_the_existing_manual_audit(self) -> None:
        self.assertEqual(
            benchmark.AUDIT_PRIORITY,
            ("ignore_uncertain", "person", "walker", "static_other", "floor_eligible"),
        )
        masks = benchmark.rasterize_labelme(labelme_payload())
        self.assertFalse(np.logical_and(masks["ignore_uncertain"], masks["floor_eligible"]).any())
        self.assertTrue(masks["valid_evaluation"][0, 0])
        self.assertFalse(masks["valid_evaluation"][3, 3])


# --------------------------------------------------------------------------- #
# integration: the written artifacts
# --------------------------------------------------------------------------- #
class ArtifactIntegrationTests(unittest.TestCase):
    def test_full_run_writes_every_required_artifact(self) -> None:
        with ascii_tempdir() as temporary:
            root = temporary
            dirs = build_inputs(root, pair_ids=(0, 160))
            output_dir = root / "run_output"
            recorded: list[dict] = []
            exit_code = run_main(
                "run_single_image_online",
                fake_stage_runner(recorded),
                full_argv(dirs, output_dir, pair_ids=(0, 160), repeats=2, warmup=1),
            )
            self.assertEqual(exit_code, 0)

            for name in (
                "EXPERIMENT.md",
                "run_metadata.json",
                "command.txt",
                "summary.json",
                "per_inference_records.jsonl",
                "parity_records.jsonl",
                "semantic_audit_left.json",
                "semantic_audit_right.json",
                "timing_summary.json",
                "combination_matrix_27_with_semantic_latency.json",
                "combination_matrix_27_with_semantic_latency.csv",
            ):
                self.assertTrue((output_dir / name).is_file(), name)

            for view in ("left", "right"):
                for pair_id in (0, 160):
                    self.assertTrue(
                        (output_dir / "visualizations" / f"{view}_pair_{pair_id:04d}_overlay.png").is_file(),
                        f"{view} pair_{pair_id:04d}",
                    )
                    self.assertTrue(
                        (output_dir / "visualizations" / f"{view}_pair_{pair_id:04d}_parity.png").is_file(),
                        f"{view} pair_{pair_id:04d} parity",
                    )

            inference_records = [
                json.loads(line)
                for line in (output_dir / "per_inference_records.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(inference_records), 2 * 2 * 2)
            self.assertEqual(len(recorded), len(inference_records) + 1)  # plus one warm-up
            for record in inference_records:
                stage = {field: record[field] for field in benchmark.STAGE_FIELDS}
                self.assertAlmostEqual(record[benchmark.FULL_ONLINE_FIELD], sum(stage.values()), places=6)
                self.assertNotIn("cache_mask_read_ms", benchmark.STAGE_FIELDS)
            self.assertEqual(sum(1 for record in inference_records if record["warmup"]), 0)

            parity_records = [
                json.loads(line)
                for line in (output_dir / "parity_records.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(parity_records), 2 * 2)
            for record in parity_records:
                self.assertEqual(record["parity_status"], benchmark.PARITY_EXACT)
                self.assertEqual(record["different_pixel_count"], 0)
                self.assertTrue(record["exact_match"])
                self.assertTrue(record["archived_cache_untouched"])

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["parity"]["exact_match_count"], len(parity_records))
            self.assertEqual(summary["parity"]["semantic_parity_status"], benchmark.PARITY_ALL_EXACT)
            self.assertEqual(summary["matrix"]["row_count"], 27)
            self.assertEqual(summary["matrix"]["realtime_false_rows"], 27)
            self.assertEqual(summary["matrix"]["measured_null_rows"], 27)
            self.assertEqual(summary["manual_2d_audit"]["left"]["labelled_image_count"], 2)
            self.assertEqual(summary["manual_2d_audit"]["right"]["labelled_image_count"], 2)

            timing = json.loads((output_dir / "timing_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(timing["main_measurement_record_count"], len(inference_records))
            self.assertEqual(timing["all_views"][benchmark.FULL_ONLINE_FIELD]["count"], len(inference_records))
            self.assertEqual(timing["left"][benchmark.FULL_ONLINE_FIELD]["count"], 2 * 2)
            self.assertIn("environment", timing)
            self.assertIn("gpu_name", timing["environment"])
            self.assertTrue(timing["stage_definition"][benchmark.FULL_ONLINE_FIELD])

            matrix = json.loads(
                (output_dir / "combination_matrix_27_with_semantic_latency.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(matrix["rows"]), 27)
            self.assertEqual(len({row["combination_id"] for row in matrix["rows"]}), 27)
            self.assertEqual(matrix["semantic_parity_status"], benchmark.PARITY_ALL_EXACT)
            for row in matrix["rows"]:
                self.assertIsNone(row["measured_end_to_end_latency_ms"])
                self.assertIs(row["realtime_compatible"], False)
                self.assertEqual(row["latency_kind"], "estimated_module_sum")

            command_text = (output_dir / "command.txt").read_text(encoding="utf-8")
            self.assertIn("--local-files-only", command_text)
            self.assertNotIn("--allow-download", command_text)

            document = (output_dir / "EXPERIMENT.md").read_text(encoding="utf-8")
            for phrase in FORBIDDEN_PHRASES:
                self.assertNotIn(phrase, document)
            self.assertIn("estimated_module_sum", document)
            self.assertIn("不是全链实测", document)

    def test_mismatch_run_marks_every_row_and_keeps_the_difference(self) -> None:
        with ascii_tempdir() as temporary:
            root = temporary
            dirs = build_inputs(root)
            output_dir = root / "mismatch_output"

            def mismatching_runner(image_path, cached_mask_path, runtime, torch, device, measure_cache_read=True, measure_mask_write=False):
                cached = cv2.imread(str(cached_mask_path), cv2.IMREAD_GRAYSCALE)
                candidate = synthetic_mask("one_pixel_different")
                stage = {field: 1.0 for field in benchmark.STAGE_FIELDS}
                record = dict(stage)
                record[benchmark.FULL_ONLINE_FIELD] = benchmark.full_online_semantic_ms(stage)
                record["cache_mask_read_ms"] = 1.0
                record["candidate_png_write_ms"] = None
                record["candidate_mask"] = candidate
                record["cached_mask_bool"] = cached > 0
                record["image_shape_hw"] = [HEIGHT, WIDTH]
                return record

            exit_code = run_main(
                "run_single_image_online",
                mismatching_runner,
                full_argv(dirs, output_dir, repeats=1, warmup=0),
            )
            self.assertEqual(exit_code, 0)

            parity_records = [
                json.loads(line)
                for line in (output_dir / "parity_records.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(parity_records), len(PAIR_IDS) * 2)
            for record in parity_records:
                self.assertEqual(record["parity_status"], benchmark.PARITY_MISMATCH)
                self.assertEqual(record["different_pixel_count"], 1)
                self.assertFalse(record["exact_match"])
                self.assertIn("difference_mask", record)
                self.assertTrue(Path(record["difference_mask"]).is_file())

            matrix = json.loads(
                (output_dir / "combination_matrix_27_with_semantic_latency.json").read_text(encoding="utf-8")
            )
            self.assertEqual(matrix["semantic_parity_status"], benchmark.PARITY_MISMATCH_PRESENT)
            for row in matrix["rows"]:
                self.assertEqual(row["semantic_parity_status"], benchmark.PARITY_MISMATCH_PRESENT)
                self.assertEqual(row["semantic_latency_boundary"], benchmark.BOUNDARY_MISMATCH)
                self.assertIsNone(row["measured_end_to_end_latency_ms"])
                self.assertIs(row["realtime_compatible"], False)

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["parity"]["exact_match_count"], 0)
            self.assertEqual(summary["parity"]["semantic_parity_status"], benchmark.PARITY_MISMATCH_PRESENT)

    def test_warmup_count_is_excluded_from_the_main_statistics(self) -> None:
        with ascii_tempdir() as temporary:
            root = temporary
            dirs = build_inputs(root)
            output_dir = root / "warmup_output"
            recorded: list[dict] = []
            run_main(
                "run_single_image_online",
                fake_stage_runner(recorded),
                full_argv(dirs, output_dir, repeats=1, warmup=3),
            )
            self.assertEqual(len(recorded), len(PAIR_IDS) * 2 + 3)
            timing = json.loads((output_dir / "timing_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(timing["main_measurement_record_count"], len(PAIR_IDS) * 2)
            self.assertEqual(timing["warmup_count"], 3)
            self.assertEqual(timing["warmup_target"], "left pair_0000")

    def test_probe_only_writes_nothing(self) -> None:
        with ascii_tempdir() as temporary:
            root = temporary
            dirs = build_inputs(root)
            recorded: list[dict] = []
            before = sorted(item.name for item in root.iterdir())
            argv = [
                "--left-image-dir", str(dirs["left_images"]),
                "--right-image-dir", str(dirs["right_images"]),
                "--left-cache-dir", str(dirs["left_cache"]),
                "--right-cache-dir", str(dirs["right_cache"]),
                "--left-label-dir", str(dirs["left_labels"]),
                "--right-label-dir", str(dirs["right_labels"]),
                "--pair-ids", "0",
                "--device", "cpu",
                "--probe-only",
                "--probe-max-pairs", "1",
            ]
            exit_code = run_main("run_single_image_online", fake_stage_runner(recorded), argv)
            self.assertEqual(exit_code, 0)
            after = sorted(item.name for item in root.iterdir())
            self.assertEqual(before, after)
            self.assertEqual(len(recorded), 2)  # left and right, no warm-up, no writes


if __name__ == "__main__":
    unittest.main()
