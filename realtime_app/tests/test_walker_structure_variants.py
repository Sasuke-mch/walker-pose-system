"""Unit tests for the controlled walker structure-variant chain.

The tests use synthetic masks and an injected stereo stub where a real stereo
run is not the object under test, so the skeleton band, the deterministic
component assignment, the grouping and the failure reporting are exercised
directly and without any calibration or model dependency.
"""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for _candidate in (str(ROOT), str(ROOT / "tools")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from pose_app import walker_structure_variants as variants  # noqa: E402
from pose_app.calibration import StereoCalibration  # noqa: E402

import benchmark_walker_structure_variants as benchmark  # noqa: E402
from benchmark_ground_walker_reconstruction import stereo_candidates  # noqa: E402
from observe_local_ground_semantic_stereo import upright_point_to_raw as frozen_upright_to_raw  # noqa: E402


SIZE = 100
FORBIDDEN_PHRASES = ("真实接触", "落地", "真实精度提高", "真实三维精度", "准确率提高")

FROZEN_STEREO_KEYS = ("runtime_width", "runtime_height", "num_disparities")


class IdentityRectification:
    """Rectification maps whose analytic inverse is the identity transport.

    With a square mask domain and unit resize scale the frozen chain's
    ``rotate + resize + remap`` becomes the identity, so a synthetic local
    pixel and its upright pixel coincide.  That makes the component assignment
    of this test observable without any calibration.
    """

    def __init__(self, size: int = SIZE):
        grid_y, grid_x = np.mgrid[0:size, 0:size]
        self.left_map_x = (size - 1 - grid_y).astype(np.float32)
        self.left_map_y = grid_x.astype(np.float32)
        self.right_map_x = self.left_map_x
        self.right_map_y = self.left_map_y


def band_stereo_stub(size: int = SIZE):
    """Return one strict 3-D point per pixel of the mask handed to the chain."""

    def stereo(left, right, left_mask, right_mask, calibration, parameters, *, mask_dilation_px=0):
        ys, xs = np.nonzero(np.asarray(left_mask) > 0)
        if len(xs) == 0:
            return {"status": "unavailable", "reasons": ["empty_semantic_mask"]}
        points = np.column_stack((xs.astype(np.float64), ys.astype(np.float64), np.full(len(xs), 1000.0)))
        return {
            "status": "candidate",
            "reasons": [],
            "points": points,
            "left_pixels": np.column_stack((xs, ys)).astype(np.int32),
            "right_pixels": np.column_stack((xs, ys)).astype(np.int32),
            "left_mask_for_matching": left_mask,
            "right_mask_for_matching": right_mask,
            "left_mask_local": left_mask,
            "right_mask_local": right_mask,
            "rectification": IdentityRectification(size),
            "mask_dilation_px": int(mask_dilation_px),
            "matching_funnel": {
                "left_mask_pixels": int(len(xs)),
                "final_candidates": int(len(xs)),
            },
        }

    return stereo


def rod_mask(xs: tuple[int, ...], size: int = SIZE) -> np.ndarray:
    """Solid rectangles: the frozen skeleton of these is one clean component each."""
    mask = np.zeros((size, size), dtype=np.uint8)
    for x in xs:
        cv2.rectangle(mask, (x - 5, 10), (x + 5, 89), 255, -1)
    return mask


def synthetic_parameters(size: int = SIZE) -> dict:
    return {
        "runtime_width": size,
        "runtime_height": size,
        "num_disparities": 160,
        "lr_consistency_px": 1.5,
    }


def write_png(path: Path, image: np.ndarray) -> None:
    """Write a PNG without OpenCV's file layer.

    ``cv2.imwrite`` cannot address this host's temporary directory because its
    path contains non-ASCII characters, so the encoder buffer is written with
    Python's own file IO instead.  Only the file names matter to the inventory
    assertions under test.
    """
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"cannot encode {path}")
    buffer.tofile(str(path))


def empty_state(frame_id: str = "pair_0000.png") -> dict:
    return {
        "frame_id": frame_id,
        "points": np.empty((0, 3), dtype=np.float64),
        "upright": np.empty((0, 2), dtype=np.float64),
        "component_id": np.empty(0, dtype=np.int64),
        "labels_left": np.zeros((SIZE, SIZE), dtype=np.int32),
        "skeleton_lengths_left": {},
        "skeleton_left": np.zeros((SIZE, SIZE), dtype=np.uint8),
        "skeleton_right": np.zeros((SIZE, SIZE), dtype=np.uint8),
        "band_left": np.zeros((SIZE, SIZE), dtype=np.uint8),
        "band_right": np.zeros((SIZE, SIZE), dtype=np.uint8),
        "candidate": None,
    }


class WalkerStructureVariantsTests(unittest.TestCase):
    def test_thick_rod_skeleton_is_far_shorter_than_the_mask_and_keeps_its_points(self):
        mask = rod_mask((50,))
        mask_pixels = int(np.count_nonzero(mask))
        record, state = benchmark.skeleton_method(
            left_upright=np.zeros((SIZE, SIZE, 3), dtype=np.uint8),
            right_upright=np.zeros((SIZE, SIZE, 3), dtype=np.uint8),
            left_mask=mask, right_mask=mask,
            calibration=None, parameters=synthetic_parameters(),
            band_radius_px=3, stereo_fn=band_stereo_stub(),
        )
        self.assertLess(record["left_skeleton_pixels"], mask_pixels / 5)
        band = state["band_left"]
        # the band can only live inside the mask dilated by its own radius, and
        # it must stay overwhelmingly next to the rod it came from
        dilated = variants.skeleton_band(mask, 3)
        self.assertTrue(np.all((band > 0) <= (dilated > 0)))
        self.assertGreater(float(((band > 0) & (mask > 0)).sum()) / float((band > 0).sum()), 0.5)
        self.assertGreater(record["left_skeleton_band_pixels"], record["left_skeleton_pixels"])
        self.assertEqual(record["status"], "candidate")
        self.assertEqual(len(record["line_primitives"]), 1)
        self.assertGreaterEqual(record["line_primitives"][0]["point_count"], 10)
        # the fitted 3-D rod must follow the synthetic image-space rod direction
        direction = np.abs(np.asarray(record["line_primitives"][0]["direction_unit"]))
        self.assertGreater(direction[1], 0.9)
        self.assertEqual(record["strict_points_without_skeleton_component"], 0)

    def test_two_disjoint_rods_stay_two_separate_components(self):
        mask = rod_mask((25, 75))
        record, state = benchmark.skeleton_method(
            left_upright=np.zeros((SIZE, SIZE, 3), dtype=np.uint8),
            right_upright=np.zeros((SIZE, SIZE, 3), dtype=np.uint8),
            left_mask=mask, right_mask=mask,
            calibration=None, parameters=synthetic_parameters(),
            band_radius_px=3, stereo_fn=band_stereo_stub(),
        )
        self.assertEqual(record["left_skeleton_component_count"], 2)
        self.assertEqual(record["status"], "candidate")
        self.assertEqual(len(record["line_primitives"]), 2)
        identifiers = sorted(item["component_id"] for item in record["line_primitives"])
        self.assertEqual(len(set(identifiers)), 2)
        centers = sorted(np.asarray(item["center_left_camera_mm"])[0] for item in record["line_primitives"])
        self.assertGreater(abs(centers[1] - centers[0]), 40.0)
        # each fitted rod stays on its own rod instead of spanning both
        for center in centers:
            self.assertLess(min(abs(center - 25.0), abs(center - 75.0)), 6.0)
        # no band pixel may be shared by both components and nothing may be lost
        assigned = sum(item["point_count"] for item in record["line_primitives"])
        self.assertEqual(assigned, record["left_skeleton_band_pixels"])
        self.assertEqual(record["strict_points_without_skeleton_component"], 0)

    def test_empty_masks_leave_all_three_methods_unavailable_without_crashing(self):
        empty = np.zeros((SIZE, SIZE), dtype=np.uint8)
        parameters = synthetic_parameters()
        record, state = benchmark.evaluate_frame(
            name="pair_0000.png",
            left_upright=np.zeros((SIZE, SIZE, 3), dtype=np.uint8),
            right_upright=np.zeros((SIZE, SIZE, 3), dtype=np.uint8),
            left_mask=empty, right_mask=empty,
            calibration=None, parameters=parameters,
            semantic_evidence="automatic_candidate", band_radius_px=3,
        )
        temporal, points = benchmark.temporal_method(
            frame_index=0, window_indices=[0], requested_window=5,
            states=[state], band_radius_px=3, line_count_skeleton=0,
        )
        record["methods"][benchmark.METHOD_TEMPORAL] = temporal
        for method in benchmark.METHODS:
            entry = record["methods"][method]
            self.assertEqual(entry["status"], "unavailable")
            self.assertEqual(entry["line_primitives"], [])
            self.assertTrue(entry["reasons"])
            self.assertEqual(entry["input_points"]["count"], 0)
        self.assertEqual(len(points), 0)
        self.assertFalse(record["mask_status"]["both_nonempty"])
        self.assertIn("empty_left_walker_mask", record["mask_status"]["reasons"])

    def test_left_only_mask_makes_the_tool_fail_instead_of_copying_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left_images, right_images = root / "left_img", root / "right_img"
            left_masks, right_masks = root / "left_mask", root / "right_mask"
            for directory in (left_images, right_images, left_masks, right_masks):
                directory.mkdir()
            image = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
            for name in ("pair_0000.png", "pair_0001.png"):
                write_png(left_images / name, image)
                write_png(right_images / name, image)
                write_png(left_masks / name, mask)
            # only the left side has a mask: nothing may be copied to the right
            with self.assertRaises(RuntimeError) as raised:
                benchmark.main([
                    "--left-dir", str(left_images), "--right-dir", str(right_images),
                    "--left-mask-dir", str(left_masks), "--right-mask-dir", str(right_masks),
                    "--semantic-evidence", "automatic_candidate",
                    "--calibration", str(root / "missing.json"),
                    "--frozen-baseline-dir", str(root / "missing_frozen"),
                    "--output-dir", str(root / "output"),
                ])
            self.assertIn("no mask is copied", str(raised.exception).replace("No mask is copied", "no mask is copied"))
            # a partially paired set keeps only the frame that has both sides
            write_png(right_masks / "pair_0000.png", mask)
            found = benchmark.inventory(benchmark.parse_args([
                "--left-dir", str(left_images), "--right-dir", str(right_images),
                "--left-mask-dir", str(left_masks), "--right-mask-dir", str(right_masks),
                "--semantic-evidence", "automatic_candidate",
                "--calibration", str(root / "missing.json"),
                "--frozen-baseline-dir", str(root / "missing_frozen"),
                "--output-dir", str(root / "output"),
            ]))
            self.assertEqual(found["paired_mask_count"], 1)
            self.assertEqual(found["usable_frame_count"], 1)
            self.assertEqual(found["left_only_mask_names"], ["pair_0001.png"])
            self.assertEqual(benchmark.select_frame_names(found, 12), ["pair_0000.png"])

    def test_temporal_window_without_second_frame_support_is_unavailable(self):
        state = empty_state("pair_0001.png")
        state["points"] = np.asarray([[0.0, 0.0, 1000.0], [5.0, 5.0, 1000.0]], dtype=np.float64)
        state["upright"] = np.asarray([[10.0, 10.0], [11.0, 11.0]], dtype=np.float64)
        state["component_id"] = np.asarray([1, 1], dtype=np.int64)
        labels = np.zeros((SIZE, SIZE), dtype=np.int32)
        labels[9:13, 9:13] = 1
        state["labels_left"] = labels
        state["skeleton_lengths_left"] = {1: 16.0}
        states = [empty_state("pair_0000.png"), state, empty_state("pair_0002.png")]
        record, points = benchmark.temporal_method(
            frame_index=1, window_indices=[0, 1, 2], requested_window=5,
            states=states, band_radius_px=3, line_count_skeleton=0,
        )
        self.assertEqual(record["minimum_frame_support"], 2)
        self.assertEqual(record["window_frame_ids"], ["pair_0000.png", "pair_0001.png", "pair_0002.png"])
        self.assertEqual(record["input_point_count"], 2)
        self.assertEqual(record["consensus_voxel_count"], 0)
        self.assertEqual(record["status"], "unavailable")
        self.assertIn("no_temporally_repeated_geometry", record["reasons"])
        self.assertEqual(len(points), 0)

    def test_every_interpretation_text_avoids_contact_support_and_accuracy_claims(self):
        mask = rod_mask((50,))
        record, state = benchmark.evaluate_frame(
            name="pair_0000.png",
            left_upright=np.zeros((SIZE, SIZE, 3), dtype=np.uint8),
            right_upright=np.zeros((SIZE, SIZE, 3), dtype=np.uint8),
            left_mask=mask, right_mask=mask,
            calibration=None, parameters=synthetic_parameters(),
            semantic_evidence="automatic_candidate", band_radius_px=3,
            stereo_fn=band_stereo_stub(),
        )
        temporal, _ = benchmark.temporal_method(
            frame_index=0, window_indices=[0], requested_window=5,
            states=[state], band_radius_px=3, line_count_skeleton=1,
        )
        record["methods"][benchmark.METHOD_TEMPORAL] = temporal
        texts = [
            record["interpretation_boundary"],
            variants.STATISTICS_BOUNDARY,
            json.dumps(record, ensure_ascii=False),
            json.dumps(benchmark.METHOD_INTERPRETATIONS, ensure_ascii=False),
            json.dumps(record["mask_status"], ensure_ascii=False),
        ]
        rendered = benchmark.render_experiment_markdown(
            run_parameters={
                "semantic_evidence": "automatic_candidate",
                "skeleton_band_radius_px": 3,
                "left_dir": "l", "right_dir": "r", "left_mask_dir": "lm", "right_mask_dir": "rm",
                "calibration": "c", "frozen_baseline_dir": "f",
            },
            found={"left_image_count": 1, "right_image_count": 1, "left_mask_count": 1,
                   "right_mask_count": 1, "paired_image_count": 1, "paired_mask_count": 1,
                   "usable_frame_count": 1},
            names=["pair_0000.png"],
            summary=benchmark.build_summary([record], {}),
        )
        texts.append(rendered)
        for text in texts:
            for phrase in FORBIDDEN_PHRASES:
                self.assertNotIn(phrase, text)
        for method in benchmark.METHODS:
            self.assertNotIn("真实接触", record["methods"][method]["interpretation"])
            self.assertNotIn("落地", record["methods"][method]["interpretation"])
            self.assertIn("not a metric accuracy claim", record["methods"][method]["interpretation"])

    def test_automatic_evidence_is_recorded_and_never_upgraded(self):
        mask = rod_mask((50,))
        record, state = benchmark.evaluate_frame(
            name="pair_0000.png",
            left_upright=np.zeros((SIZE, SIZE, 3), dtype=np.uint8),
            right_upright=np.zeros((SIZE, SIZE, 3), dtype=np.uint8),
            left_mask=mask, right_mask=mask,
            calibration=None, parameters=synthetic_parameters(),
            semantic_evidence="automatic_candidate", band_radius_px=3,
            stereo_fn=band_stereo_stub(),
        )
        temporal, _ = benchmark.temporal_method(
            frame_index=0, window_indices=[0], requested_window=5,
            states=[state], band_radius_px=3, line_count_skeleton=1,
        )
        record["methods"][benchmark.METHOD_TEMPORAL] = temporal
        self.assertEqual(record["semantic_evidence"], "automatic_candidate")
        self.assertFalse(record["semantic_evidence_upgraded"])
        summary = benchmark.build_summary([record], {})
        self.assertEqual(summary["semantic_evidence"], "automatic_candidate")
        self.assertIn("never promoted", summary["semantic_evidence_note"])
        self.assertTrue(all(
            stats["statistics_boundary"] == variants.STATISTICS_BOUNDARY
            for stats in summary["method_statistics"].values()
        ))

    def test_nearest_component_ties_go_to_the_smaller_component_id(self):
        skeleton = np.zeros((SIZE, SIZE), dtype=np.uint8)
        skeleton[5, 5] = 255
        skeleton[5, 9] = 255
        labels, stats = variants.skeleton_component_labels(skeleton)
        self.assertEqual(len(stats) - 1, 2)
        result = variants.nearest_skeleton_components(
            np.asarray([[7.0, 5.0], [5.0, 5.0], [50.0, 50.0]]), labels, band_radius_px=3
        )
        self.assertEqual(int(result["component_id"][0]), 1)
        self.assertEqual(int(result["component_id"][1]), 1)
        self.assertEqual(int(result["component_id"][2]), 0)
        self.assertEqual(result["unassigned"], 1)
        self.assertTrue(np.isinf(result["distance_px"][2]))

    def test_skeleton_band_and_pixel_transport_conventions(self):
        mask = rod_mask((50,))
        skeleton = variants.morphological_skeleton(mask)
        with self.assertRaises(ValueError):
            variants.skeleton_band(skeleton, 0)
        band = variants.skeleton_band(skeleton, 3)
        self.assertTrue(np.all((band > 0) <= (variants.skeleton_band(mask, 3) > 0)))
        upright = np.asarray([[0.0, 0.0], [SIZE - 1.0, SIZE - 1.0], [13.0, 71.0]])
        upright_size = (SIZE, SIZE)
        for side in ("left", "right"):
            raw = variants.upright_points_to_raw(upright, upright_shape=(SIZE, SIZE), side=side)
            # the vectorised convention must equal the frozen scalar convention
            for index, point in enumerate(upright):
                frozen = frozen_upright_to_raw(point, side, upright_size)
                self.assertTrue(np.allclose(frozen, raw[index]))
            back = variants.raw_points_to_upright(raw, upright_shape=(SIZE, SIZE), side=side)
            self.assertTrue(np.allclose(upright, back))
        rectification = IdentityRectification(SIZE)
        pixels = np.asarray([[11, 7], [40, 40]], dtype=np.int64)
        transported = variants.local_pixels_to_upright(
            pixels, rectification.left_map_x, rectification.left_map_y,
            runtime_size=(SIZE, SIZE), upright_shape=(SIZE, SIZE), side="left",
        )
        self.assertTrue(np.allclose(transported, pixels.astype(np.float64)))

    def test_consensus_provenance_matches_the_frozen_function(self):
        first = np.asarray([[0.0, 0.0, 900.0], [300.0, 0.0, 900.0]])
        second = np.asarray([[1.0, 1.0, 901.0], [301.0, 1.0, 902.0]])
        third = np.asarray([[500.0, 500.0, 1500.0]])
        mirrored = variants.consensus_with_provenance(
            [first, second, third], voxel_size_mm=20.0, minimum_frame_support=2
        )
        frozen = variants.camera_attached_temporal_consensus(
            [first, second, third], voxel_size_mm=20.0, minimum_frame_support=2
        )
        self.assertEqual(mirrored["consensus_voxel_count"], frozen["consensus_voxel_count"])
        self.assertEqual(mirrored["consensus_voxel_count"], 2)
        self.assertEqual(
            mirrored["consensus_points_left_camera_mm"], frozen["consensus_points_left_camera_mm"]
        )
        self.assertEqual(len(mirrored["voxel_members"]), 2)
        self.assertEqual(mirrored["voxel_members"][0], [(0, 0), (1, 0)])

    def test_grouped_line_candidates_reports_reasons_instead_of_dropping_groups(self):
        points = np.column_stack((
            np.zeros(40), np.arange(40, dtype=np.float64) * 5.0, np.full(40, 900.0),
        ))
        identifiers = np.zeros(40, dtype=np.int64)
        identifiers[:4] = 1      # too few points
        identifiers[4:] = 2      # valid line
        grouped = variants.grouped_line_candidates(points, identifiers, minimum_points=10)
        self.assertEqual(len(grouped["candidates"]), 1)
        self.assertEqual(len(grouped["failures"]), 1)
        self.assertEqual(grouped["failures"][0]["reasons"], ["insufficient_component_stereo_points"])
        degenerate = np.zeros((12, 3), dtype=np.float64)
        grouped = variants.grouped_line_candidates(
            degenerate, np.full(12, 3, dtype=np.int64), minimum_points=10
        )
        self.assertEqual(grouped["candidates"], [])
        self.assertEqual(grouped["failures"][0]["reasons"], ["degenerate_component_geometry"])
        rejected = variants.grouped_line_candidates(
            points, identifiers, minimum_points=10, group_reasons={2: "degenerate_component_geometry"}
        )
        self.assertEqual(rejected["candidates"], [])

    def test_cli_rejects_reused_output_and_invalid_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / "already_there"
            existing.mkdir()
            base = [
                "--left-dir", str(root), "--right-dir", str(root),
                "--left-mask-dir", str(root), "--right-mask-dir", str(root),
                "--semantic-evidence", "automatic_candidate",
                "--calibration", str(root / "c.json"),
                "--frozen-baseline-dir", str(root / "frozen"),
                "--output-dir", str(existing),
            ]
            with self.assertRaises(FileExistsError):
                benchmark.main(base)
            for extra, message in (
                (["--temporal-window", "4"], "--temporal-window must be a positive odd integer"),
                (["--maximum-pairs", "0"], "--maximum-pairs must be a positive integer"),
                (["--skeleton-band-radius-px", "0"], "--skeleton-band-radius-px must be a positive integer"),
            ):
                arguments = list(base)
                arguments[arguments.index("--output-dir") + 1] = str(root / "fresh")
                with self.assertRaises(ValueError) as raised:
                    benchmark.main(arguments + extra)
                self.assertIn(message, str(raised.exception))

    def test_frame_selection_uses_numeric_pair_order(self):
        found = {"usable_names": {"pair_0010.png", "pair_0002.png", "pair_0000.png"}}
        self.assertEqual(
            benchmark.select_frame_names(found, 2), ["pair_0000.png", "pair_0002.png"]
        )
        self.assertEqual(benchmark.pair_id("pair_0120.png"), 120)
        with self.assertRaises(RuntimeError):
            benchmark.pair_id("frame_0001.png")

    def test_real_stereo_chain_is_reused_unchanged_for_the_empty_mask_case(self):
        """The frozen entry point, not a stub, must report an empty mask."""
        empty = np.zeros((SIZE, SIZE), dtype=np.uint8)
        result = stereo_candidates(
            np.zeros((SIZE, SIZE, 3), dtype=np.uint8), np.zeros((SIZE, SIZE, 3), dtype=np.uint8),
            empty, empty, None, synthetic_parameters(),
        )
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["reasons"], ["empty_semantic_mask"])
        self.assertTrue(all(key in synthetic_parameters() for key in FROZEN_STEREO_KEYS))
        self.assertIsInstance(StereoCalibration, type)


if __name__ == "__main__":
    unittest.main()
