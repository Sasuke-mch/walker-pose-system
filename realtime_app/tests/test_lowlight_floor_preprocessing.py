"""Tests for the fixed classical low-light floor-preprocessing comparison.

Every test is synthetic: no camera data, no model, no calibration and no
experiment output is read or written outside a temporary directory.
"""

from __future__ import annotations

import ast
import inspect
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from pose_app import lowlight_floor_preprocessing as lowlight  # noqa: E402
from pose_app.scene_geometry_variants import estimate_region_consensus  # noqa: E402
import benchmark_lowlight_floor_preprocessing as benchmark  # noqa: E402
import observe_local_ground_semantic_stereo_lr_direction_control as corrected  # noqa: E402


def region_minimum_points() -> int:
    return benchmark.REGION_MINIMUM_POINTS


def synthetic_upright(seed: int = 20260913, height: int = 64, width: int = 64) -> np.ndarray:
    """A textured BGR image with a tile-aligned dark block of known coordinates."""
    rng = np.random.default_rng(seed)
    image = rng.integers(80, 161, size=(height, width, 3), dtype=np.uint8)
    image[24:40, 24:40, :] = 0
    return image


def ascii_tempdir() -> tempfile.TemporaryDirectory:
    """A temporary directory whose path has no non-ASCII characters.

    OpenCV's image codecs use the narrow-character Windows API, so imread and
    imwrite fail on a user-profile path that contains non-ASCII characters.
    """
    return tempfile.TemporaryDirectory(dir=str(ROOT.parent))


def write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cv2.imencode(".png", array)[1].tobytes())


def dev_payload(ious, precisions, variances, recalls=None) -> dict:
    recalls = recalls or {method: 0.5 for method in lowlight.METHODS}
    return {
        "split": "dev",
        "methods": {
            method: {
                "left_right_macro": {
                    "iou": ious[method],
                    "precision": precisions[method],
                    "recall": recalls[method],
                    "views_contributing": 2,
                    "view_macro_triples": [],
                },
                "iou_distribution": {"variance": variances[method], "count": 12},
            }
            for method in lowlight.METHODS
        },
    }


# ---------------------------------------------------------------------------
# 1-4: transforms, contract and coordinate alignment
# ---------------------------------------------------------------------------


class TransformTests(unittest.TestCase):
    def test_raw_is_a_pixel_exact_identity_with_unchanged_size(self) -> None:
        image = synthetic_upright()
        output = lowlight.apply_preprocessing(lowlight.RAW, image)
        self.assertEqual(output.shape, image.shape)
        self.assertEqual(output.dtype, np.uint8)
        self.assertTrue(np.array_equal(output, image))

    def test_gamma_returns_uint8_keeps_size_and_brightens_dark_non_zero_pixels(self) -> None:
        ramp = np.arange(256, dtype=np.uint8).reshape(1, 256, 1)
        image = np.repeat(ramp, 3, axis=2)
        output = lowlight.apply_preprocessing(lowlight.GAMMA_0P6, image)
        self.assertEqual(output.dtype, np.uint8)
        self.assertEqual(output.shape, image.shape)
        expected = np.rint(np.clip(255.0 * (np.arange(256, dtype=np.float64) / 255.0) ** 0.6, 0.0, 255.0))
        self.assertTrue(np.array_equal(output[0, :, 0].astype(np.float64), expected))
        for value in (1, 2, 3, 5, 8, 13, 50, 100, 200):
            self.assertGreater(int(output[0, value, 0]), value)
        self.assertEqual(int(output[0, 0, 0]), 0)
        self.assertEqual(int(output[0, 255, 0]), 255)

    def test_clahe_and_combined_keep_size_three_channels_and_finite_values(self) -> None:
        image = synthetic_upright()
        for method in (lowlight.CLAHE_LAB, lowlight.GAMMA_0P6_THEN_CLAHE_LAB):
            output = lowlight.apply_preprocessing(method, image)
            self.assertEqual(output.shape, image.shape, method)
            self.assertEqual(output.ndim, 3, method)
            self.assertEqual(output.shape[2], 3, method)
            self.assertEqual(output.dtype, np.uint8, method)
            self.assertTrue(np.all(np.isfinite(output.astype(np.float64))), method)

    def test_combined_group_is_gamma_then_clahe_in_that_order(self) -> None:
        image = synthetic_upright()
        self.assertTrue(
            np.array_equal(
                lowlight.apply_preprocessing(lowlight.GAMMA_0P6_THEN_CLAHE_LAB, image),
                lowlight.clahe_lab(lowlight.gamma_0p6(image)),
            )
        )

    def test_every_transform_keeps_the_marked_pixel_at_its_original_coordinates(self) -> None:
        image = synthetic_upright()
        for method in lowlight.METHODS:
            output = lowlight.apply_preprocessing(method, image)
            gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            row, column = np.unravel_index(int(np.argmin(gray)), gray.shape)
            self.assertTrue(24 <= row < 40 and 24 <= column < 40, f"{method} moved the marker to {(row, column)}")
            block = gray[24:40, 24:40]
            background = np.concatenate([gray[0:24, :].ravel(), gray[40:, :].ravel()])
            self.assertLess(float(block.mean()), float(background.mean()), method)

    def test_contract_reports_the_alignment_and_the_raw_identity(self) -> None:
        image = synthetic_upright()
        for method in lowlight.METHODS:
            checks = lowlight.assert_transform_contract(image, lowlight.apply_preprocessing(method, image), method)
            self.assertTrue(checks["same_shape_as_original"])
            self.assertTrue(checks["same_three_bgr_channels"])
            self.assertTrue(checks["all_values_finite"])
            self.assertEqual(checks["pixel_exact_identity"], method == lowlight.RAW)
            self.assertIn("coordinates", checks["coordinate_alignment"])

    def test_contract_refuses_a_resized_transform_output(self) -> None:
        image = synthetic_upright()
        resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
        for method in lowlight.METHODS:
            with self.assertRaises(lowlight.TransformContractError):
                lowlight.assert_transform_contract(image, resized, method)

    def test_contract_refuses_a_non_identity_raw_group(self) -> None:
        image = synthetic_upright()
        with self.assertRaises(lowlight.TransformContractError):
            lowlight.assert_transform_contract(image, lowlight.gamma_0p6(image), lowlight.RAW)

    def test_only_the_four_frozen_groups_exist_and_their_parameters_are_constants(self) -> None:
        self.assertEqual(len(lowlight.METHODS), 4)
        self.assertEqual(
            lowlight.METHODS,
            ("raw", "gamma_0p6", "clahe_lab", "gamma_0p6_then_clahe_lab"),
        )
        self.assertEqual(lowlight.GAMMA_EXPONENT, 0.6)
        self.assertEqual(lowlight.CLAHE_CLIP_LIMIT, 2.0)
        self.assertEqual(lowlight.CLAHE_TILE_GRID_SIZE, (8, 8))
        self.assertEqual(lowlight.method_parameters(lowlight.RAW), {"pixel_transform": "none"})
        with self.assertRaises(ValueError):
            lowlight.apply_preprocessing("gamma_0p7", synthetic_upright())
        with self.assertRaises(ValueError):
            lowlight.method_parameters("clahe_lab_plus")


class MaskCoordinateContractTests(unittest.TestCase):
    def test_exact_grid_mask_is_accepted(self) -> None:
        mask = np.zeros((16, 20), dtype=np.uint8)
        mask[4:9, 5:11] = 255
        with ascii_tempdir() as directory:
            path = Path(directory) / "pair_0000.png"
            write_png(path, mask)
            loaded = lowlight.load_binary_mask_exact(path, (16, 20))
        self.assertEqual(loaded.dtype, bool)
        self.assertEqual(int(loaded.sum()), 30)

    def test_a_mask_on_a_different_grid_is_refused_instead_of_resampled(self) -> None:
        mask = np.full((8, 10), 255, dtype=np.uint8)
        with ascii_tempdir() as directory:
            path = Path(directory) / "pair_0000.png"
            write_png(path, mask)
            with self.assertRaises(lowlight.TransformContractError):
                lowlight.load_binary_mask_exact(path, (16, 20))

    def test_a_missing_mask_is_reported(self) -> None:
        with ascii_tempdir() as directory:
            with self.assertRaises(lowlight.TransformContractError):
                lowlight.load_binary_mask_exact(Path(directory) / "missing.png", (8, 8))


# ---------------------------------------------------------------------------
# 4 bis: the frozen stereo candidate chain is reproduced by the funnel
# ---------------------------------------------------------------------------


def synthetic_stereo_case():
    height, width = 6, 12
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    forward = np.full((height, width), 3.0, dtype=np.float32)
    reverse = np.full((height, width), 3.0, dtype=np.float32)
    left_mask = np.zeros((height, width), dtype=np.uint8)
    left_mask[:, 2:] = 255
    right_mask = np.full((height, width), 255, dtype=np.uint8)
    rectification = type("SyntheticRectification", (), {})()
    rectification.translation_right = np.asarray([-30.0, 0.0, 0.0])
    rectification.virtual_K = np.asarray(
        [[330.0, 0.0, width / 2.0], [0.0, 330.0, height / 2.0], [0.0, 0.0, 1.0]]
    )
    rectification.rotation_left = np.eye(3)
    return image, forward, reverse, left_mask, right_mask, rectification


class StereoFunnelTests(unittest.TestCase):
    def test_stage_funnel_reproduces_the_frozen_candidate_pixel_set(self) -> None:
        image, forward, reverse, left_mask, right_mask, rectification = synthetic_stereo_case()
        candidate = corrected.reconstruct(
            image, image, left_mask, right_mask, rectification, forward, reverse, 160, 1.5
        )
        self.assertGreater(len(candidate["points"]), 0)
        funnel = benchmark.stereo_candidate_funnel(
            left_local=image, right_local=image, left_mask_local=left_mask, right_mask_local=right_mask,
            rectification=rectification, forward=forward, reverse=reverse, consistency_px=1.5,
            candidate=candidate,
        )
        self.assertTrue(funnel["reproduced_frozen_candidate_pixels"])
        self.assertEqual(funnel["strict_stereo_candidate_pixels"], len(candidate["points"]))
        self.assertEqual(
            funnel["strict_stereo_candidate_pixels"], funnel["pixels_passing_depth_range"]
        )
        self.assertGreaterEqual(
            funnel["pixels_reaching_left_right_consistency_test"],
            funnel["pixels_passing_left_right_consistency"],
        )
        stages = [entry["stage"] for entry in funnel["stages"]]
        self.assertEqual(stages[0], "left_mask")
        self.assertEqual(stages[-1], "depth_in_range")
        self.assertIn("left_right_consistency", stages)

    def test_a_funnel_that_does_not_match_the_frozen_chain_is_refused(self) -> None:
        image, forward, reverse, left_mask, right_mask, rectification = synthetic_stereo_case()
        candidate = corrected.reconstruct(
            image, image, left_mask, right_mask, rectification, forward, reverse, 160, 1.5
        )
        with self.assertRaises(RuntimeError):
            benchmark.stereo_candidate_funnel(
                left_local=image, right_local=image, left_mask_local=left_mask, right_mask_local=right_mask,
                rectification=rectification, forward=forward,
                reverse=np.zeros_like(reverse), consistency_px=1.5, candidate=candidate,
            )


# ---------------------------------------------------------------------------
# 5: empty mask or no prediction
# ---------------------------------------------------------------------------


def masks_for(shape=(12, 12), floor=True, person=False, walker=False) -> dict:
    height, width = shape
    masks = {name: np.zeros(shape, dtype=bool) for name in
             ("floor_eligible", "person", "walker", "static_other", "ignore_uncertain")}
    if floor:
        masks["floor_eligible"][2:10, 2:10] = True
    if person:
        masks["person"][3:5, 3:5] = True
    if walker:
        masks["walker"][6:8, 6:8] = True
    masks["valid_evaluation"] = np.ones(shape, dtype=bool)
    return masks


class SemanticEvaluationTests(unittest.TestCase):
    def test_no_prediction_is_unavailable_with_an_explicit_reason(self) -> None:
        result = benchmark.evaluate_semantic_candidate(masks_for(), None)
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("no_semantic_prediction", result["reasons"])
        self.assertIsNone(result["metrics"])

    def test_an_empty_candidate_mask_does_not_crash_and_carries_a_reason(self) -> None:
        empty = np.zeros((12, 12), dtype=bool)
        result = benchmark.evaluate_semantic_candidate(masks_for(), empty)
        self.assertEqual(result["status"], "candidate")
        self.assertIn("empty_candidate_mask", result["reasons"])
        self.assertEqual(result["metrics"]["iou"], 0.0)
        self.assertEqual(result["metrics"]["recall"], 0.0)
        self.assertEqual(result["metrics"]["precision"], 0.0)

    def test_no_reference_floor_pixels_is_unavailable(self) -> None:
        masks = masks_for(floor=False)
        result = benchmark.evaluate_semantic_candidate(masks, np.ones((12, 12), dtype=bool))
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("no_reference_floor_pixels_in_valid_area", result["reasons"])
        self.assertIsNone(result["metrics"])

    def test_a_candidate_mask_on_another_grid_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            benchmark.evaluate_semantic_candidate(masks_for(), np.ones((6, 6), dtype=bool))

    def test_person_and_walker_leakage_is_reported_with_its_sample_count(self) -> None:
        masks = masks_for(person=True, walker=True)
        candidate = np.zeros((12, 12), dtype=bool)
        candidate[3:5, 3:5] = True
        result = benchmark.evaluate_semantic_candidate(masks, candidate)
        self.assertEqual(result["person_labelled_pixels_in_valid_area"], 4)
        self.assertEqual(result["person_pixels_predicted_floor"], 4)
        self.assertEqual(result["person_to_floor_fraction"], 1.0)
        self.assertEqual(result["walker_labelled_pixels_in_valid_area"], 4)
        self.assertEqual(result["walker_to_floor_fraction"], 0.0)

    def test_a_class_without_any_label_reports_a_none_fraction_and_zero_samples(self) -> None:
        result = benchmark.evaluate_semantic_candidate(masks_for(), np.ones((12, 12), dtype=bool))
        self.assertEqual(result["walker_labelled_pixels_in_valid_area"], 0)
        self.assertIsNone(result["walker_to_floor_fraction"])

    def test_aggregation_reports_macro_values_and_the_per_frame_distribution(self) -> None:
        records = []
        for pair_id, iou in ((0, 1.0), (80, 0.0)):
            records.append({
                "pair_id": pair_id,
                "view": "left",
                "evaluation": {
                    "status": "candidate",
                    "reasons": [],
                    "metrics": {"precision": 1.0, "recall": iou, "iou": iou,
                                "candidate_floor_px_in_valid_area": 1},
                    "person_labelled_pixels_in_valid_area": 0,
                    "walker_labelled_pixels_in_valid_area": 0,
                    "static_other_labelled_pixels_in_valid_area": 0,
                    "person_pixels_predicted_floor": 0,
                    "walker_pixels_predicted_floor": 0,
                    "static_other_pixels_predicted_floor": 0,
                    "person_to_floor_fraction": None,
                    "walker_to_floor_fraction": None,
                    "static_other_to_floor_fraction": None,
                },
                "brightness": {
                    "floor_region": {"median_gray_0_255": 10.0},
                    "non_floor_region": {"median_gray_0_255": 20.0},
                    "floor_region_median_gray_delta_vs_raw": 0.0,
                },
            })
        aggregate = benchmark.aggregate_view(records)
        self.assertEqual(aggregate["frame_count"], 2)
        self.assertEqual(aggregate["macro_iou"], 0.5)
        self.assertEqual([row["iou"] for row in aggregate["per_frame"]], [1.0, 0.0])


# ---------------------------------------------------------------------------
# 6: development-split-only method selection
# ---------------------------------------------------------------------------


def function_body_source(function) -> str:
    """Source of a function without its docstring."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    node = tree.body[0]
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        node.body = node.body[1:]
    return ast.unparse(node)


class MethodSelectionTests(unittest.TestCase):
    def test_selection_code_never_references_the_holdout_split(self) -> None:
        for function in (
            benchmark.select_primary_frontend,
            benchmark._dev_macro,
            benchmark._dev_variance,
        ):
            source = function_body_source(function).replace("'holdout_pair_ids_used_for_selection'", "")
            self.assertNotIn("holdout", source.lower(), function.__name__)
            self.assertNotIn("semantic_holdout", source.lower(), function.__name__)
        signature = inspect.signature(benchmark.select_primary_frontend)
        self.assertEqual(list(signature.parameters), ["development_metrics"])

    def test_holdout_metrics_in_the_argument_cannot_change_the_decision(self) -> None:
        ious = {"raw": 0.80, "gamma_0p6": 0.83, "clahe_lab": 0.81, "gamma_0p6_then_clahe_lab": 0.79}
        precisions = {method: 0.85 for method in lowlight.METHODS}
        variances = {method: 0.01 for method in lowlight.METHODS}
        dev = dev_payload(ious, precisions, variances)
        plain = benchmark.select_primary_frontend(dev)
        dev_with_holdout = dict(dev)
        dev_with_holdout["holdout"] = {
            method: {"left_right_macro": {"iou": 0.0 if method != "raw" else 1.0}}
            for method in lowlight.METHODS
        }
        dev_with_holdout["semantic_holdout_metrics"] = dev_with_holdout["holdout"]
        contaminated = benchmark.select_primary_frontend(dev_with_holdout)
        self.assertEqual(plain, contaminated)
        self.assertEqual(plain["selected_method"], "gamma_0p6")
        self.assertEqual(plain["holdout_pair_ids_used_for_selection"], [])
        self.assertEqual(plain["criterion"], "development split left-right macro average floor IoU")

    def test_a_margin_within_the_conservative_threshold_keeps_raw(self) -> None:
        ious = {"raw": 0.80, "gamma_0p6": 0.809, "clahe_lab": 0.80, "gamma_0p6_then_clahe_lab": 0.805}
        selection = benchmark.select_primary_frontend(
            dev_payload(ious, {method: 0.85 for method in lowlight.METHODS},
                        {method: 0.01 for method in lowlight.METHODS})
        )
        self.assertTrue(selection["conservative_raw_applied"])
        self.assertEqual(selection["selected_method"], "raw")
        self.assertAlmostEqual(selection["best_margin_over_raw"], 0.009)
        self.assertEqual(selection["locked_parameters"], {"pixel_transform": "none"})

    def test_the_exact_conservative_margin_also_keeps_raw(self) -> None:
        ious = {"raw": 0.80, "gamma_0p6": 0.81, "clahe_lab": 0.79, "gamma_0p6_then_clahe_lab": 0.78}
        selection = benchmark.select_primary_frontend(
            dev_payload(ious, {method: 0.85 for method in lowlight.METHODS},
                        {method: 0.01 for method in lowlight.METHODS})
        )
        self.assertTrue(selection["conservative_raw_applied"])
        self.assertEqual(selection["selected_method"], "raw")

    def test_a_clear_development_winner_is_promoted_with_its_locked_parameters(self) -> None:
        ious = {"raw": 0.80, "gamma_0p6": 0.79, "clahe_lab": 0.86, "gamma_0p6_then_clahe_lab": 0.85}
        selection = benchmark.select_primary_frontend(
            dev_payload(ious, {method: 0.85 for method in lowlight.METHODS},
                        {method: 0.01 for method in lowlight.METHODS})
        )
        self.assertFalse(selection["conservative_raw_applied"])
        self.assertEqual(selection["selected_method"], "clahe_lab")
        self.assertEqual(selection["locked_parameters"]["clahe_clip_limit"], 2.0)
        self.assertEqual(selection["locked_parameters"]["clahe_tile_grid_size"], [8, 8])

    def test_ties_above_the_margin_prefer_higher_floor_precision(self) -> None:
        ious = {"raw": 0.70, "gamma_0p6": 0.85, "clahe_lab": 0.85, "gamma_0p6_then_clahe_lab": 0.60}
        precisions = {"raw": 0.80, "gamma_0p6": 0.91, "clahe_lab": 0.88, "gamma_0p6_then_clahe_lab": 0.99}
        variances = {"raw": 0.010, "gamma_0p6": 0.020, "clahe_lab": 0.005, "gamma_0p6_then_clahe_lab": 0.001}
        selection = benchmark.select_primary_frontend(dev_payload(ious, precisions, variances))
        self.assertEqual(selection["tie_break_applied"], "floor_precision_then_iou_variance")
        self.assertEqual(selection["selected_method"], "gamma_0p6")

    def test_a_precision_tie_is_broken_by_the_lower_iou_variance(self) -> None:
        ious = {"raw": 0.70, "gamma_0p6": 0.85, "clahe_lab": 0.85, "gamma_0p6_then_clahe_lab": 0.60}
        precisions = {method: 0.90 for method in lowlight.METHODS}
        variances = {"raw": 0.010, "gamma_0p6": 0.020, "clahe_lab": 0.005, "gamma_0p6_then_clahe_lab": 0.001}
        selection = benchmark.select_primary_frontend(dev_payload(ious, precisions, variances))
        self.assertEqual(selection["selected_method"], "clahe_lab")

    def test_selection_refuses_a_development_split_without_a_raw_iou(self) -> None:
        payload = dev_payload(
            {"raw": None, "gamma_0p6": 0.8, "clahe_lab": 0.8, "gamma_0p6_then_clahe_lab": 0.8},
            {method: 0.8 for method in lowlight.METHODS},
            {method: 0.0 for method in lowlight.METHODS},
        )
        with self.assertRaises(RuntimeError):
            benchmark.select_primary_frontend(payload)


# ---------------------------------------------------------------------------
# 7: explanation language boundary
# ---------------------------------------------------------------------------


class ExplanationBoundaryTests(unittest.TestCase):
    def test_boundary_sentences_are_accepted(self) -> None:
        for text in (
            "本结果不是真实地面精度，也不包含接触、支撑或步态。",
            "本工具不输出接触、支撑或步态判定。",
            "These records are not ground truth and not a physical accuracy measurement.",
            "No accepted ground state, no foot height, no foot contact and no support phase is produced.",
            "自动掩膜只作为候选，禁止把它当作真实地面。",
        ):
            lowlight.assert_explanation_text(text, "accepted_sample")

    def test_assertive_claims_outside_the_evidence_are_refused(self) -> None:
        for text in (
            "本方法提高了真实地面精度。",
            "该前端可以判定接触与支撑。",
            "本流程输出落地检测结果。",
            "该方法改善了足地高度估计。",
            "The enhanced front-end improved the ground accuracy.",
            "This pipeline reports foot contact and the support phase.",
        ):
            with self.assertRaises(lowlight.OutOfBoundsExplanationError, msg=text):
                lowlight.assert_explanation_text(text, "refused_sample")

    def test_the_module_vocabulary_and_docstrings_pass_the_guard(self) -> None:
        self.assertGreater(lowlight.assert_all_explanations(), 0)
        lowlight.assert_explanation_text(lowlight.__doc__ or "", "module docstring")
        lowlight.assert_explanation_text(benchmark.__doc__ or "", "tool docstring")

    def test_every_nested_payload_string_is_audited(self) -> None:
        safe = {"a": ["fine", {"b": "not a physical accuracy measurement"}], "c": 1}
        self.assertGreater(lowlight.audit_payload_texts(safe), 0)
        unsafe = {"a": ["fine", {"b": "本方法提高了真实地面精度。"}]}
        with self.assertRaises(lowlight.OutOfBoundsExplanationError):
            lowlight.audit_payload_texts(unsafe, label="synthetic")

    def test_the_output_boundary_and_forbidden_list_pass_the_guard(self) -> None:
        lowlight.assert_explanation_text(benchmark.INTERPRETATION_BOUNDARY, "interpretation_boundary")
        for item in benchmark.FORBIDDEN_ITEMS:
            lowlight.assert_explanation_text(item, "forbidden_item")

    def test_the_rejection_guard_is_reachable_and_not_dead_language(self) -> None:
        self.assertIn("真实地面", lowlight.OUT_OF_BOUNDS_ROOT_TERMS)
        self.assertIn("接触", lowlight.OUT_OF_BOUNDS_ROOT_TERMS)
        self.assertIn("支撑", lowlight.OUT_OF_BOUNDS_ROOT_TERMS)
        self.assertEqual(
            lowlight.explanation_violations("该方法提高了真实地面精度。"),
            ["该方法提高了真实地面精度"],
        )


class RegionConsensusPredicateTests(unittest.TestCase):
    def _planar_case(self, on_a_plane: bool):
        rng = np.random.default_rng(11)
        xs = rng.uniform(0.0, 100.0, 400)
        ys = rng.uniform(0.0, 100.0, 400)
        zs = 0.2 * xs + 0.1 * ys + 500.0
        if not on_a_plane:
            zs = zs + rng.uniform(-400.0, 400.0, 400)
        points = np.column_stack((xs, ys, zs))
        pixels = np.column_stack((rng.uniform(0.0, 960.0, 400), rng.uniform(0.0, 540.0, 400)))
        return estimate_region_consensus(
            points, pixels, image_shape=(540, 960), distance_threshold_mm=25.0,
            minimum_region_points=region_minimum_points(), maximum_region_angle_deg=5.0,
        )

    def test_a_successful_consensus_is_reported_as_passed(self) -> None:
        region = self._planar_case(True)
        self.assertEqual(region.status, "candidate")
        self.assertIsNotNone(region.normal_left_camera)
        self.assertTrue(benchmark.region_consensus_passed(region))

    def test_a_failed_consensus_is_reported_as_not_passed(self) -> None:
        region = self._planar_case(False)
        self.assertEqual(region.status, "unavailable")
        self.assertIsNone(region.normal_left_camera)
        self.assertFalse(benchmark.region_consensus_passed(region))

    def test_the_pass_predicate_is_not_a_comparison_with_an_available_status(self) -> None:
        source = inspect.getsource(benchmark.region_consensus_passed)
        self.assertNotIn('== "available"', source)
        self.assertIn("normal_left_camera", source)


# ---------------------------------------------------------------------------
# Geometry bookkeeping and the automatic-candidate boundary
# ---------------------------------------------------------------------------


def geometry_row(candidate_points=1000, inliers=500, fraction=0.5, coverage=0.05,
                 residual=3.0, consensus=True) -> dict:
    return {
        "bilateral_masks_present": True,
        "left_mask_fraction_upright": 0.2,
        "right_mask_fraction_upright": 0.2,
        "quality": {
            "candidate_points": candidate_points,
            "ransac_inliers": inliers,
            "ransac_inlier_fraction": fraction,
            "inlier_coverage_fraction": coverage,
            "median_plane_residual_mm": residual,
        },
        "region_consensus": {"passed": consensus, "status": "available", "reasons": [], "inlier_count": inliers,
                             "metrics": {}},
        "funnel": None,
        "unaccepted_candidate_plane": {"normal_left_camera": [0.0, 1.0, 0.0], "offset_mm": 700.0},
        "state": "unavailable",
        "reasons": ["semantic_identity_unvalidated"],
    }


class GeometryBookkeepingTests(unittest.TestCase):
    def test_weaker_internal_quantities_are_flagged_separately(self) -> None:
        raw = geometry_row()
        method = geometry_row(candidate_points=800, inliers=500, fraction=0.5, coverage=0.05,
                              residual=4.0, consensus=True)
        result = benchmark.geometry_regression_flags(raw, method)
        self.assertTrue(result["flags"]["fewer_candidate_points"])
        self.assertFalse(result["flags"]["fewer_ransac_inliers"])
        self.assertTrue(result["flags"]["higher_median_plane_residual"])
        self.assertFalse(result["flags"]["region_consensus_lost"])
        self.assertTrue(result["any_geometry_weaker"])

    def test_an_arm_without_a_plane_candidate_is_reported_not_dropped(self) -> None:
        raw = geometry_row()
        empty = geometry_row()
        empty["quality"] = None
        empty["region_consensus"] = None
        empty["reasons"] = ["empty_semantic_candidate_mask"]
        result = benchmark.geometry_regression_flags(raw, empty)
        self.assertEqual(result["status"], "partial")
        self.assertTrue(result["flags"]["geometry_unavailable_where_raw_available"])
        self.assertTrue(result["any_geometry_weaker"])

    def test_an_empty_candidate_mask_row_keeps_its_reason_and_no_accepted_plane(self) -> None:
        arm = {
            "bilateral_masks_present": False,
            "left_mask_fraction_upright": 0.0,
            "right_mask_fraction_upright": 0.0,
            "state": "unavailable",
            "reasons": ["empty_semantic_candidate_mask"],
            "funnel": None,
            "quality": None,
            "region_consensus": None,
            "unaccepted_candidate_plane": None,
            "accepted_plane": None,
        }
        row = benchmark.normalize_geometry_row(
            pair_id=40, method="raw", arm=arm, reference_comparison=None,
            mask_status=lowlight.MASK_STATUS_AUTOMATIC_CANDIDATE,
            identity_evidence=lowlight.MASK_IDENTITY_PROVIDED_UNVALIDATED,
        )
        self.assertEqual(row["reasons"], ["empty_semantic_candidate_mask"])
        self.assertIsNone(row["accepted_plane"])
        self.assertIsNone(row["unaccepted_candidate_plane"])
        self.assertFalse(row["region_consensus"]["passed"])
        self.assertIn("region_consensus_not_reached", row["region_consensus"]["reasons"][0])
        self.assertTrue(row["raw_left_right_images_used_for_geometry"])
        self.assertFalse(row["enhanced_image_used_for_geometry"])

    def test_plane_comparison_reports_the_angle_the_offset_and_the_boundary(self) -> None:
        first = {"normal_left_camera": [0.0, 1.0, 0.0], "offset_mm": 700.0}
        second = {"normal_left_camera": [0.0, 1.0, 0.0], "offset_mm": 690.0}
        result = benchmark.compare_candidate_to_reference(first, second)
        self.assertEqual(result["status"], "available")
        self.assertAlmostEqual(result["normal_angle_deg"], 0.0)
        self.assertAlmostEqual(result["offset_difference_mm"], 10.0)
        self.assertAlmostEqual(result["offset_absolute_difference_mm"], 10.0)
        self.assertIn("not a physical ground reference", result["note"])
        missing = benchmark.compare_candidate_to_reference(None, second)
        self.assertEqual(missing["status"], "unavailable")
        self.assertIn("candidate_plane_unavailable", missing["reasons"])

    def test_the_automatic_candidate_guard_refuses_a_promoted_state(self) -> None:
        source = inspect.getsource(benchmark.run_stereo_arm)
        self.assertIn('state != "unavailable"', source)
        self.assertIn("semantic_identity_unvalidated", source)
        self.assertIn("MASK_IDENTITY_PROVIDED_UNVALIDATED", source)
        source_of_funnel_call = inspect.getsource(benchmark.run_stereo_arm)
        self.assertIn("identity_evidence=MASK_IDENTITY_MANUALLY_AUDITED", inspect.getsource(benchmark.main))
        self.assertNotIn("accepted_plane\": plane", source_of_funnel_call)

    def test_mask_provenance_is_automatic_and_unvalidated_for_every_group(self) -> None:
        for method in lowlight.METHODS:
            source = benchmark.mask_source(method)
            self.assertEqual(source["mask_status"], lowlight.MASK_STATUS_AUTOMATIC_CANDIDATE)
            self.assertEqual(source["semantic_identity_evidence"], lowlight.MASK_IDENTITY_PROVIDED_UNVALIDATED)
            self.assertEqual(source["training_or_fine_tuning"], "none")
            self.assertEqual(source["preprocessing_group"], method)

    def test_geometry_aggregate_keeps_every_rejection_reason(self) -> None:
        frames = [
            benchmark.normalize_geometry_row(
                pair_id=pair_id, method="raw",
                arm=geometry_row(candidate_points=points, inliers=inliers),
                reference_comparison=None, mask_status=lowlight.MASK_STATUS_AUTOMATIC_CANDIDATE,
                identity_evidence=lowlight.MASK_IDENTITY_PROVIDED_UNVALIDATED,
            )
            for pair_id, points, inliers in ((40, 900, 400), (120, 10, 2))
        ]
        frames[1]["reasons"] = ["insufficient_stereo_candidates", "insufficient_plane_inliers"]
        frames[1]["state"] = "unavailable"
        aggregate = benchmark.geometry_aggregate(frames)
        self.assertEqual(aggregate["frame_count"], 2)
        self.assertEqual(aggregate["geometry_available_frames"], 2)
        self.assertEqual(aggregate["rejection_reason_counts"]["insufficient_stereo_candidates"], 1)
        self.assertEqual(aggregate["state_counts"]["unavailable"], 2)
        self.assertEqual(aggregate["candidate_points"]["median"], 455.0)
        self.assertIsNone(aggregate["funnel_stage_medians"]["strict_stereo_candidate_pixels"])


class OutputSafetyTests(unittest.TestCase):
    def test_an_existing_output_directory_is_refused_before_any_work(self) -> None:
        with ascii_tempdir() as directory:
            existing = Path(directory) / "already_there"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                benchmark.main([
                    "--left-dir", str(existing), "--right-dir", str(existing),
                    "--left-label-dir", str(existing), "--right-label-dir", str(existing),
                    "--calibration", str(existing / "calibration.json"),
                    "--frozen-baseline-dir", str(existing),
                    "--output-dir", str(existing),
                ])

    def test_written_json_is_audited_before_it_reaches_disk(self) -> None:
        with ascii_tempdir() as directory:
            target = Path(directory) / "summary.json"
            with self.assertRaises(lowlight.OutOfBoundsExplanationError):
                benchmark.write_json(target, {"conclusion": "本方法提高了真实地面精度。"})
            self.assertFalse(target.exists())
            benchmark.write_json(target, {"conclusion": "本结果不是真实地面精度。"})
            payload = json.loads(target.read_text(encoding="utf-8"))
        self.assertIn("不是真实地面精度", payload["conclusion"])

    def test_the_official_frame_partition_matches_the_fixed_twelve_pairs(self) -> None:
        self.assertEqual(len(lowlight.ALL_PAIR_IDS), 12)
        self.assertEqual(len(lowlight.DEV_PAIR_IDS), 6)
        self.assertEqual(len(lowlight.HOLDOUT_PAIR_IDS), 6)
        self.assertEqual(
            sorted(set(lowlight.DEV_PAIR_IDS) | set(lowlight.HOLDOUT_PAIR_IDS)),
            sorted(lowlight.ALL_PAIR_IDS),
        )
        self.assertEqual(sorted(set(lowlight.DEV_PAIR_IDS) & set(lowlight.HOLDOUT_PAIR_IDS)), [])
        self.assertEqual(lowlight.split_of(0), "dev")
        self.assertEqual(lowlight.split_of(40), "holdout")
        with self.assertRaises(ValueError):
            lowlight.split_of(20)


class PathPreflightTests(unittest.TestCase):
    def test_a_short_output_directory_passes_the_path_preflight(self) -> None:
        with ascii_tempdir() as directory:
            report = benchmark.preflight_path_lengths(Path(directory) / "out")
        self.assertLessEqual(report["worst_planned_length"], report["maximum_safe_path_length"])
        self.assertGreaterEqual(report["checked_path_count"], 4 * len(lowlight.METHODS))

    def test_a_deep_output_directory_is_refused_before_anything_is_created(self) -> None:
        deep = ROOT.parent / ("d" * 200)
        with self.assertRaises(RuntimeError):
            benchmark.preflight_path_lengths(deep)
        self.assertFalse(deep.exists())

    def test_every_arm_has_a_short_unique_code(self) -> None:
        codes = [benchmark.arm_code(method, view) for method in lowlight.METHODS for view in benchmark.VIEWS]
        self.assertEqual(len(codes), len(set(codes)))
        self.assertTrue(all(len(code) <= 8 for code in codes))


if __name__ == "__main__":
    unittest.main()
