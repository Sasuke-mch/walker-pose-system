"""Unit tests for the task-03 temporal ground modules and composition matrix.

The tests use small synthetic image sequences for the flow and VO behaviour, and
the real read-only task-01 / task-02 archives for the matrix and the snapshot
guarantees.  Nothing here re-runs a stereo matcher and nothing writes into an
upstream directory.
"""

import json
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for _candidate in (str(ROOT), str(ROOT / "tools")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import benchmark_temporal_ground_modules as benchmark  # noqa: E402
from pose_app.temporal_floor_mask_propagation import PropagationThresholds, propagate_adjacent  # noqa: E402


HEIGHT, WIDTH = 180, 240
THRESHOLDS = PropagationThresholds(
    maximum_forward_backward_error_px=1.5,
    minimum_candidate_flow_consistency=0.70,
    maximum_candidate_photometric_median=35.0,
    minimum_candidate_pixels=200,
)


def frame_with_square(offset_x: int, offset_y: int = 0) -> np.ndarray:
    """A textured background plus one bright square that moves between frames."""
    rng = np.random.default_rng(7)
    image = (rng.integers(30, 90, size=(HEIGHT, WIDTH, 3))).astype(np.uint8)
    cv2.rectangle(image, (30 + offset_x, 90 + offset_y), (110 + offset_x, 160 + offset_y), (230, 230, 230), -1)
    return image


def square_mask(offset_x: int, offset_y: int = 0, half: int = 40) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    center_x, center_y = 70 + offset_x, 125 + offset_y
    mask[center_y - half : center_y + half + 1, center_x - half : center_x + half + 1] = True
    return mask


def sequence(count: int, step_x: int = 3) -> list[np.ndarray]:
    return [frame_with_square(step_x * index) for index in range(count)]


class StubUpstream:
    """Minimal stand-in for the read-only upstream reader in matrix tests."""

    def __init__(self, pass_frames_by_matcher: dict[str, int], dynamic_window: dict | None = None):
        self._pass = pass_frames_by_matcher
        self.task02_summary = {
            "dynamicstereo_offline_window": dynamic_window or {
                "temporal_window_frames": 5, "target_position_in_window": 0,
                "future_lookahead_frames": 4, "realtime_compatible": False,
            }
        }

    def combination(self, matcher: str, plane_method: str) -> dict:
        return {
            "semantic_evidence": {"source": "mask2former_floor", "semantic_identity_evidence": "provided_unvalidated",
                                  "mask_status": "candidate", "manual_audit_available": False},
            "stereo_evidence": {"median_candidate_pixels": 1000.0, "median_reprojection_px": 0.23},
            "plane_evidence": {"median_inlier_fraction": 0.8, "median_residual_mm": 5.0,
                               "median_coverage_fraction": 0.15, "plane_candidate_frames": 12},
            "cross_region_evidence": {"pass_frames": self._pass[matcher], "fail_frames": 12 - self._pass[matcher],
                                      "unavailable_frames": 0},
            "timing_on_task01_comparable_frames": {
                "estimated_direct_pipeline_ms": {"count": 10, "median_ms": 1000.0, "p90_ms": 1100.0,
                                                 "p95_ms": 1200.0, "min_ms": 900.0, "max_ms": 1300.0, "mean_ms": 1050.0},
                "mask_load_ms": {"count": 10, "median_ms": 90.0, "p90_ms": 95.0, "p95_ms": 96.0,
                                 "min_ms": 80.0, "max_ms": 99.0, "mean_ms": 91.0},
            },
        }


    def comparable_timing(self, matcher: str, plane_method: str) -> dict:
        return self.combination(matcher, plane_method)["timing_on_task01_comparable_frames"]


def stub_flow_summary() -> dict:
    two_view = benchmark.timing_statistics([200.0, 220.0, 240.0], warmup_excluded_count=2)
    return {
        "record_count": 4, "interval_count": 2, "available_records": 4, "unavailable_records": 0,
        "reason_counts": {}, "warmup_excluded_count": 2,
        "left": {}, "right": {},
        "two_view_module_ms": two_view,
        "timing_modules": {"flow_two_view_combined_ms": two_view},
        "temporal_mode": "flow_assisted_current_frame",
        "semantics": benchmark.FLOW_ASSISTED_SEMANTICS,
    }


def vo_records_for(upstream, matcher: str, plane_method: str, intervals: list[tuple[int, int]]) -> list[dict]:
    status = "pass" if upstream._pass[matcher] else "fail"
    records = []
    for source, target in intervals:
        records.append(benchmark.evaluate_vo_interval(
            matcher=matcher, plane_method=plane_method,
            source_pair_id=source, target_pair_id=target,
            source_plane=None if status != "pass" else object(),
            source_anchor_cross_region_status=status,
            target_anchor_cross_region_status=status,
            target_plane=object(),
            relative_pose_provider=lambda a, b: (_ for _ in ()).throw(
                AssertionError("no relative pose may be requested without a pass anchor")
            ) if status != "pass" else (None, "no_per_frame_static_background_metric_correspondences_in_frozen_archives"),
            domain_certificate=lambda a, b: benchmark.static_background_domain_certificate(
                floor_masks_available=False, person_regions_available=False, walker_regions_available=False
            ),
        ))
    return records


class TemporalGroundModulesTests(unittest.TestCase):
    # ------------------------------------------------------------------ flow only adds evidence
    def test_flow_record_keeps_the_current_mask2former_mask_authoritative(self):
        previous_image, current_image = sequence(2)
        previous_mask = square_mask(3)
        current_mask = square_mask(6, half=30)   # different size, so the two masks cannot coincide
        record, propagated = benchmark.interval_flow_record(
            view="left", previous_pair_id=0, current_pair_id=1,
            previous_image=previous_image, current_image=current_image,
            previous_mask=previous_mask, current_mask=current_mask, thresholds=THRESHOLDS,
        )
        self.assertTrue(record["current_mask_is_authoritative"])
        self.assertFalse(record["propagated_mask_used_as_geometry_input"])
        # the reported current-mask fields must describe the supplied current mask
        self.assertEqual(record["current_mask_pixel_count"], int(current_mask.sum()))
        self.assertAlmostEqual(record["current_mask_area_fraction"], current_mask.mean(), places=12)
        self.assertNotEqual(record["current_mask_pixel_count"], int(propagated.sum()))
        self.assertNotEqual(record["current_mask_pixel_count"], record["propagated_pixel_count"])
        self.assertEqual(record["temporal_mode_meaning"], benchmark.FLOW_ASSISTED_SEMANTICS)
        # a mask is never replaced: the propagated raster is returned separately
        self.assertEqual(propagated.shape, current_mask.shape)
        # swapping in a different current mask must change only the current-side fields
        other_mask = square_mask(9)
        other, _ = benchmark.interval_flow_record(
            view="left", previous_pair_id=0, current_pair_id=1,
            previous_image=previous_image, current_image=current_image,
            previous_mask=previous_mask, current_mask=other_mask, thresholds=THRESHOLDS,
        )
        self.assertNotEqual(record["current_mask_pixel_count"], other["current_mask_pixel_count"])
        self.assertEqual(record["propagated_pixel_count"], other["propagated_pixel_count"])
        self.assertEqual(record["flow_status"], other["flow_status"])

    def test_flow_step_matches_the_frozen_propagation_function(self):
        previous_image, current_image = sequence(2)
        source_mask = square_mask(3)
        reference_mask, reference_record = propagate_adjacent(
            previous_image, current_image, source_mask, THRESHOLDS
        )
        timings: dict[str, float] = {}
        mirrored_mask, mirrored_record = benchmark.propagate_adjacent_timed(
            previous_image, current_image, source_mask, THRESHOLDS, timings
        )
        self.assertTrue(np.array_equal(reference_mask, mirrored_mask))
        for key in ("status", "reasons", "propagated_floor_pixel_count",
                    "candidate_flow_consistency_fraction", "candidate_photometric_median_abs_gray"):
            self.assertEqual(reference_record[key], mirrored_record[key], msg=key)
        for key in ("flow_field_ms", "consistency_ms", "decision_ms", "step_total_ms"):
            self.assertIn(key, timings)
            self.assertGreaterEqual(timings[key], 0.0)

    # ------------------------------------------------------------------ no target label leakage
    def test_anchor_evaluation_never_feeds_the_target_label_back(self):
        images = sequence(4)
        label_a = (square_mask(9), np.ones((HEIGHT, WIDTH), dtype=bool))
        label_b = (square_mask(0), np.ones((HEIGHT, WIDTH), dtype=bool))
        log_a: list[str] = []
        log_b: list[str] = []
        record_a, propagated_a, _ = benchmark.evaluate_anchor_interval(
            view="left", source_pair_id=0, target_pair_id=3,
            seed_masks={"manual_seed": square_mask(0)},
            load_image=lambda pair_id: images[pair_id], step_count=3, thresholds=THRESHOLDS,
            target_loader=lambda: label_a, phase_log=log_a,
        )
        record_b, propagated_b, _ = benchmark.evaluate_anchor_interval(
            view="left", source_pair_id=0, target_pair_id=3,
            seed_masks={"manual_seed": square_mask(0)},
            load_image=lambda pair_id: images[pair_id], step_count=3, thresholds=THRESHOLDS,
            target_loader=lambda: label_b, phase_log=log_b,
        )
        # the identical chain plus two different endpoint labels must give identical masks
        for seed in propagated_a:
            self.assertTrue(np.array_equal(propagated_a[seed], propagated_b[seed]))
        self.assertNotEqual(
            record_a["seeds"]["manual_seed"]["against_manual_target"]["one_anchor_to_next_anchor_iou"],
            record_b["seeds"]["manual_seed"]["against_manual_target"]["one_anchor_to_next_anchor_iou"],
        )
        self.assertEqual(record_a["target_label_read_phase"], "after_propagation")
        for phase_log in (log_a, log_b):
            propagation = [index for index, entry in enumerate(phase_log) if entry.startswith("propagate_chain:left")]
            loading = [index for index, entry in enumerate(phase_log) if entry == "load_target_label:left"]
            self.assertTrue(propagation and loading)
            self.assertLess(max(propagation), min(loading))

    def test_propagation_chain_cannot_receive_a_target_mask(self):
        import inspect

        signature = inspect.signature(benchmark.propagate_chains_locked)
        self.assertNotIn("target", " ".join(signature.parameters).lower())
        self.assertNotIn("target_mask", inspect.getsource(benchmark.propagate_chains_locked))

    # ------------------------------------------------------------------ VO fail-closed rules
    def test_vo_is_unavailable_without_a_pass_direct_anchor(self):
        def exploding_provider(from_id, to_id):
            raise AssertionError("the relative-pose front end must not run without a pass anchor")

        record = benchmark.evaluate_vo_interval(
            matcher="sgbm", plane_method="dense_ransac",
            source_pair_id=0, target_pair_id=40,
            source_plane=object(), source_anchor_cross_region_status="fail",
            target_anchor_cross_region_status="pass", target_plane=object(),
            relative_pose_provider=exploding_provider,
            domain_certificate=lambda a, b: benchmark.static_background_domain_certificate(
                floor_masks_available=True, person_regions_available=True, walker_regions_available=True
            ),
        )
        self.assertEqual(record["vo_status"], "unavailable")
        self.assertEqual(record["source_plane_status"], "unavailable")
        self.assertEqual(record["reasons"], ["no_direct_anchor_after_cross_region_gate"])
        self.assertIsNone(record["propagated_vs_target_normal_deg"])
        self.assertIsNone(record["propagated_vs_target_offset_mm"])
        self.assertEqual(record["confidence_boundary"], benchmark.VO_CONFIDENCE_BOUNDARY)

    def test_vo_reports_the_missing_static_background_domain_when_no_pose_exists(self):
        record = benchmark.evaluate_vo_interval(
            matcher="igev", plane_method="soft_weighted_irls",
            source_pair_id=0, target_pair_id=40,
            source_plane=object(), source_anchor_cross_region_status="pass",
            target_anchor_cross_region_status="fail", target_plane=object(),
            relative_pose_provider=lambda a, b: (None, "no_per_frame_static_background_metric_correspondences_in_frozen_archives"),
            domain_certificate=lambda a, b: benchmark.static_background_domain_certificate(
                floor_masks_available=False, person_regions_available=True, walker_regions_available=False
            ),
        )
        self.assertEqual(record["vo_status"], "unavailable")
        self.assertEqual(record["source_plane_status"], "direct")
        self.assertIn("no_per_frame_static_background_metric_correspondences_in_frozen_archives", record["reasons"])
        self.assertIn("static_background_domain_not_certifiable", record["reasons"])
        self.assertEqual(
            record["static_background_domain"]["missing_exclusions"],
            ["mask2former_floor_candidate_region_unavailable", "unverified_walker_candidate_region_unavailable"],
        )
        self.assertFalse(record["static_background_domain"]["ground_region_used_for_motion"])
        self.assertTrue(record["target_plane_used_for_comparison_only"])
        self.assertFalse(record["target_plane_fed_back_into_propagation"])
        self.assertIsNone(record["vo_runtime_ms_per_step"])

    def test_dynamicstereo_cross_region_failure_never_reaches_vo(self):
        upstream_path = benchmark.PROJECT_ROOT / "research_records/engineering_validation/G20260912_learned_stereo_replacement_benchmark_v1"
        if not (upstream_path / "summary.json").is_file():
            self.skipTest("private learned-stereo archive not provisioned")
        summary = json.loads((upstream_path / "summary.json").read_text(encoding="utf-8"))
        dynamic = [
            combination for combination in summary["combinations"]
            if combination["matcher"] == "dynamicstereo" and combination["semantic_source"] == "mask2former_floor"
        ]
        self.assertEqual(len(dynamic), 3)
        for combination in dynamic:
            self.assertEqual(combination["cross_region_evidence"]["pass_frames"], 0)
        stub = StubUpstream({"sgbm": 6, "igev": 12, "dynamicstereo": 0})
        records = vo_records_for(stub, "dynamicstereo", "dense_ransac", [(0, 40), (40, 80)])
        for record in records:
            self.assertEqual(record["vo_status"], "unavailable")
            self.assertIn("no_direct_anchor_after_cross_region_gate", record["reasons"])
            self.assertEqual(record["source_plane_status"], "unavailable")

    def test_vo_chain_propagates_a_plane_when_real_poses_are_supplied(self):
        from pose_app.local_plane_propagation import LocalPlane, RelativePose

        plane = LocalPlane(np.asarray([0.0, 0.0, 1.0]), 800.0)
        identity = np.eye(3)

        def provider(from_id, to_id):
            return RelativePose(from_id, to_id, identity, np.asarray([0.0, 0.0, -10.0]),
                                static_3d_correspondence_count=40, ransac_inlier_count=35), ""

        record = benchmark.evaluate_vo_interval(
            matcher="igev", plane_method="dense_ransac",
            source_pair_id=0, target_pair_id=3,
            source_plane=plane, source_anchor_cross_region_status="pass",
            target_anchor_cross_region_status="pass", target_plane=LocalPlane(np.asarray([0.0, 0.0, 1.0]), 830.0),
            relative_pose_provider=provider,
            domain_certificate=lambda a, b: benchmark.static_background_domain_certificate(
                floor_masks_available=True, person_regions_available=True, walker_regions_available=True
            ),
        )
        self.assertEqual(record["vo_status"], "propagated")
        self.assertEqual(record["propagation_step_count"], 3)
        self.assertEqual(record["source_plane_status"], "direct")
        self.assertEqual(record["pose_inlier_count"], 35)
        self.assertAlmostEqual(record["propagated_vs_target_normal_deg"], 0.0, places=6)
        self.assertAlmostEqual(record["propagated_vs_target_offset_mm"], 0.0, places=6)
        self.assertEqual(record["reference_pose_status_counts"], {"accepted": 3})

    # ------------------------------------------------------------------ the 27-row matrix
    def build_matrix(self, pass_frames: dict[str, int]) -> list[dict]:
        upstream = StubUpstream(pass_frames)
        intervals = [(0, 40), (40, 80)]
        vo_by_key = {}
        for matcher in benchmark.MATCHERS:
            for plane_method in benchmark.PLANE_METHODS:
                vo_by_key[(matcher, plane_method)] = vo_records_for(upstream, matcher, plane_method, intervals)
        return benchmark.build_combination_matrix(
            upstream, flow_summary=stub_flow_summary(), vo_by_key=vo_by_key,
            vo_runtime_stats=benchmark.timing_statistics([]),
            declared_frame_budget_ms=benchmark.DECLARED_FRAME_BUDGET_MS,
        )

    def test_matrix_has_exactly_27_unique_rows(self):
        rows = self.build_matrix({"sgbm": 6, "igev": 12, "dynamicstereo": 0})
        self.assertEqual(len(rows), 27)
        identifiers = [row["combination_id"] for row in rows]
        self.assertEqual(len(set(identifiers)), 27)
        expected = {
            benchmark.combination_id(matcher, plane_method, mode)
            for matcher in benchmark.MATCHERS
            for plane_method in benchmark.PLANE_METHODS
            for mode in benchmark.TEMPORAL_MODES
        }
        self.assertEqual(set(identifiers), expected)
        for row in rows:
            self.assertEqual(row["semantic_model"], benchmark.ONLINE_SEMANTIC_MODEL)
            self.assertEqual(row["direct_spatial_evidence_status"], "measured")

    def test_full_chain_measured_flags_follow_the_temporal_mode(self):
        rows = self.build_matrix({"sgbm": 6, "igev": 12, "dynamicstereo": 0})
        for row in rows:
            if row["temporal_mode"] == "direct_current_frame":
                self.assertTrue(row["full_chain_measured"])
                self.assertEqual(row["measurement_scope"], "direct_chain")
                self.assertEqual(row["flow_module_status"], "not_applicable")
            elif row["temporal_mode"] == "flow_assisted_current_frame":
                self.assertFalse(row["full_chain_measured"])
                self.assertEqual(row["measurement_scope"], "temporal_module_composition")
                self.assertEqual(row["flow_module_status"], "measured")
            else:
                self.assertFalse(row["full_chain_measured"])
                self.assertEqual(row["measurement_scope"], "temporal_module_composition")
                self.assertIn(row["vo_module_status"], ("measured", "unavailable"))

    def test_estimated_latency_is_never_labelled_measured(self):
        rows = self.build_matrix({"sgbm": 6, "igev": 12, "dynamicstereo": 0})
        for row in rows:
            self.assertIsNone(row["measured_end_to_end_latency_ms"])
            self.assertTrue(row["measured_end_to_end_latency_reason"])
            self.assertEqual(row["estimated_end_to_end_latency_kind"], "estimated_module_time_sum")
            self.assertIn("estimated", row["estimated_end_to_end_latency_caveat"])
            if row["temporal_mode"] == "vo_propagated_plane":
                self.assertIsNone(row["estimated_end_to_end_latency_ms"])
                self.assertEqual(
                    row["estimated_end_to_end_latency_unavailable_reason"],
                    "vo_module_unavailable_no_certified_static_background_pose_chain",
                )
            else:
                self.assertIsNotNone(row["estimated_end_to_end_latency_ms"])
                self.assertGreater(row["estimated_end_to_end_latency_ms"], 0.0)

    def test_dynamicstereo_rows_are_never_realtime_and_carry_the_offline_window(self):
        rows = self.build_matrix({"sgbm": 6, "igev": 12, "dynamicstereo": 0})
        dynamic = [row for row in rows if row["matcher"] == "dynamicstereo"]
        self.assertEqual(len(dynamic), 9)
        for row in dynamic:
            self.assertFalse(row["realtime_compatible"])
            self.assertEqual(row["future_lookahead_frames"], 4)
            self.assertTrue(row["offline_window_inference"])
            self.assertIn("offline_five_frame_window_requires_four_future_frames",
                          row["realtime_incompatibility_reasons"])

    def test_dynamicstereo_vo_rows_carry_the_anchor_reason(self):
        rows = self.build_matrix({"sgbm": 6, "igev": 12, "dynamicstereo": 0})
        for row in rows:
            if row["matcher"] != "dynamicstereo" or row["temporal_mode"] != "vo_propagated_plane":
                continue
            evidence = row["temporal_evidence"]
            self.assertEqual(evidence["propagated_intervals"], 0)
            self.assertIn("no_direct_anchor_after_cross_region_gate", evidence["reason_counts"])
            self.assertEqual(row["vo_module_status"], "unavailable")

    # ------------------------------------------------------------------ timing schema
    def test_every_timing_statistic_carries_the_required_fields(self):
        statistics = benchmark.timing_statistics([1.0, 2.0, None, 3.0], warmup_excluded_count=2)
        for field in benchmark.REQUIRED_TIMING_FIELDS:
            self.assertIn(field, statistics)
        self.assertEqual(statistics["count"], 3)
        self.assertEqual(statistics["warmup_excluded_count"], 2)
        self.assertEqual(statistics["min_ms"], 1.0)
        self.assertEqual(statistics["max_ms"], 3.0)
        empty = benchmark.timing_statistics([])
        for field in benchmark.REQUIRED_TIMING_FIELDS:
            self.assertIn(field, empty)
        self.assertEqual(empty["count"], 0)
        self.assertIsNone(empty["median_ms"])

    # ------------------------------------------------------------------ upstream protection
    def test_upstream_archives_are_untouched_by_a_read_only_pass(self):
        task01 = benchmark.PROJECT_ROOT / "research_records/engineering_validation/G20260912_modular_ground_benchmark_v1"
        task02 = benchmark.PROJECT_ROOT / "research_records/engineering_validation/G20260912_learned_stereo_replacement_benchmark_v1"
        if not (task01 / "summary.json").is_file() or not (task02 / "summary.json").is_file():
            self.skipTest("private ground/stereo archives not provisioned")
        before = {"task01": benchmark.snapshot_tree(task01), "task02": benchmark.snapshot_tree(task02)}
        upstream = benchmark.Upstream(task01, task02)
        for matcher in benchmark.MATCHERS:
            for plane_method in benchmark.PLANE_METHODS:
                upstream.direct_plane(matcher, plane_method, 0)
                upstream.cross_region_status(matcher, 0, plane_method)
                upstream.comparable_timing(matcher, plane_method)
        after = {"task01": benchmark.snapshot_tree(task01), "task02": benchmark.snapshot_tree(task02)}
        for key in ("task01", "task02"):
            difference = benchmark.snapshot_difference(before[key], after[key])
            self.assertTrue(difference["unchanged"], msg=f"{key}: {difference}")
            self.assertGreater(difference["files_before"], 0)

    def test_upstream_snapshot_detects_a_tampered_file(self):
        before = {"a.txt": [10, 100]}
        after = {"a.txt": [10, 200]}
        difference = benchmark.snapshot_difference(before, after)
        self.assertFalse(difference["unchanged"])
        self.assertEqual(difference["changed"], ["a.txt"])
        self.assertFalse(benchmark.snapshot_difference(before, before)["unchanged"] is False)

    def test_static_background_domain_requires_all_three_exclusions(self):
        certificate = benchmark.static_background_domain_certificate(
            floor_masks_available=True, person_regions_available=True, walker_regions_available=True
        )
        self.assertTrue(certificate["certified"])
        self.assertFalse(certificate["ground_region_used_for_motion"])
        for flag in ("floor_masks_available", "person_regions_available", "walker_regions_available"):
            arguments = {"floor_masks_available": True, "person_regions_available": True, "walker_regions_available": True}
            arguments[flag] = False
            self.assertFalse(benchmark.static_background_domain_certificate(**arguments)["certified"])
        self.assertIn("static_background_excluding_person_walker_ground", benchmark.VO_STATIC_DOMAIN_NOTE.replace(
            "Static-background VO features may only come from regions outside the Mask2Former floor candidate, the PMPose "
            "person region and the unverified walker candidate region. No ground, person or walker point may be used to "
            "make up a missing pose, and a rejected or missing adjacent pose fails the chain closed.",
            "static_background_excluding_person_walker_ground",
        ))

    def test_experiment_and_recommendation_have_no_ground_truth_claim(self):
        # claim-shaped phrases must never appear; the boundary sentences that
        # explicitly refuse those claims must appear.
        forbidden = ("真实地面精度提升", "真实相机高度", "足地高度真值", "测量得到的地面精度")
        required_disclaimers = ("不得表述为真实地面精度", "不是真实精度", "内部一致性")
        upstream = StubUpstream({"sgbm": 6, "igev": 12, "dynamicstereo": 0})
        upstream.task02_summary.update({
            "dynamicstereo_offline_latency": {
                "future_lookahead_frames": 4,
                "future_information_wait_ms_at_effective_pair_rate": 143.79,
                "window_inference_ms_median": 7325.68,
                "offline_latency_ms_median": 7469.47,
                "window_throughput_frames_per_second": 0.68,
            }
        })
        rows = self.build_matrix({"sgbm": 6, "igev": 12, "dynamicstereo": 0})
        text = benchmark.render_recommendation(
            matrix=rows,
            flow_evaluation={
                "macro_metrics": {
                    "manual_seed_vs_manual_target": {"left": {"one_anchor_to_next_anchor_iou": 0.5}, "right": {"one_anchor_to_next_anchor_iou": 0.4}},
                    "production_compatible_mask2former_seed_vs_mask2former_target": {"left": {"one_anchor_to_next_anchor_iou": 0.6}, "right": {"one_anchor_to_next_anchor_iou": 0.5}},
                }
            },
            vo_summary={"anchor_intervals_evaluated": 11, "source_anchors_eligible": 6, "propagated_intervals": 0,
                        "reason_counts": {"no_direct_anchor_after_cross_region_gate": 3}},
            upstream=upstream, declared_frame_budget_ms=benchmark.DECLARED_FRAME_BUDGET_MS,
        )
        for section in ("## A.", "## B.", "## C.", "## D.", "## E.", "## F.", "## G."):
            self.assertIn(section, text)
        for phrase in forbidden:
            self.assertNotIn(phrase, text)
        for phrase in required_disclaimers:
            self.assertIn(phrase, text)

    def test_no_mask_is_copied_between_views_and_thresholds_are_the_frozen_ones(self):
        source = Path(benchmark.__file__).read_text(encoding="utf-8")
        self.assertNotIn("left_mask_upright=right", source)
        self.assertIn("left_mask_never_copied_to_right", source)
        defaults = PropagationThresholds()
        self.assertEqual(defaults.maximum_forward_backward_error_px, 1.5)
        self.assertEqual(defaults.minimum_candidate_flow_consistency, 0.70)
        self.assertEqual(defaults.maximum_candidate_photometric_median, 35.0)
        self.assertEqual(defaults.minimum_candidate_pixels, 200)


if __name__ == "__main__":
    unittest.main()
