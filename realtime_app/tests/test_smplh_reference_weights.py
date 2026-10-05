"""Reference weights retain moderate geometry and suppress extreme conflicts."""
import unittest

import numpy as np
from pose_app.smplh_fitting.losses import observation_weights


class ReferenceWeightTests(unittest.TestCase):
    def weights(self, errors, gaps, confidence=0.8, accepted=None):
        errors, gaps = np.asarray(errors, float), np.asarray(gaps, float)
        terms = (np.full_like(errors, confidence), np.exp(-gaps/30),
                 np.exp(-errors/8), np.ones_like(errors), np.ones_like(errors))
        quality = np.prod(terms, axis=0)
        mask = np.ones_like(errors, bool) if accepted is None else accepted
        return quality, observation_weights(quality, terms, mask, "reference")

    def test_moderate_errors_retain_weight_even_when_legacy_is_tiny(self):
        old, new = self.weights([0, 10, 25], [0, 30, 80])
        np.testing.assert_allclose(new, .92)
        self.assertLess(old[-1], .003)

    def test_extreme_errors_drop_and_tail_is_monotone(self):
        _, new = self.weights([25, 50, 100, 200, 10000], [0]*5)
        self.assertTrue(np.all(np.diff(new) <= 0))
        self.assertLess(new[2], .02)
        self.assertGreater(new[-1], 0)
        self.assertGreater(new[1], .4)

    def test_either_extreme_ray_or_pixel_conflict_can_reduce_weight(self):
        _, pixel = self.weights([100], [0])
        _, ray = self.weights([0], [320])
        np.testing.assert_allclose(pixel, ray)

    def test_invalid_mask_stays_zero_even_with_perfect_geometry(self):
        _, new = self.weights([0, 0], [0, 0], accepted=np.array([True, False]))
        np.testing.assert_allclose(new, [.92, 0])

    def test_reference_confidence_has_a_floor_not_a_probability(self):
        _, new = self.weights([0], [0], confidence=0)
        np.testing.assert_allclose(new, .6)

    def test_legacy_quality_return_is_unchanged(self):
        quality = np.array([.001, .9])
        self.assertIs(observation_weights(quality, None, None, "quality"), quality)

    def test_invalid_scales_fail(self):
        for opts in ({"reference_reprojection_px": 0}, {"reference_ray_gap_mm": float("nan")},
                     {"reference_confidence_floor": 1.1}):
            with self.assertRaises(ValueError):
                observation_weights(np.ones(1), (np.ones(1),)*5, np.ones(1, bool), "reference", **opts)

    def test_invalid_quality_terms_are_not_silently_clipped(self):
        for value in (float("nan"), -0.1, 1.1):
            terms = (np.array([value]),) + (np.ones(1),)*4
            with self.assertRaises(ValueError):
                observation_weights(np.ones(1), terms, np.ones(1, bool), "reference")

    def test_cli_invalid_reference_scales_stop_before_assets(self):
        from pose_app.smplh_fitting.stages import validate_optimization_options
        from walker_tools.body.fit_smplh_wilor_sequence import build_parser

        required = ["--left-raw", "left", "--right-raw", "right", "--wilor-left", "wl",
                    "--wilor-right", "wr", "--calibration-dir", "cal", "--regressor", "reg",
                    "--smplh-model", "model", "--output-dir", "out", "--body-weighting", "reference"]
        for flag, value in [("--body-reference-reprojection-px", "0"),
                            ("--body-reference-ray-gap-mm", "nan"),
                            ("--body-reference-confidence-floor", "1.1")]:
            with self.assertRaises(ValueError):
                validate_optimization_options(build_parser().parse_args(required+[flag, value]))


if __name__ == "__main__":
    unittest.main()
